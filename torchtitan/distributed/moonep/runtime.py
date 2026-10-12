# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""MoonEP resource ownership: symmetric expert weights and per-step state.

The device-independent half of the integration (Kimi K3 report S5.2.1): every
rank receives exactly ``S*K`` tokens so all activation shapes are constants, the
redundant experts are planned per micro-batch and prefetched on the comm stream,
their gradients are staged in FP32 and reduced back to their home ranks, and no
host reads a device scalar on the critical path. All device-specific work is
delegated to a :class:`~torchtitan.distributed.moonep.backend.MoonEPBackend`.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchtitan.distributed.moonep.backend import get_moonep_backend
from torchtitan.distributed.moonep.plan import (
    compact_group_ends,
    full_group_ends,
    verify_plan_locality,
    zero_static_tail_,
)
from torchtitan.distributed.moonep.settings import MoonEPSettings
from torchtitan.distributed.moonep.streams import (
    _record_stream,
    device_stream_module,
)
from torchtitan.distributed.moonep.trace import trace

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

logger = logging.getLogger(__name__)


def _detect_device_type() -> str:
    """Resolve the accelerator this process runs on, the MoonEP way.

    torchtitan's own ``device_type`` is resolved at import time and defaults to
    "cuda" when nothing else is available, while MoonEP needs the canonical
    name: a ``torch_npu`` build may surface as a PrivateUse1 device name under
    ``torch.accelerator``, and the backend registry is keyed by "npu".
    """
    try:
        import torch_npu  # noqa: F401

        return "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("MoonEP requires a torch_npu or CUDA accelerator; none was detected.")


class MoonEPSymmetricProjection:
    """One layer's expert projection backed by MoonEP symmetric memory.

    Three views over the same memory: ``home_weight`` ``[epn, out, in]`` is the
    parameter payload, ``full_weight`` ``[E+B, out, in]`` is what prefetch reads,
    ``gemm_weight`` is the GEMM operand, global or compact per the backend.
    """

    def __init__(
        self,
        runtime: "MoonEPRuntime",
        role: str,
        source: torch.Tensor,
        ep_mesh: "DeviceMesh",
        transposed: bool = False,
    ):
        self.runtime = runtime
        self.role = role
        self.closed = False
        #: Whether the stored matrix is ``[in, out]`` instead of ``[out, in]``.
        #: Only the grouped GEMMs care; every buffer below mirrors what the model
        #: brought, which is what keeps the parameter's shape and the model's
        #: state-dict layout unchanged.
        self.transposed = transposed
        # The two matrix dimensions as stored, not as named by the linear-layer
        # convention: under ``transposed`` these are (in_features, out_features).
        self.out_features = int(source.shape[-2])
        self.in_features = int(source.shape[-1])
        self.pool = runtime.get_projection_pool(role, self.out_features, self.in_features)
        self.resources = runtime.backend.create_projection(runtime, self.pool, self.out_features, self.in_features)
        self._staged_dtype = torch.bfloat16

        epn = runtime.experts_per_rank
        start = runtime.rank * epn
        local_source = source.to_local() if isinstance(source, DTensor) else source[start : start + epn]
        self.resources.home_weight.copy_(local_source.detach())
        dist.barrier(group=runtime.group)

        local_dtensor = DTensor.from_local(self.resources.home_weight, ep_mesh, (Shard(0),), run_check=False)
        self.parameter = torch.nn.Parameter(local_dtensor, requires_grad=source.requires_grad)

    @property
    def full_weight(self) -> torch.Tensor:
        return self.resources.full_weight

    @property
    def gemm_weight(self) -> torch.Tensor:
        """The weight operand for the forward and data-gradient GEMMs."""
        return self.resources.gemm_weight

    def split_gradient(self, gemm_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a weight gradient into its home and prefetch-slot halves."""
        runtime = self.runtime
        epn = runtime.experts_per_rank
        slots = runtime.prefetch_slots
        return gemm_grad[:epn], gemm_grad[epn : epn + slots]

    def prefetch(self, buffer, plan) -> None:
        """Copy this step's planned redundant expert weights into the slots."""
        self.runtime.backend.prefetch(self.runtime, self.resources, buffer, plan)

    def stage_gradient(self, gemm_grad: torch.Tensor) -> None:
        """Publish the gradient halves into the FP32 reduction buffers.

        ``gemm_grad`` is ``[epn+B, out, in]``: the weight gradient always uses
        the compact group set, whichever layout the forward's weight operand
        had. There is no tail group to drop, because ``compact_group_ends``
        folds MoonEP's reserved rows into the final slot group.
        """
        home_slice, slot_slice = self.split_gradient(gemm_grad)
        self.runtime.backend.stage_gradient(self.runtime, self.resources, home_slice, slot_slice)
        self._staged_dtype = gemm_grad.dtype

    def finish_gradient(self, buffer, plan) -> torch.Tensor:
        """Rendezvous, then reduce the staged slot gradients home.

        Must run after :meth:`stage_gradient` on every rank. The rendezvous is per
        projection, not shared: every MoonEP barrier uses the same slot and
        generation counter, so the sequence of arrivals is part of the protocol.

        Returns ``[epn, out, in]`` in the parameter dtype.
        """
        backend = self.runtime.backend
        backend.rendezvous_before_reduce(self.runtime, buffer)
        home_grad = backend.reduce_staged_gradient(self.runtime, self.resources, buffer, plan)
        # The FP32 buffers are pooled across every layer with this matrix shape,
        # so the result has to be copied out before the next layer overwrites
        # them. The cast to the parameter dtype does exactly that.
        # Stays a plain local-shaped tensor on purpose. It is returned as the
        # gradient of ``parameter.to_local()``, whose own backward restores the
        # parameter's placements; handing back a DTensor here would instead be
        # checked against the local shape and rejected for being global-sized.
        local_grad = home_grad.to(dtype=self._staged_dtype)
        factor = self.runtime.gradient_divide_factor
        if factor != 1.0:
            # After the cast rather than before it: the FP32 buffer is the wide
            # one, and for Qwen3-30B's gate_up it is 402 MiB per layer, so an
            # extra pass over it costs real bandwidth in the backward. The factor
            # works out to the world size, hence a power of two in practice, so
            # scaling the narrower copy is exact anyway.
            local_grad.div_(factor)
        return local_grad

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.resources.close()


class MoonEPCallState:
    """State of a single expert forward: plan, group layout, pending events."""

    def __init__(self, runtime: "MoonEPRuntime", buffer, tokens_per_rank: int, hidden: int, top_k: int):
        self.runtime = runtime
        self.buffer = buffer
        self.tokens_per_rank = tokens_per_rank
        self.hidden = hidden
        self.top_k = top_k
        self.validate = runtime.should_check_plan_invariants()
        self.plan = None
        self.cu_seqlens = None
        self.group_ends = None
        self.wgrad_group_ends = None
        self.dispatch_rows = None
        self._wgrad_group_ends_host: list[int] | None = None
        self._prefetch_events: dict[int, Any] = {}

    def wgrad_group_ends_host(self) -> list[int] | None:
        """The weight-gradient group offsets on the host, read at most once.

        Both projections of a layer share this cache, so a layer costs one
        device-to-host read rather than two.
        """
        if self._wgrad_group_ends_host is None:
            self._wgrad_group_ends_host = self.wgrad_group_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
        return self._wgrad_group_ends_host

    @property
    def _comm_stream(self):
        if not self.runtime.config.async_finish or not self.runtime.backend.supports_async_comm:
            return None
        stream = self.buffer._comm_stream
        if stream is None:
            raise RuntimeError("MoonEP communication stream is unavailable.")
        return stream

    def wait(self, event) -> None:
        if event is not None:
            event.wait(self.runtime.device_module.current_stream())

    def adopt(self, *tensors: torch.Tensor | None) -> None:
        """Record tensors produced on the comm stream on the compute stream."""
        if self._comm_stream is None:
            return
        stream = self.runtime.device_module.current_stream()
        for tensor in tensors:
            _record_stream(tensor, stream)

    def set_plan(self, plan, cu_seqlens: torch.Tensor, dispatch_rows: int) -> None:
        """Record the plan and derive this step's two group layouts.

        ``group_ends`` must match the backend's weight layout; the weight gradient
        reads no weights and can always use the compact set.
        """
        runtime = self.runtime
        self.plan = plan
        self.cu_seqlens = cu_seqlens
        self.dispatch_rows = dispatch_rows

        if self.validate:
            verify_plan_locality(cu_seqlens, runtime.rank, runtime.experts_per_rank, runtime.num_experts)
        compact = compact_group_ends(
            cu_seqlens,
            runtime.rank,
            runtime.experts_per_rank,
            runtime.num_experts,
            runtime.prefetch_slots,
            dispatch_rows,
        )
        self.wgrad_group_ends = compact
        self.group_ends = (
            compact if runtime.backend.expert_weight_layout == "compact" else full_group_ends(cu_seqlens, dispatch_rows)
        )

    def neutralize_static_tail(self, rows: torch.Tensor) -> torch.Tensor:
        """Zero the rows past the last live group, where the backend has them.

        Both layouts extend their final offset to the full buffer, so the
        undefined tail would contaminate the last group's weight gradient.
        """
        if not self.runtime.backend.has_static_tail:
            return rows
        return zero_static_tail_(rows, self.cu_seqlens[-1])

    def start_prefetch(self, projection: MoonEPSymmetricProjection) -> None:
        """Queue a projection's redundant-weight copy on the comm stream.

        Slots are pooled across layers, so the copy overwrites memory an adjacent
        layer's GEMM reads: the comm stream must wait for the compute stream's
        queued work first. After that the copy proceeds concurrently, which is
        what lets the ``down`` prefetch overlap the first GEMM.
        """
        if self.plan is None:
            raise RuntimeError("MoonEP dispatch plan is not initialized.")
        trace(self.runtime, "prefetch.start", projection.role)
        with torch.profiler.record_function("moonep.prefetch"):
            comm_stream = self._comm_stream
            if comm_stream is None:
                projection.prefetch(self.buffer, self.plan)
                return
            device_module = self.runtime.device_module
            comm_stream.wait_event(device_module.current_stream().record_event())
            with device_module.stream(comm_stream):
                projection.prefetch(self.buffer, self.plan)
                self._prefetch_events[id(projection)] = comm_stream.record_event()

    def await_prefetch(self, projection: MoonEPSymmetricProjection) -> None:
        """Block the compute stream until this projection's slots are filled."""
        self.wait(self._prefetch_events.pop(id(projection), None))
        trace(self.runtime, "prefetch.await", projection.role)


class MoonEPRuntime:
    """Owns every MoonEP resource associated with one torchtitan model.

    ``config`` is duck-typed through :class:`MoonEPSettings`: torchtitan passes a
    ``MoonEPTokenDispatcher.Config``, but any object exposing a subset of the
    knob names works.
    """

    def __init__(
        self,
        group,
        ep_mesh: "DeviceMesh",
        num_experts: int,
        config: Any,
        gradient_divide_factor: float | None = None,
    ):
        # ``_detect_device_type`` reports "npu" for torch_npu builds, which
        # ``torch.accelerator`` may surface as a PrivateUse1 device name instead.
        device_type = _detect_device_type()
        self.backend = get_moonep_backend(device_type)
        # Resolved once: every stream and event this runtime touches has to come
        # from this module so it stays in MoonEP's own API family.
        self.device_module = device_stream_module(device_type)
        self.group = group
        self.ep_mesh = ep_mesh
        self.rank = dist.get_rank(group=group)
        self.size = dist.get_world_size(group=group)
        self.num_experts = int(num_experts)
        # An expert's gradient collects tokens from every EP rank, and those
        # ranks hold different micro-batches, so the sum the home-rank reduction
        # produces has to be averaged the same way expert FSDP's reduce-scatter
        # averages it for the other dispatchers. Without this the experts train
        # at ``ep_size`` times the effective learning rate of the dense
        # parameters -- largely hidden by Adam's second moment, but not by weight
        # decay and not by ``Trainer``'s clip-by-norm.
        self.gradient_divide_factor = float(gradient_divide_factor or 1.0)
        self.config = MoonEPSettings.from_config(config)
        self.config.warn_about_inert_knobs(device_type)
        self.config.report_effective_async(device_type, self.backend.supports_async_comm)
        self.closed = False
        self.finalized = False
        self.hidden_size: int | None = None
        self._static_tokens_per_rank: int | None = self.config.tokens_per_rank
        self._static_top_k: int | None = self.config.top_k
        self._plan_invariants_checked = False
        #: Monotonic expert-forward counter, used only to label trace output.
        self.call_index = -1
        self._buffers: dict[tuple[int, int, int], Any] = {}
        self._projection_pools: dict[tuple[str, int, int], Any] = {}
        self._projections: list[MoonEPSymmetricProjection] = []

        if self.num_experts % self.size:
            raise RuntimeError(f"MoonEP experts ({self.num_experts}) must be divisible by EP size ({self.size}).")
        self.experts_per_rank = self.num_experts // self.size
        # Appendix E of the K3 report proves a balanced plan always exists with
        # at most E/R redundant experts per rank and that the bound is tight, so
        # one prefetch slot per local expert makes planning unconditionally
        # feasible and training uninterruptible.
        self.prefetch_slots = self.experts_per_rank
        self.backend.validate_environment(self)

    def register_hidden_size(self, hidden_size: int) -> None:
        hidden_size = int(hidden_size)
        if self.hidden_size is None:
            self.hidden_size = hidden_size
        elif self.hidden_size != hidden_size:
            raise RuntimeError(
                "All MoonEP expert modules sharing a runtime must have the same hidden "
                f"size; expected {self.hidden_size}, got {hidden_size}."
            )

    def projection_pools(self) -> list:
        """Every pooled resource set, in the deterministic order they were created."""
        return list(self._projection_pools.values())

    def projections(self) -> list:
        """Every registered projection, in the order the modules were visited."""
        return list(self._projections)

    @property
    def configured_tokens_per_rank(self) -> int | None:
        """Reserved token capacity, or ``None`` before the first lock.

        On NPU this is filled from the data config before the heap is created.
        On CUDA it stays ``None`` until the first expert forward.
        """
        return self._static_tokens_per_rank

    @property
    def static_shape(self) -> tuple[int, int, int]:
        """The locked ``(tokens_per_rank, hidden, top_k)`` triple."""
        if self._static_tokens_per_rank is None or self._static_top_k is None or self.hidden_size is None:
            raise RuntimeError("MoonEP static token shape is not known yet.")
        return self._static_tokens_per_rank, self.hidden_size, self._static_top_k

    def get_projection_pool(self, role: str, out_features: int, in_features: int):
        key = (role, out_features, in_features)
        pool = self._projection_pools.get(key)
        if pool is None:
            pool = self.backend.create_projection_pool(self, role, out_features, in_features)
            self._projection_pools[key] = pool
        return pool

    def distribute_projection(
        self, role: str, source: torch.Tensor, ep_mesh: "DeviceMesh", transposed: bool = False
    ) -> MoonEPSymmetricProjection:
        if self.finalized:
            raise RuntimeError("MoonEP symmetric weights must be allocated before the runtime is finalized.")
        _validate_projection_source(role, source, self.size, self.backend.device_type)
        projection = MoonEPSymmetricProjection(self, role, source, ep_mesh, transposed)
        self._projections.append(projection)
        return projection

    def finalize(self) -> None:
        """Complete initialization once every expert module is parallelized.

        Ascend requires this: its symmetric heap is initialized once from the
        descriptors of every registered allocation, so all communication buffers
        and expert weights must exist first. On CUDA this only pre-creates the
        communication buffer when the static token shape is configured.
        """
        if self.finalized:
            return
        if not self._projections:
            raise RuntimeError("MoonEP runtime was finalized without any expert projections.")
        if self._static_tokens_per_rank is not None and self._static_top_k is not None:
            self._create_buffer(self._static_tokens_per_rank, self.hidden_size, self._static_top_k)
        elif self.backend.requires_preconfigured_static_shape:
            raise RuntimeError(
                f"The MoonEP {self.backend.device_type} backend needs a static "
                "tokens_per_rank and top_k before the symmetric heap is initialized. "
                "Both are derived automatically: S from data.batch_size * "
                "data.max_seq_length / (cp * tp), K from the model's routed top-k. "
                "Set data.max_seq_length (and a positive batch_size) or override "
                "moonep_config if derivation could not run."
            )
        self.backend.initialize_symmetric_heap(self)
        self._prewarm_collectives()
        self.finalized = True

    def _prewarm_collectives(self) -> None:
        """Compile the collective kernels now rather than in the first backward.

        A kernel first compiled inside the backward pass deadlocks. Not fatal if
        a backend cannot do it, since the kernels still compile lazily, so a
        failure only warns.
        """
        buffer = next(iter(self._buffers.values()), None)
        if buffer is None:
            return
        trace(self, "prewarm.collectives enter")
        try:
            self.backend.prewarm_collectives(self, buffer)
        except Exception as error:
            logger.warning(
                "MoonEP could not pre-compile its collective kernels (%s). They will "
                "be compiled on first use instead, which has been seen to hang in the "
                "backward pass; report this if the run stalls at the first step.",
                error,
            )
        trace(self, "prewarm.collectives exit")

    def should_check_plan_invariants(self) -> bool:
        """Run the device-side plan checks on the first call, then opt-in only."""
        if self.config.validate_plan_invariants:
            return True
        if self._plan_invariants_checked:
            return False
        self._plan_invariants_checked = True
        return True

    def new_call(self, tokens_per_rank: int, hidden: int, top_k: int) -> MoonEPCallState:
        if self._static_tokens_per_rank is None:
            self._static_tokens_per_rank = tokens_per_rank
        elif self._static_tokens_per_rank != tokens_per_rank:
            raise RuntimeError(
                f"MoonEP uses a static token shape: expected S={self._static_tokens_per_rank}, "
                f"got S={tokens_per_rank}. Pad/drop the batch or use another dispatcher."
            )
        if self._static_top_k is None:
            self._static_top_k = top_k
        elif self._static_top_k != top_k:
            raise RuntimeError(f"MoonEP uses a static top-k: expected K={self._static_top_k}, got K={top_k}.")

        key = (tokens_per_rank, hidden, top_k)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self._create_buffer(tokens_per_rank, hidden, top_k)
        self.call_index += 1
        return MoonEPCallState(self, buffer, tokens_per_rank, hidden, top_k)

    def _create_buffer(self, tokens_per_rank: int, hidden: int, top_k: int):
        if hidden is None:
            raise RuntimeError("MoonEP cannot create a buffer before the hidden size is known.")
        key = (tokens_per_rank, hidden, top_k)
        buffer = self._buffers.get(key)
        if buffer is not None:
            return buffer
        if self.finalized and self.backend.requires_preconfigured_static_shape:
            raise RuntimeError(
                f"The MoonEP {self.backend.device_type} backend cannot create a "
                "communication buffer after the symmetric heap is initialized; the "
                f"static shape (S={tokens_per_rank}, K={top_k}) differs from the "
                "configured one."
            )
        buffer = self.backend.create_buffer(self, tokens_per_rank, hidden, top_k)
        self._buffers[key] = buffer
        self.backend.prepare_buffer(self, buffer)
        return buffer

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.backend.release(self)
        for buffer in self._buffers.values():
            buffer.destroy()
        self._buffers.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.group)
        for projection in self._projections:
            projection.close()
        self._projections.clear()
        for pool in self._projection_pools.values():
            pool.close()
        self._projection_pools.clear()


def _validate_projection_source(name: str, tensor: torch.Tensor, ep_size: int, device_type: str) -> None:
    """Reject expert weights MoonEP's kernels cannot consume."""
    if tensor.device.type != device_type:
        raise RuntimeError(
            f"MoonEP {name} must live on {device_type}, got {tensor.device}. "
            "MoonEP maps expert weights into symmetric memory while the torchtitan "
            "integration swaps them in, so the model must already be on the "
            "accelerator at that point."
        )
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"MoonEP {name} must be BF16, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise RuntimeError(f"MoonEP {name} must be contiguous.")
    if tensor.ndim != 3:
        raise RuntimeError(
            f"MoonEP {name} must have [experts, out_features, in_features] shape, got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] % ep_size:
        raise RuntimeError(f"MoonEP {name} expert dimension {tensor.shape[0]} is not divisible by EP size {ep_size}.")
    if tensor.shape[-2] % 128 or tensor.shape[-1] % 128:
        raise RuntimeError(
            f"MoonEP {name} matrix dimensions must be multiples of 128 for the prefetch and "
            f"gradient kernels, got {tuple(tensor.shape[-2:])}."
        )
