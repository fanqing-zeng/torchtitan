# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Rank-wide runtime for MoonEP routed experts on Ascend NPUs.

Shape suffixes use ``S`` for tokens per rank, ``K`` for selected experts,
``E`` for routed experts, and ``D`` for the model (latent) dimension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor, Shard

from torchtitan.config import Configurable
from torchtitan.distributed.moonep import (
    MoonEPRuntime,
    get_moonep_experts_forward_fn,
)
from torchtitan.distributed.moonep.settings import MoonEPSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.distributed import DeviceMesh

    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.common.moonep.routed_experts import MoonEPRoutedExperts


logger = logging.getLogger(__name__)

__all__ = ["MoonEPTrainingRuntime"]


def _local_expert_source(
    source: torch.Tensor, runtime: MoonEPRuntime, ep_mesh: DeviceMesh
) -> torch.Tensor:
    """Present an already-sharded plain expert weight to the vendored runtime.

    ``MoonEPSymmetricProjection`` accepts a DTensor (``to_local()`` recovers
    this rank's slice) or a full plain tensor (it slices
    ``[rank*epn : (rank+1)*epn]`` itself). The new architecture shards expert
    weights in place during ``Module.parallelize``, so what reaches this swap
    is usually a plain per-rank slice whose dim 0 already equals
    ``experts_per_rank``; the vendored slicing would cut it to nothing on every
    rank but 0. Wrapping the slice into a zero-copy Shard(0) DTensor
    (``to_local()`` returns the same storage) steers it through the DTensor
    branch unchanged. Full plain tensors and DTensor sources pass through
    untouched.
    """
    if (
        not isinstance(source, DTensor)
        and source.shape[0] == runtime.experts_per_rank
    ):
        return DTensor.from_local(source, ep_mesh, (Shard(0),), run_check=False)
    return source


def _validate_parallel_layout(
    parallelism_context: ParallelismContext, device: torch.device
) -> list[str]:
    """Collect every v1 parallel-layout violation for one combined error."""
    errors: list[str] = []
    if device.type != "npu":
        errors.append(
            f"a NPU device (found {device.type}); the vendored MoonEP backend "
            "is Ascend-only"
        )
    if not parallelism_context.ep_enabled:
        errors.append(
            "expert_parallel_degree > 1 (MoonEP requires expert parallelism)"
        )
    if parallelism_context.pp_enabled:
        errors.append("pipeline_parallel_degree == 1 (MoonEP v1 does not support PP)")
    if parallelism_context.dp_replicate_enabled:
        errors.append(
            "dp_replicate_degree == 1 (MoonEP v1 does not support replicated DP)"
        )
    if (
        parallelism_context.dp_shard * parallelism_context.cp * parallelism_context.tp
        != parallelism_context.ep
    ):
        errors.append(
            "dp_shard * cp * tp == expert_parallel_degree (expert weights must "
            "be sharded by expert parallelism only; MoonEP replaces that "
            "sharding with symmetric memory)"
        )
    return errors


class MoonEPTrainingRuntime(Configurable):
    """Own one vendored MoonEP runtime shared by all local expert modules.

    Built by the training engine after model initialization and before the
    optimizer is created: the weight swap deregisters ``w13``/``w2`` and
    registers symmetric-memory ``gate_up_proj``/``down_proj`` parameters, so
    the optimizer and checkpointer only ever see the final parameter set.
    Closing releases the symmetric heap; the modules must not run afterwards.

    Args:
        config: User-selected symmetric-memory policy knobs.
        model_parts: Final local model or pipeline-stage modules.
        parallelism_context: Final distributed mesh topology.
        device: NPU device that owns the symmetric heap and buffers.
        num_tokens_per_microbatch_per_dp_rank: Unsharded token count used to
            derive MoonEP's static per-rank token capacity after CP and TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configure MoonEP's rank-wide symmetric-memory policy.

        Every field maps one-to-one onto a
        :class:`~torchtitan.distributed.moonep.settings.MoonEPSettings` knob.
        ``tokens_per_rank`` and ``top_k`` default to ``None`` and are derived
        from the training data shape and the expert modules unless overridden.
        """

        num_sms: int = 32
        token_padding: int = 128
        comm_stream_priority: int = -1
        async_finish: bool = False
        tokens_per_rank: int | None = None
        top_k: int | None = None
        recompute_dispatch: bool = False
        validate_plan_invariants: bool = False
        shmem_ip_port: str | None = None
        shmem_stream_barrier: bool = True

    def __init__(
        self,
        config: Config,
        *,
        model_parts: Sequence[torch.nn.Module],
        parallelism_context: ParallelismContext,
        device: torch.device,
        num_tokens_per_microbatch_per_dp_rank: int,
    ) -> None:
        from .routed_experts import MoonEPRoutedExperts

        self.config = config
        self._closed = False
        self._modules = tuple(
            dict.fromkeys(
                module
                for model_part in model_parts
                for module in model_part.modules()
                if isinstance(module, MoonEPRoutedExperts)
            )
        )
        if not self._modules:
            raise ValueError("MoonEP runtime requires at least one expert module")

        errors = _validate_parallel_layout(parallelism_context, device)
        if errors:
            raise ValueError(
                "MoonEP (ascend_moonep) requires:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )

        ep_mesh = parallelism_context.get_optional_mesh(
            "ep", include_singleton_axes=True
        )
        if ep_mesh is None:
            raise RuntimeError("MoonEP requires an expert-parallel mesh")

        first = self._modules[0]
        for module in self._modules:
            if module.num_experts != first.num_experts:
                raise ValueError(
                    "All MoE layers sharing a MoonEP runtime must have the "
                    f"same expert count; got {first.num_experts} and "
                    f"{module.num_experts}."
                )

        # Static shapes: every rank processes exactly S*K routed tokens.
        num_token_shards = parallelism_context.cp * parallelism_context.tp
        tokens_per_rank = config.tokens_per_rank
        if tokens_per_rank is None:
            if num_tokens_per_microbatch_per_dp_rank % num_token_shards:
                raise ValueError(
                    "MoonEP input tokens must divide evenly across CP and TP"
                )
            tokens_per_rank = (
                num_tokens_per_microbatch_per_dp_rank // num_token_shards
            )
        top_k = config.top_k if config.top_k is not None else first.top_k
        for module in self._modules:
            if module.top_k != top_k:
                raise ValueError(
                    "All MoE layers sharing a MoonEP runtime must select the "
                    f"same top-k; expected {top_k}, got {module.top_k}."
                )

        settings = MoonEPSettings(
            num_sms=config.num_sms,
            token_padding=config.token_padding,
            comm_stream_priority=config.comm_stream_priority,
            async_finish=config.async_finish,
            tokens_per_rank=tokens_per_rank,
            top_k=top_k,
            recompute_dispatch=config.recompute_dispatch,
            validate_plan_invariants=config.validate_plan_invariants,
            shmem_ip_port=config.shmem_ip_port,
            shmem_stream_barrier=config.shmem_stream_barrier,
        )
        self._runtime = MoonEPRuntime(
            ep_mesh.get_group(),
            ep_mesh,
            first.num_experts,
            settings,
        )
        try:
            for module in self._modules:
                self._swap_module_weights(module)
            self._runtime.finalize()
            moonep_forward = get_moonep_experts_forward_fn(self._runtime)
            for module in self._modules:
                module._moonep_forward = moonep_forward
                module._runtime = self
        except Exception:
            self._runtime.close()
            raise
        logger.info(
            "MoonEP active: %d expert layer(s), EP=%d, tokens_per_rank=%d, top_k=%d.",
            len(self._modules),
            self._runtime.size,
            tokens_per_rank,
            top_k,
        )

    def _swap_module_weights(self, module: MoonEPRoutedExperts) -> None:
        """Replace one module's w13/w2 with symmetric-memory projections."""
        runtime = self._runtime
        ep_mesh = runtime.ep_mesh

        # [E, 2F, D]: flatten the stacked gate/up axis so the fused kernel's
        # chunk(2, dim=-1) of the first GEMM's output splits into the gate
        # half and the up half in the order BinaryActivationFn unbinds them.
        gate_up_source = module.w13.weight.flatten(1, 2)
        down_source = module.w2.weight

        # The fused forward reads these module attributes.
        module.num_global_experts = module.num_experts
        runtime.register_hidden_size(module.hidden_dim)

        gate_up = runtime.distribute_projection(
            "gate_up",
            _local_expert_source(gate_up_source, runtime, ep_mesh),
            ep_mesh,
            transposed=False,
        )
        down = runtime.distribute_projection(
            "down",
            _local_expert_source(down_source, runtime, ep_mesh),
            ep_mesh,
            transposed=False,
        )

        # Both projections copied the old weights into symmetric memory above,
        # so only now can the sources be deregistered. register_parameter(
        # name, None) drops them from named_parameters()/state_dict(): the
        # state dict gains gate_up_proj/down_proj and loses w13/w2
        # (documented format change).
        module.register_parameter("gate_up_proj", gate_up.parameter)
        module.register_parameter("down_proj", down.parameter)
        module.w13.register_parameter("weight", None)
        module.w2.register_parameter("weight", None)
        object.__setattr__(module, "_moonep_gate_up", gate_up)
        object.__setattr__(module, "_moonep_down", down)

    def close(self) -> None:
        """Detach modules and release the symmetric heap and buffers."""
        if self._closed:
            return
        for module in self._modules:
            if module._runtime is self:
                module._runtime = None
                module._moonep_forward = None
        self._runtime.close()
        self._closed = True
