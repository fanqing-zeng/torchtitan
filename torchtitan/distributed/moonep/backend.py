# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan: the CUDA (NVLink multicast) backend of the upstream
# adapter is not carried over, so only the abstract contract and the backend
# factory remain here.

"""Device backends for the MoonEP expert-parallel dispatcher.

The algorithm is device independent; what is not is how the symmetric
expert-weight memory is built and how the three MoonEP kernels are launched. A
backend therefore owns the library handle, the symmetric resources of one
projection or projection pool, and the two collective kernel launches.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

import abc
import importlib
import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torchtitan.distributed.moonep.runtime import MoonEPRuntime

logger = logging.getLogger(__name__)


class ProjectionPoolResources(abc.ABC):
    """Symmetric resources shared by every layer with the same matrix shape.

    Sharing is what keeps the memory footprint flat in the number of MoE layers,
    at the cost of replaying the prefetch before each layer's backward GEMMs.
    """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the pooled symmetric resources."""


class ProjectionResources(abc.ABC):
    """Per-layer, per-projection symmetric weight resources."""

    #: ``[E+B, out, in]`` view handed to the MoonEP prefetch entry point.
    full_weight: torch.Tensor
    #: Expert GEMM operand, laid out per the backend's ``expert_weight_layout``.
    gemm_weight: torch.Tensor
    #: ``[epn, out, in]`` view of this rank's home experts; the parameter payload.
    home_weight: torch.Tensor

    @abc.abstractmethod
    def close(self) -> None:
        """Release the per-projection symmetric resources."""


class MoonEPBackend(abc.ABC):
    """Device-specific half of the MoonEP integration."""

    #: ``torch`` device type this backend drives.
    device_type: str
    #: Whether MoonEP exposes a usable asynchronous communication stream.
    supports_async_comm: bool
    #: Whether ``S``/``K`` must be known before the first forward.
    requires_preconfigured_static_shape: bool
    #: ``"full"`` when the whole ``[E+B]`` mapping is readable, else ``"compact"``.
    expert_weight_layout: str
    #: Whether MoonEP leaves the rows past the last live group undefined.
    has_static_tail: bool

    @abc.abstractmethod
    def validate_environment(self, runtime: "MoonEPRuntime") -> None:
        """Fail fast when the accelerator or topology cannot run MoonEP."""

    @abc.abstractmethod
    def create_buffer(self, runtime: "MoonEPRuntime", tokens_per_rank: int, hidden: int, top_k: int) -> Any:
        """Create the MoonEP communication buffer for one static token shape."""

    def prepare_buffer(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Hook for one-time work on a freshly created communication buffer."""

    @abc.abstractmethod
    def create_projection_pool(
        self, runtime: "MoonEPRuntime", role: str, out_features: int, in_features: int
    ) -> ProjectionPoolResources:
        """Allocate the prefetch slots and FP32 reduction buffers for a shape."""

    @abc.abstractmethod
    def create_projection(
        self, runtime: "MoonEPRuntime", pool: ProjectionPoolResources, out_features: int, in_features: int
    ) -> ProjectionResources:
        """Allocate one layer's home experts and build its weight views."""

    @abc.abstractmethod
    def prefetch(self, runtime: "MoonEPRuntime", projection: ProjectionResources, buffer: Any, plan: Any) -> None:
        """Copy this step's redundant expert weights into the prefetch slots."""

    # Three phases: the kernel reads every peer's slot gradients before its own
    # internal barrier.
    @abc.abstractmethod
    def stage_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        home_grad: torch.Tensor,
        slot_grad: torch.Tensor,
    ) -> None:
        """Publish ``[epn, out, in]`` home and ``[B, out, in]`` slot gradients."""

    def rendezvous_before_reduce(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Make every rank's staged gradients visible to every other rank."""

    @abc.abstractmethod
    def reduce_staged_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        buffer: Any,
        plan: Any,
    ) -> torch.Tensor:
        """Fold the staged slot gradients back into their home ranks.

        Returns the ``[epn, out, in]`` FP32 gradient for this rank's experts,
        including every other rank's contributions.
        """

    def initialize_symmetric_heap(self, runtime: "MoonEPRuntime") -> None:
        """Hook for backends with a one-shot collective heap initialisation."""

    def prewarm_collectives(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Hook for compiling collective kernels before the first real step."""

    def release(self, runtime: "MoonEPRuntime") -> None:
        """Hook for backend-wide teardown that must precede resource release."""


def validate_chunk_granularity(
    role: str,
    experts_per_rank: int,
    out_features: int,
    in_features: int,
    item_size: int,
    granularity: int,
) -> None:
    """Require the per-rank expert block to be virtual-memory aligned.

    The compact ``[epn+B]`` GEMM view maps the home block and the prefetch slots
    into adjacent virtual addresses. That view is only row-exact when each
    block's byte size is a whole number of allocation granules; otherwise the
    allocator's padding would shift the slot rows.
    """
    chunk_bytes = experts_per_rank * out_features * in_features * item_size
    if chunk_bytes % granularity:
        raise RuntimeError(
            f"MoonEP {role} per-rank expert block is {chunk_bytes} bytes, which is not a "
            f"multiple of the {granularity}-byte virtual-memory granularity. Choose expert "
            "matrix dimensions such that (E/R) * out_features * in_features * itemsize is "
            "granularity aligned."
        )


_BACKEND_FACTORIES: dict[str, str] = {
    "npu": "torchtitan.distributed.moonep.backend_npu:NpuMoonEPBackend",
}


def get_moonep_backend(device_type: str) -> MoonEPBackend:
    """Instantiate the MoonEP backend for ``device_type``.

    Backends are resolved lazily so that importing this module never requires
    the vendor-specific extension (``ascend_moonep``) to be installed.
    """
    target = _BACKEND_FACTORIES.get(device_type)
    if target is None:
        raise RuntimeError(
            f"No MoonEP backend for device type {device_type!r}; supported types are {sorted(_BACKEND_FACTORIES)}. "
            "The torchtitan integration vendors the Ascend NPU backend only."
        )
    module_name, class_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Failed to import the MoonEP {device_type} backend: {exc}") from exc
    return getattr(module, class_name)()
