# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""TorchTitan module boundary for MoonEP routed experts.

Shape suffixes use ``T`` for local input tokens, ``K`` for selected experts,
``E`` for routed experts, ``F`` for the expert intermediate dimension, and
``D`` for the model (latent) dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.protocols.module import Module

from .runtime import MoonEPTrainingRuntime


class MoonEPRoutedExperts(Module):
    """BF16 routed experts executed by the Ascend MoonEP backend.

    W13 and W2 are built as ordinary TorchTitan modules so parameter, FSDP,
    optimizer, and checkpoint ownership does not change during model
    construction and weight initialization. The training runtime then swaps
    them for symmetric-memory ``gate_up_proj``/``down_proj`` parameters before
    the optimizer is built. After that swap their ``forward`` methods are not
    called: MoonEP owns dispatch, the expert activation, expert GEMMs, and
    combine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure the expert shapes MoonEP's kernels consume.

        Args:
            w13: Grouped gate/up projection with stored shape ``(E, 2, F, D)``.
            w2: Grouped down projection with stored shape ``(E, D, F)``.
            top_k: Number of experts selected for each local input token.
            activation_fn: Gated expert activation, passed through unchanged
                from the model flavor. MoonEP applies it through
                ``_apply_activation`` with the concatenated ``[..., 2F]``
                first-GEMM output reshaped to ``[..., 2, F]``.
            output_postprocess: Unsupported by MoonEP v1; must be ``None``.
        """

        w13: GroupedLinear.Config
        w2: GroupedLinear.Config
        top_k: int
        activation_fn: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)
        output_postprocess: Module.Config | None = None

        def __post_init__(self) -> None:
            """Validate the expert dimensions consumed by the MoonEP kernels."""
            if self.w13.group_size != self.w2.group_size:
                raise ValueError("w13 and w2 must contain the same number of experts")
            if self.w13.in_features != self.w2.out_features:
                raise ValueError("w13 input and w2 output dimensions must match")
            if self.w13.num_linears != 2:
                raise ValueError("w13 output must contain gate and up projections")
            if self.w13.out_features != self.w2.in_features:
                raise ValueError("w13 output and w2 input dimensions must match")
            if self.w2.num_linears != 1:
                raise ValueError("w2 must contain one down projection")
            if self.top_k <= 0:
                raise ValueError("top_k must be positive")
            if self.output_postprocess is not None:
                raise ValueError("MoonEP v1 does not support an output postprocess")

    def __init__(self, config: Config):
        Module.__init__(self)
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.num_experts = config.w13.group_size
        self.top_k = config.top_k
        self._runtime: MoonEPTrainingRuntime | None = None
        self._moonep_forward: Callable[..., torch.Tensor] | None = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Leave communication and activation storage to the shared runtime."""
        del buffer_device

    def _apply_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply the expert activation to the first GEMM's concatenated output.

        MoonEP's dispatcher calls this with the raw ``[..., 2F]`` output of the
        gate_up grouped GEMM. ``BinaryActivationFn`` takes ``[..., 2, F]``, so
        reshape first; the view is zero-copy.
        """
        return self.activation_fn(
            gate_up.view(*gate_up.shape[:-1], 2, -1), offsets=None
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused MoonEP dispatch, expert computation, and combine.

        Args:
            x_TD: Local input tokens with model (latent) dimension ``D``.
            topk_scores_TK: Selected routing weights for ``K`` experts.
            topk_expert_ids_TK: Selected global expert IDs.
            num_local_tokens_per_expert_E: Router statistics retained by the
                surrounding MoE module; MoonEP derives dispatch metadata from
                the selected IDs.

        Returns:
            Combined local expert output with shape ``(T, D)``.
        """
        del num_local_tokens_per_expert_E
        moonep_forward = self._moonep_forward
        if moonep_forward is None:
            raise RuntimeError("MoonEP context is not initialized")
        # MoonEP's fused forward takes (hidden, expert ids, route weights).
        return moonep_forward(self, x_TD, topk_expert_ids_TK, topk_scores_TK)
