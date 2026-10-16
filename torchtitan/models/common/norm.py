# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared normalization modules."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.activation import Sigmoid, SiLU, UnaryActivationFn
from torchtitan.protocols.module import Module

# Shape suffixes: D = normalized feature dimension; leading dimensions are batch axes.


@lru_cache(maxsize=1)
def _register_fla_norm_gated_once() -> None:
    from fla.modules.fused_norm_gate import LayerNormGatedFunction

    spmd.register_decomposition(LayerNormGatedFunction, _fla_norm_gated_reference)


@torch.compiler.assume_constant_result
def _register_fla_norm_gated() -> None:
    # Registration is independent of tensors and must stay outside compiled graphs.
    _register_fla_norm_gated_once()


def _load_fla_rms_norm_gated():
    _register_fla_norm_gated()
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated


def _fla_norm_gated_reference(
    x_D: torch.Tensor,
    gate_D: torch.Tensor,
    weight_D: torch.Tensor | None,
    bias_D: torch.Tensor | None,
    activation: str,
    residual_D: torch.Tensor | None,
    eps: float,
    prenorm: bool,
    residual_in_fp32: bool,
    is_rms_norm: bool,
):
    """Pure torch decomposition for FLA's autograd function's SPMD types."""
    residual_out_D = x_D.float() if residual_in_fp32 else x_D
    if residual_D is not None:
        residual_out_D = residual_out_D + residual_D
    norm_input_D = residual_out_D.float()
    if is_rms_norm:
        normalized_D = F.rms_norm(norm_input_D, (x_D.shape[-1],), None, eps)
    else:
        normalized_D = F.layer_norm(norm_input_D, (x_D.shape[-1],), eps=eps)
    if weight_D is not None:
        normalized_D = normalized_D * weight_D.float()
    if bias_D is not None:
        normalized_D = normalized_D + bias_D.float()
    gate_D = gate_D.float()
    gate_D = gate_D.sigmoid() if activation == "sigmoid" else F.silu(gate_D)
    output_D = (normalized_D * gate_D).to(x_D.dtype)
    return (output_D, residual_out_D) if prenorm else output_D


class GatedRMSNorm(Module):
    """Apply RMSNorm followed by a unary gate activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float
        activation_fn: UnaryActivationFn.Config
        implementation: Literal["eager", "fla"] = "eager"

        def __post_init__(self) -> None:
            if self.implementation not in ("eager", "fla"):
                raise ValueError(
                    f"Unsupported GatedRMSNorm implementation: {self.implementation}"
                )
            if self.implementation == "fla" and not isinstance(
                self.activation_fn, (Sigmoid.Config, SiLU.Config)
            ):
                raise ValueError(
                    "FLA GatedRMSNorm supports only Sigmoid and SiLU gates"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.activation_fn = config.activation_fn.build()
        self.implementation = config.implementation
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # Preserve the public argument names used by sharding configurations.
        x_D, gate_D = x, gate
        if self.implementation == "fla" and x_D.device.type == "npu":
            # FLA's fused_norm_gate always normalizes before applying its output gate.
            return _load_fla_rms_norm_gated()(
                x_D,
                gate_D,
                self.weight,
                None,
                activation="sigmoid"
                if isinstance(self.activation_fn, Sigmoid)
                else "silu",
                residual=None,
                prenorm=False,
                eps=self.eps,
            )
        return self._forward_eager(x_D, gate_D)

    @local_compile("gated_rmsnorm", batch_invariant=False)
    def _forward_eager(
        self,
        x_D: torch.Tensor,
        gate_D: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x_D.dtype
        normalized_D = F.rms_norm(
            x_D.float(),
            (x_D.shape[-1],),
            self.weight.float(),
            self.eps,
        )
        return (normalized_D * self.activation_fn(gate_D.float())).to(input_dtype)


__all__ = ["GatedRMSNorm"]
