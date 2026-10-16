# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable wrappers around standard ``torch.nn`` modules.

Each class uses diamond inheritance (``nn.X`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- All ``nn.X`` logic (forward, state_dict, etc.) is reused as-is.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.

Each ``Config`` only exposes the fields that current callsites set;
add more if a new callsite needs them.
"""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.protocols.module import Module

# Shape suffixes: D = normalized feature dimension, R = per-row RMS statistics.


class _AscendCRMSNorm(torch.autograd.Function):
    """Expose the fused forward/backward pair to the SPMD type checker."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, x_D: torch.Tensor, weight_D: torch.Tensor, eps: float
    ):
        import torch_npu

        output_D, rstd_R = torch_npu.npu_rms_norm(x_D, weight_D, epsilon=eps)
        ctx.save_for_backward(x_D, weight_D, rstd_R)
        return output_D

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_D: torch.Tensor):  # pyrefly: ignore[bad-override]
        x_D, weight_D, rstd_R = ctx.saved_tensors
        grad_x_D, grad_weight_D = torch.ops.npu.npu_rms_norm_backward(
            grad_output_D.contiguous(), x_D, weight_D, rstd_R
        )
        return grad_x_D, grad_weight_D, None


@spmd.register_decomposition(_AscendCRMSNorm)
def _rms_norm_reference(
    x_D: torch.Tensor, weight_D: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.rms_norm(x_D, weight_D.shape, weight_D, eps)


class Conv1d(nn.Conv1d, Module):
    """Configurable nn.Conv1d."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        groups: int = 1
        # Matches the upstream ``nn.Conv1d`` default (differs from
        # ``Linear.Config.bias``, which defaults to False).
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.groups,
            bias=config.bias,
        )


class Conv2d(nn.Conv2d, Module):
    """Configurable nn.Conv2d."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        # Matches the upstream ``nn.Conv2d`` default (differs from
        # ``Linear.Config.bias``, which defaults to False).
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.bias,
        )


class GELU(nn.GELU, Module):
    """Configurable nn.GELU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        approximate: str = "none"

    def __init__(self, config: Config):
        super().__init__(approximate=config.approximate)


class GroupNorm(nn.GroupNorm, Module):
    """Configurable nn.GroupNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_groups: int
        num_channels: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__(
            config.num_groups,
            config.num_channels,
            eps=config.eps,
        )


class Identity(nn.Identity, Module):
    """Configurable nn.Identity."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()


class LayerNorm(nn.LayerNorm, Module):
    """Configurable nn.LayerNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )


class RMSNorm(nn.RMSNorm, Module):
    """Configurable RMSNorm with an optional torch_npu AscendC kernel on NPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True
        implementation: Literal["eager", "ascendc"] = "eager"

        def __post_init__(self) -> None:
            if self.implementation not in ("eager", "ascendc"):
                raise ValueError(
                    f"Unsupported RMSNorm implementation: {self.implementation}"
                )

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.implementation = config.implementation

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self, input: torch.Tensor
    ) -> torch.Tensor:
        # Preserve nn.RMSNorm's argument name used by norm sharding configs.
        # D denotes the normalized feature dimension.
        x_D = input
        if self.implementation == "ascendc" and x_D.device.type == "npu":
            weight_D = self.weight
            if weight_D is None:
                weight_D = x_D.new_ones(self.normalized_shape)
            eps = torch.finfo(x_D.dtype).eps if self.eps is None else self.eps
            return _AscendCRMSNorm.apply(x_D, weight_D, eps)
        return super().forward(x_D)


__all__ = [
    "Conv1d",
    "Conv2d",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "RMSNorm",
]
