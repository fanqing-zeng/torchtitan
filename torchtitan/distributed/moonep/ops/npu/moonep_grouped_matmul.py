# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Ascend MoonEP grouped-GEMM primitives built on ``npu_grouped_matmul``.

MoonEP's cumulative group ends map directly onto ``group_list_type=0``; the only
conversion is an on-device cast of the tiny ``[G]`` tensor to int64, so nothing
synchronizes.

The weight operand is one transposed 3-D view rather than a 2-D view per group,
which would cost ``2 * E/R`` Python view constructions per GEMM per layer. Older
CANN builds reject a strided 3-D weight, so the first failure falls back to the
per-group form for the rest of the process.
"""

from __future__ import annotations

import logging

import torch

from torchtitan.distributed.moonep.ops.registry import register_op

try:
    import torch_npu
except ImportError:  # pragma: no cover - import guard; dispatch fails loudly later
    torch_npu = None


logger = logging.getLogger(__name__)

_warned_messages: set[str] = set()


def _log_warning_once(message: str) -> None:
    if message not in _warned_messages:
        logger.warning(message)
        _warned_messages.add(message)


# ``None`` means "not probed yet"; ``True`` means the kernel accepted a single
# strided 3-D weight operand.
_SUPPORTS_STRIDED_3D_WEIGHT: bool | None = None


def _group_list(group_ends: torch.Tensor) -> torch.Tensor:
    return group_ends if group_ends.dtype == torch.int64 else group_ends.to(torch.int64)


def _weight_operands(weights: torch.Tensor, transpose: bool):
    """Build the ``weight`` argument in ``[G, K, N]`` orientation.

    ``weights`` is stored in the linear-layer convention ``[G, out, in]``.
    ``transpose=True`` requests ``[G, in, out]`` (forward and wgrad consumers),
    ``transpose=False`` keeps ``[G, out, in]`` (dgrad consumer).
    """
    oriented = weights.transpose(-2, -1) if transpose else weights
    if _SUPPORTS_STRIDED_3D_WEIGHT is not False:
        return [oriented]
    return list(oriented.unbind(0))


def _grouped_matmul(activation: torch.Tensor, weights: torch.Tensor, group_ends: torch.Tensor, transpose: bool):
    global _SUPPORTS_STRIDED_3D_WEIGHT
    group_list = _group_list(group_ends)
    try:
        output = torch_npu.npu_grouped_matmul(
            [activation],
            _weight_operands(weights, transpose),
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )[0]
    except RuntimeError:
        if _SUPPORTS_STRIDED_3D_WEIGHT is False:
            raise
        _SUPPORTS_STRIDED_3D_WEIGHT = False
        _log_warning_once(
            "npu_grouped_matmul rejected a strided 3-D weight operand; falling back "
            "to the per-group 2-D form for MoonEP grouped GEMMs."
        )
        return _grouped_matmul(activation, weights, group_ends, transpose)
    if _SUPPORTS_STRIDED_3D_WEIGHT is None:
        _SUPPORTS_STRIDED_3D_WEIGHT = True
    return output


@register_op("moonep_grouped_matmul", "npu")
def moonep_grouped_matmul_npu(inputs, group_ends, weights, weights_transposed=False):
    # A transposed layout is already in the kernel's [G, K, N] orientation.
    return _grouped_matmul(inputs, weights, group_ends, transpose=not weights_transposed)


@register_op("moonep_grouped_matmul_dgrad", "npu")
def moonep_grouped_matmul_dgrad_npu(grad_output, group_ends, weights, weights_transposed=False):
    return _grouped_matmul(grad_output, weights, group_ends, transpose=weights_transposed)


@register_op("moonep_grouped_matmul_wgrad", "npu")
def moonep_grouped_matmul_wgrad_npu(grad_output, inputs, group_ends, group_ends_host=None, weights_transposed=False):
    # Ascend exposes the K-split contraction as a first-class `group_type`, so
    # the single-kernel form is always taken here; the host offsets and the
    # `fused` request only matter to backends that fall back to a loop.
    # K-split grouped GEMM: contract along the token dimension and emit one
    # [in_features, out_features] matrix per group. That is already the stored
    # layout when weights_transposed, so only the [G, out, in] convention needs
    # the final view; callers copy slices of it into MoonEP's FP32 reduction
    # buffers, which handles strides.
    grad_weights = torch_npu.npu_grouped_matmul(
        [inputs.transpose(0, 1)],
        [grad_output],
        bias=None,
        group_list=_group_list(group_ends),
        split_item=3,
        group_type=2,
        group_list_type=0,
    )[0]
    return grad_weights if weights_transposed else grad_weights.transpose(-2, -1)
