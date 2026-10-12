# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Device-dispatched grouped GEMMs over MoonEP's cumulative group offsets.

The three GEMMs of a grouped linear layer are exposed separately, because the
MoonEP expert forward composes them into autograd nodes that route the weight
gradient into their own FP32 reduction buffers.
"""

from __future__ import annotations

import torch

from torchtitan.distributed.moonep.ops.registry import dispatch_op


def _device_type(tensor: torch.Tensor) -> str:
    """Resolve the registry key from the operand, not the global accelerator."""
    return tensor.device.type


def _validate(
    inputs: torch.Tensor,
    group_ends: torch.Tensor,
    num_groups: int,
    weights: torch.Tensor | None = None,
) -> None:
    if inputs.ndim != 2:
        raise RuntimeError(f"MoonEP grouped GEMM expects a 2-D activation, got {tuple(inputs.shape)}.")
    if group_ends.ndim != 1 or group_ends.numel() != num_groups:
        raise RuntimeError(
            f"MoonEP grouped GEMM needs one cumulative end offset per group; "
            f"got {tuple(group_ends.shape)} for {num_groups} groups."
        )
    if not inputs.is_contiguous():
        raise RuntimeError("MoonEP grouped GEMM requires contiguous activations.")
    if weights is not None and not weights.is_contiguous():
        # The device kernels consume a transposed view of the weights, which is
        # only a valid operand layout when the base tensor is contiguous.
        raise RuntimeError("MoonEP grouped GEMM requires contiguous expert weights.")


def moonep_grouped_matmul(
    inputs: torch.Tensor,
    group_ends: torch.Tensor,
    weights: torch.Tensor,
    weights_transposed: bool = False,
) -> torch.Tensor:
    """``out[m] = inputs[m] @ W[group(m)].T`` for a semantic ``W`` of ``[out, in]``.

    ``inputs`` is ``[M, in]`` and ``group_ends[-1]`` must equal ``M`` so the whole
    static buffer is covered. ``weights_transposed`` says the stored last two
    dimensions are ``[in, out]`` rather than ``[out, in]``; the arithmetic is
    the same either way.

    The flag is passed rather than inferred from the shapes, because ``out == in``
    happens (a square expert matrix) and then the two layouts are
    indistinguishable.
    """
    _validate(inputs, group_ends, weights.shape[0], weights)
    return dispatch_op(
        "moonep_grouped_matmul",
        inputs,
        group_ends,
        weights,
        weights_transposed,
        device_type=_device_type(inputs),
    )


def moonep_grouped_matmul_dgrad(
    grad_output: torch.Tensor,
    group_ends: torch.Tensor,
    weights: torch.Tensor,
    weights_transposed: bool = False,
) -> torch.Tensor:
    """``grad_inputs[m] = grad_output[m] @ W[group(m)]`` for a semantic ``[out, in]``."""
    _validate(grad_output, group_ends, weights.shape[0], weights)
    return dispatch_op(
        "moonep_grouped_matmul_dgrad",
        grad_output,
        group_ends,
        weights,
        weights_transposed,
        device_type=_device_type(grad_output),
    )


def moonep_grouped_matmul_wgrad(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    group_ends: torch.Tensor,
    group_ends_host: list[int] | None = None,
    weights_transposed: bool = False,
) -> torch.Tensor:
    """``grad_W[g] = grad_output[rows(g)].T @ inputs[rows(g)]``.

    Emitted in the weights' own layout -- ``[G, out, in]``, or ``[G, in, out]``
    under ``weights_transposed`` -- so the caller can slice it straight into
    MoonEP's gradient buffers, which mirror that layout. ``group_ends_host`` lets
    a backend that iterates over groups skip the device-to-host read.
    """
    _validate(inputs, group_ends, group_ends.numel())
    if grad_output.shape[0] != inputs.shape[0]:
        raise RuntimeError(f"wgrad operands disagree on the token count: {grad_output.shape[0]} vs {inputs.shape[0]}.")
    return dispatch_op(
        "moonep_grouped_matmul_wgrad",
        grad_output,
        inputs,
        group_ends,
        group_ends_host,
        weights_transposed,
        device_type=_device_type(inputs),
    )
