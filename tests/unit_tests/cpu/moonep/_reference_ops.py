# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep"): the CPU reference primitives
# torchtitan did not vendor, registered on the op registry for the unit tests.

"""Reference MoonEP grouped-GEMM primitives.

torchtitan vendors only the Ascend NPU kernels, so these implementations --
unchanged from FSDPTurbo's originals: they read the group boundaries back to
the host and loop -- are registered on the CPU slot of the op registry as the
numerical oracle for the unit tests. The accelerator paths must stay
bit-comparable with these within GEMM tolerance.

Importing this module has the side effect of registering the ops, which is why
the tests do ``from . import _reference_ops  # noqa: F401``.
"""

from __future__ import annotations

import torch

from torchtitan.distributed.moonep.ops.registry import register_op


def _group_slices(group_ends: torch.Tensor):
    """Yield ``(group_index, start, end)`` for every group, including empty ones."""
    ends = group_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
    start = 0
    for group, end in enumerate(ends):
        if end < start:
            raise RuntimeError(f"MoonEP group ends must be non-decreasing; group {group} ends at {end} after {start}.")
        yield group, start, end
        start = end


def _semantic(weights: torch.Tensor, weights_transposed: bool) -> torch.Tensor:
    """View the stored weights in the ``[G, out, in]`` convention."""
    return weights.transpose(-2, -1) if weights_transposed else weights


@register_op("moonep_grouped_matmul", "cpu")
def moonep_grouped_matmul_cpu(inputs, group_ends, weights, weights_transposed=False):
    weights = _semantic(weights, weights_transposed)
    output = inputs.new_zeros((inputs.shape[0], weights.shape[1]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            output[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
    return output


@register_op("moonep_grouped_matmul_dgrad", "cpu")
def moonep_grouped_matmul_dgrad_cpu(grad_output, group_ends, weights, weights_transposed=False):
    weights = _semantic(weights, weights_transposed)
    grad_inputs = grad_output.new_zeros((grad_output.shape[0], weights.shape[2]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            grad_inputs[start:end] = grad_output[start:end] @ weights[group]
    return grad_inputs


@register_op("moonep_grouped_matmul_wgrad", "cpu")
def moonep_grouped_matmul_wgrad_cpu(grad_output, inputs, group_ends, group_ends_host=None, weights_transposed=False):
    # The reference always loops, so it has no use for either hint.
    left, right = (inputs, grad_output) if weights_transposed else (grad_output, inputs)
    grad_weights = inputs.new_zeros((group_ends.numel(), left.shape[1], right.shape[1]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            grad_weights[group] = left[start:end].transpose(0, 1) @ right[start:end]
    return grad_weights
