# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the MoonEP grouped-GEMM primitives.

The CPU reference (``_reference_ops.py``, registered on import below) is the
numerical oracle; on a machine with an accelerator the same assertions run
against the device primitives. FP32 on CPU for a tight check, BF16 elsewhere
because the fused kernels accept nothing else.
"""

import pytest
import torch

# Registers the CPU reference ops on the op registry (torchtitan vendors only
# the accelerator kernels).
from . import _reference_ops  # noqa: F401
from torchtitan.distributed.moonep.ops.moonep_grouped_matmul import (
    moonep_grouped_matmul,
    moonep_grouped_matmul_dgrad,
    moonep_grouped_matmul_wgrad,
)

# One entry per accelerator that has MoonEP primitives registered.
DEVICES = ["cpu"]
if hasattr(torch, "npu") and torch.npu.is_available():  # pragma: no cover - Ascend only
    DEVICES.append("npu")


def _reference(inputs, group_ends, weights):
    """Dense per-group reference built from plain ``torch`` matmuls."""
    output = torch.zeros(inputs.shape[0], weights.shape[1], dtype=inputs.dtype, device=inputs.device)
    start = 0
    for group, end in enumerate(group_ends.tolist()):
        if end > start:
            output[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
        start = end
    return output


def _dtype_for(device):
    """The dtype this device's grouped GEMM accepts.

    CPU runs in FP32 so the reference is checked tightly. The accelerator
    primitives go through fused grouped-GEMM kernels that only accept BF16 --
    handing them FP32 raises "Expected mat_a to be BFloat16 matrix got Float" --
    so the same assertions run there in the production dtype instead.
    """
    return torch.float32 if device == "cpu" else torch.bfloat16


def _tolerance_for(device):
    """BF16 carries about three decimal digits, hence the looser accelerator bound."""
    return (1e-4, 1e-4) if device == "cpu" else (2e-2, 2e-2)


def _case(device, dtype=None, rows=48, groups=5, in_features=16, out_features=24, tail=6):
    """Build a MoonEP-shaped case: ragged groups plus an unowned static tail."""
    dtype = _dtype_for(device) if dtype is None else dtype
    torch.manual_seed(1234)
    sizes = [8, 0, 12, 6, 16]
    assert len(sizes) == groups and sum(sizes) + tail == rows
    ends = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)
    # The final group absorbs the reserved tail, mirroring compact_group_ends.
    ends[-1] = rows
    inputs = torch.randn(rows, in_features, dtype=dtype, device=device)
    weights = torch.randn(groups, out_features, in_features, dtype=dtype, device=device)
    return inputs, ends, weights


@pytest.mark.parametrize("device", DEVICES)
def test_forward_matches_dense_reference(device):
    inputs, ends, weights = _case(device)
    rtol, atol = _tolerance_for(device)
    output = moonep_grouped_matmul(inputs, ends, weights)
    assert output.shape == (inputs.shape[0], weights.shape[1])
    torch.testing.assert_close(output, _reference(inputs, ends, weights), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_dgrad_and_wgrad_match_autograd(device):
    inputs, ends, weights = _case(device)
    rtol, atol = _tolerance_for(device)
    inputs = inputs.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)

    reference = _reference(inputs, ends, weights)
    grad_output = torch.randn_like(reference)
    reference.backward(grad_output)

    dgrad = moonep_grouped_matmul_dgrad(grad_output, ends, weights.detach())
    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs.detach(), ends)

    torch.testing.assert_close(dgrad, inputs.grad, rtol=rtol, atol=atol)
    assert wgrad.shape == weights.shape
    torch.testing.assert_close(wgrad, weights.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_groups_produce_zero_weight_gradient(device):
    """Group 1 is empty, so its weight gradient must be exactly zero."""
    inputs, ends, weights = _case(device)
    grad_output = torch.randn(inputs.shape[0], weights.shape[1], dtype=inputs.dtype, device=device)
    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs, ends)
    assert torch.count_nonzero(wgrad[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_the_two_weight_orientations_agree(device):
    """A transposed weight store must produce the same three results.

    ``PatchKimiMoeExperts`` holds ``[E, in, out]`` where Qwen holds
    ``[E, out, in]``. MoonEP keeps whichever the model brought, so the only thing
    that may differ is the operand layout handed to the kernels -- the arithmetic
    must not. Comparing the two paths against each other rather than against a
    reference is what makes that an equality instead of a tolerance.

    ``out_features != in_features`` on purpose: with a square matrix the two
    layouts are indistinguishable and the test would pass vacuously.
    """
    inputs, ends, weights = _case(device)
    assert weights.shape[1] != weights.shape[2], "a square weight cannot tell the layouts apart"
    rtol, atol = _tolerance_for(device)
    flipped = weights.transpose(-2, -1).contiguous()

    forward = moonep_grouped_matmul(inputs, ends, weights)
    forward_flipped = moonep_grouped_matmul(inputs, ends, flipped, weights_transposed=True)
    torch.testing.assert_close(forward_flipped, forward, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(forward)
    dgrad = moonep_grouped_matmul_dgrad(grad_output, ends, weights)
    dgrad_flipped = moonep_grouped_matmul_dgrad(grad_output, ends, flipped, weights_transposed=True)
    torch.testing.assert_close(dgrad_flipped, dgrad, rtol=rtol, atol=atol)

    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs, ends)
    wgrad_flipped = moonep_grouped_matmul_wgrad(grad_output, inputs, ends, weights_transposed=True)
    # Emitted in the caller's own layout, so it is the transpose of the other.
    assert wgrad_flipped.shape == (wgrad.shape[0], wgrad.shape[2], wgrad.shape[1])
    torch.testing.assert_close(wgrad_flipped, wgrad.transpose(-2, -1), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_a_production_shape_matches_the_reference(device):
    """A 128-wide contraction, where the fused kernels pick a different tiling.

    The cases above are deliberately small and ragged. This one has the aligned
    dimensions a real expert matrix has, which is the path production takes.
    """
    dtype = _dtype_for(device)
    rtol, atol = _tolerance_for(device)
    torch.manual_seed(1234)
    ends = torch.tensor([3, 3, 9], dtype=torch.int32, device=device)
    inputs = torch.randn(9, 128, dtype=dtype, device=device)
    weights = torch.randn(3, 256, 128, dtype=dtype, device=device)

    torch.testing.assert_close(
        moonep_grouped_matmul(inputs, ends, weights),
        _reference(inputs, ends, weights),
        rtol=rtol,
        atol=atol,
    )


def test_group_count_mismatch_is_rejected():
    inputs, ends, weights = _case("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="one cumulative end offset per group"):
        moonep_grouped_matmul(inputs, ends[:-1], weights)
