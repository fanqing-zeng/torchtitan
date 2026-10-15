# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Translation-contract tests for the Ascend ``npu_grouped_matmul`` adapter.

A fake ``torch_npu`` records every kernel invocation and recomputes the result
from the operands it receives, following the documented Ascend semantics of
``split_item``/``group_type``. The expected values are computed independently
from the original inputs, so a wrong operand order, weight orientation, or
group layout in the adapter diverges instead of cancelling out.

These tests run on CPU without any Ascend runtime. The public entry points
dispatch on the operand's device, so they are bypassed in favor of the
registered ``*_npu`` implementations and the registry itself; the numerical
behavior of the real kernels is covered by ``test_grouped_matmul.py``, which
runs the same grouped GEMMs on every registered device.
"""

import importlib.util
import types

import pytest
import torch

from torchtitan.distributed.moonep.ops.npu import moonep_grouped_matmul as npu_gemm
from torchtitan.distributed.moonep.ops.npu.moonep_grouped_matmul import (
    moonep_grouped_matmul_dgrad_npu,
    moonep_grouped_matmul_npu,
    moonep_grouped_matmul_wgrad_npu,
)
from torchtitan.distributed.moonep.ops.registry import dispatch_op

# Groups, rows, in_features, out_features. Group 1 is empty on purpose.
G, M, K, N = 4, 48, 16, 24
ENDS = [10, 10, 30, 48]

TOL = dict(rtol=1e-5, atol=1e-5)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is not None,
    reason="a real torch_npu is installed; test_grouped_matmul.py covers the real kernels",
)

_calls: list[dict[str, object]] = []


def _fake_npu_grouped_matmul(x_list, weight_list, bias, group_list, split_item, group_type, group_list_type):
    """Recompute from the received operands per the Ascend documentation.

    Unknown ``split_item``/``group_type`` combinations fail loudly: this fake is
    the oracle for the calling convention, not a forgiving stand-in.
    """
    _calls.append(
        {
            "bias": bias,
            "group_list_dtype": group_list.dtype,
            "group_list_type": group_list_type,
            "group_type": group_type,
            "split_item": split_item,
            "weight_ptrs": tuple(w.data_ptr() for w in weight_list),
            "weight_shapes": tuple(tuple(w.shape) for w in weight_list),
            "x_shapes": tuple(tuple(x.shape) for x in x_list),
        }
    )
    if bias is not None:
        raise AssertionError("the adapter must not pass a bias")
    if group_list_type != 0:
        raise AssertionError("the adapter must pass cumulative group ends (group_list_type=0)")
    ends = group_list.detach().to(torch.int64).tolist()
    if split_item == 2 and group_type == 0:
        # Row-split GEMM: one [M, K] activation against per-group [K, N] weights,
        # handed over either as one strided [G, K, N] stack or as G 2-D views.
        (x,) = x_list
        if len(weight_list) == 1 and weight_list[0].ndim == 3:
            weights = weight_list[0].unbind(0)
        else:
            weights = list(weight_list)
        output = x.new_zeros(x.shape[0], weights[0].shape[1])
        start = 0
        for group, end in enumerate(ends):
            if end > start:
                output[start:end] = x[start:end] @ weights[group]
            start = end
        return [output]
    if split_item == 3 and group_type == 2:
        # K-split GEMM: [N, K] x [K, M] -> [G, N, M], contracting the group's
        # row range along the shared K dimension.
        (x,) = x_list
        (weight,) = weight_list
        output = x.new_zeros(len(ends), x.shape[0], weight.shape[1])
        start = 0
        for group, end in enumerate(ends):
            if end > start:
                output[group] = x[:, start:end] @ weight[start:end, :]
            start = end
        return [output]
    raise AssertionError(f"unexpected npu_grouped_matmul invocation: split_item={split_item} group_type={group_type}")


@pytest.fixture(autouse=True)
def fake_torch_npu(monkeypatch):
    """Swap in the fake module without touching ``sys.modules``.

    The vendored module imports ``torch_npu`` under ``try/except`` and reads it
    as a module global at call time, so patching the attribute is enough; the
    op registry keeps the registrations made against the real (or missing)
    import. The probe flag is reset so each test re-detects the strided form.
    """
    _calls.clear()
    fake = types.ModuleType("torch_npu")
    fake.npu_grouped_matmul = _fake_npu_grouped_matmul
    monkeypatch.setattr(npu_gemm, "torch_npu", fake)
    monkeypatch.setattr(npu_gemm, "_SUPPORTS_STRIDED_3D_WEIGHT", None)


def _case():
    """A ragged FP32 case in the linear-layer storage convention [G, out, in]."""
    torch.manual_seed(1234)
    ends = torch.tensor(ENDS, dtype=torch.int32)
    inputs = torch.randn(M, K)
    weights = torch.randn(G, N, K)
    grad_output = torch.randn(M, N)
    return inputs, ends, weights, grad_output


def _reference_forward(inputs, group_ends, weights):
    output = torch.zeros(inputs.shape[0], weights.shape[1])
    start = 0
    for group, end in enumerate(group_ends.tolist()):
        if end > start:
            output[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
        start = end
    return output


def _reference_dgrad(grad_output, group_ends, weights):
    output = torch.zeros(grad_output.shape[0], weights.shape[2])
    start = 0
    for group, end in enumerate(group_ends.tolist()):
        if end > start:
            output[start:end] = grad_output[start:end] @ weights[group]
        start = end
    return output


def _reference_wgrad(grad_output, inputs, group_ends):
    output = torch.zeros(group_ends.numel(), grad_output.shape[1], inputs.shape[1])
    start = 0
    for group, end in enumerate(group_ends.tolist()):
        if end > start:
            output[group] = grad_output[start:end].T @ inputs[start:end]
        start = end
    return output


def test_forward_passes_a_strided_transposed_weight_stack():
    inputs, ends, weights, _ = _case()
    output = moonep_grouped_matmul_npu(inputs, ends, weights)

    assert len(_calls) == 1
    call = _calls[0]
    assert call["split_item"] == 2
    assert call["group_type"] == 0
    assert call["group_list_type"] == 0
    assert call["bias"] is None
    assert call["x_shapes"] == ((M, K),)
    # The kernel must see the [G, in, out] orientation as one zero-copy stack.
    assert call["weight_shapes"] == ((G, K, N),)
    assert call["weight_ptrs"][0] == weights.data_ptr()
    assert npu_gemm._SUPPORTS_STRIDED_3D_WEIGHT is True
    torch.testing.assert_close(output, _reference_forward(inputs, ends, weights), **TOL)


def test_forward_transposed_store_is_passed_as_is():
    inputs, ends, weights, _ = _case()
    flipped = weights.transpose(-2, -1).contiguous()  # stored [G, in, out]

    output = moonep_grouped_matmul_npu(inputs, ends, flipped, weights_transposed=True)

    assert _calls[0]["weight_shapes"] == ((G, K, N),)
    assert _calls[0]["weight_ptrs"][0] == flipped.data_ptr()
    torch.testing.assert_close(output, _reference_forward(inputs, ends, weights), **TOL)


def test_dgrad_passes_the_stored_orientation():
    inputs, ends, weights, grad_output = _case()
    dgrad = moonep_grouped_matmul_dgrad_npu(grad_output, ends, weights)

    call = _calls[0]
    assert call["split_item"] == 2
    assert call["group_type"] == 0
    assert call["x_shapes"] == ((M, N),)
    # dgrad consumes the store as-is: [G, out, in].
    assert call["weight_shapes"] == ((G, N, K),)
    torch.testing.assert_close(dgrad, _reference_dgrad(grad_output, ends, weights), **TOL)

    # A transposed store still reaches the kernel in the same orientation.
    flipped = weights.transpose(-2, -1).contiguous()
    dgrad_flipped = moonep_grouped_matmul_dgrad_npu(grad_output, ends, flipped, weights_transposed=True)
    assert _calls[1]["weight_shapes"] == ((G, N, K),)
    torch.testing.assert_close(dgrad_flipped, dgrad, **TOL)


def test_wgrad_swaps_operands_and_emits_the_stored_layout():
    inputs, ends, weights, grad_output = _case()
    wgrad = moonep_grouped_matmul_wgrad_npu(grad_output, inputs, ends)

    call = _calls[0]
    assert call["split_item"] == 3
    assert call["group_type"] == 2
    assert call["group_list_type"] == 0
    assert call["bias"] is None
    # K-split contraction: inputs are transposed, grad_output is the weight.
    assert call["x_shapes"] == ((K, M),)
    assert call["weight_shapes"] == ((M, N),)
    assert wgrad.shape == (G, N, K)
    torch.testing.assert_close(wgrad, _reference_wgrad(grad_output, inputs, ends), **TOL)
    assert torch.count_nonzero(wgrad[1]) == 0  # group 1 is empty

    # The transposed request emits the raw kernel layout instead of a view.
    wgrad_flipped = moonep_grouped_matmul_wgrad_npu(grad_output, inputs, ends, weights_transposed=True)
    assert wgrad_flipped.shape == (G, K, N)
    torch.testing.assert_close(wgrad_flipped, wgrad.transpose(-2, -1), **TOL)


def test_a_strided_weight_rejection_falls_back_to_per_group_operands(monkeypatch):
    inputs, ends, weights, _ = _case()

    def reject_strided(x_list, weight_list, bias, group_list, split_item, group_type, group_list_type):
        if len(weight_list) == 1 and weight_list[0].ndim == 3:
            _calls.append({"rejected_weight_shapes": tuple(tuple(w.shape) for w in weight_list)})
            raise RuntimeError("mock CANN build: strided 3-D weight operand rejected")
        return _fake_npu_grouped_matmul(x_list, weight_list, bias, group_list, split_item, group_type, group_list_type)

    monkeypatch.setattr(npu_gemm.torch_npu, "npu_grouped_matmul", reject_strided)

    output = moonep_grouped_matmul_npu(inputs, ends, weights)

    assert len(_calls) == 2
    assert _calls[0]["rejected_weight_shapes"] == ((G, K, N),)
    assert _calls[1]["weight_shapes"] == ((K, N),) * G
    assert npu_gemm._SUPPORTS_STRIDED_3D_WEIGHT is False
    torch.testing.assert_close(output, _reference_forward(inputs, ends, weights), **TOL)

    # The probe result sticks: the next GEMM skips the strided form entirely.
    grad_output = torch.randn(M, N)
    moonep_grouped_matmul_dgrad_npu(grad_output, ends, weights)
    assert len(_calls) == 3
    assert _calls[2]["weight_shapes"] == ((N, K),) * G


def test_int32_group_ends_are_cast_before_the_call():
    inputs, ends, weights, _ = _case()
    assert ends.dtype == torch.int32

    output = moonep_grouped_matmul_npu(inputs, ends, weights)

    assert _calls[0]["group_list_dtype"] == torch.int64
    torch.testing.assert_close(output, _reference_forward(inputs, ends, weights), **TOL)


def test_ops_are_reachable_through_the_registry():
    """The public layer dispatches positionally, so pin that contract too."""
    inputs, ends, weights, grad_output = _case()

    forward = dispatch_op("moonep_grouped_matmul", inputs, ends, weights, False, device_type="npu")
    dgrad = dispatch_op("moonep_grouped_matmul_dgrad", grad_output, ends, weights, False, device_type="npu")
    wgrad = dispatch_op("moonep_grouped_matmul_wgrad", grad_output, inputs, ends, None, False, device_type="npu")

    assert len(_calls) == 3
    torch.testing.assert_close(forward, _reference_forward(inputs, ends, weights), **TOL)
    torch.testing.assert_close(dgrad, _reference_dgrad(grad_output, ends, weights), **TOL)
    torch.testing.assert_close(wgrad, _reference_wgrad(grad_output, inputs, ends), **TOL)
