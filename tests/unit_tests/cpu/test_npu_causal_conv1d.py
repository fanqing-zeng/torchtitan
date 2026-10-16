# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU numerical contract tests for AscendC short-convolution translation."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.unit_tests.causal_conv1d_test_utils import (
    case,
    load_adapter,
    reference_convolution,
)

conv = load_adapter()


class PublicKernels:
    def __init__(self):
        self.calls = []

    def forward(self, x, weight, bias, state, **kwargs):
        self.calls.append(("forward", kwargs))
        assert bias is None
        assert weight.is_contiguous()
        assert kwargs["activation_mode"] == 0
        assert kwargs["run_mode"] == 0
        boundaries = kwargs["query_start_loc"]
        heads = kwargs["head_num"]
        batch_x = x if boundaries is None else x[None]
        output = reference_convolution(
            batch_x, weight.T, boundaries, state.transpose(1, 2), activate=False
        )
        output = output.unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        return output if boundaries is None else output[0]

    def backward(
        self,
        *,
        x,
        y,
        weight,
        dy,
        initial_state,
        dht,
        query_start_loc,
        activation,
        input_layout
    ):
        self.calls.append(("backward", input_layout))
        assert dht is None
        assert activation == 1
        assert dy.is_contiguous()
        is_varlen = query_start_loc is not None
        assert input_layout == ("NTD" if is_varlen else "BNSD")
        with torch.enable_grad():
            op_x = x.detach().requires_grad_()
            op_weight = weight.detach().requires_grad_()
            state = (
                initial_state.detach().requires_grad_()
                if initial_state is not None
                else None
            )
            output = reference_convolution(
                op_x[None] if is_varlen else op_x,
                op_weight.T,
                query_start_loc,
                state.transpose(1, 2) if state is not None else None,
                activate=False,
            )
            flat_y = (
                y.transpose(0, 1).reshape_as(output)
                if is_varlen
                else y.transpose(1, 2).reshape_as(output)
            )
            torch.testing.assert_close(flat_y, output)
            grad = (
                dy.transpose(0, 1).reshape_as(output)
                if is_varlen
                else dy.transpose(1, 2).reshape_as(output)
            )
            inputs = (op_x, op_weight) if state is None else (op_x, op_weight, state)
            gradients = torch.autograd.grad(
                torch.nn.functional.silu(output), inputs, grad
            )
        return (
            gradients[0],
            gradients[1],
            None,
            gradients[2] if state is not None else None,
        )


@pytest.fixture
def kernels(monkeypatch):
    kernels = PublicKernels()
    monkeypatch.setattr(
        conv, "_load_kernels", lambda: (kernels.forward, kernels.backward)
    )
    return kernels


@pytest.mark.parametrize(
    "batch,boundaries", [(2, None), (1, [0, 11]), (1, [0, 1, 3, 3, 11])]
)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("history_length", ["none", "width", "width_minus_one"])
def test_forward_and_all_gradients(kernels, batch, boundaries, width, history_length):
    history_width = {"none": 0, "width": width, "width_minus_one": width - 1}[
        history_length
    ]
    inputs = case(
        batch=batch, width=width, boundaries=boundaries, history_width=history_width
    )
    references = tuple(
        value.detach().clone().requires_grad_() if value is not None else None
        for value in inputs
    )
    x, weight, history = inputs
    expected = reference_convolution(*references[:2], boundaries, references[2])
    actual = conv.ascendc_causal_conv1d(
        x, weight, num_heads=3, cu_seqlens=boundaries, initial_state=history
    )
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        actual, [value for value in inputs if value is not None], upstream
    )
    expected_grads = torch.autograd.grad(
        expected, [value for value in references if value is not None], upstream
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert kernels.calls[-1] == ("backward", "NTD" if boundaries else "BNSD")


@pytest.mark.parametrize("recompute", [False, True])
def test_strided_operands_and_recompute(kernels, recompute):
    raw_x, raw_weight, _ = case(channels=24)
    x = raw_x[..., ::2]
    weight = raw_weight[::2]
    boundaries = torch.tensor([0, 2, 11], dtype=torch.int32)

    def run(x, weight):
        return conv.ascendc_causal_conv1d(x, weight, num_heads=3, cu_seqlens=boundaries)

    actual = (
        checkpoint(run, x, weight, use_reentrant=False) if recompute else run(x, weight)
    )
    expected = reference_convolution(x, weight, boundaries.tolist())
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (raw_x, raw_weight), upstream)
    expected_grads = torch.autograd.grad(expected, (raw_x, raw_weight), upstream)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_documents_do_not_leak(kernels):
    x, weight, _ = case()
    boundary = 3
    output = conv.ascendc_causal_conv1d(
        x, weight, num_heads=3, cu_seqlens=[0, boundary, 11]
    )
    gradient = torch.autograd.grad(output[:, boundary:].sum(), x)[0]
    assert torch.count_nonzero(gradient[:, :boundary]) == 0


@pytest.mark.parametrize("boundaries", [[1, 11], [0, 4, 3, 11], [0, 10], [0]])
def test_rejects_invalid_packing(boundaries):
    x, weight, _ = case()
    with pytest.raises(ValueError, match="cu_seqlens"):
        conv.ascendc_causal_conv1d(x, weight, num_heads=3, cu_seqlens=boundaries)


def test_rejects_invalid_shapes_before_loading_kernels():
    x, weight, _ = case()
    with pytest.raises(ValueError, match="num_heads"):
        conv.ascendc_causal_conv1d(x, weight, num_heads=5)
    with pytest.raises(ValueError, match="channel"):
        conv.ascendc_causal_conv1d(x, weight[:4], num_heads=3)
    with pytest.raises(ValueError, match="batch"):
        conv.ascendc_causal_conv1d(
            x.expand(2, -1, -1), weight, num_heads=3, cu_seqlens=[0, 11]
        )
    with pytest.raises(ValueError, match="history"):
        conv.ascendc_causal_conv1d(
            x, weight, num_heads=3, initial_state=torch.zeros(1, 12, 1)
        )
