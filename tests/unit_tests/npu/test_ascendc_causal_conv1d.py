# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real AscendC forward/backward against independent per-document convolutions."""

import pytest
import torch

from tests.unit_tests.causal_conv1d_test_utils import (
    case,
    load_adapter,
    reference_convolution,
)

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)
conv = load_adapter()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,boundaries", [(1, None), (65, None), (129, [0, 1, 64, 129])]
)
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("history_width", [0, 3])
def test_real_short_convolution_output_and_gradients(
    dtype, tokens, boundaries, width, history_width
):
    history_width = width - 1 if history_width else 0
    values = case(
        tokens=tokens,
        channels=384,
        width=width,
        boundaries=boundaries,
        history_width=history_width,
    )
    # Cast once to the input precision, then upcast those exact values for the oracle.
    quantized = tuple(
        value.detach().to(dtype) if value is not None else None for value in values
    )
    target_inputs = tuple(
        value.to("npu").requires_grad_() if value is not None else None
        for value in quantized
    )
    oracle_inputs = tuple(
        value.double().requires_grad_() if value is not None else None
        for value in quantized
    )
    target = conv.ascendc_causal_conv1d(
        *target_inputs[:2],
        num_heads=3,
        cu_seqlens=boundaries,
        initial_state=target_inputs[2]
    )
    oracle = reference_convolution(*oracle_inputs[:2], boundaries, oracle_inputs[2])
    upstream = torch.randn_like(quantized[0])
    target_grads = torch.autograd.grad(
        target, [x for x in target_inputs if x is not None], upstream.to("npu")
    )
    oracle_grads = torch.autograd.grad(
        oracle, [x for x in oracle_inputs if x is not None], upstream.double()
    )
    # Fixed tolerances account for low-precision convolution and activation storage.
    tolerance = (
        {"atol": 0.04, "rtol": 0.04}
        if dtype == torch.bfloat16
        else {"atol": 0.004, "rtol": 0.004}
    )
    for actual, expected in zip((target, *target_grads), (oracle, *oracle_grads)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.detach().cpu().double(), expected, **tolerance
        )
