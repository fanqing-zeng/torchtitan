# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run on NPU with Triton-Ascend and the project's spmd_types dependency."""

import json
from functools import partial

import pytest
import torch

from tests.unit_tests.attn_res_test_utils import (
    forward_and_grad,
    golden_attention_residual,
    load_attn_res,
    make_inputs,
    move_inputs,
    reduction_terms,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None

pytestmark = pytest.mark.skipif(
    torch_npu is None or not torch.npu.is_available(),
    reason="Ascend NPU is unavailable",
)

# Predeclared gates from .claude/skills/kernel-numerics-verifier/SKILL.md.
FUDGE_FACTOR = 2.0
PROJECT_ATOL = 0.0
MAX_REFERENCE_RELATIVE_ERROR = 0.1


def _gate_results(reference, target, golden, inputs, upstream, eps, record_property):
    rows = []
    for name, target_tensor in target.items():
        assert target_tensor.shape == reference[name].shape == golden[name].shape
        assert target_tensor.dtype == reference[name].dtype
        expected = golden[name].cpu().double()
        actual = target_tensor.cpu().double()
        baseline = reference[name].cpu().double()
        for classification in (torch.isnan, torch.isposinf, torch.isneginf):
            torch.testing.assert_close(classification(actual), classification(expected))
            torch.testing.assert_close(
                classification(baseline), classification(expected)
            )
        if not expected.numel() or not torch.isfinite(expected).any():
            continue
        finite = torch.isfinite(expected)
        reference_error = (baseline[finite] - expected[finite]).abs().max().item()
        target_error = (actual[finite] - expected[finite]).abs().max().item()
        rounding_floor = (
            (expected[finite].to(target_tensor.dtype).double() - expected[finite])
            .abs()
            .max()
            .item()
        )
        absolute_floor = max(PROJECT_ATOL, rounding_floor)
        threshold = FUDGE_FACTOR * reference_error + absolute_floor
        scale = expected[finite].abs().max().item()
        relative_error = reference_error / max(scale, absolute_floor, 1e-300)
        flat_index = (baseline - expected).abs().nan_to_num().argmax().item()
        index = []
        for size in reversed(expected.shape):
            index.insert(0, flat_index % size)
            flat_index //= size
        terms = reduction_terms(name, inputs, upstream, tuple(index), eps=eps)
        condition = terms.abs().sum().item() / max(terms.sum().abs().item(), 1e-300)
        row = {
            "tensor": name,
            "reference_error": reference_error,
            "condition": condition,
            "target_error": target_error,
            "threshold": threshold,
            "gate_valid": relative_error <= MAX_REFERENCE_RELATIVE_ERROR,
        }
        rows.append(row)
        if not row["gate_valid"]:
            # Capture two legitimate FP32 orders; require investigation instead of
            # treating an ill-conditioned oracle comparison as a numerical pass.
            row["fp32_forward_sum"] = terms.float().sum().item()
            row["fp32_reverse_sum"] = terms.flip(0).float().sum().item()
            row["fp64_sum"] = terms.sum().item()
            pytest.fail(
                "INCONCLUSIVE: reference is ill-conditioned: " + json.dumps(row)
            )
        assert target_error <= threshold, json.dumps(row)
    record_property("attn_res_numerics", json.dumps(rows))


def _compare(
    inputs,
    upstream,
    *,
    eps=1e-5,
    backend="triton",
    strided=False,
    operation=None,
    record_property,
):
    dispatcher = load_attn_res().apply_attention_residual
    upstream_npu = move_inputs((upstream,), device="npu", strided=strided)[0].detach()
    reference = forward_and_grad(
        dispatcher,
        move_inputs(inputs, device="npu", strided=strided),
        upstream_npu,
        eps=eps,
    )
    target = forward_and_grad(
        operation or partial(dispatcher, backend=backend),
        move_inputs(inputs, device="npu", strided=strided),
        upstream_npu,
        eps=eps,
    )
    golden = forward_and_grad(
        golden_attention_residual,
        move_inputs(inputs, dtype=torch.float64),
        upstream.double(),
        eps=eps,
    )
    _gate_results(reference, target, golden, inputs, upstream, eps, record_property)
    return target


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,blocks,dim,has_partial",
    [
        (1, 0, 1, True),
        (1, 1, 1, False),
        (3, 0, 63, True),
        (7, 3, 31, True),
        (33, 7, 32, False),
        (129, 16, 33, True),
        (5, 31, 64, False),
        (5, 32, 65, True),
        (128, 6, 512, True),
        (129, 8, 7168, False),
        (0, 3, 33, True),
    ],
)
def test_triton_shapes_and_all_gradients(
    tokens, blocks, dim, has_partial, dtype, record_property
):
    inputs, upstream = make_inputs(
        tokens=tokens, blocks=blocks, dim=dim, partial=has_partial, dtype=dtype
    )
    _compare(inputs, upstream, record_property=record_property)


@pytest.mark.parametrize("has_partial", [False, True])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_triton_strided_inputs_mixed_parameter_dtype_and_custom_epsilon(
    has_partial, seed, record_property
):
    inputs, upstream = make_inputs(
        tokens=17,
        blocks=3,
        dim=129,
        partial=has_partial,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        seed=seed,
    )
    _compare(inputs, upstream, eps=0.125, strided=True, record_property=record_property)


@pytest.mark.parametrize(
    "case", ["near_zero", "large_logits", "cancellation", "zero_query"]
)
@pytest.mark.parametrize("tokens", [7, 63, 257])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_triton_adversarial_numerics_across_reduction_lengths(
    case, tokens, seed, record_property
):
    inputs, upstream = make_inputs(
        tokens=tokens, blocks=8, dim=65, dtype=torch.bfloat16, case=case, seed=seed
    )
    _compare(inputs, upstream, record_property=record_property)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_triton_nonfinite_propagation(value, record_property):
    inputs, upstream = make_inputs(tokens=1, blocks=2, dim=33)
    inputs[1][0, 0, 0] = value
    _compare(inputs, upstream, record_property=record_property)


def test_triton_identical_runs_are_deterministic(record_property):
    inputs, upstream = make_inputs(tokens=257, blocks=7, dim=129, dtype=torch.bfloat16)
    first = _compare(inputs, upstream, record_property=record_property)
    for _ in range(2):
        repeated = _compare(inputs, upstream, record_property=record_property)
        for name in first:
            torch.testing.assert_close(repeated[name], first[name], atol=0, rtol=0)


def test_triton_checkpoint_recompute(record_property):
    from torch.utils.checkpoint import checkpoint

    inputs, upstream = make_inputs(tokens=33, blocks=5, dim=65, dtype=torch.bfloat16)
    operation = partial(load_attn_res().apply_attention_residual, backend="triton")
    _compare(
        inputs,
        upstream,
        operation=partial(checkpoint, operation, use_reentrant=False),
        record_property=record_property,
    )


@pytest.mark.parametrize("has_partial", [False, True])
def test_migrated_fla_forward_and_all_gradients(has_partial, record_property):
    inputs, upstream = make_inputs(
        tokens=33, blocks=5, dim=129, partial=has_partial, dtype=torch.bfloat16
    )
    _compare(
        inputs, upstream, backend="fla", strided=True, record_property=record_property
    )
