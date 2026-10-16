# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU forward/backward comparisons against an independent FP32 recurrence.

Run explicitly on an Ascend host with the matching public fla_npu operator
package installed. Missing operator APIs fail tests once NPU is available.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu", reason="Ascend KDA numerical tests require torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend KDA numerical tests require an NPU", allow_module_level=True)

_PATH = (
    Path(__file__).resolve().parents[3]
    / "torchtitan/models/common/attention/npu_kda.py"
)
_SPEC = importlib.util.spec_from_file_location("ascendc_kda_numerics", _PATH)
kda = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kda)


def _reference(q, k, v, raw_g, raw_beta, A_log, dt_bias, *, offsets):
    """Apply decay, delta update, and readout sequentially for each document."""
    q_fp32, k_fp32 = q.float(), k.float()
    q = (q_fp32 * (q_fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(q).float()
    k = (k_fp32 * (k_fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(k).float()
    v = v.float()
    gate = (
        -5
        * (
            (raw_g.float() + dt_bias.float().reshape(q.shape[2:]))
            * A_log.float().exp()[:, None]
        ).sigmoid()
    )
    beta = raw_beta.float().sigmoid()
    documents = (
        [(0, q.shape[1])] if offsets is None else list(zip(offsets, offsets[1:]))
    )
    batches = []
    for batch in range(q.shape[0]):
        rows = []
        for start, end in documents:
            state = torch.zeros(q.shape[2], q.shape[3], v.shape[3])
            for token in range(start, end):
                state = state * gate[batch, token].exp()[..., None]
                predicted = (state * k[batch, token, :, :, None]).sum(-2)
                delta = beta[batch, token, :, None] * (v[batch, token] - predicted)
                state = state + k[batch, token, :, :, None] * delta[:, None, :]
                rows.append(
                    (state * q[batch, token, :, :, None]).sum(-2) * q.shape[-1] ** -0.5
                )
        batches.append(torch.stack(rows))
    return torch.stack(batches).to(raw_g.dtype)


def _inputs(batch, tokens, value_dim, dtype):
    generator = torch.Generator().manual_seed(42)
    shapes = [
        (batch, tokens, 2, 128),
        (batch, tokens, 2, 128),
        (batch, tokens, 2, value_dim),
        (batch, tokens, 2, 128),
        (batch, tokens, 2),
        (2,),
        (2 * 128,),
    ]
    return [torch.randn(shape, generator=generator).to(dtype) for shape in shapes]


def _assert_numerical_close(actual, expected, *, name, dtype):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), f"Non-finite {name}"
    # Relative RMS also catches a missing gradient when individual values are
    # small. Elementwise tolerances allow low-precision chunk accumulation.
    relative_rms = (actual - expected).square().mean().sqrt() / (
        expected.square().mean().sqrt().clamp_min(1e-8)
    )
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.015
    assert relative_rms < tolerance, f"{name}: relative RMS error {relative_rms.item()}"
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=2e-3 if dtype == torch.bfloat16 else 5e-4,
        msg=lambda message: f"{name}: {message}",
    )


@pytest.mark.parametrize(
    "tokens,packed", [(1, False), (63, False), (65, True), (129, True)]
)
@pytest.mark.parametrize("value_dim", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ascendc_forward_and_all_gradients_match_recurrence(
    tokens, packed, value_dim, dtype
):
    batch = 1 if packed else 2
    offsets = [0, 7, tokens] if packed else None
    inputs = _inputs(batch, tokens, value_dim, dtype)
    references = [tensor.clone().requires_grad_() for tensor in inputs]
    npu_inputs = [tensor.npu().requires_grad_() for tensor in inputs]
    output, final_state = kda.npu_chunk_kda(
        *npu_inputs[:5],
        A_log=npu_inputs[5],
        dt_bias=npu_inputs[6],
        backend="ascendc",
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device="npu")
        if packed
        else None,
    )
    expected = _reference(*references, offsets=offsets)
    assert final_state is None
    _assert_numerical_close(output, expected, name="output", dtype=dtype)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(43)).to(
        dtype
    )
    output.backward(grad.npu())
    expected.backward(grad)
    for name, actual, reference in zip(
        ("q", "k", "v", "g", "beta", "A_log", "dt_bias"), npu_inputs, references
    ):
        assert actual.grad is not None, f"Missing {name} gradient"
        _assert_numerical_close(actual.grad, reference.grad, name=name, dtype=dtype)


def test_packed_document_has_no_cross_document_dependency():
    inputs = [
        tensor.npu().requires_grad_() for tensor in _inputs(1, 65, 128, torch.bfloat16)
    ]

    def run(tensors):
        return kda.npu_chunk_kda(
            *tensors[:5],
            A_log=tensors[5],
            dt_bias=tensors[6],
            backend="ascendc",
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            cu_seqlens_cpu=torch.tensor([0, 7, 65]),
        )[0]

    output = run(inputs)
    changed = [tensor.detach().clone() for tensor in inputs]
    for tensor in changed[:5]:
        tensor[:, :7] += 2
    changed_output = run(changed)
    torch.testing.assert_close(output[:, 7:], changed_output[:, 7:], rtol=0, atol=0)
    output[:, 7:].sum().backward()
    for tensor in inputs[:5]:
        torch.testing.assert_close(
            tensor.grad[:, :7], torch.zeros_like(tensor.grad[:, :7])
        )
