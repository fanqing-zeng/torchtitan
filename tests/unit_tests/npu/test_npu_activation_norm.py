# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ascend numerical tests; run with the fused-operator dependencies installed."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)

from torchtitan.models.common.activation import Sigmoid, SiTUGLU
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.norm import GatedRMSNorm


def _compare(actual, reference, inputs, ref_inputs):
    tolerance = (
        {"atol": 3e-2, "rtol": 3e-2}
        if actual.dtype == torch.bfloat16
        else {
            "atol": 3e-5,
            "rtol": 3e-4,
        }
    )
    torch.testing.assert_close(actual, reference, **tolerance)
    grad = torch.randn_like(reference)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    reference_grads = torch.autograd.grad(reference, ref_inputs, grad)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, reference_grad, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17, 2, 64), (2, 7, 2, 128), (127, 2, 2048)])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_ascendc_situ_forward_backward(dtype, shape, linear_beta):
    torch.manual_seed(42)
    packed = (torch.randn(shape, device="npu", dtype=dtype) * 5).requires_grad_()
    ref_packed = packed.detach().clone().requires_grad_()
    fused = SiTUGLU.Config(
        beta=4.0, linear_beta=linear_beta, implementation="ascendc"
    ).build()
    eager = SiTUGLU.Config(beta=4.0, linear_beta=linear_beta).build()
    _compare(fused(packed), eager(ref_packed), (packed,), (ref_packed,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_fla_gated_rmsnorm_forward_all_gradients(dtype, head_dim, eps):
    torch.manual_seed(42)
    fused = (
        GatedRMSNorm.Config(
            dim=head_dim, eps=eps, activation_fn=Sigmoid.Config(), implementation="fla"
        )
        .build()
        .to(device="npu", dtype=dtype)
    )
    eager = (
        GatedRMSNorm.Config(dim=head_dim, eps=eps, activation_fn=Sigmoid.Config())
        .build()
        .to(device="npu", dtype=dtype)
    )
    torch.nn.init.normal_(fused.weight)
    eager.load_state_dict(fused.state_dict(), strict=True)
    x = torch.randn(2, 37, 4, head_dim, device="npu", dtype=dtype, requires_grad=True)
    gate = (torch.randn_like(x) * 4).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_gate = gate.detach().clone().requires_grad_()
    _compare(
        fused(x, gate),
        eager(ref_x, ref_gate),
        (x, gate, fused.weight),
        (ref_x, ref_gate, eager.weight),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("dim", [64, 128, 2048])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_ascendc_rmsnorm_forward_all_gradients(dtype, affine, dim, eps):
    torch.manual_seed(42)
    fused = (
        RMSNorm.Config(
            normalized_shape=dim,
            eps=eps,
            elementwise_affine=affine,
            implementation="ascendc",
        )
        .build()
        .to(device="npu", dtype=dtype)
    )
    eager = (
        RMSNorm.Config(normalized_shape=dim, eps=eps, elementwise_affine=affine)
        .build()
        .to(device="npu", dtype=dtype)
    )
    if affine:
        torch.nn.init.normal_(fused.weight)
    eager.load_state_dict(fused.state_dict(), strict=True)
    x = torch.randn(2, 37, dim, device="npu", dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    inputs = (x, fused.weight) if affine else (x,)
    ref_inputs = (ref_x, eager.weight) if affine else (ref_x,)
    _compare(fused(x), eager(ref_x), inputs, ref_inputs)


@pytest.mark.parametrize("operator", ["situ", "rmsnorm", "gated_rmsnorm"])
def test_fused_norm_activation_fullgraph_backward(operator):
    torch.manual_seed(42)
    if operator == "situ":
        fused = SiTUGLU.Config(
            beta=4.0, linear_beta=25.0, implementation="ascendc"
        ).build()
        eager = SiTUGLU.Config(beta=4.0, linear_beta=25.0).build()
        x = torch.randn(17, 2, 128, device="npu", requires_grad=True)
        inputs = (x,)
    elif operator == "rmsnorm":
        fused = (
            RMSNorm.Config(normalized_shape=128, implementation="ascendc")
            .build()
            .to("npu")
        )
        eager = RMSNorm.Config(normalized_shape=128).build().to("npu")
        inputs = (torch.randn(17, 128, device="npu", requires_grad=True),)
    else:
        fused = (
            GatedRMSNorm.Config(
                dim=128, eps=1e-5, activation_fn=Sigmoid.Config(), implementation="fla"
            )
            .build()
            .to("npu")
        )
        eager = (
            GatedRMSNorm.Config(dim=128, eps=1e-5, activation_fn=Sigmoid.Config())
            .build()
            .to("npu")
        )
        torch.nn.init.ones_(fused.weight)
        inputs = tuple(
            torch.randn(17, 4, 128, device="npu", requires_grad=True) for _ in range(2)
        )
    ref_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    if operator != "situ":
        eager.load_state_dict(fused.state_dict(), strict=True)
        params, ref_params = (fused.weight,), (eager.weight,)
    else:
        params, ref_params = (), ()
    compiled = torch.compile(fused, backend="aot_eager", fullgraph=True)
    _compare(
        compiled(*inputs), eager(*ref_inputs), inputs + params, ref_inputs + ref_params
    )
