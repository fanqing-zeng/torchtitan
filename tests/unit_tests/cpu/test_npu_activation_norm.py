# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU reference and mocked NPU contracts for fused activations and norms."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types.checker import typecheck

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.activation import (
    _AscendCSiTUGLU,
    Sigmoid,
    SiLU,
    SiTUGLU,
    Softmax,
)
from torchtitan.models.common.nn_modules import _AscendCRMSNorm, RMSNorm
from torchtitan.models.common.norm import GatedRMSNorm


def _situ_reference(packed, beta, linear_beta):
    gate, up = packed.float().unbind(-2)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(packed.dtype)


def _assert_output_and_grads(actual, expected, actual_inputs, expected_inputs):
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad)
    expected_grads = torch.autograd.grad(expected, expected_inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("implementation", ["eager", "ascendc"])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
@pytest.mark.parametrize("shape", [(7, 2, 13), (2, 3, 2, 16)])
def test_situ_cpu_preserves_packed_layout_and_gradients(
    implementation, linear_beta, shape
):
    torch.manual_seed(42)
    packed = (torch.randn(shape) * 8).requires_grad_()
    reference = packed.detach().clone().requires_grad_()
    activation = SiTUGLU.Config(
        beta=4.0, linear_beta=linear_beta, implementation=implementation
    ).build()
    actual = activation(packed, offsets=torch.tensor([2, 4]))
    expected = _situ_reference(reference, 4.0, linear_beta)
    _assert_output_and_grads(actual, expected, (packed,), (reference,))


@pytest.fixture
def cann_ops_mock(monkeypatch):
    calls = []
    module = ModuleType("cann_ops_nn")

    def forward(packed, *, dim, beta, linear_beta, activate_left):
        if not torch.compiler.is_compiling():
            calls.append(("forward", packed, dim, beta, linear_beta, activate_left))
        return _situ_reference(
            packed.unflatten(-1, (2, packed.shape[-1] // 2)),
            beta,
            None if linear_beta == 0 else linear_beta,
        )

    def backward(grad, packed, *, dim, beta, linear_beta, activate_left):
        if not torch.compiler.is_compiling():
            calls.append(("backward", packed, dim, beta, linear_beta, activate_left))
        assert grad.is_contiguous()
        gate, up = packed.float().chunk(2, dim=-1)
        sigmoid = gate.sigmoid()
        tanh = (gate / beta).tanh()
        situ = beta * tanh * sigmoid
        gate_derivative = (1 - tanh.square()) * sigmoid + situ * (1 - sigmoid)
        if linear_beta:
            up_tanh = (up / linear_beta).tanh()
            up_derivative = 1 - up_tanh.square()
            up = linear_beta * up_tanh
        else:
            up_derivative = torch.ones_like(up)
        return torch.cat(
            (grad * gate_derivative * up, grad * situ * up_derivative), dim=-1
        ).to(packed.dtype)

    module.situ_glu = forward
    module.situ_glu_grad = backward
    monkeypatch.setitem(sys.modules, "cann_ops_nn", module)
    return calls


@pytest.mark.parametrize("linear_beta", [None, 25.0])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_situ_npu_mock_forward_backward(cann_ops_mock, linear_beta, noncontiguous):
    torch.manual_seed(42)
    packed = torch.randn(5, 3, 2, 16) * 5
    if noncontiguous:
        packed = packed.transpose(0, 1)
    packed.requires_grad_()
    reference = packed.detach().clone().requires_grad_()
    activation = SiTUGLU.Config(
        beta=4.0, linear_beta=linear_beta, implementation="ascendc"
    ).build()
    with patch.object(
        torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
    ):
        actual = activation(packed, offsets=torch.tensor([1, 2]))
    expected = _situ_reference(reference, 4.0, linear_beta)
    _assert_output_and_grads(actual, expected, (packed,), (reference,))
    assert [call[0] for call in cann_ops_mock] == ["forward", "backward"]
    for call in cann_ops_mock:
        assert call[2:] == (-1, 4.0, 0.0 if linear_beta is None else linear_beta, True)
        assert call[1].is_contiguous()
    if not noncontiguous:
        assert cann_ops_mock[0][1].data_ptr() == packed.data_ptr()


def test_situ_npu_missing_dependency_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cann_ops_nn", None)
    activation = SiTUGLU.Config(implementation="ascendc").build()
    packed = torch.randn(3, 2, 8)
    with (
        patch.object(
            torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
        ),
        pytest.raises(ModuleNotFoundError, match="cann_ops_nn"),
    ):
        activation(packed)


@pytest.mark.parametrize("implementation", ["eager", "fla"])
@pytest.mark.parametrize("activation_config", [Sigmoid.Config(), SiLU.Config()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_rmsnorm_cpu_output_weight_and_gate_gradients(
    implementation, activation_config, dtype
):
    torch.manual_seed(42)
    module = (
        GatedRMSNorm.Config(
            dim=17,
            eps=3e-4,
            activation_fn=activation_config,
            implementation=implementation,
        )
        .build()
        .to(dtype=dtype)
    )
    torch.nn.init.normal_(module.weight)
    x = torch.randn(2, 5, 3, 17, dtype=dtype, requires_grad=True)
    gate = (torch.randn_like(x) * 5).requires_grad_()
    ref_x, ref_gate, ref_weight = [
        tensor.detach().clone().requires_grad_() for tensor in (x, gate, module.weight)
    ]
    normalized = F.rms_norm(ref_x.float(), (17,), ref_weight.float(), 3e-4)
    expected = (normalized * activation_config.build()(ref_gate.float())).to(dtype)
    _assert_output_and_grads(
        module(x, gate),
        expected,
        (x, gate, module.weight),
        (ref_x, ref_gate, ref_weight),
    )
    assert set(module.state_dict()) == {"weight"}


def test_gated_rmsnorm_fla_contract_and_backward(monkeypatch):
    from torchtitan.models.common import norm

    calls = []

    def fused(x, gate, weight, bias, *, activation, residual, prenorm, eps):
        calls.append((bias, activation, residual, prenorm, eps))
        return (
            F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps)
            * gate.float().sigmoid()
        )

    monkeypatch.setattr(norm, "_load_fla_rms_norm_gated", lambda: fused)
    module = GatedRMSNorm.Config(
        dim=16, eps=2e-3, activation_fn=Sigmoid.Config(), implementation="fla"
    ).build()
    torch.nn.init.normal_(module.weight)
    x = torch.randn(3, 2, 16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    ref_x, ref_gate, ref_weight = [
        tensor.detach().clone().requires_grad_() for tensor in (x, gate, module.weight)
    ]
    with patch.object(
        torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
    ):
        actual = module(x, gate)
    expected = F.rms_norm(ref_x, (16,), ref_weight, 2e-3) * ref_gate.sigmoid()
    _assert_output_and_grads(
        actual, expected, (x, gate, module.weight), (ref_x, ref_gate, ref_weight)
    )
    assert calls == [(None, "sigmoid", None, False, 2e-3)]


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "ascendc"])
def test_rmsnorm_cpu_checkpoint_and_backward(affine, implementation):
    torch.manual_seed(42)
    config = RMSNorm.Config(
        normalized_shape=17,
        eps=7e-4,
        elementwise_affine=affine,
        implementation=implementation,
    )
    module = config.build()
    reference = torch.nn.RMSNorm(17, eps=7e-4, elementwise_affine=affine)
    module.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 5, 17, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    inputs = (x, module.weight) if affine else (x,)
    ref_inputs = (ref_x, reference.weight) if affine else (ref_x,)
    _assert_output_and_grads(module(x), reference(ref_x), inputs, ref_inputs)
    assert set(module.state_dict()) == ({"weight"} if affine else set())


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("eps", [1e-5, 0.02, None])
def test_rmsnorm_npu_mock_epsilon_affine_and_backward(monkeypatch, affine, eps):
    calls = []
    mock_npu = ModuleType("torch_npu")

    def fused(x, weight, *, epsilon):
        calls.append((weight, epsilon))
        return (
            F.rms_norm(x, (x.shape[-1],), weight, epsilon),
            (x.square().mean(-1, keepdim=True) + epsilon).rsqrt(),
        )

    def fused_backward(grad, x, weight, rstd):
        scaled_grad = grad * weight
        grad_x = rstd * (
            scaled_grad - x * rstd.square() * (scaled_grad * x).mean(-1, keepdim=True)
        )
        grad_weight = (grad * x * rstd).sum(tuple(range(x.ndim - 1)))
        return grad_x, grad_weight

    mock_npu.npu_rms_norm = fused
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)
    monkeypatch.setattr(
        torch.ops.npu, "npu_rms_norm_backward", fused_backward, raising=False
    )
    module = RMSNorm.Config(
        normalized_shape=16,
        eps=eps,
        elementwise_affine=affine,
        implementation="ascendc",
    ).build()
    reference = torch.nn.RMSNorm(16, eps=eps, elementwise_affine=affine)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(4, 3, 16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    with patch.object(
        torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
    ):
        actual = module(x)
    inputs = (x, module.weight) if affine else (x,)
    ref_inputs = (ref_x, reference.weight) if affine else (ref_x,)
    _assert_output_and_grads(actual, reference(ref_x), inputs, ref_inputs)
    assert len(calls) == 1
    assert calls[0][1] == (torch.finfo(x.dtype).eps if eps is None else eps)
    if not affine:
        torch.testing.assert_close(calls[0][0], torch.ones(16))
        assert not calls[0][0].requires_grad
        assert not module.state_dict()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SiTUGLU.Config(implementation="unknown"),
        lambda: RMSNorm.Config(normalized_shape=8, implementation="unknown"),
        lambda: GatedRMSNorm.Config(
            dim=8, eps=1e-5, activation_fn=Sigmoid.Config(), implementation="unknown"
        ),
        lambda: GatedRMSNorm.Config(
            dim=8, eps=1e-5, activation_fn=Softmax.Config(), implementation="fla"
        ),
        lambda: SiTUGLU.Config(beta=0),
        lambda: SiTUGLU.Config(linear_beta=0),
    ],
)
def test_fused_activation_norm_invalid_config(factory):
    with pytest.raises(ValueError):
        factory()


def test_gated_rmsnorm_other_activation_keeps_eager_semantics():
    module = GatedRMSNorm.Config(
        dim=8, eps=1e-5, activation_fn=Softmax.Config()
    ).build()
    torch.nn.init.ones_(module.weight)
    x = torch.randn(4, 8)
    gate = torch.randn_like(x)
    torch.testing.assert_close(
        module(x, gate), F.rms_norm(x, (8,), module.weight, 1e-5) * gate.softmax(-1)
    )


@pytest.fixture
def fake_mesh():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=2
    )
    try:
        yield init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, 2])
def test_situ_spmd_retains_token_or_hidden_sharding(
    fake_mesh, cann_ops_mock, shard_dim
):
    packed = torch.randn(4, 2, 8)
    with set_current_spmd_mesh(fake_mesh), typecheck(strict_mode="strict", local=False):
        spmd.assert_type(packed, {"tp": spmd.S(shard_dim)})
        output = _AscendCSiTUGLU.apply(packed, 4.0, 25.0)
        spmd.assert_type(output, {"tp": spmd.S(0 if shard_dim == 0 else 1)})


def test_rmsnorm_spmd_retains_sequence_sharding(fake_mesh, monkeypatch):
    mock_npu = ModuleType("torch_npu")
    mock_npu.npu_rms_norm = lambda x, weight, epsilon: (
        F.rms_norm(x, weight.shape, weight, epsilon),
        (x.square().mean(-1, keepdim=True) + epsilon).rsqrt(),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)
    x = torch.randn(4, 8)
    weight = torch.ones(8)
    with set_current_spmd_mesh(fake_mesh), typecheck(strict_mode="strict", local=False):
        spmd.assert_type(x, {"tp": spmd.S(0)})
        spmd.assert_type(weight, {"tp": spmd.R})
        output = _AscendCRMSNorm.apply(x, weight, 1e-5)
        spmd.assert_type(output, {"tp": spmd.S(0)})


def test_fla_loader_registers_global_spmd_decomposition(fake_mesh, monkeypatch):
    from torchtitan.models.common import norm

    class MockFlaNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            return norm._fla_norm_gated_reference(*args)

    mock_fla = ModuleType("fla.modules.fused_norm_gate")
    mock_fla.LayerNormGatedFunction = MockFlaNorm
    mock_fla.rms_norm_gated = lambda x, gate, weight, bias, activation, eps: (
        MockFlaNorm.apply(
            x, gate, weight, bias, activation, None, eps, False, False, True
        )
    )
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate", mock_fla)
    norm._register_fla_norm_gated_once.cache_clear()
    try:
        fused = norm._load_fla_rms_norm_gated()
        x = torch.randn(4, 3, 8)
        gate = torch.randn_like(x)
        weight = torch.ones(8)
        with set_current_spmd_mesh(fake_mesh), typecheck(
            strict_mode="strict", local=False
        ):
            spmd.assert_type(x, {"tp": spmd.S(1)})
            spmd.assert_type(gate, {"tp": spmd.S(1)})
            spmd.assert_type(weight, {"tp": spmd.R})
            output = fused(x, gate, weight, None, "sigmoid", 1e-5)
            spmd.assert_type(output, {"tp": spmd.S(1)})
    finally:
        norm._register_fla_norm_gated_once.cache_clear()


def test_situ_fused_checkpoint_preserves_backward(cann_ops_mock):
    from torch.utils.checkpoint import checkpoint

    packed = torch.randn(4, 2, 8, requires_grad=True)
    reference = packed.detach().clone().requires_grad_()
    actual = checkpoint(_AscendCSiTUGLU.apply, packed, 4.0, 25.0, use_reentrant=False)
    expected = _situ_reference(reference, 4.0, 25.0)
    _assert_output_and_grads(actual, expected, (packed,), (reference,))


def test_situ_fused_fullgraph_captures_backward(cann_ops_mock):
    compiled = torch.compile(
        lambda packed: _AscendCSiTUGLU.apply(packed, 4.0, 25.0),
        backend="aot_eager",
        fullgraph=True,
    )
    packed = torch.randn(4, 2, 8, requires_grad=True)
    reference = packed.detach().clone().requires_grad_()
    _assert_output_and_grads(
        compiled(packed), _situ_reference(reference, 4.0, 25.0), (packed,), (reference,)
    )


def test_fla_loader_is_safe_in_cold_fullgraph_capture(monkeypatch):
    from torchtitan.models.common import norm

    class MockFlaNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            return norm._fla_norm_gated_reference(*args)

    mock_fla = ModuleType("fla.modules.fused_norm_gate")
    mock_fla.LayerNormGatedFunction = MockFlaNorm
    mock_fla.rms_norm_gated = lambda x, gate, weight, bias, activation, eps: (
        MockFlaNorm.apply(
            x, gate, weight, bias, activation, None, eps, False, False, True
        )
    )
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate", mock_fla)
    norm._register_fla_norm_gated_once.cache_clear()
    try:

        def forward(x, gate, weight):
            return norm._load_fla_rms_norm_gated()(
                x, gate, weight, None, "sigmoid", 1e-5
            )

        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        x = torch.randn(4, 3, 8)
        gate = torch.randn_like(x)
        weight = torch.ones(8)
        torch.testing.assert_close(
            compiled(x, gate, weight),
            F.rms_norm(x, (8,), weight, 1e-5) * gate.sigmoid(),
        )
    finally:
        norm._register_fla_norm_gated_once.cache_clear()


def test_reference_paths_compile_with_fused_configs():
    situ = SiTUGLU.Config(beta=4.0, linear_beta=25.0, implementation="ascendc").build()
    norm = GatedRMSNorm.Config(
        dim=16, eps=1e-5, activation_fn=Sigmoid.Config(), implementation="fla"
    ).build()
    torch.nn.init.ones_(norm.weight)
    packed = torch.randn(3, 2, 16, requires_grad=True)
    gate = torch.randn(3, 16, requires_grad=True)

    def forward(packed, gate):
        return norm(situ(packed), gate)

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    ref_packed = packed.detach().clone().requires_grad_()
    ref_gate = gate.detach().clone().requires_grad_()
    actual = compiled(packed, gate)
    expected = forward(ref_packed, ref_gate)
    _assert_output_and_grads(actual, expected, (packed, gate), (ref_packed, ref_gate))


def test_shared_norm_argument_names_match_sharding_contract():
    import inspect

    assert list(inspect.signature(RMSNorm.forward).parameters) == ["self", "input"]
    assert list(inspect.signature(GatedRMSNorm.forward).parameters) == [
        "self",
        "x",
        "gate",
    ]
