# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract tests for the public AscendC KDA adapter, without NPU imports."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_adapter():
    path = (
        Path(__file__).resolve().parents[3]
        / "torchtitan/models/common/attention/npu_kda.py"
    )
    spec = importlib.util.spec_from_file_location("npu_kda_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kda = _load_adapter()


def _inputs(batch=1, tokens=7, heads=2, value_dim=128, dtype=torch.bfloat16):
    torch.manual_seed(42)
    tensors = [
        torch.randn(batch, tokens, heads, 128, dtype=dtype),
        torch.randn(batch, tokens, heads, 128, dtype=dtype),
        torch.randn(batch, tokens, heads, value_dim, dtype=dtype),
        torch.randn(batch, tokens, heads, 128, dtype=dtype),
        torch.randn(batch, tokens, heads, dtype=dtype),
        torch.randn(heads, dtype=dtype),
        torch.randn(heads * 128, dtype=dtype),
    ]
    return [tensor.requires_grad_() for tensor in tensors]


def _core(q, k, v, g, beta, A_log=None, dt_bias=None):
    """Differentiable stand-in; hardware tests separately verify KDA numerics."""
    gate = g.float()
    if A_log is not None:
        bias = 0 if dt_bias is None else dt_bias.reshape(q.shape[2:])
        gate = -5 * ((gate + bias) * A_log.exp()[:, None]).sigmoid()
    score = (q.float() * k.float()).sum(-1, keepdim=True)
    return (score + v.float() * beta[..., None] + gate.mean(-1, keepdim=True)).to(v)


class _PublicKernels:
    def forward(self, q, k, v, g, beta, scale, chunk_size, **kwargs):
        self.inputs = (q, k, v, g, beta, kwargs["A_log"], kwargs["dt_bias"])
        self.kwargs = kwargs
        assert scale == 128**-0.5
        assert chunk_size == 64
        assert kwargs["layout"] == "BSND"
        assert kwargs["disable_recompute"] is True
        assert kwargs["initial_state"] is None
        assert kwargs["output_final_state"] is False
        assert kwargs["state_v_first"] is False
        assert kwargs["return_intermediate_states"] is False
        assert kwargs["chunk_indices"] is None
        assert g.dtype == beta.dtype == torch.float32
        assert all(tensor.is_contiguous() for tensor in (q, k, v, g, beta))
        batch, tokens, heads, key_dim = q.shape
        offsets = kwargs["cu_seqlens"]
        chunks = (
            (tokens + 63) // 64
            if offsets is None
            else sum((b - a + 63) // 64 for a, b in zip(offsets, offsets[1:]))
        )
        shapes = [
            (batch, heads, tokens, key_dim),
            (batch, heads, tokens, 64),
            (batch, heads, tokens, 64),
            (batch, heads, tokens, key_dim),
            (batch, heads, tokens, key_dim),
            (batch, heads, tokens, key_dim),
            (batch, heads, tokens, v.shape[-1]),
            (batch, chunks, heads, key_dim, v.shape[-1]),
        ]
        self.intermediates = [
            torch.full(shape, i + 1, dtype=torch.float32 if i == 0 else q.dtype)
            for i, shape in enumerate(shapes)
        ]
        gk, Aqk, Akk, w, qg, kg, v_new, h = self.intermediates
        return _core(*self.inputs), None, gk, Aqk, Akk, w, None, qg, kg, v_new, h, None

    def backward(self, q, k, v, beta, *args, **kwargs):
        packed = kwargs["cu_seqlens"] is not None

        def head_first(tensor):
            tensor = tensor.transpose(1, 2).contiguous()
            return tensor[0] if packed else tensor

        def sequence_first(tensor):
            return (tensor.unsqueeze(0) if packed else tensor).transpose(1, 2)

        for actual, expected in zip(
            (q, k, v, beta), (*self.inputs[:3], self.inputs[4])
        ):
            torch.testing.assert_close(actual, head_first(expected))
            assert actual.is_contiguous()
        for actual, expected in zip(args[:8], self.intermediates):
            torch.testing.assert_close(actual, expected[0] if packed else expected)
        grad_output, scale = args[8:]
        assert kwargs["disable_recompute"] is True
        assert kwargs["use_exp2"] is True
        assert kwargs["dht"] is kwargs["initial_state"] is None
        assert kwargs["cu_seqlens"] == self.kwargs["cu_seqlens"]
        if self.kwargs["use_gate_in_kernel"]:
            torch.testing.assert_close(kwargs["raw_g"], head_first(self.inputs[3]))
        else:
            assert kwargs["raw_g"] is None
        if kwargs["dt_bias"] is not None:
            assert kwargs["dt_bias"].shape == self.inputs[0].shape[2:]
        with torch.enable_grad():
            inputs = [
                tensor.detach().requires_grad_() if tensor is not None else None
                for tensor in self.inputs
            ]
            output = _core(*inputs)
            gradients = list(
                torch.autograd.grad(
                    output,
                    [tensor for tensor in inputs if tensor is not None],
                    sequence_first(grad_output),
                )
            )
        dq, dk, dv, dg, db = map(head_first, gradients[:5])
        dA = gradients[5] if inputs[5] is not None else None
        dbias = gradients[6] if inputs[6] is not None else None
        return dq, dk, dv, db, dg, None, dA, dbias


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("value_dim", [128, 256])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_forward_backward_layout_and_all_gradient_chains(
    monkeypatch, packed, value_dim, preprocess, dtype
):
    kernels = _PublicKernels()
    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (kernels.forward, kernels.backward)
    )
    inputs = _inputs(batch=1 if packed else 2, value_dim=value_dim, dtype=dtype)
    refs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    output, final_state = kda.npu_chunk_kda(
        *inputs[:5],
        A_log=inputs[5],
        dt_bias=inputs[6],
        backend="ascendc",
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=preprocess,
        use_beta_sigmoid_in_kernel=preprocess,
        allow_neg_eigval=preprocess,
        safe_gate=True,
        cu_seqlens=torch.tensor([0, 2, 7]) if packed else None,
    )
    q, k, v, g, beta, A_log, dt_bias = refs
    if preprocess:
        q_fp32, k_fp32 = q.float(), k.float()
        q = (q_fp32 * (q_fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(q)
        k = (k_fp32 * (k_fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(k)
    beta = beta.float().sigmoid() * 2 if preprocess else beta.float()
    expected = _core(q, k, v, g.float(), beta, A_log.float(), dt_bias.float())
    assert final_state is None
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    expected.backward(grad_output)
    for actual, reference in zip(inputs, refs):
        assert actual.grad is not None
        assert actual.grad.dtype == actual.dtype
        torch.testing.assert_close(actual.grad, reference.grad, atol=0, rtol=0)


def test_optional_gate_parameters_and_zero_norm_are_differentiable(monkeypatch):
    kernels = _PublicKernels()
    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (kernels.forward, kernels.backward)
    )
    inputs = _inputs()[:5]
    with torch.no_grad():
        inputs[0].zero_()
    output, _ = kda.npu_chunk_kda(
        *inputs, backend="ascendc", use_qk_l2norm_in_kernel=True
    )
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(tensor.grad).all() for tensor in inputs)


@pytest.mark.parametrize(
    "option",
    [
        {"initial_state": torch.zeros(1)},
        {"output_final_state": True},
        {"state_v_first": True},
        {"return_intermediate_states": True},
        {"cp_context": object()},
        {"chunk_size": 128},
        {"disable_recompute": False},
        {"allow_neg_eigval": True},
        {"safe_gate": True, "lower_bound": 0},
        {"use_gate_in_kernel": True},
        {"A_log": torch.ones(2)},
        {"unknown_option": True},
        {"scale": float("nan")},
        {"cu_seqlens": [1, 7]},
        {"cu_seqlens": [0, 7, 2, 7]},
        {"cu_seqlens": [0, 6]},
        {"cu_seqlens": torch.tensor([0.0, 7.0])},
    ],
)
def test_unsupported_contracts_fail_before_launch(monkeypatch, option):
    loader = Mock(side_effect=AssertionError("Invalid inputs reached operator import"))
    monkeypatch.setattr(kda, "_load_ascendc_kernels", loader)
    with pytest.raises(ValueError):
        kda.npu_chunk_kda(*_inputs()[:5], backend="ascendc", **option)
    loader.assert_not_called()


@pytest.mark.parametrize("key_dim", [64, 96, 256])
def test_explicit_ascendc_rejects_unsupported_dimensions(key_dim):
    q = torch.zeros(1, 7, 2, key_dim, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K=128"):
        kda.npu_chunk_kda(q, q, q, q, q[..., 0], backend="ascendc")


@pytest.mark.parametrize("device,key_dim", [("cpu", 128), ("npu", 64)])
def test_auto_preserves_fla_debug_and_non_npu_contract(monkeypatch, device, key_dim):
    chunk = Mock(return_value=("output", "state"))
    monkeypatch.setattr(kda, "_load_fla_kernel", lambda: chunk)
    q = SimpleNamespace(device=SimpleNamespace(type=device), shape=(1, 7, 2, key_dim))
    state = object()
    assert kda.npu_chunk_kda(q, q, q, q, q, initial_state=state, custom_flag=17) == (
        "output",
        "state",
    )
    assert chunk.call_args.kwargs["initial_state"] is state
    assert chunk.call_args.kwargs["custom_flag"] == 17
    assert chunk.call_args.kwargs["disable_recompute"] is False


def test_loader_uses_public_exports(monkeypatch):
    module = ModuleType("fla_npu.ops.ascendc")
    module.chunk_kda_fwd = Mock()
    module.chunk_kda_bwd = Mock()
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", module)
    assert kda._load_ascendc_kernels() == (module.chunk_kda_fwd, module.chunk_kda_bwd)


def test_missing_gate_gradient_is_not_silently_dropped(monkeypatch):
    kernels = _PublicKernels()

    def broken_backward(*args, **kwargs):
        result = list(kernels.backward(*args, **kwargs))
        result[6] = None
        return tuple(result)

    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (kernels.forward, broken_backward)
    )
    q, k, v, g, beta, A_log, dt_bias = _inputs()
    output, _ = kda.npu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        backend="ascendc",
    )
    with pytest.raises(RuntimeError, match="A_log gradient"):
        output.sum().backward()


@pytest.mark.parametrize("bias_shape", [None, (2, 128)])
def test_optional_bias_and_parameter_gradient_shape(monkeypatch, bias_shape):
    kernels = _PublicKernels()
    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (kernels.forward, kernels.backward)
    )
    q, k, v, g, beta, A_log, bias = _inputs()
    bias = bias.detach().reshape(bias_shape).requires_grad_() if bias_shape else None
    output, _ = kda.npu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=bias,
        use_gate_in_kernel=True,
        backend="ascendc",
    )
    output.sum().backward()
    assert A_log.grad.shape == A_log.shape
    if bias is not None:
        assert bias.grad.shape == bias_shape
        assert kernels.kwargs["dt_bias"].shape == (256,)


def test_host_metadata_preferred_without_reading_device_offsets(monkeypatch):
    kernels = _PublicKernels()
    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (kernels.forward, kernels.backward)
    )
    kda.npu_chunk_kda(
        *_inputs()[:5],
        backend="ascendc",
        cu_seqlens=object(),
        cu_seqlens_cpu=torch.tensor([0, 2, 7]),
    )
    assert kernels.kwargs["cu_seqlens"] == (0, 2, 7)


@pytest.mark.parametrize("mutation", ["missing", "layout", "dtype"])
def test_incompatible_saved_tensor_contract_is_rejected(monkeypatch, mutation):
    kernels = _PublicKernels()

    def incompatible_forward(*args, **kwargs):
        result = list(kernels.forward(*args, **kwargs))
        if mutation == "missing":
            result[2] = None
        elif mutation == "layout":
            result[2] = result[2].transpose(1, 2).contiguous()
        else:
            result[2] = result[2].bfloat16()
        return tuple(result)

    monkeypatch.setattr(
        kda, "_load_ascendc_kernels", lambda: (incompatible_forward, kernels.backward)
    )
    with pytest.raises(RuntimeError, match="saved tensor"):
        kda.npu_chunk_kda(*_inputs()[:5], backend="ascendc")


def test_gate_backward_chunk_limit_checked_before_launch():
    inputs = _inputs(tokens=1025)
    with pytest.raises(ValueError, match="1024 chunks"):
        kda.npu_chunk_kda(
            *inputs[:5],
            backend="ascendc",
            A_log=inputs[5],
            use_gate_in_kernel=True,
            cu_seqlens=list(range(1026)),
        )


def test_packed_batch_size_is_checked():
    with pytest.raises(ValueError, match="batch size 1"):
        kda.npu_chunk_kda(*_inputs(batch=2)[:5], backend="ascendc", cu_seqlens=[0, 7])


def test_fla_autograd_registration_is_lazy_and_cached(monkeypatch):
    chunk, function, register = Mock(), object(), Mock()
    monkeypatch.setitem(sys.modules, "fla.ops.kda", SimpleNamespace(chunk_kda=chunk))
    monkeypatch.setitem(
        sys.modules, "fla.ops.kda.chunk", SimpleNamespace(ChunkKDAFunction=function)
    )
    monkeypatch.setitem(
        sys.modules,
        "spmd_types",
        SimpleNamespace(register_local_autograd_function=register),
    )
    kda._register_fla_kernel_once.cache_clear()
    try:
        assert kda._load_fla_kernel() is chunk
        assert kda._load_fla_kernel() is chunk
        register.assert_called_once_with(function)
    finally:
        kda._register_fla_kernel_once.cache_clear()


@pytest.mark.parametrize("backend", ["auto", "fla"])
def test_cold_fla_loader_captures_fullgraph_forward_backward(monkeypatch, backend):
    torch._dynamo.reset()
    registered = set()

    class ChunkKDAFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, g, beta):
            ctx.save_for_backward(q, k, v, beta)
            return q * k + v * beta.unsqueeze(-1) + g

        @staticmethod
        def backward(ctx, grad):
            q, k, v, beta = ctx.saved_tensors
            return (
                grad * k,
                grad * q,
                grad * beta.unsqueeze(-1),
                grad,
                (grad * v).sum(-1),
            )

    def chunk_kda(q, k, v, g, beta, **kwargs):
        return ChunkKDAFunction.apply(q, k, v, g, beta), None

    def register(cls):
        registered.add(cls)

    modules = {
        name: ModuleType(name)
        for name in ("fla", "fla.ops", "fla.ops.kda", "fla.ops.kda.chunk", "spmd_types")
    }
    for name, module in modules.items():
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    for name in ("fla", "fla.ops", "fla.ops.kda"):
        modules[name].__path__ = []
    modules["fla"].ops = modules["fla.ops"]
    modules["fla.ops"].kda = modules["fla.ops.kda"]
    modules["fla.ops.kda"].chunk = modules["fla.ops.kda.chunk"]
    modules["fla.ops.kda"].chunk_kda = chunk_kda
    modules["fla.ops.kda.chunk"].ChunkKDAFunction = ChunkKDAFunction
    modules["spmd_types"].register_local_autograd_function = register
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    adapter = _load_adapter()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(adapter.npu_chunk_kda, backend="eager", fullgraph=True)
        inputs = _inputs(dtype=torch.float32)[:5]
        reference_inputs = [
            tensor.detach().clone().requires_grad_() for tensor in inputs
        ]
        assert not registered
        output, state = compiled(*inputs, backend=backend)
        assert state is None
        assert registered == {ChunkKDAFunction}
        expected = chunk_kda(*reference_inputs)[0]
        upstream = torch.randn_like(output)
        output.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        for tensor, reference in zip(inputs, reference_inputs):
            torch.testing.assert_close(tensor.grad, reference.grad, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
