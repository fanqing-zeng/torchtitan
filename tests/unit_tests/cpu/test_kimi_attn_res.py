# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from functools import partial
from types import ModuleType

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


@pytest.mark.parametrize(
    "blocks,has_partial", [(0, True), (1, False), (3, True), (7, False)]
)
@pytest.mark.parametrize("dim", [1, 31, 32, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_eager_matches_fp64_forward_and_all_gradients(blocks, has_partial, dim, dtype):
    inputs, upstream = make_inputs(
        blocks=blocks,
        dim=dim,
        partial=has_partial,
        dtype=dtype,
        weight_dtype=torch.float32,
    )
    actual = forward_and_grad(
        load_attn_res().apply_attention_residual,
        move_inputs(inputs, strided=True),
        upstream,
        eps=1e-5,
    )
    golden = forward_and_grad(
        golden_attention_residual,
        move_inputs(inputs, dtype=torch.float64),
        upstream.double(),
        eps=1e-5,
    )
    for name, result in actual.items():
        assert result.dtype == (
            dtype if name in ("output", "history", "partial") else torch.float32
        )
        torch.testing.assert_close(
            result.double(),
            golden[name],
            atol=2e-6 if result.dtype == torch.float32 else 0.02,
            rtol=2e-5 if result.dtype == torch.float32 else 0.01,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("blocks,has_partial", [(0, True), (1, False)])
def test_single_source_is_identity_with_zero_weight_gradients(blocks, has_partial):
    inputs, upstream = make_inputs(blocks=blocks, partial=has_partial, dim=17)
    results = forward_and_grad(
        load_attn_res().apply_attention_residual,
        move_inputs(inputs),
        upstream,
        eps=0.37,
    )
    source = inputs[0] if has_partial else inputs[1][:, 0]
    torch.testing.assert_close(results["output"], source, atol=0, rtol=0)
    gradient = results["partial"] if has_partial else results["history"][:, 0]
    torch.testing.assert_close(gradient, upstream, atol=0, rtol=0)
    assert torch.count_nonzero(results["projection"]) == 0
    assert torch.count_nonzero(results["norm"]) == 0


def test_zero_query_is_uniform_and_keeps_projection_gradient():
    inputs, upstream = make_inputs(case="zero_query")
    results = forward_and_grad(
        load_attn_res().apply_attention_residual,
        move_inputs(inputs),
        upstream,
        eps=1e-5,
    )
    expected = torch.cat((inputs[1], inputs[0].unsqueeze(1)), dim=1).mean(1)
    torch.testing.assert_close(results["output"], expected, atol=0, rtol=0)
    assert torch.count_nonzero(results["norm"]) == 0
    assert torch.count_nonzero(results["projection"]) > 0
    torch.testing.assert_close(results["partial"], upstream / 4, atol=0, rtol=0)
    torch.testing.assert_close(
        results["history"], upstream.unsqueeze(1).expand(-1, 3, -1) / 4, atol=0, rtol=0
    )


@pytest.mark.parametrize("case", ["near_zero", "cancellation", "large_logits"])
def test_eager_adversarial_inputs_and_epsilon(case):
    inputs, upstream = make_inputs(dim=65, blocks=6, case=case)
    actual = forward_and_grad(
        load_attn_res().apply_attention_residual,
        move_inputs(inputs),
        upstream,
        eps=0.125,
    )
    golden = forward_and_grad(
        golden_attention_residual,
        move_inputs(inputs, dtype=torch.float64),
        upstream.double(),
        eps=0.125,
    )
    for name in actual:
        torch.testing.assert_close(
            actual[name].double(), golden[name], atol=1e-5, rtol=2e-4
        )


def test_empty_token_batch_has_empty_output_and_zero_parameter_gradients():
    inputs, upstream = make_inputs(tokens=0)
    result = forward_and_grad(
        load_attn_res().apply_attention_residual,
        move_inputs(inputs),
        upstream,
        eps=1e-5,
    )
    assert result["output"].shape == (0, 33)
    assert torch.count_nonzero(result["projection"]) == 0
    assert torch.count_nonzero(result["norm"]) == 0


@pytest.mark.parametrize("backend", ["triton", "fla"])
def test_fused_backend_rejects_cpu_instead_of_silently_falling_back(backend):
    inputs, _ = make_inputs()
    with pytest.raises(ValueError, match="requires NPU"):
        load_attn_res().apply_attention_residual(*inputs, eps=1e-5, backend=backend)


def test_eager_import_does_not_require_fla_or_triton():
    before = set(sys.modules)
    inputs, _ = make_inputs()
    load_attn_res().apply_attention_residual(*inputs, eps=1e-5)
    newly_loaded = set(sys.modules) - before
    assert not any(name == "fla" or name.startswith("fla.") for name in newly_loaded)
    assert not any(
        name == "triton" or name.startswith("triton.") for name in newly_loaded
    )


@pytest.mark.parametrize("eps", [0, -1e-5, float("nan"), float("inf")])
def test_invalid_epsilon(eps):
    inputs, _ = make_inputs()
    with pytest.raises(ValueError, match="eps"):
        load_attn_res().apply_attention_residual(*inputs, eps=eps)


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda values: [None, values[1][:, :0], *values[2:]], "at least one"),
        (lambda values: [values[0][:1], *values[1:]], "partial_block_TD"),
        (lambda values: [values[0].double(), *values[1:]], "same dtype"),
        (
            lambda values: [values[0], values[1][:, 0], *values[2:]],
            "block_residual_TND",
        ),
        (lambda values: [*values[:2], values[2][0], values[3]], "projection_weight"),
        (lambda values: [*values[:3], values[3][None]], "norm_weight"),
    ],
)
def test_invalid_input_contract(change, error):
    inputs, _ = make_inputs()
    with pytest.raises(ValueError, match=error):
        load_attn_res().apply_attention_residual(*change(inputs), eps=1e-5)


def test_unknown_backend_fails_explicitly():
    inputs, _ = make_inputs()
    with pytest.raises(ValueError, match="Unsupported"):
        load_attn_res().apply_attention_residual(*inputs, eps=1e-5, backend="ascendc")


def test_checkpoint_recompute_preserves_all_gradients():
    from torch.utils.checkpoint import checkpoint

    inputs, upstream = make_inputs(blocks=5, partial=False)
    operation = load_attn_res().apply_attention_residual
    expected = forward_and_grad(operation, move_inputs(inputs), upstream, eps=1e-5)
    recompute = partial(checkpoint, operation, use_reentrant=False)
    actual = forward_and_grad(recompute, move_inputs(inputs), upstream, eps=1e-5)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("has_partial", [False, True])
def test_backward_formula_and_conditioning_terms_match_fp64_autograd(has_partial):
    inputs, upstream = make_inputs(tokens=7, blocks=3, dim=17, partial=has_partial)
    golden = forward_and_grad(
        golden_attention_residual,
        move_inputs(inputs, dtype=torch.float64),
        upstream.double(),
        eps=0.125,
    )
    for name, result in golden.items():
        index = tuple(size - 1 for size in result.shape)
        terms = reduction_terms(name, inputs, upstream, index, eps=0.125)
        torch.testing.assert_close(terms.sum(), result[index], atol=1e-14, rtol=1e-13)


@pytest.fixture
def fake_spmd_mesh():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=2
    )
    try:
        yield init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("blocks,has_partial", [(0, True), (3, False), (3, True)])
def test_triton_spmd_decomposition_retains_sequence_sharding(
    fake_spmd_mesh, monkeypatch, blocks, has_partial
):
    import spmd_types as spmd
    from spmd_types.checker import typecheck

    from torchtitan.distributed.spmd_types import set_current_spmd_mesh

    module = load_attn_res()
    monkeypatch.setattr(
        module._AttentionResidual,
        "forward",
        staticmethod(
            lambda ctx, current, history, projection, norm, eps: torch.empty(
                (history.shape[0], history.shape[2]), device=history.device
            )
        ),
    )
    fused = module._AttentionResidual
    current, history, projection, norm = make_inputs(
        tokens=4, blocks=blocks, dim=8, partial=has_partial
    )[0]
    with set_current_spmd_mesh(fake_spmd_mesh), typecheck(
        strict_mode="strict", local=False
    ):
        if current is not None:
            spmd.assert_type(current, {"tp": spmd.S(0)})
        spmd.assert_type(history, {"tp": spmd.S(0)})
        spmd.assert_type(projection, {"tp": spmd.R})
        spmd.assert_type(norm, {"tp": spmd.R})
        output = fused.apply(current, history, projection, norm, 1e-5)
        spmd.assert_type(output, {"tp": spmd.S(0)})


def test_fla_spmd_decomposition_retains_sequence_sharding(fake_spmd_mesh, monkeypatch):
    import spmd_types as spmd
    from spmd_types.checker import typecheck

    from torchtitan.distributed.spmd_types import set_current_spmd_mesh

    module = load_attn_res()

    class FakeFlaAttnres(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            return module._fla_attnres_reference(*args)

    def fused_attnres(query, residuals, rms_weight, rms_eps, scale):
        return FakeFlaAttnres.apply(
            query, rms_weight, None, rms_eps, scale, False, 1, *residuals
        )[0]

    mock_public = ModuleType("fla.ops.attnres")
    mock_public.fused_attnres = fused_attnres
    mock_kernel = ModuleType("fla.ops.attnres.fused")
    mock_kernel.FusedAttnresFunction = FakeFlaAttnres
    monkeypatch.setitem(sys.modules, "fla.ops.attnres", mock_public)
    monkeypatch.setitem(sys.modules, "fla.ops.attnres.fused", mock_kernel)
    module._register_fla_attnres_once.cache_clear()
    try:
        fused = module._load_fla_attnres()
        current, history, projection, norm = make_inputs(tokens=4, blocks=3, dim=8)[0]
        with set_current_spmd_mesh(fake_spmd_mesh), typecheck(
            strict_mode="strict", local=False
        ):
            spmd.assert_type(current, {"tp": spmd.S(0)})
            spmd.assert_type(history, {"tp": spmd.S(0)})
            spmd.assert_type(projection, {"tp": spmd.R})
            spmd.assert_type(norm, {"tp": spmd.R})
            residuals = [history[:, index, :].contiguous() for index in range(3)]
            residuals.append(current)
            output = fused(projection[0], residuals, norm, 1e-5, 1.0)
            spmd.assert_type(output, {"tp": spmd.S(0)})
    finally:
        module._register_fla_attnres_once.cache_clear()


@pytest.mark.parametrize("backend", ["triton", "fla"])
def test_fused_registration_is_safe_in_cold_fullgraph_capture(monkeypatch, backend):
    module = load_attn_res()
    monkeypatch.setattr(
        module._AttentionResidual,
        "forward",
        staticmethod(lambda ctx, *args: module._triton_attnres_reference(*args)),
    )

    class FakeFlaAttnres(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            return module._fla_attnres_reference(*args)

    def fused_attnres(query, residuals, rms_weight, rms_eps, scale):
        return FakeFlaAttnres.apply(
            query, rms_weight, None, rms_eps, scale, False, 1, *residuals
        )[0]

    mock_public = ModuleType("fla.ops.attnres")
    mock_public.fused_attnres = fused_attnres
    mock_kernel = ModuleType("fla.ops.attnres.fused")
    mock_kernel.FusedAttnresFunction = FakeFlaAttnres
    monkeypatch.setitem(sys.modules, "fla.ops.attnres", mock_public)
    monkeypatch.setitem(sys.modules, "fla.ops.attnres.fused", mock_kernel)
    module._register_fla_attnres_once.cache_clear()
    try:

        def forward(current, history, projection, norm):
            if backend == "triton":
                return module._AttentionResidual.apply(
                    current, history, projection, norm, 1e-5
                )
            residuals = [history[:, index, :].contiguous() for index in range(3)]
            residuals.append(current)
            return module._load_fla_attnres()(projection[0], residuals, norm, 1e-5, 1.0)

        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        inputs, _ = make_inputs(tokens=4, blocks=3, dim=8)
        actual = compiled(*inputs)
        expected = module.eager_attention_residual(*inputs, eps=1e-5)
        torch.testing.assert_close(actual, expected)
    finally:
        module._register_fla_attnres_once.cache_clear()
