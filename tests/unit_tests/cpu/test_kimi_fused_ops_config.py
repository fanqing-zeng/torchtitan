# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-level configuration, checkpoint, and callback regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.config.transform.moonep import MoonEPTransform
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.attention import kda as kda_module
from torchtitan.models.common.attention.kda import InnerKDA, KDAKernel
from torchtitan.models.common.moonep.routed_experts import MoonEPRoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.norm import GatedRMSNorm
from torchtitan.models.kimi_k3 import build_model_config, configure_fused_ops


def test_default_config_selects_every_fused_kernel():
    config = build_model_config("debugmodel", seq_len=128)
    assert config.attn_res_implementation == "triton"
    assert all(layer.attn_res_implementation == "triton" for layer in config.layers)
    for kind, expected in (
        (SiTUGLU.Config, "ascendc"),
        (RMSNorm.Config, "ascendc"),
        (GatedRMSNorm.Config, "fla"),
    ):
        selected = list(config.traverse(kind))
        assert selected
        assert all(item.implementation == expected for _, item, _, _ in selected)
    assert all(
        inner.conv_implementation == "ascendc"
        for _, inner, _, _ in config.traverse(InnerKDA.Config)
    )
    assert all(
        kernel.implementation == "auto"
        for _, kernel, _, _ in config.traverse(KDAKernel.Config)
    )


def test_reference_and_individual_overrides():
    config = build_model_config("debugmodel", use_npu_fused_ops=False)
    assert config.attn_res_implementation == "eager"
    assert all(
        norm.implementation == "eager"
        for _, norm, _, _ in config.traverse(RMSNorm.Config)
    )
    configure_fused_ops(config, causal_conv1d="eager", attn_res="fla", kda="fla")
    assert config.attn_res_implementation == "fla"
    assert config.layers[0].feed_forward.activation_fn.implementation == "ascendc"
    assert all(
        inner.conv_implementation == "eager"
        for _, inner, _, _ in config.traverse(InnerKDA.Config)
    )
    assert all(
        kernel.implementation == "fla"
        for _, kernel, _, _ in config.traverse(KDAKernel.Config)
    )


@pytest.mark.parametrize(
    "field",
    ["situ_glu", "rms_norm", "gated_rms_norm", "causal_conv1d", "kda", "attn_res"],
)
def test_invalid_selection_rejected_without_partial_mutation(field):
    config = build_model_config("debugmodel", use_npu_fused_ops=False)
    with pytest.raises(ValueError, match=field):
        configure_fused_ops(config, **{field: "invalid"})
    assert config.attn_res_implementation == "eager"


def test_checkpoint_parameter_names_and_shapes_are_unchanged():
    reference_config = build_model_config(
        "debugmodel", seq_len=128, use_npu_fused_ops=False
    )
    fused_config = build_model_config("debugmodel", seq_len=128)
    with torch.device("meta"):
        reference = reference_config.build()
        fused = fused_config.build()
    expected = reference.state_dict()
    actual = fused.state_dict()
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
    fused.load_state_dict(expected, strict=True)


def test_moonep_transform_retains_situ_and_packed_callback_gradients():
    config = build_model_config("debugmodel", seq_len=128)
    source = config.layers[1].moe.routed_experts
    activation_config = source.activation_fn
    MoonEPTransform().transform(config)
    converted = config.layers[1].moe.routed_experts
    assert isinstance(converted, MoonEPRoutedExperts.Config)
    assert converted.activation_fn is activation_config
    assert converted.activation_fn.implementation == "ascendc"
    with torch.device("meta"):
        experts = converted.build()
    packed = torch.randn(7, 2 * converted.w13.out_features, requires_grad=True)
    reference = packed.detach().clone().requires_grad_()
    output = experts._apply_activation(packed)
    gate, up = reference.chunk(2, dim=-1)
    expected = 4 * (gate / 4).tanh() * gate.sigmoid() * 25 * (up / 25).tanh()
    torch.testing.assert_close(output, expected)
    upstream = torch.randn_like(output)
    torch.testing.assert_close(
        torch.autograd.grad(output, packed, upstream)[0],
        torch.autograd.grad(expected, reference, upstream)[0],
    )


def test_kda_stages_use_ascendc_conv_and_configured_core(monkeypatch):
    inner = InnerKDA.Config(
        head_dim=128,
        conv_kernel_size=4,
        conv_implementation="ascendc",
        kernel=KDAKernel.Config(implementation="ascendc"),
    ).build()
    x = torch.randn(1, 7, 3 * 256)
    weight = torch.randn(3 * 256, 1, 4)
    boundaries = torch.tensor([0, 2, 7])
    calls = []

    def convolution(received_x, received_weight, **kwargs):
        calls.append((received_x, received_weight, kwargs))
        return received_x

    monkeypatch.setattr(kda_module, "ascendc_causal_conv1d", convolution)
    with patch.object(
        torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
    ):
        actual = inner.short_convolution(x, weight, cu_seqlens=boundaries, routing=None)
    assert actual is x
    assert calls[0][0] is x
    torch.testing.assert_close(calls[0][1], weight[:, 0])
    assert calls[0][2]["num_heads"] == 6
    assert calls[0][2]["cu_seqlens"] is boundaries
    kernel_calls = []

    def kernel(*args, **kwargs):
        kernel_calls.append(kwargs)
        return args[2], None

    monkeypatch.setattr(kda_module, "npu_chunk_kda", kernel)
    q = torch.randn(1, 7, 2, 128)
    raw_beta = torch.randn(1, 7, 2)
    inner.kernel(
        q, q, q, q, raw_beta, torch.zeros(2), torch.zeros(2, 128), cu_seqlens=boundaries
    )
    assert kernel_calls[0]["backend"] == "ascendc"
    assert kernel_calls[0]["cu_seqlens"] is boundaries


def test_attention_residual_model_wrapper_selects_configured_backend(monkeypatch):
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.kimi_k3 import attn_res
    from torchtitan.models.kimi_k3.model import _apply_attention_residual

    calls = []
    projection = Linear.Config(in_features=16, out_features=1, bias=False).build()
    norm = RMSNorm.Config(normalized_shape=16, eps=2e-4).build()
    history = torch.randn(7, 3, 16)
    partial = torch.randn(7, 16)

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        return partial

    monkeypatch.setattr(attn_res, "apply_attention_residual", operation)
    with patch.object(
        torch.Tensor, "device", property(lambda _: SimpleNamespace(type="npu"))
    ):
        result = _apply_attention_residual(
            partial, history, projection, norm, implementation="triton"
        )
    assert result is partial
    assert calls[0][0][2] is projection.weight
    assert calls[0][0][3] is norm.weight
    assert calls[0][1] == {"eps": 2e-4, "backend": "triton"}
