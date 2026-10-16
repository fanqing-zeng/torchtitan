# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-NPU integration of Kimi's fused training operators.

Both model variants use AscendC KDA; its independent recurrence oracle lives
in test_ascendc_kda_numerics.py. The reference disables fusion for convolution,
SiTU, RMSNorm, gated RMSNorm, and attention residuals. This module exercises a
small language model and MoonEP's activation callback, without EP communication,
MLA, vision, distributed sharding, or compilation. Missing dependencies fail
after an available NPU is detected. Run on the production Ascend software stack.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

pytest.importorskip("torch_npu", reason="Kimi fused integration requires torch_npu")
if not torch.npu.is_available():
    pytest.skip("Kimi fused integration requires an NPU", allow_module_level=True)

from torchtitan.config.transform.moonep import MoonEPTransform
from torchtitan.models.common.moonep import MoonEPRoutedExperts
from torchtitan.models.kimi_k3.flavors import build_model_config, configure_fused_ops


def _small_config(*, fused):
    config = build_model_config("debugmodel", seq_len=65, use_npu_fused_ops=False)
    config.layers = config.layers[:2]
    config.vision_encoder = None
    config.vocab_size = 256
    config.tok_embeddings.num_embeddings = 256
    config.lm_head.out_features = 256
    config.local_compile_regions = []
    dense = deepcopy(config.layers[0].feed_forward)
    assert dense is not None
    for layer in config.layers:
        assert layer.delta_attention is not None
        assert layer.delta_attention.head_dim == 128
        layer.attention = None
        layer.moe = None
        layer.feed_forward = deepcopy(dense)
        # Two layers close distinct blocks: the second attention residual and
        # the output head both see multiple completed residual sources.
        layer.attn_res_block_size = 1
    return configure_fused_ops(
        config,
        situ_glu="ascendc" if fused else "eager",
        rms_norm="ascendc" if fused else "eager",
        gated_rms_norm="fla" if fused else "eager",
        causal_conv1d="ascendc" if fused else "eager",
        kda="ascendc",
        attn_res="triton" if fused else "eager",
    )


def _assert_close(actual, expected, *, name, relative_tolerance, absolute_tolerance):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), f"Non-finite {name}"
    assert torch.isfinite(expected).all(), f"Non-finite reference {name}"
    error_rms = (actual - expected).square().mean().sqrt()
    reference_rms = expected.square().mean().sqrt()
    limit = absolute_tolerance + relative_tolerance * reference_rms
    assert error_rms <= limit, (
        f"{name}: RMS error {error_rms.item():.6g} exceeds {limit.item():.6g}; "
        f"reference RMS {reference_rms.item():.6g}"
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance + relative_tolerance * reference_rms.item(),
        msg=lambda message: f"{name}: {message}",
    )


def _forward_backward(model, tokens, labels, positions, *, recompute):
    metadata = model._get_attention_metadata(positions)
    assert len(metadata) == 1
    linear_metadata = next(iter(metadata.values()))
    torch.testing.assert_close(
        linear_metadata.varlen.cu_seq_q.cpu(),
        torch.tensor([0, 7, 65], dtype=linear_metadata.varlen.cu_seq_q.dtype),
    )

    def forward(input_tokens):
        return model(input_tokens, positions=positions, attention_metadata=metadata)

    logits = (
        checkpoint(forward, tokens, use_reentrant=False)
        if recompute
        else forward(tokens)
    )
    assert logits.shape == (65, 256)
    loss = F.cross_entropy(logits.float(), labels)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Missing parameter gradient: {name}"
            gradients[name] = parameter.grad.detach().float().cpu().clone()
    return logits.detach(), loss.detach(), gradients


@pytest.mark.parametrize("recompute", [False, True])
def test_combined_fusions_training_update_and_checkpoint_compatibility(recompute):
    torch.manual_seed(42)
    reference = _small_config(fused=False).build()
    reference.init_states()
    # Zero-initialized residual queries would hide norm-weight gradients in
    # the first step. Exercise nonuniform depth attention from the first run.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("res_proj.weight"):
                parameter.normal_(mean=0.0, std=0.02)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
    }
    fused = _small_config(fused=True).build()
    result = fused.load_state_dict(initial_state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert fused.state_dict().keys() == reference.state_dict().keys()
    reference = reference.to(device="npu", dtype=torch.bfloat16).train()
    fused = fused.to(device="npu", dtype=torch.bfloat16).train()
    for layer in fused.layers.values():
        assert layer.delta_attention.inner_kda.conv_implementation == "ascendc"
        assert layer.delta_attention.inner_kda.kernel.implementation == "ascendc"
        assert layer.delta_attention.output_norm.implementation == "fla"
        assert layer.feed_forward.activation_fn.implementation == "ascendc"
        assert layer.attention_norm.implementation == "ascendc"
        assert layer.attn_res_implementation == "triton"
    assert fused.attn_res_implementation == "triton"

    generator = torch.Generator().manual_seed(43)
    tokens = torch.randint(256, (65,), generator=generator).npu()
    labels = torch.randint(256, (65,), generator=generator).npu()
    positions = torch.cat((torch.arange(7), torch.arange(58))).to(
        device="npu", dtype=torch.int32
    )
    expected_logits, expected_loss, expected_gradients = _forward_backward(
        reference, tokens, labels, positions, recompute=recompute
    )
    actual_logits, actual_loss, actual_gradients = _forward_backward(
        fused, tokens, labels, positions, recompute=recompute
    )
    # These integration limits include BF16 accumulation across operators.
    # Individual operator tests provide tighter independent numerical checks.
    _assert_close(
        actual_logits,
        expected_logits,
        name="logits",
        relative_tolerance=0.04,
        absolute_tolerance=2e-3,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=5e-3, atol=2e-3)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, gradient in actual_gradients.items():
        _assert_close(
            gradient,
            expected_gradients[name],
            name=f"gradient {name}",
            relative_tolerance=0.08,
            absolute_tolerance=2e-5,
        )

    # Compare actual optimizer deltas as well as resulting parameters, so the
    # original weights cannot hide a skipped or disconnected update.
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in reference.named_parameters()
    }
    torch.optim.SGD(reference.parameters(), lr=0.03, foreach=False).step()
    torch.optim.SGD(fused.parameters(), lr=0.03, foreach=False).step()
    expected_update_square = 0.0
    update_error_square = 0.0
    for (name, actual), (reference_name, expected) in zip(
        fused.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        actual_cpu, expected_cpu = (
            actual.detach().float().cpu(),
            expected.detach().float().cpu(),
        )
        expected_update = expected_cpu - before[name]
        actual_update = actual_cpu - before[name]
        expected_update_square += expected_update.double().square().sum().item()
        update_error_square += (
            (actual_update - expected_update).double().square().sum().item()
        )
        _assert_close(
            actual_cpu,
            expected_cpu,
            name=f"updated parameter {name}",
            relative_tolerance=0.01,
            absolute_tolerance=2e-4,
        )
    assert expected_update_square > 0, "The optimizer made no update"
    # BF16 rounding amplifies differences at individual update thresholds.
    assert update_error_square**0.5 <= 0.25 * expected_update_square**0.5

    saved_state = {
        name: tensor.detach().clone() for name, tensor in fused.state_dict().items()
    }
    result = reference.load_state_dict(saved_state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(tensor, saved_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [(17,), (2, 17)])
def test_moonep_situ_callback_after_config_transform_without_communication(prefix):
    config = build_model_config("debugmodel", seq_len=65, use_npu_fused_ops=True)
    converted = MoonEPTransform().transform(config)
    fused_config = converted.layers[1].moe.routed_experts
    assert isinstance(fused_config, MoonEPRoutedExperts.Config)
    assert fused_config.activation_fn.implementation == "ascendc"
    eager_config = deepcopy(fused_config)
    eager_config.activation_fn.implementation = "eager"
    fused, reference = fused_config.build(), eager_config.build()
    assert fused._runtime is reference._runtime is None
    assert fused._moonep_forward is reference._moonep_forward is None
    hidden_dim = fused.intermediate_dim
    generator = torch.Generator().manual_seed(44)
    gate_up = torch.randn(*prefix, 2 * hidden_dim, generator=generator).to(
        torch.bfloat16
    )
    actual_input = gate_up.npu().requires_grad_()
    expected_input = gate_up.npu().requires_grad_()
    actual = fused._apply_activation(actual_input)
    expected = reference._apply_activation(expected_input)
    assert actual.shape == (*prefix, hidden_dim)
    upstream = torch.randn(*actual.shape, generator=generator).to(
        device="npu", dtype=torch.bfloat16
    )
    actual.backward(upstream)
    expected.backward(upstream)
    _assert_close(
        actual,
        expected,
        name="MoonEP activation",
        relative_tolerance=0.02,
        absolute_tolerance=2e-3,
    )
    _assert_close(
        actual_input.grad,
        expected_input.grad,
        name="MoonEP gate/up gradients",
        relative_tolerance=0.02,
        absolute_tolerance=2e-3,
    )
