# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the MoonEP expert forward's host-side helpers.

The token histogram is the one piece of the dispatcher that can be checked
without a device: MoonEP's planner needs an ``[E]`` count per step, and reading
it back with ``torch.bincount`` would move a device scalar to the host on the
critical path.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torchtitan.distributed.moonep.dispatcher import (  # noqa: E402
    _activate_expert_hidden,
    _fit_tokens_to_static_rank,
    _tokens_per_expert,
)


def test_tokens_per_expert_matches_bincount_without_host_sync():
    torch.manual_seed(7)
    num_experts = 32
    top_k_index = torch.randint(0, num_experts, (128, 4), dtype=torch.int32)
    counts = _tokens_per_expert(top_k_index, num_experts)
    expected = torch.bincount(top_k_index.reshape(-1).long(), minlength=num_experts)
    assert counts.dtype == torch.int32
    assert counts.shape == (num_experts,)
    torch.testing.assert_close(counts.long(), expected)


def test_tokens_per_expert_covers_unused_experts():
    """The histogram must always be [E] long, even when experts get no tokens."""
    top_k_index = torch.zeros(8, 2, dtype=torch.int32)
    counts = _tokens_per_expert(top_k_index, num_experts=64)
    assert counts.shape == (64,)
    assert int(counts[0]) == 16
    assert int(counts[1:].sum()) == 0


def test_swiglu_activation_splits_then_multiplies():
    module = SimpleNamespace(act_fn=torch.nn.SiLU(), limit=None)
    gate_up = torch.tensor([[1.0, -1.0, 2.0, 3.0]])
    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(_activate_expert_hidden(module, gate_up), module.act_fn(gate) * up)


def test_situ_activation_consumes_the_concatenated_projection():
    """Kimi's SiTU transforms both halves; chunk-then-multiply would be wrong."""

    class SituAndMul:
        def __call__(self, gate_up):
            return gate_up.sum(dim=-1, keepdim=True)

    module = SimpleNamespace(act_fn=SituAndMul())
    gate_up = torch.ones(2, 4)
    torch.testing.assert_close(_activate_expert_hidden(module, gate_up), torch.full((2, 1), 4.0))


def test_module_apply_activation_wins_over_generic_split():
    def apply_activation(gate_up):
        return gate_up[..., :1] + 7

    module = SimpleNamespace(act_fn=torch.nn.SiLU(), _apply_activation=apply_activation)
    out = _activate_expert_hidden(module, torch.zeros(3, 4))
    torch.testing.assert_close(out, torch.full((3, 1), 7.0))


def test_fit_tokens_is_noop_when_already_at_capacity():
    hidden = torch.ones(4, 8)
    index = torch.arange(4).unsqueeze(-1)
    weights = torch.full((4, 1), 0.5)
    out_h, out_i, out_w = _fit_tokens_to_static_rank(hidden, index, weights, 4)
    assert out_h is hidden
    assert out_i is index
    assert out_w is weights


def test_fit_tokens_pads_a_short_microbatch_without_touching_the_prefix():
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    index = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    out_h, out_i, out_w = _fit_tokens_to_static_rank(hidden, index, weights, 5)
    assert out_h.shape == (5, 3)
    assert out_i.shape == (5, 2)
    assert out_w.shape == (5, 2)
    torch.testing.assert_close(out_h[:2], hidden)
    torch.testing.assert_close(out_i[:2], index)
    torch.testing.assert_close(out_w[:2], weights)
    assert torch.count_nonzero(out_h[2:]) == 0
    assert torch.count_nonzero(out_i[2:]) == 0
    assert torch.count_nonzero(out_w[2:]) == 0


def test_fit_tokens_rejects_a_microbatch_longer_than_capacity():
    hidden = torch.zeros(5, 2)
    index = torch.zeros(5, 1, dtype=torch.int32)
    weights = torch.ones(5, 1)
    with pytest.raises(RuntimeError, match="reserved S=4"):
        _fit_tokens_to_static_rank(hidden, index, weights, 4)


def test_tokens_per_expert_counts_pad_rows_into_expert_zero():
    # If dispatcher.py later counts tokens_per_expert before padding, this
    # test fails while the rest of the file stays green.
    hidden = torch.zeros(2, 4)
    index = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    weights = torch.zeros(2, 2)
    _, padded_index, _ = _fit_tokens_to_static_rank(hidden, index, weights, 4)
    counts = _tokens_per_expert(padded_index, num_experts=8)
    assert int(counts[0]) == 4
    assert [int(c) for c in counts[1:5]] == [1, 1, 1, 1]
