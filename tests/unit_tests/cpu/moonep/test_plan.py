# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for MoonEP's compact expert-GEMM group layout.

Runs on CPU only - no accelerator, no MoonEP extension, no process group - so it
can gate every commit.
"""

import pytest
import torch

from torchtitan.distributed.moonep.plan import (
    compact_group_ends,
    full_group_ends,
    verify_plan_locality,
    verify_routing_range,
    zero_static_tail_,
)


def _layout(counts, token_padding, num_experts, slots):
    """Build a ``[E+B]`` cu_seqlens tensor from per-group token counts."""
    ends = []
    cursor = 0
    for count in counts:
        if count:
            cursor += -(-count // token_padding) * token_padding
        ends.append(cursor)
    assert len(ends) == num_experts + slots
    return torch.tensor(ends, dtype=torch.int32)


def test_compact_group_ends_selects_local_and_slot_groups():
    num_experts, num_ranks, rank = 16, 4, 2
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank
    counts = [0] * (num_experts + slots)
    # This rank's own experts carry tokens ...
    for local in range(experts_per_rank):
        counts[rank * experts_per_rank + local] = 10 * (local + 1)
    # ... and two migrated experts sit in prefetch slots.
    counts[num_experts + 0] = 40
    counts[num_experts + 1] = 25

    cu_seqlens = _layout(counts, token_padding=8, num_experts=num_experts, slots=slots)
    rows = int(cu_seqlens[-1]) + 5
    ends = compact_group_ends(cu_seqlens, rank, experts_per_rank, num_experts, slots, rows)

    assert ends.shape == (experts_per_rank + slots,)
    assert ends.dtype == cu_seqlens.dtype
    # Monotone, and the last group is extended to cover the reserved tail.
    assert torch.all(ends[1:] >= ends[:-1])
    assert int(ends[-1]) == rows

    # The per-group sizes must match the original layout for every live group
    # except the last, which absorbs the tail.
    sizes = torch.diff(ends, prepend=torch.zeros(1, dtype=ends.dtype))
    expected = [-(-counts[rank * experts_per_rank + i] // 8) * 8 for i in range(experts_per_rank)]
    expected += [-(-counts[num_experts + i] // 8) * 8 if counts[num_experts + i] else 0 for i in range(slots)]
    assert sizes[:-1].tolist() == expected[:-1]

    # The source must not be mutated: the adapter reads cu_seqlens[-1] to
    # neutralize the tail.
    assert int(cu_seqlens[-1]) == rows - 5


def test_compact_group_ends_is_a_full_cover_when_padding_is_disabled():
    """Ascend drops token padding, so the compact set covers every row exactly."""
    num_experts, num_ranks, rank = 8, 2, 1
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank
    counts = [0] * (num_experts + slots)
    for local in range(experts_per_rank):
        counts[rank * experts_per_rank + local] = 3
    counts[num_experts] = 7

    cu_seqlens = _layout(counts, token_padding=1, num_experts=num_experts, slots=slots)
    rows = int(cu_seqlens[-1])
    ends = compact_group_ends(cu_seqlens, rank, experts_per_rank, num_experts, slots, rows)

    # Without padding the compact set already ends at the buffer end, so
    # extending the last offset changes nothing.
    assert int(ends[-1]) == rows
    assert ends.tolist() == (
        cu_seqlens[rank * experts_per_rank : (rank + 1) * experts_per_rank].tolist() + cu_seqlens[num_experts:].tolist()
    )


def test_compact_group_ends_handles_rank_zero():
    """Rank 0 has no preceding groups, so the prefix slice must stay in range."""
    num_experts, num_ranks, rank = 8, 4, 0
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank
    counts = [0] * (num_experts + slots)
    counts[0] = 5
    counts[1] = 6
    counts[num_experts] = 4

    cu_seqlens = _layout(counts, token_padding=4, num_experts=num_experts, slots=slots)
    ends = compact_group_ends(cu_seqlens, rank, experts_per_rank, num_experts, slots, int(cu_seqlens[-1]) + 2)
    assert ends[0] == 8  # 5 tokens padded to 8
    assert ends.numel() == experts_per_rank + slots


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_compact_group_ends_group_count(slots):
    num_experts, num_ranks, rank = 8, 2, 0
    experts_per_rank = num_experts // num_ranks
    cu_seqlens = torch.zeros(num_experts + slots, dtype=torch.int32)
    ends = compact_group_ends(cu_seqlens, rank, experts_per_rank, num_experts, slots, 0)
    assert ends.numel() == experts_per_rank + slots


def test_full_group_ends_keeps_global_indexing_and_covers_the_tail():
    cu_seqlens = torch.tensor([8, 8, 24, 32], dtype=torch.int32)
    ends = full_group_ends(cu_seqlens, 40)
    assert ends.tolist() == [8, 8, 24, 40]
    # The source must not be mutated: MoonEP reuses it and the adapter reads
    # cu_seqlens[-1] when neutralizing the tail.
    assert cu_seqlens.tolist() == [8, 8, 24, 32]


def test_zero_static_tail_masks_in_place_without_allocating_a_copy():
    rows = torch.ones(6, 3)
    returned = zero_static_tail_(rows, torch.tensor(4))
    assert returned.data_ptr() == rows.data_ptr()
    assert torch.count_nonzero(rows[:4]) == 12
    assert torch.count_nonzero(rows[4:]) == 0


def test_verify_plan_locality_accepts_a_conforming_plan():
    num_experts, num_ranks, rank = 8, 2, 1
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank
    counts = [0] * (num_experts + slots)
    for local in range(experts_per_rank):
        counts[rank * experts_per_rank + local] = 4
    counts[num_experts] = 8
    verify_plan_locality(_layout(counts, 4, num_experts, slots), rank, experts_per_rank, num_experts)


def test_verify_plan_locality_names_the_offending_groups():
    """A remote expert outside the prefetch slots must be reported, not asserted."""
    num_experts, num_ranks, rank = 8, 2, 1
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank
    counts = [0] * (num_experts + slots)
    counts[0] = 4  # expert 0 belongs to rank 0, yet carries tokens here
    for local in range(experts_per_rank):
        counts[rank * experts_per_rank + local] = 4

    with pytest.raises(RuntimeError) as error:
        verify_plan_locality(_layout(counts, 4, num_experts, slots), rank, experts_per_rank, num_experts)
    message = str(error.value)
    assert "[0]" in message
    # No configuration escapes this: on Ascend the forward's weight operand maps
    # only these groups, so the message has to send the reader at the planner.
    assert "planner problem" in message


def test_verify_routing_range_rejects_out_of_range_ids():
    verify_routing_range(torch.tensor([[0, 3], [7, 1]], dtype=torch.int32), num_experts=8)
    with pytest.raises(RuntimeError, match=r"\[0, 8\)"):
        verify_routing_range(torch.tensor([[0, 8]], dtype=torch.int32), num_experts=8)
    with pytest.raises(RuntimeError, match=r"\[0, 8\)"):
        verify_routing_range(torch.tensor([[-1, 2]], dtype=torch.int32), num_experts=8)


def test_plan_invariants_accept_duplicate_expert_ids_from_padding():
    """Pad rows fill every K slot with expert 0; that is not a real top-k.

    ``call.validate`` in the dispatcher feeds ``[S, K]`` to
    ``verify_routing_range``. Group ends come from ``compact_group_ends`` in
    ``MoonEPCallState.set_plan``.
    """
    index = torch.tensor([[1, 2], [3, 4], [0, 0], [0, 0]], dtype=torch.int32)
    num_experts, num_ranks, rank = 8, 2, 0
    experts_per_rank = num_experts // num_ranks
    slots = experts_per_rank

    verify_routing_range(index, num_experts)

    counts = [0] * (num_experts + slots)
    for expert_id in index.reshape(-1).tolist():
        counts[expert_id] += 1
    cu_seqlens = _layout(counts, token_padding=1, num_experts=num_experts, slots=slots)
    ends = compact_group_ends(cu_seqlens, rank, experts_per_rank, num_experts, slots, int(cu_seqlens[-1]))
    assert torch.all(ends[1:] >= ends[:-1])
