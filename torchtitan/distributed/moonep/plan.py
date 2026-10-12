# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan. The upstream helpers that back-fill
# ``tokens_per_rank`` / ``top_k`` from an FSDPTurbo EP plan are not carried
# over: torchtitan derives both on the ``MoonEPTokenDispatcher.Config``.

"""Static-shape helpers for MoonEP's group layout and plan invariants.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

import torch


def compact_group_ends(
    cu_seqlens: torch.Tensor,
    rank: int,
    experts_per_rank: int,
    num_experts: int,
    prefetch_slots: int,
    num_rows: int,
) -> torch.Tensor:
    """Project MoonEP's ``[E+B]`` group ends onto the ``epn+B`` groups this rank computes.

    Valid because a rank only receives tokens for its own home experts and for
    the slots it prefetched, so every other group is empty and
    ``[home | slots]`` is already a monotone partition of the live prefix.
    The upstream report proves ``B = E/R`` always suffices for that.

    The last offset is extended to ``num_rows`` so the GEMM covers the whole
    static buffer without a host read, which folds MoonEP's reserved tail into
    the final slot group.
    """
    start = rank * experts_per_rank
    ends = torch.cat(
        (
            cu_seqlens[start : start + experts_per_rank],
            cu_seqlens[num_experts : num_experts + prefetch_slots],
        )
    ).clone()
    ends[-1] = num_rows
    return ends


def full_group_ends(cu_seqlens: torch.Tensor, num_rows: int) -> torch.Tensor:
    """MoonEP's own ``[E+B]`` group ends, used where the whole mapping is readable."""
    ends = cu_seqlens.clone()
    ends[-1] = num_rows
    return ends


def zero_static_tail_(rows: torch.Tensor, last_group_end: torch.Tensor) -> torch.Tensor:
    """Zero the rows past the last live group, in place.

    MoonEP leaves its static tail undefined. The ``[E+B]`` group set has no
    spare weight row to absorb it, so it would contaminate the last prefetch
    slot's weight gradient.
    """
    active = torch.arange(rows.shape[0], device=rows.device) < last_group_end
    return rows.mul_(active.unsqueeze(-1))


def verify_plan_locality(cu_seqlens: torch.Tensor, rank: int, experts_per_rank: int, num_experts: int) -> None:
    """Check that no tokens land outside this rank's group blocks.

    The invariant :func:`compact_group_ends` relies on. Checked on the host
    rather than with ``torch._assert_async``, whose failure poisons the device
    context and surfaces as a crash with no message.
    """
    start = rank * experts_per_rank
    ends = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    leading = ends[start - 1] if start > 0 else 0
    local_end = ends[start + experts_per_rank - 1]
    experts_end = ends[num_experts - 1]
    if leading == 0 and experts_end == local_end:
        return

    stray = [
        group
        for group in range(num_experts)
        if not start <= group < start + experts_per_rank and ends[group] > (ends[group - 1] if group else 0)
    ]
    raise RuntimeError(
        "MoonEP planned tokens for expert groups this rank does not own: groups "
        f"{stray[:16]}{'...' if len(stray) > 16 else ''}. Every routed token must land "
        f"on one of this rank's [{start}, {start + experts_per_rank}) home experts or in "
        f"one of its {cu_seqlens.numel() - num_experts} prefetch slots. This is a planner "
        "problem, not a configuration one: on Ascend the forward's weight operand maps "
        "only those groups, so there is nothing to fall back to."
    )


def verify_routing_range(top_k_index: torch.Tensor, num_experts: int) -> None:
    """Check that every routed expert id is in ``[0, E)``.

    An out-of-range id corrupts the planning kernel in ways that are hard to
    attribute, so it is worth one synchronization during bring-up.
    """
    lowest = int(top_k_index.min())
    highest = int(top_k_index.max())
    if 0 <= lowest and highest < num_experts:
        return
    raise RuntimeError(f"MoonEP expert ids must be in [0, {num_experts}); got [{lowest}, {highest}].")
