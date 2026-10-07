# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MoonEP primitives for MoE Expert Parallel.

MoonEP (https://github.com/MoonshotAI/MoonEP) keeps token loads perfectly
balanced across EP ranks via dynamic redundant experts: an online planner
moves surplus tokens to remote ranks' spare expert slots and prefetches the
corresponding expert weights over NVLink before expert computation. Every
rank always computes exactly ``S * K`` tokens with static shapes, so there is
no host sync and no per-layer dynamic allocation.

Layout contract (H = expert input dim, F = expert hidden dim, epn = E / R):

- ``dispatch`` returns tokens in expert-major VM-group order:
  ``hidden_nvsh [NvS, H]`` with ``NvS`` the padded per-rank slot count and
  ``cu_seqlens [2 * epn]`` the padded end offset per VM-group row. Rows
  ``[0, epn)`` belong to this rank's local experts; rows ``[epn, 2 * epn)``
  to this rank's prefetch slots (remote experts copied in by
  ``prefetch_weight``).
- The integrating framework exposes expert weights as a
  ``[2 * epn, H, F]`` compute view: rows ``[0, epn)`` alias the local
  expert parameters, rows ``[epn, 2 * epn)`` alias this rank's slice of the
  process-global prefetch pool. The grouped GEMM runs over that view with
  ``cumsum(cu_seqlens)`` offsets.
- ``combine`` applies the routing weights and reduces the VM-group rows
  back to token-major order.

Autograd: MoonEP's kernels are not autograd-aware, so dispatch/combine are
wrapped in ``torch.library`` custom ops with manual backward (dispatch
backward is a combine, combine backward is a re-dispatch through the saved
plan), mirroring the DeepEP integration in
``torchtitan/distributed/deepep/deepep.py``. The opaque
``MoonEPCommPlan`` crosses the op boundary via a CPU handle id + cache.

Weight prefetch happens in ``MoonEPGroupedExperts.forward`` (the experts own
the local parameters; the pools are process-global and shared across
layers). Gradient reduce for prefetch-slot experts (``reduce_grad``) is not
wired yet -- see the TODO in ``MoonEPGroupedExperts``; until it is, MoonEP
dispatch is forward-correct but the redundant-expert gradient path is
incomplete, so training with EP > 1 requires that TODO first.
"""

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup

try:
    from moonep import Buffer
except ImportError as e:
    raise ImportError(
        "MoonEP is required for this module. "
        "Install from: https://github.com/MoonshotAI/MoonEP"
    ) from e


# Process-global MoonEP buffer (one per process, recreated if the group or a
# larger size is needed). Holds the NVLink/VMM communication buffers and the
# async comm stream.
_buffer: Buffer | None = None

# Process-global weight prefetch pools and gradient reduce buffers, one
# buffer per projection. Shared by all MoE layers: pools are overwritten by
# each layer's prefetch before its expert compute, so the physical cost is
# ``epn`` expert weights per projection in total, not per layer.
# Shapes: w1/w3 pools [R, epn, F, H] bf16, w2 pool [R, epn, H, F] bf16;
# grad buffers mirror them in fp32.
_pools: dict[str, torch.Tensor] = {}
_grad_buffers: dict[str, torch.Tensor] = {}

# Global cache for dispatch plans, keyed by an int handle id. The custom ops
# pass tensors across the op boundary, so the plan is smuggled through a CPU
# int64 handle id tensor + this cache (same pattern as the DeepEP wrapper).
_plan_cache: dict = {}
_plan_counter: int = 0

# Plan handed off from the dispatcher to MoonEPGroupedExperts within one
# MoE forward: dispatch saves it, the experts' prefetch/GEMM consume it.
_current_plan = None
_current_plan_weights = None


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle id tensor on CPU to avoid a GPU-CPU sync."""
    global _plan_counter
    _plan_counter += 1
    return torch.tensor([_plan_counter], dtype=torch.int64, device="cpu")


def get_buffer(
    group: ProcessGroup,
    *,
    hidden: int,
    expert_hidden: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
) -> Buffer:
    """Get or create the process-global MoonEP ``Buffer`` and weight pools.

    Args:
        group: EP process group.
        hidden: Expert input dim (the dispatcher moves tokens in this space,
            the latent dim for latent-MoE models).
        expert_hidden: Expert FFN intermediate dim (pool weight shapes).
        num_max_tokens_per_rank: Per-rank input token capacity ``S``.
        num_topk: Routing top-k ``K``.
        num_experts: Total routed experts ``E`` in the EP group.
    """
    global _buffer, _pools
    num_ep_ranks = group.size()
    num_local_experts = num_experts // num_ep_ranks
    if (
        _buffer is not None
        and _buffer.destroyed is False
    ):
        return _buffer
    _buffer = Buffer(
        S=num_max_tokens_per_rank,
        H=hidden,
        K=num_topk,
        E=num_experts,
        num_ep_ranks=num_ep_ranks,
        explicitly_destroy=True,
    )
    rank = group.rank()
    device = torch.cuda.current_device()
    # Prefetch pools: only this rank's [epn, ...] slice is ever read by the
    # compute view, but the all-rank view [R, epn, ...] is what
    # ``prefetch_weight`` maps for remote writes.
    _pools = {
        "w1": torch.empty(
            num_ep_ranks, num_local_experts, expert_hidden, hidden,
            dtype=torch.bfloat16, device=device,
        ),
        "w2": torch.empty(
            num_ep_ranks, num_local_experts, hidden, expert_hidden,
            dtype=torch.bfloat16, device=device,
        ),
        "w3": torch.empty(
            num_ep_ranks, num_local_experts, expert_hidden, hidden,
            dtype=torch.bfloat16, device=device,
        ),
    }
    del rank  # rank slices are views taken per use; pools are whole-group
    return _buffer


def destroy_buffer() -> None:
    """Release the process-global buffer and pools before group teardown."""
    global _buffer, _pools, _grad_buffers
    if _buffer is not None and not _buffer.destroyed:
        _buffer.destroy()
    _buffer = None
    _pools = {}
    _grad_buffers = {}


def get_pools() -> dict[str, torch.Tensor]:
    """The process-global prefetch pools (see module docstring)."""
    assert _pools, "MoonEP pools must be initialized via get_buffer()"
    return _pools


def stash_current_plan(plan, route_weights_nvs: torch.Tensor | None) -> None:
    """Hand the freshly produced plan to ``MoonEPGroupedExperts.forward``."""
    global _current_plan, _current_plan_weights
    _current_plan = plan
    _current_plan_weights = route_weights_nvs


def pop_current_plan():
    """Consume the plan handoff; returns ``(plan, route_weights_nvs)``."""
    global _current_plan, _current_plan_weights
    plan, weights = _current_plan, _current_plan_weights
    _current_plan = None
    _current_plan_weights = None
    return plan, weights


# ============================================================================
# Custom op registration for SAC integration + autograd
# ============================================================================

_lib = torch.library.Library("moonep", "DEF")

# dispatch returns: (hidden_nvsh, route_weights_nvs, cu_seqlens, handle_id).
_lib.define(
    "dispatch(Tensor x, Tensor topk_scores, Tensor topk_ids, Tensor tpe, "
    "int num_experts) -> (Tensor, Tensor, Tensor, Tensor)"
)
# combine returns: combined_x. ``will_backward`` is the caller's outer grad
# state: when False the op frees the plan itself (no backward will consume
# it), matching the DeepEP wrapper's contract.
_lib.define("combine(Tensor x, Tensor handle_id, bool will_backward) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_ids: torch.Tensor,
    tpe: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute MoonEP dispatch (planning + scatter to expert-grouped slots)."""
    global _plan_cache
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
        hidden_sh=x,
        route_weights_sk=topk_scores.float(),
        topk_experts_sk=topk_ids.to(torch.int32),
        tokens_per_expert=tpe.to(torch.int32),
        zero_copy=False,
    )
    handle_id = _get_next_handle_id()
    _plan_cache[handle_id.item()] = plan
    return hidden_nvsh, route_weights_nvs, cu_seqlens, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x = inputs[0]
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_plan = _plan_cache.get(handle_id.item())


def _dispatch_backward(ctx, grad_hidden_nvsh, _grad_w, _grad_cu, _grad_hid):
    """Backward for dispatch: combine the gradients (token-major sum)."""
    global _plan_cache, _buffer
    if grad_hidden_nvsh is None:
        return None, None, None, None, None
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"
    plan = ctx.saved_plan
    assert plan is not None

    grad_x, _weights, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=grad_hidden_nvsh,
    )
    grad_x = grad_x.to(ctx.input_dtype)
    # Order matches op inputs: x, topk_scores, topk_ids, tpe, num_experts.
    return grad_x, None, None, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor, handle_id: torch.Tensor, will_backward: bool
) -> torch.Tensor:
    """Execute MoonEP combine (weights applied inside the kernel)."""
    global _plan_cache, _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    if not will_backward:
        plan = _plan_cache.pop(handle_id.item(), None)
    else:
        plan = _plan_cache.get(handle_id.item())
    assert plan is not None, f"Plan not found for handle_id={handle_id.item()}"

    combined, _gathered_weights = buffer.combine(
        plan=plan,
        hidden_nvsh=x,
    )
    return combined


def _combine_setup_context(ctx, inputs, output):
    _, handle_id, _will_backward = inputs
    ctx.saved_plan = _plan_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: re-dispatch the gradient through the saved plan.

    Planning is skipped on this path (the plan's layout is reused) and no
    weight prefetch is needed.
    """
    global _plan_cache, _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    plan = ctx.saved_plan
    assert plan is not None

    grad_nvsh, _, _, _ = buffer.dispatch(
        hidden_sh=grad_combined,
        plan=plan,
    )
    return grad_nvsh, None, None


torch.library.register_autograd(
    "moonep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "moonep::combine", _combine_backward, setup_context=_combine_setup_context
)


# CPU registrations for logic-level tests only (the kernels require CUDA and
# an NVLink domain); they route through whatever object is installed at
# moonep._buffer, so a test double exercises the same code paths.
@torch.library.impl(_lib, "dispatch", "CPU")
def _dispatch_op_impl_cpu(*args, **kwargs):
    return _dispatch_op_impl(*args, **kwargs)


@torch.library.impl(_lib, "combine", "CPU")
def _combine_op_impl_cpu(*args, **kwargs):
    return _combine_op_impl(*args, **kwargs)


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve the cached plan
    cu_seqlens: torch.Tensor  # [2 * epn] padded end offset per VM-group row
    num_recv_tokens: int  # NvS, the padded per-rank slot count


def dispatch_tokens(
    hidden_states: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via MoonEP.

    Tokens come back in expert-major VM-group order for the compute-view
    grouped GEMM (see module docstring). The produced plan is stashed for
    ``MoonEPGroupedExperts`` (prefetch + compute view) and for the combine
    custom op via the handle id.

    Args:
        hidden_states: Input tokens ``[S, H]``, ``S`` the per-rank token
            count (MoonEP requires a fixed S equal to the buffer capacity).
        top_scores: Routing scores ``[S, K]``.
        selected_experts_indices: Expert indices ``[S, K]``.
        num_local_tokens_per_expert_E: Local token counts per expert ``[E]``.
        num_experts: Total number of experts across all ranks.

    Returns:
        (routed_tokens [NvS, H], cu_seqlens [2 * epn], state)
    """
    del num_local_tokens_per_expert_E  # the planner recomputes counts
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    (
        hidden_nvsh,
        route_weights_nvs,
        cu_seqlens,
        handle_id,
    ) = torch.ops.moonep.dispatch(
        hidden_states,
        top_scores.float(),
        selected_experts_indices.to(torch.int32),
        torch.empty(0, device=hidden_states.device, dtype=torch.int32),
        num_experts=num_experts,
    )

    stash_current_plan(_plan_cache[handle_id.item()], route_weights_nvs)
    state = DispatchState(
        handle_id=handle_id,
        cu_seqlens=cu_seqlens,
        num_recv_tokens=hidden_nvsh.shape[0],
    )
    return hidden_nvsh, cu_seqlens, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via MoonEP.

    Routing weights are applied inside the MoonEP combine kernel (they were
    dispatched as ``route_weights_nvs``), so no explicit weighting here --
    unlike the DeepEP wrapper, which applies scores in PyTorch.

    Args:
        hidden_states: Raw (unweighted) expert outputs ``[NvS, H]``.
        state: Dispatch state from ``dispatch_tokens``.

    Returns:
        Combined tokens ``[S, H]``.
    """
    will_backward = torch.is_grad_enabled()
    return torch.ops.moonep.combine(
        hidden_states, state.handle_id, will_backward
    )
