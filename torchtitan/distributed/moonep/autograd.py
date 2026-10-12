# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""The autograd nodes that place MoonEP's collectives in the graph.

Each node exists to pin *when* a collective runs relative to the expert GEMMs.
The ordering is part of the distributed protocol rather than an optimisation:
MoonEP's barriers share one slot and generation counter, so every rank has to
arrive at them in the same sequence or the job deadlocks.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.moonep.ops import (
    moonep_grouped_matmul,
    moonep_grouped_matmul_dgrad,
    moonep_grouped_matmul_wgrad,
)
from torchtitan.distributed.moonep.runtime import MoonEPCallState
from torchtitan.distributed.moonep.trace import trace


class _MoonEPDispatch(torch.autograd.Function):
    """Scatter tokens to their expert-grouped slots on the owning ranks.

    ``combine`` is the exact adjoint of ``dispatch`` (a permute-with-duplication
    versus a sum-reduction), so the backward is a ``combine`` call.
    """

    @staticmethod
    def forward(ctx, hidden, route_weights, topk_experts, tokens_per_expert, state):
        config = state.runtime.config
        async_finish = config.async_finish and state.runtime.backend.supports_async_comm
        trace(state.runtime, "dispatch.forward enter")
        with torch.profiler.record_function("moonep.dispatch.forward"):
            result = state.buffer.dispatch(
                hidden,
                route_weights,
                topk_experts,
                tokens_per_expert,
                async_finish=async_finish,
                zero_copy=False,
            )
        if async_finish:
            hidden_nvs, route_nvs, cu_seqlens, plan, event = result
            # The group offsets decide the GEMMs' memory extents, so a stale read
            # is out of bounds. No overlap lost: the compute stream is idle here.
            state.wait(event)
        else:
            hidden_nvs, route_nvs, cu_seqlens, plan = result
        state.adopt(hidden_nvs, route_nvs)
        state.set_plan(plan, cu_seqlens, int(hidden_nvs.shape[0]))
        state.neutralize_static_tail(hidden_nvs)
        ctx.state = state
        trace(state.runtime, "dispatch.forward exit")
        return hidden_nvs, route_nvs

    @staticmethod
    def backward(ctx, grad_hidden_nvs, grad_route_nvs):
        state = ctx.state
        config = state.runtime.config
        async_finish = config.async_finish and state.runtime.backend.supports_async_comm
        trace(state.runtime, "dispatch.backward enter")
        with torch.profiler.record_function("moonep.dispatch.backward"):
            grad_hidden, grad_route, event = state.buffer.combine(
                plan=state.plan,
                hidden_nvsh=grad_hidden_nvs.contiguous(),
                route_weights_nvs=None if grad_route_nvs is None else grad_route_nvs.float().contiguous(),
                async_finish=async_finish,
                zero_copy=False,
            )
        state.wait(event)
        state.adopt(grad_hidden, grad_route)
        trace(state.runtime, "dispatch.backward exit")
        return grad_hidden, grad_route, None, None, None


class _MoonEPWeightsBridge(torch.autograd.Function):
    """Connect the registered parameters to their symmetric GEMM views.

    One multi-output node so both reductions are launched in the same order on
    every rank; independent nodes could be scheduled differently and deadlock on
    MoonEP's internal barriers.

    The expert weights are deliberately *not* autograd inputs: a weight gradient
    is indexed by the ``epn+B`` groups this rank computed while the GEMM operand
    may be the global ``[E+B]`` mapping, so the shapes are unrelated. The expert
    GEMMs read the weights off the projection and stage their gradients; this node
    only carries the ordering edge and performs the reduction.
    """

    @staticmethod
    def forward(ctx, local_gate_up, local_down, gate_up_projection, down_projection, state):
        ctx.gate_up_projection = gate_up_projection
        ctx.down_projection = down_projection
        ctx.state = state
        # Ordering markers, not data.
        marker = torch.zeros((), device=local_gate_up.device, dtype=local_gate_up.dtype)
        return marker.clone(), marker.clone()

    @staticmethod
    def backward(ctx, grad_gate_up_marker, grad_down_marker):
        state = ctx.state
        trace(state.runtime, "grad_reduce enter")
        with torch.profiler.record_function("moonep.grad_reduce"):
            # down-then-gate_up is fixed across ranks: one shared barrier slot.
            local_down = ctx.down_projection.finish_gradient(state.buffer, state.plan)
            local_gate_up = ctx.gate_up_projection.finish_gradient(state.buffer, state.plan)
        trace(state.runtime, "grad_reduce exit")
        return local_gate_up, local_down, None, None, None


class _MoonEPGateUpGmm(torch.autograd.Function):
    """First expert GEMM on MoonEP's compact group set.

    Optionally recomputes the dispatched activation in the backward pass instead
    of keeping it alive, trading one extra all-to-all for ``K`` times less saved
    activation memory.
    """

    @staticmethod
    def forward(ctx, dispatched, marker, hidden, state, projection):
        with torch.profiler.record_function("moonep.gmm.gate_up"):
            output = moonep_grouped_matmul(dispatched, state.group_ends, projection.gemm_weight, projection.transposed)
        ctx.state = state
        ctx.projection = projection
        ctx.recompute_dispatch = state.runtime.config.recompute_dispatch
        ctx.save_for_backward(hidden if ctx.recompute_dispatch else dispatched)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.state
        projection = ctx.projection
        (saved,) = ctx.saved_tensors
        # Slots are pooled across layers; replay this layer's weights first.
        trace(state.runtime, "gmm.gate_up.backward enter")
        state.await_prefetch(projection)
        grad_output = grad_output.contiguous()
        with torch.profiler.record_function("moonep.gmm.gate_up.backward"):
            grad_dispatched = moonep_grouped_matmul_dgrad(
                grad_output, state.group_ends, projection.gemm_weight, projection.transposed
            )
            dispatched = _recover_dispatched(state, saved) if ctx.recompute_dispatch else saved
            trace(state.runtime, "gmm.gate_up.backward dgrad done")
            projection.stage_gradient(
                moonep_grouped_matmul_wgrad(
                    grad_output,
                    dispatched,
                    state.wgrad_group_ends,
                    state.wgrad_group_ends_host(),
                    projection.transposed,
                )
            )
        trace(state.runtime, "gmm.gate_up.backward exit")
        return grad_dispatched, None, None, None, None


def _recover_dispatched(state: MoonEPCallState, hidden: torch.Tensor) -> torch.Tensor:
    """Re-run dispatch with the saved plan to rebuild the first GEMM's input."""
    with torch.profiler.record_function("moonep.dispatch.recompute"):
        dispatched = state.buffer.dispatch(hidden, plan=state.plan, async_finish=False, zero_copy=False)[0]
    state.adopt(dispatched)
    return state.neutralize_static_tail(dispatched)


class _MoonEPDownAndCombine(torch.autograd.Function):
    """Second expert GEMM, route weighting, and combine as one autograd node.

    Fusing them keeps the expert output ``[NvS, H]`` out of the saved-tensor set.
    ``dL/dp = sum_h dL/dweighted * expert_out`` needs that output; substituting
    ``expert_out = act @ W.T`` and reordering the sums gives

        dL/dp[n] = sum_j act[n,j] * u[n,j],   u = dL/dweighted @ W

    where ``u`` is the down-GEMM data gradient the backward computes anyway.
    """

    @staticmethod
    def forward(ctx, activated, marker, route_weights_nvs, state, projection, peer_projection):
        config = state.runtime.config
        async_finish = config.async_finish and state.runtime.backend.supports_async_comm
        with torch.profiler.record_function("moonep.gmm.down"):
            expert_output = moonep_grouped_matmul(
                activated, state.group_ends, projection.gemm_weight, projection.transposed
            )
        expert_output.mul_(route_weights_nvs.to(expert_output.dtype).unsqueeze(-1))
        trace(state.runtime, "gmm.down.forward done")
        with torch.profiler.record_function("moonep.combine.forward"):
            output, _, event = state.buffer.combine(
                plan=state.plan,
                hidden_nvsh=expert_output,
                async_finish=async_finish,
                zero_copy=False,
            )
        state.wait(event)
        state.adopt(output)
        trace(state.runtime, "combine.forward exit")
        ctx.state = state
        ctx.projection = projection
        ctx.peer_projection = peer_projection
        ctx.save_for_backward(activated, route_weights_nvs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.state
        config = state.runtime.config
        async_finish = config.async_finish and state.runtime.backend.supports_async_comm
        activated, route_weights_nvs = ctx.saved_tensors
        weight = ctx.projection.gemm_weight

        # Queued early so the gate_up copy overlaps this GEMM.
        trace(state.runtime, "combine.backward enter")
        state.start_prefetch(ctx.projection)
        state.start_prefetch(ctx.peer_projection)
        state.await_prefetch(ctx.projection)

        with torch.profiler.record_function("moonep.combine.backward"):
            result = state.buffer.dispatch(
                grad_output.contiguous(), plan=state.plan, async_finish=async_finish, zero_copy=False
            )
        grad_weighted = result[0]
        if async_finish:
            state.wait(result[4])
        state.adopt(grad_weighted)
        state.neutralize_static_tail(grad_weighted)
        trace(state.runtime, "combine.backward dispatch done")

        with torch.profiler.record_function("moonep.gmm.down.backward"):
            grad_activated = moonep_grouped_matmul_dgrad(
                grad_weighted, state.group_ends, weight, ctx.projection.transposed
            )
            # FP32 to match the router-weight buffer MoonEP scatters and gathers.
            grad_route = torch.sum(activated * grad_activated, dim=-1, dtype=torch.float32)
            route_column = route_weights_nvs.to(grad_activated.dtype).unsqueeze(-1)
            # In place: both operands are freshly produced, so no extra buffer.
            grad_activated.mul_(route_column)
            grad_weighted.mul_(route_column)
            trace(state.runtime, "combine.backward dgrad done")
            ctx.projection.stage_gradient(
                moonep_grouped_matmul_wgrad(
                    grad_weighted,
                    activated,
                    state.wgrad_group_ends,
                    state.wgrad_group_ends_host(),
                    ctx.projection.transposed,
                )
            )
        trace(state.runtime, "combine.backward exit")
        return grad_activated, None, grad_route, None, None, None


def moonep_dispatch(state, hidden, route_weights, topk_experts, tokens_per_expert):
    return _MoonEPDispatch.apply(hidden, route_weights, topk_experts, tokens_per_expert, state)


def moonep_weights_for_step(gate_up_parameter, down_parameter, gate_up_projection, down_projection, state):
    return _MoonEPWeightsBridge.apply(
        _local(gate_up_parameter), _local(down_parameter), gate_up_projection, down_projection, state
    )


def moonep_gate_up_gmm(state, dispatched, marker, hidden, projection):
    # Only forwarded when the dispatch is recomputed; else it must not be held.
    if not state.runtime.config.recompute_dispatch:
        hidden = None
    return _MoonEPGateUpGmm.apply(dispatched, marker, hidden, state, projection)


def moonep_down_and_combine(state, activated, marker, route_weights_nvs, projection, peer_projection):
    return _MoonEPDownAndCombine.apply(activated, marker, route_weights_nvs, state, projection, peer_projection)


def _local(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter
