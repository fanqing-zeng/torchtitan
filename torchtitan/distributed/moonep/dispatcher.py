# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Fused-expert forward implementation backed by MoonEP.

The schedule of report S5.2.1: dispatch on the comm stream, queue both
projections' prefetches behind it, run the first GEMM as soon as the gate_up
slots are ready so the down prefetch overlaps it, then the second GEMM, the
router weighting and combine. Static shapes throughout, no host reads.

FSDPTurbo's module-patching entry point (``parallelize_moonep_module``) is not
vendored: the torchtitan integration owns the parameter swap and binds the
forward returned by :func:`get_moonep_experts_forward_fn` itself. That forward
expects the expert module to already expose ``num_global_experts``,
``hidden_dim``, ``gate_up_proj``, ``down_proj``, ``act_fn`` and the
``_moonep_*`` resource holders.
"""

from __future__ import annotations

import torch

from torchtitan.distributed.moonep.autograd import (
    moonep_dispatch,
    moonep_down_and_combine,
    moonep_gate_up_gmm,
    moonep_weights_for_step,
)
from torchtitan.distributed.moonep.plan import verify_routing_range
from torchtitan.distributed.moonep.runtime import MoonEPRuntime
from torchtitan.distributed.moonep.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)


def _consumes_concatenated_gate_up(act_fn) -> bool:
    """True when ``act_fn`` splits ``[gate, up]`` itself instead of seeing the gate half.

    Kimi's ``SituAndMul`` (and the fused SiTU-GLU it wraps) take the concatenated
    first-projection output. The generic SwiGLU path is ``act_fn(gate) * up``.
    """
    return type(act_fn).__name__ == "SituAndMul"


def _activate_expert_hidden(module, gate_up: torch.Tensor) -> torch.Tensor:
    """Apply the expert activation without assuming a SwiGLU-shaped ``act_fn``.

    Prefer the module's own ``_apply_activation`` when present -- that is what
    ``PatchKimiMoeExperts`` uses to choose SiTU vs SiLU. Otherwise detect SiTU
    by the activation class, and only then fall back to the Qwen-style split.
    """
    apply = getattr(module, "_apply_activation", None)
    if callable(apply):
        return apply(gate_up).contiguous()
    if _consumes_concatenated_gate_up(module.act_fn):
        return module.act_fn(gate_up).contiguous()
    gate, up = gate_up.chunk(2, dim=-1)
    act_limit = getattr(module, "limit", None)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    return (module.act_fn(gate) * up).contiguous()


def _fit_tokens_to_static_rank(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a microbatch to MoonEP's reserved ``[S, H]`` / ``[S, K]`` buffers.

    ``tokens_per_rank`` is the capacity reserved at heap init
    (``batch_size * max_seq_length / (cp * tp)``), not a value the caller
    types into the config. A shorter live prefix is padded: hidden and route
    weights are zero, expert ids are zero. Those pad rows do not change the
    live prefix of a grouped GEMM (zero activation, zero route weight). A
    longer microbatch cannot be addressed; raise ``data.max_seq_length``.
    """
    live = hidden_states.shape[0]
    if live == tokens_per_rank:
        return hidden_states, top_k_index, top_k_weights
    if live > tokens_per_rank:
        raise RuntimeError(
            f"MoonEP reserved S={tokens_per_rank} tokens/rank from "
            "data.batch_size * data.max_seq_length (divided by cp * tp), but "
            f"this step has S={live}. Raise data.max_seq_length to cover this "
            "step; do not override tokens_per_rank in the moonep dispatcher config."
        )
    pad = tokens_per_rank - live
    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad))
    top_k_index = torch.nn.functional.pad(top_k_index, (0, 0, 0, pad))
    top_k_weights = torch.nn.functional.pad(top_k_weights, (0, 0, 0, pad))
    return hidden_states, top_k_index, top_k_weights


def _tokens_per_expert(top_k_index: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram the routed expert ids without a device-to-host synchronization.

    ``torch.bincount`` sizes its output from the data and therefore reads a
    device scalar back to the host, which would reintroduce the per-layer stall
    that MoonEP's static shapes exist to remove.
    """
    flat = top_k_index.reshape(-1).to(torch.int64)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=top_k_index.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return counts


def get_moonep_experts_forward_fn(runtime: MoonEPRuntime, fixed_router=False):
    def moonep_experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ):
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        if top_k_index.ndim == 1:
            top_k_index = top_k_index.unsqueeze(-1)
            top_k_weights = top_k_weights.unsqueeze(-1)
        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim).contiguous()
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(f"MoonEP expert input must be BF16, got {hidden_states.dtype}.")

        live_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[-1]
        if top_k_index.shape[0] != live_tokens or top_k_weights.shape != top_k_index.shape:
            raise RuntimeError(
                "MoonEP expects hidden [S,H], expert indices [S,K], and route weights [S,K]; got "
                f"{tuple(hidden_states.shape)}, {tuple(top_k_index.shape)}, {tuple(top_k_weights.shape)}."
            )

        top_k_index = top_k_index.to(dtype=torch.int32).contiguous()
        top_k_weights = top_k_weights.float().contiguous()
        reserved = runtime.configured_tokens_per_rank
        if reserved is not None:
            hidden_states, top_k_index, top_k_weights = _fit_tokens_to_static_rank(
                hidden_states, top_k_index, top_k_weights, reserved
            )
        tokens_per_rank = hidden_states.shape[0]
        tokens_per_expert = _tokens_per_expert(top_k_index, self.num_global_experts)

        call = runtime.new_call(tokens_per_rank, self.hidden_dim, top_k)
        if call.validate:
            verify_routing_range(top_k_index, self.num_global_experts)
        dispatched, route_weights_nvs = moonep_dispatch(
            call, hidden_states, top_k_weights, top_k_index, tokens_per_expert
        )
        # Both prefetches are queued behind the dispatch on the communication
        # stream; only the gate_up copy is waited on before the first GEMM.
        call.start_prefetch(self._moonep_gate_up)
        call.start_prefetch(self._moonep_down)

        # The bridge returns ordering markers, not weights: each expert GEMM
        # consumes one so the weight-gradient reduction runs after both of them.
        gate_up_marker, down_marker = moonep_weights_for_step(
            self.gate_up_proj, self.down_proj, self._moonep_gate_up, self._moonep_down, call
        )

        call.await_prefetch(self._moonep_gate_up)
        fc1 = moonep_gate_up_gmm(call, dispatched, gate_up_marker, hidden_states, self._moonep_gate_up)
        activated = _activate_expert_hidden(self, fc1)

        call.await_prefetch(self._moonep_down)
        output = moonep_down_and_combine(
            call, activated, down_marker, route_weights_nvs, self._moonep_down, self._moonep_gate_up
        )
        return output[:live_tokens].view(*hidden_shape)

    return moonep_experts_forward
