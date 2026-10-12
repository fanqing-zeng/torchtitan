# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Small routing-tensor helpers shared by the MoonEP expert forward."""

from __future__ import annotations

import torch


def normalize_expert_args(top_k_index, top_k_weights):
    """Ensure top_k_index is an integer tensor (indices) and top_k_weights is a
    float tensor (weights). Swap if necessary and adjust dimensions if needed.

    Args:
        top_k_index: Tensor that could be either indices or weights
        top_k_weights: Tensor that could be either weights or indices

    Returns:
        (correct_top_k_index, correct_top_k_weights)
    """
    # Swap if top_k_index is floating point (actually weights)
    if torch.is_floating_point(top_k_index):
        top_k_index, top_k_weights = top_k_weights, top_k_index

    # Ensure weights have the same shape as indices
    if top_k_weights.size() != top_k_index.size():
        # Gather weights using indices (assume top_k_weights has shape [batch_size, num_experts])
        # and top_k_index has shape [batch_size, top_k]
        top_k_weights = top_k_weights.gather(1, top_k_index)

    return top_k_index, top_k_weights


def fixed_router_for_debug(top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_global_experts: int):
    """Debug-only helper that replaces router output with deterministic uniform routing.

    Each token is assigned to experts in round-robin fashion with equal weights.
    Used by numerical-parity runs so every EP degree routes identically.

    Args:
        top_k_index: Original top-k expert indices [num_tokens, top_k]
        top_k_weights: Original top-k weights [num_tokens, top_k]
        num_global_experts: Total number of experts

    Returns:
        (fixed_top_k_index, fixed_top_k_weights)
    """
    is_1d = top_k_index.dim() == 1
    num_tokens = top_k_index.shape[0]
    top_k = 1 if is_1d else top_k_index.shape[1]

    with torch.no_grad():
        base = torch.arange(num_tokens, device=top_k_index.device, dtype=top_k_index.dtype)
        offset = torch.arange(top_k, device=top_k_index.device, dtype=top_k_index.dtype)
        fixed_index = (base.unsqueeze(1) + offset.unsqueeze(0)) % num_global_experts
        if is_1d:
            fixed_index = fixed_index.squeeze(1)
        fixed_weights = torch.full_like(top_k_weights, 1.0 / top_k)

    return fixed_index, fixed_weights
