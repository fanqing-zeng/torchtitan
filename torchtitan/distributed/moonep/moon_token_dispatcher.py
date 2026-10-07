# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP token dispatcher for MoE Expert Parallel.

Composes with ``RoutedExperts`` exactly like ``DeepEPTokenDispatcher``:
``dispatch`` moves tokens (in the expert-input space) to expert-major
VM-group layout across the EP group; ``RoutedExperts`` then runs the
grouped-expert compute and hands the outputs back to ``combine``.

MoonEP-specific pieces live in ``torchtitan/distributed/moonep/moonep.py``:
the process-global ``Buffer`` + weight pools, the plan handoff to
``MoonEPGroupedExperts``, and the ``torch.library`` custom ops that give
dispatch/combine full autograd (dispatch backward = combine, combine
backward = re-dispatch through the saved plan).
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.distributed.moonep import moonep
from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    EPDispatchMetadata,
)


class MoonEPTokenDispatcher(BaseEPTokenDispatcher):
    """Token dispatcher using MoonEP's balanced redundant-expert dispatch.

    MoonEP plans a small set of redundant experts online from the current
    routing, prefetches their weights into process-global pools, and moves
    every token to exactly one (possibly remote) expert slot, so each rank
    computes a static ``S * K`` rows with no host sync.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        # Per-rank input token capacity S used to size the MoonEP buffer.
        # Runtime configuration must fill it before dispatcher construction
        # (``update_ep_token_dispatcher_config``).
        num_max_tokens_per_rank: int | None = None
        # Expert FFN intermediate dim, needed to size the weight pools.
        expert_hidden_dim: int | None = None
        # Expert input dim, threaded by the builder for buffer/pool
        # initialization. For latent MoE models this is the latent dim
        # (inner_experts.dim), filled by the runtime configuration.
        hidden_dim: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        if config.num_max_tokens_per_rank is None:
            raise ValueError(
                "MoonEP requires num_max_tokens_per_rank for buffer initialization."
            )
        if config.num_max_tokens_per_rank <= 0:
            raise ValueError(
                "MoonEP num_max_tokens_per_rank must be positive, got "
                f"{config.num_max_tokens_per_rank}."
            )
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.expert_hidden_dim = config.expert_hidden_dim
        self.hidden_dim = config.hidden_dim

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.moonep import moonep  # noqa: F401

    def init_buffer(self) -> None:
        """Eagerly create the MoonEP buffer and weight pools."""
        assert self.ep_mesh is not None
        assert self.hidden_dim is not None
        assert self.expert_hidden_dim is not None

        moonep.get_buffer(
            self.ep_mesh.get_group(),
            hidden=self.hidden_dim,
            expert_hidden=self.expert_hidden_dim,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            num_topk=self.top_k,
            num_experts=self.num_experts,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, EPDispatchMetadata]:
        """Dispatch tokens through the preallocated MoonEP buffer."""
        # MoonEP's planner recomputes per-expert counts from topk itself;
        # the router-provided counts are redundant.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )

        hidden_nvsh, _cu_seqlens, state = moonep.dispatch_tokens(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E=None,
            num_experts=self.num_experts,
        )
        # num_tokens_per_local_expert is consumed by the grouped GEMM offsets;
        # MoonEP returns padded per-VM-group ends (cu_seqlens) instead. The
        # experts build offsets from cu_seqlens when the MoonEP plan is
        # present, so pass a zero placeholder here.
        num_tokens_per_local_expert_e = torch.zeros(
            self.num_experts // self.ep_mesh.get_group().size(),
            dtype=torch.int32,
            device=x_TD.device,
        )
        metadata = EPDispatchMetadata(state=state)
        return hidden_nvsh, num_tokens_per_local_expert_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: EPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Combine tokens via MoonEP (weights applied inside the kernel)."""
        del x_TD
        combined_TD = moonep.combine_tokens(routed_output_RD, metadata.state)
        return combined_TD
