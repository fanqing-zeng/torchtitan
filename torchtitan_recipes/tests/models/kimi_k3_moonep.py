# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 MoonEP smoke-test recipe (Ascend NPU only)."""

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.transform import apply_transforms
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def kimi_k3_debugmodel_moonep() -> Trainer.Config:
    """Debugmodel Kimi K3 with MoonEP experts, FSDP2 + EP2, on Ascend NPUs.

    Launch on two NPUs from the repo root::

        MODULE=torchtitan_recipes.tests.models.kimi_k3_moonep \\
        CONFIG=kimi_k3_debugmodel_moonep NGPU=2 ./run_train.sh

    The base recipe already trains in bfloat16 with CUDA graphs disabled;
    this variant additionally turns activation checkpointing off and selects
    the FSDP2 + EP2 layout MoonEP v1 requires (``dp_shard * cp * tp == ep``,
    ``pp == dp_replicate == 1``).

    The model uses the ``debugmodel-moonep`` flavor: identical to the official
    debugmodel except ``num_experts=128``, so every MoonEP VMM chunk (bf16
    home weight / prefetch slots and fp32 gradients, with
    ``prefetch_slots == experts_per_rank``) is an exact multiple of the
    Ascend 950DT VMM allocation granularity (2 MiB minimum) at ``ep=2``
    (``E/R=64``). The official debugmodel's 8 experts can never satisfy that
    alignment on 950DT.
    """
    return _moonep_debugmodel_config(flavor="debugmodel-moonep", experts_degree=2)


def kimi_k3_debugmodel_moonep_8card() -> Trainer.Config:
    """Debugmodel Kimi K3 with MoonEP experts, FSDP2 + EP8, on 8 Ascend NPUs.

    Launch on eight NPUs from the repo root::

        MODULE=torchtitan_recipes.tests.models.kimi_k3_moonep \\
        CONFIG=kimi_k3_debugmodel_moonep_8card NGPU=8 ./run_train.sh

    Same recipe as :func:`kimi_k3_debugmodel_moonep` at the ``ep=8`` layout
    MoonEP v1 requires (``dp_shard * cp * tp == ep``, ``pp == dp_replicate
    == 1``). The ``debugmodel-moonep-512`` flavor keeps the per-rank chunk
    alignment on the 950DT VMM: the binding constraint is ``E/R`` a multiple
    of 64, so ``num_experts = 64 x ep`` -- 512 experts at ``ep=8`` yields the
    same per-rank chunk sizes as the ep=2 recipe's 128 experts.
    """
    return _moonep_debugmodel_config(flavor="debugmodel-moonep-512", experts_degree=8)


def _moonep_debugmodel_config(*, flavor: str, experts_degree: int) -> Trainer.Config:
    """Shared MoonEP smoke-test body for the debugmodel recipes.

    Swaps the model to the given MoonEP-aligned flavor, selects the FSDP2 +
    EP layout MoonEP v1 requires, replaces DistMuon with plain AdamW, and
    applies the MoonEP transform.
    """
    from torchtitan.config.transform.moonep import MoonEPTransform
    from torchtitan.models.common.moonep.runtime import MoonEPTrainingRuntime
    from torchtitan.models.kimi_k3 import build_model_config

    config = kimi_k3_debugmodel()
    # num_experts is the only difference; vocab size is unchanged, so the
    # loss/global_vocab_size computed by the base recipe stays consistent.
    # The base recipe's DistMuon (built against 8-expert layouts) is replaced
    # below, so its stale compute shardings are discarded.
    config.model = build_model_config(
        flavor,
        seq_len=config.model.max_context_length,
    )
    config.activation_checkpoint = None
    config.parallelism.data_parallel_shard_degree = experts_degree
    config.parallelism.expert_parallel_degree = experts_degree
    # MoonEP's post-init weight swap replaces the w13/w2 parameters that
    # DistMuon's expert compute layouts target, so smoke-test with AdamW.
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    config.moonep = MoonEPTrainingRuntime.Config()
    return apply_transforms(config, [MoonEPTransform()])
