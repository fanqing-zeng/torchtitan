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
    """
    from torchtitan.config.transform.moonep import MoonEPTransform
    from torchtitan.models.common.moonep.runtime import MoonEPTrainingRuntime

    config = kimi_k3_debugmodel()
    config.activation_checkpoint = None
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    # MoonEP's post-init weight swap replaces the w13/w2 parameters that
    # DistMuon's expert compute layouts target, so smoke-test with AdamW.
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    config.moonep = MoonEPTrainingRuntime.Config()
    return apply_transforms(config, [MoonEPTransform()])
