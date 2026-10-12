# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""MoonEP: perfectly balanced expert parallelism over symmetric memory.

Selected through ``moe.moe_comm_backend = "moonep"`` in the job config.
``dispatcher`` carries the fused expert forward, ``runtime`` owns the symmetric
weights and per-step state, ``autograd`` places the collectives in the graph,
``plan`` holds the static-shape layout, ``backend`` / ``backend_npu`` hide all
device-specific work, and ``settings`` / ``streams`` / ``trace`` are support.
Only the two names below are imported from outside this package.
"""

from torchtitan.distributed.moonep.dispatcher import get_moonep_experts_forward_fn
from torchtitan.distributed.moonep.runtime import MoonEPRuntime

__all__ = ["MoonEPRuntime", "get_moonep_experts_forward_fn"]
