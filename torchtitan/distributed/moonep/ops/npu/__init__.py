# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan: only the MoonEP grouped-GEMM registrations are kept.

from torchtitan.distributed.moonep.ops.npu.moonep_grouped_matmul import (
    moonep_grouped_matmul_dgrad_npu,
    moonep_grouped_matmul_npu,
    moonep_grouped_matmul_wgrad_npu,
)

__all__ = [
    "moonep_grouped_matmul_npu",
    "moonep_grouped_matmul_dgrad_npu",
    "moonep_grouped_matmul_wgrad_npu",
]
