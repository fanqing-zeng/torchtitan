# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""MoonEP device ops. Importing this package registers the NPU implementations
when ``torch_npu`` is installed, so the autograd nodes can dispatch to them."""

from torchtitan.distributed.moonep.ops.moonep_grouped_matmul import (
    moonep_grouped_matmul,
    moonep_grouped_matmul_dgrad,
    moonep_grouped_matmul_wgrad,
)
from torchtitan.distributed.moonep.ops.registry import dispatch_op, register_op

try:
    import torchtitan.distributed.moonep.ops.npu  # noqa: F401
except ImportError:
    pass

__all__ = [
    "register_op",
    "dispatch_op",
    "moonep_grouped_matmul",
    "moonep_grouped_matmul_dgrad",
    "moonep_grouped_matmul_wgrad",
]
