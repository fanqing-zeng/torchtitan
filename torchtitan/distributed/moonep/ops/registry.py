# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan. Simplified from the upstream registry: MoonEP's ops
# always resolve their device from an operand, so the implicit
# ``torch.accelerator`` probing and the CPU fallback are not carried over.

"""Registry for device-specific MoonEP grouped-GEMM implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_OPERATOR_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register_op(op_name: str, device_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a device implementation for an op; used as a decorator."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _OPERATOR_REGISTRY.setdefault(op_name, {})[device_type] = func
        return func

    return decorator


def dispatch_op(op_name: str, *args: Any, device_type: str, **kwargs: Any) -> Any:
    """Call the ``device_type`` implementation of ``op_name``."""
    impl = _OPERATOR_REGISTRY.get(op_name, {}).get(device_type)
    if impl is None:
        raise KeyError(f"MoonEP op '{op_name}' is not available for device '{device_type}'")
    return impl(*args, **kwargs)
