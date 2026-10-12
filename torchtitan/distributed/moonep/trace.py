# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Per-rank progress markers for localising a hang inside MoonEP.

A hang in MoonEP's collectives yields no traceback, no device error and no
timeout, so the only way to localise it is to know which rank stopped at which
phase of which expert layer.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.distributed.moonep.runtime import MoonEPRuntime


_TRACE_ENABLED = os.environ.get("MOONEP_TRACE", "0") not in ("0", "", "false", "False")
# Every entry point only enqueues work, so a plain trace reports where the host
# got to, not where the device stalled. This turns each marker into a device
# checkpoint; debug only, it serializes the pipeline.
_TRACE_SYNC = os.environ.get("MOONEP_TRACE_SYNC", "0") not in ("0", "", "false", "False")


def trace(runtime: "MoonEPRuntime", phase: str, detail: str = "") -> None:
    """Emit a per-rank progress marker when ``MOONEP_TRACE`` is set."""
    if not _TRACE_ENABLED:
        return
    if _TRACE_SYNC:
        runtime.device_module.current_stream().synchronize()
    suffix = f" {detail}" if detail else ""
    sys.stderr.write(
        f"[moonep rank={runtime.rank} call={runtime.call_index} t={time.monotonic():.3f}] {phase}{suffix}\n"
    )
    sys.stderr.flush()
