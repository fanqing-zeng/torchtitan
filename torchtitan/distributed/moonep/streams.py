# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Device stream and event helpers, kept inside one API family."""

from __future__ import annotations

import torch


def device_stream_module(device_type: str):
    """Return the concrete device module (``torch.cuda`` or ``torch.npu``).

    Not ``torch.accelerator``: its generic ``torch.Stream`` is cast unchecked by
    the CUDA/NPU event bindings, which segfaults instead of raising.
    """
    return torch.get_device_module(device_type)


def _record_stream(tensor: torch.Tensor | None, stream) -> None:
    """Tie a tensor produced on the comm stream to the consuming stream."""
    if tensor is not None and stream is not None:
        tensor.record_stream(stream)
