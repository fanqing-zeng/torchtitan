# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent short-convolution reference shared by CPU and NPU tests."""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def load_adapter():
    path = (
        Path(__file__).resolve().parents[2]
        / "torchtitan/models/common/attention/npu_causal_conv1d.py"
    )
    spec = importlib.util.spec_from_file_location("npu_conv_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_convolution(x, weight, boundaries=None, history=None, *, activate=True):
    """One regular grouped PyTorch convolution per independent document."""
    width = weight.shape[1]
    sequences = (
        list(x.unbind(0))
        if boundaries is None
        else [x[0, start:end] for start, end in zip(boundaries, boundaries[1:])]
    )
    outputs = []
    for index, sequence in enumerate(sequences):
        if sequence.shape[0] == 0:
            outputs.append(sequence)
            continue
        padded = (
            F.pad(sequence.T.unsqueeze(0), (width - 1, 0))
            if history is None
            else torch.cat((history[index, :, -(width - 1) :], sequence.T), dim=1)[None]
        )
        output = F.conv1d(padded, weight[:, None], groups=weight.shape[0])[0].T
        outputs.append(F.silu(output) if activate else output)
    return torch.stack(outputs) if boundaries is None else torch.cat(outputs)[None]


def case(*, batch=1, tokens=11, channels=12, width=4, boundaries=None, history_width=0):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(batch, tokens, channels, generator=generator, dtype=torch.float64)
    weight = (
        torch.randn(channels, width, generator=generator, dtype=torch.float64) / width
    )
    history = None
    if history_width:
        sequences = batch if boundaries is None else len(boundaries) - 1
        history = torch.randn(
            sequences, channels, history_width, generator=generator, dtype=torch.float64
        )
    return tuple(
        value.requires_grad_() if value is not None else None
        for value in (x, weight, history)
    )
