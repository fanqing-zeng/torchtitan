# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent oracle and isolated imports for attention-residual tests."""

import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_attn_res():
    # Avoid loading model registration, vision, and FLA just to test this operator.
    model_path = (
        Path(__file__).resolve().parents[2] / "torchtitan" / "models" / "kimi_k3"
    )
    package_name = "_torchtitan_attn_res_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path)]
    sys.modules[package_name] = package
    name = package_name + ".attn_res"
    spec = importlib.util.spec_from_file_location(name, model_path / "attn_res.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def golden_attention_residual(partial, history, projection, norm, *, eps):
    """FP64 oracle with per-source vector dot products and explicit softmax."""
    values = (
        history if partial is None else torch.cat((history, partial[:, None]), dim=1)
    )
    sources = values.unbind(dim=1)
    normalized = [
        source / (source.square().sum(-1, keepdim=True) / source.shape[-1] + eps).sqrt()
        for source in sources
    ]
    logits = torch.stack(
        [
            torch.einsum("td,d,d->t", source, norm, projection[0])
            for source in normalized
        ],
        dim=1,
    )
    exponentials = (logits - logits.amax(dim=1, keepdim=True)).exp()
    probabilities = exponentials / exponentials.sum(dim=1, keepdim=True)
    return torch.einsum("tn,tnd->td", probabilities, values)


def make_inputs(
    *,
    tokens=5,
    blocks=3,
    dim=33,
    partial=True,
    dtype=torch.float32,
    weight_dtype=None,
    seed=42,
    case="random",
):
    generator = torch.Generator().manual_seed(seed)
    weight_dtype = weight_dtype or dtype
    history = torch.randn(tokens, blocks, dim, generator=generator).to(dtype)
    current = (
        torch.randn(tokens, dim, generator=generator).to(dtype) if partial else None
    )
    projection = (torch.randn(1, dim, generator=generator) / dim**0.5).to(
        weight_dtype
    )
    norm = (torch.randn(dim, generator=generator) * 0.1 + 1).to(weight_dtype)
    if case == "near_zero":
        history = (history * 1e-5).to(dtype)
        current = (current * 1e-5).to(dtype) if current is not None else None
    elif case == "large_logits":
        projection = projection * 64
    elif case == "zero_query":
        projection.zero_()
    elif case == "cancellation":
        for block in range(1, blocks, 2):
            history[:, block] = -history[:, block - 1]
        if current is not None:
            current.mul_(0.01)
        projection.mul_(0.01)
    elif case != "random":
        raise ValueError(case)
    upstream = torch.randn(tokens, dim, generator=generator).to(dtype)
    return (current, history, projection, norm), upstream


def move_inputs(inputs, *, device="cpu", dtype=None, strided=False):
    moved = []
    for tensor in inputs:
        if tensor is None:
            moved.append(None)
            continue
        tensor = tensor.detach().to(device=device, dtype=dtype or tensor.dtype)
        if strided:
            storage = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))
            storage[..., ::2] = tensor
            tensor = storage[..., ::2]
        else:
            tensor = tensor.clone()
        moved.append(tensor.detach().requires_grad_())
    return tuple(moved)


def forward_and_grad(function, inputs, upstream, *, eps):
    output = function(*inputs, eps=eps)
    names = ("partial", "history", "projection", "norm")
    active = [
        (name, tensor) for name, tensor in zip(names, inputs) if tensor is not None
    ]
    gradients = torch.autograd.grad(output, [tensor for _, tensor in active], upstream)
    return {"output": output.detach()} | {
        name: gradient.detach() for (name, _), gradient in zip(active, gradients)
    }


def reduction_terms(name, inputs, upstream, index, *, eps):
    """Return FP64 terms for one output or gradient element's conditioning."""
    current, history, projection, norm = [
        tensor.detach().cpu().double() if tensor is not None else None
        for tensor in inputs
    ]
    upstream = upstream.detach().cpu().double()
    values = history if current is None else torch.cat((history, current[:, None]), 1)
    rstd = (values.square().mean(-1) + eps).rsqrt()
    keys = values * rstd[..., None]
    scores = (keys * projection[0] * norm).sum(-1)
    probabilities = scores.softmax(-1)
    output = (values * probabilities[..., None]).sum(1)
    grad_scores = probabilities * (
        (values * upstream[:, None]).sum(-1) - (output * upstream).sum(-1, keepdim=True)
    )
    if name == "output":
        token, dim = index
        return probabilities[token] * values[token, :, dim]
    if name in ("projection", "norm"):
        dim = index[-1]
        multiplier = norm[dim] if name == "projection" else projection[0, dim]
        return (grad_scores * keys[..., dim] * multiplier).flatten()
    if name == "partial":
        token, dim = index
        source = values.shape[1] - 1
    else:
        token, source, dim = index
    return torch.stack(
        (
            probabilities[token, source] * upstream[token, dim],
            grad_scores[token, source]
            * rstd[token, source]
            * projection[0, dim]
            * norm[dim],
            -grad_scores[token, source]
            * rstd[token, source]
            * keys[token, source, dim]
            * scores[token, source]
            / values.shape[-1],
        )
    )
