# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: composite CPU/NPU kernels for the fused MLA and SwiGLU custom ops.

The fused overrides (``torchtitan/overrides/fused_mla.py``,
``torchtitan/overrides/fused_swiglu.py``) register their custom ops with
``device_types="cuda"`` -- only the triton kernels, which run on CUDA. Two
dispatch layers need kernels on Ascend NPU:

* the spmd_types type checker runs functions on CPU-representative tensors,
  so the ops also need a CPU kernel even before execution;
* real execution happens on NPU (PrivateUse1), where no kernel exists.

This module registers composite PyTorch implementations for both devices.
They reproduce the triton kernels' math exactly (fp32 intermediates, the
bf16/fp16 head-sum rounding boundary in the MLA KV backward, offset-limited
rows in SwiGLU); only floating-point summation order can differ. On CUDA the
original triton kernels stay active; this module is only imported through
``override.imports`` on NPU runs.

    torchtitan_train ... --override.imports torchtitan.overrides.npu_fused_ops.npu_fused_ops
"""

import torch

from torchtitan.config import override
from torchtitan.distributed.activation_checkpoint import FullAC


def _on_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def _active_rows(num_rows: int, offsets: torch.Tensor | None) -> int:
    """Row limit semantics of the SwiGLu triton kernels."""
    if offsets is None:
        return num_rows
    if offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    return int(offsets[-1].item())


def _silu_and_mul_composite(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)
    limit = _active_rows(gate.shape[0], offsets)
    if limit > 0:
        gate_f = gate[:limit].float()
        up_f = up[:limit].float()
        silu = gate_f * torch.sigmoid(gate_f)
        out[:limit] = (silu * up_f).to(gate.dtype)
    return out


def _silu_and_mul_backward_composite(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
    grad_up = torch.empty_like(up, memory_format=torch.contiguous_format)
    limit = _active_rows(gate.shape[0], offsets)
    if limit > 0:
        grad_f = grad_out[:limit].float()
        gate_f = gate[:limit].float()
        up_f = up[:limit].float()
        sigmoid = torch.sigmoid(gate_f)
        silu = gate_f * sigmoid
        silu_grad = sigmoid * (1.0 + gate_f * (1.0 - sigmoid))
        grad_gate[:limit] = (grad_f * up_f * silu_grad).to(gate.dtype)
        grad_up[:limit] = (grad_f * silu).to(up.dtype)
    return grad_gate, grad_up


def _cos_sin(
    rope_cache_real: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-token cos/sin rows from the (max_pos, pairs, 2) real cache."""
    cos = rope_cache_real[..., 0][positions]
    sin = rope_cache_real[..., 1][positions]
    return cos, sin


def _fused_mla_q_rope_composite(
    q: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    inverse: bool,
) -> torch.Tensor:
    # In-place rotate-half on q's rope slice, fp32 intermediates.
    cos, sin = _cos_sin(rope_cache_real, positions)
    cos = cos.unsqueeze(2)  # (B, L, 1, pairs) broadcast over heads
    sin = sin.unsqueeze(2)
    q_even = q[..., q_nope_dim::2].float()
    q_odd = q[..., q_nope_dim + 1 :: 2].float()
    if inverse:
        out_even = q_even * cos + q_odd * sin
        out_odd = q_odd * cos - q_even * sin
    else:
        out_even = q_even * cos - q_odd * sin
        out_odd = q_even * sin + q_odd * cos
    q[..., q_nope_dim::2] = out_even.to(q.dtype)
    q[..., q_nope_dim + 1 :: 2] = out_odd.to(q.dtype)
    return q


def _fused_mla_k_rope_composite(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
) -> torch.Tensor:
    batch, seq_len, n_heads, _ = kv.shape
    rope_dim = k_pe.shape[-1]
    k = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + rope_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    k[..., :q_nope_dim] = kv[..., :q_nope_dim]
    cos, sin = _cos_sin(rope_cache_real, positions)
    kpe_even = k_pe[..., 0::2].float()
    kpe_odd = k_pe[..., 1::2].float()
    # k_pe is shared across heads; rotate once, broadcast when storing.
    out_even = (kpe_even * cos - kpe_odd * sin).to(kv.dtype).unsqueeze(2)
    out_odd = (kpe_even * sin + kpe_odd * cos).to(kv.dtype).unsqueeze(2)
    k[..., q_nope_dim::2] = out_even
    k[..., q_nope_dim + 1 :: 2] = out_odd
    return k


def _fused_mla_kv_backward_composite(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, n_heads, _ = grad_k.shape
    v_dim = grad_v.shape[-1]
    grad_kv = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + v_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    grad_kv[..., :q_nope_dim] = grad_k[..., :q_nope_dim]
    grad_kv[..., q_nope_dim:] = grad_v

    # Head reduction of the rope gradient in fp32, then reproduce the stock
    # expand-backward rounding boundary before the inverse rotation.
    grad_pos_even = grad_k[..., q_nope_dim::2].float().sum(dim=2)
    grad_pos_odd = grad_k[..., q_nope_dim + 1 :: 2].float().sum(dim=2)
    if grad_k.dtype == torch.bfloat16:
        grad_pos_even = grad_pos_even.to(torch.bfloat16).float()
        grad_pos_odd = grad_pos_odd.to(torch.bfloat16).float()
    elif grad_k.dtype == torch.float16:
        grad_pos_even = grad_pos_even.to(torch.float16).float()
        grad_pos_odd = grad_pos_odd.to(torch.float16).float()

    cos, sin = _cos_sin(rope_cache_real, positions)
    out_even = grad_pos_even * cos + grad_pos_odd * sin
    out_odd = grad_pos_odd * cos - grad_pos_even * sin
    grad_k_pe = torch.empty(
        (*grad_k.shape[:2], rope_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    grad_k_pe[..., 0::2] = out_even.to(grad_k.dtype)
    grad_k_pe[..., 1::2] = out_odd.to(grad_k.dtype)
    return grad_kv, grad_k_pe


def _register_kernels() -> None:
    from torchtitan.overrides import fused_mla, fused_swiglu

    for device in ("cpu", "npu"):
        fused_swiglu.silu_and_mul_op.register_kernel(device, _silu_and_mul_composite)
        fused_swiglu.silu_and_mul_backward_op.register_kernel(
            device, _silu_and_mul_backward_composite
        )
        fused_mla._fused_mla_q_rope_op.register_kernel(
            device, _fused_mla_q_rope_composite
        )
        fused_mla._fused_mla_k_rope_op.register_kernel(
            device, _fused_mla_k_rope_composite
        )
        fused_mla._fused_mla_kv_backward_op.register_kernel(
            device, _fused_mla_kv_backward_composite
        )


if _on_npu():
    _register_kernels()


@override(
    target=FullAC.Config,
    exact=True,
    description="Composite CPU/NPU kernels for the fused MLA and SwiGLU "
    "custom ops (registration happens at module import).",
)
def npu_fused_ops(cfg: FullAC.Config) -> FullAC.Config:
    return cfg
