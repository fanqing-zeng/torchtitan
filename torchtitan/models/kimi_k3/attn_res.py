# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi block attention residuals with explicitly selected training backends."""

import math
from functools import lru_cache

import spmd_types as spmd
import torch
from torch.autograd.function import once_differentiable

# T = packed tokens, N = completed blocks, D = model dimension.


class _AttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, partial_TD, history_TND, projection_1D, norm_D, eps
    ):
        import triton

        from ._attn_res_triton import _attn_res_forward

        num_tokens, num_blocks, dim = history_TND.shape
        has_partial = partial_TD is not None
        partial_pointer = partial_TD if has_partial else history_TND
        partial_strides = partial_TD.stride() if has_partial else (0, 0)
        output_TD = history_TND.new_empty((num_tokens, dim))
        output_fp32_TD = torch.empty_like(output_TD, dtype=torch.float32)
        rstd_TS = torch.empty(
            (num_tokens, num_blocks + has_partial),
            device=history_TND.device,
            dtype=torch.float32,
        )
        scores_TS = torch.empty_like(rstd_TS)
        logsumexp_T = rstd_TS.new_empty((num_tokens,))
        if num_tokens:
            _attn_res_forward[(num_tokens,)](
                history_TND,
                partial_pointer,
                projection_1D,
                norm_D,
                output_TD,
                output_fp32_TD,
                rstd_TS,
                scores_TS,
                logsumexp_T,
                num_blocks,
                dim,
                *history_TND.stride(),
                *partial_strides,
                projection_1D.stride(-1),
                norm_D.stride(-1),
                eps,
                has_partial,
                triton.next_power_of_2(dim),
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(
            partial_TD,
            history_TND,
            projection_1D,
            norm_D,
            output_fp32_TD,
            rstd_TS,
            scores_TS,
            logsumexp_T,
        )
        return output_TD

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TD):  # pyrefly: ignore[bad-override]
        import triton

        from ._attn_res_triton import (
            _attn_res_backward_sources,
            _attn_res_backward_weights,
        )

        (
            partial_TD,
            history_TND,
            projection_1D,
            norm_D,
            output_fp32_TD,
            rstd_TS,
            scores_TS,
            logsumexp_T,
        ) = ctx.saved_tensors
        num_tokens, num_blocks, dim = history_TND.shape
        has_partial = partial_TD is not None
        partial_pointer = partial_TD if has_partial else history_TND
        partial_strides = partial_TD.stride() if has_partial else (0, 0)
        grad_history_TND = torch.empty(
            history_TND.shape, device=history_TND.device, dtype=history_TND.dtype
        )
        grad_partial_TD = (
            history_TND.new_empty((num_tokens, dim)) if has_partial else None
        )
        grad_weight_TD = torch.empty_like(output_fp32_TD)
        grad_projection_1D = torch.empty(
            projection_1D.shape, device=projection_1D.device, dtype=projection_1D.dtype
        )
        grad_norm_D = torch.empty(
            norm_D.shape, device=norm_D.device, dtype=norm_D.dtype
        )
        if num_tokens:
            _attn_res_backward_sources[(num_tokens,)](
                history_TND,
                partial_pointer,
                projection_1D,
                norm_D,
                output_fp32_TD,
                rstd_TS,
                scores_TS,
                logsumexp_T,
                grad_output_TD,
                grad_history_TND,
                grad_partial_TD if has_partial else grad_history_TND,
                grad_weight_TD,
                num_blocks,
                dim,
                *history_TND.stride(),
                *partial_strides,
                projection_1D.stride(-1),
                norm_D.stride(-1),
                *grad_output_TD.stride(),
                has_partial,
                triton.next_power_of_2(dim),
                enable_fp_fusion=False,
            )
        _attn_res_backward_weights[(triton.cdiv(dim, 32),)](
            projection_1D,
            norm_D,
            grad_weight_TD,
            grad_projection_1D,
            grad_norm_D,
            num_tokens,
            dim,
            projection_1D.stride(-1),
            norm_D.stride(-1),
            128,
            32,
            enable_fp_fusion=False,
        )
        return grad_partial_TD, grad_history_TND, grad_projection_1D, grad_norm_D, None


@spmd.register_decomposition(_AttentionResidual)
def _triton_attnres_reference(partial_TD, history_TND, projection_1D, norm_D, eps):
    return eager_attention_residual(
        partial_TD, history_TND, projection_1D, norm_D, eps=eps
    )


@lru_cache(maxsize=1)
def _register_fla_attnres_once() -> None:
    from fla.ops.attnres.fused import FusedAttnresFunction

    spmd.register_decomposition(FusedAttnresFunction, _fla_attnres_reference)


@torch.compiler.assume_constant_result
def _register_fla_attnres() -> None:
    # Registration depends only on the imported class, not on tensor inputs.
    _register_fla_attnres_once()


def _load_fla_attnres():
    _register_fla_attnres()
    from fla.ops.attnres import fused_attnres

    return fused_attnres


def _fla_attnres_reference(
    query_D,
    rms_weight_D,
    output_rms_weight,
    rms_eps,
    scale,
    return_weights,
    checkpoint_level,
    *residuals_TD,
):
    """SPMD decomposition for FLA's two-output FusedAttnresFunction API.

    This adapter uses scale=1, no output RMSNorm, and no returned probabilities.
    FLA must expose the output_rms_weight/return_weights/checkpoint_level forward
    signature; an older incompatible FLA installation needs updating.
    """
    if output_rms_weight is not None or scale != 1.0 or return_weights:
        raise ValueError(
            "Kimi's FLA adapter requires the default attention-residual output"
        )
    history_TND = torch.stack(residuals_TD, dim=1)
    output_TD = eager_attention_residual(
        None, history_TND, query_D.reshape(1, -1), rms_weight_D, eps=rms_eps
    )
    return output_TD, query_D.new_empty(0, dtype=output_TD.dtype)


def eager_attention_residual(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Compute the baseline FP32 normalization, scoring, and weighted sum."""
    values_TND = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
    )
    values_float_TND = values_TND.float()
    variance_TN1 = values_float_TND.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float_TND * torch.rsqrt(variance_TN1 + eps)
    score_weight_D = norm_weight_D.float() * projection_weight_1D.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float_TND).squeeze(1)
    return output_TD.to(values_TND.dtype)


def apply_attention_residual(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    *,
    eps: float,
    backend: str = "eager",
) -> torch.Tensor:
    """Mix completed blocks and an optional current block.

    All backends return the residual dtype and differentiate every residual and
    both weights. ``triton`` uses local strided Triton-Ascend kernels; ``fla``
    requires an FLA installation with an Ascend attention-residual backend.
    Fused backends support first-order training gradients on NPU with FP32
    accumulation. Parameters may use FP32 with BF16 or FP16 residuals.
    """
    if backend not in ("eager", "triton", "fla"):
        raise ValueError(f"Unsupported attention-residual backend: {backend!r}")
    if block_residual_TND.ndim != 3:
        raise ValueError("block_residual_TND must have shape [T, N, D]")
    num_tokens, num_blocks, dim = block_residual_TND.shape
    if dim == 0:
        raise ValueError("The attention-residual model dimension must be positive")
    if num_blocks == 0 and partial_block_TD is None:
        raise ValueError("Attention residuals require at least one residual source")
    if partial_block_TD is not None:
        if partial_block_TD.shape != (num_tokens, dim):
            raise ValueError("partial_block_TD must have shape [T, D]")
        if partial_block_TD.dtype != block_residual_TND.dtype:
            raise ValueError("Partial and completed residuals must have the same dtype")
    if projection_weight_1D.shape != (1, dim):
        raise ValueError("projection_weight_1D must have shape [1, D]")
    if norm_weight_D.shape != (dim,):
        raise ValueError("norm_weight_D must have shape [D]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Attention-residual eps must be finite and positive")
    for tensor in (partial_block_TD, projection_weight_1D, norm_weight_D):
        if tensor is not None and tensor.device != block_residual_TND.device:
            raise ValueError(
                "All attention-residual tensors must be on the same device"
            )
    if backend == "eager":
        return eager_attention_residual(
            partial_block_TD,
            block_residual_TND,
            projection_weight_1D,
            norm_weight_D,
            eps=eps,
        )
    if block_residual_TND.device.type != "npu":
        raise ValueError(
            f"The {backend!r} attention-residual backend requires NPU tensors"
        )
    for tensor in (
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
    ):
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Fused attention residuals support FP16, BF16, and FP32")
    if backend == "triton":
        try:
            return _AttentionResidual.apply(
                partial_block_TD,
                block_residual_TND,
                projection_weight_1D,
                norm_weight_D,
                eps,
            )
        except ImportError as exc:
            raise ImportError(
                "The triton attention-residual backend requires Triton-Ascend"
            ) from exc
    try:
        fused_attnres = _load_fla_attnres()
    except ImportError as exc:
        raise ImportError(
            "The fla attention-residual backend requires FLA with Ascend attnres support"
        ) from exc
    residuals = [
        block_residual_TND[:, index, :].contiguous() for index in range(num_blocks)
    ]
    if partial_block_TD is not None:
        residuals.append(partial_block_TD.contiguous())
    return fused_attnres(
        query=projection_weight_1D.squeeze(0).contiguous(),
        residuals=residuals,
        rms_weight=norm_weight_D.contiguous(),
        rms_eps=eps,
        scale=1.0,
    )
