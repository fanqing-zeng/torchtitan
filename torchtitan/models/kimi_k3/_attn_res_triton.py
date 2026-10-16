# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Strided Triton-Ascend attention residual kernels, imported only on demand."""

import triton
import triton.language as tl

# T = packed tokens, N = completed blocks, D = model dimension.
# S = all sources including the optional partial block.


@triton.jit
def _attn_res_forward(
    HISTORY,
    PARTIAL,
    PROJECTION,
    NORM,
    OUTPUT,
    OUTPUT_FP32,
    RSTD,
    SCORES,
    LOGSUMEXP,
    N: tl.constexpr,
    D: tl.constexpr,
    HS_T: tl.constexpr,
    HS_N: tl.constexpr,
    HS_D: tl.constexpr,
    PS_T: tl.constexpr,
    PS_D: tl.constexpr,
    QS_D: tl.constexpr,
    WS_D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets_D = tl.arange(0, BD)
    mask_D = offsets_D < D
    weight_D = tl.load(PROJECTION + offsets_D * QS_D, mask_D, 0).to(tl.float32)
    weight_D *= tl.load(NORM + offsets_D * WS_D, mask_D, 0).to(tl.float32)
    num_sources: tl.constexpr = N + HAS_PARTIAL
    max_score = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    numerator_D = tl.full((BD,), 0, tl.float32)
    # Online softmax keeps only one source in registers and avoids a T*N*D copy.
    for source in range(num_sources):
        if source < N:
            values_D = tl.load(
                HISTORY + token * HS_T + source * HS_N + offsets_D * HS_D,
                mask_D,
                0,
            ).to(tl.float32)
        else:
            values_D = tl.load(PARTIAL + token * PS_T + offsets_D * PS_D, mask_D, 0).to(
                tl.float32
            )
        rstd = tl.rsqrt(tl.sum(values_D * values_D, 0) / D + EPS)
        score = tl.sum((values_D * rstd) * weight_D, 0)
        new_max = tl.maximum(max_score, score)
        rescale = tl.exp(max_score - new_max)
        probability = tl.exp(score - new_max)
        numerator_D = numerator_D * rescale + probability * values_D
        denominator = denominator * rescale + probability
        max_score = new_max
        tl.store(RSTD + token * num_sources + source, rstd)
        tl.store(SCORES + token * num_sources + source, score)
    output_D = numerator_D / denominator
    tl.store(OUTPUT + token * D + offsets_D, output_D, mask_D)
    # Backward's softmax delta must use the unrounded mixture even for BF16.
    tl.store(OUTPUT_FP32 + token * D + offsets_D, output_D, mask_D)
    tl.store(LOGSUMEXP + token, max_score + tl.log(denominator))


@triton.jit
def _attn_res_backward_sources(
    HISTORY,
    PARTIAL,
    PROJECTION,
    NORM,
    OUTPUT_FP32,
    RSTD,
    SCORES,
    LOGSUMEXP,
    GRAD_OUTPUT,
    GRAD_HISTORY,
    GRAD_PARTIAL,
    GRAD_WEIGHT,
    N: tl.constexpr,
    D: tl.constexpr,
    HS_T: tl.constexpr,
    HS_N: tl.constexpr,
    HS_D: tl.constexpr,
    PS_T: tl.constexpr,
    PS_D: tl.constexpr,
    QS_D: tl.constexpr,
    WS_D: tl.constexpr,
    GS_T: tl.constexpr,
    GS_D: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets_D = tl.arange(0, BD)
    mask_D = offsets_D < D
    num_sources: tl.constexpr = N + HAS_PARTIAL
    weight_D = tl.load(PROJECTION + offsets_D * QS_D, mask_D, 0).to(tl.float32)
    weight_D *= tl.load(NORM + offsets_D * WS_D, mask_D, 0).to(tl.float32)
    grad_output_D = tl.load(
        GRAD_OUTPUT + token * GS_T + offsets_D * GS_D, mask_D, 0
    ).to(tl.float32)
    output_D = tl.load(OUTPUT_FP32 + token * D + offsets_D, mask_D, 0)
    delta = tl.sum(grad_output_D * output_D, 0)
    logsumexp = tl.load(LOGSUMEXP + token)
    grad_weight_D = tl.full((BD,), 0, tl.float32)
    for source in range(num_sources):
        if source < N:
            values_D = tl.load(
                HISTORY + token * HS_T + source * HS_N + offsets_D * HS_D,
                mask_D,
                0,
            ).to(tl.float32)
        else:
            values_D = tl.load(PARTIAL + token * PS_T + offsets_D * PS_D, mask_D, 0).to(
                tl.float32
            )
        rstd = tl.load(RSTD + token * num_sources + source)
        score = tl.load(SCORES + token * num_sources + source)
        probability = tl.exp(score - logsumexp)
        grad_score = probability * (tl.sum(grad_output_D * values_D, 0) - delta)
        key_D = values_D * rstd
        grad_values_D = probability * grad_output_D + (grad_score * rstd) * (
            weight_D - key_D * (score / D)
        )
        if source < N:
            tl.store(
                GRAD_HISTORY + (token * N + source) * D + offsets_D,
                grad_values_D,
                mask_D,
            )
        else:
            tl.store(GRAD_PARTIAL + token * D + offsets_D, grad_values_D, mask_D)
        grad_weight_D += grad_score * key_D
    tl.store(GRAD_WEIGHT + token * D + offsets_D, grad_weight_D, mask_D)


@triton.jit
def _attn_res_backward_weights(
    PROJECTION,
    NORM,
    GRAD_WEIGHT,
    GRAD_PROJECTION,
    GRAD_NORM,
    T: tl.constexpr,
    D: tl.constexpr,
    QS_D: tl.constexpr,
    WS_D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    offsets_D = tl.program_id(0) * BD + tl.arange(0, BD)
    mask_D = offsets_D < D
    accumulated_D = tl.full((BD,), 0, tl.float32)
    # A fixed reduction order avoids atomics and nondeterministic weight gradients.
    for start in range(tl.cdiv(T, BT)):
        offsets_T = start * BT + tl.arange(0, BT)
        grad_TD = tl.load(
            GRAD_WEIGHT + offsets_T[:, None].to(tl.int64) * D + offsets_D[None, :],
            (offsets_T[:, None] < T) & (offsets_D[None, :] < D),
            0,
        )
        accumulated_D += tl.sum(grad_TD, 0)
    projection_D = tl.load(PROJECTION + offsets_D * QS_D, mask_D, 0).to(tl.float32)
    norm_D = tl.load(NORM + offsets_D * WS_D, mask_D, 0).to(tl.float32)
    tl.store(GRAD_PROJECTION + offsets_D, accumulated_D * norm_D, mask_D)
    tl.store(GRAD_NORM + offsets_D, accumulated_D * projection_D, mask_D)
