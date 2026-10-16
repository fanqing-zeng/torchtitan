# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AscendC depthwise causal convolution for KDA's packed Q/K/V projections.

The public fla_npu calling convention follows MindSpeed-MM's
``fsdp/ops/gdn/causal_conv1d_ascendc.py``. Parameters remain in TorchTitan's
checkpoint layout; only operands passed to the kernel are transposed.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

# B = batch, T = tokens, C = channels, H = heads, D = channels per head,
# W = filter width, N = independent sequences.


def _load_kernels():
    from fla_npu.ops.ascendc import causal_conv1d, causal_conv1d_bwd

    return causal_conv1d, causal_conv1d_bwd


class _AscendCCausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, x_BTC, weight_CW, initial_state_NCW, num_heads, boundaries
    ):
        forward_kernel, _ = _load_kernels()
        batch, tokens, channels = x_BTC.shape
        width = weight_CW.shape[1]
        is_varlen = boundaries is not None
        num_sequences = len(boundaries) - 1 if is_varlen else batch
        op_x = x_BTC.reshape(tokens, channels) if is_varlen else x_BTC
        op_weight_WC = weight_CW.transpose(0, 1).contiguous()
        initial_state_NWC = (
            initial_state_NCW.transpose(1, 2).contiguous()
            if initial_state_NCW is not None
            else None
        )
        conv_states = (
            initial_state_NWC[:, -(width - 1) :].contiguous()
            if initial_state_NWC is not None
            else x_BTC.new_zeros(num_sequences, width - 1, channels)
        )
        # The backward API consumes the preactivation when activation=1.
        # Keep it once, rather than recomputing the convolution during backward.
        preactivation = forward_kernel(
            op_x,
            op_weight_WC,
            None,
            conv_states,
            query_start_loc=boundaries,
            initial_state_mode=[1] * num_sequences
            if initial_state_NCW is not None
            else None,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=num_heads,
        )
        output = F.silu(preactivation)
        if is_varlen:
            output = output.transpose(0, 1).reshape(1, tokens, channels)
        else:
            output = output.transpose(1, 2).reshape(batch, tokens, channels)
        ctx.save_for_backward(op_x, op_weight_WC, initial_state_NWC, preactivation)
        ctx.input_shape = x_BTC.shape
        ctx.num_heads = num_heads
        ctx.boundaries = boundaries
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_BTC):  # pyrefly: ignore[bad-override]
        _, backward_kernel = _load_kernels()
        op_x, op_weight_WC, initial_state_NWC, preactivation = ctx.saved_tensors
        batch, tokens, channels = ctx.input_shape
        grad_BTHD = grad_BTC.reshape(
            batch, tokens, ctx.num_heads, channels // ctx.num_heads
        )
        if ctx.boundaries is not None:
            op_grad = grad_BTHD.squeeze(0).transpose(0, 1).contiguous()
        else:
            op_grad = grad_BTHD.transpose(1, 2).contiguous()
        dx, dw_WC, _, dh_NWC = backward_kernel(
            x=op_x,
            y=preactivation,
            weight=op_weight_WC,
            dy=op_grad,
            initial_state=initial_state_NWC,
            dht=None,
            query_start_loc=ctx.boundaries,
            activation=1,
            input_layout="NTD" if ctx.boundaries is not None else "BNSD",
        )
        dh_NCW = (
            dh_NWC.transpose(1, 2).contiguous()
            if initial_state_NWC is not None
            else None
        )
        return dx.reshape(ctx.input_shape), dw_WC.transpose(0, 1), dh_NCW, None, None


def ascendc_causal_conv1d(
    x_BTC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    num_heads: int,
    cu_seqlens: torch.Tensor | list[int] | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply depthwise convolution and SiLU, preserving packed boundaries.

    Input has shape ``[B, T, C]`` and weight ``[C, W]``. Packed input requires
    ``B=1``. Optional history uses ``[N, C, W]`` or ``[N, C, W-1]``; gradients
    propagate through history as well as the input and filter. No final state
    is requested by TorchTitan's training path.
    """
    if x_BTC.ndim != 3 or weight_CW.ndim != 2:
        raise ValueError("causal_conv1d expects input [B, T, C] and weight [C, W].")
    batch, tokens, channels = x_BTC.shape
    if weight_CW.shape[0] != channels:
        raise ValueError("causal_conv1d input and weight channel counts must match.")
    if num_heads < 1 or channels % num_heads:
        raise ValueError("causal_conv1d channels must be divisible by num_heads.")
    width = weight_CW.shape[1]
    if width < 2:
        raise ValueError("AscendC causal_conv1d requires a filter width of at least 2.")
    boundaries = None
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("Packed causal_conv1d requires batch size 1.")
        if isinstance(cu_seqlens, torch.Tensor) and (
            cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens must be a 1D integer tensor.")
        boundaries = (
            cu_seqlens.detach().cpu().tolist()
            if isinstance(cu_seqlens, torch.Tensor)
            else list(cu_seqlens)
        )
        if any(not isinstance(offset, int) for offset in boundaries):
            raise ValueError("cu_seqlens must contain integer offsets.")
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != tokens
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("cu_seqlens must be ordered boundaries from 0 to T.")
    num_sequences = batch if boundaries is None else len(boundaries) - 1
    if initial_state is not None:
        if initial_state.ndim != 3 or initial_state.shape[:2] != (
            num_sequences,
            channels,
        ):
            raise ValueError("causal_conv1d history must have shape [N, C, W].")
        if initial_state.shape[-1] == width - 1:
            initial_state = F.pad(initial_state, (1, 0))
        elif initial_state.shape[-1] != width:
            raise ValueError("causal_conv1d history length must be W or W-1.")
    return _AscendCCausalConv1d.apply(
        x_BTC.contiguous(), weight_CW, initial_state, num_heads, boundaries
    )
