# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA training through the public fla_npu AscendC forward/backward API.

Requires ``fla_npu.ops.ascendc.chunk_kda_fwd`` (12 results, BSND output,
BNSD intermediates) and ``chunk_kda_bwd`` (8 results, BNSD or packed HTD).
The fused training contract is K=128, V=128/256, chunk_size=64, equal query
and value head counts, and no initial/final state. K=64 debug models use FLA.
Optional accelerator packages are imported only when their backend is used.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

# B = batch, T = sequence tokens, H = heads, K = query/key dimension,
# V = value dimension, C = chunk size, N = number of chunks.


def _load_ascendc_kernels():
    try:
        from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd
    except ImportError as exc:
        raise ImportError(
            "AscendC KDA requires fla_npu with the public chunk_kda_fwd and "
            "chunk_kda_bwd APIs used by MindSpeed-MM. Update fla_npu and its "
            "matching CANN operator package; private ctypes APIs are unsupported."
        ) from exc
    return chunk_kda_fwd, chunk_kda_bwd


@lru_cache(maxsize=1)
def _register_fla_kernel_once() -> None:
    try:
        import spmd_types as spmd
    except ModuleNotFoundError as exc:
        if exc.name != "spmd_types":
            raise
    else:
        from fla.ops.kda.chunk import ChunkKDAFunction

        spmd.register_local_autograd_function(ChunkKDAFunction)


@torch.compiler.assume_constant_result
def _register_fla_kernel() -> None:
    # Registration is tensor-independent setup, not part of the compiled graph.
    _register_fla_kernel_once()


def _load_fla_kernel():
    _register_fla_kernel()
    from fla.ops.kda import chunk_kda

    return chunk_kda


def _l2norm(x_BTHK: torch.Tensor) -> torch.Tensor:
    """Match FLA's FP32 L2 normalization, including its epsilon at zero."""
    x_fp32_BTHK = x_BTHK.float()
    return (
        x_fp32_BTHK * torch.rsqrt(x_fp32_BTHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(x_BTHK.dtype)


def _head_first(x: torch.Tensor, *, packed: bool) -> torch.Tensor:
    x = x.transpose(1, 2).contiguous()
    return x.squeeze(0) if packed else x


def _sequence_first(x: torch.Tensor, *, packed: bool) -> torch.Tensor:
    if packed:
        x = x.unsqueeze(0)
    return x.transpose(1, 2).contiguous()


def _host_offsets(cu_seqlens, *, batch: int, tokens: int):
    if cu_seqlens is None:
        return None
    if batch != 1:
        raise ValueError("AscendC KDA packed sequences require batch size 1.")
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("KDA cu_seqlens must be a 1D integer tensor.")
        cu_seqlens = cu_seqlens.detach().cpu().tolist()
    if any(not isinstance(offset, int) for offset in cu_seqlens):
        raise ValueError("KDA cu_seqlens must contain integer offsets.")
    offsets = tuple(cu_seqlens)
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != tokens
        or any(start > end for start, end in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            "KDA cu_seqlens must be nondecreasing, start at 0, and end at T."
        )
    return offsets


class _NPUKDA(torch.autograd.Function):
    """The AscendC core; preprocessing stays in PyTorch's autograd graph."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        q_BTHK,
        k_BTHK,
        v_BTHV,
        g_BTHK,
        beta_BTH,
        A_log_H,
        dt_bias_HK,
        scale,
        cu_seqlens,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
    ):
        forward, backward = _load_ascendc_kernels()
        (
            output_BTHV,
            _final_state,
            gk_BHTK,
            Aqk_BHTC,
            Akk_BHTC,
            w_BHTK,
            _u_BHTV,
            qg_BHTK,
            kg_BHTK,
            v_new_BHTV,
            h_BNHKV,
            _initial_state,
        ) = forward(
            q_BTHK,
            k_BTHK,
            v_BTHV,
            g_BTHK,
            beta_BTH,
            scale,
            64,
            layout="BSND",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=None,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log_H,
            dt_bias=dt_bias_HK,
            disable_recompute=True,
            return_intermediate_states=False,
            state_v_first=False,
        )
        intermediates = (
            gk_BHTK,
            Aqk_BHTC,
            Akk_BHTC,
            w_BHTK,
            qg_BHTK,
            kg_BHTK,
            v_new_BHTV,
            h_BNHKV,
        )
        if any(tensor is None for tensor in intermediates):
            raise RuntimeError(
                "AscendC KDA forward must return all saved tensors with "
                "disable_recompute=True. Install a matching fla_npu operator build."
            )
        batch, tokens, heads, key_dim = q_BTHK.shape
        num_chunks = (
            (tokens + 63) // 64
            if cu_seqlens is None
            else sum(
                (end - start + 63) // 64
                for start, end in zip(cu_seqlens, cu_seqlens[1:])
            )
        )
        key_shape = (batch, heads, tokens, key_dim)
        expected_shapes = (
            key_shape,
            (batch, heads, tokens, 64),
            (batch, heads, tokens, 64),
            key_shape,
            key_shape,
            key_shape,
            (batch, heads, tokens, v_BTHV.shape[-1]),
            (batch, num_chunks, heads, key_dim, v_BTHV.shape[-1]),
        )
        for index, (tensor, shape) in enumerate(zip(intermediates, expected_shapes)):
            dtype = torch.float32 if index == 0 else q_BTHK.dtype
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    "AscendC KDA forward returned incompatible saved tensor "
                    f"{index}: expected contiguous {shape} {dtype}, "
                    f"got {tuple(tensor.shape)} {tensor.dtype}."
                )
        ctx.save_for_backward(
            q_BTHK,
            k_BTHK,
            v_BTHV,
            g_BTHK,
            beta_BTH,
            A_log_H,
            dt_bias_HK,
            *intermediates,
        )
        ctx.kernel_backward = backward
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return output_BTHV.to(v_BTHV.dtype)

    @staticmethod
    def backward(ctx, grad_output_BTHV):  # pyrefly: ignore[bad-override]
        (
            q_BTHK,
            k_BTHK,
            v_BTHV,
            g_BTHK,
            beta_BTH,
            A_log_H,
            dt_bias_HK,
            *intermediates,
        ) = ctx.saved_tensors
        packed = ctx.cu_seqlens is not None
        if packed:
            intermediates = [tensor.squeeze(0) for tensor in intermediates]
        dq, dk, dv, db, dg, _dh0, dA, dbias = ctx.kernel_backward(
            _head_first(q_BTHK, packed=packed),
            _head_first(k_BTHK, packed=packed),
            _head_first(v_BTHV, packed=packed),
            _head_first(beta_BTH, packed=packed),
            *intermediates,
            _head_first(grad_output_BTHV, packed=packed),
            ctx.scale,
            raw_g=(
                _head_first(g_BTHK, packed=packed) if ctx.use_gate_in_kernel else None
            ),
            A_log=A_log_H,
            dt_bias=(
                dt_bias_HK.reshape(q_BTHK.shape[2:]).contiguous()
                if dt_bias_HK is not None
                else None
            ),
            initial_state=None,
            dht=None,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=None,
            chunk_size=64,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=True,
            use_exp2=True,
            state_v_first=False,
        )
        grads = [
            _sequence_first(grad, packed=packed).to(original)
            for grad, original in zip(
                (dq, dk, dv, dg, db),
                (q_BTHK, k_BTHK, v_BTHV, g_BTHK, beta_BTH),
            )
        ]
        for name, parameter, gradient in (
            ("A_log", A_log_H, dA),
            ("dt_bias", dt_bias_HK, dbias),
        ):
            if parameter is not None and parameter.requires_grad and gradient is None:
                raise RuntimeError(f"AscendC KDA backward omitted the {name} gradient.")
        return (
            *grads,
            dA.reshape_as(A_log_H).to(A_log_H) if dA is not None else None,
            dbias.reshape_as(dt_bias_HK).to(dt_bias_HK) if dbias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def npu_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    chunk_size=64,
    disable_recompute=None,
    return_intermediate_states=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    cp_context=None,
    *,
    backend="auto",
    **kwargs,
):
    """Dispatch KDA while enforcing the fused training operator's contract.

    ``auto`` uses FLA on other devices and for NPU K=64 debug models.
    ``ascendc`` explicitly selects the fused implementation. Its internal
    recomputation is unsupported: the default saves intermediates, while an
    explicit ``disable_recompute=False`` raises. Model-level activation
    checkpointing remains independent of this kernel option.
    """
    if backend not in ("auto", "ascendc", "fla"):
        raise ValueError(f"Unknown KDA backend: {backend!r}.")
    if backend == "fla" or (
        backend == "auto" and (q.device.type != "npu" or q.shape[-1] == 64)
    ):
        return _load_fla_kernel()(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
            disable_recompute=False if disable_recompute is None else disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )

    if (
        initial_state is not None
        or output_final_state
        or state_v_first
        or return_intermediate_states
        or cp_context is not None
    ):
        raise ValueError(
            "AscendC KDA training does not support initial/final/intermediate "
            "states, state_v_first, or cp_context."
        )
    if chunk_size != 64 or disable_recompute is False:
        raise ValueError(
            "AscendC KDA requires chunk_size=64 and saved intermediates "
            "(disable_recompute=True)."
        )
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("KDA expects q/k [B,T,H,K] and v [B,T,H,V].")
    batch, tokens, heads, key_dim = q.shape
    if (
        key_dim != 128
        or v.shape[:3] != q.shape[:3]
        or v.shape[-1] not in (128, 256)
        or g.shape != q.shape
        or beta.shape != q.shape[:3]
    ):
        raise ValueError(
            "AscendC KDA requires K=128, V=128/256, equal Q/K/V head counts, "
            "g [B,T,H,K], and beta [B,T,H]."
        )
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("AscendC KDA requires matching FP16/BF16 q, k, and v.")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel=True.")
    A_log_H = kwargs.pop("A_log", None)
    dt_bias_HK = kwargs.pop("dt_bias", None)
    if kwargs:
        raise ValueError(f"Unsupported AscendC KDA options: {sorted(kwargs)}.")
    if use_gate_in_kernel:
        if A_log_H is None or A_log_H.shape != (heads,):
            raise ValueError("Fused KDA gate requires A_log with shape [H].")
        if dt_bias_HK is not None and dt_bias_HK.numel() != heads * key_dim:
            raise ValueError("Fused KDA dt_bias must contain H*K elements.")
    elif A_log_H is not None or dt_bias_HK is not None:
        raise ValueError("A_log/dt_bias require use_gate_in_kernel=True.")
    if safe_gate:
        lower_bound = -5.0 if lower_bound is None else lower_bound
        if not -5.0 <= lower_bound < 0.0:
            raise ValueError("KDA safe_gate requires lower_bound in [-5, 0).")
    scale = key_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("KDA scale must be finite and positive.")
    offsets = _host_offsets(
        cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens,
        batch=batch,
        tokens=tokens,
    )
    if offsets is not None and use_gate_in_kernel:
        num_chunks = sum(
            (end - start + 63) // 64 for start, end in zip(offsets, offsets[1:])
        )
        if num_chunks > 1024:
            raise ValueError("AscendC fused KDA gate backward supports <=1024 chunks.")

    # Keep casts, L2 normalization, and sigmoid in autograd so the gradients
    # reach the original parameters, including BF16 gate parameters and logits.
    q_BTHK = _l2norm(q) if use_qk_l2norm_in_kernel else q
    k_BTHK = _l2norm(k) if use_qk_l2norm_in_kernel else k
    beta_BTH = beta.float()
    if use_beta_sigmoid_in_kernel:
        beta_BTH = beta_BTH.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    output_BTHV = _NPUKDA.apply(
        q_BTHK.contiguous(),
        k_BTHK.contiguous(),
        v.contiguous(),
        g.float().contiguous(),
        beta_BTH.contiguous(),
        A_log_H.float().contiguous() if A_log_H is not None else None,
        dt_bias_HK.float().reshape(-1).contiguous() if dt_bias_HK is not None else None,
        scale,
        offsets,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
    )
    return output_BTHV, None
