# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention using Attention Gym kernels."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat

if TYPE_CHECKING:
    # Type-only: the CP routing type belongs to the CUDA-only context-parallel
    # path; importing it eagerly would drag its kernels onto every device.
    from attn_gym.linear.context_parallel import ContextParallelRouting
from torch import nn

from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.norm import GatedRMSNorm
from torchtitan.protocols.module import Module

from .attention import (
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from .npu_causal_conv1d import _AscendCCausalConv1d, ascendc_causal_conv1d
from .npu_kda import _NPUKDA, npu_chunk_kda

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# W = convolution kernel width.


@dataclass(frozen=True, slots=True)
class LinearAttentionMetadata:
    """Sequence metadata shared by convolutional linear-attention backends."""

    varlen: VarlenAttentionMetadata | None
    num_conv_history_tokens: int

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets."""
        if self.varlen is not None:
            self.varlen.annotate_spmd_types()


# These kernels run on rank-local heads inside InnerKDA's local SPMD region.
spmd.register_local_autograd_function(_NPUKDA)
spmd.register_local_autograd_function(_AscendCCausalConv1d)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """Per-head L2 normalization over the last dimension (fla's l2norm)."""
    return x * torch.rsqrt(x.float().pow(2).sum(dim=-1, keepdim=True) + 1e-6).to(
        x.dtype
    )


def _causal_conv1d_silu(
    qkv_1TC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv1d + SiLU over packed documents, in plain torch.

    ``qkv_1TC`` is ``[1, T, C]`` and ``weight_CW`` is ``[C, W]`` (one W-tap
    filter per channel). With ``cu_seqlens`` each document convolves
    independently: taps reaching past the document start contribute zero,
    which is the varlen semantics of the previous short-conv kernel.
    ``initial_state`` is unsupported (the local path never seeds a state).
    """
    if initial_state is not None:
        raise NotImplementedError(
            "KDA short convolution does not support an initial state on this " "path."
        )
    x_TC = qkv_1TC.squeeze(0)
    seq_len, num_channels = x_TC.shape
    kernel_size = weight_CW.shape[-1]
    device = x_TC.device

    token_index = torch.arange(seq_len, device=device)
    if cu_seqlens is not None:
        # cu_seqlens = [0, end(doc0), ..., T]; each token's document starts at
        # the cu entry selected by counting how many document ends are <= t.
        doc_ids = torch.searchsorted(cu_seqlens[1:], token_index, right=True)
        doc_start = cu_seqlens[doc_ids].to(token_index.dtype)
    else:
        doc_start = torch.zeros(seq_len, device=device, dtype=token_index.dtype)

    out_TC = torch.zeros_like(x_TC)
    for tap in range(kernel_size):
        source = token_index - tap
        valid = source >= doc_start
        gathered = x_TC[source.clamp_min(0)] * valid.unsqueeze(-1).to(x_TC.dtype)
        # F.conv1d is cross-correlation: weight index k sees the tap at
        # distance (kernel_size - 1 - k).
        out_TC = out_TC + gathered * weight_CW[:, kernel_size - 1 - tap].unsqueeze(
            0
        ).to(x_TC.dtype)
    return F.silu(out_TC).unsqueeze(0)


class KDAKernel(Module):
    """Apply KDA preprocessing and the Attention Gym kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float = -5.0
        implementation: str = "auto"

        def __post_init__(self):
            if self.implementation not in ("auto", "ascendc", "fla"):
                raise ValueError("KDA implementation must be auto, ascendc, or fla.")
            if not -5.0 <= self.lower_bound < 0.0:
                raise ValueError(
                    "KDA lower_bound must be in the safe range [-5, 0), "
                    f"got {self.lower_bound}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound
        self.implementation = config.implementation

    def forward(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_1THV, _ = npu_chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
            backend=self.implementation,
        )
        return output_1THV

    def prepare_inputs(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager preprocessing shared by local and CP KDA cores.

        ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`` in FP32 is
        the bounded gate transform; the Q/K L2 norm is the standard per-head
        normalization. The local (fla) path folds both into its kernel; this
        composite stays for the context-parallel core.
        """
        gate_1THK = self.lower_bound * torch.sigmoid(
            torch.exp(A_log_H.float()).unsqueeze(-1).unsqueeze(0)
            * (raw_gate_1THK.float() + dt_bias_HK.float())
        )
        return (
            _l2norm(q_1THK),
            _l2norm(k_1THK),
            gate_1THK,
            raw_beta_1TH.float().sigmoid(),
        )


class InnerKDA(InnerAttention):
    """Run short convolution and KDA behind the vLLM replacement boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        head_dim: int
        conv_kernel_size: int
        kernel: KDAKernel.Config
        conv_implementation: str = "eager"

        def __post_init__(self):
            if self.conv_implementation not in ("eager", "ascendc"):
                raise ValueError(
                    "KDA convolution implementation must be eager or ascendc."
                )
            if self.head_dim not in (64, 128):
                raise ValueError(
                    "KDA supports head_dim 64 or 128 (the fla kernel path), "
                    f"got {self.head_dim}."
                )

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> LinearAttentionMetadata:
            """Build packed-sequence metadata consumed by KDA."""
            return LinearAttentionMetadata(
                varlen=create_varlen_metadata_for_document(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                ),
                num_conv_history_tokens=self.conv_kernel_size - 1,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.kernel = config.kernel.build()
        self.conv_implementation = config.conv_implementation

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        varlen = attention_metadata.varlen if attention_metadata is not None else None
        cu_seqlens = varlen.cu_seq_q if varlen is not None else None
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            routing=None,
        )

    def run_stages(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: "ContextParallelRouting | None",
    ) -> torch.Tensor:
        """Run the stages shared by local and context-parallel KDA."""
        raw_gate_1THK = raw_gate_THK.unsqueeze(0)
        raw_beta_1TH = raw_beta_TH.unsqueeze(0)
        mixed_qkv_1TC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        conv_output_1TC = self.short_convolution(
            mixed_qkv_1TC,
            conv_weight_C1W,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

        q_1TC, k_1TC, v_1TC = conv_output_1TC.chunk(3, dim=-1)
        q_1THK, k_1THK, v_1THV = (
            tensor.unflatten(-1, (-1, self.head_dim))
            for tensor in (q_1TC, k_1TC, v_1TC)
        )
        output_1THV = self.kda_core(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )
        return output_1THV.squeeze(0)

    def short_convolution(
        self,
        qkv_1TC: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: "ContextParallelRouting | None",
    ) -> torch.Tensor:
        del routing
        if self.conv_implementation == "ascendc" and qkv_1TC.device.type == "npu":
            return ascendc_causal_conv1d(
                qkv_1TC,
                conv_weight_C1W[:, 0],
                num_heads=qkv_1TC.shape[-1] // self.head_dim,
                cu_seqlens=cu_seqlens,
                initial_state=initial_state,
            )
        output_1TC = _causal_conv1d_silu(
            qkv_1TC,
            conv_weight_C1W[:, 0],
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
        )
        assert isinstance(output_1TC, torch.Tensor)
        return output_1TC

    def kda_core(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: "ContextParallelRouting | None",
    ) -> torch.Tensor:
        del routing
        return self.kernel(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
        )


class KDA(Module):
    """Kimi Delta Attention with checkpoint-compatible Kimi K3 parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv: Conv1d.Config
        k_conv: Conv1d.Config
        v_conv: Conv1d.Config
        forget_a: Linear.Config
        forget_b: Linear.Config
        beta: Linear.Config
        output_gate: Linear.Config
        inner_kda: Module.Config
        output_norm: GatedRMSNorm.Config
        output_proj: Linear.Config

        def __post_init__(self):
            if self.num_heads < 1:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            # head_dim 64 runs fla's triton-ascend kernel via the npu_chunk_kda
            # fall-through; head_dim 128 (the released topology) is taken over
            # by fla_npu's fused ascendc kernels inside npu_chunk_kda, because
            # fla's triton kernel hangs at 128 on this stack.
            if self.head_dim not in (64, 128):
                raise ValueError(
                    "KDA head_dim must be 64 or 128 "
                    f"(the fla/triton-ascend kernel hangs at 128, which "
                    f"npu_chunk_kda takes over), got {self.head_dim}."
                )
            if self.conv_kernel_size < 1:
                raise ValueError(
                    f"conv_kernel_size must be positive, got {self.conv_kernel_size}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv = config.q_conv.build()
        self.k_conv = config.k_conv.build()
        self.v_conv = config.v_conv.build()
        self.forget_a = config.forget_a.build()
        self.forget_b = config.forget_b.build()
        self.beta = config.beta.build()
        self.output_gate = config.output_gate.build()
        self.inner_kda = config.inner_kda.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: LinearAttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        # All KDA input projections consume x, so gather once at their common
        # module boundary.
        x_TD = maybe_gather_tp_input(self, x_TD)

        if x_TD.ndim != 2:
            raise ValueError(
                f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
            )

        raw_gate_THK = local_head_split(
            self.forget_b(self.forget_a(x_TD)),
            self.head_dim,
            cp_shard_dim=0,
        )
        raw_beta_TH = self.beta(x_TD)
        out_THV = remat.region(
            self.inner_kda,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            self.q_proj(x_TD),
            self.k_proj(x_TD),
            self.v_proj(x_TD),
            raw_gate_THK,
            raw_beta_TH,
            self.q_conv.weight,
            self.k_conv.weight,
            self.v_conv.weight,
            self.A_log,
            self.dt_bias,
            attention_metadata=attention_metadata,
        )

        output_gate_THV = local_head_split(
            self.output_gate(x_TD), self.head_dim, cp_shard_dim=0
        )
        # The gated output norm reads the inner_attention and output_gate projection
        # outputs with bare ops.
        remat.recompute_needs_tensor(out_THV, output_gate_THV)
        return self.output_proj(self.output_norm(out_THV, output_gate_THV).flatten(-2))
