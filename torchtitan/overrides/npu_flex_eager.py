# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run FlexAttention eager on Ascend NPU.

torch_npu ships its own inductor flex-attention kernel. Its mask-subgraph
lowering rejects ``aten.index``, which every packed-document mask in torchtitan
uses (``packed_document_mask`` gathers with ``offsets[document_id[q_idx]]``),
so the compiled hop fails at runtime on NPU::

    SubgraphLoweringException: Buffers cannot be created while lowering a
    pointwise subgraph.

Eager ``flex_attention`` has no such restriction and produces the same results
(compilation only fuses the same implementation), so on NPU we rebind
``FlexAttention._compiled_flex_attn`` to the eager callable. On CUDA/ROCm the
module is inert: without ``torch_npu`` installed the rebind is skipped entirely.

The rebind happens at import time of this module; the registered factory below
is an identity transform kept only so the module can be listed in
``override.imports`` (entries must name a registered ``@override`` factory):

    torchtitan_train ... --override.imports torchtitan.overrides.npu_flex_eager.npu_flex_eager
"""

import importlib

import torch

from torchtitan.config import override
from torchtitan.distributed.activation_checkpoint import FullAC


def _on_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


if _on_npu():
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    from torchtitan.models.common import attention as titan_attention
    from torchtitan.models.common.attention import FlexAttention

    FlexAttention._compiled_flex_attn = flex_attention
    # Mask construction also compiles under torch.compile and can hit the
    # broken inductor paths (SIMT triton lowering); run it eager too.
    titan_attention._compiled_create_block_mask = create_block_mask
    # Second mask-compile site: the shared vision encoder (used by the
    # multimodal debugmodels, e.g. qwen3_5 / kimi_k2_5 / muse_glimmer) has its
    # own compiled wrapper, and the per-model vision encoders re-export it via
    # from-imports -- so every holding module must be rebound, not just the
    # defining one.
    from torchtitan.models.common import vision_encoder as titan_vision_encoder

    titan_vision_encoder.compiled_create_block_mask = create_block_mask
    for module_name in (
        "torchtitan.models.qwen3_5.vision_encoder",
        "torchtitan.models.kimi_k2_7.vision_encoder",
        "torchtitan.models.muse_glimmer.vision_encoder",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "compiled_create_block_mask"):
            module.compiled_create_block_mask = create_block_mask

    # torch_npu's repeat_interleave_backward_tensor has no Meta kernel, which
    # aborts tracing under torch.compile. A backward op always returns the
    # forward input's shape, so the fake impl is just that shape.
    @torch.library.register_fake("npu::repeat_interleave_backward_tensor")
    def _repeat_interleave_backward_tensor_fake(
        grad: torch.Tensor,
        input: torch.Tensor,
        repeats: torch.Tensor,
        dim: int | None = None,
    ) -> torch.Tensor:
        return grad.new_empty(input.size())


@override(
    target=FullAC.Config,
    exact=True,
    description="NPU FlexAttention eager rebind (identity transform; the "
    "rebind happens at module import).",
)
def npu_flex_eager(cfg: FullAC.Config) -> FullAC.Config:
    return cfg
