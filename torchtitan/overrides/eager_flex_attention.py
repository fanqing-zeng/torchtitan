# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run FlexAttention eager.

``torch.compile``-lowered FlexAttention requires a complete inductor
flex-attention hop. On platforms whose lowering is incomplete or whose
compiled backward produces wrong values for some mask/shape combinations,
the compiled hop either fails at runtime or silently corrupts gradients.

Eager ``flex_attention`` computes the same results (compilation only fuses
the same implementation), so this override rebinds the class-level compiled
entries to the eager callables. It is opt-in: only recipes that list it in
``override.imports`` are affected.

The rebind happens at import time of this module; the registered factory below
is an identity transform kept only so the module can be listed in
``override.imports`` (entries must name a registered ``@override`` factory):

    torchtitan_train ... --override.imports torchtitan.overrides.eager_flex_attention.eager_flex_attention
"""

import importlib

from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

from torchtitan.config import override
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common import attention as titan_attention
from torchtitan.models.common import vision_encoder as titan_vision_encoder
from torchtitan.models.common.attention import FlexInnerAttention

FlexInnerAttention._compiled_flex_attn = flex_attention
# Mask construction also compiles under torch.compile and can hit the same
# incomplete lowering paths; run it eager too.
titan_attention._compiled_create_block_mask = create_block_mask

# Second mask-compile site: the shared vision encoder has its own compiled
# wrapper, and the per-model vision encoders re-export it via from-imports --
# so every holding module must be rebound, not just the defining one.
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


@override(
    target=FullAC.Config,
    exact=True,
    description="FlexAttention eager rebind (identity transform; the rebind "
    "happens at module import).",
)
def eager_flex_attention(cfg: FullAC.Config) -> FullAC.Config:
    return cfg
