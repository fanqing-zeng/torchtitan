# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run the ``gated_rmsnorm`` local-compile region eager.

Kimi K3 compiles its ``GatedRMSNorm`` into an independent ``torch.compile``
region (``local_compile_regions=["gated_rmsnorm", ...]``) so the norm + gate
pair fuses on stacks with a complete inductor. On stacks whose inductor
backward for this region is defective, the region's backward silently returns
garbage for the norm's weight gradient (values at ~1e38 / NaN on isolated
elements while every other gradient in the model stays at ~1e-2), which then
poisons the whole step through gradient clipping.

The eager composite computes identical math, so this override removes the
region from the model's ``local_compile_regions``. It is opt-in: only recipes
that list it in ``override.imports`` are affected.
"""

from torchtitan.config import override
from torchtitan.models.kimi_k3.model import KimiK3Model

_REGION = "gated_rmsnorm"


@override(
    target=KimiK3Model.Config,
    exact=True,
    description="Drop the gated_rmsnorm local-compile region (run the "
    "GatedRMSNorm composite eager; its compiled backward is defective on "
    "this stack).",
)
def eager_gated_rmsnorm(cfg: KimiK3Model.Config) -> KimiK3Model.Config:
    cfg.local_compile_regions = [
        region for region in cfg.local_compile_regions if region != _REGION
    ]
    return cfg
