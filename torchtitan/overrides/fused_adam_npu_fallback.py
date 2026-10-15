from __future__ import annotations

"""NPU fallback for ``aten::_fused_adam_`` (the Adam fused path).

torch_npu's PU1 kernel for ``_fused_adam_`` calls ``aclnnFusedAdam``, which is
absent from CANN 9.2.0-beta.2 ``libopapi.so`` (only the ``Adamw`` variant used
by ``_fused_adamw_``/``aclnnApplyAdamWV2`` is present).  On 910B2 the same C++
kernel falls back to CPU (``SocVersion < Ascend950``); on 950DT it aborts, so
``torch.optim.Adam(fused=True)`` crashes at ``optimizer.step()`` — exactly what
the qwen3_moe param_groups recipe (router.gate group) hits.

The kernel registration happens at import time of this module with a composite
implementation whose math matches the torch CPU reference exactly (scalar
probe: p=0.9, ea=0.15; tensor probe: bitwise-identical to
``Adam(foreach=True)`` over 3 steps).  The registered factory below is an
identity transform kept only so the module can be listed in
``override.imports`` (entries must name a registered ``@override`` factory):

    --override.imports torchtitan.overrides.fused_adam_npu_fallback.fused_adam_npu_fallback
"""

import torch
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import override


def _fused_adam_npu_composite_fallback(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    if found_inf is not None and found_inf.numel() > 0:
        if float(found_inf.item()) != 0.0:
            return
    if grad_scale is not None:
        grads = [g / grad_scale for g in grads]
    for i, p in enumerate(params):
        g = grads[i]
        if maximize:
            g = g.neg()
        if weight_decay != 0:
            g = g.add(p, alpha=weight_decay)
        ea = exp_avgs[i]
        ev = exp_avg_sqs[i]
        st = state_steps[i]
        ea.lerp_(g, 1 - beta1)
        ev.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        step = float(st.item()) if torch.is_tensor(st) else float(st)
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        if amsgrad:
            m = max_exp_avg_sqs[i]
            torch.maximum(m, ev, out=m)
            denom = m.sqrt().div_(bc2**0.5).add_(eps)
        else:
            denom = ev.sqrt().div_(bc2**0.5).add_(eps)
        p.addcdiv_(ea, denom, value=-float(lr) / bc1)


torch.library.register_kernel(
    "aten::_fused_adam_", "PrivateUse1", _fused_adam_npu_composite_fallback
)


@override(
    target=OptimizersContainer.Config,
    exact=True,
    description="fused-adam composite fallback (kernel registration happens "
    "at module import).",
)
def fused_adam_npu_fallback(cfg: OptimizersContainer.Config) -> OptimizersContainer.Config:
    return cfg
