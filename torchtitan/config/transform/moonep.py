# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Model transform for the Ascend MoonEP routed-expert backend."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moonep import MoonEPRoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module
from .base import ModelConfigTransform, ModelConfigTransformContext


__all__ = ["MoonEPTransform"]


def _replace_routed_experts(model: Module.Config) -> Module.Config:
    """Replace every stock routed-expert config with the MoonEP variant."""
    for fqn, source, parent, attr in list(model.traverse(RoutedExperts.Config)):
        if type(source) is not RoutedExperts.Config:
            raise TypeError(
                "MoonEP requires an unmodified RoutedExperts.Config source; "
                f"got {type(source).__qualname__} at {fqn!r}"
            )
        if (
            type(source.w13) is not GroupedLinear.Config
            or type(source.w2) is not GroupedLinear.Config
        ):
            raise TypeError("MoonEP requires stock GroupedLinear W13/W2 projections")
        if not isinstance(source.activation_fn, BinaryActivationFn.Config):
            raise TypeError(
                "MoonEP requires a BinaryActivationFn.Config expert activation; "
                f"got {type(source.activation_fn).__qualname__} at {fqn!r}"
            )
        if not isinstance(source.token_dispatcher, AllToAllTokenDispatcher.Config):
            raise ValueError(
                "MoonEP requires the standard all-to-all source config; "
                "the vendored runtime replaces its dispatch and combine"
            )
        replacement = MoonEPRoutedExperts.Config(
            param_init=source.param_init,
            sharding_config=source.sharding_config,
            w13=source.w13,
            w2=source.w2,
            top_k=source.token_dispatcher.top_k,
            activation_fn=source.activation_fn,
            output_postprocess=source.output_postprocess,
        )
        if parent is None:
            model = replacement
        elif isinstance(parent, list):
            assert isinstance(attr, int)
            parent[attr] = replacement
        else:
            assert isinstance(attr, str)
            setattr(parent, attr, replacement)
    return model


@dataclass(kw_only=True, slots=True)
class MoonEPTransform(ModelConfigTransform):
    """Replace stock routed experts with Ascend MoonEP execution.

    The expert activation is passed through unchanged from each source
    ``RoutedExperts.Config.activation_fn`` (including Kimi K3's SiTU-GLU):
    MoonEP applies it through ``MoonEPRoutedExperts._apply_activation``
    instead of owning a kernel activation policy.
    """

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        """Replace each stock routed-expert config with the MoonEP variant."""
        return _replace_routed_experts(model)
