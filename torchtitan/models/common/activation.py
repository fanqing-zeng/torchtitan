# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.config.configurable import Configurable
from torchtitan.config.function import Function
from torchtitan.distributed.local_compile import local_compile

# Shape suffixes: F = feed-forward hidden dimension, P = packed gate/up features.


class _AscendCSiTUGLU(torch.autograd.Function):
    """Run the ops-nn SiTU forward and backward on packed gate/up features."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, gate_up_F: torch.Tensor, beta: float, linear_beta: float | None
    ) -> torch.Tensor:
        import cann_ops_nn

        ctx.beta = beta
        ctx.linear_beta = 0.0 if linear_beta is None else linear_beta
        ctx.input_shape = gate_up_F.shape
        gate_up_P = gate_up_F.flatten(-2).contiguous()
        ctx.save_for_backward(gate_up_P)
        return cann_ops_nn.situ_glu(
            gate_up_P,
            dim=-1,
            beta=beta,
            linear_beta=ctx.linear_beta,
            activate_left=True,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_F: torch.Tensor):  # pyrefly: ignore[bad-override]
        import cann_ops_nn

        (gate_up_P,) = ctx.saved_tensors
        grad_gate_up_P = cann_ops_nn.situ_glu_grad(
            grad_output_F.contiguous(),
            gate_up_P,
            dim=-1,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            activate_left=True,
        )
        return grad_gate_up_P.reshape(ctx.input_shape), None, None


@spmd.register_decomposition(_AscendCSiTUGLU)
def _situ_glu_reference(
    gate_up_F: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    gate_F, up_F = gate_up_F.float().unbind(-2)
    gate_F = beta * torch.tanh(gate_F / beta) * torch.sigmoid(gate_F)
    if linear_beta is not None:
        up_F = linear_beta * torch.tanh(up_F / linear_beta)
    return (gate_F * up_F).to(gate_up_F.dtype)


class BinaryActivationFn(Function[torch.Tensor], ABC):
    """Base class for gated activations such as SwiGLU, applied to ``w13``'s output.

    ``__call__`` takes ``gate_up [..., 2, F]``, unbinds it, and calls the subclass's
    ``_activation_fn(gate, up)``, all inside the ``fused_binary_activation`` region.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        pass

    # recompile_limit=16: ~3 graphs per static F; DeepSeek-V3 at TP8 needs 10 (default 8).
    @local_compile("fused_binary_activation", batch_invariant=True, recompile_limit=16)
    def __call__(
        self, gate_up: torch.Tensor, *, offsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the activation to ``gate_up``: ``[..., 0, :]`` is gate, ``[..., 1, :]`` is up.

        The unbind happens here, inside the compiled region, so backward writes one
        ``[..., 2, F]`` gradient instead of autograd copying two gradients into it.

        Args:
            gate_up: ``[..., 2, F]``, as ``w13`` returns it.
            offsets: for routed experts, where each expert's rows end. Rows past
                ``offsets[-1]`` are padding that a kernel may skip (FusedSwiGLU does).
                This compiled path computes every row and never reads ``offsets``.

        Example:

            hidden_TF = activation_fn(w13(x_TD))  # w13 returns [T, 2, F]
        """
        # Keep F static. Without this, the second F this region sees (e.g. dense FFN,
        # then experts) makes Dynamo recompile it for any F, and that kernel divides by
        # a runtime F: ~1.9x slower fwd+bwd, ~1.25x slower no-grad forward.
        torch._dynamo.mark_static(gate_up, -1)
        gate, up = gate_up.unbind(-2)
        return self._activation_fn(gate, up)

    @abstractmethod
    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor, /) -> torch.Tensor:
        """Return the activation of ``gate`` and ``up``, both ``[..., F]``.

        Runs inside the compiled region, so it must compile with ``fullgraph=True`` and
        stay elementwise: the region is marked batch invariant.
        """


class UnaryActivationFn(Function[torch.Tensor], ABC):
    """Base class for configurable one-input activation functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        pass

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass


class Sigmoid(UnaryActivationFn):
    """Sigmoid activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return torch.sigmoid(x)


class SiLU(UnaryActivationFn):
    """SiLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.silu(x)


class Softmax(UnaryActivationFn):
    """Softmax activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        dim: int = -1

    def __init__(self, config: Config) -> None:
        self.dim = config.dim

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.softmax(x, dim=self.dim)


class SqrtSoftplus(UnaryActivationFn):
    """Square root of softplus activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.softplus(x).sqrt()


class SwiGLU(BinaryActivationFn):
    """SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


class ClampedSwiGLU(BinaryActivationFn):
    """Clamped SwiGLU with configurable sigmoid scale and input bound."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        swiglu_alpha: float = 1.702
        swiglu_limit: float = 7.0

        def __post_init__(self) -> None:
            if not math.isfinite(self.swiglu_alpha) or self.swiglu_alpha <= 0:
                raise ValueError(
                    "swiglu_alpha must be finite and positive, got "
                    f"{self.swiglu_alpha}"
                )
            if not math.isfinite(self.swiglu_limit) or self.swiglu_limit <= 0:
                raise ValueError(
                    "swiglu_limit must be finite and positive, got "
                    f"{self.swiglu_limit}"
                )

    def __init__(self, config: Config) -> None:
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        silu = gate * torch.sigmoid(self.alpha * gate)
        return torch.addcmul(silu, silu, up)


# TODO: move to models/kimi_k3, its only user.
class SiTUGLU(BinaryActivationFn):
    """Kimi's SiTU-GLU, with an optional AscendC implementation on NPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        beta: float = 1.0
        linear_beta: float | None = None
        implementation: Literal["eager", "ascendc"] = "eager"

        def __post_init__(self) -> None:
            if self.implementation not in ("eager", "ascendc"):
                raise ValueError(
                    f"Unsupported SiTUGLU implementation: {self.implementation}"
                )
            if not math.isfinite(self.beta) or self.beta <= 0:
                raise ValueError("SiTUGLU beta must be finite and positive")
            if self.linear_beta is not None and (
                not math.isfinite(self.linear_beta) or self.linear_beta <= 0
            ):
                raise ValueError(
                    "SiTUGLU linear_beta must be None or finite and positive"
                )

    def __init__(self, config: Config) -> None:
        self.beta = config.beta
        self.linear_beta = config.linear_beta
        self.implementation = config.implementation

    def __call__(
        self, gate_up: torch.Tensor, *, offsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.implementation == "ascendc" and gate_up.device.type == "npu":
            if gate_up.ndim < 2 or gate_up.shape[-2] != 2:
                raise ValueError(
                    "SiTUGLU expects packed gate/up with shape [..., 2, F]"
                )
            # Flatten the existing W13 layout without concatenating gate and up.
            # Compute padded expert rows too, matching BinaryActivationFn.
            return _AscendCSiTUGLU.apply(gate_up, self.beta, self.linear_beta)
        return super().__call__(gate_up, offsets=offsets)

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        input_dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(input_dtype)
