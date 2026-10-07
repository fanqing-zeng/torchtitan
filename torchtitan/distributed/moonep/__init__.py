# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP integration for MoE Expert Parallel.

Exposes :class:`MoonEPTokenDispatcher`, the token dispatcher wired into
``RoutedExperts`` when the MoonEP comm backend is selected, plus the
``moonep`` primitives module (buffer management, custom ops with manual
backward, plan handoff to the grouped experts).
"""

from torchtitan.distributed.moonep.moon_token_dispatcher import MoonEPTokenDispatcher

__all__ = ["MoonEPTokenDispatcher"]
