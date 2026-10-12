# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Vendored from FSDPTurbo (PR #78, "moonep") and trimmed to the Ascend NPU
# backend for torchtitan.

"""Normalized view of the user's MoonEP configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoonEPSettings:
    """Normalized, immutable view of the MoonEP dispatcher config.

    Read through this rather than off the config object itself so a config that
    predates a knob degrades to the default here instead of raising
    ``AttributeError`` inside an expert forward.
    """

    num_sms: int = 32
    token_padding: int = 128
    comm_stream_priority: int = -1
    async_finish: bool = False
    tokens_per_rank: int | None = None
    top_k: int | None = None
    recompute_dispatch: bool = False
    validate_plan_invariants: bool = False
    shmem_ip_port: str | None = None
    # Ascend only: CANN SHMEM's stream-ordered global barrier before the
    # weight-gradient reduction, instead of a host synchronize plus dist.barrier.
    shmem_stream_barrier: bool = True

    # Knobs that do nothing on one accelerator. Setting one is not an error, but
    # "I turned it on and saw no change" is expensive to work out from outside.
    _INERT_BY_DEVICE = {
        # Both are CANN SHMEM concepts.
        "cuda": ("shmem_stream_barrier", "shmem_ip_port"),
        # ascend_moonep runs its collectives on the calling stream whatever is
        # asked, so no side stream carries work and its priority has no target.
        "npu": ("async_finish", "comm_stream_priority"),
    }

    def inert_knobs(self, device_type: str) -> tuple:
        """Names of set-but-ignored knobs on ``device_type``, in field order."""
        candidates = self._INERT_BY_DEVICE.get(device_type, ())
        defaults = type(self)()
        return tuple(name for name in candidates if getattr(self, name) != getattr(defaults, name))

    def warn_about_inert_knobs(self, device_type: str) -> None:
        inert = self.inert_knobs(device_type)
        if inert:
            logger.warning(
                "moonep_config %s %s no effect on %s and %s ignored.",
                ", ".join(inert),
                "has" if len(inert) == 1 else "have",
                device_type,
                "is" if len(inert) == 1 else "are",
            )

    def report_effective_async(self, device_type: str, supported: bool) -> None:
        """State what asynchronous completion actually resolved to.

        A config that reads ``async_finish=True`` while nothing overlaps is how
        somebody concludes the communication is overlapping when it is not.
        """
        if self.async_finish and not supported:
            logger.info(
                "moonep_config.async_finish=True has no effect on %s: the backend "
                "has no usable asynchronous stream, so the collectives run on the "
                "compute stream.",
                device_type,
            )

    @classmethod
    def from_config(cls, config: Any) -> "MoonEPSettings":
        """Adapt any config object exposing a subset of the field names.

        The torchtitan side passes a ``MoonEPTokenDispatcher.Config``; every
        field it does not define keeps the default here.
        """
        if config is None:
            return cls()
        missing = []
        values = {}
        for field in fields(cls):
            if hasattr(config, field.name):
                values[field.name] = getattr(config, field.name)
            else:
                missing.append(field.name)
        if missing:
            logger.warning(
                "The MoonEP dispatcher config does not define %s; using the defaults.",
                ", ".join(missing),
            )
        return cls(**values)
