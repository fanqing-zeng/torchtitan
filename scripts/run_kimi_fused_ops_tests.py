# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Kimi fused-operator CPU tests and optionally require real Ascend tests."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def require_npu() -> None:
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("--npu requires an accessible Ascend device.")
    required = {
        "fla_npu.ops.ascendc": (
            "chunk_kda_fwd",
            "chunk_kda_bwd",
            "causal_conv1d",
            "causal_conv1d_bwd",
        ),
        "cann_ops_nn": ("situ_glu", "situ_glu_grad"),
        "fla.modules.fused_norm_gate": ("rms_norm_gated", "LayerNormGatedFunction"),
        "fla.ops.attnres": ("fused_attnres",),
        "triton": ("jit",),
    }
    for name, symbols in required.items():
        module = importlib.import_module(name)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            raise RuntimeError(f"{name} is missing required public APIs: {missing}")
    print(
        f"NPU: {torch.npu.get_device_name(0)}; PyTorch: {torch.__version__}", flush=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npu",
        action="store_true",
        help="Require NPU and fused dependencies; run device numerics too.",
    )
    args, pytest_args = parser.parse_known_args()
    if args.npu:
        require_npu()
    root = Path(__file__).resolve().parents[1]
    tests = [
        "tests/unit_tests/cpu/test_npu_activation_norm.py",
        "tests/unit_tests/cpu/test_npu_causal_conv1d.py",
        "tests/unit_tests/cpu/test_npu_kda.py",
        "tests/unit_tests/cpu/test_kimi_attn_res.py",
        "tests/unit_tests/cpu/test_kimi_fused_ops_config.py",
    ]
    if args.npu:
        tests.append("tests/unit_tests/npu")
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *tests, *pytest_args], cwd=root, check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
