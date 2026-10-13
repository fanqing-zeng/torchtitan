# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""One rank of the cross-rank gradient-norm agreement check.

Launched as a subprocess by ``test_grad_norm_agreement.py``; not a test
module itself. Reproduces how MoonEP registers an expert weight and how its
gradient reaches the parameter, then reports what ``clip_grad_norm_`` would see.

``MODE=production`` mirrors the real path. ``MODE=unsharded`` drops the sharding
information so the test can confirm the check would notice if it were lost.
``MODE=dtensor_grad`` has the node return a DTensor, which is the shape of the
"fix" that a reader who does not know about ``to_local()`` is likely to reach for.
"""

import json
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.distributed.moonep.autograd import _local
from torchtitan.distributed.utils import clip_grad_norm_

ROWS, COLS = 2, 3


class ExpertGradientBridge(torch.autograd.Function):
    """Stands in for _MoonEPWeightsBridge.

    Takes the weight only to carry the autograd edge, and its backward returns
    the reduced home-expert gradient as the plain local-shaped tensor MoonEP's
    reduction kernel produces.
    """

    @staticmethod
    def forward(ctx, weight, mesh=None, wrap=False):
        ctx.mesh = mesh
        ctx.wrap = wrap
        return torch.zeros((), requires_grad=False)

    @staticmethod
    def backward(ctx, grad_marker):
        reduced = torch.full((ROWS, COLS), float(dist.get_rank() + 1))
        if ctx.wrap:
            reduced = DTensor.from_local(reduced, ctx.mesh, (Shard(0),), run_check=False)
        return reduced, None, None


def build_parameter(mesh, mode):
    home = torch.zeros(ROWS, COLS)
    if mode == "unsharded":
        return torch.nn.Parameter(home)
    return torch.nn.Parameter(DTensor.from_local(home, mesh, (Shard(0),), run_check=False))


def main() -> int:
    mode = os.environ["MODE"]
    # A file rendezvous rather than TCP: picking a free port and then handing it
    # to two subprocesses is a race, and losing it made this check skip itself on
    # a busy machine. The store is constructed directly because a ``file://``
    # init_method is parsed as a URL, which mangles absolute Windows paths.
    dist.init_process_group(
        backend="gloo",
        store=dist.FileStore(os.environ["RENDEZVOUS_FILE"], int(os.environ["WORLD_SIZE"])),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("ep",))
        parameter = build_parameter(mesh, mode)

        # MoonEP hands the autograd node ``parameter.to_local()``, which is what
        # lets the node return a local-shaped gradient at all.
        ExpertGradientBridge.apply(_local(parameter), mesh, mode == "dtensor_grad").backward()

        # A plain call with no mesh: any all-reduce has to come from the
        # gradient's own placements, exactly as the ep_enabled=False path of
        # the trainer's clip_grad_norm_ sees it. An infinite max_norm leaves the
        # gradients untouched and still reports the total norm.
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "grad_type": type(parameter.grad).__name__,
                    "norm": float(clip_grad_norm_([parameter], max_norm=float("inf"), norm_type=2.0)),
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
