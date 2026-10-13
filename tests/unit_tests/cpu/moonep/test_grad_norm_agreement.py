# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Every rank must derive the same gradient norm from MoonEP's expert weights.

``clip_grad_norm_`` decides whether to all-reduce by inspecting the gradient's
placements. MoonEP's node returns a plain tensor; the placements survive because
``parameter.to_local()``'s own backward restores them. That is upstream
behaviour, so it is pinned here.

Needs more than one rank: with a single rank the local and global shapes coincide
and any arrangement looks correct. Two gloo ranks on CPU suffice.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.distributed.tensor")

WORKER = pathlib.Path(__file__).with_name("_grad_norm_agreement_worker.py")


def repository_root():
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "torchtitan").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate the repository root above {__file__}")


REPO_ROOT = repository_root()
WORLD_SIZE = 2
# Each rank's gradient is filled with rank+1 over a 2x3 shard, so the norm over
# all experts is sqrt(6*1^2 + 6*2^2) = sqrt(30) while rank 0 alone sees sqrt(6).
EXPECTED_GLOBAL_NORM = 30.0**0.5
RANK0_LOCAL_ONLY_NORM = 6.0**0.5


def run_ranks(mode, expect_failure=False):
    """Start WORLD_SIZE workers and return each rank's reported record.

    With ``expect_failure`` the ranks are expected to die, and their combined
    stderr is returned instead.
    """
    # A fresh path per call, and gone before the ranks start: FileStore appends to
    # whatever it finds, so a leftover file from an earlier call would be read as
    # a rendezvous already in progress.
    rendezvous = pathlib.Path(tempfile.gettempdir()) / f"moonep_rdzv_{mode}_{os.getpid()}_{time.monotonic_ns()}"
    rendezvous.unlink(missing_ok=True)

    env = dict(os.environ)
    env.update(
        MODE=mode,
        RENDEZVOUS_FILE=str(rendezvous),
        WORLD_SIZE=str(WORLD_SIZE),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    )
    # Gloo still opens sockets between the ranks for the collectives themselves.
    # On a multi-homed host it picks a real NIC and can fail to reach a peer in
    # the same container, so pin it to loopback where both ranks certainly are.
    # Linux only: the interface is named differently elsewhere, and naming a
    # missing one is worse than letting gloo choose.
    if sys.platform.startswith("linux"):
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    processes = []
    for rank in range(WORLD_SIZE):
        processes.append(
            subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, str(WORKER)],
                env={**env, "RANK": str(rank)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    records = {}
    failures = []
    errors = []
    for rank, process in enumerate(processes):
        try:
            out, err = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            failures.append(f"rank {rank} timed out")
            continue
        finally:
            rendezvous.unlink(missing_ok=True)
        if process.returncode != 0:
            errors.append(err)
            failures.append(f"rank {rank} exited {process.returncode}:\n{err}")
            continue
        line = next((ln for ln in out.splitlines() if ln.startswith("{")), None)
        if line is None:
            failures.append(f"rank {rank} printed no result:\n{out}\n{err}")
            continue
        records[rank] = json.loads(line)

    if expect_failure:
        if not errors:
            pytest.fail(f"the ranks were expected to fail but all succeeded: {records}")
        return "\n".join(errors)
    if failures:
        pytest.skip("cannot run a 2-rank gloo group here: " + "; ".join(failures))
    return records


def test_expert_gradient_keeps_the_parameter_sharding():
    records = run_ranks("production")
    types = {rank: record["grad_type"] for rank, record in records.items()}
    assert set(types.values()) == {"DTensor"}, (
        f"expert gradients are not DTensors ({types}), so clip_grad_norm_ would "
        "find no shard group and skip the all-reduce"
    )


def test_every_rank_computes_the_norm_over_all_experts():
    records = run_ranks("production")
    norms = {rank: record["norm"] for rank, record in records.items()}

    assert norms[0] == pytest.approx(norms[1], rel=1e-6), (
        f"ranks disagree on the gradient norm: {norms}. Each would clip by a "
        "different coefficient and the shards would stop matching."
    )
    assert norms[0] == pytest.approx(EXPECTED_GLOBAL_NORM, rel=1e-6), (
        f"norm {norms[0]} is not the norm over all experts ({EXPECTED_GLOBAL_NORM})"
    )


def test_returning_a_dtensor_from_the_node_is_rejected_by_autograd():
    """Why ``finish_gradient`` must return a plain tensor.

    The node's input is ``parameter.to_local()``, so its gradient is checked
    against the local shape; a DTensor reports the global one and autograd
    refuses it. Wrapping it looks like supplying the sharding the gradient
    appears to lack, and breaks every run with more than one expert-parallel
    rank while passing every single-rank check, where the shapes coincide.
    """
    stderr = run_ranks("dtensor_grad", expect_failure=True)
    assert "invalid gradient" in stderr, stderr[-2000:]
    # The shapes in the message are what identifies the cause: local vs global.
    assert "[2, 3]" in stderr and "[4, 3]" in stderr, stderr[-2000:]


def test_the_check_would_notice_a_gradient_without_sharding():
    """Guards the guard.

    If the two tests above ever pass for a parameter carrying no sharding
    information, they have stopped being able to detect the thing they exist for.
    """
    records = run_ranks("unsharded")
    norms = {rank: record["norm"] for rank, record in records.items()}

    assert norms[0] != pytest.approx(norms[1], rel=1e-6), (
        "ranks agreed even without sharding information, so the checks above no longer prove anything"
    )
    assert norms[0] == pytest.approx(RANK0_LOCAL_ONLY_NORM, rel=1e-6), (
        f"expected rank 0 to total only its own experts ({RANK0_LOCAL_ONLY_NORM}), got {norms[0]}"
    )
