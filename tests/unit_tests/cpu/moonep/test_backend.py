# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for MoonEP's backend selection, alignment and locality guards.

These run on any machine: they exercise the pure-Python guards that must fail
fast *before* MoonEP allocates symmetric memory, which is where a
misconfiguration is cheap to diagnose. The device paths themselves are covered
by the end-to-end NPU runs.
"""

import pytest
import torch

from torchtitan.distributed.moonep.backend import (
    get_moonep_backend,
    validate_chunk_granularity,
)

GRANULARITY = 2 * 1024 * 1024


def test_aligned_block_is_accepted():
    # 4 experts/rank x 512 x 512 x 4 bytes = 4 MiB, two granules.
    validate_chunk_granularity("down", 4, 512, 512, torch.float32.itemsize, GRANULARITY)


def test_unaligned_block_is_rejected_with_actionable_message():
    with pytest.raises(RuntimeError) as error:
        # 2 experts/rank x 256 x 256 x 2 bytes = 256 KiB, below one granule.
        validate_chunk_granularity("gate_up", 2, 256, 256, torch.bfloat16.itemsize, GRANULARITY)
    message = str(error.value)
    assert "262144 bytes" in message
    assert str(GRANULARITY) in message
    assert "granularity aligned" in message


@pytest.mark.parametrize(
    "experts_per_rank,out_features,in_features",
    [(4, 1024, 512), (8, 512, 512), (4, 512, 512)],
)
def test_reference_test_shapes_are_aligned(experts_per_rank, out_features, in_features):
    """The shapes used by the end-to-end NPU test must satisfy the contract."""
    for item_size in (torch.bfloat16.itemsize, torch.float32.itemsize):
        validate_chunk_granularity("test shape", experts_per_rank, out_features, in_features, item_size, GRANULARITY)


def test_unknown_device_type_is_rejected():
    with pytest.raises(RuntimeError, match="No MoonEP backend for device type"):
        get_moonep_backend("xpu")


def ep_group_host_count(ranks, local_world_size, monkeypatch):
    """Drive the locality probe's environment path with a stubbed process group."""
    from types import SimpleNamespace

    module = pytest.importorskip("torchtitan.distributed.moonep.backend_npu")
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: ranks, raising=False)
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
    return module._ep_group_host_count(SimpleNamespace(group=object()))


@pytest.mark.parametrize(
    "ranks,local_world_size,expected",
    [
        # One node, EP spans it entirely.
        ([0, 1, 2, 3], 4, 1),
        # Two nodes of four; EP is confined to the second one.
        ([4, 5, 6, 7], 4, 1),
        # EP straddles the node boundary, which needs the ip/port bootstrap.
        ([2, 3, 4, 5], 4, 2),
        # Eight ranks over two nodes.
        ([0, 1, 2, 3, 4, 5, 6, 7], 4, 2),
        # One rank per node.
        ([0, 1], 1, 2),
    ],
)
def test_host_count_comes_from_the_launcher_environment(ranks, local_world_size, expected, monkeypatch):
    """Node membership must be derived without a collective.

    The first collective on a subgroup is what creates that subgroup's HCCL
    communicator, and each communicator binds an NPU socket. Deriving locality
    from the environment keeps a communicator-setup failure out of what is only
    a sanity check.
    """
    assert ep_group_host_count(ranks, local_world_size, monkeypatch) == expected


@pytest.mark.parametrize("value", ["", "0", "not-a-number"])
def test_unusable_local_world_size_falls_back_to_the_collective(value, monkeypatch):
    """A malformed environment must not be silently believed."""
    from types import SimpleNamespace

    module = pytest.importorskip("torchtitan.distributed.moonep.backend_npu")
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: [0, 1], raising=False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", value)
    reached = {}

    def fake_all_gather(output, input_tensor, group=None):
        reached["called"] = True
        output.fill_(7)

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", fake_all_gather, raising=False)
    monkeypatch.setattr(module.torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    # Capture the real constructor first: the patch below replaces the global
    # torch.device, so referring to it by name inside the replacement recurses.
    real_device = torch.device
    monkeypatch.setattr(module.torch, "device", lambda *args, **kwargs: real_device("cpu"), raising=False)

    assert module._ep_group_host_count(SimpleNamespace(group=object())) == 1
    assert reached.get("called"), "the collective fallback was not used"


def test_missing_ascend_library_reports_how_to_build_it():
    """Without ``ascend_moonep`` the NPU backend must say what to install."""
    pytest.importorskip("torchtitan.distributed.moonep.backend_npu")
    try:
        import ascend_moonep  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="ascend_moonep"):
            get_moonep_backend("npu")
    else:
        pytest.skip("ascend_moonep is installed; the import guard cannot be exercised")
