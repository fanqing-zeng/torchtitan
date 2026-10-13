# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for MoonEP's per-step state and its home-rank gradient reduction.

CPU only, kernels stubbed. Two contracts: streams and events must not be mixed
across API families (the device event bindings cast without checking, so a
generic stream segfaults), and the reduction must run as
stage -> rendezvous -> launch, because the kernel reads every peer's slot
gradients before reaching its own barrier.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# The grouped-GEMM backward test dispatches the CPU reference op, which is
# registered on import (torchtitan vendors only the NPU kernels).
from . import _reference_ops  # noqa: E402,F401
from torch.distributed.tensor import DTensor  # noqa: E402

from torchtitan.distributed.moonep.runtime import (  # noqa: E402
    MoonEPCallState,
    MoonEPSymmetricProjection,
)
from torchtitan.distributed.moonep.settings import MoonEPSettings  # noqa: E402

EXPERTS, EXPERTS_PER_RANK, RANK = 4, 2, 1
OUT_FEATURES, IN_FEATURES = 2, 2
PEER_CONTRIBUTION = 100.0


class _FakeEvent:
    def __init__(self, owner):
        self.owner = owner
        self.waited_on = []

    def wait(self, stream):
        # This is the cast that crashes for real device events; assert the family.
        assert stream.owner is self.owner, "event and stream come from different families"
        self.waited_on.append(stream)


class _FakeStream:
    def __init__(self, owner):
        self.owner = owner
        self.waited_events = []

    def record_event(self):
        return _FakeEvent(self.owner)

    def wait_event(self, event):
        assert event.owner is self.owner, "stream and event come from different families"
        self.waited_events.append(event)

    def synchronize(self):
        """Used by the MOONEP_TRACE_SYNC device checkpoints."""


class _FakeDeviceModule:
    """Stands in for ``torch.cuda`` / ``torch.npu``."""

    def __init__(self):
        self.compute_stream = _FakeStream(self)
        self.entered = 0

    def current_stream(self):
        return self.compute_stream

    def stream(self, target):
        module = self

        @contextmanager
        def _context():
            module.entered += 1
            yield target

        return _context()


def _projection(role="gate_up", transposed=False, **extra):
    """A stand-in with the attributes the adapter touches on a projection."""
    return SimpleNamespace(role=role, transposed=transposed, **extra)


def _call_state(async_finish=True):
    device_module = _FakeDeviceModule()
    comm_stream = _FakeStream(device_module)
    runtime = SimpleNamespace(
        device_module=device_module,
        config=MoonEPSettings(async_finish=async_finish),
        backend=SimpleNamespace(supports_async_comm=True),
        # Trace labels; exercised whenever MOONEP_TRACE is set.
        rank=0,
        call_index=0,
    )
    buffer = SimpleNamespace(_comm_stream=comm_stream)
    state = MoonEPCallState(
        SimpleNamespace(**vars(runtime), should_check_plan_invariants=lambda: False),
        buffer,
        tokens_per_rank=8,
        hidden=4,
        top_k=2,
    )
    return state, device_module, comm_stream


def test_wait_uses_the_device_module_stream():
    state, device_module, comm_stream = _call_state()
    event = comm_stream.record_event()
    state.wait(event)
    assert event.waited_on == [device_module.compute_stream]


def test_start_prefetch_orders_the_comm_stream_behind_compute():
    state, device_module, comm_stream = _call_state()
    state.plan = object()
    recorded = []
    projection = _projection(prefetch=lambda buffer, plan: recorded.append(plan))

    state.start_prefetch(projection)

    # The communication stream waits on a compute-stream event before it
    # overwrites the pooled prefetch slots, then the copy is enqueued on it.
    assert len(comm_stream.waited_events) == 1
    assert device_module.entered == 1
    assert recorded == [state.plan]

    # The prefetch event is handed back to the compute stream, same family.
    state.await_prefetch(projection)


def test_synchronous_backend_skips_stream_bookkeeping():
    state, device_module, _ = _call_state(async_finish=False)
    state.plan = object()
    calls = []
    projection = _projection(prefetch=lambda buffer, plan: calls.append(plan))

    state.start_prefetch(projection)
    state.await_prefetch(projection)

    assert calls == [state.plan]
    assert device_module.entered == 0


def test_adopt_is_a_noop_without_a_communication_stream():
    state, _, _ = _call_state(async_finish=False)
    tensor = torch.zeros(2)
    # record_stream would raise on a CPU tensor; adopt must not reach it.
    state.adopt(tensor, None)


def test_adopt_records_comm_stream_tensors_on_the_compute_stream():
    """Cover the body of adopt, which only a side-stream backend reaches.

    Ascend returns early because it has no usable side stream, and a device run
    with async_finish off returns early too, so these lines execute in exactly
    one configuration. A stand-in tensor is used because record_stream needs a
    real device stream, which is also why the ordering test stubs adopt out --
    between them the body had no coverage at all, and a missing import in it
    went unnoticed until a run with async_finish on hit the NameError.
    """
    state, device_module, _ = _call_state(async_finish=True)
    recorded = []
    tensor = SimpleNamespace(record_stream=recorded.append)
    # The None must be skipped: dispatch passes an absent route tensor this way.
    state.adopt(tensor, None)
    assert recorded == [device_module.compute_stream]


def test_async_dispatch_waits_before_reading_its_outputs():
    """The group offsets must not be read while the planner is still writing.

    They decide the GEMM's memory extents, so an early read is out of bounds
    rather than merely wrong, on an arbitrary subset of ranks.
    """
    from torchtitan.distributed.moonep.autograd import _MoonEPDispatch

    order = []
    state, device_module, comm_stream = _call_state(async_finish=True)
    rows, hidden = 8, 2
    cu_seqlens = torch.tensor([4, 8, 8], dtype=torch.int32)

    class _TracingEvent(_FakeEvent):
        def wait(self, stream):
            super().wait(stream)
            order.append("wait")

    def dispatch(*args, **kwargs):
        assert kwargs["async_finish"] is True
        return (
            torch.zeros(rows, hidden),
            torch.zeros(rows),
            cu_seqlens,
            object(),
            _TracingEvent(device_module),
        )

    state.buffer.dispatch = dispatch
    state.runtime.experts_per_rank = 1
    state.runtime.num_experts = 2
    state.runtime.prefetch_slots = 1
    state.runtime.rank = 0
    state.runtime.backend = SimpleNamespace(
        supports_async_comm=True, expert_weight_layout="full", has_static_tail=False
    )
    state.runtime.should_check_plan_invariants = lambda: False

    original_set_plan = state.set_plan

    def traced_set_plan(*args, **kwargs):
        order.append("set_plan")
        return original_set_plan(*args, **kwargs)

    state.set_plan = traced_set_plan
    # record_stream needs a real device stream; the ordering is what matters here.
    state.adopt = lambda *tensors: order.append("adopt")

    _MoonEPDispatch.forward(SimpleNamespace(), torch.zeros(2, hidden), None, None, None, state)

    # Everything that touches a communication-stream result comes after the wait.
    assert order == ["wait", "adopt", "set_plan"], order
    assert state.group_ends is not None


def _device_module():
    """Minimal stand-in for torch.cuda / torch.npu, as the trace markers use it."""
    return SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None))


def _compact_gradient():
    """``[epn + B, out, in]``: this rank's home rows then its slot rows."""
    rows = EXPERTS_PER_RANK + EXPERTS_PER_RANK
    return torch.arange(rows * OUT_FEATURES * IN_FEATURES, dtype=torch.bfloat16).reshape(
        rows, OUT_FEATURES, IN_FEATURES
    )


def _runtime(backend, gradient_divide_factor=1.0):
    return SimpleNamespace(
        backend=backend,
        num_experts=EXPERTS,
        rank=RANK,
        experts_per_rank=EXPERTS_PER_RANK,
        prefetch_slots=EXPERTS_PER_RANK,
        config=MoonEPSettings(num_sms=8),
        gradient_divide_factor=gradient_divide_factor,
        # Trace labels; read whenever MOONEP_TRACE is set.
        call_index=0,
        device_module=_device_module(),
    )


def test_projection_splits_compact_gradient_and_returns_owned_local_result():
    compact = _compact_gradient()
    seen = {}

    def stage_gradient(runtime, resources, home_grad, slot_grad):
        seen["home"] = home_grad.clone()
        seen["slot"] = slot_grad.clone()

    def reduce_staged_gradient(runtime, resources, buffer, plan):
        # Simulate the peers' contributions landing on this rank's experts.
        return seen["home"].float() + PEER_CONTRIBUTION

    backend = SimpleNamespace(
        stage_gradient=stage_gradient,
        reduce_staged_gradient=reduce_staged_gradient,
        rendezvous_before_reduce=lambda runtime, buffer: None,
    )
    projection = object.__new__(MoonEPSymmetricProjection)
    projection.runtime = _runtime(backend)
    projection.resources = SimpleNamespace()

    projection.stage_gradient(compact)
    local_grad = projection.finish_gradient(buffer=object(), plan=object())

    # The home half and the slot half are handed over unmodified, and together
    # they cover the whole compact gradient.
    torch.testing.assert_close(seen["home"], compact[:EXPERTS_PER_RANK], rtol=0, atol=0)
    torch.testing.assert_close(seen["slot"], compact[EXPERTS_PER_RANK : 2 * EXPERTS_PER_RANK], rtol=0, atol=0)
    assert seen["home"].shape[0] + seen["slot"].shape[0] == compact.shape[0]

    expected = (compact[:EXPERTS_PER_RANK].float() + PEER_CONTRIBUTION).to(torch.bfloat16)
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)
    assert local_grad.dtype == torch.bfloat16
    # The pooled FP32 buffer must not be aliased into the autograd graph.
    assert local_grad.untyped_storage().nbytes() == local_grad.numel() * local_grad.element_size()
    # Deliberately a plain tensor, not a DTensor over the expert-parallel mesh.
    # It is returned as the gradient of ``parameter.to_local()``, so it is checked
    # against the local shape, and that call's own backward is what restores the
    # parameter's placements. Wrapping it here looks like it supplies sharding
    # information the gradient is missing, and instead breaks every run with more
    # than one expert-parallel rank; see test_grad_norm_agreement.py.
    assert isinstance(local_grad, torch.Tensor) and not isinstance(local_grad, DTensor), (
        f"finish_gradient returned {type(local_grad).__name__}; it must stay a plain local-shaped tensor"
    )


def test_home_rank_reduction_applies_the_gradient_divide_factor():
    """MoonEP must average over the EP group the way expert FSDP does.

    An expert's gradient collects tokens from every EP rank, and those ranks hold
    different micro-batches, so the home-rank reduction's sum has to be divided
    the same way expert FSDP's reduce-scatter divides it for the other
    dispatchers. MoonEP replaces that reduce-scatter, so nothing else applies the
    factor.

    Measured on a real 4-rank run before this was applied: the expert gradient
    norm was exactly 4x the fused dispatcher's on step 0, where the weights are
    still identical. Adam's second moment hides most of a constant gradient
    scale, which is why the losses still tracked; weight decay and Trainer's
    clip-by-norm do not.
    """
    factor = 4.0
    reduced = torch.full((EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES), 8.0)
    backend = SimpleNamespace(
        rendezvous_before_reduce=lambda runtime, buffer: None,
        reduce_staged_gradient=lambda runtime, resources, buffer, plan: reduced,
    )
    projection = object.__new__(MoonEPSymmetricProjection)
    projection.runtime = _runtime(backend, gradient_divide_factor=factor)
    projection.resources = SimpleNamespace()
    projection._staged_dtype = torch.float32

    result = projection.finish_gradient(buffer=object(), plan=object())

    torch.testing.assert_close(result, torch.full_like(result, 8.0 / factor), rtol=0, atol=0)


def test_bridge_reduces_down_before_gate_up():
    """The reduction order is fixed across ranks.

    Every MoonEP barrier shares one slot and generation counter, so all ranks
    must arrive in the same sequence. The staging itself happens earlier, in the
    expert GEMM backwards, which autograd guarantees run before this node.
    """
    from torchtitan.distributed.moonep.autograd import _MoonEPWeightsBridge

    order = []

    class _Projection:
        def __init__(self, name):
            self.name = name

        def finish_gradient(self, buffer, plan):
            order.append(self.name)
            return torch.zeros(EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES)

    ctx = SimpleNamespace(
        down_projection=_Projection("down"),
        gate_up_projection=_Projection("gate_up"),
        state=SimpleNamespace(
            runtime=SimpleNamespace(rank=RANK, call_index=0, device_module=_device_module()),
            buffer=None,
            plan=None,
        ),
    )

    grad_gate_up, grad_down = _MoonEPWeightsBridge.backward(ctx, torch.zeros(()), torch.zeros(()))[:2]

    assert order == ["down", "gate_up"], order
    assert grad_gate_up.shape == (EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES)
    assert grad_down.shape == (EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES)


def test_projection_rendezvouses_immediately_before_each_reduction():
    """One cross-rank sync per reduction, in that order.

    The reduction kernel reads every peer's slot gradients before its own
    internal barrier, so a rank that launches early reads buffers its peers have
    not written yet.
    """
    order = []
    backend = SimpleNamespace(
        rendezvous_before_reduce=lambda runtime, buffer: order.append("rendezvous"),
        reduce_staged_gradient=lambda runtime, resources, buffer, plan: (
            order.append("reduce") or torch.zeros(EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES)
        ),
    )
    projection = object.__new__(MoonEPSymmetricProjection)
    projection.runtime = _runtime(backend)
    projection.resources = SimpleNamespace()
    projection._staged_dtype = torch.bfloat16

    result = projection.finish_gradient(buffer=object(), plan=object())

    assert order == ["rendezvous", "reduce"]
    assert result.dtype == torch.bfloat16


def test_gemm_backward_stages_the_weight_gradient_instead_of_returning_it():
    """The weight gradient's shape is unrelated to the GEMM's weight operand.

    The forward indexes the global ``[E+B]`` mapping while the gradient covers
    only the ``epn+B`` groups this rank computed, so the weight cannot be an
    autograd input. The GEMM stages the gradient and returns ``None`` for the
    ordering marker.
    """
    from torchtitan.distributed.moonep.autograd import _MoonEPGateUpGmm

    staged = []
    rows, in_features, out_features = 8, 4, 6
    ends = torch.tensor([4, rows], dtype=torch.int32)
    weight = torch.zeros(2, out_features, in_features)

    def stage_gradient(grad):
        staged.append(grad)

    def wgrad_group_ends_host():
        return ends.tolist()

    def await_prefetch(_projection):
        return None

    projection = _projection(
        gemm_weight=weight,
        stage_gradient=stage_gradient,
    )
    state = SimpleNamespace(
        group_ends=ends,
        wgrad_group_ends=ends,
        wgrad_group_ends_host=wgrad_group_ends_host,
        await_prefetch=await_prefetch,
        runtime=SimpleNamespace(config=MoonEPSettings(), rank=RANK, call_index=0, device_module=_device_module()),
    )
    ctx = SimpleNamespace(
        state=state,
        projection=projection,
        recompute_dispatch=False,
        saved_tensors=(torch.zeros(rows, in_features),),
    )

    grads = _MoonEPGateUpGmm.backward(ctx, torch.zeros(rows, out_features))

    assert len(staged) == 1
    assert staged[0].shape == weight.shape
    assert grads[0].shape == (rows, in_features)
    assert grads[1] is None, "the marker must not receive a gradient"
