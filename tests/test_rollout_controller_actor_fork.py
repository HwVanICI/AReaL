# SPDX-License-Identifier: Apache-2.0

"""Step-0 actor-GPU inference fork: lifecycle and reclaim state machine.

The forked vLLM engines run on the actor GPUs during the first rollout, are
appended to ``self.workers`` so round-robin dispatch serves them, then
reclaimed: in-flight requests are evicted, their partial tokens surface as
continuation descriptors, and reclaim re-submits continuations to surviving
engines before tearing the fork down.

These tests exercise only the branch logic via a minimal recording stand-in
(no cluster / no real scheduler), mirroring ``test_rollout_controller_proxy_pause``.
"""

import threading

import pytest
import torch

from areal.infra.controller.fork_coordinator import (
    ForkCoordinator,
    NoOpForkCoordinator,
)
from areal.infra.controller.rollout_controller import (
    _FORK_TASK_ABANDONED,
    FORK_CONTINUATION_KEY,
    FORK_CONTINUE_FIELD,
    RolloutController,
    _RemoteRolloutTaskInput,
)
from areal.infra.workflow_executor import (
    WorkflowExecutor,
    _RolloutResult,
    _RolloutTaskInput,
)
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.fork_constants import FORK_COMPLETE_KEY, FORK_MERGE_KEY
from areal.utils.fork_continuation import ForkContinuationWorkflow
from areal.workflow.rlvr import RLVRWorkflow


class _RecordingDispatcher:
    def __init__(self, calls):
        self._calls = calls

    def pause(self):
        self._calls.append("dispatcher.pause")

    def resume(self):
        self._calls.append("dispatcher.resume")

    def submit_task_input(self, task_input):
        self._calls.append(f"dispatcher.submit:{task_input.task_id}")


class _FakeWorker:
    def __init__(self, wid):
        self.id = wid


class _RecordingController:
    """Minimal stand-in exercising the fork lifecycle + reclaim branches."""

    def __init__(self, surviving_workers, fork_workers):
        self.calls: list[str] = []
        self.workers = list(surviving_workers)
        self._current_worker_idx = 0
        self.fork_workers = list(fork_workers)
        self._fork_started = True
        self._fork_role = "actor-fork"
        self._fork_worker_ids = {w.id for w in fork_workers}
        self._task_worker_map: dict[int, str] = {}
        self._task_input_map: dict[int, _RemoteRolloutTaskInput] = {}
        self._fork = ForkCoordinator(host=self)
        self._worker_engine_names: dict[str, str] = {}
        self._task_worker_map_lock = threading.Lock()
        self._pending_futures: dict[int, object] = {}
        self._futures_lock = threading.Lock()

        class _Cfg:
            request_timeout = 60.0
            pause_grace_period = 0.0

        self.config = _Cfg()

        class _Dispatcher(_RecordingDispatcher):
            pass

        self.dispatcher = _Dispatcher(self.calls)

        class _Scheduler:
            def delete_workers(self_inner, role=None):
                self.calls.append(f"scheduler.delete_workers:{role}")

        self.scheduler = _Scheduler()

    # ---- methods under test, bound from RolloutController ----
    _fork_engine_name = RolloutController._fork_engine_name

    def _fork_collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"fork.{method}")
        if method == "evict":
            # Simulate the real evict: fork agenerate loops break, return
            # partial descriptors; coroutines deposit them and pop their
            # _task_worker_map entries.
            fork_tids = [
                tid
                for tid, wid in list(self._task_worker_map.items())
                if wid in self._fork_worker_ids
            ]
            for tid in fork_tids:
                self._task_worker_map.pop(tid, None)
                self._task_input_map.pop(tid, None)
                self._fork._fork_continuations[tid] = {
                    FORK_CONTINUATION_KEY: [
                        {
                            "prompt_ids": [1, 2],
                            "partial_ids": [3],
                            "partial_logprobs": [0.1],
                            "partial_versions": [0],
                            "rid": f"rid-{tid}",
                        }
                    ]
                }

    def _fork_collective_rpc_async(self, method, *args, **kwargs):
        self.calls.append(f"fork_async.{method}")

    def submit(self, **kwargs):
        self.calls.append(f"controller.submit:{kwargs['task_id']}")
        assert kwargs.get("priority") is True, (
            "fork re-submits (continuation / restart) must be priority "
            "(bypass staleness)"
        )
        if FORK_CONTINUE_FIELD in kwargs["data"]:
            return
        # Full restart of a lost fork task: the original input is replayed
        # untouched (no continuation fields).
        assert FORK_MERGE_KEY not in kwargs["data"]

    # Bound under the original method name (mirrors _fork_engine_name above)
    # so ``controller.reclaim_and_teardown_actor_fork()`` resolves.
    reclaim_and_teardown_actor_fork = RolloutController.reclaim_and_teardown_actor_fork


def test_reclaim_skips_when_not_started():
    """Test that reclaim is a no-op when no fork engines are active."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")], fork_workers=[]
    )
    controller._fork_started = False

    controller.reclaim_and_teardown_actor_fork()

    assert controller.calls == []


def test_reclaim_evicts_and_resubmits_continuations():
    """Test that reclaim evicts fork, collects partials, and re-submits
    continuations to surviving engines; surviving engines are never paused.

    Two in-flight tasks sit on fork engines; reclaim must:
    pause dispatcher -> fork evict -> (settle) -> remove fork workers ->
    destroy + delete -> re-submit continuations (with __continue__) to
    surviving -> resume dispatcher. No pause_generation on surviving.
    """
    surviving = [_FakeWorker("rollout/0"), _FakeWorker("rollout/1")]
    fork = [_FakeWorker("actor-fork/0"), _FakeWorker("actor-fork/1")]
    controller = _RecordingController(surviving, fork)

    # Two in-flight tasks on fork engines.
    t0 = _RemoteRolloutTaskInput(
        task_id=0,
        data={"messages": "a"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=1,
    )
    t1 = _RemoteRolloutTaskInput(
        task_id=1,
        data={"messages": "b"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=1,
    )
    controller._task_worker_map = {0: "actor-fork/0", 1: "actor-fork/1"}
    controller._task_input_map = {0: t0, 1: t1}

    class _Gen:
        def __init__(self):
            self._n = 100

        def next(self):
            self._n += 1
            return self._n

    controller._task_id_generator = _Gen()

    controller.reclaim_and_teardown_actor_fork()

    # 1. dispatcher paused first.
    assert controller.calls[0] == "dispatcher.pause"
    # 2. fork evict (not pause_generation) — aborts in-flight, deposits partials.
    #    (_fork_collective_rpc appends "fork.evict"; `in` is exact membership.)
    assert "fork.evict" in controller.calls
    # 3. fork engines destroyed + fork workers deleted.
    assert "fork.destroy" in controller.calls
    assert "scheduler.delete_workers:actor-fork" in controller.calls
    # 4. continuations re-submitted to surviving engines (with __continue__).
    assert any(c.startswith("controller.submit:") for c in controller.calls)
    # 5. dispatcher resumed last.
    assert controller.calls[-1] == "dispatcher.resume"
    # 6. fork workers removed from the dispatch pool.
    assert all(w.id.startswith("rollout") for w in controller.workers)
    # 7. fork state cleared.
    assert controller._fork_started is False
    assert controller.fork_workers == []
    # 8. surviving engines never pause_generation (point 3: only fork interrupted).
    assert not any(c == "fork.pause_generation" for c in controller.calls)


def test_reclaim_preserves_original_data_in_continuation():
    """Test that re-submitted continuations carry the original task data
    (for reward kwargs) plus the per-sample partials under __continue__."""
    surviving = [_FakeWorker("rollout/0")]
    fork = [_FakeWorker("actor-fork/0")]
    controller = _RecordingController(surviving, fork)

    orig = _RemoteRolloutTaskInput(
        task_id=5,
        data={"messages": "hello"},
        workflow="my.wf",
        workflow_kwargs={"k": 1},
        should_accept_fn=None,
        group_size=2,
    )
    controller._task_worker_map = {5: "actor-fork/0"}
    controller._task_input_map = {5: orig}

    submitted: list[dict] = []

    def capture_submit(**kwargs):
        submitted.append(kwargs)
        controller.calls.append(f"controller.submit:{kwargs['task_id']}")

    controller.submit = capture_submit

    class _Gen:
        def __init__(self):
            self._n = 99

        def next(self):
            self._n += 1
            return self._n

    controller._task_id_generator = _Gen()

    controller.reclaim_and_teardown_actor_fork()

    assert len(submitted) == 1
    sub = submitted[0]
    # Original data preserved (reward kwargs), and __continue__ injected.
    assert sub["data"]["messages"] == "hello"
    assert FORK_CONTINUE_FIELD in sub["data"]
    # group_size matches the number of per-sample partials.
    assert sub["group_size"] == len(sub["data"][FORK_CONTINUE_FIELD])


# ---------------------------------------------------------------------------
# Helpers for the Bug B (partial-evict re-merge) and bug#7 (group_size == 1)
# determinism tests. These construct the exact descriptors / trajectory dicts
# the fork salvage path exchanges, so the count-consistency and list->dict
# unpack contracts are checked without a cluster (the two NPU runs only ever
# hit the all-in-flight case, so these paths are otherwise unexercised).
# ---------------------------------------------------------------------------

N_SAMPLES = 8


def _fake_traj(n_rollouts: int, seqlen: int = 16, rid_prefix: str = "r") -> dict:
    """Minimal trajectory dict with the keys ``concat_padded_tensors`` merges.

    All fake trajectories share the same key set so concat does not reject them
    (concat_padded_tensors requires consistent keys across inputs).
    """
    return {
        "input_ids": torch.arange(n_rollouts * seqlen, dtype=torch.long).reshape(
            n_rollouts, seqlen
        ),
        "attention_mask": torch.ones((n_rollouts, seqlen), dtype=torch.long),
        "token_logprobs": torch.zeros((n_rollouts, seqlen)),
        "rids": [f"{rid_prefix}-{i}" for i in range(n_rollouts)],
    }


def _descriptor(rid: str = "rid-x", partial_len: int = 3) -> dict:
    """One per-sample continuation descriptor (as surfaced by the fork engine
    when an in-flight rollout is aborted mid-generation)."""
    return {
        "prompt_ids": [1, 2],
        "partial_ids": list(range(partial_len)),
        "partial_logprobs": [0.1 * i for i in range(partial_len)],
        "partial_versions": [0] * partial_len,
        "rid": rid,
    }


class _NoOpLogger:
    """Logger stand-in for the static ForkContinuationWorkflow helpers."""

    def info(self, *args, **kwargs):
        pass


class _RecordingBase:
    """Stand-in for RLVRWorkflow: records the ``cont`` handed to the resume
    path so route_continuation's group_size == 1 list->dict unpack is observable
    without running a real engine."""

    def __init__(self):
        self.received_cont = None
        self.received_data = None
        self.calls = 0

    async def _arun_episode_continue(self, engine, data, cont):
        self.received_cont = cont
        self.received_data = data
        self.calls += 1
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}


# ---------------------------------------------------------------------------
# Bug B: partial-evict re-merge (K completed parked + (8-K) continued -> 8)
# ---------------------------------------------------------------------------


def test_on_task_evicted_parks_complete_group_for_remerge():
    """Test that on_task_evicted parks the already-complete half of a
    partially-evicted group and pops the task maps, so reclaim can later
    re-merge it with the continuation. Bug B deposit step."""
    # Arrange: one fork task, K=3 of its 8 rollouts already complete at evict.
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    tid = 10
    controller._task_worker_map = {tid: "actor-fork/0"}
    controller._task_input_map = {
        tid: _RemoteRolloutTaskInput(
            task_id=tid,
            data={"messages": "p"},
            workflow="wf",
            workflow_kwargs={},
            should_accept_fn=None,
            group_size=N_SAMPLES,
        )
    }
    k_complete = 3
    traj = {
        FORK_CONTINUATION_KEY: [
            _descriptor(f"rid-{tid}-{i}") for i in range(N_SAMPLES - k_complete)
        ],
        FORK_COMPLETE_KEY: _fake_traj(k_complete, rid_prefix="done"),
    }

    # Act
    controller._fork.on_task_evicted(tid, _FakeWorker("actor-fork/0"), traj)

    # Assert: completed half parked (localized -> local tensor dict).
    assert tid in controller._fork._fork_complete_groups
    assert (
        controller._fork._fork_complete_groups[tid]["input_ids"].shape[0] == k_complete
    )
    # Descriptor deposited for reclaim to re-submit.
    assert tid in controller._fork._fork_continuations
    # Task popped from worker/input maps so reclaim's settle sees it done.
    assert tid not in controller._task_worker_map
    assert tid not in controller._task_input_map


@pytest.mark.parametrize("k_complete", range(1, N_SAMPLES))
def test_try_remerge_partial_evict_count_is_n_samples(k_complete):
    """Test that re-merging K parked completed + (8-K) continued yields exactly
    8 rollouts for every K in 1..7. Bug B count-consistency guard: a merged
    block of != 8 silently misaligns reward_norm's positional 8-slicer for every
    subsequent prompt in the batch (wrong advantages, no crash). K=7 is the
    bug#7 (group_size == 1) re-merge shape: a 7-group concat with a 1-single."""
    # Arrange
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    fork_tid = 42
    controller._fork._fork_complete_groups[fork_tid] = _fake_traj(
        k_complete, rid_prefix="done"
    )
    pending = _RemoteRolloutTaskInput(
        task_id=999,
        data={FORK_MERGE_KEY: fork_tid, "messages": "p"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=N_SAMPLES - k_complete,
    )
    continued = _fake_traj(N_SAMPLES - k_complete, rid_prefix="cont")

    # Act
    merged = controller._fork.try_remerge(pending, continued)

    # Assert: exactly one full n_samples group (no silent misalignment).
    assert merged["input_ids"].shape[0] == N_SAMPLES
    assert len(merged["rids"]) == N_SAMPLES
    # Parked complete half consumed (not double-merged on a second completion).
    assert fork_tid not in controller._fork._fork_complete_groups


def test_try_remerge_without_merge_key_is_noop():
    """Test that a normal (non-continuation) task without FORK_MERGE_KEY is
    returned unchanged -- re-merge never fires on regular surviving-engine
    trajectories."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[],
    )
    pending = _RemoteRolloutTaskInput(
        task_id=7,
        data={"messages": "plain"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=N_SAMPLES,
    )
    traj = _fake_traj(N_SAMPLES)

    merged = controller._fork.try_remerge(pending, traj)

    assert merged is traj


def test_try_remerge_all_in_flight_no_complete_group_is_noop():
    """Test that an all-in-flight continuation (8/8 evicted, no completed half
    parked) is accepted as-is -- the continued result is already a full group.
    This is the path both NPU runs actually exercised (112 + 120 continuations);
    it must stay a no-op so re-merge never double-concats a full group."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[],
    )
    fork_tid = 5
    # FORK_MERGE_KEY present (it IS a continuation) but no completed half parked.
    pending = _RemoteRolloutTaskInput(
        task_id=999,
        data={FORK_MERGE_KEY: fork_tid, "messages": "p"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=N_SAMPLES,
    )
    traj = _fake_traj(N_SAMPLES)

    merged = controller._fork.try_remerge(pending, traj)

    # No complete group to pop -> accept the 8-rollout continuation as-is.
    assert merged is traj
    assert merged["input_ids"].shape[0] == N_SAMPLES


@pytest.mark.parametrize("k_complete", range(1, N_SAMPLES))
def test_surface_group_descriptor_partial_evict_surfaces_both(k_complete):
    """Test that a partially-evicted group surfaces BOTH the (8-K) in-flight
    descriptors AND the K completed rollouts (merged into one group) so the
    controller can park + re-merge them. Bug B surfacing step (runs on the fork
    engine inside GroupedRolloutWorkflow.gather)."""
    # Arrange: K completed trajectories + (8-K) abort descriptors.
    complete_results = [
        _fake_traj(1, rid_prefix=f"done-{i}") for i in range(k_complete)
    ]
    descriptor_results = [
        {FORK_CONTINUATION_KEY: _descriptor(f"rid-{i}")}
        for i in range(N_SAMPLES - k_complete)
    ]
    results = complete_results + descriptor_results

    out = ForkContinuationWorkflow.surface_group_descriptor(results, _NoOpLogger())

    assert out is not None
    # (8-K) in-flight descriptors surfaced for continuation re-submit.
    assert len(out[FORK_CONTINUATION_KEY]) == N_SAMPLES - k_complete
    # K completed rollouts merged into one group for parking.
    assert out[FORK_COMPLETE_KEY]["input_ids"].shape[0] == k_complete


def test_surface_group_descriptor_all_complete_returns_none():
    """Test that a group with no in-flight descriptors (all 8 completed -> path
    A) returns None so the normal merge path runs (no spurious descriptor)."""
    results = [_fake_traj(1, rid_prefix=f"done-{i}") for i in range(N_SAMPLES)]
    assert (
        ForkContinuationWorkflow.surface_group_descriptor(results, _NoOpLogger())
        is None
    )


def test_surface_group_descriptor_all_in_flight_no_complete():
    """Test that an all-in-flight group (8/8 descriptors, 0 completed) surfaces
    the 8 descriptors with FORK_COMPLETE_KEY=None -- nothing to park/re-merge."""
    results = [
        {FORK_CONTINUATION_KEY: _descriptor(f"rid-{i}")} for i in range(N_SAMPLES)
    ]
    out = ForkContinuationWorkflow.surface_group_descriptor(results, _NoOpLogger())
    assert out is not None
    assert len(out[FORK_CONTINUATION_KEY]) == N_SAMPLES
    assert out[FORK_COMPLETE_KEY] is None


# ---------------------------------------------------------------------------
# bug#7: group_size == 1 continuation. The per-sample list arrives UNWRAPPED at
# RLVRWorkflow because the engine's _resolve_workflow only wraps
# GroupedRolloutWorkflow for group_size > 1. route_continuation must unpack the
# 1-element list to a dict before calling _arun_episode_continue, else
# cont["prompt_ids"] raises TypeError and the rollout is silently dropped.
# The group_size == 8 path (wrapped, each sample gets a dict) never exercises
# this list branch -- both NPU runs only ran the dict branch (232 continuations).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_continuation_group_size_one_unwraps_list():
    """Test that a group_size == 1 continuation (a 1-element list, unwrapped by
    the engine) is unpacked to the sole dict before _arun_episode_continue --
    the bug#7 fix. Without it, _arun_episode_continue receives a list and
    cont["prompt_ids"] raises TypeError, silently dropping the rollout."""
    sole = _descriptor("rid-single")
    data = {FORK_CONTINUE_FIELD: [sole]}  # 1-element list (group_size == 1)
    base = _RecordingBase()

    handled, _result = await ForkContinuationWorkflow.route_continuation(
        base, engine=None, data=data
    )

    assert handled is True
    assert base.calls == 1
    # _arun_episode_continue received the DICT, not the list.
    assert base.received_cont is sole
    assert isinstance(base.received_cont, dict)


@pytest.mark.asyncio
async def test_route_continuation_dict_passes_through():
    """Test that an already-unwrapped dict continuation (group_size > 1, each
    sample injected by arun_one_with_continue) is routed unchanged. This is the
    path both NPU runs exercised -- it must keep working alongside the fix."""
    cont = _descriptor("rid-wrapped")
    data = {FORK_CONTINUE_FIELD: cont}
    base = _RecordingBase()

    handled, _ = await ForkContinuationWorkflow.route_continuation(
        base, engine=None, data=data
    )

    assert handled is True
    assert base.calls == 1
    assert base.received_cont is cont


@pytest.mark.asyncio
async def test_route_continuation_empty_list_not_handled():
    """Test that an empty continuation list is treated as no continuation (cont
    becomes None) so normal generation proceeds rather than indexing into []."""
    data = {FORK_CONTINUE_FIELD: []}
    base = _RecordingBase()

    handled, result = await ForkContinuationWorkflow.route_continuation(
        base, engine=None, data=data
    )

    assert handled is False
    assert result is None
    assert base.calls == 0


@pytest.mark.asyncio
async def test_route_continuation_no_field_not_handled():
    """Test that data without FORK_CONTINUE_FIELD is not a continuation --
    normal generation runs (the common case for every non-fork rollout)."""
    base = _RecordingBase()

    handled, result = await ForkContinuationWorkflow.route_continuation(
        base, engine=None, data={"messages": "plain"}
    )

    assert handled is False
    assert result is None
    assert base.calls == 0


# ---------------------------------------------------------------------------
# post_process_result hook (Stage 1 composition refactor): the controller
# callback's per-task fork salvage is now one call on the fork strategy.
# NoOpForkCoordinator (feature off) is identity; ForkCoordinator (feature on)
# deposits descriptors, re-merges, localizes.
# ---------------------------------------------------------------------------


def test_noop_post_process_result_is_identity():
    """Test that NoOpForkCoordinator.post_process_result is identity for any
    input (normal traj / None / descriptor) -- zero fork overhead in the hot
    path when the feature is off. (A descriptor never reaches a NoOp controller
    in practice -- no fork engines -> no eviction -- but identity is the
    contract.)"""
    noop = NoOpForkCoordinator()
    traj = _fake_traj(N_SAMPLES)
    assert noop.post_process_result(None, _FakeWorker("rollout/0"), 1, traj) is traj
    assert noop.post_process_result(None, _FakeWorker("rollout/0"), 1, None) is None
    desc = {FORK_CONTINUATION_KEY: [_descriptor("rid")]}
    assert noop.post_process_result(None, _FakeWorker("rollout/0"), 1, desc) is desc


def test_noop_lifecycle_methods_are_noop():
    """Test that NoOpForkCoordinator's reclaim/destroy surface is inert."""
    noop = NoOpForkCoordinator()
    assert noop.pop_continuation(1) is None
    assert noop.pop_completed_task_ids() == {}
    noop.clear_continuations()  # must not raise
    noop.clear_completed_task_ids()  # must not raise
    noop.clear_complete_groups()  # must not raise
    assert noop.is_fork_worker("any-worker") is False


def test_post_process_result_descriptor_deposits_and_returns_none():
    """Test that post_process_result deposits an evicted descriptor (parks the
    complete half via on_task_evicted + pops the task maps) and returns None so
    the controller rejects and reclaim re-submits a continuation."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    tid = 10
    controller._task_worker_map = {tid: "actor-fork/0"}
    controller._task_input_map = {
        tid: _RemoteRolloutTaskInput(
            task_id=tid,
            data={"messages": "p"},
            workflow="wf",
            workflow_kwargs={},
            should_accept_fn=None,
            group_size=N_SAMPLES,
        )
    }
    descriptor = {
        FORK_CONTINUATION_KEY: [_descriptor(f"rid-{tid}-{i}") for i in range(5)],
        FORK_COMPLETE_KEY: _fake_traj(3, rid_prefix="done"),
    }

    out = controller._fork.post_process_result(
        pending_task=controller._task_input_map[tid],
        worker=_FakeWorker("actor-fork/0"),
        task_id=tid,
        traj=descriptor,
    )

    # Returns None -> controller rejects (reclaim re-submits a continuation).
    assert out is None
    # Complete half parked for re-merge; task maps popped inside on_task_evicted.
    assert tid in controller._fork._fork_complete_groups
    assert controller._fork._fork_complete_groups[tid]["input_ids"].shape[0] == 3
    assert tid not in controller._task_worker_map
    assert tid not in controller._task_input_map


def test_post_process_result_none_returns_none():
    """Test that a None result (failed/empty rollout) returns None for reject,
    with no salvage side effects."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[],
    )
    out = controller._fork.post_process_result(
        pending_task=None,
        worker=_FakeWorker("rollout/0"),
        task_id=1,
        traj=None,
    )
    assert out is None


def test_post_process_result_normal_traj_is_identity_when_no_fork():
    """Test that a normal trajectory on a surviving engine (no FORK_MERGE_KEY,
    worker not a fork worker) is returned unchanged -- try_remerge +
    localize_if_fork are no-ops when the feature has no fork workers."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[],
    )
    traj = _fake_traj(N_SAMPLES)
    pending = _RemoteRolloutTaskInput(
        task_id=1,
        data={"messages": "plain"},
        workflow="wf",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=N_SAMPLES,
    )

    out = controller._fork.post_process_result(
        pending_task=pending,
        worker=_FakeWorker("rollout/0"),
        task_id=1,
        traj=traj,
    )

    assert out is traj


# ---------------------------------------------------------------------------
# Descriptor short-circuit in WorkflowExecutor._execute_workflow: a fork-eviction
# descriptor returned by arun_episode must bypass check_trajectory_format +
# should_accept_fn + dump (which would reject/raise on the non-trajectory shape
# and silently lose the partial). Otherwise, with a non-None dynamic_filter_fn,
# the descriptor is rejected at the engine -> controller sees None -> the
# continuation is never deposited -> batch short + in-flight rollouts lost.
# ---------------------------------------------------------------------------


class _FakeWorkflow:
    """Workflow whose arun_episode returns a fixed result (descriptor or traj)."""

    def __init__(self, result):
        self._result = result

    async def arun_episode(self, engine, data):
        return self._result


class _FakeWorkflowExecutor:
    """Minimal stand-in for WorkflowExecutor -- only the attributes the
    _execute_workflow descriptor short-circuit + normal-reject path touch
    (staleness_manager, inference_engine, config). Binds the real
    _create_workflow_task so the production closure (with the short-circuit)
    runs against the fake self."""

    def __init__(self):
        self.inference_engine = None

        class _M:
            def on_rollout_accepted(self):
                pass

            def on_rollout_rejected(self):
                pass

        self.staleness_manager = _M()

        class _Cfg:
            check_trajectory_format = False
            dump_to_file = False
            enable_rollout_tracing = False

        self.config = _Cfg()


_FakeWorkflowExecutor._create_workflow_task = WorkflowExecutor._create_workflow_task


@pytest.mark.asyncio
async def test_descriptor_bypasses_should_accept_fn():
    """Test that a fork-eviction continuation descriptor returned by arun_episode
    is NOT run through should_accept_fn (which would reject it on its
    non-trajectory shape and silently lose the partial). The descriptor must
    pass through as-is so the controller's post_process_result can deposit it
    for reclaim. Regression guard for the workflow_executor short-circuit."""
    descriptor = {
        FORK_CONTINUATION_KEY: [
            {
                "rid": "r1",
                "prompt_ids": [1],
                "partial_ids": [2],
                "partial_logprobs": [0.1],
                "partial_versions": [0],
            }
        ],
        FORK_COMPLETE_KEY: None,
    }

    def reject_everything(traj):
        # A should_accept_fn that rejects ALL normal trajectories. Without the
        # short-circuit, the descriptor would be rejected here -> returned None
        # -> lost (never deposited for reclaim).
        return False

    pending = _RolloutTaskInput(
        task_id=42,
        data={"messages": "p"},
        workflow=_FakeWorkflow(descriptor),
        should_accept_fn=reject_everything,
        is_eval=False,
    )

    fake = _FakeWorkflowExecutor()
    closure = fake._create_workflow_task(pending)
    result = await closure()

    # Descriptor passes through (NOT rejected by should_accept_fn).
    assert result is not None
    assert isinstance(result, _RolloutResult)
    assert result.trajectory is descriptor
    assert result.task_id == 42


@pytest.mark.asyncio
async def test_normal_trajectory_still_goes_through_should_accept_fn():
    """Test that a normal (non-descriptor) trajectory still goes through
    should_accept_fn as before -- the descriptor short-circuit does NOT bypass
    filtering for real trajectories. A rejected normal trajectory returns None."""
    traj = _fake_traj(N_SAMPLES)

    def reject_everything(_traj):
        return False

    pending = _RolloutTaskInput(
        task_id=7,
        data={"messages": "p"},
        workflow=_FakeWorkflow(traj),
        should_accept_fn=reject_everything,
        is_eval=False,
    )

    fake = _FakeWorkflowExecutor()
    closure = fake._create_workflow_task(pending)
    result = await closure()

    # Normal trajectory rejected by should_accept_fn -> None (no short-circuit).
    assert result is None


# ---------------------------------------------------------------------------
# _workflow_supports_fork guard: fork only supports single-turn tensor-dict
# workflows with _arun_episode_continue (RLVRWorkflow). Agent / multi-turn
# workflows (OpenAIProxyWorkflow, online mode) are disabled. Prevents the fork
# from running with an incompatible workflow (salvage would break on
# InteractionWithTokenLogpReward; agent.run doesn't return on evict so the fork
# engine can't be torn down).
# ---------------------------------------------------------------------------


def test_workflow_supports_fork_rlvr():
    """Test that RLVRWorkflow (single-turn tensor-dict with _arun_episode_continue)
    supports fork -- both as a class and as a string import path."""
    assert PPOTrainer._workflow_supports_fork(RLVRWorkflow) is True
    assert (
        PPOTrainer._workflow_supports_fork("areal.workflow.rlvr.RLVRWorkflow") is True
    )


def test_workflow_supports_fork_agent_workflow():
    """Test that an agent/multi-turn workflow (no _arun_episode_continue, like
    OpenAIProxyWorkflow) does NOT support fork."""

    class _FakeAgentWorkflow:
        """Simulates OpenAIProxyWorkflow: has arun_episode but no
        _arun_episode_continue (multi-turn agent.run)."""

        async def arun_episode(self, engine, data):
            return None

    assert PPOTrainer._workflow_supports_fork(_FakeAgentWorkflow) is False


def test_workflow_supports_fork_online_mode():
    """Test that online agent mode (workflow=None) does not support fork."""
    assert PPOTrainer._workflow_supports_fork(None) is False


def test_workflow_supports_fork_unresolvable_string():
    """Test that an unresolvable workflow string fails safe (no fork)."""
    assert PPOTrainer._workflow_supports_fork("nonexistent.module.Workflow") is False


def test_workflow_supports_fork_callable():
    """Test that a bare callable (function-style workflow) has no
    _arun_episode_continue -> does not support fork."""

    async def _fn_workflow(engine, data):
        return None

    assert PPOTrainer._workflow_supports_fork(_fn_workflow) is False


# ---------------------------------------------------------------------------
# Reclaim snapshot accounting: descriptor vs completed-normally vs lost.
#
# ``fork_snapshot`` is the PRE-evict in-flight list. After settle, each task is
# in exactly one bucket:
#   (a) yielded a continuation descriptor  -> re-submitted to a surviving engine
#   (b) completed normally on the fork engine before evict took effect
#       (path A) -> already accepted (localize_if_fork records the task id)
#   (c) yielded neither -> truly lost (settle timeout / workflow failure)
# The old `snapshot - descriptors` count misreported (b) as "lost partial"
# (e.g. a sample that hit max_tokens exactly before the abort lands returns a
# normal "length" trajectory, remaining == 0). These tests pin the 3-way split
# and that only (a) is re-submitted.
# ---------------------------------------------------------------------------


def test_localize_if_fork_records_completed_task():
    """Test that accepting a fork trajectory records its task id AND accepted
    rollout count (path A), and that pop_completed_task_ids consumes the record
    exactly once."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    traj = _fake_traj(3)

    out = controller._fork.localize_if_fork(_FakeWorker("actor-fork/0"), 7, traj)

    # RTensor.localize returns a localized copy (dict comprehension), not the
    # same object; content is preserved.
    assert out["rids"] == traj["rids"]
    # task_id -> accepted rollout count (3 rows), not a task-only record.
    assert controller._fork.pop_completed_task_ids() == {7: 3}
    # Popped (consumed exactly once), not merely read.
    assert controller._fork.pop_completed_task_ids() == {}


def test_localize_if_fork_ignores_surviving_worker():
    """Test that normal trajectories from surviving engines are not recorded as
    fork completions (they are never in the fork snapshot)."""
    controller = _RecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    traj = _fake_traj(1)

    out = controller._fork.localize_if_fork(_FakeWorker("rollout/0"), 8, traj)

    assert out is traj
    assert controller._fork.pop_completed_task_ids() == {}


class _RecordingLogger:
    """Captures reclaim's log lines so the classification is assertable."""

    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class _MixedEvictController(_RecordingController):
    """Simulates one evict with four in-flight fork tasks:
    tid=0 -> descriptor only (all in-flight, interrupted);
    tid=1 -> completed normally before evict took effect (path A, 4 rollouts);
    tid=2 -> nothing at all (lost);
    tid=3 -> partial evict: 5 in-flight (descriptor) + 3 completed (parked).
    """

    def _fork_collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"fork.{method}")
        if method != "evict":
            return
        for tid, wid in list(self._task_worker_map.items()):
            self._task_worker_map.pop(tid, None)
            self._task_input_map.pop(tid, None)
            if tid == 0:
                self._fork._fork_continuations[tid] = {
                    FORK_CONTINUATION_KEY: [_descriptor(f"rid-{tid}")]
                }
            elif tid == 1:
                # Normal-path accept: localize_if_fork records id + rollout count.
                self._fork.localize_if_fork(_FakeWorker(wid), tid, _fake_traj(4))
            elif tid == 3:
                self._fork._fork_continuations[tid] = {
                    FORK_CONTINUATION_KEY: [
                        _descriptor(f"rid-{tid}-{i}") for i in range(5)
                    ],
                    FORK_COMPLETE_KEY: _fake_traj(3, rid_prefix="done"),
                }
            # tid == 2: no descriptor, not accepted -> truly lost (no future was
            # force-resolved in this simulation, so it is not restarted).


def test_reclaim_classifies_snapshot_and_resubmits_only_descriptors(monkeypatch):
    """Test that reclaim splits the pre-evict snapshot into descriptor /
    completed-normally / lost and reports the split in ROLLOUTS (group_size per
    task), not just task counts. Re-submits ONLY descriptor tasks; path-A
    completions must not be reported as lost."""
    controller = _MixedEvictController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    # task_id -> group_size (n_samples rollouts of one prompt).
    group_sizes = {0: 1, 1: 4, 2: 2, 3: 8}
    controller._task_worker_map = {tid: "actor-fork/0" for tid in group_sizes}
    controller._task_input_map = {
        tid: _RemoteRolloutTaskInput(
            task_id=tid,
            data={"messages": f"p{tid}"},
            workflow="wf",
            workflow_kwargs={},
            should_accept_fn=None,
            group_size=group_size,
        )
        for tid, group_size in group_sizes.items()
    }

    class _Gen:
        def __init__(self):
            self._n = 100

        def next(self):
            self._n += 1
            return self._n

    controller._task_id_generator = _Gen()

    submitted: list[dict] = []

    def capture_submit(**kwargs):
        submitted.append(kwargs)
        controller.calls.append(f"controller.submit:{kwargs['task_id']}")

    controller.submit = capture_submit

    recorder = _RecordingLogger()
    monkeypatch.setattr("areal.infra.controller.rollout_controller.logger", recorder)

    controller.reclaim_and_teardown_actor_fork()

    # Only descriptor tasks (0 all-in-flight, 3 partially evicted) are
    # re-submitted; task 1 completed normally, task 2 is lost (no future was
    # force-resolved in this simulation, so no restart either).
    assert len(submitted) == 2
    assert [s["data"][FORK_MERGE_KEY] for s in submitted] == [0, 3]
    assert all(FORK_CONTINUE_FIELD in s["data"] for s in submitted)
    # group_size = number of interrupted rollouts per descriptor task.
    assert [s["group_size"] for s in submitted] == [1, 5]

    infos = [msg for kind, msg in recorder.messages if kind == "info"]
    warnings = [msg for kind, msg in recorder.messages if kind == "warning"]
    # Per-task rollout accounting.
    assert any(
        "task_id=0 rollouts=1: 1 interrupted" in m and "0 lost" in m for m in infos
    )
    assert any("task_id=1 rollouts=4: 4 completed on fork" in m for m in infos)
    assert any("task_id=2 rollouts=2: 2 lost" in m for m in infos)
    assert any(
        "task_id=3 rollouts=8: 5 interrupted" in m
        and "3 completed on fork (parked for re-merge)" in m
        and "0 lost" in m
        for m in infos
    )
    # Aggregate in rollouts: 1+4+2+8 = 15 = 4 accepted + 3 parked + 6
    # interrupted + 2 lost.
    assert any(
        "rollout accounting across 4 fork task(s) (15 rollouts): "
        "4 completed (accepted) + 3 parked for re-merge + 6 interrupted "
        "(re-submitted as continuations) + 2 lost" in m
        for m in infos
    )
    # Only whole-task losses get the warning (task 2).
    assert any("1/4 fork task(s) yielded neither" in m for m in warnings)
    assert any("[2]" in m for m in warnings)
    # Window accounting consumed (pop) -- a later reclaim cannot double-count.
    assert controller._fork._fork_completed_task_ids == {}


# ---------------------------------------------------------------------------
# Settle timeout -> full restart
#
# A task that fails to settle is abandoned (its future is resolved with the
# _FORK_TASK_ABANDONED sentinel, so its coroutine rejects without fetching a
# result) and its original input is re-submitted as a brand-new group.
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic clock: ``sleep`` advances ``perf_counter``."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _FakeFuture:
    """Minimal asyncio-future stand-in for the settle-timeout resolve."""

    def __init__(self):
        self.resolved = False
        self.value = None

    def done(self):
        return self.resolved

    def set_result(self, value):
        self.resolved = True
        self.value = value

    def get_loop(self):
        return self

    def call_soon_threadsafe(self, cb, *args):
        cb(*args)


class _HungTaskController(_RecordingController):
    """Evict settles tid=0 with a descriptor and leaves tid=1 hung: the fork
    request never returns (the 34/38/42/43 failure mode), so its task stays in
    ``_task_worker_map`` until the settle timeout."""

    def _fork_collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"fork.{method}")
        if method != "evict":
            return
        self._task_worker_map.pop(0, None)
        self._task_input_map.pop(0, None)
        self._fork._fork_continuations[0] = {
            FORK_CONTINUATION_KEY: [_descriptor("rid-0")]
        }


def test_reclaim_restarts_settle_timeout_task(monkeypatch):
    """A fork task that never settles is abandoned (sentinel future) and its
    original input is re-submitted as a full restart with a new task id."""
    controller = _HungTaskController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    controller._task_worker_map = {0: "actor-fork/0", 1: "actor-fork/0"}
    controller._task_input_map = {
        tid: _RemoteRolloutTaskInput(
            task_id=tid,
            data={"messages": f"p{tid}"},
            workflow="wf",
            workflow_kwargs={"k": tid},
            should_accept_fn=None,
            group_size=8,
        )
        for tid in (0, 1)
    }
    future = _FakeFuture()
    controller._pending_futures[1] = future

    class _Gen:
        def __init__(self):
            self._n = 100

        def next(self):
            self._n += 1
            return self._n

    controller._task_id_generator = _Gen()

    submitted: list[dict] = []
    controller.submit = lambda **kw: submitted.append(kw)

    recorder = _RecordingLogger()
    monkeypatch.setattr("areal.infra.controller.rollout_controller.logger", recorder)
    monkeypatch.setattr("areal.infra.controller.rollout_controller.time", _FakeClock())

    controller.reclaim_and_teardown_actor_fork()

    # task 0 -> continuation; task 1 -> full restart.
    assert len(submitted) == 2
    continuation, restart = submitted
    assert FORK_CONTINUE_FIELD in continuation["data"]
    assert FORK_CONTINUE_FIELD not in restart["data"]
    assert restart["data"] == {"messages": "p1"}  # original input
    assert restart["workflow"] == "wf"
    assert restart["workflow_kwargs"] == {"k": 1}
    assert restart["group_size"] == 8  # original group size, not partial count
    assert restart["priority"] is True
    assert restart["task_id"] != 1  # new id

    # The stuck future was resolved with the sentinel: its coroutine rejects
    # without fetching, so the restart cannot be duplicated.
    assert future.resolved
    assert future.value is _FORK_TASK_ABANDONED

    infos = [msg for kind, msg in recorder.messages if kind == "info"]
    warnings = [msg for kind, msg in recorder.messages if kind == "warning"]
    assert any("did not settle within 30.0s" in m for m in warnings)
    assert any(
        "task_id=1 rollouts=8: 8 lost" in m and "full restart" in m for m in infos
    )
    assert any("re-submitted 1 continuation + 1 restart task(s)" in m for m in infos)


def test_reclaim_removes_fork_workers_from_pool_before_evict():
    """The fork workers are out of the dispatch pool before evict.
    ``_submit_then_wait`` picks a worker and registers a task under
    ``_task_worker_map_lock``; reclaim removes the fork workers from
    ``self.workers`` under that same lock before snapshotting. A task that
    starts after the snapshot therefore cannot land on a fork engine, which
    makes the snapshot complete (no late fork task can be born).
    """

    class _PoolRecordingController(_RecordingController):
        workers_at_evict: list[str] | None = None
        worker_after_snapshot: str | None = None

        _choose_worker = RolloutController._choose_worker

        def _fork_collective_rpc(self, method, *args, **kwargs):
            if method == "evict":
                self.workers_at_evict = [w.id for w in self.workers]
                # Simulate a task registering after the snapshot: it must be
                # routed to a surviving worker (and the round-robin cursor must
                # still be valid for the smaller pool).
                self.worker_after_snapshot = self._choose_worker()[0].id
            super()._fork_collective_rpc(method, *args, **kwargs)

    controller = _PoolRecordingController(
        surviving_workers=[_FakeWorker("rollout/0")],
        fork_workers=[_FakeWorker("actor-fork/0")],
    )
    controller._worker_engine_names = {
        "rollout/0": "rollout/0",
        "actor-fork/0": "actor-fork/0",
    }
    # Cursor left over from the two-worker pool; reclaim must normalize it.
    controller._current_worker_idx = 1
    controller._task_worker_map = {0: "actor-fork/0"}
    controller._task_input_map = {
        0: _RemoteRolloutTaskInput(
            task_id=0,
            data={"messages": "p0"},
            workflow="wf",
            workflow_kwargs={},
            should_accept_fn=None,
            group_size=1,
        )
    }

    class _Gen:
        def __init__(self):
            self._n = 100

        def next(self):
            self._n += 1
            return self._n

    controller._task_id_generator = _Gen()

    controller.reclaim_and_teardown_actor_fork()

    # Fork workers were already gone when evict ran, and stay gone afterwards.
    assert controller.workers_at_evict == ["rollout/0"]
    assert [w.id for w in controller.workers] == ["rollout/0"]
    # A task registering after the snapshot goes to a surviving engine.
    assert controller.worker_after_snapshot == "rollout/0"
