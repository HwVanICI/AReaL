# SPDX-License-Identifier: Apache-2.0

"""Fork-teardown SALVAGE state + logic, extracted from RolloutController.

The step-0 actor-GPU fork feature's salvage path (deposit an evicted partial,
park the already-complete half, re-merge with the continuation, localize fork
trajectories before teardown) was inlined in the controller's per-task callback,
polluting the hot path. This module owns that salvage state + logic so the
callback stays a thin delegator.

The fork LIFECYCLE (start engines / reclaim-and-teardown / collective RPC)
remains on ``RolloutController``; ``ForkCoordinator`` holds only the per-task
continuation/re-merge state shared between the callback and reclaim, accessed
via a ``host`` reference (the controller) for shared maps/locks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.utils.fork_constants import (
    FORK_COMPLETE_KEY,
    FORK_CONTINUATION_KEY,
    FORK_MERGE_KEY,
)

if TYPE_CHECKING:
    from areal.infra.controller.rollout_controller import RolloutController

logger = logging.getLogger("ForkCoordinator")


class ForkCoordinator:
    """Owns fork-teardown salvage state + the per-task salvage methods.

    State
    -----
    _fork_continuations : dict[int, dict]
        task_id -> fork-eviction continuation descriptor (group), deposited by
        the callback when a fork engine returns a partial; reclaim re-submits
        each as a continuation on a surviving engine.
    _fork_complete_groups : dict[int, dict]
        fork task_id -> localized already-complete rollouts of a partially-
        evicted group (the samples that finished before eviction). Parked at the
        deposit (while the fork worker is alive) and re-merged with the
        continuation result when that task completes, so each prompt yields one
        full n_samples group. NOT cleared at reclaim: continuations complete
        after reclaim returns (priority-bypass, step-0 training window); cleared
        on consume (pop) and defensively in ``clear_complete_groups``.
    _fork_completed_task_ids : dict[int, int]
        fork task id -> number of rollouts accepted as a normal trajectory
        (path A: the group fully completed on the fork engine before eviction
        took effect). Recorded by ``localize_if_fork`` so reclaim can report
        ROLLOUT-level accounting (not just task counts) and distinguish
        "completed normally, no continuation needed" from "yielded nothing at
        all (lost)" for its pre-evict in-flight snapshot. Popped by reclaim and
        cleared defensively in ``clear_completed_task_ids``.
    """

    def __init__(self, host: RolloutController) -> None:
        self.host = host
        self._fork_continuations: dict[int, dict[str, Any]] = {}
        self._fork_complete_groups: dict[int, dict[str, Any]] = {}
        self._fork_completed_task_ids: dict[int, int] = {}

    # -- classification ---------------------------------------------------

    @staticmethod
    def is_evict_descriptor(traj: Any) -> bool:
        """True if ``traj`` is a fork-eviction continuation descriptor (not a
        training trajectory)."""
        return (
            traj is not None
            and isinstance(traj, dict)
            and FORK_CONTINUATION_KEY in traj
        )

    def is_fork_worker(self, worker_id: str) -> bool:
        return worker_id in self.host._fork_worker_ids

    # -- callback salvage (deposit / re-merge / localize) -----------------

    def on_task_evicted(self, task_id: int, worker: Any, traj: dict[str, Any]) -> None:
        """Deposit a fork-evicted partial (descriptor + parked complete-group)
        for reclaim to re-submit as a continuation.

        Localizes the already-complete part NOW (while the fork worker is alive
        so the lazy RTensor shards can be fetched). Pops the task from the
        worker/input maps (under the host lock, BEFORE returning) so reclaim's
        settle sees it done -- the settle loop waits for fork tasks to leave
        ``_task_worker_map`` before destroying fork workers (no race).
        """
        complete_group = traj.get(FORK_COMPLETE_KEY)
        if complete_group is not None:
            complete_group = RTensor.localize(complete_group)
        with self.host._task_worker_map_lock:
            self._fork_continuations[task_id] = traj
            if complete_group is not None:
                self._fork_complete_groups[task_id] = complete_group
            self.host._task_worker_map.pop(task_id, None)
            self.host._task_input_map.pop(task_id, None)
        n_samples = len(traj.get(FORK_CONTINUATION_KEY, []))
        rids = [c.get("rid") for c in traj.get(FORK_CONTINUATION_KEY, [])]
        n_complete = (
            complete_group["input_ids"].shape[0] if complete_group is not None else 0
        )
        logger.info(
            f"[step0-fork] task_id={task_id} on worker={worker.id}: "
            f"deposited fork-evicted partial ({n_samples} in-flight, "
            f"{n_complete} completed parked for re-merge, "
            f"rids={rids}) for continuation re-submit"
        )

    def try_remerge(self, pending_task: Any, traj: dict[str, Any]) -> dict[str, Any]:
        """Re-merge a continuation result with its parked already-complete half.

        If ``pending_task.data`` carries ``FORK_MERGE_KEY`` (injected by reclaim
        when re-submitting a continuation), look up the complete-group parked at
        eviction and concat it with the continued result into one full
        n_samples group. The continued result is a lazy RTensor on the surviving
        worker; localize before concat (``concat_padded_tensors`` does not handle
        RTensor). Downstream reward normalization groups positionally by
        n_samples, so each prompt must be one contiguous block.

        No-op (returns ``traj`` unchanged) for normal tasks and for the
        all-in-flight case (no complete half parked -> ``traj`` is already a full
        group).
        """
        merge_key = pending_task.data.get(FORK_MERGE_KEY)
        if merge_key is None:
            return traj
        complete_group = self._fork_complete_groups.pop(merge_key, None)
        if complete_group is None:
            # All-in-flight case: the continued result is already a full
            # n_samples group; accept as-is (stays lazy RTensor on the surviving
            # worker, fetched later by compute_advantages).
            return traj
        continued = RTensor.localize(traj)
        n_complete = complete_group["input_ids"].shape[0]
        n_cont = continued["input_ids"].shape[0]
        merged = concat_padded_tensors([complete_group, continued])
        logger.info(
            f"[step0-fork] task_id={pending_task.task_id}: re-merged "
            f"continuation ({n_complete} completed + {n_cont} continued = "
            f"{merged['input_ids'].shape[0]} rollouts)"
        )
        return merged

    def localize_if_fork(
        self, worker: Any, task_id: int, traj: dict[str, Any]
    ) -> dict[str, Any]:
        """Localize fork-generated trajectories (lazy RTensor shards on the fork
        worker's /data/* endpoints) before the upcoming fork teardown, and log
        path A (fork self-completed prompt). No-op for surviving workers.

        Path A: a fork task that finished fully (n_samples/n_samples) before
        eviction is accepted as a normal trajectory. Logged so fork
        self-completed prompts are grep-able; evicted prompts are tracked via
        the deposit + re-merge logs, and continuations run on surviving workers
        (this does not fire for them). The task id and its accepted rollout
        count are recorded so reclaim can report task- and rollout-level
        accounting for the in-flight snapshot (see ``pop_completed_task_ids``).
        """
        if worker.id not in self.host._fork_worker_ids:
            return traj
        traj = RTensor.localize(traj)
        n_rollouts = traj["input_ids"].shape[0]
        # Recorded AFTER localize: a failed shard fetch means the trajectory
        # could not be accepted, so it is not "completed normally".
        with self.host._task_worker_map_lock:
            self._fork_completed_task_ids[task_id] = n_rollouts
        logger.info(
            f"[step0-fork] task_id={task_id} on worker={worker.id}: "
            f"fully completed on fork ({n_rollouts} rollouts), accepted"
        )
        return traj

    def post_process_result(
        self, pending_task: Any, worker: Any, task_id: int, traj: Any
    ) -> Any:
        """Per-task fork salvage hook (called from the controller callback for
        EVERY rollout, fork or not).

        Combines the three salvage steps into one call so the controller
        callback is a single-line delegator. Returns the trajectory to accept,
        or None to reject:

        * evicted descriptor -> on_task_evicted (deposit + park, which pops the
          task maps under the host lock) then None; reclaim re-submits a
          continuation. The controller's reject-path map pop is idempotent.
        * None result -> None (reject).
        * normal/continued trajectory -> try_remerge + localize_if_fork, then
          the (possibly re-merged) trajectory.

        No-op for normal surviving-engine tasks: no FORK_MERGE_KEY -> try_remerge
        returns traj unchanged; worker not in _fork_worker_ids -> localize_if_fork
        returns traj unchanged. See :class:`NoOpForkCoordinator` for the
        disabled-feature (zero-overhead) case.
        """
        if self.is_evict_descriptor(traj):
            self.on_task_evicted(task_id, worker, traj)
            return None
        if traj is None:
            return None
        traj = self.try_remerge(pending_task, traj)
        traj = self.localize_if_fork(worker, task_id, traj)
        return traj

    # -- reclaim / destroy accessors --------------------------------------

    def pop_continuation(self, task_id: int) -> dict[str, Any] | None:
        """Pop a deposited continuation descriptor (reclaim collects these to
        re-submit). None if the task yielded no descriptor (fully completed or
        lost)."""
        return self._fork_continuations.pop(task_id, None)

    def pop_completed_task_ids(self) -> dict[int, int]:
        """Pop ``{task_id: accepted_rollout_count}`` for path-A completions.

        Reclaim uses this to split its pre-evict in-flight snapshot into three
        disjoint groups: yielded a continuation descriptor, completed normally
        (no continuation needed), or yielded nothing at all (truly lost) -- and
        to report each bucket in ROLLOUTS (``group_size`` per task), not just
        task counts. Pops (rather than reads) so the window's accounting is
        consumed exactly once.
        """
        with self.host._task_worker_map_lock:
            completed = self._fork_completed_task_ids
            self._fork_completed_task_ids = {}
        return completed

    def clear_continuations(self) -> None:
        """Clear deposited descriptors (reclaim end, after re-submitting)."""
        self._fork_continuations.clear()

    def clear_completed_task_ids(self) -> None:
        """Defensive: drop the path-A accepted-task record if reclaim never
        consumed it (e.g. crash mid-step-0). Called from reclaim/destroy."""
        with self.host._task_worker_map_lock:
            self._fork_completed_task_ids.clear()

    def clear_complete_groups(self) -> None:
        """Defensive: drop any parked complete-groups whose continuation never
        ran (e.g. crash mid-step-0). Called from destroy()."""
        self._fork_complete_groups.clear()


class NoOpForkCoordinator:
    """Disabled-feature fork strategy (used when ``enable_actor_gpu_rollout``
    is off).

    Every method is inert so the controller's per-task callback hot path carries
    zero fork overhead when the feature is disabled: ``post_process_result`` is
    identity (no descriptor check, no salvage calls). The fork lifecycle
    (start/reclaim) is never invoked by the trainer when the feature is off, so
    this class only needs the surface the controller touches unconditionally --
    ``post_process_result`` (hot path) and ``clear_complete_groups`` (destroy).
    ``pop_continuation`` / ``clear_continuations`` / ``pop_completed_task_ids``
    / ``clear_completed_task_ids`` / ``is_fork_worker`` are included for
    interface parity with :class:`ForkCoordinator`.
    """

    def post_process_result(
        self, pending_task: Any, worker: Any, task_id: int, traj: Any
    ) -> Any:
        """Identity: accept the trajectory unchanged (no fork salvage)."""
        return traj

    def pop_continuation(self, task_id: int) -> dict[str, Any] | None:
        return None

    def pop_completed_task_ids(self) -> dict[int, int]:
        return {}

    def clear_continuations(self) -> None:
        pass

    def clear_completed_task_ids(self) -> None:
        pass

    def clear_complete_groups(self) -> None:
        pass

    def is_fork_worker(self, worker_id: str) -> bool:
        return False
