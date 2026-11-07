# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import shutil
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from torchdata.stateful_dataloader import StatefulDataLoader
from werkzeug.serving import make_server

from areal.api import (
    InferenceEngine,
    Job,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    Scheduler,
    WeightUpdateMeta,
    Worker,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PerfTracerConfig,
    SchedulingSpec,
    SchedulingStrategy,
)
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging, perf_tracer
from areal.utils.data import cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.fork_constants import (
    FORK_COMPLETE_KEY,
    FORK_CONTINUATION_KEY,
    FORK_CONTINUE_FIELD,
    FORK_MERGE_KEY,
)
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.utils.perf_tracer import trace_perf

from ..staleness_manager import StalenessManager
from ..workflow_executor import BatchTaskDispatcher, TaskIdGenerator
from .fork_coordinator import ForkCoordinator, NoOpForkCoordinator

logger = logging.getLogger("RolloutController")

# Resolved into a task future by reclaim when a fork task misses the settle
# timeout. The task coroutine sees it and rejects without fetching a result;
# reclaim replays the original input as a full restart. The future is a
# one-shot ownership channel: whoever resolves it (engine callback vs reclaim)
# decides who owns the task, so no shared state or locking is needed.
_FORK_TASK_ABANDONED = object()


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    task_id: int
    data: dict[str, Any]
    workflow: str | None
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str | None
    is_eval: bool = False
    group_size: int = 1
    proxy_addr: str | None = None
    # Priority tasks bypass the staleness capacity gate in the dispatcher
    # (still respect pause + runner queue size). Used for step-0 fork-evicted
    # continuation re-submits so they are pulled and run at version 0 during the
    # step-0 training window instead of being starved by overlap saturation.
    priority: bool = False


@dataclass
class _RemoteRolloutResult:
    task_id: int
    trajectory: dict[str, Any]


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Parse allocation from config.backend
        self.rollout_alloc = ModelAllocation.from_str(config.backend)

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0

        # State
        self._version_lock = Lock()
        self._version = 0

        self._task_id_generator = TaskIdGenerator()

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

        # HTTP callback server
        self._callback_app: Flask | None = None
        self._callback_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()

        # Task completion futures
        self._pending_futures: dict[int, asyncio.Future] = {}
        self._futures_lock = threading.Lock()

        # Proxy worker management (for AgentWorkflow support)
        self.proxy_workers: list[Worker] = []
        self.proxy_addrs: list[str] = []
        self._proxy_started = False

        # Proxy gateway server (for online/external access)
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread: threading.Thread | None = None
        self._proxy_gateway_port: int | None = None
        self._proxy_gateway_host: str | None = None

        # Actor-GPU fork engines (step-0 utilization feature): fork state +
        # strategy (Fork/NoOp) live in _init_fork_state.
        self._init_fork_state()

        # task_id -> worker.id, to locate which in-flight tasks sit on fork
        # engines during reclaim (so they can be aborted and redispatched).
        self._task_worker_map: dict[int, str] = {}
        # task_id -> original _RemoteRolloutTaskInput, so reclaim can replay
        # the exact input on a surviving engine (not an empty placeholder).
        self._task_input_map: dict[int, _RemoteRolloutTaskInput] = {}
        # worker.id -> engine_name. Surviving engines are named
        # "<role>/<rank>" but fork engines are named "<fork_role>/<rank>";
        # round-robin dispatch over the combined self.workers list must use the
        # engine name that matches the chosen worker (not a rollout-prefixed
        # index), or submit RPCs 404 on fork workers.
        self._worker_engine_names: dict[str, str] = {}
        self._task_worker_map_lock = threading.Lock()

    def _init_fork_state(self) -> None:
        """Init actor-GPU fork engines state + the fork strategy.

        Forked vLLM engines run on actor GPUs during the first rollout, are
        appended to ``self.workers`` so round-robin dispatch serves them, and
        are reclaimed (aborted + redispatched) before teardown so the actor can
        resume training. The fork state (``fork_workers`` / ``_fork_started`` /
        ``_fork_role`` / ``_fork_worker_ids``) is always initialized (harmless
        empty defaults when the feature is off); the per-task salvage strategy
        is :class:`ForkCoordinator` when ``enable_actor_gpu_rollout`` is on,
        else :class:`NoOpForkCoordinator` (identity) so the callback hot path
        carries zero fork overhead when the feature is disabled. The fork
        LIFECYCLE (start/reclaim/teardown) stays on this controller; the
        coordinator holds only the continuation/re-merge state + salvage logic.
        """
        self.fork_workers: list[Worker] = []
        self._fork_started: bool = False
        self._fork_role: str = "actor-fork"
        self._fork_worker_ids: set[str] = set()
        self._fork = (
            ForkCoordinator(host=self)
            if self.config.enable_actor_gpu_rollout
            else NoOpForkCoordinator()
        )

    @property
    def _proxy_role(self) -> str:
        """Generate a unique proxy role name based on the worker role.

        This avoids collisions when multiple controllers (e.g., rollout and
        eval-rollout) each fork proxy workers into the same scheduler.
        """
        if not hasattr(self, "_worker_role"):
            raise RuntimeError(
                "Cannot access _proxy_role before initialize() is called"
            )
        return f"proxy-{self._worker_role}"

    def _proxy_engine_name(self, rank: int) -> str:
        """Generate engine name for a proxy worker rank."""
        return f"{self._proxy_role}/{rank}"

    def _engine_name(self, rank: int) -> str:
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "rollout/0", "rollout/1").
        """
        return f"{self._worker_role}/{rank}"

    def initialize(
        self,
        role: str,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        instance_size = (
            self.rollout_alloc.parallel.tp_size * self.rollout_alloc.parallel.pp_size
        )
        dp_size = self.rollout_alloc.parallel.dp_size

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= instance_size
        sch_spec.mem *= instance_size
        if sch_spec.gpu > 0:
            sch_spec.gpu = instance_size

        job = Job(
            replicas=dp_size,
            tasks=[sch_spec for _ in range(dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Call async scheduler methods synchronously
        run_async_task(
            self._async_initialize, job, server_args, server_infos, *args, **kwargs
        )

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start callback server for weight sync coordination
        self._start_callback_server()

    async def _async_initialize(
        self,
        job: Job,
        server_args: dict[str, Any],
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        # Map each surviving worker to its engine name so round-robin dispatch
        # (which may later include fork workers) resolves the right engine.
        for rank, worker in enumerate(self.workers):
            self._worker_engine_names[worker.id] = self._engine_name(rank)

        logger.info("Calling engine initialization...")
        # Workers are controller-managed: the controller handles staleness
        # globally, so workers must NOT apply their own dp-scaled staleness
        # constraints. Force train_data_parallel_size=1 unless explicitly
        # configured; an explicit None must not survive, or workers fall back
        # to dividing capacity by dist.get_world_size().
        if kwargs.get("train_data_parallel_size") is None:
            kwargs["train_data_parallel_size"] = 1
        if server_infos is not None:
            # Connecting to existing local servers for evaluation
            self.server_infos = server_infos
            assert len(self.server_infos) == len(self.workers), (
                len(self.server_infos),
                len(self.workers),
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    engine_rank=rank,
                    num_engines=len(self.workers),
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        else:
            self.server_infos = await self._collective_rpc_async(
                "launch_server", server_args=server_args
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    engine_rank=rank,
                    num_engines=len(self.workers),
                    *args,
                    **kwargs,
                )
                for rank, worker in enumerate(self.workers)
            ]
            await asyncio.gather(*tasks)

        logger.info("All engines are initialized...")

    def destroy(self):
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._stop_callback_server()

        self._collective_rpc("destroy", http_timeout=60.0)

        # Delete workers via scheduler
        if hasattr(self, "_worker_role"):
            try:
                self.scheduler.delete_workers(role=self._worker_role)
                self.workers.clear()
                logger.info("Workers deleted")
            except Exception:
                logger.error(f"Error deleting workers: {traceback.format_exc()}")

        # Delete actor-GPU fork workers if still alive (defensive: handles a
        # crash mid-step-0 before reclaim_and_teardown_actor_fork ran).
        if getattr(self, "_fork_started", False):
            try:
                self._fork_collective_rpc("destroy", http_timeout=60.0)
            except Exception:
                logger.error(f"Error destroying fork engines: {traceback.format_exc()}")
            try:
                self.scheduler.delete_workers(role=self._fork_role)
                self.fork_workers.clear()
                for wid in self._fork_worker_ids:
                    self._worker_engine_names.pop(wid, None)
                self._fork_worker_ids.clear()
                self._fork_started = False
                logger.info("Fork workers deleted")
            except Exception:
                logger.error(f"Error deleting fork workers: {traceback.format_exc()}")

        # Defensive: drop any parked fork complete-groups whose continuation
        # never ran (e.g. a crash mid-step-0). Normally consumed by the
        # continuation accept branch (pop).
        if hasattr(self, "_fork"):
            self._fork.clear_complete_groups()
            self._fork.clear_completed_task_ids()

        # Delete proxy workers if initialized
        if self._proxy_started:
            try:
                self.scheduler.delete_workers(role=self._proxy_role)
                self.proxy_workers.clear()
                self.proxy_addrs.clear()
                self._proxy_started = False
                logger.info("Proxy workers deleted")
            except Exception:
                logger.error(f"Error deleting proxy workers: {traceback.format_exc()}")

        # Shutdown proxy gateway if initialized
        self._stop_proxy_gateway()
        with self._futures_lock:
            self._pending_futures.clear()

    def start_proxy(self) -> None:
        """Initialize proxy workers for AgentWorkflow support.

        Creates proxy workers colocated with rollout workers. Each proxy worker
        runs a ProxyRolloutServer that connects to the same inference server
        as its corresponding rollout worker.
        """
        if self._proxy_started:
            logger.warning("Proxy workers already initialized")
            return

        if not self.server_infos:
            raise RuntimeError(
                "Cannot initialize proxy workers: rollout not initialized. "
                "Call initialize() first."
            )

        run_async_task(self._async_start_proxy)
        self._proxy_started = True

    async def _async_start_proxy(self) -> None:
        """Async implementation of proxy worker initialization."""
        command = "areal.experimental.openai.proxy.proxy_rollout_server"
        job = Job(
            role=self._proxy_role,
            replicas=len(self.server_infos),
            tasks=[SchedulingSpec(gpu=0, port_count=2)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation",
                target=self._worker_role,
                fork=True,
            ),
        )
        worker_ids = self.scheduler.fork_workers(job, command=command)
        logger.info(f"Proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role=self._proxy_role)
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=self._proxy_engine_name(rank),
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Proxy engines created")

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos, strict=True)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._proxy_engine_name(rank),
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )
        await asyncio.gather(*init_tasks)

        logger.info(f"Proxy servers initialized. Addresses: {self.proxy_addrs}")

    # ------------------------------------------------------------------
    # Actor-GPU fork engines (step-0 utilization feature)
    # ------------------------------------------------------------------

    def _fork_engine_name(self, rank: int) -> str:
        """Engine name for a forked worker rank, e.g. 'actor-fork/0'."""
        return f"{self._fork_role}/{rank}"

    def start_actor_fork_engines(
        self,
        actor_role: str,
        fork_instance_gpus: int,
        fork_replicas: int,
        server_args: dict[str, Any],
    ) -> None:
        """Fork vLLM rollout engines onto the actor GPUs and register them.

        Mirrors ``start_proxy`` but with a **GPU** fork (one vLLM server per
        group of ``fork_instance_gpus`` actor workers), plus ``launch_server``
        and engine ``initialize`` (proxy forks only attach to existing
        servers). Forked workers are appended to ``self.workers`` so
        round-robin dispatch serves them automatically.

        ``fork_instance_gpus`` is the GPU footprint of one surviving rollout
        vLLM server = data_parallel_size x tensor_parallel_size x
        pipeline_parallel_size (matches the launch command; in the inference
        EP scenario EP size = data_parallel_size x tensor_parallel_size ==
        one instance). RayScheduler's grouped GPU fork requires
        fork_instance_gpus == actor_world_size // fork_replicas, which holds
        by construction when fork_replicas = actor_world_size // fork_instance_gpus.
        """
        if self._fork_started:
            logger.warning("Actor-GPU fork engines already started; skipping.")
            return
        run_async_task(
            self._async_start_actor_fork_engines,
            actor_role,
            fork_instance_gpus,
            fork_replicas,
            server_args,
        )
        self._fork_started = True

    async def _async_start_actor_fork_engines(
        self,
        actor_role: str,
        fork_instance_gpus: int,
        fork_replicas: int,
        server_args: dict[str, Any],
    ) -> None:
        # Reuse the rollout worker scheduling_spec (cmd/env) for the forked
        # workers, but force GPU == fork_instance_gpus (one vLLM server per
        # group), scaling cpu/mem to match the per-instance GPU count.
        fork_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        fork_spec.cpu *= fork_instance_gpus
        fork_spec.mem *= fork_instance_gpus
        fork_spec.gpu = fork_instance_gpus

        job = Job(
            replicas=fork_replicas,
            tasks=[fork_spec for _ in range(fork_replicas)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation",
                target=actor_role,
                fork=True,
            ),
            role=self._fork_role,
        )

        logger.info(
            f"[step0-fork] forking {fork_replicas} vLLM engine(s) "
            f"({fork_instance_gpus} GPUs each) onto actor role={actor_role}"
        )
        fork_start = time.perf_counter()
        worker_ids = self.scheduler.fork_workers(job)
        logger.info(f"[step0-fork] fork workers created: {worker_ids}")

        self.fork_workers = self.scheduler.get_workers(role=self._fork_role)
        logger.info(
            f"[step0-fork] fork workers ready: {[w.id for w in self.fork_workers]}"
        )

        engine_class = self.inf_engine
        create_tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._fork_engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.fork_workers)
        ]
        await asyncio.gather(*create_tasks)
        logger.info("Fork engines created.")

        # Launch the vLLM servers on the forked workers (same path as the
        # surviving engines in _async_initialize).
        await self._fork_collective_rpc_async("launch_server", server_args=server_args)
        logger.info("[step0-fork] vLLM servers launched on fork workers.")

        init_tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="initialize",
                engine_name=self._fork_engine_name(rank),
                engine_id=str(rank),
                engine_rank=rank,
                num_engines=len(self.fork_workers),
                # Controller-managed: workers must not apply their own
                # dp-scaled staleness constraints (mirrors _async_initialize).
                train_data_parallel_size=1,
            )
            for rank, worker in enumerate(self.fork_workers)
        ]
        await asyncio.gather(*init_tasks)
        logger.info("[step0-fork] fork engines initialized.")

        # Keep fork engines in lockstep with the controller version (step-0 = 0).
        await self._fork_collective_rpc_async(
            "set_version", version=self.get_version(), http_timeout=60.0
        )

        # Register forked workers into the dispatch pool. Round-robin
        # (_choose_worker) now distributes traffic to surviving + fork engines.
        self._fork_worker_ids = {w.id for w in self.fork_workers}
        # Map fork workers to their (differently-named) engines so dispatch
        # resolves engine_name="actor-fork/<rank>" instead of "rollout/<idx>".
        for rank, worker in enumerate(self.fork_workers):
            self._worker_engine_names[worker.id] = self._fork_engine_name(rank)
        surviving_n = len(self.workers)
        self.workers.extend(self.fork_workers)
        logger.info(
            f"[step0-fork] fork engines registered (surviving={surviving_n} "
            f"+ fork={len(self.fork_workers)} = {len(self.workers)} dispatch "
            f"workers, version={self.get_version()}) in "
            f"{time.perf_counter() - fork_start:.1f}s"
        )

    async def _fork_collective_rpc_async(
        self, method: str, *args, **kwargs
    ) -> list[Any]:
        """Collective RPC scoped to fork engines only (never surviving engines)."""
        return await self._generic_collective_rpc_async(
            method, self.fork_workers, self._fork_engine_name, *args, **kwargs
        )

    def _fork_collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._fork_collective_rpc_async, method, *args, **kwargs)

    def reclaim_and_teardown_actor_fork(self) -> None:
        """Reclaim in-flight requests from fork engines and tear them down.

        Step-0 reclamation state machine (called after ``prepare_batch``
        returns a full batch, before the actor is onloaded back):

        1. ``dispatcher.pause()`` — stop the producer from pulling new tasks.
        2. Snapshot in-flight fork tasks (task_id + original input) before
           evict, so continuations can be re-submitted with the original data.
        3. Fork-only ``evict`` — sets the eviction flag and aborts in-flight
           vLLM requests (``pause_generation``). Each in-flight ``agenerate``
           loop breaks and returns its partial tokens; the workflow surfaces
           them as a continuation descriptor, which the per-task coroutine
           deposits into ``ForkCoordinator`` (``self._fork``).
        4. Wait for fork task coroutines to settle (descriptors deposited).
           Tasks that never settle are abandoned: their future is resolved with
           a sentinel, so their coroutine rejects instead of accepting a late
           result, and step 7 replays the original input.
        5. Collect deposited descriptors; remove fork workers from the pool.
           Snapshot tasks are classified into continuation / completed /
           restart; only tasks reclaim itself abandoned at the settle timeout
           are restarted (that coroutine can no longer accept a result).
        6. ``destroy`` fork engines + ``scheduler.delete_workers``.
        7. Re-submit each continuation to a surviving engine (rid-pinned: all
           samples of one input land on the same surviving vLLM), then re-submit
           each lost task's original input as a full restart (new task_id,
           priority, same group_size).
        8. ``dispatcher.resume()`` — surviving engines drain continuations and
           restarts.
        """
        if not self._fork_started:
            return
        cleanup_error: Exception | None = None

        # 1. Stop dispatching new tasks.
        self.dispatcher.pause()
        reclaim_start = time.perf_counter()

        # 2. Snapshot in-flight fork tasks before evict.
        with self._task_worker_map_lock:
            self.workers = [
                w for w in self.workers if w.id not in self._fork_worker_ids
            ]
            if self.workers:
                self._current_worker_idx %= len(self.workers)
            fork_snapshot: list[tuple[int, _RemoteRolloutTaskInput]] = [
                (tid, self._task_input_map[tid])
                for tid, wid in self._task_worker_map.items()
                if wid in self._fork_worker_ids and tid in self._task_input_map
            ]
        logger.info(
            f"[step0-fork] reclaim: snapshotted {len(fork_snapshot)} in-flight "
            f"fork task(s): {[tid for tid, _ in fork_snapshot]}"
        )

        # 3. Evict fork engines: set flag + abort in-flight. In-flight
        # agenerate loops break and return partial descriptors, which the
        # per-task coroutines deposit into ForkCoordinator (self._fork).
        logger.info(
            f"[step0-fork] reclaim: evicting {len(self.fork_workers)} fork engine(s) "
            "(set flag + abort in-flight)"
        )
        try:
            self._fork_collective_rpc("evict", http_timeout=self.config.request_timeout)
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc
            logger.error(f"[step0-fork] reclaim: evict on fork engines failed: {exc}")

        # 4. Wait for fork task coroutines to settle. Bound the wait so a
        # stuck fork task cannot deadlock training. Tasks still pending at the
        # deadline are abandoned here and replayed in step 7.
        abandoned_tids: set[int] = set()
        settle_timeout = max(30.0, (self.config.pause_grace_period or 0.0) + 30.0)
        deadline = time.perf_counter() + settle_timeout
        while True:
            with self._task_worker_map_lock:
                pending_fork_tasks = [
                    tid
                    for tid, wid in self._task_worker_map.items()
                    if wid in self._fork_worker_ids
                ]
            if not pending_fork_tasks:
                logger.info(
                    f"[step0-fork] reclaim: all fork task(s) settled in "
                    f"{time.perf_counter() - reclaim_start:.1f}s"
                )
                break
            if time.perf_counter() >= deadline:
                logger.warning(
                    f"[step0-fork] reclaim: {len(pending_fork_tasks)} fork task(s) "
                    f"did not settle within {settle_timeout}s; abandoning them "
                    f"(full restart) and forcing teardown "
                    f"(pending={pending_fork_tasks})."
                )
                # Resolve the stuck futures with the ABANDONED sentinel: the
                # coroutine stops waiting (no request_timeout hang) and rejects
                # without fetching a result, so the full restart below cannot
                # duplicate it. A future the engine callback already resolved is
                # not popped here -- that task stays owned by its coroutine and
                # is not restarted.
                with self._futures_lock:
                    for tid in pending_fork_tasks:
                        fut = self._pending_futures.pop(tid, None)
                        if fut is not None and not fut.done():
                            abandoned_tids.add(tid)
                            try:
                                fut.get_loop().call_soon_threadsafe(
                                    fut.set_result, _FORK_TASK_ABANDONED
                                )
                            except Exception:  # noqa: BLE001
                                pass
                break
            time.sleep(0.2)

        # 5. Collect deposited continuation descriptors; remove fork workers
        # from the dispatch pool so re-submitted continuations route to
        # surviving engines via round-robin (_choose_worker).
        #
        # Split the pre-evict snapshot three ways so the "lost" count is exact:
        #   (a) yielded a descriptor            -> re-submit a continuation
        #   (b) completed normally on the fork  -> already accepted (path A),
        #       engine before evict took effect    no continuation needed
        #   (c) yielded nothing at all          -> re-submit a FULL RESTART of
        #       (settle timeout / workflow failure)  the original input (new
        #       task_id, priority, same group_size), instead of dropping it
        # (b) is recorded by ForkCoordinator.localize_if_fork when the normal
        # trajectory is accepted (task_id -> accepted rollout count). The
        # previous `snapshot - descriptors` count misreported (b) as lost: e.g.
        # a sample that hit max_tokens exactly before the abort lands returns a
        # normal "length" trajectory, not a descriptor (remaining == 0).
        #
        # A "task" is one prompt submission = ``group_size`` (n_samples)
        # rollouts. Accounting is reported per task AND aggregated in rollouts:
        # within a descriptor task the samples split into interrupted (partial
        # -> continuation) + parked (completed before evict, re-merged later) +
        # lost (returned None). A (c) task has no partial and no trajectory at
        # all, so the whole group is re-run -- nothing it produced is reused and
        # nothing already accepted is recomputed.
        fork_continuations: list[
            tuple[int, _RemoteRolloutTaskInput, dict[str, Any]]
        ] = []
        with self._task_worker_map_lock:
            for tid, task_input in fork_snapshot:
                desc = self._fork.pop_continuation(tid)
                if desc is not None:
                    fork_continuations.append((tid, task_input, desc))
        descriptor_by_tid = {tid: desc for tid, _, desc in fork_continuations}
        completed_by_tid = self._fork.pop_completed_task_ids()
        # Tasks reclaim abandoned at the settle timeout (their future was
        # resolved with _FORK_TASK_ABANDONED) are replayed in step 7 with their
        # original input as brand-new n_samples groups.
        restart_tasks = [
            (tid, task_input)
            for tid, task_input in fork_snapshot
            if tid in abandoned_tids
        ]
        lost_tids: list[int] = []

        total_rollouts = interrupted_rollouts = parked_rollouts = 0
        completed_rollouts = lost_rollouts = 0
        for tid, task_input in fork_snapshot:
            n_rollouts = task_input.group_size
            total_rollouts += n_rollouts
            desc = descriptor_by_tid.get(tid)
            if desc is not None:
                interrupted = len(desc.get(FORK_CONTINUATION_KEY) or [])
                parked_group = desc.get(FORK_COMPLETE_KEY)
                parked = (
                    parked_group["input_ids"].shape[0]
                    if parked_group is not None
                    else 0
                )
                lost = max(n_rollouts - interrupted - parked, 0)
                interrupted_rollouts += interrupted
                parked_rollouts += parked
                lost_rollouts += lost
                logger.info(
                    f"[step0-fork] reclaim: task_id={tid} "
                    f"rollouts={n_rollouts}: {interrupted} interrupted "
                    "(partial -> continuation), "
                    f"{parked} completed on fork (parked for re-merge), "
                    f"{lost} lost"
                )
            elif tid in completed_by_tid:
                accepted = completed_by_tid[tid]
                lost = max(n_rollouts - accepted, 0)
                completed_rollouts += accepted
                lost_rollouts += lost
                logger.info(
                    f"[step0-fork] reclaim: task_id={tid} "
                    f"rollouts={n_rollouts}: {accepted} completed on fork "
                    "(accepted, no continuation needed)"
                )
            else:
                lost_tids.append(tid)
                lost_rollouts += n_rollouts
                suffix = (
                    " -> re-submitting as a full restart"
                    if tid in abandoned_tids
                    else ""
                )
                logger.info(
                    f"[step0-fork] reclaim: task_id={tid} "
                    f"rollouts={n_rollouts}: {n_rollouts} lost "
                    f"(no descriptor, no trajectory){suffix}"
                )
        if lost_tids:
            logger.warning(
                f"[step0-fork] reclaim: {len(lost_tids)}/{len(fork_snapshot)} "
                "fork task(s) yielded neither a continuation descriptor nor an "
                "accepted trajectory (settle timeout / workflow failure): "
                f"{lost_tids}"
            )

        if self.workers:
            self._current_worker_idx %= len(self.workers)
        logger.info(
            f"[step0-fork] reclaim: rollout accounting across "
            f"{len(fork_snapshot)} fork task(s) ({total_rollouts} rollouts): "
            f"{completed_rollouts} completed (accepted) + "
            f"{parked_rollouts} parked for re-merge + "
            f"{interrupted_rollouts} interrupted (re-submitted as "
            f"continuations) + {lost_rollouts} lost"
            + (
                f" ({len(restart_tasks)} re-submitted as full restarts)"
                if restart_tasks
                else ""
            )
            + f"; collected {len(fork_continuations)} continuation descriptor(s); "
            f"dispatch pool now {len(self.workers)} surviving worker(s)"
        )

        # 6. Destroy fork engines and delete fork worker processes.
        try:
            self._fork_collective_rpc("destroy", http_timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc
            logger.error(f"destroy on fork engines failed: {exc}")
        try:
            self.scheduler.delete_workers(role=self._fork_role)
        except Exception:
            logger.error(f"Error deleting fork workers: {traceback.format_exc()}")
        self.fork_workers.clear()
        for wid in self._fork_worker_ids:
            self._worker_engine_names.pop(wid, None)
        self._fork_worker_ids.clear()
        with self._task_worker_map_lock:
            self._task_worker_map.clear()
            self._task_input_map.clear()
            self._fork.clear_continuations()
        # Late callbacks (settle-timeout race) may still record path-A
        # completions after step 5 popped the set; clear defensively.
        self._fork.clear_completed_task_ids()
        self._fork_started = False

        # 7. Re-submit continuations to surviving engines with PRIORITY so the
        # dispatcher pulls them immediately (bypassing the staleness capacity
        # gate that the step-0 overlap saturates). This makes them run at
        # version 0 during the step-0 training window (surviving engines idle)
        # instead of being starved and going off-policy by step 2-3.
        resubmitted = 0
        manifest: list[dict[str, Any]] = []
        for tid, task_input, descriptor in fork_continuations:
            cont_list = descriptor.get(FORK_CONTINUATION_KEY)
            if not cont_list:
                logger.warning(
                    "[step0-fork] reclaim: descriptor missing continuation list; "
                    "skipping re-submit"
                )
                continue
            cont_data = dict(task_input.data)
            cont_data[FORK_CONTINUE_FIELD] = cont_list
            # Tag the continuation with its originating fork task_id so the
            # accept branch can find the parked already-complete rollouts (if
            # any) and re-merge into one full n_samples group.
            cont_data[FORK_MERGE_KEY] = tid
            new_id = self._task_id_generator.next()
            rids = [c.get("rid") for c in cont_list]
            logger.info(
                f"[step0-fork] reclaim: re-submitting continuation task_id={new_id} "
                f"group_size={len(cont_list)} rids={rids} to surviving engine (priority)"
            )
            try:
                self.submit(
                    data=cont_data,
                    workflow=task_input.workflow,
                    workflow_kwargs=task_input.workflow_kwargs,
                    should_accept_fn=task_input.should_accept_fn,
                    task_id=new_id,
                    is_eval=task_input.is_eval,
                    group_size=len(cont_list),
                    proxy_addr=task_input.proxy_addr,
                    priority=True,
                )
                resubmitted += 1
                manifest.append(
                    {"cont_task_id": new_id, "rids": rids, "status": "re-submitted"}
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"[step0-fork] reclaim: failed to re-submit continuation "
                    f"task_id={new_id}: {exc}"
                )
        # Full restarts for tasks that yielded neither a descriptor nor a
        # trajectory. Same priority-bypass rationale as continuations: run them
        # during the step-0 training window on the (otherwise idle) surviving
        # engines instead of letting the saturated overlap starve them. A fresh
        # task_id is used and the original input is replayed untouched (no
        # FORK_CONTINUE_FIELD / FORK_MERGE_KEY): this is a brand-new n_samples
        # group, so it can never be confused with (or merged into) the abandoned
        # task's (nonexistent) result.
        restarted = 0
        for old_tid, task_input in restart_tasks:
            new_id = self._task_id_generator.next()
            try:
                self.submit(
                    data=dict(task_input.data),
                    workflow=task_input.workflow,
                    workflow_kwargs=task_input.workflow_kwargs,
                    should_accept_fn=task_input.should_accept_fn,
                    task_id=new_id,
                    is_eval=task_input.is_eval,
                    group_size=task_input.group_size,
                    proxy_addr=task_input.proxy_addr,
                    priority=True,
                )
                restarted += 1
                logger.info(
                    f"[step0-fork] reclaim: re-submitted lost task_id={old_tid} "
                    f"as full restart task_id={new_id} "
                    f"group_size={task_input.group_size} (priority)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"[step0-fork] reclaim: failed to re-submit lost "
                    f"task_id={old_tid} as a full restart: {exc}"
                )
        # Fork-sample manifest: one structured line listing every re-submitted
        # continuation / restart (grep `fork-manifest` to trace each fork
        # sample's fate).
        logger.info(
            f"[step0-fork] fork-manifest: re-submitted {resubmitted} continuation "
            f"task(s) with priority: {manifest}"
        )
        logger.info(
            f"[step0-fork] reclaim: re-submitted {resubmitted} continuation + "
            f"{restarted} restart task(s) to surviving engines."
        )

        # 8. Resume dispatching to surviving engines.
        self.dispatcher.resume()

        if cleanup_error is not None:
            logger.error(
                f"[step0-fork] reclaim: teardown completed with errors: {cleanup_error}"
            )
        logger.info(
            f"[step0-fork] reclaim: done in {time.perf_counter() - reclaim_start:.1f}s"
        )

    def get_proxy_addr(self, rank: int) -> str:
        """Get the proxy server address for a given rollout worker rank.

        Parameters
        ----------
        rank : int
            The rank of the rollout worker

        Returns
        -------
        str
            The HTTP address of the corresponding proxy server
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if rank >= len(self.proxy_addrs):
            raise IndexError(
                f"Invalid rank {rank}, only {len(self.proxy_addrs)} proxy workers"
            )
        return self.proxy_addrs[rank]

    def start_proxy_gateway(self) -> None:
        """Start the proxy gateway for external access.

        Creates a FastAPI server that routes requests to backend proxy
        workers. Requires ``start_proxy()`` to have been called first.
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if self._proxy_gateway_host is not None:
            logger.warning("Proxy gateway already running")
            return

        from areal.experimental.openai.proxy.proxy_gateway import (
            create_proxy_gateway_app,
        )

        agent_cfg = self.config.agent

        app = create_proxy_gateway_app(
            proxy_addrs=self.proxy_addrs,
            admin_api_key=agent_cfg.admin_api_key
            if agent_cfg is not None
            else "areal-admin-key",
        )

        self._proxy_gateway_port = find_free_ports(1)[0]
        self._proxy_gateway_host = gethostip()
        self._proxy_gateway_app = app

        def serve():
            import uvicorn

            try:
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=self._proxy_gateway_port,
                    log_level="warning",
                    access_log=False,
                )
                server = uvicorn.Server(config)
                self._proxy_gateway_server = server
                server.run()
            except Exception:
                logger.error("Proxy gateway thread crashed", exc_info=True)

        self._proxy_gateway_thread = threading.Thread(target=serve, daemon=True)
        self._proxy_gateway_thread.start()

        # Wait for uvicorn to bind the port before propagating the address
        # to worker engines via collective RPC.
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                self._proxy_gateway_server is not None
                and self._proxy_gateway_server.started
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "Proxy gateway failed to start within 10s. "
                f"Cannot propagate address "
                f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
            )

        logger.info(
            "Proxy gateway started on "
            f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
        )

        # Propagate proxy_gateway_addr to all rollout worker engines
        # so that _resolve_workflow can pick it up for online mode.
        self._collective_rpc(
            "set_proxy_gateway_addr",
            addr=self.proxy_gateway_addr,
        )

    @property
    def proxy_gateway_addr(self) -> str:
        """Single URL for external users."""
        if self._proxy_gateway_host is None:
            raise RuntimeError("Proxy gateway not started")
        return f"http://{format_hostport(self._proxy_gateway_host, self._proxy_gateway_port)}"

    def _stop_proxy_gateway(self) -> None:
        """Stop the proxy gateway server if running."""
        if self._proxy_gateway_host is None:
            return
        logger.info("Stopping proxy gateway...")
        if self._proxy_gateway_server is not None:
            self._proxy_gateway_server.should_exit = True
        if self._proxy_gateway_thread is not None:
            self._proxy_gateway_thread.join(timeout=30.0)
            if self._proxy_gateway_thread.is_alive():
                logger.warning(
                    "Proxy gateway thread did not exit within 30s; "
                    "daemon thread will be killed on process exit"
                )
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread = None
        self._proxy_gateway_port = None
        self._proxy_gateway_host = None

    def _start_callback_server(self):
        """Start Flask HTTP server to receive callbacks from RolloutCallback."""
        if self._callback_server is not None:
            logger.warning("Callback server already running")
            return

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/callback/init_weights_group", methods=["POST"])
        def init_weights_group():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.init_weights_update_group(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_xccl", methods=["POST"])
        def update_weights():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            param_specs = deserialize_value(payload.get("param_specs"))
            self._callback_loop.run_until_complete(
                self.update_weights_from_distributed(meta, param_specs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_disk", methods=["POST"])
        def update_weights_disk():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.update_weights_from_disk(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_awex", methods=["POST"])
        def update_weights_awex():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            step_id = payload.get("step_id")
            kwargs = deserialize_value(payload.get("kwargs"))
            self._callback_loop.run_until_complete(
                self.update_weights_from_awex(meta, step_id=step_id, kwargs=kwargs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/pause_generation", methods=["POST"])
        def pause_generation():
            self._callback_loop.run_until_complete(self.pause_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/continue_generation", methods=["POST"])
        def continue_generation():
            self._callback_loop.run_until_complete(self.continue_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/rollout_complete", methods=["POST"])
        def rollout_complete():
            payload = request.get_json() or {}
            task_id = payload.get("task_id")
            try:
                self._resolve_task_future(task_id)
                return jsonify({"status": "ok"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.errorhandler(Exception)
        def handle_error(e):
            logger.error(f"Callback handler error: {e}")
            return jsonify({"error": str(e)}), 500

        self._callback_port = find_free_ports(1)[0]
        self._callback_host = gethostip()
        self._callback_app = app
        self._callback_server = make_server(
            self._callback_host, self._callback_port, app, threaded=False
        )

        # Suppress Werkzeug access logs (e.g., "POST /callback/rollout_complete 200 -")
        # Override log_request directly on the request handler class
        self._callback_server.RequestHandlerClass.log_request = (
            lambda self, *args, **kwargs: None
        )

        # Also configure Werkzeug logger level for any other log messages
        import logging as stdlib_logging

        werkzeug_logger = stdlib_logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(stdlib_logging.WARNING)

        def serve_forever():
            # Create and set event loop for this thread
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            # Signal that the loop is ready
            self._callback_loop_ready.set()
            logger.info(
                f"Callback server started on {format_hostport(self._callback_host, self._callback_port)}"
            )
            self._callback_server.serve_forever()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()
        # Wait for loop to be created
        self._callback_loop_ready.wait()

    def _stop_callback_server(self):
        """Stop the callback server if running."""
        if self._callback_server is not None:
            logger.info("Stopping callback server...")
            self._callback_server.shutdown()
            if self._callback_loop is not None:
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()

    @property
    def callback_addr(self) -> str:
        """Return callback server address as 'host:port'."""
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return format_hostport(self._callback_host, self._callback_port)

    def _resolve_task_future(self, task_id: int):
        """Resolve a pending future with the task result."""
        with self._futures_lock:
            future = self._pending_futures.pop(task_id, None)
        if future:
            future.get_loop().call_soon_threadsafe(future.set_result, None)

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._collective_rpc_async, method, *args, **kwargs)

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.workers, self._engine_name, *args, **kwargs
        )

    def _proxy_collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._proxy_collective_rpc_async, method, *args, **kwargs)

    async def _proxy_collective_rpc_async(
        self, method: str, *args, **kwargs
    ) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.proxy_workers, self._proxy_engine_name, *args, **kwargs
        )

    async def _generic_collective_rpc_async(
        self,
        method: str,
        workers: list[Worker],
        engine_name_fn: Callable[[int], str],
        *args,
        **kwargs,
    ) -> list[Any]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=engine_name_fn(rank),
                *args,
                **kwargs,
            )
            for rank, worker in enumerate(workers)
        ]
        return await asyncio.gather(*tasks)

    def _choose_worker(self) -> tuple[Worker, str]:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        tuple[Worker, str]
            The chosen worker object and the engine name registered on it.
            Engine name is looked up from ``_worker_engine_names`` so it matches
            the worker whether it is a surviving ("rollout/<rank>") or a fork
            ("actor-fork/<rank>") engine.
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        worker = self.workers[self._current_worker_idx]
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        engine_name = self._worker_engine_names.get(worker.id)
        if engine_name is None:
            raise RuntimeError(
                f"Chosen worker {worker.id} has no registered engine name "
                f"(known: {self._worker_engine_names})"
            )
        return worker, engine_name

    def _resolve_workflow_str(self, workflow: WorkflowLike | None) -> str | None:
        """Resolve workflow to a string import path.

        Handles RolloutWorkflow, agent workflow instances/classes, string paths,
        and ``None`` (online mode).
        """
        # None workflow = online mode (config-driven)
        if workflow is None:
            return None

        # String paths - return as-is
        if isinstance(workflow, str):
            return workflow

        # RolloutWorkflow classes
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"

        # RolloutWorkflow instances
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

        # Agent-like workflow classes
        elif isinstance(workflow, type):
            return f"{workflow.__module__}.{workflow.__name__}"

        # Agent-like workflow instances
        else:
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            task_id = pending_task.task_id

            # Record which worker this task landed on, so reclaim can locate
            # in-flight tasks on actor-GPU fork engines (abort + redispatch).
            with self._task_worker_map_lock:
                worker, engine_name = self._choose_worker()
                self._task_worker_map[task_id] = worker.id
                self._task_input_map[task_id] = pending_task

            manager = self.staleness_manager

            try:
                # Set future for this task
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    # Proxy workers are 1:1 with surviving rollout workers and
                    # indexed by their rollout rank. engine_name is
                    # "rollout/<rank>" for surviving workers. (Fork and proxy do
                    # not co-occur: fork is separation-deployment standard
                    # rollout, proxy is AgentWorkflow online mode.)
                    _rollout_rank = int(engine_name.rsplit("/", 1)[-1])
                    proxy_addr = self.get_proxy_addr(_rollout_rank)
                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=pending_task.workflow_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                # Wait for callback to resolve the future. Reclaim's settle
                # timeout resolves it with _FORK_TASK_ABANDONED instead: it owns
                # the task and re-submits the original input as a full restart,
                # so reject here without fetching (no duplicate prompt group).
                signal = await asyncio.wait_for(
                    future, timeout=self.config.request_timeout
                )
                if signal is _FORK_TASK_ABANDONED:
                    manager.on_rollout_rejected()
                    with self._task_worker_map_lock:
                        self._task_worker_map.pop(task_id, None)
                        self._task_input_map.pop(task_id, None)
                    return None

                # Fetch the result
                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,  # A short time to prevent blocking other requests
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                # Fork salvage (deposit evicted partial + re-merge + localize +
                # path A log) is a single hook on the fork strategy. Identity
                # (NoOpForkCoordinator) when the feature is off, so this hot
                # path carries zero fork overhead in the normal case. Returns
                # the trajectory to accept, or None to reject (descriptor / None
                # result); the map pop below is idempotent for the descriptor
                # case (on_task_evicted already popped under the lock).
                traj = self._fork.post_process_result(
                    pending_task, worker, task_id, result
                )
                if traj is not None:
                    manager.on_rollout_accepted()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}"
                        )
                    with self._task_worker_map_lock:
                        self._task_worker_map.pop(task_id, None)
                        self._task_input_map.pop(task_id, None)
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    logger.info(f"Finish but reject rollout. {self._rollout_stats()}")
                with self._task_worker_map_lock:
                    self._task_worker_map.pop(task_id, None)
                    self._task_input_map.pop(task_id, None)
                return None

            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                    with self._task_worker_map_lock:
                        self._task_worker_map.pop(task_id, None)
                        self._task_input_map.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                    with self._task_worker_map_lock:
                        self._task_worker_map.pop(task_id, None)
                        self._task_input_map.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
        priority: bool = False,
    ) -> int:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        if task_id is None:
            task_id = self._task_id_generator.next()
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
            group_size=group_size,
            proxy_addr=proxy_addr,
            priority=priority,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")

        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
            )
        results = self.wait(count=len(data))
        # Return list of trajectories
        return [r for r in results if r is not None]

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=self._task_id_generator.next(),
                        group_size=group_size,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        streaming_batch_size = self.config.streaming_train_batch_size
        batch_size = (
            streaming_batch_size
            if streaming_batch_size is not None
            and streaming_batch_size < dataloader.batch_size
            else dataloader.batch_size
        )
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator,
            batch_size=batch_size,
            dynamic_bs=dynamic_bs,
            prefetch_size=dataloader.batch_size,
        )

        # Return list of trajectories
        trajectories = [r.trajectory if r is not None else None for r in results]
        return [t for t in trajectories if t is not None]

    def compute_logp(self, data: list[dict[str, Any]]) -> list[Any]:
        """Compute token log-probabilities for trajectories via remote workers."""
        if len(data) == 0:
            return []

        async def _compute():
            indexed_chunks: list[list[int]] = []
            tasks = []
            n_workers = len(self.workers)
            if n_workers == 0:
                raise RuntimeError("No workers available for compute_logp.")

            for rank, worker in enumerate(self.workers):
                idxs = list(range(rank, len(data), n_workers))
                if not idxs:
                    continue
                chunk = [data[i] for i in idxs]
                indexed_chunks.append(idxs)
                tasks.append(
                    self.scheduler.async_call_engine(
                        worker_id=worker.id,
                        method="compute_logp",
                        engine_name=self._engine_name(rank),
                        data=chunk,
                        http_timeout=self.config.request_timeout,
                    )
                )
            rpc_results = await asyncio.gather(*tasks)
            merged: list[Any] = [None] * len(data)
            for idxs, chunk_result in zip(indexed_chunks, rpc_results):
                if len(chunk_result) != len(idxs):
                    raise RuntimeError(
                        f"compute_logp result length mismatch: got {len(chunk_result)}, expected {len(idxs)}"
                    )
                for out_idx, value in zip(idxs, chunk_result):
                    merged[out_idx] = value
            return merged

        return run_async_task(_compute)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate. engine_name is looked up per-worker so it
        # matches fork engines as well as surviving ones.
        worker, engine_name = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            engine_name=engine_name,
            req=req,
        )

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="init_weights_update_group",
                engine_name=self._engine_name(rank),
                meta=meta,
                xccl_group_ranks=[rank],
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        await self._collective_rpc_async(
            "update_weights_from_distributed", meta=meta, param_specs=param_specs
        )

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        meta.clear_checkpoint_after_load = False
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)
        shutil.rmtree(meta.path, ignore_errors=True)

    async def update_weights_from_awex(
        self,
        meta: WeightUpdateMeta,
        step_id: int | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        await self._collective_rpc_async(
            "update_weights_from_awex", meta=meta, step_id=step_id, kwargs=kwargs
        )

    async def pause_generation(self):
        await self._collective_rpc_async("pause_generation")

    async def continue_generation(self):
        await self._collective_rpc_async("continue_generation")

    def offload(self) -> None:
        """Offload rollout model memory on all inference workers."""
        self._collective_rpc("offload")

    def onload(self, tags: list[str] | None = None) -> None:
        """Onload rollout model memory on all inference workers."""
        self._collective_rpc("onload", tags=tags)

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)
            if self._proxy_started:
                self._proxy_collective_rpc(
                    "set_version", version=version, http_timeout=60.0
                )

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.dispatcher.pause()
        # Proxy workers hold their own engines against the same inference
        # servers. Without pausing them too they keep issuing generation
        # requests during the weight-update window, while pause_generation is
        # clearing those servers' caches — which corrupts version attribution
        # and, for VLM rollouts, kills the engine on a stale multimodal hash.
        # Pause them before the rollout workers so that the fewest requests are
        # still in flight once pause_generation aborts and clears.
        if self._proxy_started:
            self._proxy_collective_rpc("pause", http_timeout=60.0)
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        self._collective_rpc("resume", http_timeout=60.0)
        if self._proxy_started:
            self._proxy_collective_rpc("resume", http_timeout=60.0)
        self.dispatcher.resume()

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        async def _call():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        self._collective_rpc("save_perf_tracer", step=step, force=force)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
