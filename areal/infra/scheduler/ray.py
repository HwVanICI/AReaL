# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import getpass
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import ray
import ray.exceptions
import requests
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api import Job, Scheduler, Worker
from areal.api.cli_args import (
    BaseExperimentConfig,
    NameResolveConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.infra.platforms import current_platform
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    RPCConnectionError,
    SchedulerError,
    WorkerConfigurationError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.http import get_default_connector
from areal.infra.utils.launcher import (
    get_env_vars,
    get_thread_env_vars,
)
from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.infra.utils.ray import create_resource_spec, ray_resource_type
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("RayScheduler")


def _ray_device_resource() -> str:
    device = ray_resource_type()
    return "GPU" if device == "CPU" else device


def _read_log_tail(log_file: str, lines: int = 50) -> str:
    try:
        with open(log_file) as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"[Could not read log file: {e}]"


@ray.remote
class RayWorkerProcessLauncher:
    def __init__(
        self,
        role: str,
        log_file: str,
        merged_log: str,
        env_vars: dict[str, str] | None = None,
    ):
        self.role = role
        self.log_file = log_file
        self.merged_log = merged_log
        self.env_vars = env_vars or {}
        self.host = gethostip()
        self.worker_processes: dict[str, Any] = {}
        self.backend_worker_processes: dict[str, Any] = {}
        self.visible_devices = self._get_visible_devices()

    def _get_visible_devices(self) -> list[str]:
        env_var = current_platform.device_control_env_var
        visible = os.environ.get(env_var) if env_var else None
        if visible:
            return [x for x in visible.split(",") if x != ""]
        try:
            ids = ray.get_runtime_context().get_accelerator_ids()
            device = _ray_device_resource()
            if device in ids:
                return [str(x) for x in ids[device]]
            if "GPU" in ids:
                return [str(x) for x in ids["GPU"]]
            if "NPU" in ids:
                return [str(x) for x in ids["NPU"]]
        except Exception:
            pass
        return []

    def node_info(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "visible_devices": self.visible_devices,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def alloc_ports(self, count: int) -> dict[str, Any]:
        return {"host": self.host, "ports": find_free_ports(count)}

    def start_worker(
        self,
        role: str,
        worker_index: int,
        port_count: int,
        gpu_devices: list[str],
        cmd: str | None,
        experiment_name: str,
        trial_name: str,
        name_resolve_type: str,
        nfs_record_root: str,
        etcd3_addr: str,
        fileroot: str | None,
    ) -> dict[str, Any]:
        worker_id = f"{role}/{worker_index}"
        if not cmd:
            cmd = "python -m areal.infra.rpc.rpc_server"
        if "--port" in cmd:
            raise RuntimeError(
                "Custom command should not include --port argument. "
                "The scheduler automatically allocates and provides the port."
            )

        ports = find_free_ports(port_count)
        cmd_prefix = shlex.split(cmd)
        if cmd_prefix[0].startswith("python"):
            cmd_prefix[0] = sys.executable
        cmd_suffix = [
            "--host",
            "0.0.0.0",
            "--port",
            str(ports[0]),
            "--experiment-name",
            str(experiment_name),
            "--trial-name",
            str(trial_name),
            "--role",
            role,
            "--worker-index",
            str(worker_index),
            "--name-resolve-type",
            name_resolve_type,
            "--nfs-record-root",
            nfs_record_root,
            "--etcd3-addr",
            etcd3_addr,
        ]
        if fileroot:
            cmd_suffix.extend(["--fileroot", str(fileroot)])
        full_cmd = [*cmd_prefix, *cmd_suffix]

        env = os.environ.copy()
        env.update(self.env_vars)
        if current_platform.device_control_env_var:
            env[current_platform.device_control_env_var] = ",".join(gpu_devices)

        logger.info(
            "Starting Ray worker %s on %s devices=%s ports=%s: %s",
            worker_id,
            self.host,
            gpu_devices,
            ports,
            " ".join(full_cmd),
        )
        process = run_with_streaming_logs(
            full_cmd,
            self.log_file,
            self.merged_log,
            role,
            env=env,
        )
        time.sleep(0.1)
        if process.poll() is not None:
            raise RuntimeError(
                f"Worker {worker_id} exited immediately with code {process.returncode}\n"
                f"{_read_log_tail(self.log_file)}"
            )
        self.processes[worker_id] = process
        return {
            "worker_id": worker_id,
            "ip": self.host,
            "worker_ports": [str(p) for p in ports],
            "pid": process.pid,
            "gpu_devices": gpu_devices,
        }

    def process_status(self, worker_id: str) -> dict[str, Any]:
        process = self.worker_processes.get(worker_id)
        if process is None:
            return {"exists": False, "returncode": None}
        return {"exists": True, "returncode": process.poll()}

    def launch_llm_server(
        self, backend: str, server_args: dict[str, Any]
    ) -> dict[str, Any]:
        server_args = server_args.copy()
        if backend == "sglang":
            from areal.api.cli_args import SGLangConfig

            cmd = SGLangConfig.build_cmd_from_args(server_args)
        elif backend == "vllm":
            from areal.api.cli_args import vLLMConfig

            cmd = vLLMConfig.build_cmd_from_args(server_args)
        else:
            raise RuntimeError(f"Unsupported multi-node inference backend: {backend}")

        key = (
            f"{backend}:{server_args.get('node_rank', 0)}:{server_args.get('port', '')}"
        )
        env = os.environ.copy()
        env.update(self.env_vars)
        if current_platform.device_control_env_var and self.visible_devices:
            env[current_platform.device_control_env_var] = ",".join(
                self.visible_devices
            )
        process = run_with_streaming_logs(
            cmd,
            self.log_file,
            self.merged_log,
            self.role,
            env=env,
        )
        time.sleep(0.1)
        if process.poll() is not None:
            raise RuntimeError(
                f"{backend} server node {server_args.get('node_rank')} exited "
                f"immediately with code {process.returncode}\n{_read_log_tail(self.log_file)}"
            )
        self.backend_worker_processes[key] = process
        return {"host": self.host, "pid": process.pid}

    def stop_all(self) -> None:
        for process in list(self.backend_worker_processes.values()):
            try:
                kill_process_tree(process.pid, timeout=5, graceful=True)
            except Exception:
                logger.warning("Failed to stop Ray-managed LLM server", exc_info=True)
        self.backend_worker_processes.clear()

        for process in list(self.worker_processes.values()):
            try:
                kill_process_tree(process.pid, timeout=5, graceful=True)
            except Exception:
                logger.warning("Failed to stop Ray-managed HTTP server", exc_info=True)
        self.worker_processes.clear()


@dataclass
class RayWorkerInfo:
    """Ray worker information."""

    worker: Worker
    role: str
    ray_job_id: int  # -1 for forked workers (managed by parent)
    task_index: int
    discovered: bool = False
    spec: SchedulingSpec | None = None
    node: str | None = None
    launchers: list[Any] = field(default_factory=list)
    placement_group: Any | None = None
    gpu_devices: list[str] = field(default_factory=list)


class RayScheduler(Scheduler):
    def __init__(
        self,
        n_gpus_per_node: int = 8,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
        startup_timeout: float = 300.0,
        health_check_interval: float = 5.0,
        enable_tms_offload: bool | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        exp_config: BaseExperimentConfig | None = None,
    ):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Get n_gpus_per_node from parameter or config
        self._n_gpus_per_node = n_gpus_per_node
        if exp_config is not None:
            self._n_gpus_per_node = exp_config.cluster.n_gpus_per_node

        # Get other params from config if provided
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.enable_tms_offload = bool(enable_tms_offload)
        if exp_config is not None:
            self.experiment_name = exp_config.experiment_name
            self.trial_name = exp_config.trial_name
            self.fileroot = exp_config.cluster.fileroot
            self.enable_tms_offload = exp_config.enable_offload
        if self.experiment_name is None or self.trial_name is None:
            raise ValueError("experiment_name and trial_name must be provided")

        # name_resolve config (exp_config overwrites direct params)
        self.name_resolve_config = NameResolveConfig(
            type=name_resolve_type,
            nfs_record_root=nfs_record_root,
            etcd3_addr=etcd3_addr,
        )
        if exp_config is not None:
            self.name_resolve_config = exp_config.cluster.name_resolve

        if self.fileroot:
            validate_shared_path(self.fileroot, "cluster.fileroot")
        if self.name_resolve_config.type == "nfs":
            validate_shared_path(
                self.name_resolve_config.nfs_record_root,
                "name_resolve.nfs_record_root",
            )

        # Reconfigure name_resolve and clear old entries
        if self.experiment_name and self.trial_name:
            name_resolve.reconfigure(self.name_resolve_config)
            name_resolve.clear_subtree(
                names.trial_root(self.experiment_name, self.trial_name)
            )

        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.exp_config = exp_config

        # Internal state
        self._workers: dict[str, list[RayWorkerInfo]] = {}
        self._jobs: dict[str, list[Any]] = {}  # role -> Ray launcher actors
        self._placement_groups: dict[str, Any] = {}

        # Colocation tracking: colocated roles reuse workers from target role
        # For forked roles, they also track target but have their own workers in _workers
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

        logger.info(
            f"Initialized RayScheduler: exp={self.experiment_name}, "
            f"trial={self.trial_name}, fileroot={self.fileroot}, "
            f"n_gpus_per_node={self.n_gpus_per_node}"
        )

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    def _log_path_of(self, role: str) -> str:
        log_path = (
            Path(self.fileroot)
            / "logs"
            / getpass.getuser()
            / self.experiment_name
            / self.trial_name
        )
        log_path.mkdir(parents=True, exist_ok=True)
        return str(log_path / f"{role}.log")

    def _merged_log_path(self) -> str:
        log_path = (
            Path(self.fileroot)
            / "logs"
            / getpass.getuser()
            / self.experiment_name
            / self.trial_name
        )
        log_path.mkdir(parents=True, exist_ok=True)
        return str(log_path / "merged.log")

    def _read_log_tail(self, role: str, lines: int = 50) -> str:
        return _read_log_tail(self._log_path_of(role), lines=lines)

    def _find_worker_by_id(self, worker_id: str) -> RayWorkerInfo | None:
        """Find worker by ID across all roles."""
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _check_worker_process_status(self, role: str) -> None:
        """Check Ray worker process status and raise if failed."""
        # For colocated/forked roles, check the target role's process status instead
        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            return self._check_worker_process_status(target_role)

        if role not in self._jobs:
            raise WorkerNotFoundError(f"Role '{role}' not found")

        for worker_info in self._workers.get(role, []):
            if not worker_info.launchers:
                continue
            try:
                status = ray.get(
                    worker_info.launchers[0].process_status.remote(worker_info.worker.id),
                    timeout=2,
                )
            except Exception as e:
                logs = self._read_log_tail(role)
                raise WorkerFailedError(
                    worker_info.worker.id,
                    -1,
                    f"Ray worker status query failed: {e}. Logs:\n{logs}",
                ) from e
            if status.get("exists") and status.get("returncode") is not None:
                logs = self._read_log_tail(role)
                raise WorkerFailedError(
                    worker_info.worker.id,
                    status["returncode"],
                    logs,
                )

    def _verify_worker_alive(self, worker_id: str) -> RayWorkerInfo:
        """Verify worker exists and job is running."""
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Check Ray worker process status
        self._check_worker_process_status(worker_info.role)

        return worker_info

    def _wait_worker_ready(self, worker_info: RayWorkerInfo, timeout: int = 60):
        tik = time.time()
        while time.time() - tik < timeout:
            if self._is_worker_ready(worker_info):
                return
            time.sleep(1)

    def _is_worker_ready(self, worker_info: RayWorkerInfo) -> bool:
        """Check if worker is ready via health endpoint."""
        if not worker_info.discovered:
            return False

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/health"

        try:
            response = requests.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _configure_worker(self, worker_info: RayWorkerInfo, worker_rank: int) -> None:
        # Wait for worker to be ready
        while not self._is_worker_ready(worker_info):
            time.sleep(0.1)

        worker_id = worker_info.worker.id
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/configure"

        try:
            response = requests.post(
                url,
                data=orjson.dumps(
                    serialize_value(
                        dict(
                            config=self.exp_config,
                            role=worker_info.role,
                            rank=worker_rank,
                        )
                    )
                ),
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )

            if response.status_code == 200:
                logger.info(f"Configuration successful on worker '{worker_id}'")
                return
            elif response.status_code == 400:
                error_detail = response.json().get("error", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(400))
            elif response.status_code == 500:
                error_detail = response.json().get("error", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(500))
            else:
                raise WorkerConfigurationError(
                    worker_id,
                    f"Unexpected status code: {response.status_code}",
                    str(response.status_code),
                )

        except requests.exceptions.ConnectionError as e:
            self._check_worker_process_status(worker_info.role)
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except requests.exceptions.Timeout as e:
            raise WorkerConfigurationError(worker_id, f"Request timed out: {e}") from e

        except WorkerConfigurationError:
            raise

        except Exception as e:
            raise WorkerConfigurationError(
                worker_id, f"Unexpected error: {str(e)}"
            ) from e

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        """Prepare scheduling specs for workers."""
        if schedulings is None or len(schedulings) == 0:
            raise ValueError(f"No scheduling specs provided for role '{role}'")

        # Amend environment variables
        for sch in schedulings:
            if sch.additional_bash_cmds:
                raise ValueError(
                    "RayScheduler does not support SchedulingSpec.additional_bash_cmds. "
                    "Use SchedulingSpec.env_vars for Ray worker environment setup."
                )
            # AReaL env var forwarding
            if self.enable_tms_offload:
                sch.env_vars.update(get_tms_env_vars())
            sch.env_vars.update(get_env_vars())
            thread_env = get_thread_env_vars(
                cpus_per_task=sch.cpu,
                existing_env_vars=sch.env_vars,
            )
            sch.env_vars.update(thread_env)

        if len(schedulings) == 1:
            # Expand single spec to all workers
            return [schedulings[0]] * num_workers
        elif len(schedulings) == num_workers:
            return list(schedulings)
        else:
            raise ValueError(
                f"Number of scheduling specs ({len(schedulings)}) must be 1 or match "
                f"number of workers ({num_workers})"
            )

    @staticmethod
    async def _wait_for_fork_ready(
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        timeout: float = 60,
    ) -> bool:
        url = f"http://{format_hostport(host, port)}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return True
            except (TimeoutError, aiohttp.ClientError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def _fork_single_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: RayWorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> RayWorkerInfo:
        """Fork a single worker asynchronously.

        Parameters
        ----------
        command : str, optional
            Custom module path to run instead of the default rpc_server.
        """
        worker_id = f"{role}/{idx}"
        guard_url = f"http://{format_hostport(target_wi.worker.ip, int(target_wi.worker.worker_ports[0]))}"

        try:
            # 1. Allocate a port on the target guard
            async with session.post(
                f"{guard_url}/alloc_ports",
                json={"count": 1},
            ) as alloc_resp:
                if alloc_resp.status != 200:
                    error_text = await alloc_resp.text()
                    raise WorkerCreationError(
                        role,
                        f"Port allocation failed for worker {idx}",
                        f"HTTP {alloc_resp.status}: {error_text}",
                    )
                alloc_data = await alloc_resp.json()
                forked_host = alloc_data["host"]
                forked_port = alloc_data["ports"][0]

            # 2. Build the full raw command
            module_path = command or "areal.infra.rpc.rpc_server"
            raw_cmd = [
                sys.executable,
                "-m",
                module_path,
                "--host",
                "0.0.0.0",
                "--port",
                str(forked_port),
                "--experiment-name",
                str(self.experiment_name),
                "--trial-name",
                str(self.trial_name),
                "--role",
                role,
                "--worker-index",
                str(idx),
            ]
            if self.name_resolve_config.type:
                raw_cmd.extend(["--name-resolve-type", self.name_resolve_config.type])
            if self.name_resolve_config.nfs_record_root:
                raw_cmd.extend(
                    ["--nfs-record-root", self.name_resolve_config.nfs_record_root]
                )
            if self.name_resolve_config.etcd3_addr:
                raw_cmd.extend(["--etcd3-addr", self.name_resolve_config.etcd3_addr])
            if self.fileroot:
                raw_cmd.extend(["--fileroot", str(self.fileroot)])

            # 3. Fork via raw_cmd
            payload = {
                "role": role,
                "worker_index": idx,
                "raw_cmd": raw_cmd,
            }
            async with session.post(
                f"{guard_url}/fork",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        f"HTTP {response.status}: {error_text}",
                    )

                result = await response.json()

                if result.get("status") != "success":
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        result.get("error", "Unknown error"),
                    )

                forked_pid = result.get("pid")

            # 4. Wait for the forked worker to become ready
            if not await self._wait_for_fork_ready(session, forked_host, forked_port):
                try:
                    async with session.post(
                        f"{guard_url}/kill_forked_worker",
                        json={"role": role, "worker_index": idx},
                    ):
                        pass
                except Exception:
                    pass
                raise WorkerCreationError(
                    role,
                    f"Forked worker {idx} failed to become ready",
                    f"Readiness timeout at {forked_host}:{forked_port}",
                )

            logger.info(
                f"Forked worker {worker_id} created at {forked_host}:{forked_port} "
                f"(pid={forked_pid}) from {target_role}/{idx}"
            )

        except aiohttp.ClientError as e:
            raise WorkerCreationError(
                role,
                f"Failed to fork worker {idx} from {target_role}/{idx}",
                str(e),
            ) from e

        worker = Worker(
            id=worker_id,
            ip=forked_host,
            worker_ports=[str(forked_port)],
            engine_ports=[],
        )
        port_cnt = len(self._workers[target_role][0].worker.worker_ports)
        if port_cnt > 1:
            async with session.post(
                f"http://{format_hostport(forked_host, forked_port)}/alloc_ports",
                json=dict(count=port_cnt - 1),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        f"HTTP {response.status}: {error_text}",
                    )
                new_ports = (await response.json())["ports"]
                worker.worker_ports += list(map(str, new_ports))

        return RayWorkerInfo(
            worker=worker,
            role=role,
            ray_job_id=-1,  # Not a separate Ray job
            task_index=idx,
            discovered=True,  # Already discovered during fork
            spec=target_wi.spec,  # Inherit from target
            node=target_wi.node,  # Same node as target
            launchers=target_wi.launchers,
            placement_group=target_wi.placement_group,
            gpu_devices=target_wi.gpu_devices,
        )

    async def _kill_forked_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: RayWorkerInfo,
    ) -> None:
        """Kill a single forked worker via its parent's RPC server."""
        target_url = f"http://{format_hostport(target_wi.worker.ip, int(target_wi.worker.worker_ports[0]))}/kill_forked_worker"

        try:
            payload = {"role": role, "worker_index": idx}
            async with session.post(
                target_url,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to kill forked worker {role}/{idx}: "
                        f"HTTP {response.status}: {error_text}"
                    )
                else:
                    result = await response.json()
                    logger.info(
                        result.get("message", f"Killed forked worker {role}/{idx}")
                    )
        except Exception as e:
            logger.warning(f"Exception killing forked worker {role}/{idx}: {e}")

    async def _cleanup_forked_workers_async(
        self,
        role: str,
        target_role: str,
        workers: list[RayWorkerInfo],
    ) -> None:
        """Cleanup forked workers by calling kill endpoint on parent workers."""
        target_workers = self._workers.get(target_role, [])
        if not target_workers:
            logger.warning(
                f"Cannot cleanup forked workers: target role '{target_role}' not found"
            )
            return

        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            tasks = []
            for worker_info in workers:
                worker_index = int(worker_info.worker.id.split("/")[-1])
                if worker_index < len(target_workers):
                    tasks.append(
                        self._kill_forked_worker(
                            session, role, worker_index, target_workers[worker_index]
                        )
                    )
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _create_forked_workers_async(
        self,
        role: str,
        target_role: str,
        target_workers: list[RayWorkerInfo],
        command: str | None = None,
    ) -> list[str]:
        """Create forked workers concurrently using async requests.

        Parameters
        ----------
        command : str, optional
            Custom module path to run instead of the default rpc_server.
            If specified, the forked processes run this module.
        """
        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            # Launch all fork requests concurrently with exception handling
            tasks = [
                self._fork_single_worker(
                    session, role, idx, target_wi, target_role, command
                )
                for idx, target_wi in enumerate(target_workers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful workers from failures
        workers = []
        failed_indices = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed_indices.append(idx)
                logger.error(
                    f"Failed to fork worker {role}/{idx} from {target_role}/{idx}: {result}"
                )
            else:
                workers.append(result)

        # If any fork failed, cleanup successful workers and raise
        if failed_indices:
            if workers:
                logger.warning(
                    f"Cleaning up {len(workers)} successfully forked workers due to partial failure"
                )
                # Kill the forked processes via parent RPC servers
                try:
                    await self._cleanup_forked_workers_async(role, target_role, workers)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup forked workers: {cleanup_error}")

            raise WorkerCreationError(
                role,
                f"Failed to fork {len(failed_indices)} out of {len(target_workers)} workers",
                f"Failed indices: {failed_indices}",
            )

        self._workers[role] = list(workers)
        self._colocated_roles[role] = target_role
        worker_ids = [w.worker.id for w in workers]

        logger.info(
            f"Role '{role}' forked from '{target_role}': "
            f"created {len(workers)} new worker processes"
        )

        # Configure forked workers if exp_config is available
        if self.exp_config is not None:
            for worker_rank, worker_info in enumerate(workers):
                self._configure_worker(worker_info, worker_rank)

        return worker_ids

    def fork_workers(
        self,
        role: str,
        target_role: str,
        command: str | None = None,
    ) -> list[str]:
        """Fork new worker processes from existing workers.

        Creates new worker processes by forking from existing workers of the target role.
        The forked workers are colocated on the same nodes as their target workers.

        Parameters
        ----------
        role : str
            Role name for the new forked workers (e.g., "proxy")
        target_role : str
            Role of existing workers to fork from (e.g., "rollout")
        command : str, optional
            Custom module path to run instead of the default rpc_server.
            If specified, the forked process runs this module.

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["proxy/0", "proxy/1"])
        """
        if target_role not in self._workers:
            raise WorkerNotFoundError(f"Target role '{target_role}' not found for fork")
        target_workers = self._workers[target_role]

        try:
            return run_async_task(
                self._create_forked_workers_async,
                role,
                target_role,
                target_workers,
                command,
            )
        except Exception:
            # Cleanup on failure
            if role in self._workers:
                del self._workers[role]
            if role in self._colocated_roles:
                del self._colocated_roles[role]
            raise

    def _create_placement_group(
        self, role: str, bundles: list[dict[str, Any]], timeout: float
    ) -> Any:
        """Generate Ray placement group for worker job with bundle-per-node layout."""
        pg = placement_group(bundles=bundles, strategy="PACK")
        try:
            ray.get(pg.ready(), timeout=timeout)
        except ray.exceptions.GetTimeoutError as e:
            remove_placement_group(pg)
            logger.error(
                "Ray placement group timeout, please check if the resource requirement "
                "for your experiment exceeds the available resources in the cluster. \n"
                f"ray.nodes(): {ray.nodes()} \n"
                f"Placement Group bundles: {bundles}"
            )
            raise WorkerCreationError(
                role,
                "Ray placement group timeout",
                f"Placement Group bundles: {bundles}",
            ) from e
        except Exception as e:
            remove_placement_group(pg)
            raise WorkerCreationError(
                role,
                "Ray placement group creation failed",
                f"{type(e).__name__}: {e}",
            ) from e
        return pg

    def _build_node_plan(
        self, replicas: int, spec: SchedulingSpec
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        if spec.gpu <= 0:
            bundles = [
                {"CPU": spec.cpu * replicas, "memory": spec.mem * replicas * 1024**3}
            ]
            return bundles, [dict(bundle_index=0, node_rank=0, workers=replicas)], 1

        if spec.gpu > self.n_gpus_per_node:
            if spec.gpu % self.n_gpus_per_node != 0:
                raise ValueError(
                    "Ray multi-node instances must use an integer number of nodes. "
                    f"Requesting {spec.gpu} GPUs but each node has {self.n_gpus_per_node}."
                )
            nodes_per_worker = spec.gpu // self.n_gpus_per_node
            if spec.cpu % nodes_per_worker != 0 or spec.mem % nodes_per_worker != 0:
                raise ValueError(
                    "Ray multi-node instances must evenly split CPU and memory "
                    "across nodes. "
                    f"Requesting cpu={spec.cpu}, mem={spec.mem} across "
                    f"{nodes_per_worker} nodes."
                )
            per_node_cpu = spec.cpu // nodes_per_worker
            per_node_mem = spec.mem // nodes_per_worker
            bundles = []
            plan = []
            for worker_idx in range(replicas):
                for node_rank in range(nodes_per_worker):
                    bundle_index = len(bundles)
                    bundles.append(
                        {
                            "CPU": per_node_cpu,
                            _ray_device_resource(): float(self.n_gpus_per_node),
                            "memory": per_node_mem * 1024**3,
                        }
                    )
                    plan.append(
                        dict(
                            bundle_index=bundle_index,
                            worker_idx=worker_idx,
                            node_rank=node_rank,
                            workers=1 if node_rank == 0 else 0,
                        )
                    )
            return bundles, plan, nodes_per_worker

        total_gpus = spec.gpu * replicas
        bundles = []
        plan = []
        remaining_workers = replicas
        while remaining_workers > 0:
            workers_on_node = min(remaining_workers, self.n_gpus_per_node // spec.gpu)
            gpus_on_node = workers_on_node * spec.gpu
            bundle_index = len(bundles)
            bundles.append(
                {
                    "CPU": spec.cpu * workers_on_node,
                    _ray_device_resource(): float(gpus_on_node),
                    "memory": spec.mem * workers_on_node * 1024**3,
                }
            )
            plan.append(
                dict(
                    bundle_index=bundle_index,
                    node_rank=0,
                    workers=workers_on_node,
                    gpus_on_node=gpus_on_node,
                )
            )
            remaining_workers -= workers_on_node

        if (
            total_gpus >= self.n_gpus_per_node
            and total_gpus % self.n_gpus_per_node != 0
        ):
            logger.warning(
                "Ray role uses a partial final node: total_gpus=%s, n_gpus_per_node=%s. "
                "This is allowed for Ray but differs from Slurm's full-node-only policy.",
                total_gpus,
                self.n_gpus_per_node,
            )
        return bundles, plan, 1

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """Create workers via Ray placement group creation.

        Parameters
        ----------
        job : Job
            Job specification with replicas, tasks, and scheduling strategy

        Returns
        -------
        list[str]
            List of worker IDs created

        Raises
        ------
        WorkerCreationError
            If worker creation fails
        """
        role = job.role
        replicas = job.replicas
        if ":" in role:
            raise ValueError("Invalid worker name.")
        num_workers = job.replicas

        # Validation
        if role in self._workers:
            raise WorkerCreationError(role, f"Role '{role}' already exists")
        if num_workers <= 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        # Prepare scheduling specs
        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)

        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target
        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        # Determine node allocation and handle colocation
        if strategy_type == SchedulingStrategyType.colocation:
            colocate_role = strategy.target
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(
                    f"Cannot colocate with role '{colocate_role}' - role not found"
                )

            target_workers = self._workers[colocate_role]
            if num_workers != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    f"Colocated role must have same replica count as target "
                    f"({num_workers} != {len(target_workers)})",
                )

            # Check if fork mode is enabled
            if strategy.fork:
                # Fork mode: spawn new processes on same nodes via /fork endpoint
                return self.fork_workers(role, colocate_role)

            # Reuse existing workers - no new Ray job submitted
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: calculate nodes needed and submit new Ray job
        spec = schedulings[0]
        total_gpus = spec.gpu * replicas

        # Calculate resource requirements
        nodes = max(1, (total_gpus + self.n_gpus_per_node - 1) // self.n_gpus_per_node)
        n_gpus_per_node = min(
            self.n_gpus_per_node, (spec.gpu * replicas + nodes - 1) // nodes
        )
        cpus_per_task = spec.cpu
        mem_per_task = spec.mem * 1024  # Convert GB to MB

        logger.info(
            f"Creating {replicas} workers for role '{role}': "
            f"nodes={nodes}, gpus_per_node={n_gpus_per_node}, "
            f"cpus={cpus_per_task}, mem={mem_per_task}MB"
        )

        launchers = []
        try:
            bundles, plan, nodes_per_worker = self._build_node_plan(replicas, spec)
            pg = self._create_placement_group(role, bundles, self.startup_timeout)
            self._placement_groups[role] = pg

            for item in plan:
                bundle = bundles[item["bundle_index"]]
                gpu_count = int(bundle.get(_ray_device_resource(), 0))
                cpu_count = int(bundle.get("CPU", 0))
                mem_gb = max(1, int(bundle.get("memory", 0) // 1024**3))
                options = create_resource_spec(
                    _ray_device_resource(), cpu_count, gpu_count, mem_gb * 1024**3
                )
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=item["bundle_index"],
                    placement_group_capture_child_tasks=True,
                )
                launcher = RayWorkerProcessLauncher.options(**options).remote(
                    role,
                    self._log_path_of(role),
                    self._merged_log_path(),
                    spec.env_vars,
                )
                launchers.append((item, launcher))

            node_infos = ray.get(
                [launcher.node_info.remote() for _, launcher in launchers]
            )
            ordered = sorted(
                zip(launchers, node_infos, strict=True),
                key=lambda x: (
                    x[1]["host"],
                    min([int(v) for v in x[1]["visible_devices"]] or [0]),
                    x[0][0]["bundle_index"],
                ),
            )
            all_launchers = [launcher for (item, launcher), info in ordered]
            self._jobs[role] = all_launchers

            workers = []
            worker_ids = []
            if nodes_per_worker > 1:
                grouped: dict[
                    int, list[tuple[dict[str, Any], Any, dict[str, Any]]]
                ] = {}
                for (item, launcher), info in ordered:
                    grouped.setdefault(item["worker_idx"], []).append(
                        (item, launcher, info)
                    )
                for worker_idx in range(replicas):
                    group = sorted(grouped[worker_idx], key=lambda x: x[0]["node_rank"])
                    head_item, head_launcher, head_info = group[0]
                    worker_id = f"{role}/{worker_idx}"
                    result = ray.get(
                        head_launcher.start_worker.remote(
                            role,
                            worker_idx,
                            spec.port_count,
                            head_info["visible_devices"],
                            spec.cmd,
                            self.experiment_name,
                            self.trial_name,
                            self.name_resolve_config.type,
                            self.name_resolve_config.nfs_record_root,
                            self.name_resolve_config.etcd3_addr,
                            self.fileroot,
                        )
                    )
                    worker = Worker(
                        id=worker_id,
                        ip=result["ip"],
                        worker_ports=result["worker_ports"],
                        engine_ports=[],
                    )
                    worker_info = RayWorkerInfo(
                        worker=worker,
                        role=role,
                        ray_job_id=worker_idx,
                        task_index=worker_idx,
                        discovered=True,
                        spec=spec,
                        node=head_info["host"],
                        launchers=[launcher for _, launcher, _ in group],
                        placement_group=pg,
                        gpu_devices=result["gpu_devices"],
                    )
                    workers.append(worker_info)
                    worker_ids.append(worker_id)
            else:
                next_worker_idx = 0
                for (item, launcher), info in ordered:
                    visible = info["visible_devices"]
                    workers_on_node = item["workers"]
                    for local_idx in range(workers_on_node):
                        worker_idx = next_worker_idx
                        next_worker_idx += 1
                        start = local_idx * max(1, spec.gpu)
                        end = start + max(1, spec.gpu)
                        gpu_devices = visible[start:end] if spec.gpu > 0 else []
                        worker_id = f"{role}/{worker_idx}"
                        result = ray.get(
                            launcher.start_worker.remote(
                                role,
                                worker_idx,
                                schedulings[worker_idx].port_count,
                                gpu_devices,
                                schedulings[worker_idx].cmd,
                                self.experiment_name,
                                self.trial_name,
                                self.name_resolve_config.type,
                                self.name_resolve_config.nfs_record_root,
                                self.name_resolve_config.etcd3_addr,
                                self.fileroot,
                            )
                        )
                        worker = Worker(
                            id=worker_id,
                            ip=result["ip"],
                            worker_ports=result["worker_ports"],
                            engine_ports=[],
                        )
                        worker_info = RayWorkerInfo(
                            worker=worker,
                            role=role,
                            ray_job_id=worker_idx,
                            task_index=worker_idx,
                            discovered=True,
                            spec=schedulings[worker_idx],
                            node=info["host"],
                            launchers=[launcher],
                            placement_group=pg,
                            gpu_devices=result["gpu_devices"],
                        )
                        workers.append(worker_info)
                        worker_ids.append(worker_id)

            self._workers[role] = workers

            logger.info(f"Created {replicas} workers for role '{role}' with Ray")
        except WorkerCreationError:
            if role in self._jobs:
                for launcher in self._jobs[role]:
                    try:
                        ray.get(launcher.stop_all.remote(), timeout=10)
                    except Exception:
                        pass
                del self._jobs[role]
            if role in self._placement_groups:
                remove_placement_group(self._placement_groups[role])
                del self._placement_groups[role]
            raise
        except Exception as e:
            if role in self._jobs:
                for launcher in self._jobs[role]:
                    try:
                        ray.get(launcher.stop_all.remote(), timeout=10)
                    except Exception:
                        pass
                del self._jobs[role]
            if role in self._placement_groups:
                remove_placement_group(self._placement_groups[role])
                del self._placement_groups[role]
            logs = self._read_log_tail(role)
            raise WorkerCreationError(
                role,
                "Ray worker creation failed",
                f"{type(e).__name__}: {e}\nLogs:\n{logs}",
            ) from e

        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        """Wait for workers to be ready and return their information.

        Parameters
        ----------
        role : str
            Role name to query
        timeout : float, optional
            Maximum wait time in seconds

        Returns
        -------
        list[Worker]
            List of ready workers

        Raises
        ------
        WorkerNotFoundError
            If role doesn't exist
        WorkerTimeoutError
            If timeout exceeded
        WorkerFailedError
            If workers failed
        """
        # Handle colocated/forked roles
        if role in self._colocated_roles:
            # Forked roles have their own workers in _workers
            if role in self._workers:
                workers = self._workers[role]
                # Forked workers are already discovered and configured during creation
                # Just verify they're still healthy
                for worker_info in workers:
                    if not self._is_worker_ready(worker_info):
                        raise WorkerFailedError(
                            worker_info.worker.id, -1, "Forked worker not responding"
                        )
                logger.info(
                    f"All {len(workers)} forked workers ready for role '{role}'"
                )
                return [w.worker for w in workers]
            else:
                # Colocated roles delegate to target role's workers
                target_role = self._colocated_roles[role]
                logger.debug(
                    f"Role '{role}' is colocated with '{target_role}', "
                    "returning target role's workers"
                )
                return self.get_workers(target_role, timeout)

        if role not in self._workers:
            raise WorkerNotFoundError(f"Role '{role}' not found")

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout
        start_time = time.time()

        logger.info(
            f"Waiting for {len(workers)} workers of role '{role}' to be ready..."
        )

        while time.time() - start_time < timeout:
            # Check job status
            try:
                self._check_worker_process_status(role)
            except WorkerFailedError:
                raise

            # Health check all workers
            ready_workers = []

            for worker_info in workers:
                if self._is_worker_ready(worker_info):
                    ready_workers.append(worker_info)

            # All ready
            if len(ready_workers) == len(workers):
                logger.info(f"All {len(workers)} workers ready for role '{role}'")

                # Configure workers if exp_config is available
                if self.exp_config is not None:
                    for worker_rank, worker_info in enumerate(workers):
                        self._configure_worker(worker_info, worker_rank)

                return [w.worker for w in workers]

            # Log progress
            if ready_workers:
                logger.debug(f"{len(ready_workers)}/{len(workers)} workers ready")

            time.sleep(self.health_check_interval)

        raise WorkerTimeoutError(role, timeout)

    def _destroy_engines_on_workers(
        self, workers: list[RayWorkerInfo], timeout: float = 30.0
    ) -> None:
        """Call ``engine.destroy()`` on every worker via HTTP before killing jobs.

        All calls are dispatched concurrently so that the engine-side CPU
        barrier (``dist.barrier`` + ``dist.destroy_process_group``) can
        complete across all ranks.  A bounded *timeout* prevents indefinite
        blocking when a worker is already unreachable.
        """
        if not workers:
            return

        async def _destroy_all():
            destroy_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(
                timeout=destroy_timeout,
                connector=get_default_connector(),
            ) as session:
                tasks = []
                for wi in workers:
                    port = int(wi.worker.worker_ports[0])
                    url = f"http://{format_hostport(wi.worker.ip, port)}/call"
                    payload = {
                        "method": "destroy",
                        "engine_name": wi.worker.id,
                        "args": serialize_value([]),
                        "kwargs": serialize_value({}),
                        "rpc_meta": None,
                    }
                    tasks.append(
                        session.post(
                            url,
                            data=orjson.dumps(payload),
                            headers={"Content-Type": "application/json"},
                        )
                    )
                results = await asyncio.gather(
                    *[self._safe_destroy_request(t) for t in tasks],
                    return_exceptions=True,
                )
                for wi, res in zip(workers, results):
                    if isinstance(res, BaseException):
                        logger.warning(
                            f"engine.destroy() on {wi.worker.id} failed: "
                            f"{type(res).__name__}: {res}"
                        )

        try:
            run_async_task(_destroy_all)
        except Exception as e:
            logger.warning(f"Failed to destroy engines before cancel: {e}")

    @staticmethod
    async def _safe_destroy_request(coro):
        """Await an aiohttp context-manager response, suppressing errors."""
        try:
            async with coro as resp:
                await resp.read()
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        """Delete workers and cancel Ray jobs.

        Teardown follows a two-phase protocol analogous to the Ray and Local
        schedulers:

        1. **Engine destroy** – call ``engine.destroy()`` on every worker via
           HTTP concurrently.  This runs the engine-side CPU barrier and
           ``dist.destroy_process_group`` so that NCCL communicators and the
           TCPStore are shut down cleanly on all ranks.
        2. **Job cancel** – stop the Ray-managed launcher actors.  At this
           point process groups are already torn down, so killing the
           processes will not produce spurious ``TCPStore.recvValue failed``
           warnings.

        Parameters
        ----------
        role : str, optional
            Role to delete. If None, deletes all roles.
        reverse_order : bool, optional
            Accepted for API compatibility with other schedulers but ignored
            here: Ray launchers tear down worker processes by launcher group,
            so per-rank ordering cannot be enforced globally.
        """
        del reverse_order  # unused, see docstring
        if role is None:
            # Delete colocated/forked roles first (they don't own Ray jobs)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r)
            # Then delete actual worker roles
            for r in list(self._workers.keys()):
                self.delete_workers(r)
            return

        # Handle colocated/forked role
        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            # Forked roles have their own workers that need cleanup
            if role in self._workers:
                logger.info(f"Removing forked role '{role}' (managed by parent worker)")
                try:
                    run_async_task(
                        self._cleanup_forked_workers_async,
                        role,
                        target_role,
                        self._workers[role],
                    )
                except Exception as e:
                    logger.warning(f"Failed to cleanup forked role '{role}': {e}")
                del self._workers[role]
            else:
                logger.info(f"Removing colocated role '{role}' mapping")
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning(f"Role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        # Phase 1: destroy engines so that the CPU barrier and
        # dist.destroy_process_group complete on every rank.
        self._destroy_engines_on_workers(workers)

        # Phase 2: cancel the Ray job. Process groups are already torn
        # down, so stopping actors will not cause TCPStore race conditions.
        for launcher in self._jobs.get(role, []):
            try:
                ray.get(launcher.stop_all.remote(), timeout=30)
            except Exception as e:
                logger.error(f"Error stopping Ray launcher for role {role}: {e}")
            try:
                ray.kill(launcher, no_restart=True)
            except Exception:
                pass

        if role in self._placement_groups:
            try:
                remove_placement_group(self._placement_groups[role])
            except Exception as e:
                logger.warning(f"Failed to remove placement group for role {role}: {e}")

        # Clean up internal state
        del self._workers[role]
        self._jobs.pop(role, None)
        self._placement_groups.pop(role, None)

        logger.info(f"Successfully deleted workers for role '{role}'")

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        """Set environment variables on a worker before engine creation.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        env : dict[str, str]
            Environment variables to set
        """
        worker_info = self._verify_worker_alive(worker_id)
        if not env:
            return

        payload = {"env": env}
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/set_env"

        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return
                    detail = (await response.json()).get("error", "Unknown error")
                    raise SchedulerError(
                        worker_id,
                        f"Failed to set env on worker (status={response.status}): {detail}",
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            self._check_worker_process_status(worker_info.role)
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e
        except TimeoutError as e:
            raise SchedulerError(worker_id, f"set_env timed out: {e}") from e

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Create an engine instance on a remote worker.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        engine : str
            Import path to engine class
        engine_name : str, optional
            Unique name for this engine instance. Defaults to worker_id.
        *args
            Initialization arguments
        **kwargs
            Initialization keyword arguments

        Returns
        -------
        Any
            Result from engine initialization

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker has failed
        EngineCreationError
            If engine creation fails
        """
        worker_info = self._verify_worker_alive(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )

        payload = {
            "engine": engine,
            "engine_name": engine_name,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/create_engine"

        try:
            logger.debug(
                f"Creating engine '{engine_name}' (class: {engine}) on worker '{worker_id}'"
            )

            timeout = aiohttp.ClientTimeout(total=300.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(
                            f"Engine created successfully on worker '{worker_id}'"
                        )
                        return result.get("result")
                    elif response.status == 400:
                        error_detail = (await response.json()).get(
                            "error", "Unknown error"
                        )
                        if "Failed to import" in error_detail:
                            raise EngineImportError(engine, error_detail)
                        else:
                            raise EngineCreationError(worker_id, error_detail, 400)
                    elif response.status == 500:
                        error_detail = (await response.json()).get(
                            "error", "Unknown error"
                        )
                        raise EngineCreationError(worker_id, error_detail, 500)
                    else:
                        raise EngineCreationError(
                            worker_id,
                            f"Unexpected status code: {response.status}",
                            response.status,
                        )

        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            self._check_worker_process_status(worker_info.role)
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except TimeoutError as e:
            raise EngineCreationError(
                worker_id, f"Engine creation timed out: {e}"
            ) from e

    def _prepare_multi_node_server_args(
        self,
        worker_info: RayWorkerInfo,
        server_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        backend = (
            "vllm"
            if "distributed_executor_backend" in server_args or "model" in server_args
            else "sglang"
        )
        head_launcher = worker_info.launchers[0]
        host_ports = ray.get(head_launcher.alloc_ports.remote(1))
        master_host = host_ports["host"]
        master_port = host_ports["ports"][0]
        n_nodes = len(worker_info.launchers)

        head_args = copy.deepcopy(server_args)
        worker_args = []
        if backend == "sglang":
            dist_init_addr = f"{master_host}:{master_port}"
            head_args.update(nnodes=n_nodes, node_rank=0, dist_init_addr=dist_init_addr)
            for node_rank in range(1, n_nodes):
                args = copy.deepcopy(server_args)
                args.pop("host", None)
                args.pop("port", None)
                args.update(
                    nnodes=n_nodes, node_rank=node_rank, dist_init_addr=dist_init_addr
                )
                worker_args.append(args)
        else:
            head_args.update(
                nnodes=n_nodes,
                node_rank=0,
                master_addr=master_host,
                master_port=str(master_port),
            )
            for node_rank in range(1, n_nodes):
                args = copy.deepcopy(server_args)
                args.pop("host", None)
                args.pop("port", None)
                args.update(
                    nnodes=n_nodes,
                    node_rank=node_rank,
                    master_addr=master_host,
                    master_port=str(master_port),
                    headless=True,
                )
                worker_args.append(args)

        return backend, head_args, worker_args

    async def _launch_multi_node_server(
        self,
        worker_info: RayWorkerInfo,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        server_args = kwargs.get("server_args")
        if not isinstance(server_args, dict) or len(worker_info.launchers) <= 1:
            return kwargs

        backend, head_args, worker_args = self._prepare_multi_node_server_args(
            worker_info, server_args
        )
        refs = []
        for launcher, args in zip(worker_info.launchers[1:], worker_args, strict=True):
            refs.append(launcher.launch_llm_server.remote(backend, args))
        if refs:
            await asyncio.to_thread(ray.get, refs)
        new_kwargs = kwargs.copy()
        new_kwargs["server_args"] = head_args
        return new_kwargs

    def call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Call a method on an engine instance (synchronous)."""
        return run_async_task(
            self.async_call_engine,
            worker_id,
            method,
            engine_name,
            *args,
            rpc_meta=rpc_meta,
            http_timeout=http_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs,
        )

    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Call a method on an engine instance (asynchronous).

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Name of method to call
        engine_name : str, optional
            Name of the engine to call. Defaults to worker_id.
        *args
            Method arguments
        http_timeout : float, default=7200.0
            HTTP request timeout in seconds
        max_retries : int, default=3
            Maximum retry attempts
        retry_delay : float, default=1.0
            Initial retry delay in seconds
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from engine method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker has failed
        EngineCallError
            If method call fails
        """
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        if method == "launch_server" and len(worker_info.launchers) > 1:
            kwargs = await self._launch_multi_node_server(worker_info, kwargs)

        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "rpc_meta": rpc_meta,
        }

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/call"
        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check job status before each attempt
            try:
                self._check_worker_process_status(worker_info.role)
            except WorkerFailedError:
                raise

            try:
                timeout = aiohttp.ClientTimeout(total=http_timeout)
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    read_bufsize=1024 * 1024 * 10,
                    connector=get_default_connector(),
                ) as session:
                    async with session.post(
                        url,
                        data=orjson.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return deserialize_value(result.get("result"))
                        elif response.status == 500:
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            if (
                                attempt < max_retries
                                and "timeout" in error_detail.lower()
                            ):
                                last_error = f"Engine method timeout: {error_detail}"
                                logger.warning(
                                    f"Retryable error on attempt {attempt}/{max_retries}: {last_error}"
                                )
                            else:
                                raise EngineCallError(
                                    worker_id, method, error_detail, attempt=attempt
                                )
                        elif response.status == 503:
                            last_error = "Service unavailable (503)"
                            logger.warning(
                                f"Worker temporarily unavailable, retry {attempt}/{max_retries}"
                            )
                        else:
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id,
                                method,
                                f"HTTP {response.status}: {error_detail}",
                                attempt=attempt,
                            )

            except TimeoutError as e:
                last_error = f"Request timeout: {e}"
                logger.warning(f"Request timeout on attempt {attempt}/{max_retries}")
            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                self._check_worker_process_status(worker_info.role)
                last_error = f"Connection error: {e}"
                logger.warning(f"Connection error on attempt {attempt}/{max_retries}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(
                    f"Unexpected error on attempt {attempt}/{max_retries}: {e}"
                )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
        )

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
