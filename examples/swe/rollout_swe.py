"""Rollout-only script for SWE-bench agents with AReaL proxy mode."""

import sys
import warnings
from pathlib import Path
from typing import Any

from examples.swe.dataset import get_swe_dataset, resolve_swe_dataset_path
from examples.swe.utils import SWEPPOConfig

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import SGLangConfig, load_expr_config, vLLMConfig
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, RolloutController, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.data import get_batch_size
from areal.utils.dataloader import create_dataloader
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("SWERollout")


def group_filter(x: dict[str, Any]):
    """Filter out groups where all rollouts already solved the task."""
    return x["rewards"].mean() <= 0.95


def _install_aweagent_deps_on_ray_nodes(aweagent_root: str):
    """Install AReaL-SWEAgent dependencies on all Ray GPU nodes.

    Each node runs in a separate container with its own venv,
    so we must ensure packages like ``aenv`` are installed everywhere.
    """
    if not aweagent_root:
        aweagent_root = str(
            Path(__file__).resolve().parents[2].parent / "AReaL-SWEAgent"
        )
    try:
        import ray

        if not ray.is_initialized():
            return

        @ray.remote(num_gpus=0)
        def _install():
            import os
            import socket
            import subprocess

            ip = socket.gethostbyname(socket.gethostname())
            req_path = os.path.join(aweagent_root, "requirements.txt")
            result = subprocess.run(
                ["uv", "pip", "install", "-r", req_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return (
                ip,
                result.returncode,
                result.stderr[-200:] if result.stderr else "",
            )

        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
        ]
        refs = []
        for node in nodes:
            node_ip = node["NodeManagerAddress"]
            refs.append(_install.options(resources={f"node:{node_ip}": 0.01}).remote())

        results = ray.get(refs, timeout=180)
        for ip, rc, err in results:
            if rc != 0:
                logger.warning(f"Failed to install AReaL-SWEAgent deps on {ip}: {err}")
            else:
                logger.info(f"AReaL-SWEAgent deps installed on {ip}")
    except Exception as e:
        logger.warning(f"Could not install AReaL-SWEAgent deps on Ray nodes: {e}")


def _resolve_aweagent_root(econfig) -> str:
    return (
        getattr(econfig, "agent_root", "")
        or getattr(econfig, "aweagent_root", "")
        or getattr(econfig, "swe_agent_root", "")
    )


def _init_scheduler(config):
    cfg = config.scheduler
    if cfg.type == "local":
        return LocalScheduler(exp_config=config)
    if cfg.type == "ray":
        return RayScheduler(exp_config=config)
    if cfg.type == "slurm":
        return SlurmScheduler(exp_config=config)
    raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")


def _configure_rollout_only(config):
    # Rollout-only keeps one immutable model version, so training staleness limits
    # must not cap how many finite-dataset requests can finish.
    config.rollout.max_head_offpolicyness = int(1e12)
    # Preserve source order and submit every item, including the final partial batch.
    config.train_dataset.shuffle = False
    config.train_dataset.drop_last = False


def _init_rollout(config, scheduler):
    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")

    if rollout_alloc.backend == "sglang":
        if config.rollout.return_routed_experts:
            config.sglang.enable_return_routed_experts = True
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
            base_gpu_id=0,
        )
    elif rollout_alloc.backend == "vllm":
        if config.rollout.return_routed_experts:
            raise ValueError(
                "return_routed_experts is not supported with vLLM backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
        )
    else:
        raise ValueError(
            f"Invalid backend: {rollout_alloc.backend}, expected sglang or vllm"
        )

    rollout = engine_cls.as_controller(config.rollout, scheduler)
    rollout.initialize(role="rollout", server_args=server_args)

    # V1 agent workflows need colocated proxy workers. V2 starts its gateway stack
    # as part of controller initialization.
    if isinstance(rollout, RolloutController):
        logger.info("Initializing proxy workers for AgentWorkflow support")
        rollout.start_proxy()

    return rollout


def _count_trajectories(rollout_batch: list[dict[str, Any]]) -> int:
    count = 0
    for data in rollout_batch:
        batch_size = get_batch_size(data)
        if batch_size == 0 and isinstance(data.get("interactions"), list):
            batch_size = len(data["interactions"])
        if batch_size == 0:
            logger.warning("Could not infer trajectory count; counting result as one")
            batch_size = 1
        count += batch_size
    return count


def _run_rollouts(config, train_dataloader, rollout, workflow_kwargs):
    workflow = "examples.swe.agent.SWEAgentWorkflow"
    should_accept_fn = getattr(config, "should_accept_fn", None)
    step_sizes = []
    submitted_data = 0

    # Traverse the finite dataloader exactly once. Submit everything up front to
    # preserve cross-step pipelining, then consume results in original batch sizes.
    for data_batch in train_dataloader:
        if not data_batch:
            continue
        step_sizes.append(len(data_batch))
        for data in data_batch:
            rollout.submit(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=config.gconfig.n_samples,
            )
            submitted_data += 1

    total_steps = len(step_sizes)

    logger.info(
        "Starting rollout-only run: dataset_items=%d, steps=%d",
        submitted_data,
        total_steps,
    )
    completed_steps = 0
    completed_data = 0
    completed_trajectories = 0
    for global_step, step_data_count in enumerate(step_sizes):
        results = rollout.wait(step_data_count, timeout=None)
        rollout_batch = [result for result in results if result is not None]
        step_trajectory_count = _count_trajectories(rollout_batch)
        completed_steps += 1
        completed_data += step_data_count
        completed_trajectories += step_trajectory_count

        stats = rollout.export_stats()
        stats.update(
            {
                "rollout/data_count": step_data_count,
                "rollout/trajectory_count": step_trajectory_count,
            }
        )
        logger.info(
            "Rollout step %d/%d done: data=%d, accepted=%d.",
            global_step + 1,
            total_steps,
            step_data_count,
            len(rollout_batch),
        )
        logger.info("Stats:\n%s", tabulate_stats(stats))

    logger.info(
        "Rollout complete: steps=%d, data=%d, trajectories=%d",
        completed_steps,
        completed_data,
        completed_trajectories,
    )


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, SWEPPOConfig)
    _configure_rollout_only(config)
    econfig = config.econfig

    # When using Ray scheduler, ensure SWEAgent deps are on all nodes
    if config.scheduler.type == "ray":
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        _install_aweagent_deps_on_ray_nodes(_resolve_aweagent_root(econfig))

    train_dataset = get_swe_dataset(
        dataset_path=resolve_swe_dataset_path(
            config.train_dataset.path, econfig.dataset_path
        ),
        split="train",
        min_items=1,
    )

    # Build workflow kwargs
    from dataclasses import asdict

    econfig_dict = asdict(econfig)
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=econfig.timeout,
    )

    seeding.set_random_seed(config.seed, key="rollout")
    train_dataloader = create_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    scheduler = _init_scheduler(config)
    rollout = None
    try:
        rollout = _init_rollout(config, scheduler)
        _run_rollouts(config, train_dataloader, rollout, workflow_kwargs)
    finally:
        if rollout is not None:
            rollout.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])
