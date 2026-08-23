"""Training script for SWE-bench agent RL with AReaL proxy mode."""

import sys
import warnings
from pathlib import Path
from typing import Any

from examples.swe.dataset import get_swe_dataset, resolve_swe_dataset_path
from examples.swe.utils import SWEPPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("SWETrain")


def group_filter(x: dict[str, Any]):
    """Filter out groups where all rollouts already solved the task."""
    return x["rewards"].mean() <= 0.95


def _install_aweagent_deps_on_ray_nodes(
    aweagent_root: str,
    sandbox_backend: str,
):
    """Install AReaL-SWEAgent dependencies on all Ray GPU nodes.

    Each node runs in a separate container with its own venv, so the selected
    sandbox client's optional dependencies must be installed everywhere.
    """
    if not aweagent_root:
        areal_root = Path(__file__).resolve().parents[2]
        bundled = areal_root / "AReaL-SWEAgent"
        aweagent_root = str(
            bundled if bundled.is_dir() else areal_root.parent / "AReaL-SWEAgent"
        )
    try:
        import ray

        if not ray.is_initialized():
            return

        @ray.remote(num_gpus=0)
        def _install():
            import socket
            import subprocess
            import sys

            ip = socket.gethostbyname(socket.gethostname())
            install_target = f"{aweagent_root}[{sandbox_backend}]"
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", install_target],
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


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, SWEPPOConfig)
    econfig = config.econfig

    # When using Ray scheduler, ensure SWEAgent deps are on all nodes
    if config.scheduler.type == "ray":
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        _install_aweagent_deps_on_ray_nodes(
            _resolve_aweagent_root(econfig),
            econfig.sandbox_backend,
        )

    train_dataset = get_swe_dataset(
        dataset_path=resolve_swe_dataset_path(
            config.train_dataset.path, econfig.dataset_path
        ),
        split="train",
        min_items=64,
    )
    valid_dataset = get_swe_dataset(
        dataset_path=resolve_swe_dataset_path(
            config.valid_dataset.path, econfig.dataset_path
        ),
        split="test",
        min_items=64,
    )

    # Build workflow kwargs
    from dataclasses import asdict

    econfig_dict = asdict(econfig)
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_tokens=config.gconfig.max_tokens,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=econfig.timeout,
    )

    # Eval workflow with lower temperature for deterministic evaluation
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gen_args"] = dict(
        temperature=0.0,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.swe.agent.SWEAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=getattr(config, "should_accept_fn", None),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
