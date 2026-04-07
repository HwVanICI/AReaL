from __future__ import annotations

import argparse
import copy
import getpass
import os
import time
from dataclasses import MISSING, dataclass, field
from typing import Any

from omegaconf import OmegaConf

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    SchedulerConfig,
    ValidDatasetConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.dataset import get_custom_dataset
from areal.engine.ray_vllm_remote import RayRemotevLLMEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.infra.scheduler.local import LocalScheduler
from areal.infra.scheduler.ray import RayScheduler
from areal.infra.scheduler.slurm import SlurmScheduler
from areal.utils import logging, name_resolve, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.printing import tabulate_stats


@dataclass
class EvalOnlyConfig:
    experiment_name: str = field(default=MISSING)
    trial_name: str = field(default=MISSING)
    workflow: str = field(default=MISSING)
    cluster: ClusterSpecConfig = field(default_factory=ClusterSpecConfig)
    allocation_mode: str = field(default="")
    seed: int = field(default=1)
    tokenizer_path: str = field(default="")
    processor_path: str | None = field(default=None)
    eval_split: str = field(default="test")
    workflow_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_workflow: str | None = field(default=None)
    eval_workflow_kwargs: dict[str, Any] | None = field(default=None)
    valid_dataset: ValidDatasetConfig = field(default_factory=ValidDatasetConfig)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: vLLMConfig = field(default_factory=vLLMConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


def parse_eval_cli_args(args: list[str]) -> tuple[list[str], int | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of evaluation samples to submit.",
    )
    parsed, remaining = parser.parse_known_args(args)
    if parsed.max_items is not None and parsed.max_items < 1:
        raise ValueError("--max-items must be >= 1")
    return remaining, parsed.max_items


def load_eval_config(argv: list[str]) -> tuple[EvalOnlyConfig, str]:
    cfg, config_file = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=EvalOnlyConfig)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, EvalOnlyConfig)
    name_resolve.reconfigure(cfg.cluster.name_resolve)
    return cfg, str(config_file)


def _configure_rollout_defaults(config: EvalOnlyConfig) -> None:
    if not config.rollout.experiment_name:
        config.rollout.experiment_name = config.experiment_name
    if not config.rollout.trial_name:
        config.rollout.trial_name = config.trial_name
    if not config.rollout.fileroot:
        config.rollout.fileroot = config.cluster.fileroot
    if not config.rollout.tokenizer_path:
        config.rollout.tokenizer_path = config.tokenizer_path
    config.rollout.max_head_offpolicyness = int(1e12)


def _get_log_path(config: EvalOnlyConfig) -> str:
    path = (
        f"{config.cluster.fileroot}/logs/{getpass.getuser()}/"
        f"{config.experiment_name}/{config.trial_name}"
    )
    os.makedirs(path, exist_ok=True)
    return path


def setup_eval_logging(config: EvalOnlyConfig, log_filename: str) -> str:
    log_dir = _get_log_path(config)
    log_path = f"{log_dir}/{log_filename}"
    logging.setup_file_logging(log_path)
    return log_path


def create_scheduler(config: EvalOnlyConfig):
    cfg = config.scheduler
    if cfg.type == "local":
        return LocalScheduler(exp_config=config)
    if cfg.type == "ray":
        return RayScheduler(exp_config=config)
    if cfg.type == "slurm":
        return SlurmScheduler(exp_config=config)
    raise ValueError(
        "scheduler.type must be set to one of {local, ray, slurm} for eval_only."
    )


def build_tokenizer_and_processor(config: EvalOnlyConfig):
    if config.processor_path:
        processor, tokenizer = load_hf_processor_and_tokenizer(config.processor_path)
        return tokenizer, processor
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    return tokenizer, None


def build_valid_dataloader(
    config: EvalOnlyConfig,
    tokenizer,
    processor=None,
):
    valid_dataset = get_custom_dataset(
        split=config.eval_split,
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    return create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )


def create_rollout_controller(
    config: EvalOnlyConfig,
    allocation_mode: AllocationMode,
    scheduler,
):
    if allocation_mode.gen_backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=allocation_mode.gen.tp_size,
            base_gpu_id=0,
        )
    elif allocation_mode.gen_backend == "vllm":
        if allocation_mode.gen_instance_size > config.cluster.n_gpus_per_node:
            engine_cls = RayRemotevLLMEngine
        else:
            engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=allocation_mode.gen.tp_size,
            pp_size=allocation_mode.gen.pp_size,
        )
    else:
        raise ValueError(f"Invalid backend: {allocation_mode.gen_backend}")

    controller = engine_cls.as_controller(config.rollout, scheduler)
    controller.initialize(
        role="eval-rollout",
        alloc_mode=allocation_mode,
        server_args=server_args,
    )
    return controller


def resolve_workflow_spec(config: EvalOnlyConfig) -> tuple[str, dict[str, Any]]:
    workflow = config.eval_workflow or config.workflow
    workflow_kwargs = config.eval_workflow_kwargs
    if workflow_kwargs is None:
        workflow_kwargs = config.workflow_kwargs
    return workflow, copy.deepcopy(workflow_kwargs)


def workflow_requires_proxy(workflow: str | None) -> bool:
    if workflow is None:
        return True
    imported = import_from_string(workflow)
    if isinstance(imported, RolloutWorkflow):
        return False
    if isinstance(imported, type):
        return not issubclass(imported, RolloutWorkflow)
    return True


def ensure_proxy_ready(controller, *, timeout_s: float = 60.0) -> None:
    controller.start_proxy()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(controller.proxy_addrs) >= len(controller.workers):
            return
        time.sleep(0.2)
    raise TimeoutError(
        f"Timed out waiting for proxy workers to become ready after {timeout_s}s"
    )


def iter_eval_items(valid_dataloader, max_items: int | None = None):
    submitted = 0
    for batch in valid_dataloader:
        for item in batch:
            yield item
            submitted += 1
            if max_items is not None and submitted >= max_items:
                return


def submit_eval_requests(
    controller,
    valid_dataloader,
    *,
    workflow: str,
    workflow_kwargs: dict[str, Any],
    group_size: int,
    max_items: int | None,
) -> int:
    submitted = 0
    for item in iter_eval_items(valid_dataloader, max_items=max_items):
        controller.submit(
            item,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
            is_eval=True,
        )
        submitted += 1
    return submitted


def run_eval(argv: list[str], *, logger_name: str, log_filename: str) -> dict[str, float]:
    remaining_args, max_items = parse_eval_cli_args(argv)
    config, config_path = load_eval_config(remaining_args)
    _configure_rollout_defaults(config)
    setup_eval_logging(config, log_filename)
    logger = logging.getLogger(logger_name)

    logger.info("Loaded eval-only config from %s", config_path)
    seeding.set_random_seed(config.seed, key=logger_name.lower())

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    scheduler = create_scheduler(config)
    tokenizer, processor = build_tokenizer_and_processor(config)
    valid_dataloader = build_valid_dataloader(
        config,
        tokenizer=tokenizer,
        processor=processor,
    )
    workflow, workflow_kwargs = resolve_workflow_spec(config)
    controller = create_rollout_controller(config, allocation_mode, scheduler)
    try:
        if workflow_requires_proxy(workflow):
            logger.info(
                "Workflow %s requires proxy workers. Starting proxy...", workflow
            )
            ensure_proxy_ready(controller)
            logger.info("Proxy workers ready: %s", controller.proxy_addrs)

        submitted = submit_eval_requests(
            controller,
            valid_dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=config.gconfig.n_samples,
            max_items=max_items,
        )
        if submitted == 0:
            logger.warning("No evaluation items were submitted.")
            return {}

        logger.info(
            "Submitted %d evaluation items to %s with group_size=%d.",
            submitted,
            workflow,
            config.gconfig.n_samples,
        )
        controller.wait(submitted, timeout=None)
        eval_stats = controller.export_stats()
        if eval_stats:
            logger.info("Evaluation Results: %s", tabulate_stats(eval_stats))
        else:
            logger.info("Evaluation finished without exported stats.")
        return eval_stats
    finally:
        controller.destroy()
