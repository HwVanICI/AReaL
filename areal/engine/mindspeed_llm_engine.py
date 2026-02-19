from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import DeviceRuntimeInfo, SaveLoadMeta, WeightUpdateMeta
from areal.engine.adapters.mindspeed_llm_adapter import apply_mindspeed_llm_patches


class MindSpeedLLMEngine(TrainEngine):
    """Thin EngineAPI wrapper delegating to MindSpeedLLMRuntime.

    This module intentionally avoids importing Megatron directly.
    """

    _runtime_class_name = "MindSpeedLLMRuntime"

    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self._runtime = None
        self._parallel_strategy: ParallelStrategy | None = None

    def _build_runtime_argv(self) -> list[str]:
        strategy = self._parallel_strategy or ParallelStrategy()
        return apply_mindspeed_llm_patches(
            backend_cfg=self.config.mindspeed_llm,
            parallel_strategy=strategy,
        )

    def _init_runtime_with_patched_argv(self) -> None:
        if self._runtime is not None:
            return
        old_argv = sys.argv
        try:
            sys.argv = self._build_runtime_argv()
            runtime_mod = importlib.import_module("areal.engine.bootstrap.mindspeed_llm_runtime")
            runtime_cls = getattr(runtime_mod, self._runtime_class_name)
            self._runtime = runtime_cls(self.config)
        finally:
            sys.argv = old_argv

    @property
    def _rt(self):
        if self._runtime is None:
            raise RuntimeError("MindSpeedLLMRuntime is not initialized.")
        return self._runtime

    def __getattr__(self, name: str):
        return getattr(self._rt, name)

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        self._parallel_strategy = parallel_strategy
        self._init_runtime_with_patched_argv()
        return self._rt.create_process_group(parallel_strategy)

    def initialize(self, *args, **kwargs):
        self._init_runtime_with_patched_argv()
        return self._rt.initialize(*args, **kwargs)

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self._rt.data_parallel_group

    @property
    def data_parallel_rank(self) -> int:
        return self._rt.data_parallel_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self._rt.data_parallel_world_size

    def current_data_parallel_head(self) -> int:
        return self._rt.current_data_parallel_head()

    def is_data_parallel_head(self) -> bool:
        return self._rt.is_data_parallel_head()

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self._rt.context_and_model_parallel_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        return self._rt.cpu_group

    def destroy(self):
        if self._runtime is None:
            return None
        return self._rt.destroy()

    @property
    def initialized(self) -> bool:
        if self._runtime is None:
            return False
        return self._rt.initialized

    def train(self, mode: bool = True):
        return self._rt.train(mode)

    def update_weights(self, meta: WeightUpdateMeta):
        return self._rt.update_weights(meta)

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        return self._rt.connect_engine(engine, meta)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        return self._rt.rollout_batch(data, workflow, workflow_kwargs, group_size)

    def prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        return self._rt.prepare_batch(
            dataloader,
            workflow,
            workflow_kwargs,
            should_accept_fn,
            group_size,
            dynamic_bs,
        )

    def set_version(self, version: int):
        return self._rt.set_version(version)

    def get_version(self) -> int:
        return self._rt.get_version()

    def save(self, meta: SaveLoadMeta):
        return self._rt.save(meta)

    def load(self, meta: SaveLoadMeta):
        return self._rt.load(meta)

    def optimizer_zero_grad(self):
        return self._rt.optimizer_zero_grad()

    def optimizer_step(self):
        return self._rt.optimizer_step()

    def lr_scheduler_step(self):
        return self._rt.lr_scheduler_step()

    def forward_backward_batch(self, mb_list, process_output_fn, forward_only: bool = False):
        return self._rt.forward_backward_batch(mb_list, process_output_fn, forward_only)

    def train_batch(self, input_: dict[str, Any], loss_fn, loss_weight_fn):
        return self._rt.train_batch(input_, loss_fn, loss_weight_fn)

    @torch.no_grad()
    def eval_batch(self, input_: dict[str, Any], loss_fn, loss_weight_fn):
        return self._rt.eval_batch(input_, loss_fn, loss_weight_fn)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ):
        return self._rt.forward_batch(input_, output_seqlens, aggregate_fn)

    def export_stats(self) -> dict[str, float]:
        return self._rt.export_stats()

    def onload(self) -> None:
        return self._rt.onload()

    def offload(self) -> None:
        return self._rt.offload()

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return self._rt.get_device_stats()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        return self._rt.save_perf_tracer(step=step, force=force)

    def config_perf_tracer(self, config: PerfTracerConfig, rank: int, role: str) -> None:
        return self._rt.config_perf_tracer(config=config, rank=rank, role=role)


class MindSpeedLLMPPOActor(MindSpeedLLMEngine):
    _runtime_class_name = "MindSpeedLLMPPOActorRuntime"

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs):
        return self._rt.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs):
        return self._rt.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        return self._rt.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.engine.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class MindSpeedLLMPPOCritic(MindSpeedLLMEngine):
    _runtime_class_name = "MindSpeedLLMPPOCriticRuntime"

    @torch.no_grad()
    def compute_values(self, *args, **kwargs):
        return self._rt.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        return self._rt.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.engine.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class MindSpeedLLMLMEngine(MindSpeedLLMEngine):
    _runtime_class_name = "MindSpeedLLMLMRuntime"

    def train_lm(self, data):
        return self._rt.train_lm(data)

    def evaluate_lm(self, data):
        return self._rt.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.engine.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)
