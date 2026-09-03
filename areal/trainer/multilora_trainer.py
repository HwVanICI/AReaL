# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import asyncio
import time
import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    RolloutWorkflow,
    SaveLoadMeta,
    Scheduler,
    StepInfo,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    TeacherConfig,
    TrainDatasetConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.experimental.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.infra.data_service import DataController
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.rdataset import RDataset
from areal.infra.rpc.rtensor import RTensor
from areal.infra.utils.concurrent import call_maybe_async
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.awex_runtime import prepare_awex_runtime
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import (
        FSDPPPOActor,
        FSDPPPOCritic,
        MegatronPPOActor,
        MegatronPPOCritic,
    )
    from areal.experimental.engine.archon_engine import ArchonPPOActor, ArchonPPOCritic
    from areal.trainer.ppo.actor import PPOActorController
    from areal.trainer.ppo.critic import PPOCriticController

logger = logging.getLogger("RLTrainer")

import socket
import threading
from queue import Queue
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from areal.utils.stats_tracker import get, scope, scalar
from areal.trainer.multi_task.manager import TaskState, MultiTaskManager
from concurrent.futures import ThreadPoolExecutor
import shutil

logger = logging.getLogger("RLTrainer")


class _AddLoraRequest(BaseModel):
    lora_name: str
    config_path: str


# Shared queue (lives in this process).
_lora_queue = Queue()


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex((host, port))
        return result != 0  # 0 means port is in use.


def _start_lora_api_server(host: str = "0.0.0.0", port: int = 8001):
    app = FastAPI()

    @app.post("/add_lora")
    def add_lora(req: _AddLoraRequest):
        _lora_queue.put(
            {
                "lora_name": req.lora_name,
                "config_path": req.config_path,
            }
        )
        return {
            "status": "queued",
            "lora_name": req.lora_name,
        }

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.run()


HOST = "0.0.0.0"
PORT = 8001

if _is_port_free("127.0.0.1", PORT):
    _api_thread = threading.Thread(
        target=_start_lora_api_server,
        kwargs={
            "host": HOST,
            "port": PORT,
        },
        daemon=True,
    )
    _api_thread.start()

    print(
        f"LoRA API server started on "
        f"http://{HOST}:{PORT}/add_lora"
    )
else:
    print(
        f"Port {PORT} is already in use. "
        "LoRA API server not started."
    )


class _EmptyDataLoader:
    """Minimal dataloader for online mode that yields empty dicts.

    Compatible with ``cycle_dataloader()`` and ``len()`` expectations.
    ``steps_per_epoch`` controls how many steps constitute one epoch,
    derived from ``total_train_steps // total_train_epochs`` to ensure
    epoch-frequency-gated components (Saver, RecoverHandler) behave
    correctly.
    """

    def __init__(
        self,
        batch_size: int = 1,
        steps_per_epoch: int = 1,
    ):
        self.batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        while True:
            yield [{} for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: ARG002
        pass


class MultiLoRATrainer:
    def __init__(
        self,
        config: PPOConfig,
        multi_task_manager: MultiTaskManager,
        lora_model_path: str,
        lora_model_base_path: str,
    ):
        self.lora_model_path = lora_model_path
        self.lora_model_base_path = lora_model_base_path
        self.multi_task_manager = multi_task_manager

        rank = int(os.getenv("RANK", "0"))

        if is_single_controller():
            # Set up file logging for controller process.
            logging.setup_file_logging(
                StatsLogger.get_log_path(config.stats_logger)
            )

        self.config = config
        self._awex_runtime = prepare_awex_runtime(config)

        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )

        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()

        self.data_controller: DataController | None = None
        self._train_rdataset: RDataset | None = None
        self._valid_rdataset: RDataset | None = None

        # Set seed.
        seeding.set_random_seed(
            config.seed,
            key=f"trainer{rank}",
        )

        # Parse per-engine allocations from config.
        self.actor_alloc = ModelAllocation.from_str(
            config.actor.backend,
            name="actor",
        )
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend,
            name="rollout",
        )

        self._should_offload_rollout = self._is_actor_rollout_colocated(
            config
        )
        self._should_offload_actor = (
            self._should_offload_rollout or config.actor.offload
        )
        self._should_offload_critic = (
            config.critic is not None and config.critic.offload
        )
        self._should_offload_ref = (
            config.ref is not None and config.ref.offload
        )

        teacher_configs = (
            config.teacher.teachers
            if config.teacher is not None
            else []
        )
        self._should_offload_teacher = any(
            t.offload for t in teacher_configs
        )

        # Validate config before proceeding with weight initialization.
        self._validate_cfg()
        self._amend_xccl_weight_update_envvar()

        agent_cfg = config.rollout.agent
        self._online_mode = (
            agent_cfg is not None and agent_cfg.mode == "online"
        )

        # Create models: actor, critic, ref — each with its own allocation.
        self.actor = self._create_train_engine(
            config.actor,
            self.actor_alloc,
        )

        self.critic = None
        if config.critic is not None:
            critic_alloc = ModelAllocation.from_str(
                config.critic.backend,
                name="critic",
            )
            self.critic = self._create_critic(
                config.critic,
                critic_alloc,
            )

        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            ref_alloc = ModelAllocation.from_str(
                config.ref.backend,
                name="ref",
            )
            self.ref = self._create_train_engine(
                config.ref,
                ref_alloc,
            )

        self.teachers: list[Any] = []
        self.teacher_configs = teacher_configs
        self.teacher_allocs = []
        self.teacher_mixture_weights: list[float] = []
        self.teacher_domain_to_idx: dict[str, int] = {}

        if len(self.teacher_configs) > 0:
            total_weight = sum(
                t.weight for t in self.teacher_configs
            )
            self.teacher_mixture_weights = [
                t.weight / total_weight
                for t in self.teacher_configs
            ]

            self.teacher_domain_to_idx = {
                t_config.domain: idx
                for idx, t_config in enumerate(self.teacher_configs)
                if t_config.domain is not None
            }

            for idx, t_config in enumerate(self.teacher_configs):
                if t_config.engine_type == "rollout":
                    self.teacher_allocs.append(
                        ModelAllocation.from_str(
                            t_config.rollout.backend,
                            name=f"teacher-{idx}",
                        )
                    )
                else:
                    assert t_config.train is not None

                    self.teacher_allocs.append(
                        ModelAllocation.from_str(
                            t_config.train.backend,
                            name=f"teacher-{idx}",
                        )
                    )

                    logger.warning(
                        "teacher.engine_type='train' uses legacy "
                        "train-engine teacher path and is deprecated; "
                        "please migrate to engine_type='rollout'."
                    )

        # NOTE: Need to register the dataloaders for self.multi_task_manager.
        self.register_dataloaders()

        # -- FinetuneSpec -----------------------------------------------------
        if self._online_mode:
            assert steps_per_epoch is not None

            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=(
                    steps_per_epoch
                    * config.train_dataset.batch_size
                ),
                train_batch_size=config.train_dataset.batch_size,
            )
        else:
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=(
                    len(
                        next(
                            iter(
                                self.multi_task_manager.tasks.values()
                            )
                        ).train_dataloader
                    )
                    * config.train_dataset.batch_size
                ),
                train_batch_size=config.train_dataset.batch_size,
            )

        # Initialize engines first — the scheduler must know about roles
        # before the data controller can colocate with them.
        engine_init_kwargs = {
            "addr": None,
            "ft_spec": ft_spec,
        }

        self.actor.initialize(
            **engine_init_kwargs,
            role="actor",
        )

        if self.critic is not None:
            self.critic.initialize(
                **engine_init_kwargs,
                role="critic",
            )

        if self.ref is not None:
            self.ref.initialize(
                **engine_init_kwargs,
                role="ref",
            )

        if (
            len(self.teacher_configs) > 0
            and self.teacher_configs[0].engine_type == "train"
        ):
            for idx, teacher_config in enumerate(
                self.teacher_configs
            ):
                assert teacher_config.train is not None

                teacher = self._create_train_engine(
                    teacher_config.train,
                    self.teacher_allocs[idx],
                )

                teacher.initialize(
                    **engine_init_kwargs,
                    role=f"teacher-{idx}",
                )
                self.teachers.append(teacher)

        # Save initial LoRA weights if enabled (for inference server
        # pre-loading).
        initial_lora_path = self._save_initial_lora_weights(self.lora_model_path)

        if self._should_offload_actor:
            self._offload_model(
                self.actor,
                role="actor",
            )

        # Initialize inference with LoRA path.
        self.rollout = self._init_rollout(
            config.rollout,
            is_eval=False,
            lora_path=initial_lora_path,
        )

        self.eval_rollout = None
        if not self._online_mode:
            self.eval_rollout = self._init_rollout(
                config.rollout,
                is_eval=True,
                lora_path=initial_lora_path,
            )

        if (
            len(self.teacher_configs) > 0
            and self.teacher_configs[0].engine_type == "rollout"
        ):
            self.teachers = [
                self._init_teacher_rollout(
                    teacher_config,
                    idx,
                )
                for idx, teacher_config in enumerate(
                    self.teacher_configs
                )
            ]

        # Proxy worker initialization (lazy, for AgentWorkflow support).
        self._proxy_started = False

        # Prepare weight update meta and connect to inference engine.
        if self.config.actor._version == "v2":
            awex_kwargs: dict[str, Any] = {}

            if config.actor.use_lora:
                awex_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )

            self.weight_update_meta = WeightUpdateMeta.from_awex(
                **awex_kwargs
            )

        elif self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": False,
            }

            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                        # Keep enough recent adapter versions for
                        # off-policy rollouts (max_head_offpolicyness)
                        # plus a safety margin; older versions are
                        # unloaded to bound sglang VRAM and avoid the
                        # adapter-accumulation hang.
                        "lora_keep_versions": (
                            config.rollout.max_head_offpolicyness + 10
                        ),
                    }
                )

            if self._is_actor_rollout_colocated(config):
                disk_kwargs.update(
                    {
                        "colocate_mode": True,
                    }
                )

            self.weight_update_meta = WeightUpdateMeta.from_disk(
                **disk_kwargs
            )

        elif self.config.actor.weight_update_mode == "xccl":
            # NCCL/XCCL weight update.
            xccl_kwargs: dict[str, Any] = {
                "gen_allocation": self.rollout_alloc,
            }

            if config.actor.use_lora:
                xccl_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )

            if self.actor_alloc.backend == "megatron":
                self.weight_update_meta = (
                    WeightUpdateMeta.from_megatron_xccl(
                        **xccl_kwargs
                    )
                )
            else:
                self.weight_update_meta = (
                    WeightUpdateMeta.from_fsdp_xccl(
                        **xccl_kwargs
                    )
                )

        elif self.config.actor.weight_update_mode == "awex":
            awex_cfg = getattr(config, "awex", None)

            if awex_cfg is None:
                raise ValueError(
                    "Awex config is required when weight_update_mode "
                    "is 'awex'."
                )

            if not awex_cfg.meta_server_addr:
                raise ValueError(
                    "awex.meta_server_addr must be set when using awex."
                )

            comm_backend = awex_cfg.comm_backend
            ipc_backend = awex_cfg.weights_exchange_ipc_backend
            use_mindspeed = (
                awex_cfg.use_mindspeed
                or awex_cfg.device_backend == "npu"
            )

            if awex_cfg.device_backend == "npu":
                # Default to HCCL for NPU when NCCL is requested.
                if comm_backend == "nccl":
                    comm_backend = "hccl"

                # CUDA IPC is not available on NPU; fall back to CPU.
                if ipc_backend == "cuda":
                    ipc_backend = "cpu"

            self.weight_update_meta = WeightUpdateMeta.from_awex(
                meta_server_addr=awex_cfg.meta_server_addr,
                comm_backend=comm_backend,
                weights_exchange_ipc_backend=ipc_backend,
                weights_comm_nccl_group_size=(
                    awex_cfg.weights_comm_nccl_group_size
                ),
                enable_debug_mode=awex_cfg.enable_debug_mode,
                debug_mode_config=awex_cfg.debug_mode_config,
                disable_weights_exchange_pipeline=(
                    awex_cfg.disable_weights_exchange_pipeline
                ),
                enable_colocate_mode=awex_cfg.enable_colocate_mode,
                weights_validation_steps=(
                    awex_cfg.weights_validation_steps
                ),
                validate_weights_every_n_steps=(
                    awex_cfg.validate_weights_every_n_steps
                ),
                dump_weights_list_for_validation=(
                    awex_cfg.dump_weights_list_for_validation
                ),
                dump_weights_dir_for_validation=(
                    awex_cfg.dump_weights_dir_for_validation
                ),
                nnodes=awex_cfg.nnodes,
                node_rank=awex_cfg.node_rank,
                use_mindspeed=use_mindspeed,
            )

            if awex_cfg.comm_backend == "file":
                disk_meta = WeightUpdateMeta.from_disk(
                    config.experiment_name,
                    config.trial_name,
                    config.cluster.fileroot,
                )
                self.weight_update_meta.path = disk_meta.path

        else:
            raise ValueError(
                f"Invalid weight update mode: "
                f"{self.config.actor.weight_update_mode}"
            )

        self.actor.connect_engine(
            self.rollout,
            self.weight_update_meta,
        )

        # Set up evaluation (skip in online mode).
        self.evaluator = Evaluator(
            config.evaluator,
            ft_spec,
        )

        # Set up save as HF model.
        self.saver = Saver(
            config.saver,
            ft_spec,
        )
        self.recover_handler = RecoverHandler(
            config.recover,
            ft_spec,
        )

        # Set up statistics logging (wandb, tensorboard, etc.).
        self.stats_logger = StatsLogger(
            config,
            ft_spec,
        )

        self.recover_info = None

        # After recovery, sync the staleness manager so its capacity
        # formula stays bounded despite the version jumping from 0 to
        # recovery_version.
        if self.recover_info is not None:
            recovery_version = (
                self.recover_info.last_step_info.global_step + 1
            )

            if is_single_controller():
                sm = self.rollout.staleness_manager
            else:
                sm = (
                    self.rollout.workflow_executor.staleness_manager
                )

            if sm is not None:
                sm.on_version_recovered(recovery_version)

        self._config_perf_tracer()
        self._apply_initial_offload_policy()

        # NOTE: Creating the rollout queue and clearing paths for
        # multi-LoRA.
        self.rollout_result_queue = asyncio.Queue()

        target_dir = os.path.join(
            config.cluster.name_resolve.nfs_record_root,
            "root",
            config.experiment_name,
        )

        if os.path.exists(target_dir):
            for item in os.listdir(target_dir):
                item_path = os.path.join(
                    target_dir,
                    item,
                )

                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

        def start_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        # In trainer init.
        self.loop = asyncio.new_event_loop()

        threading.Thread(
            target=start_event_loop,
            args=(self.loop,),
            daemon=True,
        ).start()

    def _register_dataloader(self, task):
        # ---------------------------------------------------------
        # Shared DataController
        # ---------------------------------------------------------
        if (
            is_single_controller()
            and isinstance(task.train_dataset, RDataset)
        ):
            if self.data_controller is None:
                ds_cfg = DataServiceConfig.from_dataset_config(
                    task.config.train_dataset,
                    seed=task.config.seed,
                )

                assert self.scheduler is not None

                self.data_controller = DataController(
                    ds_cfg,
                    self.scheduler,
                )

                self.data_controller.initialize(
                    role="data",
                    num_dataset_workers=ds_cfg.num_workers,
                )

            train_dataset_id = (
                f"{task.config.experiment_name}_"
                f"{task.config.trial_name}_"
                f"{task.lora_name}_train"
            )

            print(
                f"[DATA] Registering train dataset: "
                f"{train_dataset_id}"
            )

            task.train_dataset.connect(
                self.data_controller,
                dataset_id=train_dataset_id,
                tokenizer_or_processor_path=(
                    task.config.tokenizer_path
                ),
                shuffle=task.config.train_dataset.shuffle,
                drop_last=task.config.train_dataset.drop_last,
            )

            task._train_rdataset = task.train_dataset

        # ---------------------------------------------------------
        # Train dataloader
        # ---------------------------------------------------------
        task.train_dataloader = self._create_dataloader(
            task.train_dataset,
            dataset_config=task.config.train_dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
        )

        task._register_dataloaders(
            task.train_dataloader,
            "train",
        )

        # Initialize iterator sequentially.
        # This avoids concurrent DataController/DataLoader
        # initialization when rollout tasks are submitted from
        # multiple threads.
        task._train_iterator = iter(task.train_dataloader)

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        if (
            task.config.valid_dataset is not None
            and task.valid_dataset is not None
        ):
            if (
                is_single_controller()
                and isinstance(task.valid_dataset, RDataset)
            ):
                assert self.data_controller is not None

                valid_dataset_id = (
                    f"{task.config.experiment_name}_"
                    f"{task.config.trial_name}_"
                    f"{task.lora_name}_valid"
                )

                print(
                    f"[DATA] Registering valid dataset: "
                    f"{valid_dataset_id}"
                )

                task.valid_dataset.connect(
                    self.data_controller,
                    dataset_id=valid_dataset_id,
                    tokenizer_or_processor_path=(
                        task.config.tokenizer_path
                    ),
                    shuffle=task.config.valid_dataset.shuffle,
                    drop_last=task.config.valid_dataset.drop_last,
                )

                task._valid_rdataset = task.valid_dataset

            task.valid_dataloader = self._create_dataloader(
                task.valid_dataset,
                dataset_config=task.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

            task._register_dataloaders(
                task.valid_dataloader,
                "validation",
            )

        else:
            task.valid_dataloader = None

    def register_dataloaders(self):
        for task_id in self.multi_task_manager.tasks:
            task = self.multi_task_manager.get_task(task_id)
            self._register_dataloader(task)

    def _register_init_loras(self):
        from areal.api.io_struct import get_versioned_lora_name

        self.config.vllm.lora_modules = []

        # Then we add the LoRA for each task.
        for task_id in self.multi_task_manager.tasks:
            task = self.multi_task_manager.get_task(task_id)

            self.config.vllm.lora_modules.append(
                f"{task.lora_name}-v0={self._init_lora_path}"
            )

    ### TODO: We still need to implement these to dynamically add and remove LoRAs    
    # # async def add_lora_adapter(self, lora_request) -> tuple[bool, str]:
    # #     """
    # #     Async API to add a LoRA adapter at any time.
        
    # #     Args:
    # #         lora_request: dict with keys:
    # #             - lora_name
    # #             - lora_int_id
    # #             - lora_model_path
    # #             - base_model_name

    # #     Returns:
    # #         Tuple (success: bool, message: str)
    # #     """
    # #     try:
    # #         # from vllm_ascend.worker.worker import NPUWorker
    # #         # await asyncio.run(add_lora(lora_request, NPUWorker))
    # #         await self.rollout.load_lora_adapter(lora_request)
    # #         print(f"!!! line 700 {lora_request=}")
    # #         time.sleep(19999)
            
    # #         # await self.eval_rollout.load_lora_adapter(lora_request)
    # #         return True, "LoRA adapter added successfully."
    # #     except Exception as e:
    # #         return False, f"Failed to add LoRA adapter: {e}"
        
    # # async def remove_lora_adapter(self, lora_request: dict) -> tuple[bool, str]:
    # #     """
    # #     Async API to remove a LoRA adapter at any time.
        
    # #     Args:
    # #         lora_request: dict with keys:
    # #             - lora_name
    # #             - lora_int_id
    # #             - lora_model_path
    # #             - base_model_name

    # #     Returns:
    # #         Tuple (success: bool, message: str)
    # #     """
    # #     try:
    # #         await self.rollout.remove_lora_adapter(lora_request)
    # #         await self.eval_rollout.remove_lora_adapter(lora_request)
    # #         return True, "LoRA adapter removed successfully."
    # #     except Exception as e:
    # #         return False, f"Failed to remove LoRA adapter: {e}"

    def submit_rollout_async(
        self,
        data_batch,
        workflow,
        workflow_kwargs,
        should_accept_fn=None,
        group_size=1,
        lora_name="",
        stamp="",
    ):
        batch_submit_time = time.time()

        async def _submit():
            result = await self.actor.rollout_batch_async(
                data_batch=data_batch,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                multilora_name=lora_name + stamp,
            )

            await self.rollout_result_queue.put(
                (result, lora_name, batch_submit_time)
            )

        asyncio.run_coroutine_threadsafe(_submit(), self.loop)


    def wait_for_rollouts(self):
        future = asyncio.run_coroutine_threadsafe(
            self.rollout_result_queue.get(),
            self.loop,
        )

        # Block until an item is available.
        result = future.result()

        return result


    def submit_n_batches(self, lora_int_id, data, n=1):
        current_task = self.multi_task_manager.get_task(lora_int_id)

        task_req = {
            "data": data,
            "workflow": current_task.workflow,
            "workflow_kwargs": current_task.workflow_kwargs,
            "lora_name": current_task.lora_name,
            "group_size": current_task.config.gconfig.n_samples,
            "lora_int_id": lora_int_id,
        }

        rollout_requests = [task_req] * n

        step = current_task.global_step % current_task.steps_per_epoch
        global_step = current_task.global_step

        with stats_tracker.scope(current_task.lora_name):
            for req in rollout_requests:
                with (
                    stats_tracker.record_timing("rollout"),
                    perf_tracer.trace_scope(
                        "train.rollout",
                        category=Category.COMPUTE,
                        args={
                            "global_step": global_step,
                            "epoch_step": step,
                        },
                    ),
                ):
                    self.submit_rollout_async(
                        data_batch=req["data"],
                        workflow=req["workflow"],
                        workflow_kwargs=req["workflow_kwargs"],
                        should_accept_fn=None,
                        group_size=req["group_size"],
                        lora_name=req["lora_name"],
                    )

        self.multi_task_manager.register_submission(
            lora_int_id=lora_int_id,
            n=n,
        )


    def remove_old_versioned_folder(self, path: str):
        """
        Remove the folder if it exists. Only run on rank 0 to avoid races.
        """
        if dist.get_rank() == 0:
            if os.path.exists(path):
                print(
                    f"[Rank 0] Removing old checkpoint folder: {path}"
                )
                shutil.rmtree(
                    path,
                    ignore_errors=True,
                )

        # Ensure all ranks wait until deletion is complete.
        dist.barrier()

        
    @staticmethod
    def _is_colocation(strategy: SchedulingStrategy | None) -> bool:
        if strategy is None:
            return False
        return strategy.type in (
            SchedulingStrategyType.colocation,
            SchedulingStrategyType.colocation.value,
            "colocation",
        )

    def _is_actor_rollout_colocated(self, config: PPOConfig) -> bool:
        actor_s = config.actor.scheduling_strategy
        rollout_s = config.rollout.scheduling_strategy
        return (self._is_colocation(actor_s) and actor_s.target == "rollout") or (
            self._is_colocation(rollout_s) and rollout_s.target == "actor"
        )

    def _onload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_onload"),
            perf_tracer.trace_scope(
                f"train.{role}_onload",
                category=Category.IO,
            ),
        ):
            engine.onload()

    def _offload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_offload"),
            perf_tracer.trace_scope(
                f"train.{role}_offload",
                category=Category.IO,
            ),
        ):
            engine.offload()

    def _onload_teachers(self) -> None:
        for idx, (teacher, teacher_config) in enumerate(
            zip(self.teachers, self.teacher_configs, strict=True)
        ):
            if teacher_config.offload:
                self._onload_model(teacher, role=f"teacher-{idx}")

    def _offload_teachers(self) -> None:
        for idx, (teacher, teacher_config) in enumerate(
            zip(self.teachers, self.teacher_configs, strict=True)
        ):
            if teacher_config.offload:
                self._offload_model(teacher, role=f"teacher-{idx}")

    def _set_distillation_fields(self, traj: dict[str, Any]) -> None:
        assert self.config.teacher is not None
        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
        traj["distill_loss_weight"] = self.config.teacher.distill_loss_weight
        traj["distill_loss_type"] = self.config.teacher.distill_loss_type
        traj["mopd_adv_clip"] = self.config.teacher.mopd_adv_clip

    def _trajectory_for_teacher(self, traj: dict[str, Any]) -> dict[str, Any]:
        assert self.config.teacher is not None
        domain_key = self.config.teacher.domain_key
        if domain_key not in traj:
            return traj
        return {k: v for k, v in traj.items() if k != domain_key}

    def _assign_mixed_teacher_logps(self, rollout_batch: list[dict[str, Any]]) -> None:
        teacher_batch = [self._trajectory_for_teacher(traj) for traj in rollout_batch]
        with ThreadPoolExecutor(max_workers=len(self.teachers)) as executor:
            futures = [
                executor.submit(teacher.compute_logp, teacher_batch)
                for teacher in self.teachers
            ]
            per_teacher_logps = [future.result() for future in futures]
        per_teacher_logps = RTensor.localize(per_teacher_logps)
        log_weights = torch.log(
            torch.tensor(
                self.teacher_mixture_weights,
                dtype=torch.float32,
                device=per_teacher_logps[0][0].device,
            )
        )
        for traj_idx, traj in enumerate(rollout_batch):
            stacked = torch.stack(
                [teacher_logps[traj_idx] for teacher_logps in per_teacher_logps],
                dim=0,
            )
            traj["teacher_logp"] = torch.logsumexp(
                stacked + log_weights[:, None, None], dim=0
            )
            self._set_distillation_fields(traj)

    def _assign_domain_teacher_logps(self, rollout_batch: list[dict[str, Any]]) -> None:
        assert self.config.teacher is not None
        domain_key = self.config.teacher.domain_key
        teacher_to_traj_indices: dict[int, list[int]] = {}
        for traj_idx, traj in enumerate(rollout_batch):
            domain = traj.get(domain_key)
            if domain is None:
                raise ValueError(
                    f"Trajectory is missing domain key {domain_key!r}; use a "
                    "domain router workflow or add domain metadata to trajectories."
                )
            if domain not in self.teacher_domain_to_idx:
                raise ValueError(
                    f"No teacher configured for domain {domain!r}. Available "
                    f"teacher domains: {sorted(self.teacher_domain_to_idx)}"
                )
            teacher_idx = self.teacher_domain_to_idx[domain]
            teacher_to_traj_indices.setdefault(teacher_idx, []).append(traj_idx)

        with ThreadPoolExecutor(max_workers=len(teacher_to_traj_indices)) as executor:
            futures = {}
            for teacher_idx, traj_indices in teacher_to_traj_indices.items():
                teacher_batch = [
                    self._trajectory_for_teacher(rollout_batch[i]) for i in traj_indices
                ]
                futures[teacher_idx] = executor.submit(
                    self.teachers[teacher_idx].compute_logp,
                    teacher_batch,
                )

            for teacher_idx, future in futures.items():
                teacher_logps = RTensor.localize(future.result())
                traj_indices = teacher_to_traj_indices[teacher_idx]
                for traj_idx, teacher_logp in zip(
                    traj_indices,
                    teacher_logps,
                    strict=True,
                ):
                    traj = rollout_batch[traj_idx]
                    traj["teacher_logp"] = teacher_logp
                    self._set_distillation_fields(traj)

    def _assign_teacher_logps(self, rollout_batch: list[dict[str, Any]]) -> None:
        assert self.config.teacher is not None
        if self.config.teacher.routing_mode == "domain":
            self._assign_domain_teacher_logps(rollout_batch)
        else:
            self._assign_mixed_teacher_logps(rollout_batch)

    def _offload_rollout(self, is_eval: bool = False):
        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        with (
            stats_tracker.record_timing("rollout_pause"),
            perf_tracer.trace_scope(
                "train.rollout_pause",
                category=Category.INSTR,
            ),
        ):
            rollout.pause()

        with (
            stats_tracker.record_timing("rollout_pause_generation"),
            perf_tracer.trace_scope(
                "train.rollout_pause_generation",
                category=Category.INSTR,
            ),
        ):
            call_maybe_async(rollout.pause_generation)

        with (
            stats_tracker.record_timing("rollout_offload"),
            perf_tracer.trace_scope(
                "train.rollout_offload",
                category=Category.IO,
            ),
        ):
            rollout.offload()

    def _onload_rollout(self, is_eval: bool = False) -> None:
        cleanup_error: Exception | None = None

        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        try:
            with (
                stats_tracker.record_timing("rollout_onload"),
                perf_tracer.trace_scope(
                    "train.rollout_onload",
                    category=Category.IO,
                ),
            ):
                rollout.onload()
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_continue_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_continue_generation",
                    category=Category.INSTR,
                ),
            ):
                call_maybe_async(rollout.continue_generation)
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_resume"),
                perf_tracer.trace_scope(
                    "train.rollout_resume",
                    category=Category.INSTR,
                ),
            ):
                rollout.resume()
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        if cleanup_error is not None:
            raise cleanup_error

    def _apply_initial_offload_policy(self) -> None:
        if self._should_offload_ref:
            self._offload_model(self.ref, role="ref")
        if self._should_offload_critic:
            self._offload_model(self.critic, role="critic")
        if self._should_offload_teacher:
            self._offload_teachers()
        if self._should_offload_actor:
            self._offload_model(self.actor, role="actor")

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")

        # Initialize tasks and set weight_update_meta
        self.weight_update_meta_mapping = dict()
        for task_id in self.multi_task_manager.tasks:
            task = self.multi_task_manager.get_task(task_id)
            task._initialize_step_config()
            
            # changing lora_name of weight_update_meta
            self.weight_update_meta_mapping[task.lora_name] = deepcopy(self.weight_update_meta)
            self.weight_update_meta_mapping[task.lora_name].lora_name = task.lora_name

        while True:                                
            if not self.multi_task_manager.has_running_tasks():
                print("No pending tasks.. waiting 5 seconds")
                time.sleep(5)
                continue

            # Fetch sequentially.
            rollout_inputs = []

            for task_id in self.multi_task_manager.tasks:
                current_task = self.multi_task_manager.get_task(task_id)
                data = current_task.next_train_batch()
                rollout_inputs.append((task_id, data))

            # Submit concurrently.
            with ThreadPoolExecutor(
                max_workers=len(rollout_inputs)
            ) as executor:

                futures = [
                    executor.submit(
                        self.submit_n_batches,
                        lora_int_id=task_id,
                        data=data,
                        n=1,
                    )
                    for task_id, data in rollout_inputs
                ]

                for future in futures:
                    future.result()

            # STEP 2: Collect exactly 1 rollout per LoRA
            step_rollouts = {}
            reward_dict = {}
            
            for _ in range(len(self.multi_task_manager.tasks)):
                rollout_batch, lora_name, batch_submit_time = self.wait_for_rollouts()
                            
                step_rollouts[lora_name] = (
                    rollout_batch,
                    batch_submit_time,
                )
                
                # -----------------------
                # Record rollout reward
                # -----------------------
                mean_reward = None
                    
                if rollout_batch:
                    rewards = [
                        traj["rewards"]
                        for traj in rollout_batch
                        if traj is not None and traj.get("rewards") is not None
                    ]

                    if rewards:
                        reward_tensors = []

                        for i, r in enumerate(rewards):

                            if isinstance(r, RTensor):
                                r = r.to_local()

                            elif not isinstance(r, torch.Tensor):
                                r = torch.as_tensor(r)

                            reward_tensors.append(r.detach().float())

                        reward_tensor = torch.cat(
                            [r.reshape(-1) for r in reward_tensors]
                        )

                        mean_reward = reward_tensor.mean().item()

                        with scope(lora_name):
                            scalar(rollout_reward=mean_reward)

                reward_dict[lora_name] = mean_reward          
            
                print(
                    f"[SYNC] Received rollout for {lora_name} "
                    f"with rollout reward: {mean_reward} "
                    f"{time.time() - batch_submit_time:.2f}s after submission",
                    flush=True,
                )
            
            # STEP 3: Train each LoRA
            for task_id in self.multi_task_manager.tasks:
                task_state = self.multi_task_manager.get_task(task_id)
                lora_name = task_state.lora_name

                rollout_batch, batch_submit_time = step_rollouts[lora_name]
                
                global_step = task_state.global_step
                epoch = global_step // task_state.steps_per_epoch
                step = global_step % task_state.steps_per_epoch

                # -----------------------
                # Training steps
                # -----------------------
                self.current_running_task = task_state
                
                ## double check that lora name is correct
                assert self.weight_update_meta_mapping[lora_name].lora_name == lora_name
                
                current_version = global_step
                versioned_meta = self.weight_update_meta_mapping[lora_name].with_version(current_version)

                # LoRA version handling
                if global_step > 0:
                    print(f"!!! line 1349 {versioned_meta=}")
                    self.actor.update_lora(versioned_meta, global_step, lora_name)
                    
                new_version = global_step + 1
                versioned_meta = self.weight_update_meta_mapping[lora_name].with_version(new_version)

                if config.actor.should_compute_prox_logp():
                    with (
                        stats_tracker.record_timing("recompute_logp"),
                        perf_tracer.trace_scope(
                            "train.recompute_logp",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        prox_logps = self.actor.compute_logp(rollout_batch)
                        for traj, logp in zip(rollout_batch, prox_logps):
                            traj["prox_logp"] = logp
                        self.actor.get_device_stats().log("recompute logp")

                if config.actor.should_compute_prox_logp():
                    with (
                        stats_tracker.record_timing("recompute_logp"),
                        perf_tracer.trace_scope(
                            "train.recompute_logp",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        prox_logps = self.actor.compute_logp(rollout_batch)
                        for traj, logp in zip(rollout_batch, prox_logps):
                            traj["prox_logp"] = logp
                        self.actor.get_device_stats().log("recompute logp")

                with (
                    stats_tracker.record_timing("compute_advantage"),
                    perf_tracer.trace_scope(
                        "train.compute_advantage",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    adv_batch = self.actor.compute_advantages(rollout_batch)
                    self.actor.get_device_stats().log("compute advantages")

                # Wait for async checkpoint staging to complete before modifying parameters
                self.saver.maybe_wait_for_staging()

                if (
                    config.memory_profiler is not None
                    and global_step in config.memory_profiler.profile_steps
                ):
                    self.actor.start_memory_profile(config.memory_profiler.max_entries)

                with (
                    stats_tracker.record_timing("train_step"),
                    perf_tracer.trace_scope(
                        "train.ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.actor.ppo_update(adv_batch)
                    self.actor.step_lr_scheduler()
                    self.actor.get_device_stats().log("ppo update")

                if (
                    config.memory_profiler is not None
                    and global_step in config.memory_profiler.profile_steps
                ):
                    log_dir = StatsLogger.get_log_path(config.stats_logger)
                    snapshot_dir = os.path.join(
                        log_dir, "memory_snapshots", f"step_{global_step}"
                    )
                    os.makedirs(snapshot_dir, exist_ok=True)
                    self.actor.stop_memory_profile(snapshot_dir)
                    logger.info(f"Memory snapshots saved to {snapshot_dir}")

                if self.critic is not None:
                    with (
                        stats_tracker.record_timing("critic_train_step"),
                        perf_tracer.trace_scope(
                            "train.critic_ppo_update",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        self.critic.ppo_update(adv_batch)
                        self.critic.step_lr_scheduler()
                        self.critic.get_device_stats().log("ppo critic update")
                    if self._should_offload_critic:
                        self._offload_model(self.critic, role="critic")

                with (
                    stats_tracker.record_timing("save"),
                    perf_tracer.trace_scope(
                        "train.save",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

                with (
                    stats_tracker.record_timing("checkpoint_for_recover"),
                    perf_tracer.trace_scope(
                        "train.checkpoint",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    self._save_recover_checkpoint(
                        epoch=epoch, epoch_step=step, global_step=global_step, task=self.current_running_task
                    )

                self.rollout.pause()

                actor_stats = None

                with (
                    stats_tracker.record_timing("update_weights"),
                    perf_tracer.trace_scope(
                        "train.update_weights",
                        category=Category.COMM,
                        args={"global_step": global_step},
                    ),
                ):
                    # Use versioned path for weight updates
                    
                    self.actor.update_weights(versioned_meta)
                    if versioned_meta.colocate_mode:
                        actor_stats = self.actor.export_stats()
                        self._offload_model(self.actor, role="actor")
                        self._onload_rollout()
                        stage_meta = versioned_meta.with_colocate_stage(1)
                        self.actor.update_weights(stage_meta)

                    self.actor.set_version(new_version)
                    if self.critic is not None:
                        self.critic.set_version(new_version)
                    self.rollout.set_version(new_version)
                    if self.eval_rollout is not None:
                        self.eval_rollout.set_version(new_version)

                with (
                    stats_tracker.record_timing("eval"),
                    perf_tracer.trace_scope(
                        "train.eval",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self._evaluate(
                        eval_workflow=eval_workflow,
                        eval_workflow_kwargs=eval_workflow_kwargs,
                        epoch=epoch,
                        epoch_step=step,
                        global_step=global_step,
                        task=self.current_running_task
                    )

                with (
                    stats_tracker.record_timing("clear_batches"),
                    perf_tracer.trace_scope(
                        "train.clear_batches",
                        category=Category.INSTR,
                        args={"global_step": global_step},
                    ),
                ):
                    # Each role runs in its own Python process with a
                    # process-local ``_fetch_buffer``; one HTTP DELETE to the
                    # storage owner clears ``_storage`` but not per-consumer
                    # caches. Fan out ``clear_batches`` to every role that
                    # localized the batch — see areal-project/AReaL#1209.
                    # SPMD mode never populates ``_fetch_buffer`` (no RTensor
                    # round-trip), so the fan-out is single-controller only.
                    if is_single_controller():
                        self.actor.clear_batches(rollout_batch, adv_batch)
                        if self.critic is not None:
                            self.critic.clear_batches(rollout_batch, adv_batch)
                        if self.ref is not None:
                            self.ref.clear_batches(rollout_batch)
                        if self.data_controller is not None:
                            self.data_controller.clear_batches()
                        # Defensive sweep: drain RTensors created by auxiliary RPC
                        # returns (stats dicts, etc.) that aren't tracked by the
                        # standard batch lifecycle above. See #1209.
                        self.actor.clear_all_local_rtensors()
                        if self.critic is not None:
                            self.critic.clear_all_local_rtensors()
                        if self.ref is not None:
                            self.ref.clear_all_local_rtensors()

                with perf_tracer.trace_scope(
                    "train.log_stats",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ):
                    self._export_and_commit_stats(
                        epoch=epoch,
                        epoch_step=step,
                        global_step=global_step,
                        actor_stats=actor_stats,
                    )
                    
                self.rollout.resume()

                self._save_perf_tracer(step=global_step)
                self.remove_old_versioned_folder(f"{self.weight_update_meta_mapping[lora_name].path}_v{task_state.global_step-1}")

                # Increment step
                task_state.global_step += 1
                

    def close(self):
        self.saver.finalize()
        if hasattr(self, "_train_rdataset") and self._train_rdataset is not None:
            self._train_rdataset.close()
        if hasattr(self, "_valid_rdataset") and self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        self.rollout.destroy()
        for teacher in self.teachers:
            teacher.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)
        self._awex_runtime.close()

    def _config_perf_tracer(self):
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")

        if not is_single_controller():
            return

        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if self.critic is not None:
            self.critic.config_perf_tracer(self.config.perf_tracer, role="critic")
        if self.ref is not None:
            self.ref.config_perf_tracer(self.config.perf_tracer, role="ref")
        self.rollout.config_perf_tracer(self.config.perf_tracer, role="rollout")
        if self.eval_rollout is not None:
            self.eval_rollout.config_perf_tracer(
                self.config.perf_tracer, role="eval-rollout"
            )

    def _save_perf_tracer(self, step: int):
        self.actor.save_perf_tracer(step=step)
        if self.ref is not None:
            self.ref.save_perf_tracer(step=step)
        if self.critic is not None:
            self.critic.save_perf_tracer(step=step)
        if self.eval_rollout is not None:
            self.eval_rollout.save_perf_tracer(step=step)
        self.rollout.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _amend_xccl_weight_update_envvar(self):
        if not is_single_controller():
            # These environs are set by the launcher in the SPMD mode.
            return
        if self.rollout_alloc.backend != "sglang":
            return

        # Disable some environ for NCCL weight update.
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> FSDPPPOActor | MegatronPPOActor | ArchonPPOActor | PPOActorController:
        """Create a training engine (actor or ref) based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOActor

            actor_cls = FSDPPPOActor
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOActor

            actor_cls = MegatronPPOActor
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOActor

            actor_cls = ArchonPPOActor
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def _create_critic(
        self, critic_config: PPOCriticConfig, alloc: ModelAllocation
    ) -> FSDPPPOCritic | MegatronPPOCritic | ArchonPPOCritic | PPOCriticController:
        """Create a critic engine based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOCritic

            critic_cls = FSDPPPOCritic
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOCritic

            critic_cls = MegatronPPOCritic
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOCritic

            critic_cls = ArchonPPOCritic
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            critic = critic_cls.as_controller(critic_config, self.scheduler)
        else:
            critic = critic_cls(config=critic_config)
        critic.create_process_group(parallel_strategy=alloc.parallel)
        return critic

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ) -> InferenceEngine | RolloutController:
        
        self._init_lora_path = lora_path
        if lora_path is not None and not is_single_controller():
            raise ValueError(
                "LoRA is only supported in single-controller mode. "
                "Use `python3 train.py scheduler.type=local` instead of "
                "`python3 -m areal.infra.launcher.local`."
            )
        # Create a working copy of config
        config = deepcopy(rollout_config)
        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            config.max_head_offpolicyness = int(1e12)
            # eval-rollout uses the same inference servers as rollout
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        # Determine engine class and server args based on backend
        rollout_backend = self.rollout_alloc.backend
        if rollout_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            if lora_path is not None and self.config.actor.use_lora:
                self.config.sglang.lora_paths = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_backend == "vllm":
            if self.config.rollout.return_routed_experts:
                raise ValueError(
                    "return_routed_experts is not supported with vLLM backend. Please disable return_routed_experts or switch to SGLang backend."
                )
            if lora_path is not None and self.config.actor.use_lora:
                self._register_init_loras()
            
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
            # vLLM does not require LoRA paths during initialization.
            # LoRA is attached to generation requests.
        else:
            raise ValueError(
                f"Invalid backend: {rollout_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine

        # Single-controller mode - no engine instantiation needed
        if config._version == "v2":
            controller = RolloutControllerV2(
                config=config, scheduler=cast(Scheduler, self.scheduler)
            )
        else:
            controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            server_args=server_args,
        )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "eval-rollout"
            
        controller.initialize(**init_kwargs)
        return controller

    def _init_teacher_rollout(
        self, teacher_config: TeacherConfig, idx: int
    ) -> InferenceEngine | RolloutController:
        assert teacher_config.rollout is not None
        rollout_config = teacher_config.rollout
        rollout_alloc = self.teacher_allocs[idx]
        config = deepcopy(rollout_config)
        if rollout_alloc.backend == "sglang":
            engine_cls = RemoteSGLangEngine
            teacher_sglang_cfg = deepcopy(teacher_config.sglang or self.config.sglang)
            if teacher_config.path:
                teacher_sglang_cfg.model_path = teacher_config.path
            server_args = SGLangConfig.build_args(
                sglang_config=teacher_sglang_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_alloc.backend == "vllm":
            engine_cls = RemotevLLMEngine
            teacher_vllm_cfg = deepcopy(teacher_config.vllm or self.config.vllm)
            if teacher_config.path:
                teacher_vllm_cfg.model = teacher_config.path
                if not rollout_config.tokenizer_path:
                    config.tokenizer_path = teacher_config.path
            server_args = vLLMConfig.build_args(
                vllm_config=teacher_vllm_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid teacher rollout backend: {rollout_alloc.backend}, expected sglang or vllm"
            )
        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine
        controller = engine_cls.as_controller(config, self.scheduler)
        controller.initialize(role=f"teacher-{idx}", server_args=server_args)
        return controller

    def _save_initial_lora_weights(self, path: str) -> str | None:
        """Save initial LoRA weights for inference server pre-loading.

        Returns path to saved LoRA weights, or None if LoRA is disabled.
        """
        if not self.config.actor.use_lora:
            return None

        meta = SaveLoadMeta(
            path=path,
            weight_format="hf",
            with_optim=False,
            tokenizer=self.tokenizer,
            processor=self.processor,
            base_model_path=self.config.actor.path,
        )
        # Save LoRA weights using engine's HuggingFace save
        self.actor.save(meta=meta)

        return path

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        if self.critic is not None:
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
            )
        # Async mode: synchronization handled by AsyncCheckpointManager
        if not self.saver.is_async and not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int, task: TaskState):
        # Save recoverable checkpoints
        to_save: dict = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(task.train_dataloader),
        )

        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            task.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
        valid_dataloader,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate(
        self,
        eval_workflow: WorkflowLike | None,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
        task: TaskState,
    ):
        if (
            self.eval_rollout is None
            or task.valid_dataloader is None
            or eval_workflow is None
        ):
            return
        self.evaluator.evaluate(
            functools.partial(
                self._evaluate_fn,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                valid_dataloader=task.valid_dataloader,
            ),
            epoch,
            epoch_step,
            global_step,
        )
        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _export_and_commit_stats(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
        actor_stats: dict[str, float] | None = None,
    ):
        # Upload statistics to the logger (e.g., wandb)
        stats = actor_stats if actor_stats is not None else self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _validate_cfg(self):
        """validate config for incompatible settings before weight initialization, to avoid wasted resources on spawning workers and loading models."""
        rollout_backend = self.rollout_alloc.backend
        actor_backend = self.actor_alloc.backend
        requires_train_engine_offload = any(
            (
                self._should_offload_rollout,
                self._should_offload_actor,
                self._should_offload_critic,
                self._should_offload_ref,
                self._should_offload_teacher,
            )
        )

        if requires_train_engine_offload and not self.config.enable_offload:
            raise ValueError(
                "enable_offload must be True when colocation scheduling or train-engine "
                "offload is enabled. Please set enable_offload=True."
            )

        if rollout_backend == "vllm" and self.config.rollout.return_routed_experts:
            raise ValueError(
                "return_routed_experts is only supported with SGLang backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )
        if (
            actor_backend == "megatron"
            and self.config.actor.use_lora
            and rollout_backend == "sglang"
        ):
            raise ValueError(
                "Megatron actor with LoRA is not supported with SGLang rollout in "
                "RL trainer. Please use vLLM rollout backend, or disable LoRA, or "
                "switch actor backend from Megatron."
            )

        # Ensure actor and rollout controller versions match.
        actor_version = self.config.actor._version
        rollout_version = self.config.rollout._version
        if actor_version != rollout_version:
            raise ValueError(
                f"actor._version ('{actor_version}') and rollout._version "
                f"('{rollout_version}') must match. Both must be 'v1' or both 'v2'."
            )

    def _requires_proxy_workflow(
        self,
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Check if workflow requires proxy workers (i.e., not a RolloutWorkflow).

        Returns True if:
        - Workflow is NOT a RolloutWorkflow instance
        - Workflow is NOT a RolloutWorkflow class
        - Workflow is a string that does NOT import to a RolloutWorkflow

        This enables any callable object with a compatible signature to work
        without requiring inheritance from AgentWorkflow.
        """
        # None workflow is handled separately in train()
        if workflow is None:
            return False

        resolved_workflow = workflow
        if isinstance(workflow, str):
            from areal.utils.dynamic_import import import_from_string

            try:
                resolved_workflow = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError):
                # If import fails, assume it needs proxy (fail-safe)
                return True

        workflow_specs = getattr(resolved_workflow, "_iter_workflow_specs", None)
        if callable(workflow_specs):
            return any(
                self._requires_proxy_workflow(child, child_kwargs)
                for child, child_kwargs in workflow_specs(workflow_kwargs or {})
            )

        # Direct RolloutWorkflow instances
        if isinstance(workflow, RolloutWorkflow):
            return False

        # RolloutWorkflow classes
        if isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return False

        # String import paths
        if isinstance(workflow, str):
            # Check if imported object is RolloutWorkflow
            if isinstance(resolved_workflow, RolloutWorkflow):
                return False
            if isinstance(resolved_workflow, type) and issubclass(
                resolved_workflow, RolloutWorkflow
            ):
                return False

        # Everything else requires proxy workers
        return True

    def _ensure_proxy_started(self) -> None:
        """Lazily initialize proxy workers when agent workflows are used.

        This method is called before training when a non-RolloutWorkflow is detected
        or when online mode is configured. It creates proxy workers colocated with
        rollout workers to handle OpenAI-compatible API requests.

        In online mode, also starts the proxy gateway for external access.
        """
        if self._proxy_started:
            return

        # Only initialize proxy in single-controller mode with RolloutController
        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if not isinstance(self.rollout, RolloutController):
            self._proxy_started = True
            return

        # v1 controller needs an explicit proxy launch call
        logger.info("Initializing proxy workers for AgentWorkflow support")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()

        # Start proxy gateway for online mode.
        agent_cfg = self.config.rollout.agent
        if agent_cfg is not None and agent_cfg.mode == "online":
            self.rollout.start_proxy_gateway()
            logger.info(
                "Proxy gateway available at %s",
                self.rollout.proxy_gateway_addr,
            )

        self._proxy_started = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False
