from __future__ import annotations

import dataclasses
import functools
import gc
import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mindspeed_llm import megatron_adaptor  # noqa: F401
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_model_config
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, PretrainedConfig

from areal.api.alloc_mode import (
    MegatronParallelStrategy,
    ParallelStrategy,
)
from areal.api.cli_args import MicroBatchSpec, PerfTracerConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    FinetuneSpec,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.workflow_api import WorkflowLike
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.bootstrap.mindspeed_llm_bootstrap import (
    create_gpt_model_from_mindspeed_args,
)
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names, perf_tracer, stats_tracker
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unpad_logits,
)
from areal.utils.distributed import init_custom_process_group
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.mcore.determinisitc import set_deterministic_algorithms
from areal.utils.mcore.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.utils.mcore.pipeline_parallel import configure_pipeline_layer_splits
from areal.utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.utils.megatron_checkpointer import MegatronCheckpointManager
from areal.utils.model import disable_dropout_in_model
from areal.utils.network import find_free_ports, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.seeding import get_seed

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler
    from areal.engine.ppo.actor import PPOActorConfig
    from areal.engine.ppo.critic import PPOCriticConfig


class _MegatronModelList(list):
    """List wrapper that exposes module-like helpers for Megatron model chunks."""

    def forward(self, *args, **kwargs) -> Any:
        if len(self) == 1:
            return self[0](*args, **kwargs)
        raise RuntimeError(
            "Direct forward calls are only supported for single-chunk model list."
        )

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        for module in self:
            yield from module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_parameters(*args, **kwargs):
            yield parameter


class _ValueHead(torch.nn.Linear):
    def __init__(self, input_size: int, *, sequence_parallel: bool, dtype: torch.dtype):
        super().__init__(in_features=input_size, out_features=1, bias=False)
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
        self.weight.data.normal_(mean=0.0, std=0.02)
        self.to(dtype=dtype)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        del weight, runtime_gather_output
        logits = super().forward(input_).float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


class MindSpeedLLMRuntime(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config: PretrainedConfig
        self.tf_config: TransformerConfig
        self.model: _MegatronModelList | None = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.mindspeed_config = config.mindspeed
        self.parallel_strategy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.process_group_initialized = False
        self._initialized = False
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None
        self.weight_update_group_initialized: bool = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int
        self._version: int = 0
        self.rank: int | None = None
        self.is_pp_head: bool
        self.world_size: int | None = None
        self.rank_generator: mpu.RankGenerator | None = None
        self.checkpointer: MegatronCheckpointManager | None = None
        self.lr_scheduler: OptimizerParamScheduler | None = None
        self.seed: int = 0
        self.own_global_group: bool = False
        self.is_offload: bool = False
        self.mindspeed_llm_config = config.mindspeed_llm
        if self.mindspeed_llm_config.modeling_mode != "spec":
            raise ValueError(
                "MindSpeed-LLM backend is spec-only now; "
                f"got modeling_mode={self.mindspeed_llm_config.modeling_mode!r}."
            )
        self._mindspeed_llm_argv = list(sys.argv)
        self.mindspeed_llm_args = None
        self.logger = logging.getLogger("MindSpeedLLMRuntime")

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        # Spec mode uses initialize_megatron() as single entry; groups are initialized there.
        self.process_group_initialized = False

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        if self.mindspeed_llm_config.stage != "sft":
            raise NotImplementedError(
                "MindSpeed-LLM backend currently supports stage='sft' only."
            )
        self.logger.info("MindSpeed-LLM modeling_mode=spec is enabled.")
        try:
            self.seed = get_seed()
        except ValueError:
            self.seed = 42
        assert addr is None
        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.alloc_mode = kwargs.get("alloc_mode", None)

        old_argv = sys.argv
        try:
            sys.argv = self._mindspeed_llm_argv
            initialize_megatron(
                extra_args_provider=None,
                args_defaults={},
                ignore_unknown_args=False,
                allow_no_cuda=True,
            )
        finally:
            sys.argv = old_argv

        args = get_args()
        self.mindspeed_llm_args = args
        self.is_pp_head = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        self.weight_update_group_name = f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
        self.engine_lock = DistributedLock("train_engine_lock")
        self.logger = logging.getLogger(f"[MindSpeedLLMRuntime Rank {torch.distributed.get_rank()}]")
        self._init_context_and_model_parallel_group()
        self._cpu_group = torch.distributed.new_group(backend="gloo")
        self.process_group_initialized = True
        self.tokenizer = load_hf_tokenizer(self.config.path)
        self.hf_config = AutoConfig.from_pretrained(self.config.path, trust_remote_code=True)
        self._rebind_preimported_moe_symbols()
        self._log_moe_binding_diagnostics()
        self._log_npu_gmm_dispatch_diagnostics()

        self.tf_config = core_transformer_config_from_args(args)
        self._validate_grouped_gemm_patch()
        self.tf_config = configure_pipeline_layer_splits(self.parallel_strategy, self.hf_config, self.tf_config)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        with self.device:
            model = create_gpt_model_from_mindspeed_args(
                pre_process=pre_process,
                post_process=post_process,
            )
            if self.config.is_critic:
                model.output_layer = _ValueHead(
                    input_size=self.tf_config.hidden_size,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    dtype=self.tf_config.params_dtype,
                )
                model.vocab_size = 1
            if self.mcore_config.wrap_with_ddp:
                ddp_config = MCoreDDPConfig(**dataclasses.asdict(self.mcore_config.ddp))
                model = DDP(
                    config=self.tf_config,
                    ddp_config=ddp_config,
                    module=model,
                    disable_bucketing=False,
                )
        self.model = _MegatronModelList([model])

        with self.device:
            self._load_model_from_hf_via_hf2mg(self.config.path)

        primary_model = self.model[0]
        model_config = get_model_config(primary_model)
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)
        model_config.finalize_model_grads_func = finalize_model_grads
        self._create_optimizer(ft_spec)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def data_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_rank()

    @property
    def data_parallel_world_size(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_world_size()

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return mpu.get_data_parallel_group()

    def current_data_parallel_head(self) -> int:
        """Get the rank of the head of the current data parallel group."""
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0]

    def is_data_parallel_head(self) -> bool:
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0] == self.rank

    @property
    def pipeline_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_pipeline_model_parallel_rank()

    def is_pipeline_parallel_head(self) -> bool:
        assert self.process_group_initialized
        return self.is_pp_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._context_and_model_parallel_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        self.process_group_initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            self.model = None
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()
            self.own_global_group = False

    def train(self, mode: bool = True):
        assert self.model is not None
        for model in self.model:
            model.train(mode=mode)
        return self

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self.weight_update_group_initialized:
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        if meta.type == "xccl":
            assert self.weight_update_group_initialized
            # In offload mode, wakes up parameters as needed to perform the update.
            tms_context = (
                torch_memory_saver.disable()
                if self.is_offload and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state saving, please use DCP format instead."
                )
            self._save_model_to_hf(
                meta.path,
                tokenizer=meta.tokenizer,
                processor=meta.processor,
                base_model_path=meta.base_model_path,
            )
        elif meta.weight_format == "dcp":
            self.checkpointer.save_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state loading, please use DCP format instead."
                )
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self.checkpointer.load_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def optimizer_zero_grad(self):
        assert self.optimizer is not None, "Optimizer is not initialized."
        self.optimizer.zero_grad()
        for model in self.model:
            model.zero_grad_buffer()

    def optimizer_step(self):
        with trace_scope("megatron_engine.step"):
            update_successful, grad_norm, _ = self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        self._ensure_ready()
        def forward_step(batch_iter, model):
            mb_input = next(batch_iter)
            cu_seqlens = mb_input.padded_mb.get("cu_seqlens", None)
            self._set_mindspeed_runtime_context(mb_input.padded_mb)
            output = packed_context_parallel_forward(model, mb_input.padded_mb)

            def _process_output(input_, output_):
                loss = process_output_fn(output_, input_)
                if loss is None:
                    loss = torch.tensor(1.0, device=output_.device)
                return loss, {}

            if mpu.is_pipeline_last_stage(ignore_virtual=False):
                output = unpad_logits(
                    output,
                    padding_length=mb_input.padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=mb_input.old_cu_seqlens,
                )
            return output, functools.partial(_process_output, mb_input.orig_mb)

        forward_backward_func = get_forward_backward_func()
        data_iterator = [iter(mb_list) for _ in range(len(self.model))] if len(self.model) > 1 else iter(mb_list)
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model if len(self.model) > 1 else self.model[0],
            num_microbatches=len(mb_list),
            seq_length=mb_list.max_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Step 3: Forward-backward using Megatron's pipeline function
        loss_multiplier = (
            mpu.get_data_parallel_world_size() * self.optimizer.get_loss_scale().item()
        )

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            return self._compute_logprobs_and_loss(
                output,
                inputs,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=loss_multiplier,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        self._ensure_ready()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Step 3: Forward using Megatron's pipeline function, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            loss = self._compute_logprobs_and_loss(
                output, inputs, loss_fn, loss_weight_fn, total_loss_weight
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate losses
        if mpu.is_pipeline_last_stage():
            return aggregate_eval_losses(losses, mpu.get_data_parallel_group())
        return None

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        self._ensure_ready()

        # Step 1: Prepare sequence lengths
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 3: Forward using Megatron's pipeline function, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(output: torch.Tensor, inputs: dict[str, Any]) -> None:
            result = self._compute_forward_result(output, inputs)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate, reorder, and broadcast outputs
        res = None
        if mpu.is_pipeline_last_stage():
            res = reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)
        res = broadcast_tensor(
            res,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return res

    def export_stats(self) -> dict[str, float]:
        data = stats_tracker.export_all(reduce_group=self.data_parallel_group)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # Some log info only exist in last pipeline rank
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            data.update(data_list[0])
        return data

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        self.get_device_stats().log("before offload model")
        current_platform.clear_memory()
        torch_memory_saver.pause()

        # TODO: NCCL offload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        torch_memory_saver.resume()
        current_platform.clear_memory()

        # TODO: NCCL onload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> MegatronParallelStrategy:
        base_strategy = dataclasses.asdict(parallel_strategy)
        vpp_size = self.mcore_config.virtual_pipeline_parallel_size
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            virtual_pipeline_parallel_size=vpp_size,
            **base_strategy,
        )

    def _init_context_and_model_parallel_group(self) -> None:
        # Initialize context and model parallel groups, which are only used in AReaL
        # for data distribution
        rank_generator = mpu.RankGenerator(
            tp=self.parallel_strategy.tensor_parallel_size,
            ep=1,
            dp=self.parallel_strategy.data_parallel_size,
            pp=self.parallel_strategy.pipeline_parallel_size,
            cp=self.parallel_strategy.context_parallel_size,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")
        # create context and model_parallel_groups
        for dp_rank, ranks in enumerate(context_and_model_parallel_ranks):
            group = mpu.create_group(
                ranks,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self._context_and_model_parallel_group = group

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        if self.optimizer_config is None:
            return
        assert self.model is not None and len(self.model) > 0

        assert self.optimizer_config.type in [
            "adam",
            "sgd",
        ], "Only AdamW/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                "Using the 'sgd' optimizer with Megatron may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability."
            )

        # Make megatron optimizer config
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer=self.optimizer_config.type,
            lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            params_dtype=self.dtype,
            clip_grad=self.optimizer_config.gradient_clipping,
        )
        mcore_opt_config.overlap_param_gather_with_optimizer_step = (
            self.mcore_config.overlap_param_gather_with_optimizer_step
        )
        mcore_opt_config.use_precision_aware_optimizer = (
            self.mcore_config.use_precision_aware_optimizer
        )
        mcore_opt_config.main_grads_dtype = getattr(
            torch, self.mcore_config.main_grads_dtype
        )
        mcore_opt_config.main_params_dtype = getattr(
            torch, self.mcore_config.main_params_dtype
        )
        mcore_opt_config.exp_avg_dtype = getattr(torch, self.mcore_config.exp_avg_dtype)
        mcore_opt_config.exp_avg_sq_dtype = getattr(
            torch, self.mcore_config.exp_avg_sq_dtype
        )

        self.optimizer = get_megatron_optimizer(
            mcore_opt_config,
            self.model,
            no_weight_decay_cond=lambda n, p: any(
                k in n for k in ["bias", "ln.weight", "ln_f.weight"]
            ),
            scale_lr_cond=None,
            lr_mult=1.0,
        )

        warmup_steps_proportion = self.optimizer_config.warmup_steps_proportion
        warmup_steps = int(warmup_steps_proportion * ft_spec.total_train_steps)
        lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
            max_lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            lr_warmup_steps=warmup_steps,
            lr_decay_steps=ft_spec.total_train_steps - warmup_steps,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            start_wd=self.optimizer_config.weight_decay,
            end_wd=self.optimizer_config.weight_decay,
            wd_incr_steps=ft_spec.total_train_steps,
            wd_incr_style="constant",
        )
        self.lr_scheduler = lr_scheduler

        self.checkpointer = MegatronCheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
            async_save=self.mcore_config.async_save,
        )

    def _check_rollout_engine_connected(self) -> None:
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self) -> None:
        if self.is_offload:
            self.onload()

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        # Early exit when chunk size is relatively small
        if not converted_named_tensors:
            return

        self.engine_lock.acquire()

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in converted_named_tensors
        ]

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, param in converted_named_tensors:
            handles.append(
                dist.broadcast(
                    param.data, 0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        converted_named_tensors.clear()

        self.engine_lock.release()

    def _collect_param(
        self,
        name: str,
        param: nn.Parameter | torch.Tensor,
    ) -> tuple[nn.Parameter | torch.Tensor, int]:
        """Collect and prepare a parameter for conversion.

        This method handles:
        - All-gathering the parameter across tensor parallel ranks
        - Removing padding for vocabulary-related parameters
        - Dequantizing parameters when needed
        - Calculating the parameter size in bytes

        Returns:
            Tuple of (prepared_param, param_size_in_bytes)
        """
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.hf_config.vocab_size)
        param_size = param.numel() * param.element_size()

        return param, param_size

    def _impl_update_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param, param_size = self._collect_param(name, param)

        if not self.is_pipeline_parallel_head():
            return buffer_size

        if buffer_size + param_size > weight_chunked_mem_size:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            buffer_size = 0

        converted_named_tensors.extend(
            convert_to_hf(
                self.tf_config,
                self.hf_config.model_type,
                name,
                param,
            )
        )
        buffer_size += param_size
        return buffer_size

    def _update_bucket_expert_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        """Gather a bucket of MoE expert weights and broadcast them.

        This function handles the distributed update for a bucket of Mixture-of-Experts
        (MoE) parameters. Since expert parameters are sharded across the expert
        parallel group, this function first performs an `all_gather` to collect all
        shards from all expert ranks.

        Once the full expert parameters are reconstructed on the pipeline parallel
        head, it converts them to the HuggingFace format and calls
        `_update_bucket_weights_from_distributed` to perform the actual broadcast
        to the inference engine.
        """

        # Early exit when chunk size is relatively small
        if not named_tensors:
            return

        group = mpu.get_expert_model_parallel_group()
        world_size = mpu.get_expert_model_parallel_world_size()

        names = [name for name, _ in named_tensors]
        all_names: list[list[str]] = [None] * world_size
        dist.all_gather_object(all_names, names, group=group)

        for rank_names in all_names:
            if len(named_tensors) != len(rank_names):
                raise RuntimeError(
                    "Named tensor count mismatch across expert parallel ranks: "
                    f"expected {len(rank_names)} but got {len(named_tensors)}"
                )

        gathered_params = [[] for _ in range(world_size)]
        handles = []
        for idx, (_, tensor) in enumerate(named_tensors):
            params = [
                torch.empty_like(tensor.data, device=current_platform.current_device())
                for _ in range(world_size)
            ]
            handle = dist.all_gather(params, tensor.data, group=group, async_op=True)
            handles.append(handle)
            for ep_rank, rank_names in enumerate(all_names):
                gathered_params[ep_rank].append((rank_names[idx], params[ep_rank]))

        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self.is_pipeline_parallel_head():
            return

        gathered_params = sum(gathered_params, [])

        converted_hf_tensors = []
        for name, param in gathered_params:
            converted_hf_tensors.extend(
                convert_to_hf(
                    self.tf_config,
                    self.hf_config.model_type,
                    name,
                    param,
                )
            )

        self._update_bucket_weights_from_distributed(meta, converted_hf_tensors)

    def _impl_update_expert_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param, param_size = self._collect_param(name, param)

        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > weight_chunked_mem_size:
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta) -> None:
        assert meta.type == "xccl"
        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr = gethostip()
        meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]
        meta.nccl_group_name = self.weight_update_group_name

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if self.is_pipeline_parallel_head():
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={self.weight_update_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=self.weight_update_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("megatron_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta) -> None:
        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        converted_named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            buffer_size = self._impl_update_weight_from_distributed(
                meta,
                name,
                param,
                converted_named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        # Only pipeline parallel heads CAN contain named tensors here
        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)

        dist.barrier(group=self.cpu_group)

        buffer_size = 0
        named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." not in name:
                continue
            buffer_size = self._impl_update_expert_weight_from_distributed(
                meta,
                name,
                param,
                named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        if named_tensors:
            # This function will early return if not pipeline parallel head
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("megatron_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta) -> None:
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        self._save_model_to_hf(meta.path, self.tokenizer, None)
        # dist.barrier() are called when _save_model_to_hf finished

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )

            fut.result()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _rebind_preimported_moe_symbols(self) -> None:
        experts_mod = sys.modules.get("megatron.core.transformer.moe.experts")
        moe_specs_mod = sys.modules.get("megatron.core.models.gpt.moe_module_specs")
        if experts_mod is None or moe_specs_mod is None:
            return
        for name in ("GroupedMLP", "SequentialMLP", "TEGroupedMLP"):
            if hasattr(experts_mod, name):
                setattr(moe_specs_mod, name, getattr(experts_mod, name))
        gg_mod = sys.modules.get("megatron.core.transformer.moe.grouped_gemm_util")
        if gg_mod is None:
            return
        from mindspeed.core.fusions.grouped_matmul import Ops as MindSpeedGroupedMatmulOps
        from mindspeed.core.fusions.grouped_matmul import (
            assert_grouped_gemm_is_available as ms_assert_grouped_gemm_is_available,
        )
        from mindspeed.core.fusions.grouped_matmul import (
            grouped_gemm_is_available as ms_grouped_gemm_is_available,
        )

        setattr(gg_mod, "ops", MindSpeedGroupedMatmulOps)
        setattr(gg_mod, "grouped_gemm_is_available", ms_grouped_gemm_is_available)
        setattr(
            gg_mod,
            "assert_grouped_gemm_is_available",
            ms_assert_grouped_gemm_is_available,
        )

    def _log_moe_binding_diagnostics(self) -> None:
        try:
            import megatron.core.transformer.moe.experts as experts_mod
            import megatron.core.transformer.moe.grouped_gemm_util as gg_mod
            from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec

            grouped_mlp = getattr(experts_mod, "GroupedMLP", None)
            spec_grouped_mlp = get_moe_module_spec.__globals__.get("GroupedMLP", None)
            gg_ops = getattr(gg_mod, "ops", None)
            self.logger.info(
                "MindSpeed MoE binding: experts.GroupedMLP=%s.%s ; "
                "moe_module_specs.GroupedMLP=%s.%s ; gg.ops=%s.%s",
                getattr(grouped_mlp, "__module__", None),
                getattr(grouped_mlp, "__name__", None),
                getattr(spec_grouped_mlp, "__module__", None),
                getattr(spec_grouped_mlp, "__name__", None),
                getattr(gg_ops, "__module__", None),
                getattr(gg_ops, "__name__", None),
            )
        except Exception as e:
            self.logger.warning("MindSpeed MoE binding diagnostics failed: %s", e)

    def _log_npu_gmm_dispatch_diagnostics(self) -> None:
        try:
            import mindspeed.ops.gmm as gmm_mod

            npu_gmm = getattr(gmm_mod, "npu_gmm", None)
            self.logger.info(
                "MindSpeed GMM symbol: mindspeed.ops.gmm=%s ; npu_gmm=%s.%s",
                getattr(gmm_mod, "__file__", None),
                getattr(npu_gmm, "__module__", None),
                getattr(npu_gmm, "__name__", None),
            )
        except Exception as e:
            self.logger.warning("MindSpeed GMM python import diagnostics failed: %s", e)

        try:
            has_privateuse1_tensor = torch._C._dispatch_has_kernel_for_dispatch_key(
                "mindspeed::npu_gmm.Tensor", "PrivateUse1"
            )
            has_privateuse1_list = torch._C._dispatch_has_kernel_for_dispatch_key(
                "mindspeed::npu_gmm.List", "PrivateUse1"
            )
            has_autograd_privateuse1_tensor = torch._C._dispatch_has_kernel_for_dispatch_key(
                "mindspeed::npu_gmm.Tensor", "AutogradPrivateUse1"
            )
            has_autograd_privateuse1_list = torch._C._dispatch_has_kernel_for_dispatch_key(
                "mindspeed::npu_gmm.List", "AutogradPrivateUse1"
            )
            self.logger.info(
                "MindSpeed GMM dispatch: npu_gmm.Tensor PrivateUse1=%s AutogradPrivateUse1=%s ; "
                "npu_gmm.List PrivateUse1=%s AutogradPrivateUse1=%s",
                has_privateuse1_tensor,
                has_autograd_privateuse1_tensor,
                has_privateuse1_list,
                has_autograd_privateuse1_list,
            )
        except Exception as e:
            self.logger.warning("MindSpeed GMM dispatch diagnostics failed: %s", e)

    def _validate_grouped_gemm_patch(self) -> None:
        args = get_args()
        if not getattr(args, "moe_grouped_gemm", False):
            return
        if getattr(args, "transformer_impl", None) != "local":
            return
        from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec

        grouped_mlp_cls = get_moe_module_spec.__globals__.get("GroupedMLP", None)
        grouped_mlp_module = getattr(grouped_mlp_cls, "__module__", "")
        if not grouped_mlp_module.startswith("mindspeed."):
            raise RuntimeError(
                "MindSpeed grouped_gemm patch is not effective: "
                f"GroupedMLP resolves to {grouped_mlp_module}.{getattr(grouped_mlp_cls, '__name__', 'Unknown')}."
            )

    def _set_mindspeed_runtime_context(self, mb: dict[str, Any]) -> None:
        from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len
        from mindspeed.utils import set_position_ids
        from mindspeed_llm.training.utils import compute_actual_seq_len

        cu_seqlens = mb.get("cu_seqlens", None)
        if cu_seqlens is not None:
            set_actual_seq_len(cu_seqlens.to(dtype=torch.int64))
        else:
            position_ids = mb.get("position_ids", None)
            if torch.is_tensor(position_ids):
                set_actual_seq_len(compute_actual_seq_len(position_ids))

        position_ids = mb.get("position_ids", None)
        if position_ids is None:
            return
        args = get_args()
        if getattr(args, "reset_position_ids", False) and position_ids.ndim == 2:
            set_position_ids(position_ids.transpose(0, 1).contiguous())
        else:
            set_position_ids(position_ids)

    def _load_model_from_hf_via_hf2mg(self, path: str) -> None:
        from megatron.training.checkpointing import load_checkpoint
        from mindspeed_llm.training.checkpointing import _convert_weights_if_needed
        from mindspeed_llm.training.utils import is_shared_path

        ms_args = self.mindspeed_llm_args
        if ms_args is None:
            raise RuntimeError("MindSpeed-LLM args are not initialized.")
        self._validate_runtime_parallel_alignment(ms_args)

        is_hf_dir = os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))
        if is_hf_dir:
            setattr(ms_args, "enable_hf2mg_convert", True)
            setattr(ms_args, "load", path)
            if not getattr(ms_args, "mg_save_dir", None):
                tp = int(getattr(ms_args, "tensor_model_parallel_size", 1))
                pp = int(getattr(ms_args, "pipeline_model_parallel_size", 1))
                ep = int(getattr(ms_args, "expert_model_parallel_size", 1))
                etp = int(getattr(ms_args, "expert_tensor_parallel_size", 1))
                setattr(ms_args, "mg_save_dir", os.path.join(path, f"areal_hf2mg_tp{tp}pp{pp}ep{ep}etp{etp}"))
            if not self._is_readable_hf2mg_cache(ms_args.mg_save_dir):
                os.makedirs(ms_args.mg_save_dir, exist_ok=True)
                _convert_weights_if_needed(ms_args, is_shared_path(ms_args.mg_save_dir))
            load_path = ms_args.mg_save_dir
        else:
            load_path = path
        setattr(ms_args, "load", load_path)
        load_checkpoint(self.model, None, None, strict=True)

    def _validate_runtime_parallel_alignment(self, ms_args) -> None:
        checks = [
            ("tensor_model_parallel_size", self.parallel_strategy.tensor_parallel_size, "tp"),
            ("pipeline_model_parallel_size", self.parallel_strategy.pipeline_parallel_size, "pp"),
            ("expert_model_parallel_size", self.parallel_strategy.expert_parallel_size, "ep"),
            ("expert_tensor_parallel_size", self.parallel_strategy.expert_tensor_parallel_size, "etp"),
            ("context_parallel_size", self.parallel_strategy.context_parallel_size, "cp"),
        ]
        mismatches = []
        for key, expected, short in checks:
            raw = getattr(ms_args, key, None)
            parsed = int(raw) if raw is not None else None
            if parsed in (None, expected):
                continue
            mismatches.append(f"{short}: allocation_mode={expected}, megatron_args={parsed}")
        if mismatches:
            raise ValueError(
                "MindSpeed-LLM runtime parallel args mismatch with allocation_mode: " + "; ".join(mismatches)
            )

    def _is_readable_hf2mg_cache(self, cache_root: str) -> bool:
        if not os.path.isdir(cache_root):
            return False
        tracker = os.path.join(cache_root, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(tracker):
            return False
        try:
            with open(tracker, encoding="utf-8") as f:
                iteration = int(f.read().strip())
        except (OSError, ValueError):
            return False
        iter_dir = os.path.join(cache_root, f"iter_{iteration:07d}")
        if not os.path.isdir(iter_dir):
            return False
        for name in os.listdir(iter_dir):
            if name.startswith("mp_rank_") and os.path.isfile(os.path.join(iter_dir, name, "model_optim_rng.pt")):
                return True
        return False

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ) -> None:
        del path, tokenizer, processor, base_model_path
        raise RuntimeError(
            "MindSpeed-LLM backend is spec-only and does not support mbridge HF save path."
        )

    def _load_model_from_hf(self, path: str) -> None:
        del path
        raise RuntimeError(
            "MindSpeed-LLM backend is spec-only and does not support mbridge HF load path."
        )

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
        # Amend position ids
        input_ = amend_position_ids(input_)
        # Split the input into micro-batches
        # NOTE: Here we use 2*pp_size in forward to align logprob precision
        # TODO: Performance check
        min_n_mbs = (
            2 * pp_size if pp_size > 1 else 1
        )  # avoid pipeline bubbles in training
        # NOTE: self.config.mb_spec.max_tokens_per_mb determines
        # the expected **total** number of tokens per micro-batch **in the forward pass**.
        # The micro batch list splitted here will be splitted to each
        # context parallel rank, so the total number of tokens per
        # GPU in a forward pass here will be `max_tokens_per_mb / cp_size`.
        mb_spec = MicroBatchSpec.new(
            self.config.mb_spec,
            n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs),
            n_mbs_divisor=pp_size,
        )
        mb_list = split_padded_tensor_dict_into_mb_list(
            input_,
            mb_spec,
            group=mpu.get_data_parallel_group(),
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        # NOTE: Pad micro-batches to:
        # 1. Reduce GPU memory fragmentation, pad actual # tokens per mb to integer multiples
        #  of GPU page size or max_tokens_per_mb
        # 2. Align sequence lengths to integer multiples of `align_to_multiple_of=tp_size*cp_size*2`
        #    to satisfy the requirement of Megatron parallelism.
        align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            seq_align_to=align_to_multiple_of,
        )
        self.logger.info(
            f"#microbatch: {len(mb_list.group_lens)}, microbatch #tokens: {mb_list.group_lens}, "
            f"aligned to: {mb_list.align_to_lengths}, padded to: {mb_list.padded_to_lengths}, "
            f"padding lengths: {mb_list.padding_lengths}."
        )
        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        return mb_list

    def _compute_logprobs_and_loss(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        if not self.config.is_critic:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs, entropy = gather_logprobs_entropy(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            loss = loss_fn(logprobs, entropy, inputs)
        else:
            values = output.squeeze(-1)
            loss = loss_fn(values, inputs)

        loss_scale = loss_weight_fn(inputs) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        if not self.config.is_critic:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            return logprobs
        else:
            values = output.squeeze(-1)
            return values

class MindSpeedLLMPPOActorRuntime(MindSpeedLLMRuntime):
    def __init__(self, config: PPOActorConfig):
        from areal.engine.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


class MindSpeedLLMPPOCriticRuntime(MindSpeedLLMRuntime):
    def __init__(self, config: PPOCriticConfig):
        from areal.engine.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)


class MindSpeedLLMLMRuntime(MindSpeedLLMRuntime):
    def __init__(self, config: TrainEngineConfig):
        from areal.engine.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)
