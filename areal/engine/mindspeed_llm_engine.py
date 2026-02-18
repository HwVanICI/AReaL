from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.distributed import finalize_model_grads
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import get_model_config
from transformers import AutoConfig

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.engine.adapters.mindspeed_llm_adapter import apply_mindspeed_llm_patches
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.mcore.determinisitc import set_deterministic_algorithms
from areal.utils.mcore.pipeline_parallel import configure_pipeline_layer_splits
from areal.utils.model import disable_dropout_in_model
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.seeding import get_seed

if TYPE_CHECKING:
    from areal.api.io_struct import FinetuneSpec
    from areal.api.scheduler_api import Scheduler
    from areal.engine.ppo.actor import PPOActorConfig
    from areal.engine.ppo.critic import PPOCriticConfig


class _MegatronModelList(list):
    def forward(self, *args, **kwargs) -> Any:
        if len(self) == 1:
            return self[0](*args, **kwargs)
        raise RuntimeError(
            "Direct forward calls are only supported for single-chunk model list."
        )

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for module in self:
            yield from module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Iterator[torch.nn.Parameter]:
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
        logits = super().forward(input_).float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


class MindSpeedLLMEngine(MegatronEngine):
    """Megatron engine variant patched by MindSpeed-LLM (mcore-only)."""

    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.mindspeed_llm_config = config.mindspeed_llm
        self.mindspeed_llm_args = None
        self.logger = logging.getLogger("MindSpeedLLMEngine")

    def _patch_mindspeed(self, parallel_strategy: ParallelStrategy):
        self.mindspeed_llm_args = apply_mindspeed_llm_patches(
            backend_cfg=self.mindspeed_llm_config,
            parallel_strategy=parallel_strategy,
        )

    def initialize(self, addr, ft_spec, *args, **kwargs):
        if self.mindspeed_llm_config.stage != "sft":
            raise NotImplementedError(
                "MindSpeed-LLM backend currently supports stage='sft' only."
            )
        if self.mindspeed_llm_config.modeling_mode == "spec":
            self.logger.info(
                "MindSpeed-LLM modeling_mode=spec is enabled. "
                "Current implementation applies MindSpeed-LLM mcore patches and "
                "reuses AReaL Megatron model construction."
            )
            if self.mindspeed_llm_args is not None:
                effective = vars(self.mindspeed_llm_args)
                self.logger.info(
                    "MindSpeed-LLM effective args snapshot: %s",
                    {
                        k: effective[k]
                        for k in sorted(effective.keys())
                        if k
                        in {
                            "stage",
                            "spec",
                            "use_mcore_models",
                            "tensor_model_parallel_size",
                            "pipeline_model_parallel_size",
                            "expert_model_parallel_size",
                            "context_parallel_size",
                        }
                    },
                )
            return self._initialize_spec_mode(addr=addr, ft_spec=ft_spec, **kwargs)
        elif self.mindspeed_llm_config.modeling_mode != "mbridge":
            raise ValueError(
                f"Invalid mindspeed_llm.modeling_mode: {self.mindspeed_llm_config.modeling_mode}"
            )
        return super().initialize(addr, ft_spec, *args, **kwargs)

    def _initialize_spec_mode(
        self,
        addr: str | None,
        ft_spec: "FinetuneSpec",
        **kwargs,
    ) -> None:
        try:
            self.seed = get_seed()
        except ValueError:
            self.logger.warning("Seed not set, using default seed 42.")
            self.seed = 42

        assert addr is None, "MegatronEngine does not support remote initialization."
        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.is_pp_head = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        self.weight_update_group_name = (
            f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
        )
        self.engine_lock = DistributedLock("train_engine_lock")
        self.alloc_mode = kwargs.get("alloc_mode", None)
        self.tokenizer = load_hf_tokenizer(self.config.path)

        # Build HF config directly and construct mcore model by --spec.
        self.hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        from megatron.training import get_args
        from megatron.training.arguments import core_transformer_config_from_args

        self.tf_config = core_transformer_config_from_args(get_args())
        self.tf_config = configure_pipeline_layer_splits(
            self.parallel_strategy, self.hf_config, self.tf_config
        )
        self.quantization_config = getattr(self.hf_config, "quantization_config", None)
        self._check_and_apply_fp8_config()
        self._validate_fp8_consistency()

        spec = getattr(self.mindspeed_llm_args, "spec", None)
        if not spec:
            raise ValueError(
                "MindSpeed-LLM spec mode requires `--spec ...` in mindspeed_llm.extra_cli_args."
            )

        from megatron.core.transformer.spec_utils import import_module

        transformer_layer_spec = import_module(spec)
        with self.device:
            model = GPTModel(
                config=self.tf_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=True,
                post_process=True,
                share_embeddings_and_output_weights=False,
                position_embedding_type="rope",
                rotary_base=getattr(self.hf_config, "rope_theta", 10000.0),
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
            models = [model]
        self.model = _MegatronModelList(models)

        with self.device:
            self._load_model_from_hf_via_hf2mg(self.config.path)

        for model in self.model:
            for _, param in model.named_parameters():
                if hasattr(param, "get_high_precision_init_val"):
                    param.clear_high_precision_init_val()
                    delattr(param, "get_high_precision_init_val")
                    delattr(param, "clear_high_precision_init_val")

        assert self.model, "Megatron models failed to initialize."
        modules = [m.module if isinstance(m, DDP) else m for m in self.model]
        total_params = sum(
            param.numel() for module in modules for param in module.parameters()
        )
        self.logger.info(
            f"Model parameter count: {total_params / 1e6:.2f}M, pp_stage={mpu.get_pipeline_model_parallel_rank()}, vpp_chunks={len(self.model)}"
        )

        if self.config.disable_dropout:
            for model in self.model:
                disable_dropout_in_model(model)

        primary_model = self.model[0]
        model_config = get_model_config(primary_model)
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)

        for i, model_chunk in enumerate(self.model):
            if (
                isinstance(model_chunk, DDP)
                and self.mcore_config.virtual_pipeline_parallel_size > 1
            ):
                vp_stage = getattr(model_chunk.module, "vp_stage", None)
                self.logger.info(f"Setting vp_stage {vp_stage} for model chunk {i}.")
                setattr(model_chunk, "vp_stage", vp_stage)

        if self.mcore_config.ddp.overlap_grad_reduce and isinstance(primary_model, DDP):
            model_config.no_sync_func = [
                model_chunk.no_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]

        if (
            self.mcore_config.ddp.overlap_param_gather
            and self.mcore_config.ddp.align_param_gather
        ):
            model_config.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.param_sync_func = model_config.param_sync_func[0]
        model_config.finalize_model_grads_func = finalize_model_grads
        self._create_optimizer(ft_spec)
        self._initialized = True

    def _load_model_from_hf_via_hf2mg(self, path: str) -> None:
        assert self.model is not None, "Model is not initialized."
        from megatron.training.checkpointing import load_checkpoint
        from mindspeed_llm.training.checkpointing import _convert_weights_if_needed
        from mindspeed_llm.training.utils import is_shared_path

        ms_args = self.mindspeed_llm_args
        if ms_args is None:
            raise RuntimeError("MindSpeed-LLM args are not initialized.")

        is_hf_dir = False
        if os.path.isdir(path):
            try:
                files = os.listdir(path)
            except OSError:
                files = []
            has_config = "config.json" in files
            has_weight = any(
                name.endswith((".bin", ".safetensors")) and "model" in name.lower()
                for name in files
            )
            is_hf_dir = has_config and has_weight

        if is_hf_dir:
            setattr(ms_args, "enable_hf2mg_convert", True)
            setattr(ms_args, "load", path)
            if not getattr(ms_args, "model_type_hf", None):
                model_type_hf = str(getattr(self.hf_config, "model_type", "qwen3"))
                model_type_hf = model_type_hf.replace("_", "-")
                setattr(ms_args, "model_type_hf", model_type_hf)
            if not getattr(ms_args, "mg_save_dir", None):
                tp = self.parallel_strategy.tensor_parallel_size
                pp = self.parallel_strategy.pipeline_parallel_size
                ep = self.parallel_strategy.expert_parallel_size
                mg_dir = os.path.join(path, f"areal_hf2mg_tp{tp}pp{pp}ep{ep}")
                setattr(ms_args, "mg_save_dir", mg_dir)
            os.makedirs(ms_args.mg_save_dir, exist_ok=True)
            _convert_weights_if_needed(ms_args, is_shared_path(ms_args.mg_save_dir))
            load_path = ms_args.mg_save_dir
        else:
            load_path = path

        setattr(ms_args, "load", load_path)
        iteration = load_checkpoint(self.model, None, None, strict=True)
        self.logger.info("Loaded MindSpeed-LLM Megatron checkpoint, iteration=%s", iteration)


class MindSpeedLLMPPOActor(MindSpeedLLMEngine):
    """PPO Actor implementation using MindSpeed-LLM backend."""

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

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        from areal.engine.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class MindSpeedLLMPPOCritic(MindSpeedLLMEngine):
    """PPO Critic implementation using MindSpeed-LLM backend."""

    def __init__(self, config: PPOCriticConfig):
        from areal.engine.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOCriticConfig, scheduler: Scheduler):
        from areal.engine.ppo.critic import PPOCriticController

        return PPOCriticController(
            train_engine=cls,
            config=config,
            scheduler=scheduler,
        )


class MindSpeedLLMLMEngine(MindSpeedLLMEngine):
    """Language model engine for SFT using MindSpeed-LLM backend."""

    def __init__(self, config: TrainEngineConfig):
        from areal.engine.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        from areal.engine.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)
