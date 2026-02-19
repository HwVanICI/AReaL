from __future__ import annotations

from mindspeed_llm import megatron_adaptor  # noqa: F401
import mindspeed.ops.gmm  # noqa: F401

import dataclasses
import functools
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.distributed import finalize_model_grads
from megatron.core.utils import get_model_config
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from transformers import AutoConfig

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.engine.bootstrap.mindspeed_llm_bootstrap import (
    create_gpt_model_from_mindspeed_args,
)
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


class MindSpeedLLMRuntime(MegatronEngine):
    """Megatron engine variant patched by MindSpeed-LLM (mcore-only)."""

    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.mindspeed_llm_config = config.mindspeed_llm
        self._mindspeed_llm_argv: list[str] = list(sys.argv)
        self.mindspeed_llm_args = None
        self.logger = logging.getLogger("MindSpeedLLMRuntime")

    def _patch_mindspeed(self, parallel_strategy: ParallelStrategy):
        del parallel_strategy

    def _rebind_preimported_moe_symbols(self) -> None:
        # If Megatron modules were imported before MindSpeed-LLM patching,
        # moe_module_specs may hold stale class references. Rebind them in-place.
        experts_mod = sys.modules.get("megatron.core.transformer.moe.experts")
        moe_specs_mod = sys.modules.get("megatron.core.models.gpt.moe_module_specs")
        if experts_mod is None or moe_specs_mod is None:
            return

        for name in ("GroupedMLP", "SequentialMLP", "TEGroupedMLP"):
            if hasattr(experts_mod, name):
                setattr(moe_specs_mod, name, getattr(experts_mod, name))

        gg_mod = sys.modules.get("megatron.core.transformer.moe.grouped_gemm_util")
        if gg_mod is not None:
            from mindspeed.core.fusions.grouped_matmul import (
                Ops as MindSpeedGroupedMatmulOps,
            )
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
            has_autograd_privateuse1_tensor = (
                torch._C._dispatch_has_kernel_for_dispatch_key(
                    "mindspeed::npu_gmm.Tensor", "AutogradPrivateUse1"
                )
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
                f"GroupedMLP resolves to {grouped_mlp_module}.{getattr(grouped_mlp_cls, '__name__', 'Unknown')}. "
                "This usually means Megatron modules were imported before MindSpeed-LLM patching."
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
                # Align with MindSpeed-LLM data pipeline semantics.
                set_actual_seq_len(compute_actual_seq_len(position_ids))
            else:
                attention_mask = mb.get("attention_mask", None)
                if (
                    attention_mask is not None
                    and torch.is_tensor(attention_mask)
                    and attention_mask.ndim == 2
                ):
                    lens = attention_mask.sum(dim=1, dtype=torch.int64)
                    actual_seq_len = torch.cumsum(lens, dim=0, dtype=torch.int64)
                    actual_seq_len = torch.nn.functional.pad(
                        actual_seq_len, (1, 0), value=0
                    )
                    set_actual_seq_len(actual_seq_len)

        position_ids = mb.get("position_ids", None)
        if position_ids is None:
            return
        args = get_args()
        if (
            getattr(args, "reset_position_ids", False)
            and torch.is_tensor(position_ids)
            and position_ids.ndim == 2
        ):
            set_position_ids(position_ids.transpose(0, 1).contiguous())
        else:
            set_position_ids(position_ids)

    def forward_backward_batch(
        self,
        mb_list,
        process_output_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor | None],
        forward_only: bool = False,
    ) -> None:
        self._ensure_ready()
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from areal.utils.mcore.packed_context_parallel import packed_context_parallel_forward
        from areal.utils.data import unpad_logits

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
        if len(self.model) > 1:
            data_iterator = [iter(mb_list) for _ in range(len(self.model))]
        else:
            data_iterator = iter(mb_list)

        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model if len(self.model) > 1 else self.model[0],
            num_microbatches=len(mb_list),
            seq_length=mb_list.max_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )

    def initialize(self, addr, ft_spec, *args, **kwargs):
        if self.mindspeed_llm_config.stage != "sft":
            raise NotImplementedError(
                "MindSpeed-LLM backend currently supports stage='sft' only."
            )
        if self.mindspeed_llm_config.modeling_mode == "spec":
            self.logger.info(
                "MindSpeed-LLM modeling_mode=spec is enabled."
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
        old_argv = sys.argv
        try:
            sys.argv = self._mindspeed_llm_argv
            initialize_megatron(
                extra_args_provider=None,
                args_defaults={},
                ignore_unknown_args=False,
                allow_no_cuda=True,
                skip_mpu_initialization=True,
            )
        finally:
            sys.argv = old_argv

        args = get_args()
        self.mindspeed_llm_args = args
        self._rebind_preimported_moe_symbols()
        self._log_moe_binding_diagnostics()
        self._log_npu_gmm_dispatch_diagnostics()

        self.tf_config = core_transformer_config_from_args(args)
        self._validate_grouped_gemm_patch()
        self.tf_config = configure_pipeline_layer_splits(
            self.parallel_strategy, self.hf_config, self.tf_config
        )
        self.quantization_config = getattr(self.hf_config, "quantization_config", None)
        self._check_and_apply_fp8_config()
        self._validate_fp8_consistency()

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
        self._validate_runtime_parallel_alignment(ms_args)

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
                tp = int(getattr(ms_args, "tensor_model_parallel_size", 1))
                pp = int(getattr(ms_args, "pipeline_model_parallel_size", 1))
                ep = int(getattr(ms_args, "expert_model_parallel_size", 1))
                etp = int(getattr(ms_args, "expert_tensor_parallel_size", 1))
                mg_dir = os.path.join(
                    path, f"areal_hf2mg_tp{tp}pp{pp}ep{ep}etp{etp}"
                )
                setattr(ms_args, "mg_save_dir", mg_dir)
            if self._is_readable_hf2mg_cache(ms_args.mg_save_dir):
                self.logger.info(
                    "Found readable HF2MG cache at %s, skipping conversion.",
                    ms_args.mg_save_dir,
                )
            else:
                os.makedirs(ms_args.mg_save_dir, exist_ok=True)
                _convert_weights_if_needed(ms_args, is_shared_path(ms_args.mg_save_dir))
            load_path = ms_args.mg_save_dir
        else:
            load_path = path

        setattr(ms_args, "load", load_path)
        iteration = load_checkpoint(self.model, None, None, strict=True)
        self.logger.info("Loaded MindSpeed-LLM Megatron checkpoint, iteration=%s", iteration)

    def _validate_runtime_parallel_alignment(self, ms_args) -> None:
        checks = [
            (
                "tensor_model_parallel_size",
                self.parallel_strategy.tensor_parallel_size,
                "tp",
            ),
            (
                "pipeline_model_parallel_size",
                self.parallel_strategy.pipeline_parallel_size,
                "pp",
            ),
            (
                "expert_model_parallel_size",
                self.parallel_strategy.expert_parallel_size,
                "ep",
            ),
            (
                "expert_tensor_parallel_size",
                self.parallel_strategy.expert_tensor_parallel_size,
                "etp",
            ),
            ("context_parallel_size", self.parallel_strategy.context_parallel_size, "cp"),
        ]
        mismatches: list[str] = []
        for key, expected, short in checks:
            raw = getattr(ms_args, key, None)
            try:
                parsed = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                parsed = None
            if parsed in (None, expected):
                continue
            mismatches.append(f"{short}: allocation_mode={expected}, megatron_args={parsed}")
        if mismatches:
            raise ValueError(
                "MindSpeed-LLM runtime parallel args mismatch with allocation_mode: "
                + "; ".join(mismatches)
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

        checkpoint_file = None
        for name in os.listdir(iter_dir):
            if not name.startswith("mp_rank_"):
                continue
            candidate = os.path.join(iter_dir, name, "model_optim_rng.pt")
            if os.path.isfile(candidate):
                checkpoint_file = candidate
                break
        if checkpoint_file is None:
            return False

        try:
            state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not isinstance(state, dict):
            return False
        return "args" in state and (
            "model" in state or any(key.startswith("model") for key in state.keys())
        )


class MindSpeedLLMPPOActorRuntime(MindSpeedLLMRuntime):
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


class MindSpeedLLMPPOCriticRuntime(MindSpeedLLMRuntime):
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


class MindSpeedLLMLMRuntime(MindSpeedLLMRuntime):
    """Language model engine for SFT using MindSpeed-LLM backend."""

    def __init__(self, config: TrainEngineConfig):
        from areal.engine.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)
