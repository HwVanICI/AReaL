from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.engine.adapters.mindspeed_llm_adapter import (
    apply_mindspeed_llm_patches,
    namespace_to_dict,
)
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler
    from areal.engine.ppo.actor import PPOActorConfig
    from areal.engine.ppo.critic import PPOCriticConfig


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
            megatron_cfg=self.mcore_config,
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
                # A compact snapshot helps diff against script-style args.
                effective = namespace_to_dict(self.mindspeed_llm_args)
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
        elif self.mindspeed_llm_config.modeling_mode != "mbridge":
            raise ValueError(
                f"Invalid mindspeed_llm.modeling_mode: {self.mindspeed_llm_config.modeling_mode}"
            )
        return super().initialize(addr, ft_spec, *args, **kwargs)


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

