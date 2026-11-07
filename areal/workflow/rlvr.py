# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.fork_continuation import ForkContinuationWorkflow
from areal.utils.hf_utils import apply_chat_template
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from areal.workflow.reward_metrics import log_reward_metrics

logger = logging.getLogger("RLVRWorkflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    return apply_chat_template(
        tokenizer,
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        # Support string paths for get_input_ids_fn
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        # Support string paths for data_extract_prompt_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function, and logs metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        # Fork-teardown eviction: the engine returned a partial (stop_reason
        # "abort" only happens on eviction; normal aborts are converted to
        # "length" inside agenerate). Skip reward and surface the partial.
        if resp.stop_reason == "abort" and getattr(engine, "_evicting", False):
            logger.info(
                f"[step0-fork] rid={req.rid}: workflow surfacing fork-evicted "
                f"partial (prompt={resp.input_len} partial={resp.output_len} "
                "tokens) as continuation descriptor"
            )
            return resp, None

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        log_reward_metrics(reward, task_data)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # Fork-continuation re-entry: if the controller injected FORK_CONTINUE_FIELD
        # (re-submitting a partial to a surviving engine), route to the resume
        # path. Isolated in ForkContinuationWorkflow (handles the group_size == 1
        # unwrapped-list case); no-op for normal generation.
        handled, result = await ForkContinuationWorkflow.route_continuation(
            self, engine, data
        )
        if handled:
            return result

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Fork-teardown eviction: surface the partial as a continuation
        # descriptor (resp.stop_reason == "abort" only happens on eviction;
        # normal aborts are converted to "length" inside agenerate). Packaging
        # isolated in ForkContinuationWorkflow.
        if resp.stop_reason == "abort":
            return ForkContinuationWorkflow.build_continuation_descriptor(req, resp)

        return self._build_trajectory(resp, reward)

    async def _arun_episode_continue(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
        cont: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Resume a partial generation on a surviving engine.

        Re-enters the single-turn flow with prompt + partial as the prefix,
        generates the remaining tokens (reusing agenerate's prefix-continuation
        loop), then stitches partial + new into a full trajectory with reward
        computed on the complete output. The rid is preserved so the surviving
        engine's rid->server affinity keeps all samples of one input on the
        same vLLM (KV reuse). At step 0 fork and surviving engines share the
        same (version-0) weights, so stitching partial + new logprobs is
        on-policy correct.
        """
        prompt_ids: list[int] = list(cont["prompt_ids"])
        partial_ids: list[int] = list(cont["partial_ids"])
        partial_logprobs: list[float] = list(cont["partial_logprobs"])
        partial_versions: list[int] = list(cont["partial_versions"])
        rid: str = cont["rid"]

        input_ids = prompt_ids + partial_ids
        remaining = min(
            self.gconfig.max_tokens - len(input_ids),
            self.gconfig.max_new_tokens - len(partial_ids),
        )
        req = ModelRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        # agenerate re-validates max_new_tokens internally.
        req.gconfig.max_new_tokens = max(remaining, 0)

        prompt_str = self.tokenizer.decode(prompt_ids)
        logger.info(
            f"[step0-fork] rid={rid}: resuming continuation on surviving engine "
            f"(prompt={len(prompt_ids)} partial={len(partial_ids)} "
            f"remaining={remaining} tokens)"
        )

        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        # Stitch partial + newly generated tokens into a full response.
        full_resp = ModelResponse(
            input_tokens=prompt_ids,
            output_tokens=partial_ids + list(resp.output_tokens),
            output_logprobs=partial_logprobs + list(resp.output_logprobs),
            output_versions=partial_versions + list(resp.output_versions),
            stop_reason=resp.stop_reason,
            tokenizer=self.tokenizer,
        )
        logger.info(
            f"[step0-fork] rid={rid}: continuation done "
            f"(new={resp.output_len} full_output={full_resp.output_len} "
            f"stop_reason={resp.stop_reason}); computing reward"
        )

        reward = await self._compute_rewards(full_resp, prompt_str, data)
        log_reward_metrics(reward, data)
        return self._build_trajectory(full_resp, reward)

    @staticmethod
    def _build_trajectory(
        resp: ModelResponse, reward: float
    ) -> dict[str, torch.Tensor]:
        """Build the result tensor dict with batch dim 1 from a response."""
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
