"""GRPO on DAPO-Math-17k with AReaL's math_verify reward.

Mirrors ``examples/math/boba_grpo.py``: a custom dataset loader + a rule-based
``reward_fn`` + ``areal.workflow.rlvr.RLVRWorkflow``. The dataset and reward
conventions follow the DAPO-Math-17k recipes in
``D:\\workplace\\tdt_verl`` (verl, parquet ``{prompt, extra_info,
reward_model.ground_truth}``) and ``D:\\workspace_latest2\\vime`` (jsonl
``{prompt, label}``), but the grader is AReaL's ``get_math_verify_worker`` --
the general counterpart of vime's ``math`` rm-type (``grade_answer_verl``) and,
unlike vime's ``deepscaler`` rm-type, it does NOT depend on a
triple-quote separator, so it works for plain chat/Qwen3 thinking outputs that only emit
``\\boxed{}``.

The algorithm is plain GRPO (group reward-norm + batch adv-norm, single
``eps_clip``), NOT DAPO -- there is no overlong penalty or decoupled clip.

Run::

    python3 examples/math/math17k_grpo.py --config examples/math/math17k_grpo_npu.yaml
"""

import os
import sys

from datasets import load_dataset
from math_verify import parse, verify

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("Math17kGRPO")


def get_math17k_dataset(path: str, tokenizer, max_length: int | None = None):
    """Load DAPO-Math-17k into AReaL RL format.

    Accepts either a HuggingFace repo id (e.g. ``BytedTsinghua-SIA/DAPO-Math-17k``
    -- verl's source) or a local ``.jsonl`` file (vime's
    ``zhuzilin/dapo-math-17k`` mirror layout with ``prompt``/``label`` keys).

    Each returned sample carries ``messages`` (a single user turn) and ``label``
    (the ground-truth answer string). The boxed-answer instruction is appended
    to the prompt when not already present. Compatible with both verl-style
    samples (ground truth under ``reward_model.ground_truth``) and vime-style
    samples (top-level ``label``).
    """
    if path.endswith(".jsonl") or (os.path.exists(path) and os.path.isfile(path)):
        dataset = load_dataset("json", split="train", data_files=path)
    else:
        dataset = load_dataset(path, split="train")

    def process(sample):
        prompt = sample["prompt"]
        # prompt may be a plain string or a list of chat messages
        if isinstance(prompt, list):
            content = ""
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
        else:
            content = str(prompt)

        if "\\boxed" not in content and "final answer" not in content.lower():
            content = content + "\nPlease put your final answer within \\boxed{}."

        reward_model = sample.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            label = reward_model["ground_truth"]
        else:
            label = sample.get("label", sample.get("answer", ""))

        return {
            "messages": [{"role": "user", "content": content}],
            "label": str(label),
        }

    keep = {"messages", "label"}
    dataset = dataset.map(process).remove_columns(
        [c for c in dataset.column_names if c not in keep]
    )

    if max_length is not None:
        dataset = dataset.filter(
            lambda x: len(tokenizer.encode(x["messages"][0]["content"])) <= max_length
        )
    return dataset


# Fast-path answer extraction adapted from verl/utils/reward_score/math_dapo.py
# (strict-box path). Pure string + brace matching, no sympy: cannot trigger the
# _eval_power pathological path that caused the rollout-worker OOM. Used as a
# fast path before falling back to math_verify (sympy) for symbolic equivalence.


def _last_boxed_only_string(string: str) -> str | None:
    """Extract the last ``\\boxed{...}`` expression via brace matching."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                return string[idx : i + 1]
        i += 1
    return None


def _remove_boxed(s: str) -> str | None:
    """Strip the ``\\boxed{...}`` wrapper and return the inner content."""
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return None
    return s[len(left) : -1]


def _strict_box_correct(pred: str, gt: str) -> bool:
    """verl math_dapo strict-box: extract the last boxed from ``pred[-100:]``
    and compare to ``gt`` by string equality (``gt`` is NOT normalized)."""
    boxed = _last_boxed_only_string(pred[-100:])
    extracted = _remove_boxed(boxed) if boxed is not None else None
    return extracted is not None and extracted == gt


def math17k_reward_fn(
    prompts, completions, prompt_ids, completion_ids, label, **kwargs
) -> float:
    """Rule-based math reward: verl ``math_dapo`` strict-box fast-path + ``math_verify`` fallback.

    Fast-path (pure string, no sympy): extract the last ``\\boxed{}`` from the
    completion tail and compare to ``label`` by string equality. Hits when the
    model emits the same LaTeX rendering as the ground truth (the common case)
    and never touches sympy, so it cannot trigger the ``_eval_power`` pathological
    path that caused the rollout-worker OOM.

    Fallback (sympy via ``math_verify``): when the string comparison misses,
    take the LAST ``\\boxed{}`` in the completion and compare it to ``label``
    with ``math_verify`` for symbolic equivalence (e.g. ``\\frac{1}{2}`` vs
    ``0.5``). Only a ``\\boxed{}`` is ever accepted as the answer -- the raw
    completion is never ``parse()``d, because math_verify's extraction would
    harvest bare numbers / inline latex from the reasoning chain and match the
    gt by accident (scoring 1.0 for a response truncated before its final
    answer); verl's strict-box path has the same tail-only property. Uses
    math_verify's default ``signal.alarm`` timeout, valid in the
    ``ProcessPoolExecutor`` worker's main thread.

    Returns 1.0 on a match, -1.0 otherwise (including parse/timeout failure).
    """
    completion = str(completions)
    gt = str(label)

    # Fast path: verl math_dapo strict-box string comparison (no sympy).
    try:
        if _strict_box_correct(completion, gt):
            return 1.0
    except Exception:
        pass

    # Fallback: take the LAST \boxed{} in the completion (not just the tail)
    # and compare with math_verify symbolic equivalence. We do NOT parse() the
    # raw completion: math_verify's extraction (try_extract_without_anchor /
    # LatexExtractionConfig) harvests bare numbers and inline latex ($113$)
    # from the reasoning chain, which can equal the gt by chance and score 1.0
    # for a response that never emitted a final answer (e.g. truncated mid-CoT
    # by max_tokens). verl avoids this by only looking at the tail "Answer:" or
    # \boxed{}; we mirror that by only ever accepting a \boxed{} as the answer.
    try:
        boxed = _last_boxed_only_string(completion)
        if boxed:
            extracted = _remove_boxed(boxed)
            if extracted is not None:
                ans = parse("\\boxed{" + extracted + "}")
                gold = parse("\\boxed{" + gt + "}")
                if ans and gold and verify(gold, ans):
                    return 1.0
    except Exception:
        pass

    return -1.0


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_math17k_dataset(
        config.train_dataset.path,
        tokenizer,
        getattr(config.train_dataset, "max_length", None),
    )

    # Eval on AIME-2024 if configured; degrade gracefully if unavailable.
    valid_dataset = None
    eval_workflow = None
    eval_workflow_kwargs = None
    vcfg = getattr(config, "valid_dataset", None)
    if vcfg is not None and getattr(vcfg, "path", None):
        try:
            valid_dataset = get_math17k_dataset(
                vcfg.path, tokenizer, getattr(vcfg, "max_length", None)
            )
            eval_workflow = "areal.workflow.rlvr.RLVRWorkflow"
            eval_workflow_kwargs = dict(
                reward_fn="examples.math.math17k_grpo.math17k_reward_fn",
                gconfig=config.gconfig,
                tokenizer=config.tokenizer_path,
                enable_thinking=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load valid_dataset from {vcfg.path!r}: {e}. "
                "Evaluation will be skipped."
            )
            valid_dataset = None

    workflow_kwargs = dict(
        reward_fn="examples.math.math17k_grpo.math17k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
