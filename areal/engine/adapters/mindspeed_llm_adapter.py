from __future__ import annotations

import importlib
import shlex

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MindSpeedLLMEngineConfig
from areal.utils import logging

logger = logging.getLogger("MindSpeedLLMAdapter")


def _extract_explicit_values(tokens: list[str]) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key, sep, val = tok[2:].partition("=")
        key = key.replace("-", "_")
        if not key:
            i += 1
            continue
        if sep:
            values[key] = val
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            values[key] = tokens[i + 1]
            i += 2
            continue
        values[key] = True
        i += 1
    return values


def _build_base_cli_tokens(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> list[str]:
    return [
        "--use-mcore-models",
        "--stage",
        str(backend_cfg.stage),
        "--tensor-model-parallel-size",
        str(parallel_strategy.tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(parallel_strategy.pipeline_parallel_size),
        "--expert-model-parallel-size",
        str(parallel_strategy.expert_parallel_size),
        "--expert-tensor-parallel-size",
        str(parallel_strategy.expert_tensor_parallel_size),
        "--context-parallel-size",
        str(parallel_strategy.context_parallel_size),
    ]


def validate_parallel_consistency(
    *,
    explicit_values: dict[str, str | bool],
    parallel_strategy: ParallelStrategy,
    strict: bool,
) -> None:
    if not strict:
        return

    checks = [
        ("tensor_model_parallel_size", parallel_strategy.tensor_parallel_size, "tp"),
        ("pipeline_model_parallel_size", parallel_strategy.pipeline_parallel_size, "pp"),
        ("expert_model_parallel_size", parallel_strategy.expert_parallel_size, "ep"),
        (
            "expert_tensor_parallel_size",
            parallel_strategy.expert_tensor_parallel_size,
            "etp",
        ),
        ("context_parallel_size", parallel_strategy.context_parallel_size, "cp"),
    ]
    mismatches: list[str] = []
    for key, alloc_value, short in checks:
        raw = explicit_values.get(key, None)
        if raw is None:
            continue
        try:
            parsed_value = int(raw)
        except (TypeError, ValueError):
            parsed_value = None
        if parsed_value in (None, alloc_value):
            continue
        mismatches.append(
            f"{short}: allocation_mode={alloc_value}, extra_cli_args={parsed_value}"
        )

    if mismatches:
        raise ValueError(
            "MindSpeed-LLM parallel settings conflict with allocation_mode: "
            + "; ".join(mismatches)
        )


def _build_mindspeed_llm_argv(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> list[str]:
    extra_tokens = shlex.split(backend_cfg.extra_cli_args or "", posix=True)
    explicit_values = _extract_explicit_values(extra_tokens)

    validate_parallel_consistency(
        explicit_values=explicit_values,
        parallel_strategy=parallel_strategy,
        strict=backend_cfg.strict_arg_validation,
    )
    base_tokens = _build_base_cli_tokens(
        backend_cfg=backend_cfg,
        parallel_strategy=parallel_strategy,
    )
    return ["areal-mindspeed-llm", *base_tokens, *extra_tokens]


def apply_mindspeed_llm_patches(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> list[str]:
    # Hard dependency checks.
    for package in ("mindspeed_llm", "mindspeed"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(
                f"Required package '{package}' is not installed or importable. "
            "MindSpeed-LLM backend requires mindspeed + mindspeed_llm."
            )

    argv = _build_mindspeed_llm_argv(
        backend_cfg=backend_cfg,
        parallel_strategy=parallel_strategy,
    )

    logger.info(
        "MindSpeed-LLM adaptor prepared argv (stage=%s, modeling_mode=%s).",
        backend_cfg.stage,
        backend_cfg.modeling_mode,
    )
    return argv
