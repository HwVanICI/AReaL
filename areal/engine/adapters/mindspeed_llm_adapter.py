from __future__ import annotations

import argparse
import importlib
import shlex
import sys

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MindSpeedLLMEngineConfig
from areal.utils import logging

logger = logging.getLogger("MindSpeedLLMAdapter")


def _extract_explicit_keys(tokens: list[str]) -> set[str]:
    keys: set[str] = set()
    for tok in tokens:
        if not tok.startswith("--"):
            continue
        name = tok.split("=", maxsplit=1)[0][2:].replace("-", "_")
        if name:
            keys.add(name)
    return keys


def _unwrap_function(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


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


def parse_extra_cli_args(
    *,
    extra_cli_args: str,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> tuple[argparse.Namespace, set[str]]:
    tokens = shlex.split(extra_cli_args or "", posix=True)
    explicit_keys = _extract_explicit_keys(tokens)

    base_tokens = _build_base_cli_tokens(
        backend_cfg=backend_cfg,
        parallel_strategy=parallel_strategy,
    )
    argv = ["areal-mindspeed-llm"] + base_tokens + tokens
    import megatron.training.arguments as megatron_arguments
    from megatron.training import get_args
    from megatron.training.global_vars import unset_global_variables
    from megatron.training.initialize import initialize_megatron

    old_argv = sys.argv
    old_parse_args = megatron_arguments.parse_args
    try:
        megatron_arguments.parse_args = _unwrap_function(old_parse_args)
        sys.argv = argv
        # Make this parser path idempotent in long-lived processes.
        unset_global_variables()
        initialize_megatron(
            extra_args_provider=None,
            args_defaults={},
            ignore_unknown_args=False,
            allow_no_cuda=True,
            skip_mpu_initialization=True,
        )
        args = get_args()
    finally:
        megatron_arguments.parse_args = old_parse_args
        sys.argv = old_argv
    return args, explicit_keys


def validate_parallel_consistency(
    *,
    extra_args: argparse.Namespace,
    explicit_keys: set[str],
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
        if key not in explicit_keys:
            continue
        raw = getattr(extra_args, key, None)
        try:
            parsed_value = int(raw) if raw is not None else None
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


def build_mindspeed_llm_args(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> argparse.Namespace:
    args, explicit_keys = parse_extra_cli_args(
        extra_cli_args=backend_cfg.extra_cli_args,
        backend_cfg=backend_cfg,
        parallel_strategy=parallel_strategy,
    )

    validate_parallel_consistency(
        extra_args=args,
        explicit_keys=explicit_keys,
        parallel_strategy=parallel_strategy,
        strict=backend_cfg.strict_arg_validation,
    )

    # Keep explicit semantic alignment with mcore path in MindSpeed-LLM.
    args.use_mcore_models = True
    args.use_legacy_models = False
    return args


def apply_mindspeed_llm_patches(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> argparse.Namespace:
    # Hard dependency checks.
    for package in ("mindspeed_llm", "mindspeed"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(
                f"Required package '{package}' is not installed or importable. "
                "MindSpeed-LLM backend requires mindspeed + mindspeed_llm."
            )

    args = build_mindspeed_llm_args(
        backend_cfg=backend_cfg,
        parallel_strategy=parallel_strategy,
    )

    logger.info(
        "MindSpeed-LLM adaptor initialized (stage=%s, modeling_mode=%s, use_mcore_models=%s).",
        getattr(args, "stage", None),
        backend_cfg.modeling_mode,
        getattr(args, "use_mcore_models", None),
    )
    return args
