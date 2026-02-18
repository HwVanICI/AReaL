from __future__ import annotations

import argparse
import importlib
import shlex
import sys
from dataclasses import dataclass
from typing import Any

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MegatronEngineConfig, MindSpeedLLMEngineConfig
from areal.utils import logging

logger = logging.getLogger("MindSpeedLLMAdapter")


@dataclass
class ParsedCLIArgs:
    namespace: argparse.Namespace
    explicit_keys: set[str]

def _extract_explicit_keys(tokens: list[str]) -> set[str]:
    keys: set[str] = set()
    for tok in tokens:
        if not tok.startswith("--"):
            continue
        name = tok.split("=", maxsplit=1)[0][2:].replace("-", "_")
        if name:
            keys.add(name)
    return keys


def _build_base_cli_tokens(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    megatron_cfg: MegatronEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> list[str]:
    tokens = [
        "--use-mcore-models",
        "--stage",
        str(backend_cfg.stage),
        "--tensor-model-parallel-size",
        str(parallel_strategy.tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(parallel_strategy.pipeline_parallel_size),
        "--expert-model-parallel-size",
        str(parallel_strategy.expert_parallel_size),
        "--context-parallel-size",
        str(parallel_strategy.context_parallel_size),
    ]
    if megatron_cfg.recompute_method is not None:
        tokens.extend(["--recompute-method", str(megatron_cfg.recompute_method)])
    if megatron_cfg.recompute_granularity is not None:
        tokens.extend(
            ["--recompute-granularity", str(megatron_cfg.recompute_granularity)]
        )
    if megatron_cfg.recompute_num_layers is not None:
        tokens.extend(["--recompute-num-layers", str(megatron_cfg.recompute_num_layers)])
    return tokens


def parse_extra_cli_args(
    *,
    extra_cli_args: str,
    backend_cfg: MindSpeedLLMEngineConfig,
    megatron_cfg: MegatronEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> ParsedCLIArgs:
    tokens = shlex.split(extra_cli_args or "", posix=True)
    explicit_keys = _extract_explicit_keys(tokens)
    base_tokens = _build_base_cli_tokens(
        backend_cfg=backend_cfg,
        megatron_cfg=megatron_cfg,
        parallel_strategy=parallel_strategy,
    )
    argv = ["areal-mindspeed-llm"] + base_tokens + tokens
    from megatron.training import get_args
    from megatron.training.global_vars import unset_global_variables
    from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
    from mindspeed_llm.features_manager import create_features_list
    from mindspeed_llm.training.initialize import initialize_megatron

    MindSpeedFeaturesManager.set_features_list(create_features_list())

    old_argv = sys.argv
    try:
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
        sys.argv = old_argv
    return ParsedCLIArgs(namespace=args, explicit_keys=explicit_keys)


def _maybe_cast_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        ("context_parallel_size", parallel_strategy.context_parallel_size, "cp"),
    ]
    mismatches: list[str] = []
    for key, alloc_value, short in checks:
        if key not in explicit_keys:
            continue
        parsed_value = _maybe_cast_int(getattr(extra_args, key, None))
        if parsed_value is None or parsed_value == alloc_value:
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
    megatron_cfg: MegatronEngineConfig,
    parallel_strategy: ParallelStrategy,
) -> argparse.Namespace:
    parsed = parse_extra_cli_args(
        extra_cli_args=backend_cfg.extra_cli_args,
        backend_cfg=backend_cfg,
        megatron_cfg=megatron_cfg,
        parallel_strategy=parallel_strategy,
    )

    validate_parallel_consistency(
        extra_args=parsed.namespace,
        explicit_keys=parsed.explicit_keys,
        parallel_strategy=parallel_strategy,
        strict=backend_cfg.strict_arg_validation,
    )

    args = parsed.namespace
    # Keep explicit semantic alignment with mcore path in MindSpeed-LLM.
    args.use_mcore_models = True
    args.use_legacy_models = False
    return args


def apply_mindspeed_llm_patches(
    *,
    backend_cfg: MindSpeedLLMEngineConfig,
    megatron_cfg: MegatronEngineConfig,
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
        megatron_cfg=megatron_cfg,
        parallel_strategy=parallel_strategy,
    )

    from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
    from mindspeed_llm.features_manager import create_features_list

    # Ensure a clean patch manager state before applying MindSpeed-LLM patches.
    MindSpeedFeaturesManager.remove_patches()
    MindSpeedFeaturesManager.set_features_list(create_features_list())

    MindSpeedFeaturesManager.apply_features_pre_patches(args)
    MindSpeedFeaturesManager.apply_features_patches(args)

    logger.info(
        "Applied MindSpeed-LLM patches (stage=%s, modeling_mode=%s, use_mcore_models=%s).",
        getattr(args, "stage", None),
        backend_cfg.modeling_mode,
        getattr(args, "use_mcore_models", None),
    )
    return args


def namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(args))
