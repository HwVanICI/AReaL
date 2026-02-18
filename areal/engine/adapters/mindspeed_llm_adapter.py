from __future__ import annotations

import argparse
import importlib
import shlex
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

def _add_unknown_args(args: dict[str, Any], key: str | None, value: list[str] | None):
    if key is None:
        return
    name = key[2:].replace("-", "_")
    if value is None:
        args[name] = True
    elif len(value) == 1:
        args[name] = value[0]
    else:
        args[name] = value


def _parse_unknown_tokens(tokens: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    key: str | None = None
    value: list[str] | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            _add_unknown_args(parsed, key, value)
            splits = tok.split("=", maxsplit=1)
            if len(splits) == 2:
                key, value = splits[0], [splits[1]]
            else:
                key, value = tok, None
        else:
            if value is None:
                value = [tok]
            else:
                value.append(tok)
        i += 1
    _add_unknown_args(parsed, key, value)
    return parsed


def _extract_explicit_keys(tokens: list[str]) -> set[str]:
    keys: set[str] = set()
    for tok in tokens:
        if not tok.startswith("--"):
            continue
        name = tok.split("=", maxsplit=1)[0][2:].replace("-", "_")
        if name:
            keys.add(name)
    return keys


def parse_extra_cli_args(extra_cli_args: str) -> ParsedCLIArgs:
    tokens = shlex.split(extra_cli_args or "", posix=True)
    explicit_keys = _extract_explicit_keys(tokens)
    # Parse args through the same registration path as MindSpeed-LLM launcher:
    # Megatron native args + MindSpeed args + MindSpeed-LLM args.
    try:
        from mindspeed_llm.training.arguments import process_args_v2
    except Exception as e:  # pragma: no cover - covered by dependency checks
        raise RuntimeError(
            "Failed to load MindSpeed-LLM argument registry. "
            "Please ensure mindspeed and mindspeed_llm are installed correctly."
        ) from e

    parser = argparse.ArgumentParser(
        description="MindSpeed-LLM extra args", allow_abbrev=False
    )
    parser = process_args_v2(parser)
    known_args, unknown = parser.parse_known_args(tokens)
    merged = vars(known_args)
    merged.update(_parse_unknown_tokens(unknown))
    return ParsedCLIArgs(namespace=argparse.Namespace(**merged), explicit_keys=explicit_keys)


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
    parsed = parse_extra_cli_args(backend_cfg.extra_cli_args)

    validate_parallel_consistency(
        extra_args=parsed.namespace,
        explicit_keys=parsed.explicit_keys,
        parallel_strategy=parallel_strategy,
        strict=backend_cfg.strict_arg_validation,
    )

    base = {
        "use_mcore_models": True,
        "use_legacy_models": False,
        "stage": backend_cfg.stage,
        "tensor_model_parallel_size": parallel_strategy.tensor_parallel_size,
        "pipeline_model_parallel_size": parallel_strategy.pipeline_parallel_size,
        "expert_model_parallel_size": parallel_strategy.expert_parallel_size,
        "context_parallel_size": parallel_strategy.context_parallel_size,
        "recompute_method": megatron_cfg.recompute_method,
        "recompute_granularity": megatron_cfg.recompute_granularity,
        "recompute_num_layers": megatron_cfg.recompute_num_layers,
    }
    parsed_dict = vars(parsed.namespace)
    for key, value in parsed_dict.items():
        if key not in base:
            base[key] = value
    for key in parsed.explicit_keys:
        if key in parsed_dict:
            base[key] = parsed_dict[key]
    return argparse.Namespace(**base)


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

    # MegatronEngine imports mindspeed.megatron_adaptor at module load time,
    # which may already register/apply a patch set. Clear it before applying
    # MindSpeed-LLM feature patches to avoid duplicate registration errors
    # (e.g. "the patch of compile exist !").
    try:
        MindSpeedFeaturesManager.remove_patches()
    except Exception:
        logger.warning("Failed to remove existing MindSpeed patches before re-patching.")

    MindSpeedFeaturesManager.set_features_list(create_features_list())

    if backend_cfg.set_megatron_global_args:
        try:
            from megatron.training.global_vars import set_args

            set_args(args)
        except Exception:
            logger.warning(
                "Failed to set megatron global args before applying MindSpeed-LLM patches."
            )

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
