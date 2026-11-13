# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm_ascend.quantization.methods.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

from areal.infra.platforms import current_platform


def is_ascend_mxfp8(model_runner) -> bool:
    """Return whether this worker is an explicitly configured MXFP8 worker."""
    if current_platform.device_type != "npu":
        return False

    if model_runner.vllm_config.model_config.quantization != "ascend":
        return False

    quant_config = model_runner.vllm_config.quant_config
    if not isinstance(quant_config, AscendModelSlimConfig):
        raise RuntimeError(
            "quantization=ascend did not create an AscendModelSlimConfig"
        )

    quant_description = quant_config.quant_description
    weight_schemes = {
        value
        for key, value in quant_description.items()
        if key.endswith(".weight") and isinstance(value, str)
    }
    unsupported = weight_schemes - {"W8A8_MXFP8", "FLOAT"}
    if unsupported:
        raise RuntimeError(
            "Online Ascend FP8 rollout only supports W8A8_MXFP8/FLOAT, got "
            f"{sorted(unsupported)}"
        )
    if "W8A8_MXFP8" not in weight_schemes:
        raise RuntimeError(
            "quant_model_description.json does not contain W8A8_MXFP8 weights"
        )
    for name, module in model_runner.model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        wrapper = getattr(module, "quant_method", None)
        scheme = getattr(wrapper, "quant_method", None)
        if not isinstance(scheme, AscendW8A8MXFP8DynamicFusedMoEMethod):
            raise RuntimeError(
                "Online Ascend MXFP8 rollout requires every FusedMoE module "
                f"to use W8A8_MXFP8; module {name!r} is FLOAT"
            )
    group_size = quant_description.get("group_size", 32)
    if group_size != 32:
        raise RuntimeError("Online Ascend MXFP8 rollout requires group_size=32")

    return True


def _get_module_from_weight_name(model, name: str):
    module_path = name.split(".")[:-1]
    if not module_path:
        return None

    packed_mapping = getattr(model, "packed_modules_mapping", {})
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names in packed_mapping.items()
        for original_name in original_names
    }
    module_path[-1] = reversed_mapping.get(module_path[-1], module_path[-1])

    module = model
    try:
        for part in module_path:
            if isinstance(module, FusedMoE):
                return module
            if isinstance(module, torch.nn.ModuleList):
                module = module[int(part)]
            else:
                module = getattr(module, part)
    except (AttributeError, IndexError, ValueError):
        return None
    return module


def _get_mxfp8_parameters(model, name: str):
    if not name.endswith("weight"):
        return None
    module = _get_module_from_weight_name(model, name)
    if module is None:
        raise RuntimeError(
            f"Could not resolve actor weight {name!r} to a vLLM module; refusing "
            "to load it without an MXFP8 scale"
        )

    if isinstance(module, LinearBase):
        if module.weight.dtype != torch.float8_e4m3fn:
            return None
        scale = getattr(module, "weight_scale", None)
        if scale is None:
            raise RuntimeError(f"MXFP8 module for {name!r} has no weight_scale")
        return module.weight, scale, "linear"
    if isinstance(module, FusedMoE):
        if ".down_proj." in name:
            target_names = "w2_weight", "w2_weight_scale"
        elif ".gate_proj." in name or ".up_proj." in name:
            target_names = "w13_weight", "w13_weight_scale"
        else:
            raise RuntimeError(
                f"Could not map MoE weight {name!r} to an MXFP8 parameter"
            )
        target_weight = getattr(module, target_names[0], None)
        target_scale = getattr(module, target_names[1], None)
        if target_weight is None:
            raise RuntimeError(
                f"MoE module for {name!r} has no {target_names[0]} Parameter"
            )
        if target_weight.dtype != torch.float8_e4m3fn:
            return None
        if target_scale is None:
            raise RuntimeError(
                f"MXFP8 module for {name!r} has no {target_names[1]} Parameter"
            )
        kind = "moe_w2" if target_names[0] == "w2_weight" else "moe_w13"
        return target_weight, target_scale, kind
    return None


def _validate_target_tensor(
    name: str,
    tensor: torch.Tensor,
    target: torch.Tensor,
    kind: str,
) -> None:
    if tensor.dtype != target.dtype:
        raise RuntimeError(
            f"MXFP8 tensor {name} has dtype {tensor.dtype}, but target Parameter "
            f"has dtype {target.dtype}"
        )

    source_shape = tuple(tensor.shape)
    target_shape = tuple(target.shape)
    dense_rank = kind == "linear" and tensor.ndim == target.ndim
    moe_rank = (
        kind in {"moe_w13", "moe_w2"}
        and target.ndim == tensor.ndim + 1
        and target_shape[0] > 0
    )
    source_last = source_shape[-1] if source_shape else 0
    target_last = target_shape[-1] if target_shape else 0
    compatible_last_dim = (
        source_last > 0
        and target_last > 0
        and (source_last % target_last == 0 or target_last % source_last == 0)
    )
    # Actor tensors use full HF shapes. vLLM's weight_loader performs the final
    # TP, packed-linear, and expert shard validation; this helper only rejects
    # rank, group/input dimension, and dtype combinations that cannot be valid.
    if not ((dense_rank or moe_rank) and compatible_last_dim):
        raise RuntimeError(
            f"MXFP8 tensor {name} has shape {source_shape}, but target Parameter "
            f"has incompatible shape {target_shape}"
        )


def iter_mxfp8_modules(
    model: torch.nn.Module,
) -> Iterator[
    tuple[
        str,
        torch.nn.Module,
        AscendW8A8MXFP8DynamicLinearMethod | AscendW8A8MXFP8DynamicFusedMoEMethod,
    ]
]:
    """Yield local modules using the supported Ascend MXFP8 schemes."""
    for name, module in model.named_modules():
        wrapper = getattr(module, "quant_method", None)
        scheme = getattr(wrapper, "quant_method", None)
        if isinstance(
            scheme,
            (
                AscendW8A8MXFP8DynamicLinearMethod,
                AscendW8A8MXFP8DynamicFusedMoEMethod,
            ),
        ):
            yield name, module, scheme


def _validate_loading_shapes(name, module, weight_name, scale_name, *, moe=False):
    original = getattr(module, "_mxfp8_original_shapes", {})
    if getattr(module, "_mxfp8_transformed", False) and not original:
        raise ValueError(f"Ascend MXFP8 {name}: missing original loading shapes")
    weight = getattr(module, weight_name)
    scale = getattr(module, scale_name)
    shape = tuple(original.get(weight_name, weight.shape))
    scale_shape = tuple(original.get(scale_name, scale.shape))
    k = shape[-1]
    # Row-parallel shards start at tp_rank * local K. N shards and EP do not
    # divide quantization groups. Use actual local shapes, not config guesses.
    offset = k * getattr(module, "tp_rank", 0)
    if k <= 0 or k % 32 or offset % 32 or (moe and k % 64):
        alignment = 64 if moe else 32
        raise ValueError(
            f"Ascend MXFP8 {name}: local K={k}, offset={offset}; "
            f"requires K divisible by {alignment} and aligned 32-element groups"
        )
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or scale_shape != (*shape[:-1], k // 32)
    ):
        raise ValueError(
            f"Ascend MXFP8 {name}: incompatible weight/scale loading shapes"
        )


def get_ascend_mxfp8_description(model_runner) -> dict:
    """Describe actual initialized modules in HF names, without reading JSON."""
    if not is_ascend_mxfp8(model_runner):
        raise ValueError("Ascend MXFP8 transfer requires an initialized MXFP8 model")
    model = model_runner.model
    quantized = {name for name, _, _ in iter_mxfp8_modules(model)}
    packed = getattr(model, "packed_modules_mapping", {})
    policies = {}
    described = set()
    for name, module in model.named_modules():
        if isinstance(module, LinearBase):
            parent, _, leaf = name.rpartition(".")
            names = [
                f"{parent + '.' if parent else ''}{part}.weight"
                for part in packed.get(leaf, [leaf])
            ]
            scheme = "W8A8_MXFP8" if name in quantized else "FLOAT"
            if name in quantized:
                _validate_loading_shapes(name, module, "weight", "weight_scale")
                described.add(name)
            policies.update(dict.fromkeys(names, scheme))
        elif isinstance(module, FusedMoE) and name in quantized:
            for weight_name in ("w13_weight", "w2_weight"):
                _validate_loading_shapes(
                    name, module, weight_name, weight_name + "_scale", moe=True
                )
            # The model's loader mapping supplies canonical expert names and
            # global expert IDs, including experts not owned by this EP rank.
            mapping = model.get_expert_mapping()
            parent, _, leaf = name.rpartition(".")
            expert_names = [
                f"{parent + '.' if parent else ''}{source}weight"
                for target, source, _, _ in mapping
                if target in {leaf + ".w13_", leaf + ".w2_"}
            ]
            if not expert_names:
                raise ValueError(
                    f"Cannot map initialized MXFP8 experts {name} to HF names"
                )
            policies.update(dict.fromkeys(expert_names, "W8A8_MXFP8"))
            described.add(name)
    if described != quantized:
        raise ValueError(
            f"Cannot describe MXFP8 modules: {sorted(quantized - described)}"
        )
    return {"group_size": 32, "weights": policies}


def restore_mxfp8_weights_for_loading(model) -> None:
    for _, module, scheme in iter_mxfp8_modules(model):
        scheme.restore_weights_for_rl_loading(module)


def process_mxfp8_weights_after_loading(model) -> None:
    for _, module, _ in iter_mxfp8_modules(model):
        module.quant_method.process_weights_after_loading(module)


def _decode_mxfp8_weights(
    weights: Iterable[tuple[str, torch.Tensor]], model
) -> Iterator[tuple[str, torch.Tensor]]:
    weights = dict(weights)
    consumed_scales = set()
    for name in weights:
        if (
            name.endswith(".weight_scale")
            and name.removesuffix("_scale") not in weights
        ):
            raise ValueError(f"Orphan Ascend MXFP8 scale {name}")
    # Senders place each FP8 weight before its scale in the same bucket.
    # Metadata and broadcast calls preserve this order, and the receiver
    # appends tensors in metadata order, even with asynchronous sends.
    # dict preserves insertion order, so loading a weight records its scale
    # in consumed_scales before the loop reaches that scale entry.
    for name, tensor in weights.items():
        if name in consumed_scales:
            continue
        # Every rollout worker receives every actor bucket. Pipeline-parallel
        # workers must discard weights owned by another vLLM stage before module
        # resolution, matching vLLM's native load_weights behavior.
        if is_pp_missing_parameter(name, model):
            continue
        target_parameters = _get_mxfp8_parameters(model, name)
        if target_parameters is None:
            if name.endswith(".weight_scale") or tensor.dtype == torch.uint8:
                raise ValueError(f"Unexpected Ascend MXFP8 payload {name}")
            yield name, tensor
            continue
        if tensor.dtype != torch.uint8 or tensor.ndim != 2 or tensor.shape[-1] % 32:
            raise ValueError(
                f"Ascend MXFP8 {name}: expected aligned uint8 weight bytes"
            )
        target_weight, target_scale, target_kind = target_parameters
        scale_name = f"{name}_scale"
        weight_scale = weights.get(scale_name)
        if (
            weight_scale is None
            or weight_scale.dtype != torch.uint8
            or tuple(weight_scale.shape) != (tensor.shape[0], tensor.shape[1] // 32)
        ):
            raise ValueError(
                f"Ascend MXFP8 {name}: missing or incompatible uint8 scale"
            )
        consumed_scales.add(scale_name)
        fp8_weight = tensor.view(torch.float8_e4m3fn)
        _validate_target_tensor(name, fp8_weight, target_weight, target_kind)
        _validate_target_tensor(scale_name, weight_scale, target_scale, target_kind)
        yield name, fp8_weight
        yield scale_name, weight_scale


def load_mxfp8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    model_runner,
    *,
    restore: bool,
):
    model = model_runner.model
    if restore:
        restore_mxfp8_weights_for_loading(model)

    return model.load_weights(_decode_mxfp8_weights(weights, model))
