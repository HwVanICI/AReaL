# SPDX-License-Identifier: Apache-2.0

"""Ascend MXFP8 quantization and static weight-transfer descriptions."""

from collections.abc import Iterable, Iterator

import torch


def quantize_mxfp8_tensor(
    tensor: torch.Tensor, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 weights and the native paired E8M0 scale layout."""
    import torch_npu

    fp8_weight, weight_scale = torch_npu.npu_dynamic_mx_quant(
        tensor, axis=-1, dst_type=torch_npu.float8_e4m3fn
    )
    if fp8_weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"MXFP8 quantization returned invalid dtype for {name}")
    if weight_scale.dtype != torch.uint8:
        raise RuntimeError(f"MXFP8 quantization returned invalid scale for {name}")
    return fp8_weight, weight_scale


@torch.no_grad()
def quantize_mxfp8_for_transfer(
    name: str, tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 byte views and contiguous logical [N, K/32] E8M0 scales."""
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16:
        raise ValueError(f"Ascend MXFP8 {name}: expected a BF16 [N,K] matrix")
    n, k = tensor.shape
    if k <= 0 or k % 32:
        raise ValueError(f"Ascend MXFP8 {name}: K={k} must be divisible by 32")
    weight, scale = quantize_mxfp8_tensor(tensor.contiguous(), name)
    if weight.shape != tensor.shape or tuple(scale.shape) != (n, (k + 63) // 64, 2):
        raise RuntimeError(
            f"Ascend MXFP8 {name}: quantizer returned incompatible shapes"
        )
    # A dtype view preserves every FP8 bit; .to(uint8) would change the values.
    return weight.view(torch.uint8), scale.reshape(n, -1)[:, : k // 32].contiguous()


def merge_ascend_mxfp8_descriptions(
    descriptions: Iterable[dict], *, replicas: bool = False
) -> dict:
    """Union PP/EP worker policies, or require identical complete DP policies."""
    merged = None
    for description in descriptions:
        if not isinstance(description, dict) or description.get("group_size") != 32:
            raise ValueError(
                "Missing or invalid Ascend MXFP8 initialization description"
            )
        weights = description.get("weights")
        if not isinstance(weights, dict) or any(
            scheme not in {"W8A8_MXFP8", "FLOAT"} for scheme in weights.values()
        ):
            raise ValueError("Invalid Ascend MXFP8 weight policy")
        if merged is None:
            merged = dict(weights)
        elif replicas:
            if merged != weights:
                raise ValueError(
                    "Inference replicas have different Ascend MXFP8 policies"
                )
        else:
            for name, scheme in weights.items():
                if name in merged and merged[name] != scheme:
                    raise ValueError(f"Conflicting Ascend MXFP8 policy for {name}")
                merged[name] = scheme
    if merged is None or "W8A8_MXFP8" not in merged.values():
        raise ValueError("No Ascend MXFP8 weights were described by initialized vLLM")
    return {"group_size": 32, "weights": merged}


def iter_ascend_mxfp8_payloads(
    weights: Iterable[tuple[str, torch.Tensor]], description: dict
) -> Iterator[tuple[str, torch.Tensor]]:
    """Quantize full HF matrices selected by the initialized inference model."""
    policies = description["weights"]
    for name, tensor in weights:
        if policies.get(name) == "W8A8_MXFP8":
            weight, scale = quantize_mxfp8_for_transfer(name, tensor)
            yield name, weight
            yield name + "_scale", scale
        else:
            yield name, tensor
