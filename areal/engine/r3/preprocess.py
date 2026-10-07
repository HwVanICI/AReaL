# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import pybase64
import torch

from areal.engine.r3.asserts import r3_error


def decode_vllm_routed_experts(
    raw_routed_experts: str,
    *,
    source: str,
) -> np.ndarray:
    """Decode vLLM base64 ``.npy`` routing into ``[tokens, layers, topk]``."""
    try:
        decoded = pybase64.b64decode(raw_routed_experts)
        routed_experts = np.load(io.BytesIO(decoded), allow_pickle=False)
    except Exception as exc:
        raise r3_error("Failed to decode vLLM routed_experts", source=source) from exc
    if routed_experts.ndim != 3:
        raise r3_error(
            "vLLM routed_experts must have shape [tokens, layers, topk]",
            source=source,
            shape=tuple(routed_experts.shape),
        )
    return routed_experts


def decode_sglang_routed_experts(
    raw_routed_experts: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    source: str,
) -> np.ndarray:
    """Decode SGLang base64 int32 routed_experts into ``[tokens, flat_dim]``."""

    num_sgl_tokens = int(prompt_tokens) + int(completion_tokens) - 1
    if num_sgl_tokens < 0:
        raise r3_error(
            "Invalid SGLang token counts for routed_experts",
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    try:
        decoded = pybase64.b64decode(raw_routed_experts.encode("utf-8"))
    except Exception as exc:
        raise r3_error(
            "Failed to base64 decode SGLang routed_experts",
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ) from exc

    if len(decoded) % np.dtype(np.int32).itemsize != 0:
        raise r3_error(
            "Decoded SGLang routed_experts byte length is not int32-aligned",
            source=source,
            byte_len=len(decoded),
        )
    flat = np.frombuffer(decoded, dtype=np.int32)
    if num_sgl_tokens == 0:
        if flat.size != 0:
            raise r3_error(
                "SGLang routed_experts returned data for an empty token range",
                source=source,
                flat_size=flat.size,
            )
        return flat.reshape(0, 0)
    if flat.size % num_sgl_tokens != 0:
        raise r3_error(
            "SGLang routed_experts cannot be reshaped by token count",
            source=source,
            flat_size=flat.size,
            num_sgl_tokens=num_sgl_tokens,
        )
    return flat.reshape(num_sgl_tokens, -1)


def _reshape_routed_experts(
    routed_experts: np.ndarray,
    *,
    num_moe_layers: int,
    topk: int,
    moe_layer_indices: Sequence[int] | None = None,
) -> np.ndarray:
    arr = np.asarray(routed_experts, dtype=np.int32)
    if arr.ndim == 2:
        expected_flat_dim = num_moe_layers * topk
        if arr.shape[1] % topk != 0:
            raise r3_error(
                "routed_experts flat_dim does not match Megatron MoE config",
                raw_shape=arr.shape,
                flat_dim=arr.shape[1],
                num_moe_layers=num_moe_layers,
                topk=topk,
                expected_flat_dim=expected_flat_dim,
            )
        returned_layers = arr.shape[1] // topk
        if returned_layers < num_moe_layers:
            raise r3_error(
                "routed_experts flat_dim does not match Megatron MoE config",
                raw_shape=arr.shape,
                flat_dim=arr.shape[1],
                num_moe_layers=num_moe_layers,
                topk=topk,
                expected_flat_dim=expected_flat_dim,
                returned_layers=returned_layers,
            )
        arr = arr.reshape(arr.shape[0], returned_layers, topk)
    if arr.ndim == 3:
        if arr.shape[2] == topk and arr.shape[1] > num_moe_layers:
            if moe_layer_indices is not None:
                return arr[:, moe_layer_indices, :]
            prefix_len = arr.shape[1] - num_moe_layers
            if np.any(arr[:, :prefix_len, :] != 0):
                raise r3_error(
                    "Cannot discard nonzero routed_experts layer slots; "
                    "only leading dense slots are supported",
                    raw_shape=arr.shape,
                    num_moe_layers=num_moe_layers,
                    topk=topk,
                )
            return arr[:, prefix_len:, :]
        if arr.shape[1:] != (num_moe_layers, topk):
            raise r3_error(
                "routed_experts layer/topk dims do not match Megatron MoE config",
                raw_shape=arr.shape,
                num_moe_layers=num_moe_layers,
                topk=topk,
            )
        return arr
    raise r3_error("Unsupported routed_experts ndim", raw_shape=arr.shape)


def _preprocess_one(
    routed_experts: np.ndarray | None,
    *,
    seq_len: int,
    num_moe_layers: int,
    topk: int,
    moe_layer_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, bool]:
    out = np.zeros((seq_len, num_moe_layers, topk), dtype=np.int32)
    if routed_experts is None:
        return out, False

    arr = _reshape_routed_experts(
        routed_experts,
        num_moe_layers=num_moe_layers,
        topk=topk,
        moe_layer_indices=moe_layer_indices,
    )
    if arr.shape[0] > seq_len:
        raise r3_error(
            "routed_experts has more token rows than the training sequence",
            raw_shape=arr.shape,
            seq_len=seq_len,
        )

    copied = arr.shape[0]
    if copied > 0:
        out[:copied] = arr
        if copied < seq_len:
            out[copied:] = arr[-1]

    valid = copied >= max(seq_len - 1, 0)

    return out, valid


def preprocess_routed_experts_batch(
    routed_experts: Sequence[np.ndarray | None],
    *,
    seq_lens: Sequence[int],
    num_moe_layers: int,
    topk: int,
    moe_layer_indices: Sequence[int] | None = None,
    max_seq_len: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    if len(routed_experts) != len(seq_lens):
        raise r3_error(
            "routed_experts batch size does not match seq_lens",
            routed_batch_size=len(routed_experts),
            seq_lens_size=len(seq_lens),
        )
    if max_seq_len is None:
        max_seq_len = max(seq_lens, default=0)

    batch = np.zeros(
        (len(routed_experts), max_seq_len, num_moe_layers, topk),
        dtype=np.int32,
    )
    valid: list[bool] = []
    for idx, (sample_routing, seq_len) in enumerate(zip(routed_experts, seq_lens)):
        if seq_len > max_seq_len:
            raise r3_error(
                "seq_len exceeds max_seq_len",
                sample_idx=idx,
                seq_len=seq_len,
                max_seq_len=max_seq_len,
            )
        sample, sample_valid = _preprocess_one(
            sample_routing,
            seq_len=int(seq_len),
            num_moe_layers=num_moe_layers,
            topk=topk,
            moe_layer_indices=moe_layer_indices,
        )
        batch[idx, : int(seq_len)] = sample
        valid.append(sample_valid)

    return {
        "routed_experts": torch.as_tensor(batch, dtype=torch.int32, device=device),
        "r3_routing_valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
    }
