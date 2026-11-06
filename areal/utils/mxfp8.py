# SPDX-License-Identifier: Apache-2.0

"""Ascend MXFP8 quantization shared by training and inference loaders."""

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
