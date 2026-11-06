# SPDX-License-Identifier: Apache-2.0

"""Worker-side assertions invoked through AWEX's existing test RPC entrypoint."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def check_mxfp8_update(
    *, model, model_context, prepare=False, expected_value=None, check_graph=False
):
    from areal.engine.vllm_ext import ascend_mxfp8

    worker = model_context["scheduler"]
    if check_graph:
        assert worker.model_runner.use_aclgraph, "ACL Graph must be enabled"
    modules = list(ascend_mxfp8.iter_mxfp8_modules(model))
    pointers = {
        f"{prefix}.{name}": getattr(module, name).data_ptr()
        for prefix, module, _ in modules
        for name in module._mxfp8_original_shapes
    }
    torch.npu.synchronize()
    if prepare:
        worker._awex_test_mxfp8_pointers = pointers
        # Do not let temporary allocations from the previous reference checks
        # inflate the next update's reserved-memory baseline.
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        return {}

    # Read the update peak before allocating any reference tensors for assertions.
    result = {
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "checked_weights": 0,
    }
    assert pointers == worker._awex_test_mxfp8_pointers, (
        "An update or sleep/wake changed an inference parameter's address"
    )
    receiver = worker._awex_ascend_mxfp8_receiver
    assert not receiver._receiving, "MXFP8 receive is still in progress"

    import torch_npu

    for _, module, _ in modules:
        for name, shape in module._mxfp8_original_shapes.items():
            if name.endswith("_scale"):
                continue
            weight = getattr(module, name)
            scale = getattr(module, name + "_scale")
            k = shape[-1]
            # The actor writes the same controlled value to every matrix element.
            # One reference row suffices for all local rows and experts.
            source = torch.full(
                (1, k), expected_value, dtype=torch.bfloat16, device=weight.device
            )
            expected_weight, native_scale = torch_npu.npu_dynamic_mx_quant(
                source, axis=-1, dst_type=torch_npu.float8_e4m3fn
            )
            expected_scale = native_scale.reshape(1, -1)[:, : (k + 31) // 32]
            if expected_scale.shape[-1] % 2:
                expected_scale = F.pad(expected_scale, (0, 1))
            expected_scale = expected_scale.reshape(1, -1, 2).transpose(0, 1)
            matrices = [(weight, scale)] if len(shape) == 2 else zip(weight, scale)
            for local_weight, local_scale in matrices:
                assert tuple(local_weight.shape) == (k, shape[-2])
                assert tuple(local_scale.shape) == ((k + 63) // 64, shape[-2], 2)
                # Bound assertion temporaries instead of making a full FP32 copy.
                for start in range(0, shape[-2], 256):
                    weight_slice = local_weight[:, start : start + 256]
                    scale_slice = local_scale[:, start : start + 256, :]
                    torch.testing.assert_close(
                        weight_slice.float(),
                        expected_weight.T.float().expand_as(weight_slice),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scale_slice,
                        expected_scale.expand_as(scale_slice),
                        rtol=0,
                        atol=0,
                    )
            result["checked_weights"] += 1
    return result
