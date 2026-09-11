import sys
from types import ModuleType

import pytest

from areal.engine.patch_awex import _patch_vllm_expert_names


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "model.layers.0.mlp.experts.routed_experts.w13_weight",
            "model.layers.0.mlp.experts.w13_weight",
        ),
        (
            "model.layers.47.mlp.experts.routed_experts.w2_weight",
            "model.layers.47.mlp.experts.w2_weight",
        ),
        (
            "model.layers.0.mlp.experts.routed_experts.w13_weight_scale_inv",
            "model.layers.0.mlp.experts.w13_weight_scale_inv",
        ),
        (
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w13_weight",
        ),
        (
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
        ),
        (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
        ),
    ],
)
def test_vllm_expert_names_patch_preserves_normalizer(monkeypatch, name, expected):
    """Normalize nested experts once and delegate to the original converter."""
    calls = []

    class Converter:
        def _normalize_name(self, value):
            calls.append(value)
            return "normalized:" + value

    module = ModuleType("awex.converter.vllm_converter")
    module.VLLMToHFWeightConverter = Converter
    monkeypatch.setitem(sys.modules, module.__name__, module)

    _patch_vllm_expert_names()
    installed = Converter._normalize_name
    _patch_vllm_expert_names()

    assert Converter._normalize_name is installed
    assert Converter()._normalize_name(name) == "normalized:" + expected
    assert calls == [expected]
