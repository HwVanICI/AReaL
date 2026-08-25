# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

try:
    from areal.api.cli_args import FP8EngineConfig, MindSpeedEngineConfig
    from areal.engine import megatron_engine
except ImportError as exc:
    pytest.skip(
        f"MindSpeed test dependencies are unavailable: {exc}",
        allow_module_level=True,
    )


def _make_engine(fp8_config: FP8EngineConfig):
    engine = megatron_engine.MegatronEngine.__new__(megatron_engine.MegatronEngine)
    engine.enable_fp8 = True
    engine.fp8_config = fp8_config
    engine.mcore_config = SimpleNamespace(
        recompute_method="uniform",
        recompute_granularity="full",
        recompute_num_layers=1,
        ddp=SimpleNamespace(
            use_distributed_optimizer=True,
            fp8_param_gather=False,
        ),
    )
    engine.mindspeed_config = MindSpeedEngineConfig()
    engine.bridge_cls = "mbridge"
    return engine


def test_mindspeed_repatch_receives_fp8_compute_config(monkeypatch):
    engine = _make_engine(
        FP8EngineConfig(
            mode="e4m3",
            recipe="delayed",
            param=False,
        )
    )
    captured = {}

    import mindspeed.megatron_adaptor

    monkeypatch.setattr(megatron_engine, "is_npu_available", True)
    monkeypatch.setattr(
        mindspeed.megatron_adaptor,
        "repatch",
        lambda config: captured.update(config),
    )
    monkeypatch.setattr(
        megatron_engine, "_patch_mindspeed_mlp_init_tp_group", lambda: None
    )
    monkeypatch.setattr(
        megatron_engine,
        "_patch_mindspeed_npu_groupmatmul_add_fp32_dtype",
        lambda: None,
    )

    engine._patch_mindspeed(
        SimpleNamespace(
            context_parallel_size=1,
            expert_model_parallel_size=1,
            tensor_parallel_size=2,
        )
    )

    assert captured["fp8"] == "e4m3"
    assert captured["fp8_format"] == "e4m3"
    assert captured["fp8_recipe"] == "delayed"
    assert captured["transformer_impl"] == "transformer_engine"
