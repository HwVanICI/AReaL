# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

try:
    from areal.api.cli_args import FP8EngineConfig, MindSpeedEngineConfig
    from areal.api.io_struct import WeightUpdateMeta
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


@pytest.mark.parametrize(
    ("stage_kind", "expected_last_flags"),
    [("dense", [True]), ("moe", [False, True])],
)
def test_megatron_ascend_marks_dense_or_expert_final_bucket(
    monkeypatch, stage_kind, expected_last_flags
):
    captured = []

    class _Lock:
        def acquire(self):
            return None

        def release(self):
            return None

    class _Handle:
        def wait(self):
            return None

    class _Future:
        def result(self):
            return None

    class _Rollout:
        def update_weights_from_distributed(self, meta, _param_specs):
            captured.append(
                (
                    meta.is_last_bucket,
                    meta.version,
                    meta.pp_rank,
                    meta.pp_world_size,
                )
            )
            return _Future()

        def pause_generation(self):
            return None

        def continue_generation(self):
            return None

    engine = megatron_engine.MegatronEngine.__new__(megatron_engine.MegatronEngine)
    engine.weight_update_master_addr = "127.0.0.1"
    engine.weight_update_master_port = 12345
    engine.weight_update_group_name = "pp2"
    engine._cpu_group = object()
    engine.process_group_initialized = True
    engine.engine_lock = _Lock()
    engine.rollout_engine = _Rollout()
    engine.weight_update_group = object()
    engine.config = SimpleNamespace(use_lora=False, use_merged_lora=False)
    engine.bridge_cls = "mbridge"
    engine.mcore_config = SimpleNamespace(use_bridge_for_update_weights=False)
    engine.quantization_config = None
    engine.tf_config = SimpleNamespace(
        num_moe_experts=2 if stage_kind == "moe" else None
    )
    engine.model = object()
    engine.is_pipeline_parallel_head = lambda: True

    dense = ("model.weight", torch.ones(2, 2, dtype=torch.bfloat16))
    expert = (
        "model.layers.0.experts.weight",
        torch.ones(2, 2, dtype=torch.bfloat16),
    )
    parameters = [dense, expert] if stage_kind == "moe" else [dense]

    def _impl_update_weight(
        _meta, name, param, converted_named_tensors, _buffer_size, _chunk_size
    ):
        converted_named_tensors.append((name, param))
        return param.numel() * param.element_size()

    def _impl_update_expert(
        _meta, name, param, named_tensors, _buffer_size, _chunk_size
    ):
        named_tensors.append((name, param))
        return param.numel() * param.element_size()

    def _update_expert_bucket(meta, named_tensors, is_last_bucket=False):
        sent_payload = bool(named_tensors)
        engine._update_bucket_weights_from_distributed(
            meta, named_tensors, is_last_bucket=is_last_bucket
        )
        return sent_payload

    engine._impl_update_weight_from_distributed = _impl_update_weight
    engine._impl_update_expert_weight_from_distributed = _impl_update_expert
    engine._update_bucket_expert_weights_from_distributed = _update_expert_bucket
    monkeypatch.setattr(
        megatron_engine,
        "get_named_parameters",
        lambda _model, _num_experts: parameters,
    )

    monkeypatch.setattr(
        megatron_engine.dist,
        "broadcast",
        lambda *_args, **_kwargs: _Handle(),
    )
    monkeypatch.setattr(
        megatron_engine.dist, "get_rank", lambda: 1
    )
    monkeypatch.setattr(
        megatron_engine.dist, "barrier", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        megatron_engine.mpu, "get_pipeline_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_pipeline_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        megatron_engine.current_platform, "synchronize", lambda: None
    )

    meta = WeightUpdateMeta(
        type="xccl", online_quantization="ascend", version=11
    )
    engine._update_weights_from_distributed(meta)

    assert [bucket[0] for bucket in captured] == expected_last_flags
    assert all(bucket[1:4] == (11, 1, 2) for bucket in captured)
