# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from areal.api.io_struct import ParamSpec, WeightUpdateMeta

ascend_mxfp8 = pytest.importorskip(
    "areal.engine.vllm_ext.ascend_mxfp8",
    reason="Ascend MXFP8 tests require torch_npu and vllm-ascend",
)

from areal.engine.vllm_ext import (
    areal_vllm_server,
    vllm_worker_extension,
)
from areal.engine.vllm_remote import VLLMBackend


def _run_with_generation_paused(coro):
    areal_vllm_server._generation_run_event.clear()
    try:
        return asyncio.run(coro)
    finally:
        areal_vllm_server._generation_run_event.set()


@pytest.fixture(autouse=True)
def _assume_weight_is_owned_by_local_pp_stage(monkeypatch):
    monkeypatch.setattr(
        ascend_mxfp8,
        "is_pp_missing_parameter",
        lambda _name, _model: False,
    )
    monkeypatch.setattr(
        ascend_mxfp8, "AscendW8A8MXFP8DynamicLinearMethod", _Scheme
    )
    monkeypatch.setattr(
        ascend_mxfp8, "AscendW8A8MXFP8DynamicFusedMoEMethod", _Scheme
    )


class _Scheme:
    def __init__(self):
        self.restore_count = 0

    def restore_weights_for_rl_loading(self, layer):
        self.restore_count += 1
        layer._mxfp8_transformed = False


class _Wrapper:
    def __init__(self):
        self.quant_method = _Scheme()
        self.process_count = 0

    def process_weights_after_loading(self, layer):
        self.process_count += 1
        layer._mxfp8_transformed = True


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(2, 32, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.empty(2, 1, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = _Wrapper()
        self._mxfp8_original_shapes = {
            "weight": self.weight.shape,
            "weight_scale": self.weight_scale.shape,
        }
        self._mxfp8_transformed = True


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _Linear()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.input_layernorm = torch.nn.LayerNorm(32)


class _Model(torch.nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer()])
        self.loaded = []

    def load_weights(self, weights):
        self.loaded.extend(weights)
        return {name for name, _ in self.loaded}


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.empty(2, 4, 32, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(2, 2, 32, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.empty(2, 4, 1, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.empty(2, 2, 1, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = _Wrapper()
        self._mxfp8_original_shapes = {"w13_weight": self.w13_weight.shape}
        self._mxfp8_transformed = True


class _MoEModel(torch.nn.Module):
    packed_modules_mapping = {}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = _MoE()
        self.model.layers = torch.nn.ModuleList([layer])
        self.loaded = []

    def load_weights(self, weights):
        self.loaded.extend(weights)
        return {name for name, _ in self.loaded}


def _mxfp8_capability_runner(monkeypatch, model):
    class _QuantConfig:
        quant_description = {
            "model.layers.0.self_attn.q_proj.weight": "W8A8_MXFP8",
            "model.layers.0.input_layernorm.weight": "FLOAT",
            "group_size": 32,
        }

    monkeypatch.setattr(ascend_mxfp8.current_platform, "device_type", "npu")
    monkeypatch.setattr(ascend_mxfp8, "AscendModelSlimConfig", _QuantConfig)
    return SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(quantization="ascend"),
            quant_config=_QuantConfig(),
        ),
    )


@pytest.mark.parametrize("float_moe", [False, True])
def test_mxfp8_capability_allows_float_non_moe_but_rejects_float_moe(
    monkeypatch, float_moe
):
    model = _MoEModel()
    monkeypatch.setattr(ascend_mxfp8, "FusedMoE", _MoE)
    runner = _mxfp8_capability_runner(monkeypatch, model)

    if float_moe:
        model.model.layers[0].mlp.experts.quant_method = SimpleNamespace(
            quant_method=object()
        )
        runner.vllm_config.quant_config.quant_description[
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ] = "FLOAT"
        with pytest.raises(RuntimeError):
            ascend_mxfp8.is_ascend_mxfp8(runner)
    else:
        assert ascend_mxfp8.is_ascend_mxfp8(runner)


def test_vllm_ascend_bucket_uses_atomic_request_metadata():
    """Ascend sends tensor and transaction metadata in one XCCL request."""
    meta = WeightUpdateMeta(
        type="xccl",
        nccl_group_name="weight-update",
        is_last_bucket=True,
        online_quantization="ascend",
        version=7,
        pp_rank=1,
        pp_world_size=2,
    )
    requests = VLLMBackend().build_distributed_weight_update_requests(
        meta,
        [ParamSpec(name="model.weight", shape=(2, 2), dtype="bfloat16")],
    ).requests

    assert len(requests) == 1
    request = requests[0]
    assert request.endpoint == "/areal_update_weights_xccl"
    assert request.payload == {
        "names": ["model.weight"],
        "dtypes": ["bfloat16"],
        "shapes": [(2, 2)],
        "group_name": "weight-update",
        "is_last_bucket": True,
        "online_quantization": "ascend",
        "version": 7,
        "pp_rank": 1,
        "pp_world_size": 2,
    }


def test_load_mxfp8_weights_maps_packed_linear_weight_and_scale(monkeypatch):
    model = _Model()
    linear = model.model.layers[0].self_attn.qkv_proj
    linear.weight = torch.nn.Parameter(
        torch.empty(6, 32, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    linear.weight_scale = torch.nn.Parameter(
        torch.empty(6, 1, dtype=torch.uint8), requires_grad=False
    )
    linear._mxfp8_original_shapes = {
        "weight": linear.weight.shape,
        "weight_scale": linear.weight_scale.shape,
    }
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            quant_config=SimpleNamespace(
                quant_description={"group_size": 32}
            ),
        ),
    )

    def _npu_dynamic_mx_quant(tensor, **_kwargs):
        return (
            tensor.to(torch.float8_e4m3fn),
            torch.ones(8, 1, 1, dtype=torch.uint8),
        )

    monkeypatch.setattr(
        ascend_mxfp8.torch_npu,
        "npu_dynamic_mx_quant",
        _npu_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        ascend_mxfp8.torch_npu, "float8_e4m3fn", torch.float8_e4m3fn
    )
    monkeypatch.setattr(ascend_mxfp8, "LinearBase", _Linear)
    monkeypatch.setattr(ascend_mxfp8, "FusedMoE", ())

    weight_name = "model.layers.0.self_attn.q_proj.weight"
    norm_name = "model.layers.0.input_layernorm.weight"
    norm_weight = torch.ones(32, dtype=torch.bfloat16)
    ascend_mxfp8.load_mxfp8_weights(
        [
            (weight_name, torch.ones(8, 32, dtype=torch.bfloat16)),
            (norm_name, norm_weight),
        ],
        runner,
        restore=True,
    )

    loaded = dict(model.loaded)
    assert linear._mxfp8_transformed is False
    assert loaded[weight_name].dtype == torch.float8_e4m3fn
    assert loaded[f"{weight_name}_scale"].shape == (8, 1)
    assert loaded[norm_name] is norm_weight


def test_load_mxfp8_weights_quantizes_moe_weights_and_scales(monkeypatch):
    model = _MoEModel()
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            quant_config=SimpleNamespace(
                quant_description={"group_size": 32}
            ),
        ),
    )

    def _npu_dynamic_mx_quant(tensor, *, axis, dst_type):
        scale_shape = (*tensor.shape[:-1], 1, 1)
        return tensor.to(dst_type), torch.ones(scale_shape, dtype=torch.uint8)

    monkeypatch.setattr(
        ascend_mxfp8.torch_npu,
        "npu_dynamic_mx_quant",
        _npu_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        ascend_mxfp8.torch_npu, "float8_e4m3fn", torch.float8_e4m3fn
    )
    monkeypatch.setattr(ascend_mxfp8, "LinearBase", _Linear)
    monkeypatch.setattr(ascend_mxfp8, "FusedMoE", _MoE)

    gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
    down = "model.layers.0.mlp.experts.0.down_proj.weight"
    ascend_mxfp8.load_mxfp8_weights(
        [
            (gate, torch.ones(3, 32, dtype=torch.bfloat16)),
            (down, torch.ones(5, 32, dtype=torch.bfloat16)),
        ],
        runner,
        restore=True,
    )

    loaded = dict(model.loaded)
    assert loaded[gate].dtype == torch.float8_e4m3fn
    assert loaded[f"{gate}_scale"].dtype == torch.uint8
    assert loaded[f"{gate}_scale"].shape == (3, 1)
    assert loaded[down].dtype == torch.float8_e4m3fn
    assert loaded[f"{down}_scale"].dtype == torch.uint8
    assert loaded[f"{down}_scale"].shape == (5, 1)


def test_load_mxfp8_weights_skips_remote_pp_weight(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        ascend_mxfp8,
        "is_pp_missing_parameter",
        lambda name, _model: name.startswith("model.layers.1."),
    )
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            quant_config=SimpleNamespace(
                quant_description={"group_size": 32}
            ),
        ),
    )

    ascend_mxfp8.load_mxfp8_weights(
        [
            (
                "model.layers.1.self_attn.q_proj.weight",
                torch.ones(2, 32, dtype=torch.bfloat16),
            )
        ],
        runner,
        restore=False,
    )

    assert model.loaded == []


def test_worker_restores_once_and_finalizes_once(monkeypatch):
    worker = vllm_worker_extension.VLLMWorkerExtension()
    worker.model_runner = SimpleNamespace(
        model=_Model(),
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16)
        ),
    )
    worker.weight_update_groups = {"pp0": object(), "pp1": object()}
    restore_flags = []
    process_count = 0

    monkeypatch.setattr(
        ascend_mxfp8, "is_ascend_mxfp8", lambda _runner: True
    )
    monkeypatch.setattr(
        vllm_worker_extension,
        "set_current_vllm_config",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        vllm_worker_extension.torch.distributed,
        "broadcast",
        lambda *_args, **_kwargs: None,
    )

    def _load(weights, _runner, *, restore):
        list(weights)
        restore_flags.append(restore)

    def _process(_model):
        nonlocal process_count
        process_count += 1

    monkeypatch.setattr(ascend_mxfp8, "load_mxfp8_weights", _load)
    monkeypatch.setattr(ascend_mxfp8, "process_mxfp8_weights_after_loading", _process)
    monkeypatch.setattr(
        vllm_worker_extension.current_platform, "synchronize", lambda: None
    )

    worker.areal_update_weight_xccl(
        [],
        [],
        [],
        "pp0",
        1,
        True,
        "ascend",
    )
    worker.areal_update_weight_xccl(
        [],
        [],
        [],
        "pp1",
        1,
        False,
        "ascend",
    )
    assert process_count == 0

    worker.areal_finalize_mxfp8_update(1)
    worker.areal_update_weight_xccl(
        [], [], [], "pp0", 2, True, "ascend"
    )

    assert restore_flags == [True, False, True]
    assert process_count == 1


def test_server_finalizes_mxfp8_only_after_all_pp_stages_complete():
    """MXFP8 finalization waits for the final bucket from every PP stage."""
    pp_world_size = 2
    buckets = ((1, False), (0, False), (1, True), (0, True))

    class _LLM:
        def __init__(self):
            self.calls = []

        async def collective_rpc(self, method, args=()):
            self.calls.append((method, args))
            return [(True, "Success")]

    llm = _LLM()
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=llm))
    )

    async def _run():
        for index, (pp_rank, is_last_bucket) in enumerate(buckets):
            request = areal_vllm_server.UpdateWeightBucketRequest(
                names=[],
                dtypes=[],
                shapes=[],
                group_name=f"pp{pp_rank}",
                online_quantization="ascend",
                version=3,
                pp_rank=pp_rank,
                pp_world_size=pp_world_size,
                is_last_bucket=is_last_bucket,
            )
            await areal_vllm_server.areal_update_weight_xccl(request, raw_request)
            if index < len(buckets) - 1:
                assert not any(
                    call[0] == "areal_finalize_mxfp8_update"
                    for call in llm.calls
                )

    _run_with_generation_paused(_run())

    finalize_calls = [
        call for call in llm.calls if call[0] == "areal_finalize_mxfp8_update"
    ]
    load_calls = [
        call for call in llm.calls if call[0] == "areal_update_weight_xccl"
    ]
    assert [call[1][5] for call in load_calls] == [True, False, False, False]
    assert finalize_calls == [("areal_finalize_mxfp8_update", (3,))]
