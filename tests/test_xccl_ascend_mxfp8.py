# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Ascend wire protocol and initialization response path."""

import ast
import asyncio
import dataclasses
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest_plugins = ("tests.test_awex_ascend_mxfp8",)


def _method(path, owner, name, scope):
    """Execute production methods without importing optional training runtimes."""
    path = Path(__file__).parents[1] / path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = (
        tree.body
        if owner is None
        else next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    )
    node = next(node for node in body if getattr(node, "name", None) == name)
    node.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def _description(**weights):
    return {"group_size": 32, "weights": weights}


@pytest.mark.parametrize("enabled", [False, True])
def test_xccl_requests_use_boolean_ascend_flag(enabled):
    scope = {
        "HttpRequest": SimpleNamespace,
        "WeightUpdateRequests": SimpleNamespace,
        "current_platform": SimpleNamespace(communication_backend="hccl"),
        "format_host_for_url": lambda address: address,
    }
    init_request = _method(
        "areal/engine/vllm_remote.py",
        "VLLMBackend",
        "build_init_weights_group_request",
        scope,
    )
    update_requests = _method(
        "areal/engine/vllm_remote.py",
        "VLLMBackend",
        "build_distributed_weight_update_requests",
        scope,
    )
    meta = SimpleNamespace(
        ascend_online_quantization=enabled,
        gen_allocation=SimpleNamespace(
            parallel=SimpleNamespace(tp_size=2, pp_size=1, world_size=4)
        ),
        nccl_master_address="localhost",
        nccl_master_port=1,
        nccl_group_name="weights",
        backend=None,
        use_lora=False,
        version=7,
        is_last_bucket=True,
        pp_rank=0,
        pp_world_size=1,
    )
    init = init_request(None, "localhost", 0, meta)
    updates = update_requests(
        None,
        meta,
        [SimpleNamespace(name="linear.weight", shape=(3, 96), dtype="uint8")],
    ).requests
    if enabled:
        assert init.payload["ascend_online_quantization"] is True
        assert len(updates) == 1
        assert updates[0].payload["ascend_online_quantization"] is True
    else:
        assert "ascend_online_quantization" not in init.payload
        assert len(updates) == 2
        assert all("ascend_online_quantization" not in req.payload for req in updates)


def _linear(module, n=3, k=96):
    layer = module.LinearBase()
    layer.weight = torch.nn.Parameter(
        torch.empty(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty(n, k // 32, dtype=torch.uint8), requires_grad=False
    )
    layer.quant_method = SimpleNamespace(
        quant_method=module.AscendW8A8MXFP8DynamicLinearMethod()
    )
    layer._mxfp8_original_shapes = {
        "weight": layer.weight.shape,
        "weight_scale": layer.weight_scale.shape,
    }
    layer._mxfp8_transformed = True
    return layer


def test_initialized_modules_export_packed_and_expert_policies(
    mxfp8_module, monkeypatch
):
    """Actual modules, including global expert IDs, define the sender policy."""
    model = torch.nn.Module()
    model.qkv_proj = _linear(mxfp8_module)
    model.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    model.router = mxfp8_module.LinearBase()
    model.experts = mxfp8_module.FusedMoE()
    model.experts.quant_method = SimpleNamespace(
        quant_method=mxfp8_module.AscendW8A8MXFP8DynamicFusedMoEMethod()
    )
    for name, shape in {"w13_weight": (2, 6, 64), "w2_weight": (2, 3, 128)}.items():
        model.experts.register_parameter(
            name,
            torch.nn.Parameter(
                torch.empty(shape, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
        )
        model.experts.register_parameter(
            name + "_scale",
            torch.nn.Parameter(
                torch.empty(*shape[:-1], shape[-1] // 32, dtype=torch.uint8),
                requires_grad=False,
            ),
        )
    model.get_expert_mapping = lambda: [
        (
            f"experts.{'w2_' if projection == 'down_proj' else 'w13_'}",
            f"experts.{expert}.{projection}.",
            expert,
            "w2" if projection == "down_proj" else "w1",
        )
        for expert in range(4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    monkeypatch.setattr(mxfp8_module, "is_ascend_mxfp8", lambda _: True)
    description = mxfp8_module.get_ascend_mxfp8_description(
        SimpleNamespace(model=model)
    )
    expected = {
        f"{part}.weight": "W8A8_MXFP8" for part in ("q_proj", "k_proj", "v_proj")
    }
    expected["router.weight"] = "FLOAT"
    expected.update(
        {
            f"experts.{expert}.{part}.weight": "W8A8_MXFP8"
            for expert in range(4)
            for part in ("gate_proj", "up_proj", "down_proj")
        }
    )
    assert description == _description(**expected)


def test_initialized_local_k_rejects_partial_groups(mxfp8_module, monkeypatch):
    model = torch.nn.Module()
    model.linear = _linear(mxfp8_module, k=48)
    monkeypatch.setattr(mxfp8_module, "is_ascend_mxfp8", lambda _: True)
    with pytest.raises(ValueError, match="local K=48"):
        mxfp8_module.get_ascend_mxfp8_description(SimpleNamespace(model=model))


def test_description_union_checks_replica_completeness(mxfp8_module):
    from areal.utils.mxfp8 import merge_ascend_mxfp8_descriptions

    first = _description(**{"stage0.weight": "W8A8_MXFP8"})
    second = _description(**{"stage1.weight": "W8A8_MXFP8"})
    complete = merge_ascend_mxfp8_descriptions([first, second])
    assert len(complete["weights"]) == 2
    assert (
        merge_ascend_mxfp8_descriptions([complete, complete], replicas=True) == complete
    )
    with pytest.raises(ValueError, match="replicas"):
        merge_ascend_mxfp8_descriptions([complete, first], replicas=True)
    with pytest.raises(ValueError, match="Conflicting"):
        merge_ascend_mxfp8_descriptions(
            [first, _description(**{"stage0.weight": "FLOAT"})]
        )


@pytest.mark.parametrize("k", [32, 64, 96])
def test_wire_decode_preserves_fp8_bits_and_float_weights(mxfp8_module, k):
    model = torch.nn.Module()
    model.linear = _linear(mxfp8_module, k=k)
    model.norm = torch.nn.LayerNorm(k)
    fp8 = torch.arange(3 * k).reshape(3, k).remainder(16).to(torch.float8_e4m3fn)
    scale = torch.arange(3 * (k // 32), dtype=torch.uint8).reshape(3, -1)
    wire = fp8.view(torch.uint8)
    norm = torch.ones(k, dtype=torch.bfloat16)
    loaded = dict(
        mxfp8_module._decode_mxfp8_weights(
            [
                ("linear.weight", wire),
                ("linear.weight_scale", scale),
                ("norm.weight", norm),
            ],
            model,
        )
    )
    assert loaded["linear.weight"].data_ptr() == fp8.data_ptr()
    assert loaded["linear.weight_scale"] is scale
    assert loaded["norm.weight"] is norm
    torch.testing.assert_close(
        loaded["linear.weight"].view(torch.uint8), wire, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "invalid", ["missing_scale", "scale_shape", "scale_dtype", "bf16_weight"]
)
def test_wire_decode_rejects_invalid_pairs(mxfp8_module, invalid):
    model = torch.nn.Module()
    model.linear = _linear(mxfp8_module)
    weight = torch.zeros(3, 96, dtype=torch.uint8)
    scale = torch.zeros(3, 3, dtype=torch.uint8)
    if invalid == "bf16_weight":
        weight = weight.to(torch.bfloat16)
    elif invalid == "scale_shape":
        scale = scale[:, :2]
    elif invalid == "scale_dtype":
        scale = scale.float()
    weights = [("linear.weight", weight)]
    if invalid != "missing_scale":
        weights.append(("linear.weight_scale", scale))
    with pytest.raises(ValueError):
        list(mxfp8_module._decode_mxfp8_weights(weights, model))


def test_init_response_returns_model_description_to_actor(mxfp8_module, monkeypatch):
    """Run worker→HTTP→remote engine→controller→callback return handling."""
    description = _description(**{"linear.weight": "W8A8_MXFP8"})
    calls = []
    worker_init = _method(
        "areal/engine/vllm_ext/vllm_worker_extension.py",
        "VLLMWorkerExtension",
        "areal_init_update_weight_group",
        {
            "dist": SimpleNamespace(),
            "init_custom_process_group": lambda **_: object(),
            "get_world_group": lambda: SimpleNamespace(rank=0),
            "DIST_GROUP_DEFAULT_TIMEOUT": 5,
            "logger": SimpleNamespace(error=lambda *_: None),
        },
    )
    monkeypatch.setitem(sys.modules, "areal.engine.vllm_ext.ascend_mxfp8", mxfp8_module)
    monkeypatch.setattr(
        mxfp8_module,
        "get_ascend_mxfp8_description",
        lambda runner: calls.append(runner) or description,
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace())

    class Client:
        async def collective_rpc(self, method, args):
            assert method == "areal_init_update_weight_group"
            assert args[-1] is request.ascend_online_quantization
            return [worker_init(worker, *args)]

    endpoint = _method(
        "areal/engine/vllm_ext/areal_vllm_server.py",
        None,
        "areal_init_weights_update_group",
        {
            "logger": SimpleNamespace(info=lambda *_: None),
            "JSONResponse": lambda body: body,
            "build_response": lambda results: {
                "success": all(result[0] for result in results)
            },
            "to_json_response": lambda success, message: {
                "success": success,
                "message": message,
            },
        },
    )
    request = SimpleNamespace(
        master_address="localhost",
        master_port="1",
        rank_offset=1,
        world_size=2,
        backend="gloo",
        group_name="weights",
        ascend_online_quantization=True,
    )
    raw = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=Client()))
    )
    response = asyncio.run(endpoint(request, raw))
    assert response["ascend_mxfp8"] == description
    request.group_name = "weights_pp1"
    asyncio.run(endpoint(request, raw))
    assert (
        len(calls) == 1
    )  # One export per initialized worker, reused across actor PP groups.

    class Session:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    async def http_request(**_):
        return await endpoint(request, raw)

    remote_init = _method(
        "areal/infra/remote_inf_engine.py",
        None,
        "_init_weights_update_group_remote",
        {
            "asyncio": asyncio,
            "uvloop": SimpleNamespace(run=asyncio.run),
            "aiohttp": SimpleNamespace(
                ClientSession=Session, ClientTimeout=SimpleNamespace
            ),
            "get_default_connector": lambda: None,
            "arequest_with_retry": http_request,
        },
    )
    backend = SimpleNamespace(
        build_init_weights_group_request=lambda *args: SimpleNamespace(
            endpoint="/areal_init_weights_update_group", payload={}, method="POST"
        )
    )
    meta = SimpleNamespace(ascend_online_quantization=True)
    remote_description = remote_init(backend, meta, ["replica0", "replica1"], 5)
    assert remote_description == description

    controller_init = _method(
        "areal/infra/controller/rollout_controller.py",
        "RolloutController",
        "init_weights_update_group",
        {"asyncio": asyncio},
    )

    async def schedule_init(**_):
        return remote_description

    controller = SimpleNamespace(
        workers=[SimpleNamespace(id="replica0"), SimpleNamespace(id="replica1")],
        scheduler=SimpleNamespace(async_call_engine=schedule_init),
        _engine_name=lambda rank: str(rank),
    )
    result = asyncio.run(controller_init(controller, meta))
    with ThreadPoolExecutor(max_workers=1) as executor:
        callback_init = _method(
            "areal/infra/controller/rollout_callback.py",
            "RolloutCallback",
            "init_weights_update_group",
            {
                "get_executor": lambda: executor,
                "serialize_value": lambda meta: meta,
            },
        )
        callback = SimpleNamespace(
            _post=lambda *args: {"status": "ok", "ascend_mxfp8": result}
        )
        assert callback_init(callback, meta).result(timeout=5) == description

        error = RuntimeError("description request failed")

        def failed_post(*args):
            raise error

        callback._post = failed_post
        with pytest.raises(RuntimeError, match="description request failed") as caught:
            callback_init(callback, meta).result(timeout=5)
        assert caught.value is error

    # Ordinary BF16 initialization preserves the existing None result.
    meta.ascend_online_quantization = False
    request.ascend_online_quantization = False
    assert remote_init(backend, meta, ["replica0"], 5) is None
    assert asyncio.run(controller_init(controller, meta)) is None


def test_megatron_prepares_pairs_and_propagates_quantization_error(
    mxfp8_module, monkeypatch
):
    mxfp8 = sys.modules["areal.utils.mxfp8"]

    monkeypatch.setattr(
        mxfp8,
        "quantize_mxfp8_tensor",
        lambda tensor, name: (
            tensor.to(torch.float8_e4m3fn),
            torch.full((3, 2, 2), 127, dtype=torch.uint8),
        ),
    )
    prepare = _method(
        "areal/engine/megatron_engine.py",
        "MegatronEngine",
        "_prepare_ascend_mxfp8_weights",
        {"iter_ascend_mxfp8_payloads": mxfp8.iter_ascend_mxfp8_payloads},
    )
    engine = SimpleNamespace(
        _ascend_mxfp8_description=_description(**{"linear.weight": "W8A8_MXFP8"}),
    )
    meta = SimpleNamespace(ascend_online_quantization=True)
    wire = prepare(
        engine, meta, [("linear.weight", torch.ones(3, 96, dtype=torch.bfloat16))]
    )
    assert [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in wire] == [
        ("linear.weight", torch.uint8, (3, 96)),
        ("linear.weight_scale", torch.uint8, (3, 3)),
    ]
    with pytest.raises(ValueError, match="K=48"):
        prepare(
            engine, meta, [("linear.weight", torch.ones(3, 48, dtype=torch.bfloat16))]
        )


@pytest.mark.parametrize("failure", [None, "quant", "receiver"])
def test_fsdp_preserves_pending_pipeline_and_propagates_errors(
    mxfp8_module, monkeypatch, failure
):
    """Execute the real FSDP bucket loop with CPU tensors and transport handles."""
    mxfp8 = sys.modules["areal.utils.mxfp8"]
    gathered = []
    updates = []
    events = []

    @dataclasses.dataclass
    class Meta:
        ascend_online_quantization: bool = True
        weight_chunked_mem_mb: int = 0
        is_last_bucket: bool = False

    def quantize(tensor, name):
        if failure == "quant" and name == "second.weight":
            raise ValueError("quant failed")
        return tensor.to(torch.float8_e4m3fn), torch.ones(3, 2, 2, dtype=torch.uint8)

    monkeypatch.setattr(mxfp8, "quantize_mxfp8_tensor", quantize)

    class ResultFuture(Future):
        def result(self):
            events.append("result")
            return super().result()

    class Rollout:
        def pause_generation(self):
            events.append("pause")

        def continue_generation(self):
            events.append("continue")

        def update_weights_from_distributed(self, meta, specs):
            updates.append((meta.is_last_bucket, specs))
            future = ResultFuture()
            if failure == "receiver":
                future.set_exception(ValueError("receiver rejected pair"))
            else:
                future.set_result(None)
            return future

    scope = {
        "dataclasses": dataclasses,
        "dist": SimpleNamespace(
            get_rank=lambda: 0,
            barrier=lambda **_: None,
            broadcast=lambda *_, **__: SimpleNamespace(
                wait=lambda: events.append("wait")
            ),
        ),
        "current_platform": SimpleNamespace(
            device_type="npu", synchronize=lambda: None
        ),
        "torch": torch,
        "nullcontext": nullcontext,
        "ParamSpec": SimpleNamespace,
        "_PendingWeightUpdateBucket": SimpleNamespace,
        "iter_ascend_mxfp8_payloads": mxfp8.iter_ascend_mxfp8_payloads,
    }
    methods = {
        name: _method("areal/engine/fsdp_engine.py", "FSDPEngine", name, scope)
        for name in (
            "_update_weights_from_distributed",
            "_update_bucket_weights_from_distributed_async",
            "_wait_pending_weight_update_bucket",
            "_update_bucket_weights_from_distributed",
        )
    }
    engine_type = type("Engine", (), methods)
    engine = engine_type()
    engine.rollout_engine = Rollout()
    engine.weight_update_master_addr = "localhost"
    engine.weight_update_master_port = 1
    engine.weight_update_group_names = ["weights"]
    engine.weight_update_groups = [object()]
    engine.cpu_group = object()
    engine.config = SimpleNamespace(use_lora=False)
    parameters = [
        (name + ".weight", torch.ones(3, 96, dtype=torch.bfloat16))
        for name in ("first", "second", "third")
    ]
    engine._get_model_name_parameters = lambda meta: parameters
    engine._get_full_tensor = lambda tensor: gathered.append(tensor) or tensor
    engine._cast_to_compute_dtype = lambda tensor: tensor
    engine._ascend_mxfp8_description = _description(
        **{
            name: "FLOAT" if name == "third.weight" else "W8A8_MXFP8"
            for name, _ in parameters
        }
    )
    if failure is None:
        engine._update_weights_from_distributed(Meta())
        assert [last for last, _ in updates] == [False, False, True]
        assert events[-1] == "continue"
        for _, specs in updates[:2]:
            assert [spec.dtype for spec in specs] == ["uint8", "uint8"]
            assert [spec.shape for spec in specs] == [(3, 96), (3, 3)]
            assert specs[1].name == specs[0].name + "_scale"
        assert [(spec.name, spec.dtype, spec.shape) for spec in updates[2][1]] == [
            ("third.weight", "bfloat16", (3, 96))
        ]
    else:
        message = "receiver rejected" if failure == "receiver" else "quant failed"
        with pytest.raises(ValueError, match=message):
            engine._update_weights_from_distributed(Meta())
        assert "continue" not in events
        assert not any(last for last, _ in updates)
    assert len(gathered) == (3 if failure in {None, "receiver"} else 2)
