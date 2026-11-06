# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle tests; native Ascend checks skip individually without NPU deps."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def awex_metadata():
    """Use AWEX's actual metadata builder and sharding strategy on CPU."""
    resolver = pytest.importorskip("awex.meta.meta_resolver")
    sharding = pytest.importorskip("awex.sharding.param_sharding")

    def build(raw, tp):
        class Resolver(resolver.ParamMetaResolver):
            def get_model_arch_name(self):
                return "TestModel"

            def get_parameters_meta(self):
                return self._build_params_meta()

            def _get_params_raw_meta(self):
                return raw

            def _get_sharding_info(self, name, rank_info, param_meta):
                strategy = sharding.ShardingStrategy(
                    "mcore", False, False, tp, tp, 1, 1, rank_info
                )
                return strategy.get_sharding_strategy(
                    name, rank_info=rank_info, param_meta=param_meta
                )

        return Resolver(SimpleNamespace(num_hidden_layers=1)).get_parameters_meta()

    return build


def _raw_metadata(name, tp, *, n=7, k=384, dim=1, dtype=torch.bfloat16):
    shape = (n // tp, k) if dim == 0 else (n, k // tp)
    return [
        {
            "rank_info": SimpleNamespace(
                tp_rank=rank,
                tp_size=tp,
                attn_tp_rank=rank,
                attn_tp_size=tp,
                pp_rank=0,
                pp_size=1,
                ep_rank=0,
                ep_size=1,
                ep_tp_rank=0,
                ep_tp_size=1,
                global_rank=rank,
                engine_rank=0,
                world_size=tp,
                is_infer=False,
            ),
            "params_meta": [
                {
                    "name": name,
                    "shape": shape,
                    "numel": shape[0] * shape[1],
                    "dtype": dtype,
                    "vp_stage": 0,
                }
            ],
        }
        for rank in range(tp)
    ]


def _project_raw(module, monkeypatch, raw, name):
    monkeypatch.setattr(
        module, "_original_collect_model_param_raw_info", lambda *args, **kwargs: raw
    )
    resolver = SimpleNamespace(
        _train_engine=SimpleNamespace(
            weights_exchange_writer=SimpleNamespace(_ascend_mxfp8_names={name})
        )
    )
    return module._ascend_mxfp8_collect_model_param_raw_info(resolver, True)


@pytest.mark.parametrize("tp,dim", [(1, 1), (2, 1), (4, 1), (2, 0)])
def test_raw_metadata_projects_weight_scale_without_quantization(
    receiver_module, monkeypatch, awex_metadata, tp, dim
):
    """Original BF16 metadata becomes byte/scale records before AWEX builds shards."""
    name = (
        "model.layers.0.attention.dense.weight"
        if dim == 1
        else "model.layers.0.mlp.gate_proj.weight"
    )
    raw = _raw_metadata(name, tp, n=6 if dim == 0 else 7, dim=dim)
    monkeypatch.setattr(
        receiver_module,
        "quantize_mxfp8_tensor",
        lambda *args: pytest.fail("metadata must not quantize"),
    )
    projected = _project_raw(receiver_module, monkeypatch, raw, name)
    parameters = awex_metadata(projected, tp)
    receiver_module._validate_wire_metadata(parameters, {name})
    assert parameters[0].dtype == torch.uint8
    assert parameters[1].global_shape == (6 if dim == 0 else 7, 12)


@pytest.mark.parametrize("k", [6144, 6160])
def test_unlisted_projection_uses_actual_metadata(
    receiver_module, monkeypatch, awex_metadata, k
):
    """New projection names are accepted or rejected by actual K alignment."""
    name = "model.layers.0.linear_attn.out_proj.weight"
    raw = _raw_metadata(name, 2, n=6, k=k, dim=0)
    if k % 32:
        with pytest.raises(ValueError, match="32-element K groups"):
            _project_raw(receiver_module, monkeypatch, raw, name)
    else:
        metadata = awex_metadata(
            _project_raw(receiver_module, monkeypatch, raw, name), 2
        )
        receiver_module._validate_wire_metadata(metadata, {name})
        assert metadata[0].global_shape == (6, k)
        assert metadata[1].global_shape == (6, k // 32)


def test_raw_debug_metadata_remains_bf16(receiver_module, monkeypatch):
    """The optional raw debug path bypasses wire metadata projection."""
    name = "model.layers.0.attention.dense.weight"
    raw = _raw_metadata(name, 1)
    monkeypatch.setattr(
        receiver_module, "_original_collect_model_param_raw_info", lambda *args: raw
    )
    resolver = SimpleNamespace(
        _train_engine=SimpleNamespace(
            weights_exchange_writer=SimpleNamespace(_ascend_mxfp8_names={name})
        )
    )
    assert (
        receiver_module._ascend_mxfp8_collect_model_param_raw_info(resolver, False)
        is raw
    )
    assert raw[0]["params_meta"][0]["dtype"] == torch.bfloat16


@pytest.mark.parametrize("stage", ["metadata", "quantization"])
@pytest.mark.parametrize(
    "shape,dtype", [((7, 32), torch.float32), ((2, 7, 32), torch.bfloat16)]
)
def test_actor_matrix_contract_checked_before_metadata_and_quantization(
    receiver_module, monkeypatch, stage, shape, dtype
):
    """Both initial metadata and real update tensors must satisfy the contract."""
    name = "model.layers.0.attention.dense.weight"
    monkeypatch.setattr(
        receiver_module,
        "quantize_mxfp8_tensor",
        lambda *args: pytest.fail("must reject before calling the native quantizer"),
    )
    with pytest.raises(ValueError, match="BF16.*matrix"):
        if stage == "metadata":
            raw = _raw_metadata(name, 1)
            raw[0]["params_meta"][0].update(shape=shape, dtype=dtype)
            _project_raw(receiver_module, monkeypatch, raw, name)
        else:
            receiver_module._quantize_for_transfer(
                name, torch.empty(shape, dtype=dtype)
            )


@pytest.mark.parametrize(
    "corruption",
    ["offset", "weight_offset", "weight_shape", "shape", "dtype", "missing"],
)
def test_metadata_rejects_incompatible_scale_coordinates(
    receiver_module, monkeypatch, awex_metadata, corruption
):
    """Shape-compatible weights alone cannot hide a misplaced scale shard."""
    name = "model.layers.0.attention.dense.weight"
    raw = _project_raw(receiver_module, monkeypatch, _raw_metadata(name, 2), name)
    parameters = awex_metadata(raw, 2)
    scale = parameters[1]
    if corruption == "offset":
        scale.shards[1].global_offset = (0, 0)
    elif corruption == "weight_offset":
        parameters[0].shards[1].global_offset = (0, 193)
    elif corruption == "weight_shape":
        # Global K remains aligned; one local K violates the quantization groups.
        parameters[0].shards[0].shape = (7, 176)
    elif corruption == "shape":
        scale.shards[0].shape = (7, 5)
    elif corruption == "dtype":
        scale.dtype = torch.float32
    else:
        parameters.pop()
    with pytest.raises(ValueError, match="AWEX.*MXFP8|Missing AWEX"):
        receiver_module._validate_wire_metadata(parameters, {name})


@pytest.mark.parametrize("source_tp,target_tp", [(2, 4), (4, 2)])
def test_real_awex_plan_reshards_fp8_bytes_and_scales(
    receiver_module, monkeypatch, awex_metadata, source_tp, target_tp
):
    """Real AWEX plans preserve row/group patterns through K resharding."""
    transfer = pytest.importorskip("awex.transfer.transfer_plan")
    name = "model.layers.0.attention.dense.weight"
    source = awex_metadata(
        _project_raw(
            receiver_module, monkeypatch, _raw_metadata(name, source_tp), name
        ),
        source_tp,
    )
    target = awex_metadata(
        _project_raw(
            receiver_module, monkeypatch, _raw_metadata(name, target_tp), name
        ),
        target_tp,
    )
    full = {
        name: (torch.arange(7 * 384).reshape(7, 384) % 251).to(torch.uint8),
        name + "_scale_inv": (torch.arange(7 * 12).reshape(7, 12) + 100).to(
            torch.uint8
        ),
    }

    def slice_for(shard):
        return tuple(
            slice(offset, offset + size)
            for offset, size in zip(shard.global_offset, shard.shape)
        )

    sources = {
        (shard.global_rank, p.name): full[p.name][slice_for(shard)].clone()
        for p in source
        for shard in p.shards
    }
    for rank in range(target_tp):
        plan = transfer.TransferPlanBuilder(
            target_tp, source_tp, 1, False
        ).build_local_transfer_plan(target, source, rank)
        targets = {
            p.name: torch.zeros_like(full[p.name][slice_for(p.shards[rank])])
            for p in target
        }
        for operations in plan.operations.values():
            for operation in operations:
                key = operation.send_shard_meta.name
                targets[key][operation.inf_slices].copy_(
                    sources[(operation.send_shard_meta.global_rank, key)][
                        operation.train_slices
                    ]
                )
        for p in target:
            torch.testing.assert_close(
                targets[p.name], full[p.name][slice_for(p.shards[rank])], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "required", [None, [], ["weight"], ["weight_scale_inv"], ["norm"]]
)
def test_writer_filters_wire_requests_after_original_converter(
    receiver_module, monkeypatch, required
):
    """Even empty and scale-only requests retain the original converter call."""
    source = torch.ones(7, 96, dtype=torch.bfloat16)
    norm = torch.ones(7, dtype=torch.float32)
    calls = []

    def convert(self, names):
        calls.append(names)
        return (
            {"weight": source, "norm": norm}
            if not names
            else {
                key: tensor
                for key, tensor in {"weight": source, "norm": norm}.items()
                if key in names
            }
        )

    monkeypatch.setattr(receiver_module, "_original_convert_parameters", convert)
    monkeypatch.setattr(
        receiver_module,
        "quantize_mxfp8_tensor",
        lambda tensor, name: (
            tensor.to(torch.float8_e4m3fn),
            torch.full((7, 2, 2), 127, dtype=torch.uint8),
        ),
    )
    writer = SimpleNamespace(
        _ascend_mxfp8_names={"weight"},
        current_worker_parameters_map={
            "weight": SimpleNamespace(shards=[SimpleNamespace(shape=(7, 96))])
        },
    )
    result = receiver_module._ascend_mxfp8_convert_parameters(writer, required)
    assert len(calls) == 1
    assert calls[0] == (
        None
        if required is None
        else {key.removesuffix("_scale_inv") for key in required}
    )
    assert set(result) == (
        {"weight", "weight_scale_inv", "norm"} if required is None else set(required)
    )
    if "norm" in result:
        assert result["norm"] is norm
    if "weight" in result:
        torch.testing.assert_close(
            result["weight"].view(torch.float8_e4m3fn).float(),
            source.float(),
            rtol=0,
            atol=0,
        )


def test_writer_rejects_changed_local_shape(receiver_module, monkeypatch):
    """Published local metadata is checked again before each quantization."""
    writer = SimpleNamespace(
        _ascend_mxfp8_names={"weight"},
        current_worker_parameters_map={
            "weight": SimpleNamespace(shards=[SimpleNamespace(shape=(7, 64))])
        },
    )
    monkeypatch.setattr(
        receiver_module,
        "_original_convert_parameters",
        lambda *args: {"weight": torch.ones(7, 96, dtype=torch.bfloat16)},
    )
    with pytest.raises(ValueError, match="published local metadata"):
        receiver_module._ascend_mxfp8_convert_parameters(writer)


@pytest.mark.parametrize("mismatch", [False, True])
def test_writer_discovers_quantization_before_collecting_training_metadata(
    receiver_module, monkeypatch, awex_metadata, mismatch
):
    """Inference metadata is available before the original writer initializes."""
    common = pytest.importorskip("awex.util.common")
    name = "model.layers.0.attention.dense.weight"
    metadata = awex_metadata(
        _project_raw(receiver_module, monkeypatch, _raw_metadata(name, 2), name), 2
    )
    events = []

    def get_binary(key, timeout):
        events.append(key)
        return common.to_binary(metadata)

    def initialize(self):
        assert self._ascend_mxfp8_names == {name}
        events.append("original_initialize")
        self.parameters_meta = (
            awex_metadata(
                _project_raw(
                    receiver_module, monkeypatch, _raw_metadata(name, 2, n=8), name
                ),
                2,
            )
            if mismatch
            else metadata
        )
        return "initialized"

    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(receiver_module, "_original_writer_initialize", initialize)
    writer = SimpleNamespace(
        train_engine=SimpleNamespace(enable_ascend_mxfp8=True, config={}),
        comm_backend="hccl",
        timeout=10,
        meta_server_client=SimpleNamespace(get_binary=get_binary),
    )
    if mismatch:
        with pytest.raises(ValueError, match="Inconsistent"):
            receiver_module._ascend_mxfp8_writer_initialize(writer)
    else:
        assert receiver_module._ascend_mxfp8_writer_initialize(writer) == "initialized"
    assert events == ["infer_params_meta", "original_initialize"]


@pytest.mark.parametrize(
    "device,flags",
    [
        ("cuda", {"enable_ascend_mxfp8": True}),
        ("npu", {"enable_ascend_mxfp8": False}),
        ("npu", {"online_quantization": "ascend"}),
    ],
)
def test_writer_requires_ascend_flag_and_npu(
    receiver_module, monkeypatch, device, flags
):
    """GPU and generic quantization flags cannot activate Ascend wire conversion."""
    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type=device)
    )
    monkeypatch.setattr(
        receiver_module, "_original_writer_initialize", lambda writer: "original"
    )
    writer = SimpleNamespace(train_engine=SimpleNamespace(**flags))
    assert receiver_module._ascend_mxfp8_writer_initialize(writer) == "original"
    assert not hasattr(writer, "_ascend_mxfp8_names")


@pytest.mark.parametrize("post_failure", [False, True])
def test_actual_colocate_patch_post_load_precedes_ack(
    receiver_module, support, monkeypatch, post_failure
):
    """Execute the actual local patch; failed post-load must never send an ACK."""
    tree = ast.parse(
        (Path(__file__).parents[1] / "areal/engine/patch_awex.py").read_text()
    )
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_update_weights_in_colocate_mode"
    )
    events = []
    monkeypatch.setattr(
        receiver_module,
        "current_platform",
        SimpleNamespace(device_type="npu", synchronize=lambda: events.append("sync")),
    )
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda **kwargs: events.append("barrier")
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(empty_cache=lambda: None), raising=False
    )

    def post(model):
        events.append("post")
        if post_failure:
            raise RuntimeError("post-load failed")

    monkeypatch.setattr(support, "process_mxfp8_weights_after_loading", post)
    reader = _reader(receiver_module, _make_runner(), _Converter())
    receiver = receiver_module._get_receiver("vllm", reader.model_context)
    reader.parameters = receiver.begin_receive(reader.weight_converter)
    reader.__dict__.update(
        enable_colocate_mode=True,
        collect_training_weights=lambda *args, **kwargs: None,
        transfer_plan=SimpleNamespace(operations={}),
        send_ranks_sample=[],
        rank_coordinate="0",
        colocate_transport=SimpleNamespace(
            update_weights_in_colocate_mode=lambda *args, **kwargs: events.append(
                "transport"
            )
        ),
        train_to_infer_device_mapping={0: 0},
        infer_to_train_device_mapping={0: 0},
        transfer_rank=0,
        infer_world_size=1,
        send_transfer_plan=None,
        weights_update_group=None,
        deserialized_weights={},
        _history_update_weights_time={},
        meta_server_client=SimpleNamespace(
            put_object=lambda *args: events.append("ack"),
            get_object_then_delete=lambda *args: None,
        ),
    )
    scope = {
        "logger": SimpleNamespace(info=lambda *args: None),
        "time": SimpleNamespace(time=lambda: 0),
        "gc": SimpleNamespace(collect=lambda: None),
        "print_current_gpu_status": lambda *args: None,
        "compute_statistics": lambda *args: None,
        "get_ip_address": lambda: "localhost",
        "device_util": SimpleNamespace(
            current_device=lambda: 0, get_device_type=lambda: "npu"
        ),
        "torch": torch,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "patch_awex.py", "exec"),
        scope,
    )
    if post_failure:
        with pytest.raises(RuntimeError, match="post-load failed"):
            scope[method.name](reader, 1)
        assert events == ["transport", "post"]
    else:
        scope[method.name](reader, 1)
        assert events == ["transport", "post", "sync", "ack", "barrier"]


@pytest.fixture
def mxfp8_module(monkeypatch):
    """Load the shared implementation with isolated optional dependency imports."""
    path = Path(__file__).parents[1] / "areal/engine/vllm_ext/ascend_mxfp8.py"
    platform = types.ModuleType("areal.infra.platforms")
    platform.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    quant_spec = importlib.util.spec_from_file_location(
        "areal.utils.mxfp8", path.parents[2] / "utils/mxfp8.py"
    )
    quant_module = importlib.util.module_from_spec(quant_spec)
    monkeypatch.setitem(sys.modules, quant_spec.name, quant_module)
    quant_spec.loader.exec_module(quant_module)
    optional_imports = {
        "torch_npu": {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "npu_dynamic_mx_quant": None,
        },
        "vllm.model_executor.layers.fused_moe": {
            "FusedMoE": type("FusedMoE", (torch.nn.Module,), {})
        },
        "vllm.model_executor.layers.linear": {
            "LinearBase": type("LinearBase", (torch.nn.Module,), {})
        },
        "vllm.model_executor.models.utils": {
            "is_pp_missing_parameter": lambda name, model: False
        },
        "vllm_ascend.quantization.modelslim_config": {
            "AscendModelSlimConfig": type("AscendModelSlimConfig", (), {})
        },
        "vllm_ascend.quantization.methods.w8a8_mxfp8": {
            "AscendW8A8MXFP8DynamicLinearMethod": _LinearScheme,
            "AscendW8A8MXFP8DynamicFusedMoEMethod": _MoEScheme,
        },
    }
    spec = importlib.util.spec_from_file_location("ascend_mxfp8_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    # Remove the dependency stubs immediately after import so NPU tests can
    # independently import the actual runtime (or skip when it is absent).
    with monkeypatch.context() as dependencies:
        for name, attributes in optional_imports.items():
            dependency = types.ModuleType(name)
            dependency.__dict__.update(attributes)
            dependencies.setitem(sys.modules, name, dependency)
        spec.loader.exec_module(module)
    module.torch_npu = SimpleNamespace(**optional_imports["torch_npu"])
    return module


@pytest.fixture
def receiver_module(monkeypatch, mxfp8_module):
    # Isolate AReaL's training-engine imports while using the real shared code.
    directory = Path(__file__).parents[1] / "areal/engine/vllm_ext"
    package = types.ModuleType("areal.engine.vllm_ext")
    package.__path__ = [str(directory)]
    package.ascend_mxfp8 = mxfp8_module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    path = directory / "awex_ascend_mxfp8.py"
    spec = importlib.util.spec_from_file_location("awex_mxfp8_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setitem(sys.modules, "areal.engine.vllm_ext.awex_ascend_mxfp8", module)
    spec.loader.exec_module(module)
    return module


class _LinearScheme:
    pass


class _MoEScheme:
    pass


def _parameter(shape, dtype):
    return torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


def test_normal_awex_import_does_not_load_ascend_dependencies(
    receiver_module, monkeypatch
):
    """Importing the hooks/config guard does not require the Ascend runtime."""
    original_import = builtins.__import__

    def import_without_ascend(name, *args, **kwargs):
        if name == "areal.engine.vllm_ext" or name.startswith(("torch_npu", "vllm")):
            raise AssertionError(f"Unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "awex_import_test", receiver_module.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", import_without_ascend)
        spec.loader.exec_module(module)
        module.validate_awex_ascend_mxfp8_config({"comm_backend": "hccl"})
        assert module._get_receiver("vllm", {}) is None


@pytest.mark.parametrize(
    "backend,mode,comm,error",
    [
        ("fsdp", "awex", "hccl", "AWEX currently requires a Megatron actor"),
        ("megatron", "awex", "hccl", None),
        ("megatron", "awex", "astate", "comm_backend=astate"),
        ("fsdp", "xccl", "hccl", None),
        ("megatron", "xccl", "hccl", None),
    ],
)
def test_trainer_rejects_unimplemented_awex_backend_before_initialization(
    receiver_module, backend, mode, comm, error
):
    """Run the actual trainer guard without importing distributed training engines."""
    path = Path(__file__).parents[1] / "areal/trainer/rl_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer"
    )
    validate = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_cfg"
    )
    scope = {"current_platform": SimpleNamespace(device_type="npu")}
    exec(
        compile(ast.Module(body=[validate], type_ignores=[]), str(path), "exec"), scope
    )
    actor = SimpleNamespace(
        _version="v1",
        weight_update_mode=mode,
        use_lora=False,
        dtype="bfloat16",
        megatron=SimpleNamespace(fp8_config=None),
    )
    instance = SimpleNamespace(
        actor_alloc=SimpleNamespace(backend=backend, parallel=None),
        rollout_alloc=SimpleNamespace(backend="vllm", parallel=None),
        _should_offload_rollout=False,
        _should_offload_actor=False,
        _should_offload_critic=False,
        _should_offload_ref=False,
        _should_offload_teacher=False,
        config=SimpleNamespace(
            actor=actor,
            rollout=SimpleNamespace(_version="v1", return_routed_experts=False),
            vllm=SimpleNamespace(
                quantization="ascend", load_format="dummy", dtype="bfloat16"
            ),
            awex={"comm_backend": comm},
        ),
    )
    if error:
        with pytest.raises(ValueError, match=error):
            scope["_validate_cfg"](instance)
    else:
        scope["_validate_cfg"](instance)


def test_shared_module_iterator_drives_restore_and_post_load(mxfp8_module):
    """Only MXFP8 modules receive restore/post-load calls, in model order."""
    model = torch.nn.Module()
    events = []
    for name, scheme_type in [("dense", _LinearScheme), ("experts", _MoEScheme)]:
        module = torch.nn.Module()
        scheme = scheme_type()
        scheme.restore_weights_for_rl_loading = lambda layer, name=name: events.append(
            ("restore", name)
        )
        module.quant_method = SimpleNamespace(
            quant_method=scheme,
            process_weights_after_loading=lambda layer, name=name: events.append(
                ("post", name)
            ),
        )
        model.add_module(name, module)
    model.normal = torch.nn.Module()
    model.normal.quant_method = SimpleNamespace(quant_method=object())

    found = list(mxfp8_module.iter_mxfp8_modules(model))
    mxfp8_module.restore_mxfp8_weights_for_loading(model)
    mxfp8_module.process_mxfp8_weights_after_loading(model)

    assert [name for name, _, _ in found] == ["dense", "experts"]
    assert found[0][1] is model.dense
    assert found[1][1] is model.experts
    assert events == [
        ("restore", "dense"),
        ("restore", "experts"),
        ("post", "dense"),
        ("post", "experts"),
    ]


@pytest.mark.parametrize("k", [48, 64, 96])
def test_shared_quantizer_preserves_xccl_scale_validation(mxfp8_module, monkeypatch, k):
    """Common quantization retains XCCL's existing alignment and strict scale checks."""
    model = torch.nn.Module()
    model.linear = mxfp8_module.LinearBase()
    model.linear.weight = _parameter((2, k), torch.float8_e4m3fn)
    model.linear.weight_scale = _parameter((2, (k + 31) // 32), torch.uint8)
    source = torch.ones((2, k), dtype=torch.bfloat16)
    native_scale = torch.arange(2 * ((k + 63) // 64) * 2, dtype=torch.uint8).reshape(
        2, -1, 2
    )
    calls = []

    def quantize(tensor, *, axis, dst_type):
        calls.append((axis, dst_type))
        return tensor.to(dst_type), native_scale

    monkeypatch.setattr(mxfp8_module.torch_npu, "npu_dynamic_mx_quant", quantize)
    monkeypatch.setitem(sys.modules, "torch_npu", mxfp8_module.torch_npu)
    weights = [("linear.weight", source)]
    if k == 48:
        with pytest.raises(RuntimeError, match="divisible by 32"):
            list(mxfp8_module._quantize_weights(weights, model, group_size=32))
        assert not calls
    elif k == 96:
        with pytest.raises(RuntimeError, match="MXFP8 scale linear.weight_scale"):
            list(mxfp8_module._quantize_weights(weights, model, group_size=32))
        assert calls == [(-1, torch.float8_e4m3fn)]
    else:
        loaded = dict(mxfp8_module._quantize_weights(weights, model, group_size=32))
        assert calls == [(-1, torch.float8_e4m3fn)]
        torch.testing.assert_close(
            loaded["linear.weight"].float(), source.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            loaded["linear.weight_scale"], native_scale.reshape(2, 2), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "weight_dtype,scale_dtype,error",
    [
        (torch.bfloat16, torch.uint8, "invalid dtype"),
        (torch.float8_e4m3fn, torch.float32, "invalid scale"),
    ],
)
def test_shared_quantizer_rejects_invalid_dtypes(
    mxfp8_module, monkeypatch, weight_dtype, scale_dtype, error
):
    monkeypatch.setitem(sys.modules, "torch_npu", mxfp8_module.torch_npu)
    """Both weight transfer paths use the same native output dtype checks."""
    monkeypatch.setattr(
        mxfp8_module.torch_npu,
        "npu_dynamic_mx_quant",
        lambda tensor, **kwargs: (
            torch.zeros((2, 64), dtype=weight_dtype),
            torch.zeros((2, 1, 2), dtype=scale_dtype),
        ),
    )
    with pytest.raises(RuntimeError, match=error):
        mxfp8_module.quantize_mxfp8_tensor(
            torch.ones((2, 64), dtype=torch.bfloat16), "local.weight"
        )


def _make_runner(k=96, moe=False):
    model = torch.nn.Module()
    model.config = SimpleNamespace(tie_word_embeddings=True)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 8, dtype=torch.bfloat16)
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    layer.input_layernorm = torch.nn.LayerNorm(8, dtype=torch.float32)
    if moe:
        layer.mlp = torch.nn.Module()
        quantized = torch.nn.Module()
        layer.mlp.experts = quantized
        shapes = {"w13_weight": (2, 128, k), "w2_weight": (2, 8, k)}
        scheme = _MoEScheme()
    else:
        layer.self_attn = torch.nn.Module()
        quantized = torch.nn.Module()
        layer.self_attn.qkv_proj = quantized
        shapes = {"weight": (8, k)}
        scheme = _LinearScheme()
    quantized._mxfp8_original_shapes = {}
    for name, shape in shapes.items():
        setattr(quantized, name, _parameter(shape, torch.float8_e4m3fn))
        scale_shape = (*shape[:-1], (k + 31) // 32)
        setattr(quantized, name + "_scale", _parameter(scale_shape, torch.uint8))
        quantized._mxfp8_original_shapes[name] = shape
        quantized._mxfp8_original_shapes[name + "_scale"] = scale_shape
    quantized.quant_method = SimpleNamespace(quant_method=scheme)
    return SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(quantization="ascend")
        ),
    )


class _Converter:
    """Exercise AWEX's packed gate/up and Ascend expert transpose contract."""

    def __init__(self, pp_rank=0, pp_size=1, ep_rank=1):
        self.rank_info = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
        self.ep_rank = ep_rank

    def convert_param(self, name, tensor):
        if "experts.w" not in name:
            return [(name, tensor)]
        result = []
        for local_id in range(tensor.shape[0]):
            expert_id = self.ep_rank * tensor.shape[0] + local_id
            logical = tensor[local_id].transpose(0, 1)
            prefix = name.rsplit(".", 1)[0] + f".{expert_id}."
            if name.endswith("w13_weight"):
                gate, up = logical.chunk(2, dim=0)
                result.extend(
                    [
                        (prefix + "gate_proj.weight", gate),
                        (prefix + "up_proj.weight", up),
                    ]
                )
            else:
                result.append((prefix + "down_proj.weight", logical))
        return result


@pytest.fixture
def support(mxfp8_module, monkeypatch):
    events = []
    monkeypatch.setattr(
        mxfp8_module.current_platform, "synchronize", lambda: None, raising=False
    )
    monkeypatch.setattr(
        mxfp8_module,
        "restore_mxfp8_weights_for_loading",
        lambda model: events.append("restore"),
    )
    monkeypatch.setattr(
        mxfp8_module,
        "process_mxfp8_weights_after_loading",
        lambda model: events.append("post"),
    )
    monkeypatch.setattr(mxfp8_module, "is_ascend_mxfp8", lambda runner: True)
    monkeypatch.setattr(mxfp8_module, "events", events, raising=False)
    return mxfp8_module


@pytest.mark.parametrize("moe,k", [(False, 32), (False, 64), (False, 96), (True, 64)])
def test_metadata_and_receive_share_logical_views(receiver_module, support, moe, k):
    """Byte/scale views write the actual FP8 parameters without a BF16 cache."""
    runner = _make_runner(k=k, moe=moe)
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)
    converter = _Converter()
    meta = receiver.metadata_parameters(converter)
    shell = receiver._shell

    actual = receiver.begin_receive(converter)

    assert meta.keys() == actual.keys()
    assert all(t.device.type == "meta" for t in meta.values())
    assert actual["lm_head.weight"] is actual["model.embed_tokens.weight"]
    assert actual["model.layers.0.input_layernorm.weight"].dtype == torch.float32
    for name in meta:
        assert meta[name].shape == actual[name].shape
        assert meta[name].dtype == actual[name].dtype
    if moe:
        base_name = "model.layers.0.mlp.experts.w13_weight"
        actual["model.layers.0.mlp.experts.2.gate_proj.weight"].fill_(3)
        actual["model.layers.0.mlp.experts.2.up_proj.weight"].fill_(5)
        base = runner.model.get_parameter(base_name).view(torch.uint8)
        torch.testing.assert_close(
            base[0, :64], torch.full_like(base[0, :64], 3), rtol=0, atol=0
        )
        torch.testing.assert_close(
            base[0, 64:], torch.full_like(base[0, 64:], 5), rtol=0, atol=0
        )
        scale_name = "model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv"
        actual[scale_name].fill_(125)
        base_scale = runner.model.get_parameter(base_name + "_scale")[0, :64]
        torch.testing.assert_close(
            base_scale, torch.full_like(base_scale, 125), rtol=0, atol=0
        )
    else:
        assert meta["model.layers.0.self_attn.qkv_proj.weight"].dtype == torch.uint8
        assert meta["model.layers.0.self_attn.qkv_proj.weight_scale_inv"].shape == (
            8,
            k // 32,
        )
    actual.clear()
    receiver.release()
    assert receiver._shell is shell
    assert not receiver._receiving
    assert not hasattr(receiver, "_buffers")


@pytest.mark.parametrize("k", [32, 96])
def test_moe_unsupported_scale_pairs_fail_before_receiving(receiver_module, support, k):
    """MoE rejects the current library's unsupported local K before transport."""
    with pytest.raises(ValueError, match="local K=.*K % 64"):
        receiver_module.AwexAscendMXFP8Receiver(_make_runner(k=k, moe=True))


def test_receive_rebinds_normal_parameters_after_wake(receiver_module, support):
    """A second round resolves live parameters instead of keeping stale views."""
    runner = _make_runner()
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)
    first = receiver.begin_receive(_Converter())
    old = first["model.layers.0.input_layernorm.weight"]
    first.clear()
    receiver.release()
    runner.model.model.layers[0].input_layernorm.weight = _parameter(
        (8,), torch.float32
    )

    second = receiver.begin_receive(_Converter())

    assert second["model.layers.0.input_layernorm.weight"].data_ptr() != old.data_ptr()
    assert (
        second["model.layers.0.input_layernorm.weight"].data_ptr()
        == runner.model.model.layers[0].input_layernorm.weight.data_ptr()
    )


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_tied_alias_only_on_last_pp_stage(receiver_module, support, pp_rank):
    """Tied lm_head aliases follow the original AWEX last-PP-stage rule."""
    receiver = receiver_module.AwexAscendMXFP8Receiver(_make_runner())
    meta = receiver.metadata_parameters(_Converter(pp_rank=pp_rank, pp_size=2))
    assert ("lm_head.weight" in meta) == (pp_rank == 1)


def test_float_only_local_pp_stage_needs_no_bf16_cache(receiver_module, support):
    """A worker with no local quantized layers still receives its FLOAT weights."""
    runner = _make_runner()
    layer = runner.model.model.layers[0].self_attn.qkv_proj
    layer.quant_method = None
    layer.weight = _parameter((8, 80), torch.bfloat16)
    del layer.weight_scale
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)

    parameters = receiver.begin_receive(_Converter(pp_rank=1, pp_size=2))
    receiver.finish_receive()

    assert (
        parameters["model.layers.0.self_attn.qkv_proj.weight"].data_ptr()
        == layer.weight.data_ptr()
    )
    assert not hasattr(receiver, "_buffers")
    assert support.events == ["restore", "post"]


def _fake_quant(tensor):
    n, k = tensor.shape
    scale = torch.full((n, (k + 31) // 32), 127, dtype=torch.uint8)
    return tensor.to(torch.float8_e4m3fn), scale


def _reader(module, runner, converter):
    return SimpleNamespace(
        engine_name="vllm",
        model_context={
            "scheduler": SimpleNamespace(model_runner=runner),
            "infer_engine_config": SimpleNamespace(comm_backend="hccl"),
        },
        weight_converter=converter,
        parameters={},
    )


def test_megatron_awex_uses_target_version_before_actor_version_advances():
    """Initial sync and the first trained update must not reuse CPU IPC keys."""
    path = Path(__file__).parents[1] / "areal/engine/megatron_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine"
    )
    method = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_update_weights_from_awex"
    )
    method.decorator_list = []
    scope = {
        "WeightUpdateMeta": object,
        "dist": SimpleNamespace(get_rank=lambda: 0, barrier=lambda **kwargs: None),
        "current_platform": SimpleNamespace(synchronize=lambda: None),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    writer_steps, reader_steps = [], []

    def update_rollout(meta, step_id, kwargs):
        reader_steps.append(step_id)
        return SimpleNamespace(result=lambda: None)

    engine = SimpleNamespace(
        version=0,
        cpu_group=None,
        awex_writer=SimpleNamespace(
            set_global_step=writer_steps.append,
            write_weights=lambda: None,
        ),
        rollout_engine=SimpleNamespace(
            pause_generation=lambda: None,
            continue_generation=lambda: None,
            update_weights_from_awex=update_rollout,
        ),
    )
    engine.get_version = lambda: engine.version
    update = scope["_update_weights_from_awex"]
    for target_version in (0, 1, 2):
        update(engine, SimpleNamespace(version=target_version, comm_backend="hccl"))
        # The trainer commits only after synchronization.
        engine.version = target_version
    assert writer_steps == reader_steps == [0, 1, 2]

    # Keep unversioned callers working, and distinguish explicit zero from None.
    engine.version = 7
    for target_version in (None, 0):
        update(engine, SimpleNamespace(version=target_version, comm_backend="hccl"))
    assert writer_steps == reader_steps == [0, 1, 2, 7, 0]


def test_integration_worker_check_rejects_stale_weight(receiver_module, monkeypatch):
    """A successful update response cannot hide unchanged FP8 weight data."""
    from tests.experimental.awex.mxfp8_checks import check_mxfp8_update

    runner = _make_runner(k=64)
    layer = runner.model.model.layers[0].self_attn.qkv_proj
    layer.weight.data = layer.weight.data.T  # Inference layout, still all zeros.
    layer.weight_scale.data = layer.weight_scale.data.reshape(8, 1, 2).transpose(0, 1)
    worker = SimpleNamespace(
        model_runner=runner,
        _awex_ascend_mxfp8_receiver=SimpleNamespace(_receiving=False),
        _awex_test_mxfp8_pointers={
            f"model.layers.0.self_attn.qkv_proj.{name}": getattr(layer, name).data_ptr()
            for name in ("weight", "weight_scale")
        },
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        ),
        raising=False,
    )
    native = types.ModuleType("torch_npu")
    native.float8_e4m3fn = torch.float8_e4m3fn
    native.npu_dynamic_mx_quant = lambda tensor, **kwargs: (
        tensor.to(torch.float8_e4m3fn),
        torch.full((1, 1, 2), 127, dtype=torch.uint8),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", native)
    with pytest.raises(AssertionError, match="Mismatched elements"):
        check_mxfp8_update(
            model=runner.model,
            model_context={"scheduler": worker},
            expected_value=1,
        )


@pytest.mark.parametrize("colocate", [False, True])
def test_three_rounds_receive_fp8_and_scales_before_post_load(
    receiver_module, support, monkeypatch, colocate
):
    """Each round binds live targets, writes bytes/scales, and post-loads once."""
    events = support.events
    monkeypatch.setattr(
        receiver_module,
        "current_platform",
        SimpleNamespace(device_type="npu", synchronize=lambda: events.append("sync")),
    )
    reader = _reader(receiver_module, _make_runner(), _Converter())
    runner = reader.model_context["scheduler"].model_runner
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)
    reader.model_context["scheduler"]._awex_ascend_mxfp8_receiver = receiver
    value = 0

    def receive(self, step_id, **kwargs):
        weight = self.parameters["model.layers.0.self_attn.qkv_proj.weight"]
        source = torch.full(weight.shape, value, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        weight[:, :32].copy_(source.view(torch.uint8)[:, :32])
        events.append("source0")
        weight[:, 32:].copy_(source.view(torch.uint8)[:, 32:])
        self.parameters["model.layers.0.self_attn.qkv_proj.weight_scale_inv"].fill_(
            120 + value
        )
        events.append("source1")
        # The real colocation patch finishes before ACK. A second finish is a no-op.
        if colocate:
            receiver_module.finish_awex_ascend_mxfp8_receive(self)
            events.append("ack")
        else:
            events.append("transport_return")
        return "received"

    monkeypatch.setattr(receiver_module, "_original_update_weights", receive)
    for value, step in enumerate([0, 0, 1], 1):
        events.clear()
        assert receiver_module._ascend_mxfp8_update_weights(reader, step) == "received"
        assert events == ["restore", "source0", "source1"] + (
            ["post", "sync", "ack"]
            if colocate
            else ["transport_return", "post", "sync"]
        )
        assert not receiver._receiving
        assert all(t.device.type == "meta" for t in reader.parameters.values())
        weight = runner.model.model.layers[0].self_attn.qkv_proj.weight
        torch.testing.assert_close(
            weight.float(), torch.full_like(weight.float(), value), rtol=0, atol=0
        )


def test_receive_uses_new_fp8_and_scale_targets_after_restore(receiver_module, support):
    """Restore may replace Parameters; receive resolves both new weight and scale."""
    runner = _make_runner()
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)
    originals = dict(runner.model.named_parameters())

    def restore(model):
        for name, (_, scale_name, _) in receiver._weights.items():
            for key in (name, scale_name):
                parent, _, leaf = key.rpartition(".")
                model.get_submodule(parent)._parameters[leaf] = _parameter(
                    originals[key].shape, originals[key].dtype
                )

    support.restore_mxfp8_weights_for_loading = restore
    actual = receiver.begin_receive(_Converter())
    for name in receiver._weights:
        target = runner.model.get_parameter(name)
        assert target is not originals[name]
        assert actual[name].data_ptr() == target.data_ptr()
        assert (
            actual[name + "_scale_inv"].data_ptr()
            == runner.model.get_parameter(name + "_scale").data_ptr()
        )


@pytest.mark.parametrize("post_failure", [False, True])
def test_failed_update_blocks_reuse(
    receiver_module, support, monkeypatch, post_failure
):
    """Partial transfers or failed layout processing cannot be reused."""
    monkeypatch.setattr(
        receiver_module,
        "current_platform",
        SimpleNamespace(device_type="npu", synchronize=lambda: None),
    )
    reader = _reader(receiver_module, _make_runner(), _Converter())

    def receive(self, step_id):
        if not post_failure:
            raise RuntimeError("transport failed")

    def post(model):
        raise RuntimeError("post-load failed")

    monkeypatch.setattr(receiver_module, "_original_update_weights", receive)
    monkeypatch.setattr(support, "process_mxfp8_weights_after_loading", post)
    with pytest.raises(RuntimeError, match="failed"):
        receiver_module._ascend_mxfp8_update_weights(reader, 0)
    receiver = reader.model_context["scheduler"]._awex_ascend_mxfp8_receiver
    assert receiver._receiving
    assert all(t.device.type != "meta" for t in reader.parameters.values())
    assert "post" not in support.events
    with pytest.raises(RuntimeError, match="restart this worker"):
        receiver.begin_receive(_Converter())


@pytest.mark.parametrize(
    "config",
    [
        {"comm_backend": "file"},
        {"comm_backend": "astate"},
        {"weights_validation_steps": -1},
        {"weights_validation_steps": 1},
        {"dump_weights_list_for_validation": ["model.norm.weight"]},
    ],
)
def test_unsupported_debug_and_transport_configs_fail(receiver_module, config):
    """Reject unsupported validation and transports before preparing wire metadata."""
    with pytest.raises(ValueError, match="Online AWEX Ascend MXFP8"):
        receiver_module.validate_awex_ascend_mxfp8_config(config)


@pytest.mark.parametrize("backend", ["hccl", "nccl"])
def test_receiver_accepts_tensor_backends_with_sender_quantization(
    receiver_module, support, monkeypatch, backend
):
    """The existing HCCL/NCCL tensor writers use the sender quantization hook."""
    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    reader = _reader(receiver_module, _make_runner(), _Converter())
    reader.model_context["infer_engine_config"].comm_backend = backend
    assert receiver_module._get_receiver("vllm", reader.model_context) is not None
    receiver_module.validate_awex_ascend_mxfp8_config(
        {"comm_backend": backend, "dump_weights_dir_for_validation": "/unused"}
    )


@pytest.mark.parametrize("backend", ["file", "astate"])
def test_writer_rejects_unsupported_transport_before_metadata(
    receiver_module, monkeypatch, backend
):
    """Direct writer callers cannot publish uint8 metadata for a BF16 sender."""
    pytest.importorskip("awex.util.common")
    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(
        receiver_module,
        "_original_writer_initialize",
        lambda *args: pytest.fail("original writer initialization must not run"),
    )
    writer = SimpleNamespace(
        train_engine=SimpleNamespace(enable_ascend_mxfp8=True, config={}),
        comm_backend=backend,
        meta_server_client=SimpleNamespace(
            get_binary=lambda *args, **kwargs: pytest.fail(
                "must reject before querying metadata"
            )
        ),
        timeout=10,
    )
    with pytest.raises(ValueError, match=f"comm_backend={backend}"):
        receiver_module._ascend_mxfp8_writer_initialize(writer)
    assert not hasattr(writer, "_ascend_mxfp8_names")


@pytest.mark.parametrize(
    "engine,device,quantization",
    [("sglang", "npu", "ascend"), ("vllm", "cuda", "ascend"), ("vllm", "npu", None)],
)
def test_other_modes_use_original_reader(
    receiver_module, monkeypatch, engine, device, quantization
):
    """Normal BF16 and other engines retain the original initialize/update behavior."""
    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type=device)
    )
    reader = _reader(receiver_module, _make_runner(), _Converter())
    reader.model_context["infer_engine_config"].comm_backend = "astate"
    reader.engine_name = engine
    reader.model_context[
        "scheduler"
    ].model_runner.vllm_config.model_config.quantization = quantization
    monkeypatch.setattr(receiver_module, "_original_initialize", lambda self: "init")
    monkeypatch.setattr(
        receiver_module, "_original_update_weights", lambda self, step_id: step_id
    )
    assert receiver_module._ascend_mxfp8_initialize(reader) == "init"
    assert receiver_module._ascend_mxfp8_update_weights(reader, 3) == 3


def test_sender_trims_paired_padding_without_changing_fp8_bits(
    receiver_module, monkeypatch
):
    """K=96 has three valid scales; the fourth native paired slot is padding."""
    tensor = torch.arange(192, dtype=torch.float32).reshape(2, 96).to(torch.bfloat16)
    fp8 = tensor.to(torch.float8_e4m3fn)
    scale = torch.tensor([[[1, 2], [3, 99]], [[4, 5], [6, 99]]], dtype=torch.uint8)
    monkeypatch.setattr(
        receiver_module, "quantize_mxfp8_tensor", lambda *args: (fp8, scale)
    )
    wire, logical = receiver_module._quantize_for_transfer("local.weight", tensor)
    assert wire.data_ptr() == fp8.data_ptr()
    torch.testing.assert_close(
        wire.view(torch.float8_e4m3fn).float(), fp8.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        logical, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("k", [48, 80])
def test_direct_transfer_rejects_partial_groups(receiver_module, k):
    """This path refuses partial groups even if the quantizer can pad them."""
    with pytest.raises(ValueError, match="32-element K groups"):
        receiver_module._quantize_for_transfer(
            "local.weight", torch.empty(7, k, dtype=torch.bfloat16)
        )


def test_installation_is_idempotent_and_rpc_callable_is_importable(
    receiver_module, monkeypatch
):
    """Installing twice cannot wrap itself; metadata RPC retains an importable name."""

    class Resolver:
        _get_model_param_info = staticmethod(lambda *args, **kwargs: "meta")

    class Reader:
        def initialize(self):
            return "init"

        def update_weights(self, step_id):
            return step_id

    original = Reader.update_weights
    for name, attributes in {
        "awex.meta.infer_meta_resolver": {"InferParamMetaResolver": Resolver},
        "awex.reader.weights_reader": {"WorkerWeightsReader": Reader},
    }.items():
        fake = types.ModuleType(name)
        fake.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, fake)
    for name, attributes in {
        "awex.meta.train_meta_resolver": {
            "McoreParamMetaResolver": type(
                "TrainResolver", (), {"_collect_model_param_raw_info": lambda *args: []}
            )
        },
        "awex.writer.weights_writer": {
            "WeightsExchangeShardingWriter": type(
                "Writer",
                (),
                {
                    "_initialize": lambda *args: None,
                    "convert_parameters": lambda *args: {},
                },
            )
        },
    }.items():
        fake = types.ModuleType(name)
        fake.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, fake)
    receiver_module.install_awex_ascend_mxfp8_hooks()
    receiver_module.install_awex_ascend_mxfp8_hooks()
    assert receiver_module._original_update_weights is original
    fn = Resolver._get_model_param_info
    assert getattr(sys.modules[fn.__module__], fn.__qualname__) is fn
    assert fn.__qualname__.endswith("_get_model_param_info")


def test_metadata_rpc_reports_fp8_byte_and_scale_shards(
    receiver_module, support, monkeypatch
):
    """Metadata advertises weight bytes and paired effective uint8 scales."""
    monkeypatch.setattr(
        receiver_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    runner = _make_runner(k=64, moe=True)
    reader = _reader(receiver_module, runner, _Converter())
    converter = reader.weight_converter
    registry = types.ModuleType("awex.models.registry")
    registry.get_infer_weights_converter = lambda *args, **kwargs: converter
    sharding = types.ModuleType("awex.sharding")
    sharding.get_rank_info_extractor = (
        lambda engine: lambda context, rank: converter.rank_info
    )
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, sharding.__name__, sharding)
    monkeypatch.setattr(
        receiver_module, "_original_get_model_param_info", lambda *args, **kwargs: "raw"
    )
    info = receiver_module._ascend_mxfp8_get_model_param_info(
        "vllm",
        reader.model_context["infer_engine_config"],
        convert_params=True,
        model=runner.model,
        model_context=reader.model_context,
    )
    meta = {item["name"]: item for item in info["params_meta"]}
    assert not any(name.endswith("_scale") for name in meta)
    gate = meta["model.layers.0.mlp.experts.2.gate_proj.weight"]
    assert gate["dtype"] == "uint8"
    assert meta[gate["name"] + "_scale_inv"]["shape"] == (64, 2)
    assert gate["shape"] == (64, 64)
    assert gate["numel"] == 64 * 64
    assert (
        receiver_module._ascend_mxfp8_get_model_param_info(
            "vllm", {}, convert_params=False, model_context=reader.model_context
        )
        == "raw"
    )


@pytest.mark.parametrize("ep_size", [1, 2])
def test_real_awex_converter_expert_views(receiver_module, support, ep_size):
    """The pinned AWEX converter sees HF gate/up rows, including EP global ids."""
    converters = pytest.importorskip("awex.converter.vllm_converter")
    runner = _make_runner(k=64, moe=True)
    runner.model.config.num_attention_heads = 8
    runner.model.config.num_key_value_heads = 8
    runner.model.config.num_experts = 4 if ep_size == 2 else 2
    rank = SimpleNamespace(tp_rank=0, ep_rank=ep_size - 1, pp_rank=0, pp_size=1)
    config = SimpleNamespace(tp_size=1, ep_size=ep_size, device_backend="npu")
    converter = converters.VLLMToHFWeightConverter(runner.model.config, config, rank)
    receiver = receiver_module.AwexAscendMXFP8Receiver(runner)
    parameters = receiver.begin_receive(converter)
    expert_id = 2 if ep_size == 2 else 0
    prefix = f"model.layers.0.mlp.experts.{expert_id}."
    parameters[prefix + "gate_proj.weight"].fill_(1)
    parameters[prefix + "up_proj.weight"].fill_(2)
    base = runner.model.model.layers[0].mlp.experts.w13_weight.view(torch.uint8)[0]
    torch.testing.assert_close(base[:64], torch.ones_like(base[:64]), rtol=0, atol=0)
    torch.testing.assert_close(base[64:], torch.full_like(base[64:], 2), rtol=0, atol=0)
    parameters[prefix + "gate_proj.weight_scale_inv"].fill_(123)
    scale = runner.model.model.layers[0].mlp.experts.w13_weight_scale[0]
    torch.testing.assert_close(
        scale[:64], torch.full_like(scale[:64], 123), rtol=0, atol=0
    )


@pytest.mark.parametrize("k", [32, 48, 64, 80, 96, 128])
def test_pinned_dense_layout_roundtrip_preserves_content_and_pointers(k):
    """Execute the actual patched methods on CPU, including odd scale pair padding."""
    source = os.environ.get("AREAL_TEST_PATCHED_MXFP8_SOURCE")
    if not source:
        ascend = importlib.util.find_spec("vllm_ascend")
        if ascend is None:
            pytest.skip("Requires vllm-ascend or AREAL_TEST_PATCHED_MXFP8_SOURCE")
        source = Path(ascend.origin).parent / "quantization/methods/w8a8_mxfp8.py"
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "AscendW8A8MXFP8DynamicLinearMethod"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"process_weights_after_loading", "restore_weights_for_rl_loading"}
    ]
    scope = {"F": F, "logger": logging.getLogger("test")}
    exec(compile(ast.Module(body=methods, type_ignores=[]), source, "exec"), scope)
    layer = torch.nn.Module()
    layer.weight = _parameter((8, k), torch.float8_e4m3fn)
    layer.weight_scale = _parameter((8, (k + 31) // 32), torch.uint8)
    process = scope["process_weights_after_loading"]
    restore = scope["restore_weights_for_rl_loading"]
    process(None, layer)
    pointers = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    for value in [1, 2, 3]:
        restore(None, layer)
        layer.weight.fill_(value)
        layer.weight_scale.fill_(value + 120)
        process(None, layer)
        assert pointers == (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
        torch.testing.assert_close(
            layer.weight.float(),
            torch.full((k, 8), value, dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        expected = torch.full((8, (k + 31) // 32), value + 120, dtype=torch.uint8)
        if expected.shape[-1] % 2:
            expected = F.pad(expected, (0, 1))
        expected = expected.reshape(8, -1, 2).transpose(0, 1).contiguous()
        torch.testing.assert_close(layer.weight_scale, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [32, 64, 96, 128])
def test_native_npu_sender_quantization(receiver_module, k):
    """The actor sends exact native FP8 bytes and effective E8M0 scales."""
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    tensor = torch.randn((16, k), dtype=torch.bfloat16, device="npu")
    native_weight, native_scale = torch_npu.npu_dynamic_mx_quant(
        tensor, axis=-1, dst_type=torch_npu.float8_e4m3fn
    )
    weight, scale = receiver_module._quantize_for_transfer("test.weight", tensor)
    torch.testing.assert_close(weight, native_weight.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        scale, native_scale.reshape(16, -1)[:, : k // 32], rtol=0, atol=0
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_native_npu_matmul_tail_layout_compatibility(
    receiver_module, mxfp8_module, monkeypatch
):
    """Optional K=80 smoke check using the installed vLLM-Ascend post-load method."""
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    compat = pytest.importorskip("vllm_ascend.device.mxfp_compat")
    if compat.FLOAT8_E8M0FNU_DTYPE is None:
        pytest.skip("Native E8M0 scale dtype is unavailable")
    methods = pytest.importorskip("vllm_ascend.quantization.methods.w8a8_mxfp8")
    k = 80  # A partial 32-element group and an odd number of valid scales.
    x = torch.randn((16, k), device="npu", dtype=torch.bfloat16)
    weight = torch.randn((32, k), device="npu", dtype=torch.bfloat16)
    qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)
    qw, sw = receiver_module.quantize_mxfp8_tensor(weight, "test.weight")
    sw = sw.reshape(32, -1)[:, : (k + 31) // 32]
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(qw, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sw, requires_grad=False)
    methods.AscendW8A8MXFP8DynamicLinearMethod.process_weights_after_loading(
        None, layer
    )
    output = torch_npu.npu_quant_matmul(
        qx,
        layer.weight,
        layer.weight_scale,
        scale_dtype=compat.FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=sx,
        pertoken_scale_dtype=compat.FLOAT8_E8M0FNU_DTYPE,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )

    assert output.shape == (16, 32)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
