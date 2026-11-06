# SPDX-License-Identifier: Apache-2.0

"""Quantize on the actor and receive MXFP8 bytes through AWEX's tensor path."""

from __future__ import annotations

import math

import torch

from areal.infra.platforms import current_platform
from areal.utils.mxfp8 import quantize_mxfp8_tensor

_original_get_model_param_info = None
_original_initialize = None
_original_update_weights = None
_original_writer_initialize = None
_original_collect_model_param_raw_info = None
_original_convert_parameters = None

_SCALE_SUFFIX = "_scale_inv"


def validate_awex_ascend_mxfp8_config(config) -> None:
    """Reject paths which bypass the tensor receiver or inspect BF16 model data."""

    def value(name, default):
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    backend = value("comm_backend", "nccl")
    if backend in {"file", "astate"}:
        raise ValueError(
            f"Online AWEX Ascend MXFP8 does not support comm_backend={backend}; "
            "direct FP8 transfer requires the convert_parameters tensor path"
        )
    if value("weights_validation_steps", 0) != 0 or value(
        "dump_weights_list_for_validation", []
    ):
        raise ValueError(
            "Online AWEX Ascend MXFP8 requires weights_validation_steps=0 and "
            "an empty dump_weights_list_for_validation"
        )


def _validate_matrix_k(name: str, k: int, offset: int = 0) -> None:
    if k <= 0 or k % 32 or offset % 32:
        raise ValueError(
            f"AWEX Ascend MXFP8 {name}: K={k}, offset={offset}; "
            "direct transfer requires complete, aligned 32-element K groups"
        )


def _validate_bf16_matrix(
    name: str, shape: tuple[int, ...], dtype: torch.dtype | str
) -> None:
    if len(shape) != 2 or dtype not in (torch.bfloat16, "bfloat16"):
        raise ValueError(f"AWEX MXFP8 {name}: actor must supply a BF16 [N,K] matrix")
    _validate_matrix_k(name, shape[-1])


def _quantized_names(parameters_meta) -> set[str]:
    names = {parameter.name for parameter in parameters_meta}
    return {
        name.removesuffix(_SCALE_SUFFIX)
        for name in names
        if name.endswith(_SCALE_SUFFIX)
    }


def _validate_wire_metadata(parameters_meta, quantized: set[str]) -> None:
    """Validate actual shard boundaries and matching scale coordinates."""
    parameters = {parameter.name: parameter for parameter in parameters_meta}
    for name in quantized:
        # ParameterMeta describes a logical matrix across ranks, not a tensor.
        weight = parameters.get(name)
        scale = parameters.get(name + _SCALE_SUFFIX)
        if weight is None or scale is None:
            raise ValueError(f"Missing AWEX MXFP8 weight/scale pair for {name}")
        if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
            raise ValueError(f"AWEX MXFP8 {name}: weight bytes and scale must be uint8")
        shape = tuple(weight.global_shape)
        if len(shape) != 2:
            raise ValueError(f"AWEX MXFP8 {name}: expected a logical [N,K] matrix")
        if tuple(scale.global_shape) != (shape[0], shape[1] // 32):
            raise ValueError(f"AWEX MXFP8 {name}: incompatible global scale shape")
        scales = {(s.engine_rank, s.global_rank): s for s in scale.shards}
        if len(scales) != len(weight.shards):
            raise ValueError(f"AWEX MXFP8 {name}: incompatible scale shard count")
        # AWEX derives global K from these shards; checking every local K also
        # covers global K alignment. Keep each rank's scale coordinates checked.
        for shard in weight.shards:
            k, offset = shard.shape[-1], shard.global_offset[-1]
            _validate_matrix_k(name, k, offset)
            scale_shard = scales.get((shard.engine_rank, shard.global_rank))
            if (
                scale_shard is None
                or tuple(scale_shard.shape) != (*shard.shape[:-1], k // 32)
                or tuple(scale_shard.global_offset)
                != (*shard.global_offset[:-1], offset // 32)
                or scale_shard.sharding_dim != shard.sharding_dim
                or scale_shard.dtype != torch.uint8
                or shard.dtype != torch.uint8
            ):
                raise ValueError(
                    f"AWEX MXFP8 {name}: incompatible weight/scale shard coordinates"
                )


def _ascend_mxfp8_writer_initialize(self):
    if current_platform.device_type != "npu" or not getattr(
        self.train_engine, "enable_ascend_mxfp8", False
    ):
        return _original_writer_initialize(self)
    from awex.util.common import check_train_infer_params_meta, from_binary

    validate_awex_ascend_mxfp8_config(
        self.train_engine.config | {"comm_backend": self.comm_backend}
    )
    # The reader publishes this before waiting for training metadata. Keep the
    # original initialization/collectives; only its input dtype/scale records change.
    infer_meta = from_binary(
        self.meta_server_client.get_binary("infer_params_meta", timeout=self.timeout)
    )
    self._ascend_mxfp8_names = _quantized_names(infer_meta)
    if not self._ascend_mxfp8_names:
        raise ValueError("AWEX Ascend MXFP8 inference metadata contains no scale pairs")
    _validate_wire_metadata(infer_meta, self._ascend_mxfp8_names)
    result = _original_writer_initialize(self)
    _validate_wire_metadata(self.parameters_meta, self._ascend_mxfp8_names)
    check_train_infer_params_meta(
        self.parameters_meta, infer_meta, raise_exception=True
    )
    return result


def _ascend_mxfp8_collect_model_param_raw_info(self, convert_params=False, **kwargs):
    # Run real converter collectives and PP canonicalization exactly as before.
    metadata = _original_collect_model_param_raw_info(self, convert_params, **kwargs)
    writer = getattr(self._train_engine, "weights_exchange_writer", None)
    quantized = getattr(writer, "_ascend_mxfp8_names", None)
    if not convert_params or quantized is None:
        return metadata
    for rank in metadata:
        projected = []
        for parameter in rank["params_meta"]:
            name = parameter["name"]
            if name not in quantized:
                projected.append(parameter)
                continue
            shape = tuple(parameter["shape"])
            _validate_bf16_matrix(name, shape, parameter["dtype"])
            scale_shape = (*shape[:-1], shape[-1] // 32)
            projected.extend(
                (
                    {**parameter, "dtype": torch.uint8},
                    {
                        **parameter,
                        "name": name + _SCALE_SUFFIX,
                        "shape": scale_shape,
                        "numel": math.prod(scale_shape),
                        "dtype": torch.uint8,
                    },
                )
            )
        rank["params_meta"] = projected
    return metadata


def _quantize_for_transfer(
    name: str, tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_bf16_matrix(name, tensor.shape, tensor.dtype)
    n, k = tensor.shape
    weight, scale = quantize_mxfp8_tensor(tensor.contiguous(), name)
    if weight.shape != tensor.shape or tuple(scale.shape) != (n, (k + 63) // 64, 2):
        raise RuntimeError(f"AWEX MXFP8 {name}: quantizer returned incompatible shapes")
    # view reinterprets FP8 bits. It is never a numeric conversion to integers.
    return weight.view(torch.uint8), scale.reshape(n, -1)[:, : k // 32]


@torch.no_grad()
def _ascend_mxfp8_convert_parameters(self, required_names=None):
    quantized = getattr(self, "_ascend_mxfp8_names", None)
    if quantized is None:
        return _original_convert_parameters(self, required_names)
    required = None if required_names is None else set(required_names)
    base_required = (
        None
        if required is None
        else {
            name.removesuffix(_SCALE_SUFFIX) if name.endswith(_SCALE_SUFFIX) else name
            for name in required
        }
    )
    converted = _original_convert_parameters(self, base_required)
    output = {}
    # Do not skip the converter even for an empty set: it owns TP collectives.
    for name in list(converted):
        tensor = converted.pop(name)
        if required is not None and name not in base_required:
            continue
        if name in quantized:
            expected = self.current_worker_parameters_map[name].shards
            if len(expected) != 1 or tuple(tensor.shape) != tuple(expected[0].shape):
                raise ValueError(
                    f"AWEX MXFP8 {name}: actor tensor does not match published local metadata"
                )
            weight, scale = _quantize_for_transfer(name, tensor)
            if required is None or name in required:
                output[name] = weight
            scale_name = name + _SCALE_SUFFIX
            if required is None or scale_name in required:
                output[scale_name] = scale
        else:
            output[name] = tensor
    missing = set() if required is None else required - output.keys()
    if missing:
        raise ValueError(f"Missing AWEX MXFP8 payloads: {sorted(missing)}")
    return output


class AwexAscendMXFP8Receiver:
    """Bind AWEX's receive views directly to restored FP8 weights and scales."""

    def __init__(self, model_runner):
        from areal.engine.vllm_ext import ascend_mxfp8

        self.model = model_runner.model
        self._failure: str | None = None
        self._receiving = False
        self._finished = False
        # name -> (logical loading shape, scale parameter name, is MoE)
        self._weights: dict[str, tuple[tuple[int, ...], str, bool]] = {}
        scales = set()
        for prefix, module, scheme in ascend_mxfp8.iter_mxfp8_modules(self.model):
            is_moe = isinstance(
                scheme, ascend_mxfp8.AscendW8A8MXFP8DynamicFusedMoEMethod
            )
            weight_names = ("w13_weight", "w2_weight") if is_moe else ("weight",)
            shapes = getattr(module, "_mxfp8_original_shapes", {})
            for weight_name in weight_names:
                name = f"{prefix}.{weight_name}" if prefix else weight_name
                scale_name = f"{name}_scale"
                shape = tuple(shapes.get(weight_name, ()))
                if len(shape) != (3 if is_moe else 2) or min(shape) <= 0:
                    raise RuntimeError(
                        f"Missing logical MXFP8 loading shape for {name}"
                    )
                _validate_matrix_k(name, shape[-1])
                if is_moe and shape[-1] % 64:
                    raise ValueError(
                        f"AWEX Ascend MXFP8 MoE {name} has local K={shape[-1]}; "
                        "the current vllm-ascend MoE scale layout requires K % 64 == 0"
                    )
                expected_scale = (*shape[:-1], shape[-1] // 32)
                if tuple(shapes.get(f"{weight_name}_scale", ())) != expected_scale:
                    raise RuntimeError(
                        f"Incompatible MXFP8 loading scale shape for {name}"
                    )
                self._weights[name] = (shape, scale_name, is_moe)
                scales.add(scale_name)
        # A PP stage may own only FLOAT parameters even though the global model
        # uses MXFP8. Such a worker still follows AWEX's normal receive lifecycle.
        self._shell: dict[str, torch.Tensor] = {}
        for name, parameter in self.model.named_parameters():
            if name in scales:
                continue
            if name in self._weights:
                shape = self._weights[name][0]
                dtype = torch.uint8
                self._shell[self._weights[name][1]] = torch.empty(
                    (*shape[:-1], shape[-1] // 32), dtype=torch.uint8, device="meta"
                )
            else:
                if str(parameter.dtype).startswith("torch.float8_"):
                    raise RuntimeError(f"Unrecognized FP8 receive parameter {name}")
                shape, dtype = tuple(parameter.shape), parameter.dtype
            self._shell[name] = torch.empty(shape, dtype=dtype, device="meta")
        if not self._weights.keys() <= self._shell.keys():
            raise RuntimeError(
                "MXFP8 weights are missing from model.named_parameters()"
            )

    def _convert(self, parameters, converter) -> dict[str, torch.Tensor]:
        converted = {}
        scales = {scale: name for name, (_, scale, _) in self._weights.items()}
        for name, parameter in parameters.items():
            weight_name = scales.get(name, name)
            if weight_name in self._weights and self._weights[weight_name][2]:
                # AWEX's NPU converter accepts [E,K,N] and exposes HF [N,K]
                # expert views. Loading views are [E,N,K] for weights and
                # [E,N,K/32] for scales.
                parameter = parameter.transpose(-1, -2)
            for hf_name, tensor in converter.convert_param(weight_name, parameter):
                converted[hf_name + _SCALE_SUFFIX if name in scales else hf_name] = (
                    tensor
                )
        rank = converter.rank_info
        if (
            getattr(self.model.config, "tie_word_embeddings", False)
            and rank.pp_rank == rank.pp_size - 1
            and "lm_head.weight" not in converted
            and "model.embed_tokens.weight" in converted
        ):
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        return converted

    def metadata_parameters(self, converter) -> dict[str, torch.Tensor]:
        return self._convert(self._shell, converter)

    def begin_receive(self, converter) -> dict[str, torch.Tensor]:
        if self._failure is not None:
            raise RuntimeError(
                "AWEX Ascend MXFP8 receiver failed; restart this worker: "
                + self._failure
            )
        if self._receiving:
            raise RuntimeError("AWEX Ascend MXFP8 receive is already in progress")
        from areal.engine.vllm_ext import ascend_mxfp8

        self._receiving = True
        self._finished = False
        ascend_mxfp8.restore_mxfp8_weights_for_loading(self.model)
        targets = dict(self.model.named_parameters())
        parameters = {}
        for name, shell in self._shell.items():
            target = targets[name].detach()
            if name in self._weights:
                if target.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"AWEX MXFP8 {name}: target must be float8_e4m3fn")
                target = target.view(torch.uint8)
            if target.shape != shell.shape or target.dtype != shell.dtype:
                raise ValueError(
                    f"AWEX MXFP8 {name}: target does not match published loading metadata"
                )
            parameters[name] = target
        return self._convert(parameters, converter)

    def finish_receive(self) -> None:
        if self._finished:
            return
        from areal.engine.vllm_ext import ascend_mxfp8

        ascend_mxfp8.process_mxfp8_weights_after_loading(self.model)
        current_platform.synchronize()
        self._finished = True

    def release(self) -> None:
        self._receiving = False


def _get_receiver(engine_name, model_context):
    if engine_name != "vllm" or current_platform.device_type != "npu":
        return None
    worker = model_context["scheduler"]
    model_runner = worker.model_runner
    if model_runner.vllm_config.model_config.quantization != "ascend":
        return None
    validate_awex_ascend_mxfp8_config(model_context["infer_engine_config"])
    receiver = getattr(worker, "_awex_ascend_mxfp8_receiver", None)
    if receiver is None:
        from areal.engine.vllm_ext import ascend_mxfp8

        if not ascend_mxfp8.is_ascend_mxfp8(model_runner):
            return None
        receiver = AwexAscendMXFP8Receiver(model_runner)
        worker._awex_ascend_mxfp8_receiver = receiver
    return receiver


def _ascend_mxfp8_get_model_param_info(
    engine_name, infer_engine_config, convert_params=False, engine_rank=0, **kwargs
):
    # This must be a module-level callable: AWEX dispatches module + qualname.
    # Its suffix also preserves vLLM's detection of metadata collection on all DP ranks.
    receiver = _get_receiver(engine_name, kwargs["model_context"])
    if receiver is None or not convert_params:
        return _original_get_model_param_info(
            engine_name, infer_engine_config, convert_params, engine_rank, **kwargs
        )
    validate_awex_ascend_mxfp8_config(infer_engine_config)
    from awex.models.registry import get_infer_weights_converter
    from awex.sharding import get_rank_info_extractor

    rank = get_rank_info_extractor(engine_name)(kwargs["model_context"], engine_rank)
    model = receiver.model
    architecture = type(model).__name__
    converter = get_infer_weights_converter(
        engine_name,
        architecture,
        hf_config=model.config,
        infer_engine_config=infer_engine_config,
        rank_info=rank,
    )
    return {
        "rank_info": rank,
        "model_arch_name": architecture,
        "params_meta": [
            {
                "name": name,
                "numel": tensor.numel(),
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for name, tensor in receiver.metadata_parameters(converter).items()
        ],
    }


def _ascend_mxfp8_initialize(self):
    receiver = _get_receiver(self.engine_name, self.model_context)
    if receiver is None:
        return _original_initialize(self)
    from awex.util.common import check_train_infer_params_meta

    quantized = _quantized_names(self.parameters_meta)
    _validate_wire_metadata(self.parameters_meta, quantized)
    _validate_wire_metadata(self.training_params_meta, quantized)
    check_train_infer_params_meta(
        self.training_params_meta, self.parameters_meta, raise_exception=True
    )
    self.parameters = receiver.metadata_parameters(self.weight_converter)


def finish_awex_ascend_mxfp8_receive(reader) -> None:
    """Complete post-load before a colocated reader acknowledges its update."""
    receiver = _get_receiver(reader.engine_name, reader.model_context)
    if receiver is not None:
        receiver.finish_receive()


def _ascend_mxfp8_update_weights(self, step_id, **kwargs):
    receiver = _get_receiver(self.engine_name, self.model_context)
    if receiver is None:
        return _original_update_weights(self, step_id, **kwargs)
    try:
        self.parameters = receiver.begin_receive(self.weight_converter)
        result = _original_update_weights(self, step_id, **kwargs)
        # Colocation already finishes before ACK; separation finishes here,
        # before the worker RPC returns and the API server wakes the KV cache.
        receiver.finish_receive()
    except Exception as exc:
        # Receives might still be in flight. Keep their views and block reuse;
        # never post-load in finally or publish success for a partial update.
        receiver._failure = str(exc)
        raise
    self.parameters = receiver.metadata_parameters(self.weight_converter)
    receiver.release()
    return result


def install_awex_ascend_mxfp8_hooks() -> None:
    from awex.meta.infer_meta_resolver import InferParamMetaResolver
    from awex.meta.train_meta_resolver import McoreParamMetaResolver
    from awex.reader.weights_reader import WorkerWeightsReader
    from awex.writer.weights_writer import WeightsExchangeShardingWriter

    global _original_get_model_param_info
    global _original_initialize
    global _original_update_weights
    global _original_writer_initialize
    global _original_collect_model_param_raw_info
    global _original_convert_parameters
    if WorkerWeightsReader.initialize is _ascend_mxfp8_initialize:
        return
    _original_get_model_param_info = InferParamMetaResolver._get_model_param_info
    _original_initialize = WorkerWeightsReader.initialize
    _original_update_weights = WorkerWeightsReader.update_weights
    _original_writer_initialize = WeightsExchangeShardingWriter._initialize
    _original_collect_model_param_raw_info = (
        McoreParamMetaResolver._collect_model_param_raw_info
    )
    _original_convert_parameters = WeightsExchangeShardingWriter.convert_parameters
    InferParamMetaResolver._get_model_param_info = staticmethod(
        _ascend_mxfp8_get_model_param_info
    )
    WorkerWeightsReader.initialize = _ascend_mxfp8_initialize
    WorkerWeightsReader.update_weights = _ascend_mxfp8_update_weights
    WeightsExchangeShardingWriter._initialize = _ascend_mxfp8_writer_initialize
    McoreParamMetaResolver._collect_model_param_raw_info = (
        _ascend_mxfp8_collect_model_param_raw_info
    )
    WeightsExchangeShardingWriter.convert_parameters = _ascend_mxfp8_convert_parameters
