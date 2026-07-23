# SPDX-License-Identifier: Apache-2.0

import torch
from awex.converter.mcore_converter import (
    McoreToHFWeightConverter,
    _process_mcore_pp_name,
)
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name
from awex.sharding.param_sharding import (
    ShardingStrategy,
    ShardingType,
    get_default_sharding_dim,
)
from transformers import PretrainedConfig


class Qwen3VLShardingStrategy(ShardingStrategy):
    _visual_sharding_dims = {
        "merger.linear_fc1.weight": 0,
        "merger.linear_fc1.bias": 0,
        "merger.linear_fc2.weight": 1,
        "attn.qkv.weight": 0,
        "attn.qkv.bias": 0,
        "attn.proj.weight": 1,
        "mlp.linear_fc1.weight": 0,
        "mlp.linear_fc1.bias": 0,
        "mlp.linear_fc2.weight": 1,
    }

    _gdn_sharding_dims = {}

    def get_shared_expert_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for shared expert parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        if self.tp_size > 1:
            return ShardingType.TP_SHARDING, sharding_dim, self.tp_size
        else:
            return ShardingType.NO_SHARDING, sharding_dim, 1

    def _get_gdn_sharding_strategy(self):
        # todo
        return ShardingType.NO_SHARDING, 0, 1

    def _get_visual_sharding_strategy(self, parameter_name, **kwargs):
        tp_size = self.rank_info.tp_size
        if tp_size == 1:
            return ShardingType.NO_SHARDING, 0, 1

        visual_prefix = "model.visual."
        if parameter_name.startswith(visual_prefix):
            suffix = parameter_name[len(visual_prefix) :]

            if suffix.startswith("blocks."):
                # Skip "blocks.{i}."
                parts = suffix.split(".", 2)
                if len(parts) >= 3:
                    suffix = parts[
                        2
                    ]  # e.g. "attn.qkv.weight" or "mlp.linear_fc1.weight"

            sharding_dim = self._visual_sharding_dims.get(suffix)
            if sharding_dim is not None:
                return ShardingType.TP_SHARDING, sharding_dim, tp_size

        return ShardingType.NO_SHARDING, 0, 1

    def get_sharding_strategy(self, parameter_name, **kwargs):
        if "visual." in parameter_name:
            return self._get_visual_sharding_strategy(parameter_name, **kwargs)
        if "linear_attn" in parameter_name:
            return self._get_gdn_sharding_strategy()
        if (
            "shared_expert_gate" in parameter_name
            or "shared_experts.gate_weight" in parameter_name
        ):
            return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)


def _split_mcore_gated_attn_qkv(
    linear_qkv: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    return [("qkv", linear_qkv)]


def _reshard_mcore_gdn_conv1d(
    conv1d: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    return [("qkvz", conv1d)]


def _split_mcore_gdn_in_proj(
    in_proj: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    return [("qkvz", in_proj)]


def reshard_visual_attn_qkv(
    parameter: torch.Tensor,
    infer_atten_tp_size: int,
    vision_config: PretrainedConfig,
    train_tp_rank: int,
    train_tp_size: int,
):
    from awex.converter.mcore_converter import get_full_tensor

    weight = get_full_tensor(parameter, dim=0)
    num_heads = vision_config.num_heads
    head_dim = vision_config.hidden_size // num_heads
    query_list = []
    key_list = []
    value_list = []
    for qkv in torch.chunk(weight, num_heads, dim=0):
        q, k, v = qkv.split([head_dim, head_dim, head_dim], dim=0)
        query_list.append(q)
        key_list.append(k)
        value_list.append(v)
    # concat the query, key, value
    all_query = torch.cat(query_list, dim=0)
    all_key = torch.cat(key_list, dim=0)
    all_value = torch.cat(value_list, dim=0)

    query_shards = all_query.chunk(infer_atten_tp_size, dim=0)
    key_shards = all_key.chunk(infer_atten_tp_size, dim=0)
    value_shards = all_value.chunk(infer_atten_tp_size, dim=0)
    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    merged = torch.cat(qkv_tp_groups, dim=0)
    if train_tp_size and train_tp_size > 1:
        if train_tp_rank is None:
            raise ValueError("train_tp_rank is required when train_tp_size > 1")
        shards = torch.chunk(merged, train_tp_size, dim=0)
        if train_tp_rank >= len(shards):
            raise ValueError(
                f"train_tp_rank {train_tp_rank} out of range for tp_size {train_tp_size}"
            )
        return shards[train_tp_rank]
    return merged


class McoreToHFWeightConverterQwen3VL(McoreToHFWeightConverter):
    def __init__(self, hf_config, rank_info, infer_conf, tf_config):
        super().__init__(hf_config.text_config, rank_info, infer_conf, tf_config)
        self.vision_config = hf_config.vision_config

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_vision_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert vision encoder (vision_model) parameters from mcore to HF format.

        Name mapping:
          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_qkv.weight
          HF:    model.visual.blocks.{i}.attn.qkv.weight

          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_proj.weight
          HF:    model.visual.blocks.{i}.attn.proj.weight

          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight
          HF:    model.visual.blocks.{i}.norm1.weight

          mcore: module.vision_model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight
          HF:    model.visual.blocks.{i}.norm2.weight

          mcore: module.vision_model.merger.patch_norm.weight
          HF:    model.visual.merger.norm.weight

          mcore: module.vision_model.decoder.layers.{i}  →  model.visual.blocks.{i}
        """
        # Strip "module.vision_model." prefix (already stripped "module.module.")
        # After stripping, remaining starts with "vision_model."
        assert name.startswith("vision_model."), f"Expected vision_model prefix: {name}"
        remaining = name[len("vision_model.") :]

        # --- Top-level vision params (patch_embed, pos_embed, merger) ---
        if remaining.startswith("patch_embed."):
            return [(f"model.visual.{remaining}", parameter)]
        if remaining.startswith("pos_embed."):
            return [(f"model.visual.{remaining}", parameter)]
        if remaining.startswith("merger."):
            # merger.patch_norm → merger.norm
            remaining = remaining.replace("merger.patch_norm.", "merger.norm.", 1)
            return [(f"model.visual.{remaining}", parameter)]

        # --- Block-level vision params ---
        # mcore: decoder.layers.{i}.self_attention.* or decoder.layers.{i}.mlp.*
        if remaining.startswith("decoder.layers."):
            # Extract layer index and sub-name
            rest = remaining[len("decoder.layers.") :]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse vision block name: {name}")
            block_idx = parts[0]
            sub_name = parts[1]

            # self_attention → attn
            if sub_name.startswith("self_attention."):
                attn_sub = sub_name[len("self_attention.") :]
                # linear_qkv.layer_norm_weight → norm1.weight
                if attn_sub == "linear_qkv.layer_norm_weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.norm1.weight", parameter)
                    ]
                if attn_sub == "linear_qkv.layer_norm_bias":
                    return [(f"model.visual.blocks.{block_idx}.norm1.bias", parameter)]
                # linear_qkv.weight/bias → attn.qkv.weight/bias
                if attn_sub == "linear_qkv.weight":
                    reshard_param = reshard_visual_attn_qkv(
                        parameter,
                        self.infer_atten_tp_size,
                        self.vision_config,
                        self.rank_info.attn_tp_rank,
                        self.rank_info.attn_tp_size,
                    )
                    return [
                        (
                            f"model.visual.blocks.{block_idx}.attn.qkv.weight",
                            reshard_param,
                        )
                    ]
                if attn_sub == "linear_qkv.bias":
                    reshard_param = reshard_visual_attn_qkv(
                        parameter,
                        self.infer_atten_tp_size,
                        self.vision_config,
                        self.rank_info.attn_tp_rank,
                        self.rank_info.attn_tp_size,
                    )
                    return [
                        (
                            f"model.visual.blocks.{block_idx}.attn.qkv.bias",
                            reshard_param,
                        )
                    ]
                # linear_proj.weight/bias → attn.proj.weight/bias
                if attn_sub == "linear_proj.weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.attn.proj.weight", parameter)
                    ]
                if attn_sub == "linear_proj.bias":
                    return [
                        (f"model.visual.blocks.{block_idx}.attn.proj.bias", parameter)
                    ]
                raise NotImplementedError(f"Unsupported vision attn param: {name}")

            # mlp
            if sub_name.startswith("mlp."):
                mlp_sub = sub_name[len("mlp.") :]
                # linear_fc1.layer_norm_weight → norm2.weight
                if mlp_sub == "linear_fc1.layer_norm_weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.norm2.weight", parameter)
                    ]
                if mlp_sub == "linear_fc1.layer_norm_bias":
                    return [(f"model.visual.blocks.{block_idx}.norm2.bias", parameter)]
                # linear_fc1/fc2 weight/bias → mlp.linear_fc1/fc2 weight/bias
                if mlp_sub in (
                    "linear_fc1.weight",
                    "linear_fc1.bias",
                    "linear_fc2.weight",
                    "linear_fc2.bias",
                ):
                    return [
                        (f"model.visual.blocks.{block_idx}.mlp.{mlp_sub}", parameter)
                    ]
                raise NotImplementedError(f"Unsupported vision mlp param: {name}")

            raise NotImplementedError(f"Unsupported vision block param: {name}")

        raise NotImplementedError(f"Unsupported vision param: {name}")

    def _convert_gdn_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> list[tuple[str, torch.Tensor]]:
        if "in_proj.weight" in name:
            return _split_mcore_gdn_in_proj(
                parameter,
                self.hf_config,
                self.infer_atten_tp_size,
                self.rank_info.attn_tp_rank,
                self.rank_info.attn_tp_size,
            )
        elif "in_proj.layer_norm_weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "conv1d.weight" in name:
            return _reshard_mcore_gdn_conv1d(
                parameter,
                self.hf_config,
                self.infer_atten_tp_size,
                self.rank_info.attn_tp_rank,
                self.rank_info.attn_tp_size,
            )
        elif "dt_bias" in name:
            return [("linear_attn.dt_bias", parameter)]
        elif "A_log" in name:
            return [("linear_attn.A_log", parameter)]
        elif "out_norm.weight" in name:
            return [("linear_attn.norm.weight", parameter + 1.0)]
        elif "out_proj.weight" in name:
            return [("linear_attn.out_proj.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported GDN parameter name: {name}")

    def _is_linear_attn_layer(self, layer_number: int, name: str) -> bool:
        # first time here,self._pp_stage_layer_id_map is {}
        # when pp > 1,layer_number is local rank。it does not support the num_hidden_layers % (pp * full_attention_interval) != 0
        if not self._pp_stage_layer_id_map:
            gdn_keys = ["dt_bias", "A_log", "in_proj", "conv1d", "out_norm", "out_proj"]
            for key in gdn_keys:
                if key in name:
                    return True
            return False

        text_config = self.hf_config
        layer_types = getattr(text_config, "layer_types", [])
        if layer_types:
            return layer_types[layer_number] == "linear_attention"
        interval = getattr(text_config, "full_attention_interval", 4)
        return (layer_number + 1) % interval != 0

    def _convert_attn_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> list[tuple[str, torch.Tensor]]:
        name = _process_mcore_pp_name(
            name,
            self.rank_info,
            self.hf_config,
            self.tf_config,
            vp_stage=vp_stage,
            pp_stage_layer_id_map=self._pp_stage_layer_id_map,
        )
        rest = name.split("decoder.layers.", 1)[1]
        layer_str = rest.split(".", 1)[0]
        layer_number = int(layer_str)

        if self._is_linear_attn_layer(layer_number, name):
            # GDN linear attention
            converted = []
            for sub_name, param in self._convert_gdn_param(
                name, parameter, str(layer_number)
            ):
                converted.append((f"model.layers.{layer_number}.{sub_name}", param))
            return converted
        else:
            # Gated full attention with attention_output_gate
            if "linear_qkv.weight" in name or name.endswith("linear_qkv"):
                converted = _split_mcore_gated_attn_qkv(
                    parameter,
                    self.hf_config,
                    self.infer_atten_tp_size,
                    self.rank_info.attn_tp_rank,
                    self.rank_info.attn_tp_size,
                )
                result = []
                for sub_name, param in converted:
                    sub_name = self._normalize_attn_name(sub_name)
                    result.append((f"model.layers.{layer_number}.{sub_name}", param))
                return result
            else:
                converted = []
                for attn_name, param in self._convert_attention_param(
                    name, parameter, layer_str
                ):
                    attn_name = self._normalize_attn_name(attn_name)
                    converted.append(
                        (f"model.layers.{layer_number}.{attn_name}", param)
                    )
                return converted

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> list[tuple[str, torch.Tensor]]:
        name = name.replace("module.", "")

        # ---- Vision encoder parameters ----
        if name.startswith("vision_model."):
            return self._convert_vision_param(name, parameter)

        language_prefix = "language_model."
        name = name.replace(language_prefix, "")

        if "self_attention" in name:
            converted_params = self._convert_attn_param(
                name, parameter, vp_stage=vp_stage
            )
        else:
            converted_params = super().convert_param(name, parameter, vp_stage=vp_stage)

        if len(converted_params) == 1 and converted_params[0][0] == "lm_head.weight":
            return converted_params
        return [
            (s.replace("model.", "model.language_model.", 1), t)
            for s, t in converted_params
        ]


class VLLMToHFWeightConverterQwen3VL(
    SGlangToHFWeightConverter,
):
    """vLLM-side converter for Qwen3-VL multimodal models.

    Handles the ``model.language_model.*`` prefix, GDN linear_attn
    parameters, and standard MHA self_attn with separate Q/K/V projections.
    vLLM uses the same parameter names as the HF checkpoint for Qwen3-VL,
    so minimal normalization is needed.
    """

    def __init__(
        self,
        model_config,
        infer_engine_config,
        rank_info,
    ):
        super().__init__(model_config.text_config, infer_engine_config, rank_info)

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_visual_param(self, name: str, parameter):
        return [(f"model.{name}", parameter)]

    def _convert_gdn_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        rest = name.split("layers.", 1)[1]
        layer_str = rest.split(".", 1)[0]
        prefix = f"model.layers.{layer_str}."

        if "in_proj_qkvz" in name:
            return [(f"{prefix}linear_attn.in_proj_qkvz.weight", parameter)]
        elif "in_proj_ba" in name:
            return [(f"{prefix}linear_attn.in_proj_ba.weight", parameter)]
        elif "conv1d.weight" in name:
            return [(f"{prefix}linear_attn.conv1d.weight", parameter)]
        elif "dt_bias" in name:
            return [(f"{prefix}linear_attn.dt_bias", parameter)]
        elif "A_log" in name:
            return [(f"{prefix}linear_attn.A_log", parameter)]
        elif "norm.weight" in name:
            return [(f"{prefix}linear_attn.norm.weight", parameter)]
        elif "out_proj.weight" in name:
            return [(f"{prefix}linear_attn.out_proj.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported vLLM GDN parameter name: {name}")

    def _normalize_name(self, name: str) -> str:
        name, has_scale_inv = normalize_scale_inv_name(name)
        replacements = [
            (".self_attn.attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.attn.o_proj", ".attention.dense"),
            (".self_attn.o_proj", ".attention.dense"),
            (".self_attn.proj", ".attention.dense"),
            (".self_attn.q_norm", ".attention.query_layernorm"),
            (".self_attn.k_norm", ".attention.key_layernorm"),
        ]
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
        # Guard against double normalization.
        name = name.replace("query_key_value_proj_proj", "query_key_value_proj")
        return append_scale_inv(name, has_scale_inv)

    def convert_param(self, name, parameter):
        if name.startswith("visual."):
            converted_params = self._convert_visual_param(name, parameter)
            return converted_params

        language_prefix = "language_model."
        name = name.replace(language_prefix, "").replace(
            "shared_expert.", "shared_experts."
        )

        if "linear_attn" in name:
            converted_params = self._convert_gdn_param(name, parameter)
        else:
            converted_params = super().convert_param(
                self._normalize_name(name), parameter
            )

        if len(converted_params) == 1 and converted_params[0][0] == "lm_head.weight":
            return converted_params
        return [
            (s.replace("model.", "model.language_model.", 1), t)
            for s, t in converted_params
        ]


CONFIG = [
    {
        "model_name": "Qwen3_5ForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "sglang_converter": VLLMToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
    {
        "model_name": "Qwen3_5MoeForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "sglang_converter": VLLMToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
]
