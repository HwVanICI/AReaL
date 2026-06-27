# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from areal.utils import logging

if TYPE_CHECKING:
    from awex.sharding.rank_info import RankInfo

logger = logging.getLogger(__name__)


class _AwexWriterAdapterBase:
    """Common adapter surface expected by Awex training writers."""

    def __init__(self, engine, meta):
        if getattr(meta, "use_mindspeed", False) or (meta.comm_backend == "hccl"):
            # Ensure MindSpeed patches are enabled before Awex imports Megatron.
            os.environ.setdefault("AWEX_USE_MINDSPEED", "1")
        try:
            from awex.writer.weights_writer import get_weights_exchange_writer
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Awex is not available. Install awex or ensure it is on PYTHONPATH."
            ) from exc

        self._engine = engine
        self._get_writer = get_weights_exchange_writer
        self.weights_exchange_writer = None

        self.hf_config = self._get_hf_config(engine)
        self.model = engine.model
        self.global_step = -1

        self.meta_server_addr = meta.meta_server_addr or ""
        self.comm_backend = meta.comm_backend or "file"
        self.enable_debug_mode = meta.enable_debug_mode
        self.enable_colocate_mode = meta.enable_colocate_mode

        self.config = {
            "weights_validation_steps": meta.weights_validation_steps,
            "validate_weights_every_n_steps": meta.validate_weights_every_n_steps,
            "dump_weights_list_for_validation": meta.dump_weights_list_for_validation,
            "dump_weights_dir_for_validation": meta.dump_weights_dir_for_validation,
            "disable_weights_exchange_pipeline": meta.disable_weights_exchange_pipeline,
            "debug_mode_config": meta.debug_mode_config,
            "weights_exchange_ipc_backend": meta.weights_exchange_ipc_backend,
        }

        self._export_meta_server_env(self.meta_server_addr)

    def _get_hf_config(self, engine):
        return getattr(engine, "hf_config", getattr(engine, "model_config", None))

    def _export_meta_server_env(self, meta_server_addr: str) -> None:
        os.environ["AWEX_META_SERVER_ADDR"] = meta_server_addr or ""
        ip, port = (meta_server_addr or ":").split(":")
        os.environ["AWEX_META_SERVER_IP"] = ip
        os.environ["AWEX_META_SERVER_PORT"] = port

    def initialize(self) -> None:
        if self.weights_exchange_writer is not None:
            return
        self.weights_exchange_writer = self._get_writer(self)
        self.weights_exchange_writer.initialize()
        if self.enable_colocate_mode:
            self.release_memory_occupation()

    def set_global_step(self, global_step: int) -> None:
        # Awex writer uses this for logging and synchronization metadata.
        self.global_step = global_step

    def write_weights(self, **kwargs) -> None:
        if self.weights_exchange_writer is None:
            raise RuntimeError("Awex writer not initialized.")
        self.weights_exchange_writer.write_weights(step_id=self.global_step, **kwargs)
        if self.enable_colocate_mode:
            self.release_memory_occupation()

    def save_hf_checkpoint(self, path: str) -> None:
        self._engine._save_model_to_hf(path)

    def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but MegatronEngine does not "
                "support fine-grained memory release. No-op."
            )

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but MegatronEngine does not "
                "support fine-grained memory resume. No-op."
            )

    def release_grad_memory(self, empty_cache: bool = True) -> None:
        # Placeholder to satisfy Awex TrainingEngine API.
        return None


class AwexMegatronWriterAdapter(_AwexWriterAdapterBase):
    """Adapter that exposes AReaL MegatronEngine to Awex writer API."""

    def __init__(self, engine, meta):
        super().__init__(engine, meta)
        self.engine_name = "mcore"


class AwexFSDPWriterAdapter(_AwexWriterAdapterBase):
    """Adapter that exposes AReaL FSDPEngine to Awex writer API."""

    def __init__(self, engine, meta):
        super().__init__(engine, meta)
        self.engine_name = "fsdp"
        self._meta = meta

    def save_hf_checkpoint(self, path: str) -> None:
        self._engine._save_model_to_hf(
            path, self._engine.tokenizer, self._engine.processor
        )

    def get_awex_rank_info(self) -> RankInfo:
        from awex.sharding.rank_info import RankInfo

        mesh = self._engine.world_mesh
        dim_names = tuple(mesh.mesh_dim_names or ())

        tp_size = mesh.size(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 1
        tp_rank = (
            mesh.get_local_rank(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 0
        )
        cp_size = mesh.size(dim_names.index("sp")) if "sp" in dim_names else 1
        cp_rank = mesh.get_local_rank(dim_names.index("sp")) if "sp" in dim_names else 0
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.dp_rank,
            ep_rank=0,
            ep_size=1,
            ep_tp_rank=0,
            ep_tp_size=1,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.dp_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="ulysses" if cp_size > 1 else "none",
        )

    def iter_awex_named_parameters(
        self,
    ) -> Iterator[tuple[str, nn.Parameter | torch.Tensor]]:
        yield from self._engine._get_model_name_parameters(self._meta)

    def get_awex_local_param_metadata(self) -> list[dict]:
        rank_info = self.get_awex_rank_info()
        metadata = []

        for name, param in self.iter_awex_named_parameters():
            if self._skip_tied_lm_head(name):
                continue
            tensor = param.data if isinstance(param, nn.Parameter) else param
            if isinstance(tensor, DTensor):
                local_tensor = tensor._local_tensor
                sharding_dim, num_shards = self._extract_dtensor_sharding(tensor)
                global_offset = self._compute_dtensor_offset(tensor)
                global_shape = tuple(tensor.shape)
                global_numel = int(tensor.numel())
            else:
                local_tensor = tensor
                sharding_dim = 0
                num_shards = 1
                global_offset = tuple(0 for _ in tuple(local_tensor.shape))
                global_shape = tuple(local_tensor.shape)
                global_numel = int(local_tensor.numel())

            metadata.append(
                {
                    "name": name,
                    "shape": tuple(local_tensor.shape),
                    "numel": int(local_tensor.numel()),
                    "dtype": local_tensor.dtype,
                    "global_shape": global_shape,
                    "global_numel": global_numel,
                    "global_offset": global_offset,
                    "sharding_dim": sharding_dim,
                    "num_shards": num_shards,
                    "rank_info": rank_info,
                }
            )

        return metadata

    def get_awex_local_parameters(
        self, required_names: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        params = {}
        for name, param in self.iter_awex_named_parameters():
            if self._skip_tied_lm_head(name):
                continue
            if required_names is not None and name not in required_names:
                continue
            tensor = param.data if isinstance(param, nn.Parameter) else param
            if isinstance(tensor, DTensor):
                tensor = tensor._local_tensor
            tensor = self._engine._cast_to_compute_dtype(tensor)
            params[name] = tensor.contiguous()
        return params

    def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but FSDPEngine does not "
                "support fine-grained memory release. No-op."
            )

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but FSDPEngine does not "
                "support fine-grained memory resume. No-op."
            )

    def _skip_tied_lm_head(self, name: str) -> bool:
        return (
            getattr(self.hf_config, "tie_word_embeddings", False)
            and name == "lm_head.weight"
        )

    @staticmethod
    def _compute_dtensor_offset(dtensor: DTensor) -> tuple[int, ...]:
        global_shape = tuple(dtensor.shape)
        offset = [0] * len(global_shape)
        remaining_shape = list(global_shape)

        for mesh_dim, placement in enumerate(dtensor.placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                mesh_size = dtensor.device_mesh.size(mesh_dim)
                coord = dtensor.device_mesh.get_local_rank(mesh_dim)
                local_size, local_offset = placement._local_shard_size_and_offset(
                    remaining_shape[shard_dim], mesh_size, coord
                )
                offset[shard_dim] += local_offset
                remaining_shape[shard_dim] = local_size

        return tuple(offset)

    @staticmethod
    def _extract_dtensor_sharding(dtensor: DTensor) -> tuple[int, int]:
        shard_info: dict[int, int] = {}
        for mesh_dim, placement in enumerate(dtensor.placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                mesh_size = dtensor.device_mesh.size(mesh_dim)
                shard_info[dim] = shard_info.get(dim, 1) * mesh_size

        if not shard_info:
            return 0, 1

        primary_dim = max(shard_info.items(), key=lambda item: item[1])[0]
        return primary_dim, shard_info[primary_dim]
