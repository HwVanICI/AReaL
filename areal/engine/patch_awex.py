# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import os
import pkgutil
import time

import torch


def _patch_vllm_adapter():
    import awex
    from awex.vllm_awex_adapter import logger

    def resume_memory_occupation(self, tags=None) -> None:
        if isinstance(tags, str):
            tags = [tags]
        logger.info("Resume memory occupation via vLLM wake_up.")
        self._call_engine_async("wake_up", tags)

    awex.vllm_awex_adapter.AwexVLLMServerAdapter.resume_memory_occupation = (
        resume_memory_occupation
    )


def _patch_nccl_comm():
    from awex.transfer import nccl_comm
    from awex.transfer.nccl_comm import logger
    from awex.transfer.transfer_plan import slice_tensor
    from awex.util import device as device_util

    def execute_tensors_to_copy(tensors_to_copy, copy_ops, recv_parameters, stage: str):
        start_time = time.time()
        num_ops = len(copy_ops)
        logger.info(f"Start to execute {num_ops} copy operations for {stage}")
        assert len(copy_ops) == len(tensors_to_copy), (
            f"Number of copy operations mismatch: {len(copy_ops)} != {len(tensors_to_copy)}"
        )
        with torch.no_grad():
            for send_tensor, recv_op in zip(tensors_to_copy, copy_ops):
                recv_tensor = recv_parameters[recv_op.recv_shard_meta.name]
                recv_tensor_sliced = slice_tensor(recv_tensor, recv_op, False)
                if not recv_tensor_sliced.is_contiguous():
                    dst_slice = recv_tensor_sliced.contiguous()
                    dst_slice.copy_(send_tensor)
                    recv_tensor_sliced.copy_(dst_slice)
                else:
                    recv_tensor_sliced.copy_(send_tensor)
        duration = time.time() - start_time
        device_util.synchronize(device_id=device_util.current_device())
        logger.info(
            f"Finished executing {num_ops} copy operations for {stage}, took {duration:.4f} seconds"
        )

    nccl_comm.execute_tensors_to_copy = execute_tensors_to_copy


def _patch_colocate_reader():
    import awex
    from awex.reader.nccl_reader import logger
    from awex.transfer.transfer_plan import (
        TransferPlanBuilder,
    )
    from awex.util import device as device_util
    from awex.util.common import (
        compute_statistics,
        get_ip_address,
    )
    from awex.util.gpu import get_gpu_status, print_current_gpu_status
    from awex.util.system_util import count_open_fds
    from awex.util.tensor_util import (
        cuda_ipc_deserialize,
        ipc_deserialize,
        reconstruct_tensors_from_groups,
    )

    device_phy_ids = [str(i) for i in range(8)]
    visible_env_key = "CUDA_VISIBLE_DEVICES"
    if device_util.get_device_type() == "npu":
        device_phy_ids = [str(i) for i in range(16)]
        visible_env_key = "ASCEND_RT_VISIBLE_DEVICES"

    def _init_reader_in_colocate_mode(self):
        def _get_current_phy_id():
            device_phy_id = os.getenv(visible_env_key)
            if device_phy_id in device_phy_ids:
                return device_phy_id
            ids = device_phy_id.split(",")
            gpu_id = getattr(self.scheduler, "gpu_id", None) or getattr(
                self.scheduler, "local_rank", None
            )
            if gpu_id is None:
                gpu_id = int(os.environ.get("LOCAL_RANK", 0))
            current_phy_id = ids[int(gpu_id)]
            logger.info(
                f"_get_current_phy_id:{current_phy_id=} {gpu_id=} {ids=} {os.environ.get('LOCAL_RANK', 0)=}"
            )
            return current_phy_id

        device_id = _get_current_phy_id()

        self.meta_server_client.add_object_to_set(
            "inference_device_rank_entries",
            (get_ip_address(), device_id, self.transfer_rank),
        )
        self.meta_server_client.wait_set_until_size(
            "inference_device_rank_entries", self.infer_world_size, timeout=self.timeout
        )
        self.inference_device_mapping = self.meta_server_client.get_set(
            "inference_device_rank_entries"
        )
        self.inference_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in self.inference_device_mapping
        }

        self.meta_server_client.wait_set_until_size(
            "training_device_rank_entries",
            self.training_world_size,
            timeout=self.timeout,
        )
        device_rank_entries = self.meta_server_client.get_set(
            "training_device_rank_entries"
        )
        self.training_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in device_rank_entries
        }
        self.train_to_infer_device_mapping = {}
        self.infer_to_train_device_mapping = {}
        for ip_address, device_id, transfer_rank in device_rank_entries:
            infer_rank = self.inference_device_mapping[(ip_address, device_id)]
            self.train_to_infer_device_mapping[transfer_rank] = infer_rank
            self.infer_to_train_device_mapping[infer_rank] = transfer_rank
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.send_transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.infer_to_train_device_mapping[self.transfer_rank],
        )
        from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport

        self.colocate_transport = NcclColocateStreamBatchTransport(
            self.transfer_rank, self.infer_world_size
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank} in colocate mode"
        )

    def collect_training_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return
        # Can't serialize IPC tensors at initialization since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        # We'll get serialized weights from meta server each step instead
        # Get serialized weights from meta server
        ip_address = get_ip_address()
        device_id = device_util.current_device()
        training_rank = self.infer_to_train_device_mapping[self.transfer_rank]
        key = f"training_serialized_weights_{ip_address}_{training_rank}_{step_id}"
        logger.info(
            f"Start to get serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        self.send_rank, self.send_rank_info, serialized_weights = (
            self.meta_server_client.get_object(key, timeout=self.timeout)
        )
        logger.info(
            f"Finished getting serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        logger.info(
            f"GPU status before deserialization:\n{get_gpu_status()} for rank {self.rank_coordinate}"
        )
        logger.info(f"Open fds before deserialization: {count_open_fds()}")
        # Deserialize weights into tensors
        if self.ipc_backend in ("cpu", "npu"):
            group_shared, metadata, names = ipc_deserialize(serialized_weights)
            group_shared = [t.to(device_id) for t in group_shared]
        else:
            group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        device_util.synchronize(device_id=device_util.current_device())
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        device_util.synchronize(device_id=device_util.current_device())
        self.deserialized_weights = dict(zip(names, tensors))
        logger.info(
            f"Deserialized {len(self.deserialized_weights)} parameters and {len(group_shared)} groups"
        )
        logger.info(
            f"GPU status after deserialization for rank {self.rank_coordinate}:\n{get_gpu_status()}"
        )
        logger.info(f"Open fds after deserialization: {count_open_fds()}")

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        import torch.distributed as dist

        assert self.enable_colocate_mode, "Colocate mode is not enabled"
        self.collect_training_weights(step_id, **kwargs)
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} "
            f"ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )
        start_time = time.time()
        self.colocate_transport.update_weights_in_colocate_mode(
            self.train_to_infer_device_mapping,
            self.infer_to_train_device_mapping,
            self.transfer_rank,
            self.rank_coordinate,
            self.infer_world_size,
            self.send_transfer_plan,
            self.transfer_plan,
            self.weights_update_group,
            self.deserialized_weights,
            self.parameters,
            step_id=step_id,
        )
        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}"
        )
        self.deserialized_weights = None
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        ip_address = get_ip_address()
        # device_id = device_util.current_device()
        training_rank = self.infer_to_train_device_mapping[self.transfer_rank]
        key_suffix = f"_{ip_address}_{training_rank}_{step_id}"
        # Signal completion to training process
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        gc.collect()
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        write_finished_key = f"write_finished{key_suffix}"
        self.meta_server_client.get_object_then_delete(write_finished_key)
        logger.info(
            f"Finished updating weights in colocate mode for rank {self.transfer_rank}"
        )

    awex.reader.nccl_reader.NCCLWorkerWeightsReader._init_reader_in_colocate_mode = (
        _init_reader_in_colocate_mode
    )
    awex.reader.nccl_reader.NCCLWorkerWeightsReader.collect_training_weights = (
        collect_training_weights
    )
    awex.reader.nccl_reader.NCCLWorkerWeightsReader._update_weights_in_colocate_mode = (
        _update_weights_in_colocate_mode
    )


def _patch_colocate_writer():
    import awex
    from awex.util import device as device_util
    from awex.util.common import compute_statistics, get_ip_address
    from awex.util.gpu import print_current_gpu_status
    from awex.util.system_util import count_open_fds
    from awex.util.tensor_util import (
        cuda_ipc_serialize,
        group_tensors_by_shape_and_dtype,
        ipc_serialize,
        release_tensors,
    )
    from awex.writer.nccl_writer import logger

    device_phy_ids = [str(i) for i in range(8)]
    visible_env_key = "CUDA_VISIBLE_DEVICES"
    if device_util.get_device_type() == "npu":
        device_phy_ids = [str(i) for i in range(16)]
        visible_env_key = "ASCEND_RT_VISIBLE_DEVICES"

    def _init_writer_in_colocate_mode(self):
        self.ipc_backend = "cuda"
        if device_util.get_device_type() == "npu":
            self.ipc_backend = "cpu"

        # Don't get IPC tensors here since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        ip_address = get_ip_address()
        self._set_device()

        def _get_current_phy_id():
            device_phy_id = os.getenv(visible_env_key)
            if device_phy_id in device_phy_ids:
                return device_phy_id
            ids = device_phy_id.split(",")
            gpu_id = int(os.environ.get("LOCAL_RANK", 0))
            current_phy_id = ids[gpu_id]
            logger.info(
                f"_get_current_phy_id:{current_phy_id=} {gpu_id=} {ids=} {os.environ.get('LOCAL_RANK', 0)=}"
            )
            return current_phy_id

        device_id = _get_current_phy_id()
        self.meta_server_client.add_object_to_set(
            "training_device_rank_entries", (ip_address, device_id, self.transfer_rank)
        )
        logger.info(
            f"Initialized NCCL weights writer for rank {self.transfer_rank} in colocate mode"
        )

    @torch.no_grad()
    def _write_weights_in_colocate_mode(self, step_id, **kwargs):
        start_time = time.time()
        tensors, names = self._prepare_params_for_colocate()
        num_tensors = len(tensors)
        if self.ipc_backend in ("cpu", "npu"):
            tensors = [t.cpu() for t in tensors]
        logger.info(
            f"Start to group tensors by shape and dtype for rank {self.transfer_rank}"
        )
        # this will copy tensor by concatenate
        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        device_util.synchronize(device_id=device_util.current_device())
        logger.info(
            f"Finished grouping tensors by shape and dtype for rank {self.transfer_rank}"
        )
        print_current_gpu_status(
            f"after group_tensors_by_shape_and_dtype for rank {self.transfer_rank}"
        )
        logger.info(f"Open fds before serialize: {count_open_fds()}")

        release_tensors(tensors)
        del tensors
        self.train_engine.release_memory_occupation("weights")
        self.meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self.transfer_rank
        )
        print_current_gpu_status(
            f"after offloaded weights for rank {self.transfer_rank}"
        )

        if self.ipc_backend in ("cpu", "npu"):
            group_shared = [tensor.cpu().share_memory_() for tensor in group_tensors]
            serialized_weights = ipc_serialize((group_shared, metadata, names))
        else:
            group_shared = [
                tensor.to(device_util.get_torch_device()).share_memory_()
                for tensor in group_tensors
            ]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        device_util.synchronize(device_id=device_util.current_device())
        logger.info(
            f"Finished serializing ipc weights with {num_tensors} params, and {len(group_shared)} groups "
            f"for rank {self.transfer_rank}"
        )
        logger.info(f"Open fds after serialize: {count_open_fds()}")

        # Put serialized weights to meta server
        ip_address = get_ip_address()
        # device_id = device_util.current_device()
        key_suffix = f"_{ip_address}_{self.transfer_rank}_{step_id}"
        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        self.meta_server_client.put_object(
            serialized_weights_key,
            (self.transfer_rank, self.rank_info, serialized_weights),
        )
        logger.info(
            f"Put {len(group_shared)} serialized training weights to meta server "
            f"with key {serialized_weights_key} for step {step_id}"
        )
        # Wait for inference engines to finish processing
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.get_object(update_finished_key, timeout=self.timeout)
        self.meta_server_client.delete_if_exists(update_finished_key)
        release_tensors(group_tensors)
        release_tensors(group_shared)
        del group_tensors
        del group_shared
        device_util.synchronize(device_id=device_util.current_device())
        gc.collect()
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        print_current_gpu_status(
            f"after clear group_shared for rank {self.transfer_rank}"
        )
        write_finished_key = f"write_finished{key_suffix}"
        self.meta_server_client.put_object(write_finished_key, True)
        duration = time.time() - start_time
        compute_statistics(
            self._history_write_weights_time,
            step_id,
            duration,
            "Send weights using NCCL in colocate mode",
        )
        logger.info(
            f"Finished writing weights in colocate mode for rank {self.transfer_rank}"
        )

    awex.writer.nccl_writer.NCCLWeightsWriter._init_writer_in_colocate_mode = (
        _init_writer_in_colocate_mode
    )
    awex.writer.nccl_writer.NCCLWeightsWriter._write_weights_in_colocate_mode = (
        _write_weights_in_colocate_mode
    )


def _patch_meta_resolver_for_vlm():
    import awex

    _orig_init = awex.meta.meta_resolver.ParamMetaResolver.__init__

    def init_for_vlm(self, hf_config):
        if hasattr(hf_config, "text_config") and hf_config.text_config:
            if not hasattr(hf_config, "num_hidden_layers"):
                setattr(
                    hf_config,
                    "num_hidden_layers",
                    hf_config.text_config.num_hidden_layers,
                )
        _orig_init(self, hf_config)

    awex.meta.meta_resolver.ParamMetaResolver.__init__ = init_for_vlm


def _import_model_configs():
    model_arch_name_to_config = {}
    package_name = "areal.engine.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            module = importlib.import_module(name)
            if hasattr(module, "CONFIG"):
                entry = module.CONFIG
                if isinstance(entry, list):
                    for tmp in entry:
                        model_name = tmp["model_name"]
                        assert model_name not in model_arch_name_to_config, (
                            f"Duplicated model config for {model_name}"
                        )
                        model_arch_name_to_config[model_name] = tmp
                else:
                    model_name = entry["model_name"]
                    assert model_name not in model_arch_name_to_config, (
                        f"Duplicated model config for {model_name}"
                    )
                    model_arch_name_to_config[model_name] = entry

    return model_arch_name_to_config


def _registry_models():
    from awex.models.registry import ModelRegistry

    model_dict = _import_model_configs()
    for model_name, config in model_dict.items():
        if model_name in ModelRegistry.models:
            print(f"model {model_name} already registered, skipping.", flush=True)
            continue
        ModelRegistry.models[model_name] = config
        print(f"areal register model {model_name} for awex success.", flush=True)


def patch_awex():
    _registry_models()
    _patch_vllm_adapter()
    _patch_nccl_comm()
    _patch_colocate_reader()
    _patch_colocate_writer()
    _patch_meta_resolver_for_vlm()
