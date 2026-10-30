# SPDX-License-Identifier: Apache-2.0

"""DP2/PP2/VPP2 snapshots with optimizer-step gather overlap disabled.

Run on four NPUs with Megatron-Core 0.16.1 and the target MindSpeed stack.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any

import areal.utils.torch_npu_compat  # noqa: F401  # isort: skip
import mindspeed.megatron_adaptor  # noqa: F401  # isort: skip

import torch  # noqa: I001
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.package_info import __version__ as mcore_version

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    DistributedDataParallelConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine.megatron_engine import MegatronEngine
from areal.infra.platforms import current_platform
from areal.utils import logging, seeding
from areal.utils.data import broadcast_tensor_container

logger = logging.getLogger("MegatronSnapshotNPUTest")


def make_engine(model_path: str) -> MegatronEngine:
    config = TrainEngineConfig(
        backend="megatron:d2p2t1",
        experiment_name="test",
        trial_name="parameter_snapshot_npu",
        path=model_path,
        dtype="bfloat16",
        disable_dropout=True,
        mb_spec=MicroBatchSpec(n_mbs=4, max_tokens_per_mb=256),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            virtual_pipeline_parallel_size=2,
            bridge_type="mbridge",
            ddp=DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                overlap_param_gather=True,
                align_param_gather=False,
            ),
            # MindSpeed's force-sync path cannot wait on pending gather lists.
            overlap_param_gather_with_optimizer_step=False,
        ),
    )
    engine = MegatronEngine(config)
    allocation = ModelAllocation.from_str(config.backend)
    engine.create_process_group(parallel_strategy=allocation.parallel)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1, dataset_size=128, train_batch_size=8
        ),
    )
    return engine


def mock_input(device: torch.device) -> dict[str, Any]:
    seqlens = torch.randint(32, 64, (8,), dtype=torch.int, device=device)
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(10000, 50000, (8, max_seqlen), device=device)
    attention_mask = torch.arange(max_seqlen, device=device).unsqueeze(
        0
    ) < seqlens.unsqueeze(1)
    input_ids.masked_fill_(~attention_mask, 0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    return torch.mean(logprobs)


def run_parameter_snapshot(model_path: str, output: Path) -> None:
    rank = int(os.environ["RANK"])
    engines = []
    control_parameters = []
    control_logp = None
    ok = True
    for use_snapshot in (False, True):
        seeding.set_random_seed(0, key=f"engine{rank}")
        engine = make_engine(model_path)
        engines.append(engine)
        assert engine.device.type == "npu"
        assert mpu.get_data_parallel_world_size() == 2
        assert mpu.get_pipeline_model_parallel_world_size() == 2
        assert len(engine.model) == 2
        seeding.set_random_seed(0, key=f"data{rank}")
        input_ = broadcast_tensor_container(
            mock_input(engine.device),
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )

        def forward():
            engine.eval()
            return engine.forward(
                input_=input_, aggregate_fn=lambda xs: torch.cat(xs, dim=0)
            )

        def train():
            nonlocal ok
            engine.train()
            stats = engine.train_batch(
                input_=input_,
                loss_fn=mock_loss_fn,
                loss_weight_fn=lambda x: x["cu_seqlens"][-1],
            )
            if (
                stats["update_successful"] != 1
                or not math.isfinite(stats["grad_norm"])
                or stats["grad_norm"] <= 0
            ):
                logger.error("rank %s invalid optimizer update: %s", rank, stats)
                ok = False

        def check_close(actual, expected):
            nonlocal ok
            try:
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-6)
            except AssertionError as exc:
                logger.error("rank %s snapshot parity failed: %s", rank, exc)
                ok = False

        initial_logp = forward().detach().cpu()
        if use_snapshot:
            check_close(initial_logp, control_logp)
            engine.capture_parameter_snapshot()
        else:
            control_logp = initial_logp
        train()

        if use_snapshot:
            with engine.parameter_snapshot_context():
                check_close(forward().detach().cpu(), initial_logp)
            engine.clear_parameter_snapshot()

        updated_logp = forward().detach().cpu()
        if torch.allclose(updated_logp, initial_logp, rtol=3e-3, atol=1e-6):
            logger.error("rank %s updated logp matches the initial policy", rank)
            ok = False

        for update_index in range(2):
            parameters = {
                (part_index, name): parameter.detach().cpu()
                for part_index, model_chunk in enumerate(engine.model)
                for name, parameter in model_chunk.named_parameters()
            }
            if use_snapshot:
                assert parameters.keys() == control_parameters[update_index].keys()
                for name, parameter in parameters.items():
                    check_close(parameter, control_parameters[update_index][name])
            else:
                control_parameters.append(parameters)
            if update_index == 0:
                # Check optimizer master state and the next update after switching.
                train()
                forward()

    verdict = torch.tensor(int(ok), dtype=torch.int32, device=engines[-1].device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    ok = bool(verdict.item())
    for engine in reversed(engines):
        engine.destroy()
    if rank == 0:
        output.write_text("Passed" if ok else "Failed", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if mcore_version != "0.16.1":
        raise RuntimeError(
            f"requires target Megatron-Core 0.16.1, loaded {mcore_version}"
        )
    if current_platform.device_type != "npu":
        raise RuntimeError("this snapshot runner requires Ascend NPUs")
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("DP2/PP2/VPP2 snapshot test requires exactly 4 workers")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    run_parameter_snapshot(args.model_path, args.output)


if __name__ == "__main__":
    main()
