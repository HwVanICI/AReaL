# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils.stats_tracker import DistributedStatsTracker, ReduceType


def _collectors(tracker, group=None):
    # Compile the real recording branch and exporter without NPU initialization.
    path = Path(__file__).resolve().parents[1] / "areal/engine/megatron_engine.py"
    engine = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine"
    )
    methods = {
        node.name: node for node in engine.body if isinstance(node, ast.FunctionDef)
    }
    recording = next(
        node
        for node in ast.walk(methods["forward_backward_batch"])
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Name) and part.id == "r3_completed"
            for part in ast.walk(node.test)
        )
    )
    namespace = {
        "torch": torch,
        "ReduceType": ReduceType,
        "stats_tracker": SimpleNamespace(
            scope=tracker.scope,
            scalar=tracker.scalar,
            stat_compact=tracker.stat_compact,
            export_all=tracker.export,
        ),
        "mpu": SimpleNamespace(get_pipeline_model_parallel_world_size=lambda: 1),
    }
    code = compile(ast.Module(body=[recording], type_ignores=[]), str(path), "exec")
    exec(
        compile(
            ast.Module(body=[methods["export_stats"]], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    instance = SimpleNamespace(
        parallel_strategy=SimpleNamespace(context_parallel_size=1),
        _offload_aware_context=nullcontext,
        data_parallel_group=group,
        data_parallel_group_gloo=group,
    )

    def record(stats):
        exec(code, namespace | {"r3_completed": True, "r3_context": {"stats": stats}})

    return record, lambda: namespace["export_stats"](instance)


def _counters(total, valid, active, replayed):
    counters = {
        "batches_with_side_channel": 1,
        "samples_total": total,
        "routing_valid_samples": valid,
        "routing_invalid_samples": total - valid,
        "routing_valid_fraction": valid / max(total, 1),
        "router_stage_microbatches": active,
        "mode_replay_microbatches": replayed,
        "skipped_microbatches": active - replayed,
        "tokens_real": total * 10,
    }
    if active > replayed:
        counters["skip_invalid_sample"] = active - replayed
    return counters


def _assert_combined(stats, prefix):
    expected = {
        "batches_with_side_channel": 2,
        "samples_total": 10,
        "routing_valid_samples": 5,
        "routing_invalid_samples": 5,
        "router_stage_microbatches": 5,
        "mode_replay_microbatches": 2,
        "skipped_microbatches": 3,
        "skip_invalid_sample": 3,
        "tokens_real": 100,
        "routing_valid_fraction": 0.5,
        "replay_fraction_active": 0.4,
        "skip_fraction_active": 0.6,
        "enabled": 1.0,
    }
    for key, value in expected.items():
        assert stats[prefix + key] == pytest.approx(value), key


@pytest.mark.parametrize("scope", ["", "ppo_actor/update"])
def test_r3_stats_unequal_calls_export_totals_and_weighted_ratios(scope):
    record, export = _collectors(DistributedStatsTracker(scope))

    record(_counters(total=4, valid=4, active=2, replayed=2))
    record(_counters(total=6, valid=1, active=3, replayed=0))
    stats = export()

    _assert_combined(stats, f"{scope}/r3/" if scope else "r3/")
    assert export() == {}


def test_r3_stats_no_router_stage_exports_zero_active_ratios():
    record, export = _collectors(DistributedStatsTracker())

    record(_counters(total=3, valid=3, active=0, replayed=0))
    stats = export()

    assert stats["r3/routing_valid_fraction"] == 1.0
    assert stats["r3/replay_fraction_active"] == 0.0
    assert stats["r3/skip_fraction_active"] == 0.0


def test_r3_export_without_routing_stats_preserves_other_metrics():
    tracker = DistributedStatsTracker()
    _, export = _collectors(tracker)
    tracker.scalar(loss=2.0)

    assert export() == {"loss": 2.0, "loss__count": 1}


def _check_distributed_stats(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        tracker = DistributedStatsTracker()
        record, export = _collectors(tracker, dist.group.WORLD)
        record(_counters(4, 4, 2, 2) if rank == 0 else _counters(6, 1, 3, 0))
        _assert_combined(export(), "r3/")

        if rank == 0:
            with tracker.scope("ppo_actor/update"):
                record(_counters(4, 4, 2, 2))
        stats = export()
        assert stats["ppo_actor/update/r3/samples_total"] == 4
        assert stats["ppo_actor/update/r3/batches_with_side_channel"] == 1
        assert stats["ppo_actor/update/r3/replay_fraction_active"] == 1.0
        assert stats["ppo_actor/update/r3/skip_fraction_active"] == 0.0
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo support")
def test_r3_stats_two_cpu_ranks_reduce_unequal_and_missing_populations(tmp_path):
    mp.spawn(
        _check_distributed_stats,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
