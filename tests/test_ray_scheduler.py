# SPDX-License-Identifier: Apache-2.0

from areal.api.cli_args import SchedulingSpec
from areal.infra.scheduler.ray import RayScheduler


def _scheduler(n_gpus_per_node: int = 16) -> RayScheduler:
    scheduler = object.__new__(RayScheduler)
    scheduler._n_gpus_per_node = n_gpus_per_node
    scheduler.ray_device_resource = "GPU"
    return scheduler


def test_build_node_plan_large_training_keeps_contiguous_node_groups():
    scheduler = _scheduler(n_gpus_per_node=16)
    spec = SchedulingSpec(cpu=1, gpu=1, mem=1)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=64, spec=spec)

    assert nodes_per_worker == 1
    assert [int(bundle["GPU"]) for bundle in bundles] == [16, 16, 16, 16]
    assert [item["workers"] for item in plan] == [16, 16, 16, 16]


def test_build_node_plan_partial_single_node_role():
    scheduler = _scheduler(n_gpus_per_node=16)
    spec = SchedulingSpec(cpu=2, gpu=1, mem=3)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=12, spec=spec)

    assert nodes_per_worker == 1
    assert len(bundles) == 1
    assert int(bundles[0]["GPU"]) == 12
    assert bundles[0]["CPU"] == 24
    assert plan == [
        {"bundle_index": 0, "node_rank": 0, "workers": 12, "gpus_on_node": 12}
    ]


def test_build_node_plan_multi_node_instance_uses_head_and_worker_nodes():
    scheduler = _scheduler(n_gpus_per_node=16)
    spec = SchedulingSpec(cpu=32, gpu=32, mem=64)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=2, spec=spec)

    assert nodes_per_worker == 2
    assert [int(bundle["GPU"]) for bundle in bundles] == [16, 16, 16, 16]
    assert [
        (item["worker_idx"], item["node_rank"], item["workers"]) for item in plan
    ] == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]
