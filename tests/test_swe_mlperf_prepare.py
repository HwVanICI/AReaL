"""Dataset identity, deployment routing and model-independent grader smoke checks."""

import hashlib
import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from examples.swe import check_r2e_env, prepare_mlperf_data


@pytest.mark.parametrize("select_task", [False, True])
def test_prepare_dataset_checks_source_before_mapping_or_selection(
    tmp_path, monkeypatch, select_task
):
    task_id = "owner__repo-" + "a" * 40
    instance = {
        "instance_id": task_id,
        "problem_statement": "fix",
        "base_commit": "b" * 40,
        "dataset_name": "R2E-Gym/R2E-Gym-Subset",
        "docker_image": "original",
    }
    metadata = {key: value for key, value in instance.items() if key != "docker_image"}
    metadata["instance_dict"] = json.dumps(instance)
    source = tmp_path / "official.jsonl"
    source.write_text(
        json.dumps({"responses_create_params": {"metadata": metadata}}) + "\n"
    )
    target = tmp_path / "prepared.jsonl"
    monkeypatch.setitem(
        prepare_mlperf_data.DATASETS,
        "test",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    assert (
        prepare_mlperf_data.prepare_dataset(
            source,
            target,
            "test",
            {task_id: "provisioned"},
            task_id if select_task else None,
        )
        == 1
    )
    row = json.loads(target.read_text())
    assert row["image"] == "provisioned"
    assert row["r2e_instance"]["docker_image"] == "original"
    with pytest.raises(ValueError, match="not found"):
        prepare_mlperf_data.prepare_dataset(source, target, "test", {}, "missing")
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_mlperf_data.prepare_dataset(
            source, target, "test", {task_id: "provisioned"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("gold_reward", [1.0, 0.0])
async def test_grader_sanity_uses_fresh_sandboxes_and_checks_gold(
    monkeypatch, gold_reward
):
    sandboxes = [AsyncMock(), AsyncMock()]
    enter = AsyncMock(side_effect=sandboxes)
    grade = AsyncMock(side_effect=[(gold_reward, "gold report"), (0.0, "empty report")])
    env = ModuleType("aweagent.envs.e2b")
    env._enter_e2b_sandbox = enter
    grader = ModuleType("aweagent.grading.r2e")
    grader.evaluate_r2e_patch = grade
    monkeypatch.setitem(sys.modules, "aweagent.envs.e2b", env)
    monkeypatch.setitem(sys.modules, "aweagent.grading.r2e", grader)

    task = {"image": "prepared", "r2e_instance": {}}
    if gold_reward:
        await check_r2e_env.check_task(task, "both")
        assert enter.await_count == 2
        assert grade.await_args_list[0].kwargs["mode"] == "gold"
        assert grade.await_args_list[1].kwargs["mode"] == "empty"
        sandboxes[1].__aexit__.assert_awaited_once()
    else:
        with pytest.raises(RuntimeError, match="gold: expected"):
            await check_r2e_env.check_task(task, "both")
    sandboxes[0].__aexit__.assert_awaited_once()
