import pytest
import torch

from areal.api import RolloutWorkflow
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_context import WorkflowContext, set as set_workflow_context
from areal.utils import logging


def _make_tensor_traj(batch_size: int = 1, seqlen: int = 4, reward: float = 1.0):
    input_ids = torch.arange(batch_size * seqlen, dtype=torch.int32).view(
        batch_size, seqlen
    )
    attention_mask = torch.ones(batch_size, seqlen, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, seqlen, dtype=torch.int32)
    loss_mask[:, -1] = 1
    logprobs = torch.zeros(batch_size, seqlen, dtype=torch.float32)
    versions = torch.zeros(batch_size, seqlen, dtype=torch.int32)
    rewards = torch.full((batch_size,), reward, dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "versions": versions,
        "rewards": rewards,
    }


class _SequenceWorkflow(RolloutWorkflow):
    def __init__(self, results):
        self._results = list(results)
        self._idx = 0

    async def arun_episode(self, engine, data):
        del engine, data
        result = self._results[self._idx]
        self._idx += 1
        return result


@pytest.mark.asyncio
async def test_grouped_rollout_drop_group_rejects_partial_results():
    set_workflow_context(WorkflowContext(task_id=17))
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([_make_tensor_traj(), None]),
        group_size=2,
        logger=logging.getLogger("TestGroupedRollout"),
        result_policy="drop_group",
        expected_samples_per_subrollout=1,
    )

    result = await grouped.arun_episode(engine=None, data={})
    assert result is None


@pytest.mark.asyncio
async def test_grouped_rollout_allow_partial_keeps_valid_results_and_adds_metadata():
    set_workflow_context(WorkflowContext(task_id=23))
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([_make_tensor_traj(), None]),
        group_size=2,
        logger=logging.getLogger("TestGroupedRollout"),
        result_policy="allow_partial",
        expected_samples_per_subrollout=1,
    )

    result = await grouped.arun_episode(engine=None, data={})

    assert result is not None
    assert result["input_ids"].shape[0] == 1
    assert torch.equal(result["group_ids"], torch.tensor([23], dtype=torch.long))
    assert torch.equal(
        result["group_expected_size"], torch.tensor([2], dtype=torch.long)
    )
    assert torch.equal(result["group_actual_size"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(result["group_is_complete"], torch.tensor([False]))


@pytest.mark.asyncio
async def test_grouped_rollout_drop_group_rejects_unexpected_subrollout_cardinality():
    set_workflow_context(WorkflowContext(task_id=31))
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([_make_tensor_traj(batch_size=2), _make_tensor_traj()]),
        group_size=2,
        logger=logging.getLogger("TestGroupedRollout"),
        result_policy="drop_group",
        expected_samples_per_subrollout=1,
    )

    result = await grouped.arun_episode(engine=None, data={})
    assert result is None


@pytest.mark.asyncio
async def test_grouped_rollout_allow_partial_preserves_variable_group_size():
    set_workflow_context(WorkflowContext(task_id=41))
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([_make_tensor_traj(batch_size=2), _make_tensor_traj()]),
        group_size=2,
        logger=logging.getLogger("TestGroupedRollout"),
        result_policy="allow_partial",
        expected_samples_per_subrollout=1,
    )

    result = await grouped.arun_episode(engine=None, data={})

    assert result is not None
    assert result["input_ids"].shape[0] == 3
    assert torch.equal(
        result["group_ids"], torch.tensor([41, 41, 41], dtype=torch.long)
    )
    assert torch.equal(
        result["group_expected_size"], torch.tensor([2, 2, 2], dtype=torch.long)
    )
    assert torch.equal(
        result["group_actual_size"], torch.tensor([3, 3, 3], dtype=torch.long)
    )
    assert torch.equal(result["group_is_complete"], torch.tensor([False, False, False]))
