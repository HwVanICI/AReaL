from __future__ import annotations

import pytest

from areal.experimental.openai.proxy.workflow import (
    _log_interaction_reward_metrics,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import stats_tracker
from areal.workflow.rlvr import log_reward_metrics


@pytest.fixture(autouse=True)
def reset_stats_tracker():
    stats_tracker.export_all(reset=True)
    yield
    stats_tracker.export_all(reset=True)


def test_log_reward_metrics_records_domain_reward():
    log_reward_metrics(
        0.75,
        {
            "domain": "math",
        },
    )

    stats = stats_tracker.export_all(reset=True)

    assert stats["rollout/reward"] == pytest.approx(0.75)
    assert stats["rollout/reward/math"] == pytest.approx(0.75)
    assert stats["rollout/n_seqs/math"] == 1


def test_log_reward_metrics_keeps_aggregate_reward_without_task_metadata():
    log_reward_metrics(0.25, {})

    stats = stats_tracker.export_all(reset=True)

    assert stats["rollout/reward"] == pytest.approx(0.25)
    assert not any(key.startswith("rollout/reward/") for key in stats)


def test_agent_reward_metrics_records_domain_reward():
    interactions = {
        "response-1": InteractionWithTokenLogpReward(reward=0.5),
        "response-2": InteractionWithTokenLogpReward(reward=1.0),
    }

    _log_interaction_reward_metrics(interactions, {"domain": "leetcode"})

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/reward"] == pytest.approx(1.0)
    assert stats["rollout/reward/leetcode"] == pytest.approx(1.0)
    assert stats["rollout/n_seqs/leetcode"] == 2
