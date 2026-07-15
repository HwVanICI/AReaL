from __future__ import annotations

import pytest

from areal.utils import stats_tracker
from areal.workflow.rlvr import log_reward_metrics


@pytest.fixture(autouse=True)
def reset_stats_tracker():
    stats_tracker.export_all(reset=True)
    yield
    stats_tracker.export_all(reset=True)


def test_log_reward_metrics_records_domain_and_dataset_rewards():
    log_reward_metrics(
        0.75,
        {
            "domain": "math",
            "dataset_name": "gsm8k",
        },
    )

    stats = stats_tracker.export_all(reset=True)

    assert stats["rollout/reward"] == pytest.approx(0.75)
    assert stats["rollout/reward/math"] == pytest.approx(0.75)
    assert stats["rollout/reward_dataset/gsm8k"] == pytest.approx(0.75)


def test_log_reward_metrics_keeps_aggregate_reward_without_task_metadata():
    log_reward_metrics(0.25, {})

    stats = stats_tracker.export_all(reset=True)

    assert stats["rollout/reward"] == pytest.approx(0.25)
    assert not any(key.startswith("rollout/reward/") for key in stats)
    assert not any(key.startswith("rollout/reward_dataset/") for key in stats)
