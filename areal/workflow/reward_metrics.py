# SPDX-License-Identifier: Apache-2.0

from typing import Any

from areal.infra import workflow_context
from areal.utils import stats_tracker


def log_reward_metrics(reward: float, task_data: dict[str, Any]) -> None:
    """Log aggregate reward and an optional domain-specific reward."""
    tracker = stats_tracker.get(workflow_context.stat_scope())
    tracker.scalar(reward=reward)

    domain = task_data.get("domain")
    if domain:
        tracker.scalar(**{f"reward/{domain}": reward})
