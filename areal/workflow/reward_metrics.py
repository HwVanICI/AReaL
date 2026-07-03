# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.infra import workflow_context
from areal.utils import stats_tracker


def log_reward_metrics(
    reward: float,
    task_data: dict[str, Any],
    n_seqs: int = 1,
) -> None:
    """Log aggregate reward and optional domain-specific reward and count."""
    if n_seqs < 1:
        raise ValueError(f"n_seqs must be positive, got {n_seqs}")

    tracker = stats_tracker.get(workflow_context.stat_scope())
    tracker.scalar(reward=reward)

    domain = task_data.get("domain")
    if domain:
        tracker.scalar(**{f"reward/{domain}": reward})
        tracker.denominator(
            **{
                f"n_seqs/{domain}": torch.ones(
                    n_seqs,
                    dtype=torch.bool,
                )
            }
        )
