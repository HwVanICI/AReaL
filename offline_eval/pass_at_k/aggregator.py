from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from areal.utils import logging

from .metrics import (
    aggregate_pass_at_k_over_dataset,
    compute_problem_level_pass_at_k,
    infer_successes,
)
from .reader import EvalRecord

logger = logging.getLogger("PassAtKAggregator")


@dataclass(frozen=True)
class TaskPassMetrics:
    tail_version: int
    task_id: int
    n_samples: int
    n_successes: int | None
    mean_reward: float
    pass_metrics: dict[str, float] = field(default_factory=dict)
    skipped_reason: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = {
            "tail_version": self.tail_version,
            "task_id": self.task_id,
            "n_samples": self.n_samples,
            "n_successes": self.n_successes,
            "mean_reward": self.mean_reward,
            "skipped_reason": self.skipped_reason,
        }
        row.update(self.pass_metrics)
        return row


@dataclass(frozen=True)
class VersionPassSummary:
    tail_version: int
    n_tasks_total: int
    n_tasks_used: int
    n_tasks_skipped: int
    avg_samples: float
    avg_reward: float
    pass_metrics: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "tail_version": self.tail_version,
            "n_tasks_total": self.n_tasks_total,
            "n_tasks_used": self.n_tasks_used,
            "n_tasks_skipped": self.n_tasks_skipped,
            "avg_samples": self.avg_samples,
            "avg_reward": self.avg_reward,
        }
        row.update(self.pass_metrics)
        return row


def group_records_by_version_and_task(
    records: list[EvalRecord],
) -> dict[tuple[int, int], list[EvalRecord]]:
    grouped: dict[tuple[int, int], list[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.tail_version, record.task_id)].append(record)
    return dict(grouped)


def _sort_group(records: list[EvalRecord]) -> list[EvalRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.sample_idx,
            record.line_no if record.line_no is not None else -1,
        ),
    )


def aggregate_task_metrics(
    records: list[EvalRecord],
    *,
    ks: list[int] | None = None,
    success_threshold: float | None = None,
    atol: float = 1e-6,
    strict: bool = True,
) -> list[TaskPassMetrics]:
    grouped = group_records_by_version_and_task(records)
    results: list[TaskPassMetrics] = []
    for (tail_version, task_id), group in sorted(grouped.items()):
        ordered_group = _sort_group(group)
        rewards = [record.reward for record in ordered_group]
        pass_metrics = compute_problem_level_pass_at_k(
            rewards,
            ks=ks,
            success_threshold=success_threshold,
            atol=atol,
        )
        if pass_metrics is None:
            message = (
                f"Task {task_id} at version {tail_version} has non-binary rewards "
                "and no success_threshold was provided."
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s Skipping task.", message)
            results.append(
                TaskPassMetrics(
                    tail_version=tail_version,
                    task_id=task_id,
                    n_samples=len(rewards),
                    n_successes=None,
                    mean_reward=sum(rewards) / len(rewards),
                    skipped_reason="non_binary_reward",
                )
            )
            continue

        successes = infer_successes(
            rewards,
            success_threshold=success_threshold,
            atol=atol,
        )
        assert successes is not None
        results.append(
            TaskPassMetrics(
                tail_version=tail_version,
                task_id=task_id,
                n_samples=len(rewards),
                n_successes=sum(successes),
                mean_reward=sum(rewards) / len(rewards),
                pass_metrics=pass_metrics,
            )
        )
    return results


def aggregate_version_summaries(
    task_metrics: list[TaskPassMetrics],
) -> list[VersionPassSummary]:
    grouped: dict[int, list[TaskPassMetrics]] = defaultdict(list)
    for metric in task_metrics:
        grouped[metric.tail_version].append(metric)

    summaries: list[VersionPassSummary] = []
    for tail_version, metrics in sorted(grouped.items()):
        used_metrics = [metric for metric in metrics if metric.skipped_reason is None]
        pass_metrics = aggregate_pass_at_k_over_dataset(
            [metric.pass_metrics for metric in used_metrics]
        )
        summaries.append(
            VersionPassSummary(
                tail_version=tail_version,
                n_tasks_total=len(metrics),
                n_tasks_used=len(used_metrics),
                n_tasks_skipped=len(metrics) - len(used_metrics),
                avg_samples=(
                    sum(metric.n_samples for metric in metrics) / len(metrics)
                    if metrics
                    else 0.0
                ),
                avg_reward=(
                    sum(metric.mean_reward for metric in metrics) / len(metrics)
                    if metrics
                    else 0.0
                ),
                pass_metrics=pass_metrics,
            )
        )
    return summaries
