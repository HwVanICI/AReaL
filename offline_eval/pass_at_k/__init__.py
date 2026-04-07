"""Offline pass@k aggregation helpers."""

from .aggregator import (
    TaskPassMetrics,
    VersionPassSummary,
    aggregate_task_metrics,
    aggregate_version_summaries,
    group_records_by_version_and_task,
)
from .metrics import (
    compute_problem_level_pass_at_k,
    default_pass_at_ks,
    estimate_pass_at_k,
    infer_successes,
    normalize_pass_at_ks,
)
from .reader import EvalRecord, load_eval_records, resolve_record_files

__all__ = [
    "EvalRecord",
    "TaskPassMetrics",
    "VersionPassSummary",
    "aggregate_task_metrics",
    "aggregate_version_summaries",
    "group_records_by_version_and_task",
    "load_eval_records",
    "resolve_record_files",
    "compute_problem_level_pass_at_k",
    "default_pass_at_ks",
    "estimate_pass_at_k",
    "infer_successes",
    "normalize_pass_at_ks",
]
