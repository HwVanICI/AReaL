from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def parse_pass_metric_key(metric_key: str) -> int:
    prefix = "pass@"
    if not metric_key.startswith(prefix):
        raise ValueError(f"Invalid pass@k metric key: {metric_key}")
    return int(metric_key[len(prefix) :])


def sort_pass_metric_keys(metric_keys: Iterable[str]) -> list[str]:
    return sorted(metric_keys, key=parse_pass_metric_key)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if num_correct < 0 or num_correct > num_samples:
        raise ValueError(
            f"num_correct must be in [0, num_samples], got {num_correct}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > num_samples:
        raise ValueError(
            f"k must be <= num_samples, got k={k}, num_samples={num_samples}"
        )
    if num_samples - num_correct < k:
        return 1.0

    prod = 1.0
    for denom in range(num_samples - num_correct + 1, num_samples + 1):
        prod *= 1.0 - (k / denom)
    return 1.0 - prod


def default_pass_at_ks(max_k: int) -> list[int]:
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    ks = []
    current = 1
    while current < max_k:
        ks.append(current)
        current *= 2
    ks.append(max_k)
    return ks


def normalize_pass_at_ks(ks: Sequence[int] | None, *, max_k: int) -> list[int]:
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    if ks is None:
        return default_pass_at_ks(max_k)

    normalized = sorted({int(k) for k in ks})
    if not normalized:
        raise ValueError("ks must not be empty")
    if normalized[0] <= 0:
        raise ValueError(f"ks must contain only positive integers, got {ks}")
    return [k for k in normalized if k <= max_k]


def infer_successes(
    rewards: Iterable[float],
    *,
    success_threshold: float | None = None,
    atol: float = 1e-6,
) -> list[bool] | None:
    reward_list = [float(reward) for reward in rewards]
    if not reward_list:
        raise ValueError("rewards must not be empty")

    if success_threshold is not None:
        return [reward >= success_threshold for reward in reward_list]

    successes: list[bool] = []
    for reward in reward_list:
        if math.isclose(reward, 0.0, abs_tol=atol):
            successes.append(False)
        elif math.isclose(reward, 1.0, abs_tol=atol):
            successes.append(True)
        else:
            return None
    return successes


def compute_problem_level_pass_at_k(
    rewards: Sequence[float],
    *,
    ks: Sequence[int] | None = None,
    success_threshold: float | None = None,
    atol: float = 1e-6,
) -> dict[str, float] | None:
    successes = infer_successes(
        rewards,
        success_threshold=success_threshold,
        atol=atol,
    )
    if successes is None:
        return None

    num_samples = len(successes)
    num_correct = sum(successes)
    metrics = {}
    for k in normalize_pass_at_ks(ks, max_k=num_samples):
        metrics[f"pass@{k}"] = estimate_pass_at_k(num_samples, num_correct, k)
    return metrics


def aggregate_pass_at_k_over_dataset(
    per_problem_metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    if not per_problem_metrics:
        return {}

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in per_problem_metrics:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: totals[key] / counts[key]
        for key in sort_pass_metric_keys(totals.keys())
        if counts[key] > 0
    }
