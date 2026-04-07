from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from .aggregator import VersionPassSummary
from .metrics import parse_pass_metric_key, sort_pass_metric_keys


@dataclass(frozen=True)
class PassKCurve:
    label: str
    pass_metrics: dict[str, float]


def plot_pass_k_curves(
    curves: list[PassKCurve],
    output_path: str | Path,
) -> None:
    if not curves:
        raise ValueError("No pass@k curves available for plotting")

    output_path = Path(output_path)
    metric_keys = sort_pass_metric_keys(
        {
            key
            for curve in curves
            for key in curve.pass_metrics.keys()
        }
    )
    if not metric_keys:
        raise ValueError("No pass@k metrics available for plotting")

    xs = [parse_pass_metric_key(metric_key) for metric_key in metric_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in curves:
        ys = [curve.pass_metrics.get(metric_key, float("nan")) for metric_key in metric_keys]
        ax.plot(xs, ys, marker="o", label=curve.label)

    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Curve")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_version_summaries(
    summaries: list[VersionPassSummary],
    output_path: str | Path,
) -> None:
    if not summaries:
        raise ValueError("No version summaries available for plotting")

    curves = [
        PassKCurve(
            label=(
                f"tail_version={summary.tail_version}"
                if len(summaries) > 1
                else "pass@k"
            ),
            pass_metrics=summary.pass_metrics,
        )
        for summary in summaries
    ]
    plot_pass_k_curves(curves, output_path)
