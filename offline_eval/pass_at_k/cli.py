from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

from .aggregator import (
    aggregate_task_metrics,
    aggregate_version_summaries,
)
from .metrics import parse_pass_metric_key
from .plotting import PassKCurve, plot_pass_k_curves
from .reader import load_eval_records


def _parse_ks(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--ks must not be empty")
    return [int(item) for item in items]


def _parse_labels(value: str | None, *, expected_count: int) -> list[str] | None:
    if value is None:
        return None
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if len(labels) != expected_count:
        raise ValueError(
            f"--labels must provide exactly {expected_count} labels, got {len(labels)}"
        )
    return labels


def _field_sort_key(field_name: str) -> tuple[int, Any]:
    if field_name.startswith("pass@"):
        return (1, parse_pass_metric_key(field_name))
    return (0, field_name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(
        {key for row in rows for key in row.keys()},
        key=_field_sort_key,
    )
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def _default_source_label(source: str) -> str:
    path = Path(source).expanduser()
    if path.exists():
        if path.is_file():
            return path.stem
        if path.name in {"eval-rollout", "rollout"} and path.parent.name:
            return path.parent.name
        if path.name:
            return path.name
    return source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate eval-rollout dump into pass@k metrics.")
    parser.add_argument(
        "--source",
        required=True,
        nargs="+",
        help="One or more JSONL files, directories, or globs to read.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels for --source entries when plotting or combining outputs.",
    )
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated k values, e.g. 1,2,4,8. Defaults to powers of two up to n_samples.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Treat reward >= threshold as success for non-binary rewards.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write summary artifacts into.",
    )
    parser.add_argument("--emit-csv", action="store_true", help="Write summary.csv.")
    parser.add_argument("--emit-json", action="store_true", help="Write summary.json.")
    parser.add_argument(
        "--emit-task-csv",
        action="store_true",
        help="Write task_metrics.csv with per-task details.",
    )
    parser.add_argument("--plot", action="store_true", help="Write pass_at_k.png.")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on malformed records or non-binary rewards without success_threshold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ks = _parse_ks(args.ks)
    labels = _parse_labels(args.labels, expected_count=len(args.source))
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    emit_csv = args.emit_csv
    emit_json = args.emit_json
    emit_task_csv = args.emit_task_csv
    if output_dir is not None and not (emit_csv or emit_json or emit_task_csv or args.plot):
        emit_csv = True
        emit_json = True

    all_summary_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    curves: list[PassKCurve] = []
    multi_source = len(args.source) > 1

    for idx, source in enumerate(args.source):
        label = labels[idx] if labels is not None else _default_source_label(source)
        records = load_eval_records(source, strict=args.strict)
        task_metrics = aggregate_task_metrics(
            records,
            ks=ks,
            success_threshold=args.success_threshold,
            strict=args.strict,
        )
        summaries = aggregate_version_summaries(task_metrics)

        for summary in summaries:
            row = summary.to_row()
            if multi_source:
                row = {"source": label, **row}
            all_summary_rows.append(row)

        for metric in task_metrics:
            row = metric.to_row()
            if multi_source:
                row = {"source": label, **row}
            all_task_rows.append(row)

        if args.plot:
            if len(summaries) == 1:
                curve_label = label
                curves.append(
                    PassKCurve(
                        label=curve_label,
                        pass_metrics=summaries[0].pass_metrics,
                    )
                )
            else:
                for summary in summaries:
                    curves.append(
                        PassKCurve(
                            label=f"{label}@v{summary.tail_version}",
                            pass_metrics=summary.pass_metrics,
                        )
                    )

    if all_summary_rows:
        print(
            tabulate(all_summary_rows, headers="keys", tablefmt="github", floatfmt=".6f")
        )
    else:
        print("No version summaries were produced.", file=sys.stderr)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if emit_csv:
            _write_csv(output_dir / "summary.csv", all_summary_rows)
        if emit_json:
            _write_json(output_dir / "summary.json", all_summary_rows)
        if emit_task_csv:
            _write_csv(output_dir / "task_metrics.csv", all_task_rows)
        if args.plot:
            plot_pass_k_curves(curves, output_dir / "pass_at_k.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
