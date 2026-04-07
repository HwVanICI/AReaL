from __future__ import annotations

import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("PassAtKReader")


@dataclass(frozen=True)
class EvalRecord:
    task_id: int
    sample_idx: int
    reward: float
    tail_version: int
    prompt: str | None = None
    completion: str | None = None
    head_version: int | None = None
    source_file: str | None = None
    line_no: int | None = None


def resolve_record_files(source: str | Path) -> list[Path]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_file():
            return [source_path]
        if source_path.is_dir():
            return sorted(p for p in source_path.rglob("*.jsonl") if p.is_file())
    matches = [Path(p).expanduser() for p in glob(str(source_path), recursive=True)]
    return sorted(p for p in matches if p.is_file())


def _require_field(
    payload: dict[str, Any],
    field_name: str,
    *,
    source_file: Path,
    line_no: int,
) -> Any:
    if field_name not in payload:
        raise ValueError(
            f"Missing required field `{field_name}` in {source_file}:{line_no}"
        )
    return payload[field_name]


def _parse_record(
    payload: dict[str, Any],
    *,
    source_file: Path,
    line_no: int,
) -> EvalRecord:
    return EvalRecord(
        task_id=int(_require_field(payload, "task_id", source_file=source_file, line_no=line_no)),
        sample_idx=int(
            _require_field(payload, "sample_idx", source_file=source_file, line_no=line_no)
        ),
        reward=float(_require_field(payload, "reward", source_file=source_file, line_no=line_no)),
        tail_version=int(
            _require_field(payload, "tail_version", source_file=source_file, line_no=line_no)
        ),
        prompt=payload.get("prompt"),
        completion=payload.get("completion"),
        head_version=(
            int(payload["head_version"]) if payload.get("head_version") is not None else None
        ),
        source_file=str(source_file),
        line_no=line_no,
    )


def load_eval_records(
    source: str | Path,
    *,
    strict: bool = True,
) -> list[EvalRecord]:
    files = resolve_record_files(source)
    if not files:
        raise FileNotFoundError(f"No jsonl files found from source: {source}")

    records: list[EvalRecord] = []
    for path in files:
        with path.open("r", encoding="utf-8") as fin:
            for line_no, raw_line in enumerate(fin, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    record = _parse_record(
                        payload,
                        source_file=path,
                        line_no=line_no,
                    )
                except Exception as exc:
                    if strict:
                        raise
                    logger.warning("Skipping invalid record at %s:%d: %s", path, line_no, exc)
                    continue
                records.append(record)
    return records
