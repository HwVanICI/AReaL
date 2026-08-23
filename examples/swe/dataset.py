"""Dataset loading helpers shared by SWE training and rollout scripts."""

import json
from pathlib import Path

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWEDataset")


def resolve_swe_dataset_path(path: str, dataset_root: str = "") -> str:
    """Resolve a dataset path relative to an optional SWE dataset root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return path
    if dataset_root:
        rooted = Path(dataset_root) / candidate
        if rooted.exists():
            return str(rooted)
    return path


def get_swe_dataset(
    dataset_path: str,
    split: str = "train",
    min_items: int | None = None,
) -> Dataset:
    """Create a Hugging Face dataset from SWE-bench-style JSONL.

    Each valid row must contain ``instance_id`` and ``problem_statement``.
    Other fields, including E2B image routing and evaluation metadata, are
    preserved unchanged. When ``min_items`` is set, the input rows are repeated
    in source order until the dataset contains at least that many items.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench dataset not found: {dataset_path}")

    dataset_items = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "instance_id" not in item:
                logger.warning(f"Skipping item missing 'instance_id': {line[:100]}")
                continue
            if "problem_statement" not in item:
                logger.warning(
                    "Skipping item missing 'problem_statement': "
                    f"{item.get('instance_id')}"
                )
                continue
            dataset_items.append(item)

    if not dataset_items:
        raise ValueError(f"No valid items found in dataset: {dataset_path}")

    if min_items is not None and len(dataset_items) < min_items:
        original_items = dataset_items.copy()
        while len(dataset_items) < min_items:
            dataset_items.extend(original_items)

    dataset = Dataset.from_list(dataset_items)
    logger.info(
        f"Created SWE dataset with {len(dataset)} items "
        f"from {dataset_path} (split={split})"
    )
    return dataset
