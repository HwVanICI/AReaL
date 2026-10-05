"""Dataset loading helpers shared by SWE training and rollout scripts."""

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWEDataset")


def normalize_swe_row(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize an official NeMo Gym row, leaving ordinary SWE rows unchanged.

    ``r2e_instance`` and the original metadata are private harness/grader data,
    not model prompt content. Only ``problem_statement`` supplies the task text;
    gold patches, expected outputs, and source prompts are never promoted.
    """
    agent_ref = item.get("agent_ref")
    is_gym_agent = (
        isinstance(agent_ref, dict) and agent_ref.get("type") == "responses_api_agents"
    )
    if "responses_create_params" not in item and not is_gym_agent:
        return item

    params = item.get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError("Official SWE row: responses_create_params must be an object")
    metadata = params.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "Official SWE row: responses_create_params.metadata must be an object"
        )

    serialized = metadata.get("instance_dict")
    if not isinstance(serialized, str):
        raise ValueError(
            "Official SWE row: metadata.instance_dict must be a JSON string"
        )
    try:
        instance = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Official SWE row: metadata.instance_dict is not valid JSON"
        ) from exc
    if not isinstance(instance, dict):
        raise ValueError(
            "Official SWE row: metadata.instance_dict must decode to an object"
        )

    for field in ("instance_id", "problem_statement", "base_commit", "dataset_name"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Official SWE row: metadata.{field} must be a nonempty string"
            )
        if instance.get(field) != value:
            raise ValueError(
                f"Official SWE row: instance_dict.{field} does not match metadata.{field}"
            )
        if field in item and item[field] != value:
            raise ValueError(
                f"Official SWE row: top-level {field} does not match metadata.{field}"
            )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", metadata["instance_id"]):
        raise ValueError("Official SWE row: instance_id must be a shell-safe filename")
    if "R2E-Gym" not in metadata["dataset_name"]:
        raise ValueError(
            "Official SWE row: dataset_name must identify an R2E-Gym dataset"
        )
    docker_image = instance.get("docker_image")
    if not isinstance(docker_image, str) or not docker_image.strip():
        raise ValueError(
            "Official SWE row: instance_dict.docker_image must be a nonempty string"
        )

    # Keep gold-bearing fields nested for the trusted harness, never in task text.
    normalized = item.copy()
    normalized.update(
        instance_id=metadata["instance_id"],
        problem_statement=metadata["problem_statement"],
        base_commit=metadata["base_commit"],
        dataset_name=metadata["dataset_name"],
        r2e_instance=instance,
        image=item.get("image") or docker_image,
        workdir="/testbed",
    )
    return normalized


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
    """Create a Hugging Face dataset from ordinary SWE or official NeMo Gym JSONL.

    Each ordinary row must contain ``instance_id`` and ``problem_statement``.
    Official rows are normalized and validated by :func:`normalize_swe_row`;
    malformed official rows raise instead of being skipped.
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
            item = normalize_swe_row(json.loads(line))
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
