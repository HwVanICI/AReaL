from __future__ import annotations

import pytest
from datasets import Dataset

from areal.api.cli_args import DatasetSourceConfig, TrainDatasetConfig
from areal.dataset import get_custom_dataset


def test_get_custom_dataset_with_sources_tags_domain_and_dataset_name(monkeypatch):
    """Mixed dataset sources are tagged for workflow routing."""

    def _fake_get_custom_dataset(**kwargs):
        return Dataset.from_list([{"text": kwargs["path"]}])

    monkeypatch.setenv("AREAL_SPMD_MODE", "1")
    monkeypatch.setattr("areal.dataset._get_custom_dataset", _fake_get_custom_dataset)

    cfg = TrainDatasetConfig(
        path=None,
        type=None,
        scheduling_spec=None,
        datasets=[
            DatasetSourceConfig(
                name="gsm8k",
                domain="math",
                path="math-path",
                type="rl",
            ),
            DatasetSourceConfig(
                name="coding_tasks",
                domain="coding",
                path="coding-path",
                type="rl",
            ),
        ],
    )

    dataset = get_custom_dataset(split="train", dataset_config=cfg)

    assert len(dataset) == 2
    assert dataset[0]["domain"] == "math"
    assert dataset[0]["dataset_name"] == "gsm8k"
    assert dataset[1]["domain"] == "coding"
    assert dataset[1]["dataset_name"] == "coding_tasks"


def test_get_custom_dataset_with_sources_preserves_heterogeneous_columns(monkeypatch):
    """Mixed datasets can combine domains with different sample schemas."""

    def _fake_get_custom_dataset(**kwargs):
        if kwargs["path"] == "math-path":
            return Dataset.from_list([{"question": "1+1?", "answer": "2"}])
        return Dataset.from_list([{"repo": "demo", "instruction": "fix test"}])

    monkeypatch.setenv("AREAL_SPMD_MODE", "1")
    monkeypatch.setattr("areal.dataset._get_custom_dataset", _fake_get_custom_dataset)

    cfg = TrainDatasetConfig(
        path=None,
        type=None,
        scheduling_spec=None,
        datasets=[
            DatasetSourceConfig(
                name="gsm8k",
                domain="math",
                path="math-path",
                type="rl",
            ),
            DatasetSourceConfig(
                name="coding_tasks",
                domain="coding",
                path="coding-path",
                type="rl",
            ),
        ],
    )

    dataset = get_custom_dataset(split="train", dataset_config=cfg)

    assert dataset[0]["question"] == "1+1?"
    assert dataset[0]["repo"] is None
    assert dataset[1]["repo"] == "demo"
    assert dataset[1]["question"] is None


def test_get_custom_dataset_with_sources_uses_source_split_override(monkeypatch):
    """Source split and kwargs override parent mixed dataset settings."""

    calls = []

    def _fake_get_custom_dataset(**kwargs):
        calls.append(kwargs)
        return Dataset.from_list([{"x": kwargs["path"]}])

    monkeypatch.setenv("AREAL_SPMD_MODE", "1")
    monkeypatch.setattr("areal.dataset._get_custom_dataset", _fake_get_custom_dataset)

    cfg = TrainDatasetConfig(
        path=None,
        type=None,
        max_length=128,
        dataset_kwargs={"revision": "parent", "common": "yes"},
        scheduling_spec=None,
        datasets=[
            DatasetSourceConfig(
                name="default_split",
                domain="math",
                path="math-path",
                type="rl",
            ),
            DatasetSourceConfig(
                name="override_split",
                domain="coding",
                path="coding-path",
                type="rl",
                split="validation",
                max_length=256,
                dataset_kwargs={"revision": "v1"},
            ),
        ],
    )

    get_custom_dataset(split="train", dataset_config=cfg)

    assert calls[0]["split"] == "train"
    assert calls[0]["max_length"] == 128
    assert calls[0]["revision"] == "parent"
    assert calls[0]["common"] == "yes"
    assert calls[1]["split"] == "validation"
    assert calls[1]["max_length"] == 256
    assert calls[1]["revision"] == "v1"
    assert calls[1]["common"] == "yes"


def test_get_custom_dataset_with_sources_rejects_remote_rdataset(monkeypatch):
    """Remote data service mixtures are intentionally not supported yet."""

    monkeypatch.setenv("AREAL_SPMD_MODE", "0")

    cfg = TrainDatasetConfig(
        path=None,
        type=None,
        datasets=[
            DatasetSourceConfig(
                name="gsm8k",
                domain="math",
                path="math-path",
                type="rl",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Remote RDataset loading does not support"):
        get_custom_dataset(split="train", dataset_config=cfg)
