from examples.swe.dataset import get_swe_dataset, resolve_swe_dataset_path


def _write_dataset(path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"instance_id":"task-1","problem_statement":"one"}',
                '{"instance_id":"task-2","problem_statement":"two"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_get_swe_dataset_without_min_items_preserves_source_rows(tmp_path):
    dataset_path = tmp_path / "swe.jsonl"
    _write_dataset(dataset_path)

    dataset = get_swe_dataset(str(dataset_path))

    assert len(dataset) == 2
    assert list(dataset["instance_id"]) == ["task-1", "task-2"]


def test_get_swe_dataset_with_min_items_repeats_source_order(tmp_path):
    dataset_path = tmp_path / "swe.jsonl"
    _write_dataset(dataset_path)

    dataset = get_swe_dataset(str(dataset_path), min_items=3)

    assert len(dataset) == 4
    assert list(dataset["instance_id"]) == [
        "task-1",
        "task-2",
        "task-1",
        "task-2",
    ]


def test_resolve_swe_dataset_path_uses_existing_dataset_root(tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = dataset_root / "swe.jsonl"
    _write_dataset(dataset_path)

    resolved = resolve_swe_dataset_path("swe.jsonl", str(dataset_root))

    assert resolved == str(dataset_path)
