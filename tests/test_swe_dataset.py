import json

import pytest

from examples.swe.dataset import (
    get_swe_dataset,
    normalize_swe_row,
    resolve_swe_dataset_path,
)


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


def _official_row(index: int = 1) -> dict:
    commit = f"{index:040x}"
    instance_id = f"example-owner__example-repo-{commit}"
    instance = {
        "instance_id": instance_id,
        "problem_statement": f"Fix synthetic issue {index}",
        "base_commit": "b" * 40,
        "dataset_name": "R2E-Gym/R2E-Gym-Subset",
        "repo": "example-owner/example-repo",
        "docker_image": f"registry.example/r2e/example-repo_final:{commit}",
        "container_formatter": f"/synthetic-images/example-{index}.sif",
        "container_id": index,
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "prompt": "PRIVATE_GOLD_SOURCE_PROMPT",
        "patch": "PRIVATE_GOLD_PATCH",
        "parsed_commit_content": json.dumps({"diff": "PRIVATE_GOLD_DIFF"}),
        "execution_result_content": "PRIVATE_GOLD_EXECUTION",
        "expected_output_json": json.dumps({"PRIVATE_GOLD_TEST": "PASSED"}),
        "custom_harness_field": {"value": index},
    }
    metadata = {
        key: instance[key]
        for key in (
            "instance_id",
            "problem_statement",
            "base_commit",
            "dataset_name",
            "repo",
        )
    }
    metadata["instance_dict"] = json.dumps(instance)
    metadata["split"] = "train"
    return {
        "responses_create_params": {
            "input": [],
            "metadata": metadata,
            "model": "synthetic-model",
            "temperature": 1.0,
        },
        "agent_ref": {"type": "responses_api_agents", "name": "synthetic-agent"},
        "source_index": index,
    }


def _write_rows(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_get_swe_dataset_official_rows_preserves_order_and_harness_fields(tmp_path):
    """Official rows retain their payload while exposing safe task/routing fields."""
    rows = [_official_row(2), _official_row(1), _official_row(3)]
    dataset_path = tmp_path / "official.jsonl"
    _write_rows(dataset_path, rows)

    dataset = get_swe_dataset(str(dataset_path))

    assert len(dataset) == len(rows)
    assert list(dataset["source_index"]) == [2, 1, 3]
    for actual, original in zip(dataset, rows, strict=True):
        metadata = original["responses_create_params"]["metadata"]
        instance = json.loads(metadata["instance_dict"])
        for key, value in original.items():
            assert actual[key] == value
        for key in ("instance_id", "problem_statement", "base_commit", "dataset_name"):
            assert actual[key] == metadata[key]
        assert actual["r2e_instance"] == instance
        assert actual["image"] == instance["docker_image"]
        assert not actual["image"].endswith(".sif")
        assert actual["workdir"] == "/testbed"
        assert "PRIVATE_GOLD" not in actual["problem_statement"]
        for gold_field in (
            "prompt",
            "patch",
            "parsed_commit_content",
            "execution_result_content",
            "expected_output_json",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ):
            assert gold_field not in actual
            assert actual["r2e_instance"][gold_field] == instance[gold_field]


def test_normalize_swe_row_official_input_is_not_mutated():
    """Normalization is additive and does not alter the source row."""
    row = _official_row()
    original = json.loads(json.dumps(row))

    normalized = normalize_swe_row(row)

    assert row == original
    assert normalized is not row
    assert normalized["responses_create_params"] == original["responses_create_params"]


def test_normalize_swe_row_preserves_provisioned_image():
    row = _official_row()
    row["image"] = "registry.example/provisioned-runtime:1"

    assert normalize_swe_row(row)["image"] == row["image"]


def test_get_swe_dataset_official_min_items_repeats_source_order(tmp_path):
    """Official rows use the same whole-source repetition as ordinary rows."""
    dataset_path = tmp_path / "official.jsonl"
    _write_rows(dataset_path, [_official_row(2), _official_row(1)])

    dataset = get_swe_dataset(str(dataset_path), min_items=3)

    assert len(dataset) == 4
    assert list(dataset["source_index"]) == [2, 1, 2, 1]


@pytest.mark.parametrize("serialized", ["{broken", "null", "[]", '"text"', 7, {}])
def test_get_swe_dataset_invalid_official_instance_json_raises(tmp_path, serialized):
    """Broken or non-object instance JSON cannot silently remove a source row."""
    row = _official_row()
    row["responses_create_params"]["metadata"]["instance_dict"] = serialized
    dataset_path = tmp_path / "official.jsonl"
    _write_rows(dataset_path, [_official_row(2), row, _official_row(3)])

    with pytest.raises(ValueError, match="instance_dict"):
        get_swe_dataset(str(dataset_path))


@pytest.mark.parametrize(
    "field", ["instance_id", "problem_statement", "base_commit", "dataset_name"]
)
@pytest.mark.parametrize("location", ["metadata", "instance_dict", "top-level"])
def test_get_swe_dataset_official_inconsistent_fields_raises(tmp_path, field, location):
    """Conflicting IDs, task text, and grading identity are rejected."""
    row = _official_row()
    metadata = row["responses_create_params"]["metadata"]
    if location == "metadata":
        metadata[field] = "different"
    elif location == "instance_dict":
        instance = json.loads(metadata["instance_dict"])
        instance[field] = "different"
        metadata["instance_dict"] = json.dumps(instance)
    else:
        row[field] = "different"
    dataset_path = tmp_path / "official.jsonl"
    _write_rows(dataset_path, [_official_row(2), row])

    with pytest.raises(ValueError, match=field):
        get_swe_dataset(str(dataset_path))


@pytest.mark.parametrize(
    "instance_id",
    [
        "../owner__repo-" + "a" * 40,
        "owner/repo-" + "a" * 40,
        "owner__repo-" + "a" * 40 + "\n",
        "owner__repo-" + "a" * 40 + ";command",
    ],
)
def test_normalize_swe_row_unsafe_official_id_raises(instance_id):
    """Matching metadata alone does not make an unsafe ID acceptable."""
    row = _official_row()
    metadata = row["responses_create_params"]["metadata"]
    instance = json.loads(metadata["instance_dict"])
    metadata["instance_id"] = instance["instance_id"] = instance_id
    metadata["instance_dict"] = json.dumps(instance)

    with pytest.raises(ValueError, match="instance_id"):
        normalize_swe_row(row)


@pytest.mark.parametrize(
    "instance_id", ["task-1", "owner__repo-short", "owner__repo-" + "A" * 40]
)
def test_normalize_swe_row_shell_safe_official_id_preserved(instance_id):
    """Normalization does not impose an ID naming convention beyond shell safety."""
    row = _official_row()
    metadata = row["responses_create_params"]["metadata"]
    instance = json.loads(metadata["instance_dict"])
    metadata["instance_id"] = instance["instance_id"] = instance_id
    metadata["instance_dict"] = json.dumps(instance)

    normalized = normalize_swe_row(row)

    assert normalized["instance_id"] == instance_id
    assert normalized["r2e_instance"]["instance_id"] == instance_id


@pytest.mark.parametrize(
    "field",
    ["instance_id", "problem_statement", "base_commit", "dataset_name", "docker_image"],
)
@pytest.mark.parametrize("value", [None, "", " \t", 42])
def test_normalize_swe_row_missing_required_official_values_raises(field, value):
    """Required official strings cannot be null, blank, or non-string."""
    row = _official_row()
    metadata = row["responses_create_params"]["metadata"]
    instance = json.loads(metadata["instance_dict"])
    instance[field] = value
    if field != "docker_image":
        metadata[field] = value
    metadata["instance_dict"] = json.dumps(instance)

    with pytest.raises(ValueError, match=field):
        normalize_swe_row(row)


def test_normalize_swe_row_non_r2e_dataset_raises():
    """An otherwise valid NeMo row must belong to R2E-Gym."""
    row = _official_row()
    metadata = row["responses_create_params"]["metadata"]
    instance = json.loads(metadata["instance_dict"])
    metadata["dataset_name"] = instance["dataset_name"] = "other/dataset"
    metadata["instance_dict"] = json.dumps(instance)

    with pytest.raises(ValueError, match="R2E-Gym"):
        normalize_swe_row(row)


@pytest.mark.parametrize(
    "location, field",
    [
        ("row", "responses_create_params"),
        ("params", "metadata"),
        ("metadata", "instance_dict"),
        *[
            ("metadata", key)
            for key in (
                "instance_id",
                "problem_statement",
                "base_commit",
                "dataset_name",
            )
        ],
        *[
            ("instance", key)
            for key in (
                "instance_id",
                "problem_statement",
                "base_commit",
                "dataset_name",
                "docker_image",
            )
        ],
    ],
)
def test_get_swe_dataset_partial_official_row_raises(tmp_path, location, field):
    """A partial official row must fail even when valid rows remain."""
    row = _official_row()
    params = row["responses_create_params"]
    metadata = params["metadata"]
    instance = json.loads(metadata["instance_dict"])
    target = {"row": row, "params": params, "metadata": metadata, "instance": instance}[
        location
    ]
    target.pop(field)
    if location == "instance":
        metadata["instance_dict"] = json.dumps(instance)
    dataset_path = tmp_path / "partial.jsonl"
    _write_rows(dataset_path, [_official_row(2), row])

    with pytest.raises(ValueError, match=field):
        get_swe_dataset(str(dataset_path))


@pytest.mark.parametrize("value", [None, [], "not an object"])
@pytest.mark.parametrize("field", ["responses_create_params", "metadata"])
def test_get_swe_dataset_malformed_official_envelope_raises(tmp_path, field, value):
    """Malformed envelopes are detected before ordinary-row skip behavior."""
    row = _official_row()
    target = (
        row if field == "responses_create_params" else row["responses_create_params"]
    )
    target[field] = value
    dataset_path = tmp_path / "malformed.jsonl"
    _write_rows(dataset_path, [_official_row(2), row])

    with pytest.raises(ValueError, match=field):
        get_swe_dataset(str(dataset_path))


def test_get_swe_dataset_ordinary_rows_preserves_fields_and_skips_incomplete(tmp_path):
    """Ordinary SWE rows keep their existing routing, metadata, and skip behavior."""
    ordinary = {
        "instance_id": "task-1",
        "problem_statement": "one",
        "image": "ordinary-image",
        "workdir": "/ordinary-repo",
        "base_commit": "ordinary-base",
        "custom_metadata": {"value": 1},
    }
    dataset_path = tmp_path / "ordinary.jsonl"
    _write_rows(
        dataset_path,
        [
            {"problem_statement": "missing id"},
            ordinary,
            {"instance_id": "missing-task"},
        ],
    )

    dataset = get_swe_dataset(str(dataset_path))

    assert list(dataset) == [ordinary]
    assert normalize_swe_row(ordinary) is ordinary
