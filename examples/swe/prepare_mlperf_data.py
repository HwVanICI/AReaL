"""Select official MLPerf R2E tasks and map them to provisioned E2B images."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.swe.dataset import normalize_swe_row

DATASETS = {
    "train": "e772af09599270ab16a04ccbbcef395bcb2929607f050478033a84b087a606fc",
    "validation": "2d5bddf717afb4389510a7d197f26ce5eb7f610aab4b20d6af6e0150de6f2c79",
}


def prepare_dataset(
    source: Path,
    target: Path,
    split: str,
    image_map: dict[str, str],
    instance_id: str | None = None,
) -> int:
    """Verify the original artifact before selecting or changing image routing."""
    expected_hash = DATASETS[split]
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Official {split} dataset SHA-256 mismatch: {digest}")
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = normalize_swe_row(json.loads(line))
            if instance_id is None or row["instance_id"] == instance_id:
                image = image_map.get(row["instance_id"])
                if not isinstance(image, str) or not image:
                    raise ValueError(
                        f"Missing provisioned image for {row['instance_id']}"
                    )
                row["image"] = image
                rows.append(row)
    if instance_id is not None and not rows:
        raise ValueError(
            f"Task {instance_id} not found in the official {split} dataset"
        )
    if source.resolve() == target.resolve():
        raise ValueError("The prepared dataset must not overwrite the official source")
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=DATASETS, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    image_map = json.loads(args.image_map.read_text())
    prepare_dataset(args.dataset, args.output, args.split, image_map, args.instance_id)


if __name__ == "__main__":
    main()
