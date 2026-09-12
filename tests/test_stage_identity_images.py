import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.stage_identity_images import promote_mapping, stage_images


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 12, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_stage_deduplicates_and_preserves_mapping_template(tmp_path):
    segmentation = tmp_path / "segmentation_export" / "train"
    keypoint = tmp_path / "keypoint_export" / "valid"
    _write_image(segmentation / "one.jpg", (10, 20, 30))
    _write_image(keypoint / "same.jpg", (10, 20, 30))
    _write_image(keypoint / "two.jpg", (40, 50, 60))
    output = tmp_path / "identity_data"

    first = stage_images([str(tmp_path / "segmentation_export"),
                          str(tmp_path / "keypoint_export")], str(output))
    assert first["image_count"] == 2
    assert first["copied"] == 2
    assert first["duplicate_exact"] == 1
    template = output / "_review" / "unassigned" / "identity_mapping_template.csv"
    assert template.exists()

    template.write_text(template.read_text(encoding="utf-8") + "", encoding="utf-8")
    second = stage_images([str(tmp_path / "segmentation_export"),
                           str(tmp_path / "keypoint_export")], str(output))
    assert second["already_present"] == 2
    assert second["template"] == "preserved_existing"


def test_promote_mapping_copies_only_explicit_assignments(tmp_path):
    source = tmp_path / "segmentation_export" / "train"
    _write_image(source / "one.jpg", (10, 20, 30))
    output = tmp_path / "identity_data"
    stage_images([str(tmp_path / "segmentation_export")], str(output))
    manifest = json.loads((output / "_review" / "unassigned" /
                           "direct_closeups_manifest.json").read_text())
    staged_path = next(row["staged_path"] for row in manifest["records"]
                       if "staged_path" in row)
    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["staged_path", "dog_id", "target_split"])
        writer.writeheader()
        writer.writerow({"staged_path": staged_path, "dog_id": "dog_0001",
                         "target_split": "train"})

    result = promote_mapping(str(mapping), str(output))
    assert result["assigned_images"] == 1
    assert (output / "train" / "dog_0001" / "one.jpg").exists()
