"""Build metadata for the organized identity dataset.

The script records facts available from the files and computes an image-quality
score from the image itself. It deliberately does not invent camera, lighting,
or session provenance; unavailable values remain ``unknown``.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _quality_score(image: np.ndarray) -> tuple[int, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    sharp_score = min(100.0, max(0.0, sharpness / 300.0 * 100.0))
    brightness_score = max(0.0, 100.0 - abs(brightness - 128.0) / 128.0 * 100.0)
    resolution_score = min(100.0, min(image.shape[:2]) / 1000.0 * 100.0)
    score = round(0.5 * sharp_score + 0.3 * brightness_score + 0.2 * resolution_score)
    return int(max(0, min(100, score))), sharpness, brightness


def _capture_date(name: str) -> str:
    match = DATE_RE.search(name)
    return "-".join(match.groups()) if match else "unknown"


def build_metadata(root: str, output: str) -> dict:
    base = Path(root)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for split in SPLITS:
        split_dir = base / split
        for image_path in sorted(split_dir.rglob("*")) if split_dir.is_dir() else []:
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"unreadable image: {image_path}")
                continue
            score, sharpness, brightness = _quality_score(image)
            dog_id = image_path.parent.name
            rows.append({
                "image_path": image_path.relative_to(base).as_posix(),
                "dog_id": dog_id,
                "session_id": f"source_burst_{dog_id}",
                "capture_date": _capture_date(image_path.name),
                "camera": "unknown",
                "conditions": "unknown",
                "quality": score,
                "notes": (
                    "auto_measured; "
                    f"width={image.shape[1]}; height={image.shape[0]}; "
                    f"laplacian={sharpness:.2f}; brightness={brightness:.2f}; "
                    "camera_and_conditions_require_manual_confirmation"
                ),
            })

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "dog_id", "session_id", "capture_date", "camera",
              "conditions", "quality", "notes"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(destination), "rows": len(rows), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="identity_data")
    parser.add_argument("--output", default="identity_data/identity_metadata.csv")
    args = parser.parse_args()
    result = build_metadata(args.data, args.output)
    print(result)
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
