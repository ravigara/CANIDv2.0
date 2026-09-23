"""Register every dog folder in one model-loaded run.

Expected layout::

    registration/
      dog1/*.jpg
      dog2/*.jpg

The active anatomy-feature gallery is updated once after all folders are
processed. Existing gallery entries are preserved and same-ID entries are
replaced with the new enrollment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_directory(source: str, index_path: str, min_valid: int = 3) -> dict:
    from noseid.biometric import build_pipeline, enroll_dog, registration_response
    from noseid.matching import FeatureIndex

    root = Path(source)
    destination = Path(index_path)
    if not root.is_dir():
        raise FileNotFoundError(f"registration directory not found: {root}")
    if not (destination / "feature_meta.json").is_file():
        raise FileNotFoundError(
            f"active feature gallery not found: {destination / 'feature_meta.json'}")

    index = FeatureIndex.load(str(destination))
    pipeline = build_pipeline()
    results = []
    dog_dirs = sorted((path for path in root.iterdir() if path.is_dir()),
                      key=lambda path: path.name.lower())
    for dog_dir in dog_dirs:
        files = sorted(path for path in dog_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        images = []
        unreadable = []
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable.append(path.name)
                continue
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            enrolled = enroll_dog(
                pipeline, index, dog_dir.name, images, min_valid=min_valid,
                metadata={"source": str(root)},
            )
            row = registration_response(enrolled)
            row["status"] = enrolled.status
            row.update({
                "input_images": len(files),
                "unreadable_images": unreadable,
            })
        except Exception as exc:
            row = {
                "nose_print_id": dog_dir.name,
                "status": "Failed",
                "input_images": len(files),
                "unreadable_images": unreadable,
                "error": str(exc),
            }
        results.append(row)

    index.save(str(destination))
    return {
        "source": str(root),
        "gallery": str(destination),
        "registered": sum(row.get("status") == "Enrolled" for row in results),
        "failed": sum(row.get("status") != "Enrolled" for row in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="registration")
    parser.add_argument("--index", default="output/index")
    parser.add_argument("--min-valid", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(register_directory(args.source, args.index, args.min_valid),
                     indent=2))


if __name__ == "__main__":
    main()
