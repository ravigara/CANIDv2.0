"""Validate a real per-dog identity dataset before cropping or training.

Expected layout::

    identity_data/train/dog_0001/*.jpg
    identity_data/valid/dog_0001/*.jpg
    identity_data/test/dog_0001/*.jpg

The validator checks image readability, per-dog counts, exact duplicate
content across splits, and whether the test set contains dogs absent from
training for open-set evaluation. It cannot infer capture sessions; those
must be recorded and reviewed by the dataset owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


SPLITS = ("train", "valid", "test")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_identity_dataset(root: str, min_dogs: int = 2,
                              min_images_per_dog: int = 2) -> dict:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"identity dataset does not exist: {base}")
    split_data = {}
    hashes: dict[str, list[str]] = {}
    errors = []
    for split in SPLITS:
        split_dir = base / split
        dogs = {}
        if not split_dir.exists():
            errors.append(f"missing split directory: {split_dir}")
            split_data[split] = {"dogs": {}, "images": 0}
            continue
        for dog_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = sorted(p for p in dog_dir.iterdir()
                            if p.suffix.lower() in EXTS)
            dogs[dog_dir.name] = len(images)
            if len(images) < min_images_per_dog:
                errors.append(
                    f"{split}/{dog_dir.name}: {len(images)} images; "
                    f"need at least {min_images_per_dog}")
            for image_path in images:
                data = image_path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                hashes.setdefault(digest, []).append(f"{split}/{dog_dir.name}/{image_path.name}")
                if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                    errors.append(f"unreadable image: {image_path}")
        split_data[split] = {"dogs": dogs, "images": sum(dogs.values())}

    train_dogs = set(split_data["train"]["dogs"])
    valid_dogs = set(split_data["valid"]["dogs"])
    test_dogs = set(split_data["test"]["dogs"])
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_duplicates = [paths for paths in duplicate_groups
                              if len({p.split("/", 1)[0] for p in paths}) > 1]
    if cross_split_duplicates:
        errors.append(f"exact duplicate content crosses splits: {len(cross_split_duplicates)} groups")

    open_set_test_dogs = sorted(test_dogs - train_dogs)
    if not open_set_test_dogs:
        errors.append("test has no dogs absent from train; open-set rejection cannot be evaluated")
    if len(train_dogs) < min_dogs:
        errors.append(f"train has {len(train_dogs)} dogs; need at least {min_dogs}")

    result = {
        "root": str(base),
        "splits": split_data,
        "train_dogs": sorted(train_dogs),
        "valid_dogs": sorted(valid_dogs),
        "test_dogs": sorted(test_dogs),
        "open_set_test_dogs": open_set_test_dogs,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "errors": errors,
        "valid": not errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="identity_data")
    parser.add_argument("--min-dogs", type=int, default=2)
    parser.add_argument("--min-images-per-dog", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        result = validate_identity_dataset(
            args.data, args.min_dogs, args.min_images_per_dog)
    except FileNotFoundError as exc:
        print(json.dumps({"root": args.data, "valid": False,
                          "errors": [str(exc)]}, indent=2))
        raise SystemExit(2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 2)


if __name__ == "__main__":
    main()
