"""Build the previous anatomy-only identity gallery.

This remains available as the rollback/comparison builder. The active gallery
is built by ``scripts/build_cascade_gallery.py``. It replays the already-trained
detector -> landmarks -> segmentation pipeline and stores one centroid per
detected anatomical feature. No upstream model retraining is performed here.

Example:
    python scripts/build_identity_gallery.py --source-data identity_data --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _source_images(root: Path, split: str = "train"):
    split_dir = root / split
    if not split_dir.is_dir():
        return
    for dog_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(dog_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, dog_dir.name


def build_gallery(source_data: str, weights: str, output: str,
                  device_value: str, min_valid: int = 1) -> dict:
    from noseid.biometric import build_pipeline, run_to_feature_embeddings
    from noseid.config import get_config
    from noseid.matching import FeatureIndex

    source_root = Path(source_data).resolve()
    checkpoint = Path(weights).resolve()
    destination = Path(output).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"identity source not found: {source_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"embedding checkpoint not found: {checkpoint}")

    cfg = get_config({"embedding": {"weights": str(checkpoint)}})
    cfg["detection"]["device"] = device_value
    cfg["landmarks"]["device"] = device_value
    cfg["segmentation"]["device"] = device_value
    pc = build_pipeline(cfg)

    feature_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    counts = Counter()
    failures = []
    total = 0
    for image_path, dog_id in _source_images(source_root, "train") or ():
        total += 1
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            failures.append({"image": str(image_path), "reason": "unreadable"})
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            _, _, _, result = run_to_feature_embeddings(pc, image, dog_id=dog_id)
            if len(result.embeddings) < 2:
                failures.append({
                    "image": str(image_path),
                    "dog_id": dog_id,
                    "reason": "fewer_than_two_features",
                    "features": result.features_present,
                })
                continue
            for name, vector in result.embeddings.items():
                feature_vectors[dog_id][name].append(
                    np.asarray(vector, dtype=np.float32))
            counts[dog_id] += 1
        except Exception as exc:
            failures.append({"image": str(image_path), "dog_id": dog_id,
                             "reason": "processing_error", "message": str(exc)})

    index = FeatureIndex(dim=int(cfg["embedding"]["dim"]), cfg=cfg)
    enrollment = []
    skipped = []
    for dog_id in sorted(feature_vectors):
        if counts[dog_id] < min_valid:
            skipped.append(dog_id)
            continue
        try:
            result = index.enroll(
                dog_id, dict(feature_vectors[dog_id]), num_images=counts[dog_id])
            enrollment.append(result.to_dict())
        except ValueError as exc:
            skipped.append(dog_id)
            failures.append({"dog_id": dog_id, "reason": "enrollment_error",
                             "message": str(exc)})

    destination.mkdir(parents=True, exist_ok=True)
    index.save(str(destination))
    manifest = {
        "format": FeatureIndex.FORMAT,
        "weights": str(checkpoint),
        "embedding_version": index.embedding_version,
        "embedding_dim": index.dim,
        "device": device_value,
        "source_data": str(source_root),
        "source_split": "train",
        "source_images": total,
        "usable_images": int(sum(counts.values())),
        "enrolled_identities": sorted(index._templates),
        "enrolled_identity_count": index.size,
        "images_per_identity": dict(counts),
        "skipped_identities": skipped,
        "feature_counts": {
            name: int(sum(name in features for features in index._templates.values()))
            for name in ("rhinarium", "left_nare", "right_nare", "philtrum")
        },
        "enrollment": enrollment,
        "failures": failures,
        "development_only": True,
        "warning": (
            "This gallery is built from the current identity training split. "
            "Validate on session-separated known dogs and unknown dogs before use."
        ),
    }
    (destination / "feature_gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", default="identity_data",
                        help="original per-dog source images")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--output", default="output/index")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-valid", type=int, default=1)
    args = parser.parse_args()
    manifest = build_gallery(
        args.source_data, args.weights, args.output, args.device, args.min_valid)
    print(json.dumps({
        "gallery": args.output,
        "embedding_version": manifest["embedding_version"],
        "enrolled_identity_count": manifest["enrolled_identity_count"],
        "usable_images": manifest["usable_images"],
        "feature_counts": manifest["feature_counts"],
        "failures": len(manifest["failures"]),
    }, indent=2))


if __name__ == "__main__":
    main()
