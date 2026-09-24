"""Build the whole-nose retrieval plus anatomy-feature reranking gallery.

This reuses the existing detector, segmentation, landmark, and embedding
checkpoints. It recomputes gallery vectors but does not train any model.

Typical current-project build::

    python scripts/build_cascade_gallery.py --device 0
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


def _folders(root: Path, split: str | None = None):
    base = root / split if split else root
    if not base.is_dir():
        return
    for dog_dir in sorted((path for path in base.iterdir() if path.is_dir()),
                          key=lambda path: path.name.lower()):
        images = sorted(path for path in dog_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            yield dog_dir.name, images


def _source_images(identity_source: Path, registration_source: Path,
                   registry_source: Path):
    """Yield each dog folder once, preferring original source photos."""
    seen: set[str] = set()
    for root, split in ((identity_source, "train"),
                        (registration_source, None),
                        (registry_source, None)):
        for dog_id, images in _folders(root, split) or ():
            if dog_id in seen:
                continue
            seen.add(dog_id)
            yield dog_id, images


def build_gallery(identity_source: str, registration_source: str,
                  registry_source: str, weights: str, output: str,
                  device_value: str, min_valid: int = 1) -> dict:
    from noseid.biometric import build_pipeline, run_to_cascade_embeddings
    from noseid.config import get_config
    from noseid.matching import CascadeFeatureIndex, FeatureIndex

    identity_root = Path(identity_source).resolve()
    registration_root = Path(registration_source).resolve()
    registry_root = Path(registry_source).resolve()
    checkpoint = Path(weights).resolve()
    destination = Path(output).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"embedding checkpoint not found: {checkpoint}")

    cfg = get_config({
        "embedding": {"weights": str(checkpoint)},
        "matching": {"embedding_version": "noseid-v3-cascade"},
    })
    cfg["detection"]["device"] = device_value
    cfg["landmarks"]["device"] = device_value
    cfg["segmentation"]["device"] = device_value
    pipeline = build_pipeline(cfg)

    old_index = None
    old_path = destination / "feature_meta.json"
    if old_path.is_file():
        old_index = FeatureIndex.load(str(destination))

    whole_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    feature_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    counts = Counter()
    failures = []
    total = 0
    source_map = {}
    for dog_id, image_paths in _source_images(
            identity_root, registration_root, registry_root) or ():
        source_map[dog_id] = [str(path) for path in image_paths]
        for image_path in image_paths:
            total += 1
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                failures.append({"image": str(image_path), "dog_id": dog_id,
                                 "reason": "unreadable"})
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                _, _, _, whole, features = run_to_cascade_embeddings(
                    pipeline, image, dog_id=dog_id)
                if len(features.embeddings) < 2:
                    failures.append({
                        "image": str(image_path), "dog_id": dog_id,
                        "reason": "fewer_than_two_features",
                        "features": features.features_present,
                    })
                    continue
                whole_vectors[dog_id].append(
                    np.asarray(whole.embedding, dtype=np.float32))
                for name, vector in features.embeddings.items():
                    feature_vectors[dog_id][name].append(
                        np.asarray(vector, dtype=np.float32))
                counts[dog_id] += 1
            except Exception as exc:
                failures.append({"image": str(image_path), "dog_id": dog_id,
                                 "reason": "processing_error", "message": str(exc)})

    index = CascadeFeatureIndex(dim=int(cfg["embedding"]["dim"]), cfg=cfg)
    enrollment = []
    skipped = []
    for dog_id in sorted(feature_vectors):
        if counts[dog_id] < min_valid:
            skipped.append(dog_id)
            continue
        try:
            metadata = ((old_index.get_record(dog_id) or {}).get("metadata", {})
                        if old_index else {})
            result = index.enroll(
                dog_id, whole_vectors[dog_id], dict(feature_vectors[dog_id]),
                metadata=metadata, num_images=counts[dog_id])
            enrollment.append(result.to_dict())
        except ValueError as exc:
            skipped.append(dog_id)
            failures.append({"dog_id": dog_id, "reason": "enrollment_error",
                             "message": str(exc)})

    destination.mkdir(parents=True, exist_ok=True)
    index.save(str(destination))
    manifest = {
        "format": CascadeFeatureIndex.FORMAT,
        "weights": str(checkpoint),
        "embedding_version": index.embedding_version,
        "embedding_dim": index.dim,
        "device": device_value,
        "sources": {
            "identity_data_train": str(identity_root),
            "registration": str(registration_root),
            "registry_fallback": str(registry_root),
        },
        "source_images": total,
        "usable_images": int(sum(counts.values())),
        "enrolled_identities": sorted(index._templates),
        "enrolled_identity_count": index.size,
        "images_per_identity": dict(counts),
        "source_map": source_map,
        "skipped_identities": skipped,
        "feature_counts": {
            name: int(sum(name in features for features in index._templates.values()))
            for name in ("rhinarium", "left_nare", "right_nare", "philtrum")
        },
        "enrollment": enrollment,
        "failures": failures,
        "development_only": True,
        "warning": (
            "Calibrate the cascade on session-separated known dogs and unknown "
            "same-breed dogs before production use."
        ),
    }
    (destination / "cascade_gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-source", default="identity_data")
    parser.add_argument("--registration-source", default="registration")
    parser.add_argument("--registry-source", default="output/registry")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--output", default="output/index")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-valid", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_gallery(
        args.identity_source, args.registration_source, args.registry_source,
        args.weights, args.output, args.device, args.min_valid,
    ), indent=2))


if __name__ == "__main__":
    main()
