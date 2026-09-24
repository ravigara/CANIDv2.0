"""Build the reversible full-photo + nose cascade gallery.

The existing ``cascade_meta.json`` is never overwritten.  This command writes
``full_photo_cascade_meta.json`` beside it, reusing all existing detector,
segmentation, landmark, and nose-embedding checkpoints without training.

Run::

    python scripts/build_full_photo_gallery.py --device 0
"""
from __future__ import annotations

import argparse
import copy
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


def build_gallery(identity_source: str, registration_source: str,
                  registry_source: str, weights: str, output: str,
                  device_value: str, min_valid: int = 1) -> dict:
    from noseid.biometric import (build_pipeline,
                                  run_to_full_photo_cascade_embeddings)
    from noseid.config import get_config
    from noseid.embedding.full_photo import FullPhotoEmbedder
    from noseid.matching import (CascadeFeatureIndex, FeatureIndex,
                                 FullPhotoCascadeIndex)
    from scripts.build_cascade_gallery import _source_images

    identity_root = Path(identity_source).resolve()
    registration_root = Path(registration_source).resolve()
    registry_root = Path(registry_source).resolve()
    checkpoint = Path(weights).resolve()
    destination = Path(output).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"embedding checkpoint not found: {checkpoint}")

    cfg = get_config({
        "embedding": {"weights": str(checkpoint)},
        "matching": {"embedding_version": "noseid-v4-full-photo-cascade"},
    })
    old_index = None
    if (destination / "full_photo_cascade_meta.json").is_file():
        old_index = FullPhotoCascadeIndex.load(str(destination))
    elif (destination / "cascade_meta.json").is_file():
        old_index = CascadeFeatureIndex.load(str(destination))
    elif (destination / "feature_meta.json").is_file():
        old_index = FeatureIndex.load(str(destination))

    photo_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    whole_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    feature_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    counts = Counter()
    failures = []
    source_map = {}
    total = 0
    # A prior cascade already contains the expensive nose/anatomy templates.
    # For those identities, only the new full-frame vectors need computing.
    reusable_nose = isinstance(old_index, CascadeFeatureIndex)
    photo_embedder = FullPhotoEmbedder(cfg)
    pipeline = None
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
                if reusable_nose and dog_id in old_index._templates:
                    photo = photo_embedder.embed(image, dog_id)
                    photo_vectors[dog_id].append(
                        np.asarray(photo.embedding, dtype=np.float32))
                    counts[dog_id] += 1
                    continue
                if pipeline is None:
                    cfg["detection"]["device"] = device_value
                    cfg["landmarks"]["device"] = device_value
                    cfg["segmentation"]["device"] = device_value
                    pipeline = build_pipeline(cfg)
                _, _, _, photo, whole, features = \
                    run_to_full_photo_cascade_embeddings(
                        pipeline, image, dog_id=dog_id)
                if len(features.embeddings) < 2:
                    failures.append({
                        "image": str(image_path), "dog_id": dog_id,
                        "reason": "fewer_than_two_features",
                        "features": features.features_present,
                    })
                    continue
                photo_vectors[dog_id].append(
                    np.asarray(photo.embedding, dtype=np.float32))
                whole_vectors[dog_id].append(
                    np.asarray(whole.embedding, dtype=np.float32))
                for name, vector in features.embeddings.items():
                    feature_vectors[dog_id][name].append(
                        np.asarray(vector, dtype=np.float32))
                counts[dog_id] += 1
            except Exception as exc:
                failures.append({"image": str(image_path), "dog_id": dog_id,
                                 "reason": "processing_error",
                                 "message": str(exc)})

    index = FullPhotoCascadeIndex(dim=int(cfg["embedding"]["dim"]), cfg=cfg)
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
                dog_id, photo_vectors[dog_id], whole_vectors[dog_id],
                dict(feature_vectors[dog_id]), metadata=metadata,
                num_images=counts[dog_id])
            enrollment.append(result.to_dict())
        except ValueError as exc:
            skipped.append(dog_id)
            failures.append({"dog_id": dog_id, "reason": "enrollment_error",
                             "message": str(exc)})

    # Copy the existing whole-nose/anatomy templates and attach the newly
    # computed full-photo template for identities handled by the fast path.
    if reusable_nose:
        for dog_id in sorted(photo_vectors):
            if dog_id in feature_vectors or counts[dog_id] < min_valid:
                continue
            try:
                index._templates[dog_id] = {
                    name: vector.copy()
                    for name, vector in old_index._templates[dog_id].items()
                }
                index._whole_templates[dog_id] = old_index._whole_templates[
                    dog_id].copy()
                index._records[dog_id] = copy.deepcopy(
                    old_index.get_record(dog_id) or {"dog_id": dog_id})
                result = index.enroll_full_photo_template(
                    dog_id, photo_vectors[dog_id],
                    metadata=(index._records[dog_id].get("metadata", {}) or {}),
                    num_images=counts[dog_id])
                enrollment.append(result.to_dict())
            except (KeyError, ValueError) as exc:
                skipped.append(dog_id)
                failures.append({"dog_id": dog_id,
                                 "reason": "photo_enrollment_error",
                                 "message": str(exc)})

    destination.mkdir(parents=True, exist_ok=True)
    index.save(str(destination))
    manifest = {
        "format": FullPhotoCascadeIndex.FORMAT,
        "weights": str(checkpoint),
        "embedding_version": index.embedding_version,
        "embedding_dim": index.dim,
        "full_photo_backend": cfg.get("full_photo", {}).get(
            "backend", "hybrid_numpy"),
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
        "weights": {
            "full_photo": index.photo_weight,
            "whole_nose": index.whole_weight,
            "anatomical_features": index.feature_weight,
        },
        "enrollment": enrollment,
        "failures": failures,
        "development_only": True,
        "warning": (
            "Calibrate on session-separated known dogs and unknown same-breed "
            "dogs. Set NOSEID_GALLERY_MODE=cascade to revert to the prior "
            "whole-nose cascade without rebuilding artifacts."
        ),
    }
    (destination / "full_photo_gallery_manifest.json").write_text(
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
