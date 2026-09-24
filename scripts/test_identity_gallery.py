"""Smoke-test the active identity gallery.

This deliberately re-runs the complete detector -> anatomy -> embedding path.
It catches representation drift that a crop-only gallery test would miss.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_identity_gallery import IMAGE_EXTENSIONS, _source_images  # noqa: E402


def evaluate_gallery(source_data: str, weights: str, index_dir: str,
                     output: str, device_value: str,
                     feature_only: bool = False) -> dict:
    from noseid.biometric import (build_pipeline,
                                  run_to_cascade_embeddings,
                                  run_to_feature_embeddings,
                                  run_to_full_photo_cascade_embeddings)
    from noseid.config import get_config
    from noseid.matching import (CascadeFeatureIndex, FeatureIndex,
                                 FullPhotoCascadeIndex)

    source_root = Path(source_data).resolve()
    gallery_path = Path(index_dir).resolve()
    is_full_photo = ((gallery_path / "full_photo_cascade_meta.json").is_file()
                     and not feature_only)
    is_cascade = ((gallery_path / "cascade_meta.json").is_file()
                  and not feature_only and not is_full_photo)
    if (not is_full_photo and not is_cascade
            and not (gallery_path / "feature_meta.json").is_file()):
        raise FileNotFoundError(f"identity gallery not found: {gallery_path}")
    if not Path(weights).exists():
        raise FileNotFoundError(f"embedding checkpoint not found: {weights}")
    index = (FullPhotoCascadeIndex.load(str(gallery_path)) if is_full_photo
             else CascadeFeatureIndex.load(str(gallery_path)) if is_cascade
             else FeatureIndex.load(str(gallery_path)))
    cfg = get_config({"embedding": {"weights": str(Path(weights).resolve())}})
    cfg["detection"]["device"] = device_value
    cfg["landmarks"]["device"] = device_value
    cfg["segmentation"]["device"] = device_value
    pc = build_pipeline(cfg)

    split_results = {}
    for split in ("train", "valid", "test"):
        statuses = Counter()
        correct = 0
        rows = []
        for image_path, dog_id in _source_images(source_root, split) or ():
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                statuses["Unreadable"] += 1
                continue
            try:
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                if is_full_photo:
                    _, _, _, photo, whole, features = \
                        run_to_full_photo_cascade_embeddings(
                            pc, image, dog_id=dog_id)
                    result = index.identify(
                        photo.embedding, whole.embedding,
                        {name: vector for name, vector in features.embeddings.items()})
                elif is_cascade:
                    _, _, _, whole, features = run_to_cascade_embeddings(
                        pc, image, dog_id=dog_id)
                    result = index.identify(
                        whole.embedding,
                        {name: vector for name, vector in features.embeddings.items()})
                else:
                    _, _, _, features = run_to_feature_embeddings(
                        pc, image, dog_id=dog_id)
                    result = index.identify({
                        name: vector for name, vector in features.embeddings.items()
                    })
            except Exception as exc:
                statuses["ProcessingError"] += 1
                rows.append({"image": str(image_path), "expected_dog_id": dog_id,
                             "status": "ProcessingError", "error": str(exc)})
                continue
            statuses[result.status] += 1
            if split == "train" and result.status == "Verified" and result.dog_id == dog_id:
                correct += 1
            rows.append({
                "image": str(image_path),
                "expected_dog_id": dog_id,
                "returned_dog_id": result.dog_id,
                "similarity": result.similarity,
                "status": result.status,
                "features": features.features_present,
                "whole_nose_similarity": (
                    result.candidates[0].get("whole_nose_similarity")
                    if result.candidates else None
                ),
                "full_photo_similarity": (
                    result.candidates[0].get("full_photo_similarity")
                    if result.candidates else None
                ),
            })
        total = sum(statuses.values())
        split_results[split] = {
            "samples": total,
            "known_training_gallery": split == "train",
            "status_counts": dict(sorted(statuses.items())),
            "rank1_verified_accuracy_percent": (
                float(100.0 * correct / total) if split == "train" and total else None
            ),
            "rows": rows,
        }

    result = {
        "gallery": str(gallery_path),
        "weights": str(Path(weights).resolve()),
        "embedding_version": index.embedding_version,
        "device": device_value,
        "gallery_identities": sorted(index._templates),
        "thresholds": {
            "verify_cosine": index.verify_thr,
            "reject_cosine": index.reject_thr,
            "margin": index.margin_thr,
            "min_features": index.min_features,
        },
        "cascade": ({
            "retrieval_k": index.retrieval_k,
            "photo_weight": getattr(index, "photo_weight", None),
            "whole_weight": index.whole_weight,
            "feature_weight": index.feature_weight,
        } if (is_full_photo or is_cascade) else None),
        "split_results": split_results,
        "warning": (
            "Training results are in-sample. Valid/test identities absent from "
            "the gallery are open-set checks, not known-dog accuracy."
        ),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", default="identity_data")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--index", default="output/index")
    parser.add_argument("--output", default="identity_test_review/gallery_smoke.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--feature-only", action="store_true",
                        help="test the previous anatomy-only gallery")
    args = parser.parse_args()
    result = evaluate_gallery(
        args.source_data, args.weights, args.index, args.output, args.device,
        args.feature_only)
    print(json.dumps({
        "output": args.output,
        "gallery_identities": len(result["gallery_identities"]),
        "splits": {
            split: {key: value for key, value in values.items() if key != "rows"}
            for split, values in result["split_results"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
