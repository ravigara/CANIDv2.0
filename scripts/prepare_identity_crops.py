"""Create per-dog nose-crop folders for identity embedding training.

The embedding model must see the nose pattern rather than the full dog face.
This script applies the trained detector, landmark model, and anatomy segmenter
to each image while preserving the train/valid/test and dog identity folder
structure. Use ``--detector-only`` only for an explicit baseline.

Example:
    python scripts/prepare_identity_crops.py \
        --weights output/models/detection_best.pt \
        --source identity_data --out identity_crops --device 0
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


def prepare_identity_crops(weights: str, source: str, out: str,
                           conf: float = 0.5, pad: float = 0.15,
                           device: str | None = None,
                           segmentation_weights: str | None = None,
                           landmark_weights: str | None = None,
                           detector_only: bool = False,
                           direct_closeups: bool = False) -> dict:
    from noseid.config import get_config
    from noseid.pipeline.crop import normalize_nose_crop
    from noseid.pipeline.detection import NoseDetector

    cfg = get_config()
    src = Path(source)
    dst = Path(out)
    if not src.exists():
        raise FileNotFoundError(f"identity source does not exist: {src}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dst.mkdir(parents=True, exist_ok=True)
    detector = None if direct_closeups else NoseDetector(
        weights=weights, conf=conf, backend="torch", device=device)
    landmarks = segmenter = None
    if not detector_only and not direct_closeups:
        from noseid.pipeline.landmarks import LandmarkDetector
        from noseid.pipeline.segmentation import NoseSegmenter
        segmentation_weights = segmentation_weights or cfg["segmentation"]["weights"]
        landmark_weights = landmark_weights or cfg["landmarks"]["weights"]
        landmarks = LandmarkDetector(
            weights=landmark_weights, backend="torch", device=device,
            crop_pad=pad,
            min_confidence=float(cfg["landmarks"].get("min_confidence", 0.50)))
        segmenter = NoseSegmenter(
            weights=segmentation_weights, backend="torch", cfg=cfg,
            device=device or cfg.get("segmentation", {}).get("device", "auto"))

    manifest = {
        "weights": weights,
        "segmentation_weights": segmentation_weights,
        "landmark_weights": landmark_weights,
        "source": source,
        "detector_only": detector_only,
        "direct_closeups": direct_closeups,
        "conf": conf,
        "pad": pad,
        "splits": {},
    }

    for split_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        split_out = dst / split_dir.name
        total = saved = skipped = 0
        records = []
        for dog_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            dog_out = split_out / dog_dir.name
            for image_path in sorted(p for p in dog_dir.iterdir()
                                     if p.suffix.lower() in exts):
                total += 1
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    skipped += 1
                    continue
                if direct_closeups:
                    target_size = int(cfg.get("image", {}).get("crop_size", 224))
                    direct = cv2.resize(image, (target_size, target_size),
                                        interpolation=cv2.INTER_AREA)
                    dog_out.mkdir(parents=True, exist_ok=True)
                    target_path = dog_out / image_path.name
                    if cv2.imwrite(str(target_path), direct):
                        saved += 1
                        records.append({
                            "source": str(image_path),
                            "output": str(target_path),
                            "status": "direct_closeup",
                            "detector_bypassed": True,
                        })
                    else:
                        skipped += 1
                        records.append({"source": str(image_path),
                                        "status": "write_failed"})
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = detector.detect(rgb)
                if not detections:
                    skipped += 1
                    records.append({"source": str(image_path), "status": "no_detection"})
                    continue
                detection = max(detections, key=lambda item: item.confidence)
                bbox = detection.bbox
                lm = landmarks.detect(rgb, bbox) if landmarks is not None else None
                seg = segmenter.segment(rgb, bbox) if segmenter is not None else None
                normalized = normalize_nose_crop(
                    rgb, bbox, lm, seg.masks if seg is not None else None,
                    seg.classes if seg is not None else None,
                    target_size=int(cfg.get("image", {}).get("crop_size", 224)),
                    pad=pad,
                    min_confidence=float(cfg["landmarks"].get("min_confidence", 0.50)),
                    mask_background=not detector_only)
                dog_out.mkdir(parents=True, exist_ok=True)
                target_path = dog_out / image_path.name
                if cv2.imwrite(str(target_path), cv2.cvtColor(
                        normalized.image, cv2.COLOR_RGB2BGR)):
                    saved += 1
                    records.append({
                        "source": str(image_path),
                        "output": str(target_path),
                        "status": "saved",
                        "detector_confidence": detection.confidence,
                        "bbox": bbox,
                        "aligned": normalized.aligned,
                        "mask_used": normalized.mask_used,
                        "fallback_reason": normalized.fallback_reason,
                        "landmarks_valid": bool(lm.valid) if lm is not None else False,
                    })
                else:
                    skipped += 1
                    records.append({"source": str(image_path), "status": "write_failed"})
        manifest["splits"][split_dir.name] = {
            "total": total, "saved": saved, "skipped": skipped,
            "files": records,
        }

    (dst / "crop_manifest.json").write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True,
                   help="per-dog source root with train/valid/test folders")
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--device", default=None)
    p.add_argument("--segmentation-weights", default=None)
    p.add_argument("--landmark-weights", default=None)
    p.add_argument("--detector-only", action="store_true",
                   help="disable landmark/segmentation normalization")
    p.add_argument("--direct-closeups", action="store_true",
                   help="treat input files as already-cropped nose images; bypass detection")
    args = p.parse_args()
    print(json.dumps(prepare_identity_crops(
        args.weights, args.source, args.out, args.conf, args.pad, args.device,
        args.segmentation_weights, args.landmark_weights, args.detector_only,
        args.direct_closeups),
        indent=2))


if __name__ == "__main__":
    main()
