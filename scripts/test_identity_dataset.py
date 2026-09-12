"""Create disposable qualitative review artifacts for the identity dataset.

This does not train or evaluate identity embeddings. It runs the real
detector -> landmark -> segmentation -> normalized-crop path and writes
per-image overlays, crops, and JSON reports to a disposable directory.

Quick review:
    python scripts/test_identity_dataset.py --max-per-dog 2 --device 0

Full review:
    python scripts/test_identity_dataset.py --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
CLASS_COLORS = {
    "rhinarium": (0, 180, 255),
    "left_nare": (255, 80, 80),
    "right_nare": (80, 80, 255),
    "philtrum": (80, 220, 80),
    "fold": (220, 80, 220),
}
POINT_COLORS = {
    "left_nare": (255, 80, 80),
    "right_nare": (80, 80, 255),
    "philtrum": (80, 220, 80),
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_metadata(root: Path) -> dict[str, dict]:
    path = root / "identity_metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row.get("image_path", ""): row for row in csv.DictReader(handle)}


def _draw_overlay(image_bgr: np.ndarray, bbox: list[float], confidence: float,
                  masks: np.ndarray, classes: list[str], landmarks) -> np.ndarray:
    overlay = image_bgr.copy()
    for index, name in enumerate(classes):
        if name == "background" or index >= masks.shape[-1]:
            continue
        mask = masks[..., index].astype(bool)
        if not mask.any() or name not in CLASS_COLORS:
            continue
        color = np.asarray(CLASS_COLORS[name], dtype=np.float32)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, f"NOSE01 {confidence:.3f}", (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA)
    for name in ("left_nare", "right_nare", "philtrum"):
        point = getattr(landmarks, name)
        px, py = int(round(point[0])), int(round(point[1]))
        color = POINT_COLORS[name]
        cv2.circle(overlay, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, name, (px + 7, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def review_dataset(data: str, output: str, detector_weights: str,
                   segmentation_weights: str, landmark_weights: str,
                   conf: float, pad: float, device: str,
                   max_per_dog: int | None = None) -> dict:
    from noseid.config import get_config
    from noseid.pipeline.crop import normalize_nose_crop, padded_bbox
    from noseid.pipeline.detection import NoseDetector
    from noseid.pipeline.landmarks import LandmarkDetector
    from noseid.pipeline.segmentation import NoseSegmenter

    root = Path(data).resolve()
    out = Path(output).resolve()
    if not root.exists():
        raise FileNotFoundError(f"identity dataset does not exist: {root}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = get_config()
    detector = NoseDetector(weights=detector_weights, conf=conf,
                            backend="torch", device=device)
    landmarks_detector = LandmarkDetector(
        weights=landmark_weights, backend="torch", device=device,
        crop_pad=pad,
        min_confidence=float(cfg["landmarks"].get("min_confidence", 0.50)),
    )
    segmenter = NoseSegmenter(
        weights=segmentation_weights, backend="torch", cfg=cfg,
        device=device,
    )
    metadata = _load_metadata(root)
    summary = {
        "data": str(root),
        "output": str(out),
        "detector_weights": detector_weights,
        "segmentation_weights": segmentation_weights,
        "landmark_weights": landmark_weights,
        "confidence_threshold": conf,
        "crop_pad": pad,
        "device": device,
        "max_per_dog": max_per_dog,
        "counts": Counter(),
        "per_split": {},
        "per_dog": {},
        "reports": [],
    }

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        split_counts = Counter()
        for dog_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = [
                p for p in sorted(dog_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if max_per_dog is not None:
                images = images[:max_per_dog]
            dog_key = f"{split}/{dog_dir.name}"
            dog_counts = Counter()
            for image_path in images:
                relative = _relative(image_path, root)
                destination = out / split / dog_dir.name
                destination.mkdir(parents=True, exist_ok=True)
                stem = image_path.stem
                report_path = destination / f"{stem}_report.json"
                report = {
                    "source": relative,
                    "split": split,
                    "dog_id": dog_dir.name,
                    "status": "error",
                    "metadata": metadata.get(relative),
                }
                try:
                    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        raise ValueError("image could not be decoded")
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    detections = detector.detect(image_rgb)
                    if not detections:
                        report["status"] = "no_detection"
                        cv2.imwrite(str(destination / f"{stem}_no_detection.jpg"), image_bgr)
                        dog_counts["no_detection"] += 1
                        summary["counts"]["no_detection"] += 1
                        _write_report(report_path, report)
                        summary["reports"].append(str(report_path))
                        continue

                    detection = max(detections, key=lambda item: item.confidence)
                    bbox = [float(value) for value in detection.bbox]
                    landmarks = landmarks_detector.detect(image_rgb, bbox)
                    segmentation = segmenter.segment(image_rgb, bbox)
                    normalized = normalize_nose_crop(
                        image_rgb, bbox, landmarks, segmentation.masks,
                        segmentation.classes, target_size=224, pad=pad,
                        min_confidence=float(cfg["landmarks"].get(
                            "min_confidence", 0.50)), mask_background=True,
                    )
                    crop_box = padded_bbox(image_rgb.shape, bbox, pad)
                    x1, y1, x2, y2 = crop_box
                    crop = image_bgr[y1:y2, x1:x2]
                    overlay = _draw_overlay(
                        image_bgr, bbox, float(detection.confidence),
                        segmentation.masks, segmentation.classes, landmarks,
                    )
                    crop_path = destination / f"{stem}_crop.jpg"
                    overlay_path = destination / f"{stem}_overlay.jpg"
                    normalized_path = destination / f"{stem}_normalized_crop.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    cv2.imwrite(str(overlay_path), overlay)
                    cv2.imwrite(str(normalized_path), cv2.cvtColor(
                        normalized.image, cv2.COLOR_RGB2BGR))
                    present = {
                        name: bool(segmentation.masks[..., index].any())
                        for index, name in enumerate(segmentation.classes)
                    }
                    report.update({
                        "status": "ok",
                        "detector": detection.to_dict(),
                        "landmarks": landmarks.to_dict(),
                        "segmentation": {
                            "classes": segmentation.classes,
                            "present": present,
                        },
                        "preprocessing": {
                            "aligned": normalized.aligned,
                            "mask_used": normalized.mask_used,
                            "fallback_reason": normalized.fallback_reason,
                            "source_bbox": normalized.source_bbox,
                        },
                        "outputs": {
                            "crop": str(crop_path),
                            "overlay": str(overlay_path),
                            "normalized_crop": str(normalized_path),
                        },
                    })
                    dog_counts["ok"] += 1
                    if normalized.mask_used:
                        dog_counts["mask_used"] += 1
                        summary["counts"]["mask_used"] += 1
                    else:
                        dog_counts["mask_fallback"] += 1
                        summary["counts"]["mask_fallback"] += 1
                    summary["counts"]["ok"] += 1
                except Exception as exc:  # keep reviewing other images
                    report["error"] = f"{type(exc).__name__}: {exc}"
                    dog_counts["error"] += 1
                    summary["counts"]["error"] += 1
                _write_report(report_path, report)
                summary["reports"].append(str(report_path))

            summary["per_dog"][dog_key] = dict(dog_counts)
            split_counts.update(dog_counts)
        summary["per_split"][split] = dict(split_counts)

    summary["counts"] = dict(summary["counts"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    from noseid.config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="identity_data")
    parser.add_argument("--output", default="identity_test_review")
    parser.add_argument("--detector-weights", default=cfg["detection"]["weights"])
    parser.add_argument("--segmentation-weights", default=cfg["segmentation"]["weights"])
    parser.add_argument("--landmark-weights", default=cfg["landmarks"]["weights"])
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-per-dog", type=int, default=None,
                        help="review only the first N images in each dog folder")
    args = parser.parse_args()
    if args.max_per_dog is not None and args.max_per_dog < 1:
        parser.error("--max-per-dog must be positive")
    result = review_dataset(
        args.data, args.output, args.detector_weights,
        args.segmentation_weights, args.landmark_weights,
        args.conf, args.pad, args.device, args.max_per_dog,
    )
    print(json.dumps({
        "output": result["output"],
        "counts": result["counts"],
        "reports": len(result["reports"]),
    }, indent=2))


if __name__ == "__main__":
    main()
