"""Evaluate a trained landmark checkpoint on imported COCO-keypoint data.

Reports per-keypoint pixel error and PCK. The evaluation preprocessing matches
training: RGB image -> square resize -> tensor / 255, without augmentation.

Example:
    python scripts/evaluate_landmarks.py --data keypoint_data --split test --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noseid.pipeline.landmark_model import (  # noqa: E402
    KEYPOINT_NAMES, decode_heatmaps, load_landmark_model)


def _resolve_device(value: str) -> str:
    import torch
    value = str(value).lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return f"cuda:{value}"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return value


def _scan_samples(data_dir: str, split: str):
    base = Path(data_dir) / split
    kbase = Path(data_dir) / "keypoints" / split
    samples = []
    for dog_dir in sorted(base.iterdir() if base.exists() else []):
        if not dog_dir.is_dir():
            continue
        for image in sorted(dog_dir.glob("*.[jp][pn]g")):
            keypoint = kbase / dog_dir.name / f"{image.stem}.json"
            if keypoint.exists():
                samples.append((image, keypoint))
    return samples


def evaluate(data_dir: str, weights: str, split: str = "test",
             device: str = "auto", image_size: int = 256) -> dict:
    import torch

    dev = _resolve_device(device)
    model = load_landmark_model(weights, dev)
    samples = _scan_samples(data_dir, split)
    if not samples:
        raise FileNotFoundError(f"no keypoint samples found under {data_dir}/{split}")

    errors = {name: [] for name in KEYPOINT_NAMES}
    normalized_errors = {name: [] for name in KEYPOINT_NAMES}
    confidences = {name: [] for name in KEYPOINT_NAMES}
    image_confidences = []
    evaluated = 0
    with torch.inference_mode():
        for image_path, keypoint_path in samples:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"unable to read image: {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            resized = cv2.resize(rgb, (image_size, image_size),
                                 interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
            tensor = (tensor / 255.0).unsqueeze(0).to(dev)
            points, point_confidences = decode_heatmaps(
                model(tensor), input_size=(image_size, image_size))
            points[:, 0] *= float(w) / float(image_size)
            points[:, 1] *= float(h) / float(image_size)

            truth = np.asarray(
                json.loads(keypoint_path.read_text(encoding="utf-8"))["keypoints"],
                dtype=np.float32)[:len(KEYPOINT_NAMES)]
            image_confidences.append([float(v) for v in point_confidences])
            for index, name in enumerate(KEYPOINT_NAMES):
                if index >= len(truth) or truth[index, 2] <= 0:
                    continue
                distance = float(np.linalg.norm(points[index] - truth[index, :2]))
                errors[name].append(distance)
                normalized_errors[name].append(distance / max(h, w))
                confidences[name].append(float(point_confidences[index]))
            evaluated += 1

    per_keypoint = {}
    all_errors = []
    for name, values in errors.items():
        arr = np.asarray(values, dtype=np.float32)
        norm = np.asarray(normalized_errors[name], dtype=np.float32)
        all_errors.extend(arr.tolist())
        per_keypoint[name] = {
            "count": int(arr.size),
            "mean_error_px": float(arr.mean()) if arr.size else None,
            "median_error_px": float(np.median(arr)) if arr.size else None,
            "pck_at_5_percent": float(np.mean(norm <= 0.05)) if norm.size else None,
            "pck_at_10_percent": float(np.mean(norm <= 0.10)) if norm.size else None,
            "mean_confidence": float(np.mean(confidences[name])) if confidences[name] else None,
            "p10_confidence": float(np.percentile(confidences[name], 10)) if confidences[name] else None,
            "p90_confidence": float(np.percentile(confidences[name], 90)) if confidences[name] else None,
        }
    all_arr = np.asarray(all_errors, dtype=np.float32)
    all_norm = np.concatenate([
        np.asarray(normalized_errors[name], dtype=np.float32)
        for name in KEYPOINT_NAMES if normalized_errors[name]
    ]) if all_errors else np.asarray([], dtype=np.float32)
    threshold_sweep = []
    for threshold in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70):
        retained_errors = []
        retained_norm = []
        retained = total = 0
        valid_images = 0
        for image_conf in image_confidences:
            if all(float(c) >= threshold for c in image_conf):
                valid_images += 1
        for name in KEYPOINT_NAMES:
            for distance, norm, confidence in zip(
                    errors[name], normalized_errors[name], confidences[name]):
                total += 1
                if confidence >= threshold:
                    retained += 1
                    retained_errors.append(distance)
                    retained_norm.append(norm)
        threshold_sweep.append({
            "threshold": threshold,
            "point_coverage": float(retained / max(total, 1)),
            "all_point_image_coverage": float(valid_images / max(evaluated, 1)),
            "retained_mean_error_px": float(np.mean(retained_errors)) if retained_errors else None,
            "retained_pck_at_10_percent": float(np.mean(np.asarray(retained_norm) <= 0.10))
            if retained_norm else None,
        })
    return {
        "data": str(data_dir), "split": split, "weights": str(weights),
        "device": dev, "images": evaluated,
        "keypoints": KEYPOINT_NAMES,
        "per_keypoint": per_keypoint,
        "overall": {
            "count": int(all_arr.size),
            "mean_error_px": float(all_arr.mean()) if all_arr.size else None,
            "median_error_px": float(np.median(all_arr)) if all_arr.size else None,
            "pck_at_5_percent": float(np.mean(all_norm <= 0.05)) if all_norm.size else None,
            "pck_at_10_percent": float(np.mean(all_norm <= 0.10)) if all_norm.size else None,
        },
        "confidence_threshold_sweep": threshold_sweep,
    }


def main() -> None:
    from noseid.config import get_config
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="keypoint_data")
    parser.add_argument("--weights", default=cfg["landmarks"]["weights"])
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", default="output/results/landmarks_test.json")
    args = parser.parse_args()
    result = evaluate(args.data, args.weights, args.split,
                      args.device, args.image_size)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
