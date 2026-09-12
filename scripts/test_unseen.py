"""Run the trained detector, landmarks, and anatomy segmenter on one unseen image.

The detector runs on the original image. Landmarks and segmentation then run
on the padded detector crop, matching the training/inference contract.

Example:
    python scripts/test_unseen.py --image unseen/my_nose.jpg --device 0
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


CLASS_NAMES = ["rhinarium", "left_nare", "right_nare", "philtrum", "fold", "background"]
COLORS = {
    "rhinarium": (0, 180, 255),
    "left_nare": (255, 80, 80),
    "right_nare": (80, 80, 255),
    "philtrum": (80, 220, 80),
    "fold": (220, 80, 220),
}


def _device(value: str) -> str:
    if value.isdigit():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return f"cuda:{value}"
    return value


def _padded_box(box: np.ndarray, shape: tuple[int, int, int], pad: float):
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh)),
            min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh)))


def run(image_path: str, output_dir: str, detector_weights: str,
        segmentation_weights: str, landmark_weights: str,
        conf: float = 0.50,
        pad: float = 0.15, device: str = "0") -> dict:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from ultralytics import YOLO
    from noseid.pipeline.landmarks import LandmarkDetector
    from noseid.pipeline.crop import normalize_nose_crop

    device = _device(str(device))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(image_path)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(src)

    detector = YOLO(detector_weights)
    result = detector.predict(str(src), conf=conf, device=device,
                              verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    stem = src.stem
    report = {
        "image": str(src),
        "detector_weights": detector_weights,
        "segmentation_weights": segmentation_weights,
        "landmark_weights": landmark_weights,
        "confidence_threshold": conf,
        "status": "no_detection",
    }

    if boxes is None or len(boxes) == 0:
        cv2.imwrite(str(out / f"{stem}_no_detection.jpg"), image)
        (out / f"{stem}_report.json").write_text(json.dumps(report, indent=2))
        return report

    scores = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    best = int(scores.argmax())
    x1, y1, x2, y2 = _padded_box(xyxy[best], image.shape, pad)
    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmark_detector = LandmarkDetector(
        weights=landmark_weights, backend="torch", device=device,
        crop_pad=pad)
    landmarks = landmark_detector.detect(
        image_rgb, [float(v) for v in xyxy[best].tolist()])

    model = SegformerForSemanticSegmentation.from_pretrained(
        segmentation_weights, local_files_only=True).to(device).eval()
    proc = SegformerImageProcessor.from_pretrained(
        segmentation_weights, local_files_only=True)
    inputs = proc(images=crop_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=crop.shape[:2], mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    overlay = image.copy()
    for cid, name in enumerate(CLASS_NAMES):
        if name == "background":
            continue
        mask = pred == cid
        if not mask.any():
            continue
        region = overlay[y1:y2, x1:x2]
        region[mask] = (
            0.55 * region[mask] + 0.45 * np.asarray(COLORS[name], dtype=np.float32)
        ).astype(np.uint8)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, f"NOSE01 {scores[best]:.3f}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    point_colors = {
        "left_nare": (255, 80, 80),
        "right_nare": (80, 80, 255),
        "philtrum": (80, 220, 80),
    }
    for name in ("left_nare", "right_nare", "philtrum"):
        point = getattr(landmarks, name)
        px, py = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (px, py), 6, point_colors[name], -1, cv2.LINE_AA)
        cv2.putText(overlay, name, (px + 7, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_colors[name], 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(out / f"{stem}_crop.jpg"), crop)
    cv2.imwrite(str(out / f"{stem}_overlay.jpg"), overlay)

    present = {name: bool((pred == cid).any()) for cid, name in enumerate(CLASS_NAMES)}
    source_masks = np.zeros((*image_rgb.shape[:2], len(CLASS_NAMES)), dtype=np.uint8)
    source_masks[y1:y2, x1:x2] = np.stack(
        [(pred == cid).astype(np.uint8) for cid in range(len(CLASS_NAMES))], axis=-1)
    normalized = normalize_nose_crop(
        image_rgb, [float(v) for v in xyxy[best].tolist()], landmarks,
        source_masks, CLASS_NAMES, target_size=224, pad=pad,
        min_confidence=0.50, mask_background=True)
    normalized_path = out / f"{stem}_normalized_crop.jpg"
    cv2.imwrite(str(normalized_path), cv2.cvtColor(
        normalized.image, cv2.COLOR_RGB2BGR))
    report.update({
        "status": "ok",
        "detector_confidence": float(scores[best]),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "present": present,
        "landmarks": landmarks.to_dict(),
        "outputs": {
            "crop": str(out / f"{stem}_crop.jpg"),
            "overlay": str(out / f"{stem}_overlay.jpg"),
            "normalized_crop": str(normalized_path),
        },
        "preprocessing": {
            "aligned": normalized.aligned,
            "mask_used": normalized.mask_used,
            "fallback_reason": normalized.fallback_reason,
            "source_bbox": normalized.source_bbox,
        },
    })
    (out / f"{stem}_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    from noseid.config import get_config

    cfg = get_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="output/results/unseen_model")
    p.add_argument("--detector-weights",
                   default=cfg["detection"]["weights"])
    p.add_argument("--segmentation-weights",
                    default=cfg["segmentation"]["weights"])
    p.add_argument("--landmark-weights",
                   default=cfg["landmarks"]["weights"])
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    print(json.dumps(run(args.image, args.output, args.detector_weights,
                         args.segmentation_weights, args.landmark_weights,
                         args.conf, args.pad,
                         args.device), indent=2))


if __name__ == "__main__":
    main()
