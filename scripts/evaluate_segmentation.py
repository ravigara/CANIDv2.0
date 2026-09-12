"""Evaluate the trained anatomy segmenter on imported COCO-mask data.

The evaluator uses the same class sidecars and background derivation as the
training loader. It reports per-class IoU/Dice and an aggregate that excludes
the optional unsupervised ``fold`` channel.
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

from noseid.config import get_config  # noqa: E402
from noseid.data.synth import CLASSES, CLASS_ID  # noqa: E402
from noseid.pipeline.segmentation import (  # noqa: E402
    NoseSegmenter, compute_iou_dice)

EVALUATED_CLASSES = [c for c in CLASSES if c != "fold"]


def _scan(data: Path, split: str):
    base = data / split
    masks = data / "masks" / split
    for dog_dir in sorted(base.iterdir() if base.exists() else []):
        if not dog_dir.is_dir():
            continue
        for image in sorted(dog_dir.glob("*.[jp][pn]g")):
            mask = masks / dog_dir.name / f"{image.stem}.png"
            sidecar = masks / dog_dir.name / f"{image.stem}.classes.txt"
            if mask.exists():
                yield image, mask, sidecar


def _read_labels(mask_path: Path, sidecar: Path) -> np.ndarray:
    stack = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if stack is None:
        raise ValueError(f"unable to read mask: {mask_path}")
    if stack.ndim == 2:
        return stack.astype(np.uint8)
    labels = np.full(stack.shape[:2], CLASS_ID["background"], dtype=np.uint8)
    names = [x.strip() for x in sidecar.read_text(encoding="utf-8").splitlines()
             if x.strip()] if sidecar.exists() else []
    for channel, name in enumerate(names[:stack.shape[-1]]):
        if name in CLASS_ID:
            labels[stack[..., channel] > 0] = CLASS_ID[name]
    return labels


def evaluate(data: str, weights: str, split: str = "test",
             device: str = "auto") -> dict:
    cfg = get_config()
    segmenter = NoseSegmenter(weights=weights, backend="torch", cfg=cfg,
                               device=device)
    per_class = {name: {"iou": [], "dice": [], "images": 0,
                        "intersection": 0, "pred": 0, "gt": 0}
                 for name in CLASSES}
    samples = 0
    for image_path, mask_path, sidecar in _scan(Path(data), split):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unable to read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = _read_labels(mask_path, sidecar)
        result = segmenter.segment(rgb)
        pred = np.argmax(result.masks, axis=-1).astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch for {image_path}: {pred.shape} vs {gt.shape}")
        for name in CLASSES:
            iou, dice = compute_iou_dice(pred, gt, CLASS_ID[name])
            per_class[name]["iou"].append(iou)
            per_class[name]["dice"].append(dice)
            per_class[name]["images"] += 1
            p = pred == CLASS_ID[name]
            g = gt == CLASS_ID[name]
            per_class[name]["intersection"] += int(np.logical_and(p, g).sum())
            per_class[name]["pred"] += int(p.sum())
            per_class[name]["gt"] += int(g.sum())
        samples += 1

    if not samples:
        raise FileNotFoundError(f"no segmentation samples found under {data}/{split}")
    summary = {}
    for name, values in per_class.items():
        summary[name] = {
            "images": values["images"],
            "iou": float(np.mean(values["iou"])),
            "dice": float(np.mean(values["dice"])),
            "global_iou": float(values["intersection"] / max(
                values["pred"] + values["gt"] - values["intersection"], 1)),
            "global_dice": float(2 * values["intersection"] / max(
                values["pred"] + values["gt"], 1)),
        }
    aggregate = {
        "mIoU": float(np.mean([summary[n]["iou"] for n in EVALUATED_CLASSES])),
        "mDice": float(np.mean([summary[n]["dice"] for n in EVALUATED_CLASSES])),
        "classes": EVALUATED_CLASSES,
    }
    return {"data": data, "split": split, "weights": weights,
            "device": segmenter.device, "images": samples,
            "per_class": summary, "aggregate": aggregate}


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="anatomy_data_retrained_20260730")
    parser.add_argument("--weights", default=cfg["segmentation"]["weights"])
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", default=str(cfg["segmentation"].get("device", "auto")))
    parser.add_argument("--output", default="output/results/segmentation_test.json")
    args = parser.parse_args()
    result = evaluate(args.data, args.weights, args.split, args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
