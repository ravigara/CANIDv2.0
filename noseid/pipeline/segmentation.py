"""Stage 5 - Nose Segmentation.

Segments: rhinarium, left nare, right nare, philtrum, fold regions, background.

torch backend : HuggingFace SegFormer (``nvidia/segformer-b0-finetuned-ade-512-512``
                re-headed to 6 classes) or a U-Net fallback via smp.
numpy backend : classical pipeline - Otsu for rhinarium, blob detection for
                nares, geometric construction for philtrum/fold. Good enough to
                produce realistic masks + IoU/Dice for the demo.

Metrics: IoU + Dice; targets Dice > 0.95 (compute_segmentation_metrics).
"""
from __future__ import annotations

import os
import numpy as np
import cv2

from ..backend import resolve_backend
from ..config import get_config, cfg_get
from ..data.synth import CLASSES, CLASS_ID
from ..types import SegmentationResult

NUM_CLASSES = len(CLASSES)  # 6


def _padded_crop(image: np.ndarray, bbox: list[float], pad: float = 0.15):
    """Return detector crop plus its integer origin in the source image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    ix1 = max(0, int(x1 - pad * bw)); iy1 = max(0, int(y1 - pad * bh))
    ix2 = min(w, int(x2 + pad * bw)); iy2 = min(h, int(y2 + pad * bh))
    return image[iy1:iy2, ix1:ix2], ix1, iy1, ix2, iy2


def compute_iou_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> tuple[float, float]:
    p = pred == class_id
    g = gt == class_id
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = float(inter / union) if union else 1.0
    dice = float(2 * inter / (p.sum() + g.sum())) if (p.sum() + g.sum()) else 1.0
    return iou, dice


def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Mean IoU/Dice across classes + per-class dict."""
    per = {}
    ious, dices = [], []
    for c in CLASSES:
        iou, dice = compute_iou_dice(pred, gt, CLASS_ID[c])
        per[c] = {"iou": iou, "dice": dice}
        ious.append(iou); dices.append(dice)
    return {"per_class": per, "mIoU": float(np.mean(ious)),
            "mDice": float(np.mean(dices))}


class NoseSegmenter:
    def __init__(self, weights: str | os.PathLike | None = None,
                 backend: str = "auto", cfg: dict | None = None,
                 device: str = "auto"):
        self.weights = str(weights) if weights else None
        self.cfg = cfg or get_config()
        self.num_classes = int(cfg_get(self.cfg, "segmentation.num_classes", NUM_CLASSES))
        self.crop_pad = float(cfg_get(self.cfg, "segmentation.crop_pad", 0.15))
        self.backend = resolve_backend(backend, "segmentation")
        self.device = self._resolve_device(device) if self.backend == "torch" else "cpu"
        self._model = None
        if self.backend == "torch":
            try:
                self._init_torch()
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(f"[noseid] SegFormer init failed: {e}; numpy backend")
                self.backend = "numpy"

    def _init_torch(self) -> None:
        # Lazy: only built when weights exist, otherwise fall back.
        if not (self.weights and os.path.exists(self.weights)):
            self.backend = "numpy"
            return
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        try:
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.weights, num_labels=self.num_classes,
                ignore_mismatched_sizes=True, local_files_only=True)
            self._proc = SegformerImageProcessor.from_pretrained(
                self.weights, local_files_only=True)
        except OSError:
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.weights, num_labels=self.num_classes,
                ignore_mismatched_sizes=True)
            self._proc = SegformerImageProcessor.from_pretrained(self.weights)
        self._model.to(self.device).eval()

    @staticmethod
    def _resolve_device(value: str) -> str:
        import torch
        value = str(value).lower()
        if value == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if value.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError("a CUDA segmentation device was requested but CUDA is unavailable")
            return f"cuda:{value}"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA segmentation device requested but CUDA is unavailable")
        return value

    # -- public API -------------------------------------------------------
    def segment(self, image: np.ndarray, bbox: list[float] | None = None) -> SegmentationResult:
        if bbox is not None:
            crop, x1, y1, x2, y2 = _padded_crop(image, bbox, self.crop_pad)
            if crop.size == 0:
                raise ValueError("segmentation crop is empty")
            local = (self._segment_torch(crop)
                     if self.backend == "torch" and self._model is not None
                     else self._segment_numpy(crop))
            # Preserve the public API's source-image dimensions while making
            # both backends operate on the same padded detector crop.
            masks = np.zeros((*image.shape[:2], self.num_classes), dtype=np.uint8)
            masks[..., CLASS_ID["background"]] = 1
            masks[y1:y2, x1:x2] = local
        elif self.backend == "torch" and self._model is not None:
            masks = self._segment_torch(image)
        else:
            masks = self._segment_numpy(image)
        present = {c: bool(masks[..., CLASS_ID[c]].sum() > 0) for c in CLASSES}
        # IoU/dice left 0 absent ground truth; caller can recompute against GT
        return SegmentationResult(classes=CLASSES, present=present,
                                  iou=0.0, dice=0.0, masks=masks)

    __call__ = segment

    # -- backends ---------------------------------------------------------
    def _segment_torch(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        pil = Image.fromarray(image if image.ndim == 3 else
                              np.stack([image] * 3, -1))
        inp = self._proc(pil, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self._model(**inp).logits
        out = torch.nn.functional.interpolate(out, size=image.shape[:2],
                                              mode="bilinear", align_corners=False)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
        # convert single-channel label map -> HxWxC one-hot for uniform API
        masks = np.zeros((*pred.shape, self.num_classes), dtype=np.uint8)
        for cid in range(self.num_classes):
            masks[..., cid] = (pred == cid).astype(np.uint8)
        return masks

    def _segment_numpy(self, image: np.ndarray,
                       bbox: list[float] | None = None) -> np.ndarray:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        masks = np.zeros((h, w, self.num_classes), dtype=np.uint8)

        # rhinarium via Otsu (dark blob) + morphology
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
        # keep only the largest blob = rhinarium
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        rhin_mask = np.zeros((h, w), np.uint8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            rhin_mask = (th == biggest).astype(np.uint8)
        masks[..., CLASS_ID["rhinarium"]] = rhin_mask

        # nares = dark blobs inside the rhinarium
        dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 5)
        dark = cv2.bitwise_and(dark, dark, mask=rhin_mask)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        ln_mask = np.zeros((h, w), np.uint8)
        rn_mask = np.zeros((h, w), np.uint8)
        if len(cnts) >= 2:
            cx0 = w / 2
            placed = []
            for c in cnts:
                if cv2.contourArea(c) < 5:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                side = "L" if cx < cx0 else "R"
                if side == "L" and ln_mask.sum() == 0:
                    cv2.drawContours(ln_mask, [c], -1, 1, -1); placed.append("L")
                elif side == "R" and rn_mask.sum() == 0:
                    cv2.drawContours(rn_mask, [c], -1, 1, -1); placed.append("R")
                if {"L", "R"}.issubset(placed):
                    break
        masks[..., CLASS_ID["left_nare"]] = ln_mask
        masks[..., CLASS_ID["right_nare"]] = rn_mask

        # philtrum = vertical band between nares centers
        if ln_mask.sum() and rn_mask.sum():
            ly, lx = np.where(ln_mask); ry, rx = np.where(rn_mask)
            midx = ((lx.mean() + rx.mean()) / 2)
            top = max(ly.mean(), ry.mean())
            xs = np.arange(w)
            band = (np.abs(xs - midx) < 4).astype(np.uint8)
            band = np.tile(band, (h, 1))
            ymask = np.zeros((h, w), bool); ymask[int(top):int(top + 0.25 * h), :] = True
            masks[..., CLASS_ID["philtrum"]] = (band & ymask & rhin_mask).astype(np.uint8)

        # fold = annular ring near rhinarium edge
        eroded = cv2.erode(rhin_mask, k, iterations=2)
        masks[..., CLASS_ID["fold"]] = (rhin_mask & (1 - eroded)).astype(np.uint8)
        masks[..., CLASS_ID["background"]] = (rhin_mask == 0).astype(np.uint8)
        return masks
