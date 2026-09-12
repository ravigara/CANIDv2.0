"""Stage 4 - Anatomical Landmark Detection.

Detects: left nare, right nare, philtrum, nose boundary, major fold lines,
texture regions.

torch backend : a small CNN heatmap regressor (HRNet/MediaPipe-style) trained
                on the COCO keypoint annotations produced by Stage 1.
numpy backend : blob/contour analysis on the rhinarium crop to localize the
                two darkest blobs (nares) and the philtrum midpoint below them.

Output: LandmarkResult(left_nare, right_nare, philtrum, ...).
"""
from __future__ import annotations

import os
import numpy as np
import cv2

from ..backend import resolve_backend
from ..types import LandmarkResult
from .landmark_model import decode_heatmaps, load_landmark_model


def _crop(image: np.ndarray, bbox: list[float], pad: float = 0.1) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    return image[y1:y2, x1:x2], x1, y1


def _blob_centers(crop_gray: np.ndarray, min_area: int, max_area: int,
                  n: int) -> list[tuple[float, float]]:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold, params.maxThreshold = 10, 100
    params.filterByArea = True
    params.minArea, params.maxArea = min_area, max_area
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    det = cv2.SimpleBlobDetector_create(params)
    kps = det.detect(crop_gray)
    kps = sorted(kps, key=lambda k: k.size, reverse=True)[:n]
    return [(float(k.pt[0]), float(k.pt[1])) for k in kps]


class LandmarkDetector:
    def __init__(self, weights: str | os.PathLike | None = None,
                 backend: str = "auto", device: str = "auto",
                 input_size: int = 256, crop_pad: float = 0.15,
                 min_confidence: float = 0.50):
        self.weights = str(weights) if weights else None
        self.backend = resolve_backend(backend, "landmarks")
        self.device = self._resolve_device(device) if self.backend == "torch" else "cpu"
        self.input_size = int(input_size)
        self.crop_pad = float(crop_pad)
        self.min_confidence = float(min_confidence)
        self._model = None
        if self.backend == "torch":
            if not self.weights and str(backend).lower() == "auto":
                # Auto mode without a checkpoint should retain the lightweight
                # heuristic path without emitting a model-load warning.
                self.backend = "numpy"
                return
            try:
                self._init_torch()
            except Exception as e:  # pragma: no cover - dependency/model failure
                import warnings
                warnings.warn(
                    f"[noseid] landmark model init failed: {e}; using numpy backend")
                self.backend = "numpy"

    @staticmethod
    def _resolve_device(value: str) -> str:
        import torch
        value = str(value).lower()
        if value == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if value.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError("a CUDA landmark device was requested but CUDA is unavailable")
            return f"cuda:{value}"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA landmark device requested but CUDA is unavailable")
        return str(value)

    def _init_torch(self) -> None:
        if not (self.weights and os.path.exists(self.weights)):
            raise FileNotFoundError(f"landmark weights not found: {self.weights}")
        self._model = load_landmark_model(self.weights, self.device)

    def detect(self, image: np.ndarray, bbox: list[float] | None = None) -> LandmarkResult:
        if self.backend == "torch" and self._model is not None:
            return self._detect_torch(image, bbox)
        return self._detect_numpy(image, bbox)

    def _detect_torch(self, image: np.ndarray,
                      bbox: list[float] | None = None) -> LandmarkResult:
        import torch

        if bbox is None:
            h, w = image.shape[:2]
            bbox = [0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h]
        crop, ox, oy = _crop(image, bbox, self.crop_pad)
        if crop.size == 0:
            raise ValueError("landmark crop is empty")
        rgb = crop if crop.ndim == 3 else np.stack([crop] * 3, axis=-1)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = (tensor / 255.0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            heatmaps = self._model(tensor)
        points, confidences = decode_heatmaps(
            heatmaps, input_size=(self.input_size, self.input_size))
        crop_h, crop_w = crop.shape[:2]
        points[:, 0] *= float(crop_w) / float(self.input_size)
        points[:, 1] *= float(crop_h) / float(self.input_size)
        points[:, 0] += ox
        points[:, 1] += oy
        names = ["left_nare", "right_nare", "philtrum"]
        confidence = {n: float(c) for n, c in zip(names, confidences)}
        valid = bool(np.all(confidences >= self.min_confidence))
        return LandmarkResult(
            left_nare=points[0].tolist(),
            right_nare=points[1].tolist(),
            philtrum=points[2].tolist(),
            confidence=confidence,
            valid=valid)

    def _detect_numpy(self, image: np.ndarray,
                      bbox: list[float] | None = None) -> LandmarkResult:
        if bbox is None:
            h, w = image.shape[:2]
            bbox = [0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h]
        crop, ox, oy = _crop(image, bbox, self.crop_pad)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        ch, cw = gray.shape

        # nares = dark blobs inside the upper two-thirds of the rhinarium crop
        upper = gray[: 2 * ch // 3, :]
        crop_area = float(ch * cw)
        min_a = max(3, int(0.0015 * crop_area))
        max_a = max(min_a + 1, int(0.03 * crop_area))
        centers = _blob_centers(upper, min_a, max_a, 8)

        # split blobs into left/right halves of the crop; take the strongest
        # blob on each side of the vertical midline (anatomically one nare/side)
        midx = cw / 2
        left_cands = [c for c in centers if c[0] < midx]
        right_cands = [c for c in centers if c[0] >= midx]
        left_local = left_cands[0] if left_cands else (cw * 0.4, ch * 0.35)
        right_local = right_cands[0] if right_cands else (cw * 0.6, ch * 0.35)

        left = [left_local[0] + ox, left_local[1] + oy]
        right = [right_local[0] + ox, right_local[1] + oy]
        # philtrum = below nares midpoint, on the vertical axis
        midx = (left[0] + right[0]) / 2
        phil = [midx, max(left[1], right[1]) + 0.18 * (bbox[3] - bbox[1])]

        # nose boundary = rhinarium contour (largest dark contour)
        boundary = self._boundary(gray, ox, oy)
        # fold lines = strong edges within the rhinarium ring
        folds = self._fold_lines(gray, ox, oy, boundary)
        return LandmarkResult(
            left_nare=[float(left[0]), float(left[1])],
            right_nare=[float(right[0]), float(right[1])],
            philtrum=[float(phil[0]), float(phil[1])],
            nose_boundary=boundary,
            fold_lines=folds,
        )

    __call__ = detect

    @staticmethod
    def _boundary(gray: np.ndarray, ox: int, oy: int) -> list[list[float]]:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        c = max(cnts, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        return [[float(p[0] + ox), float(p[1] + oy)] for p in approx]

    @staticmethod
    def _fold_lines(gray: np.ndarray, ox: int, oy: int,
                    boundary: list[list[float]]) -> list[list[float]]:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                                minLineLength=gray.shape[0] // 6,
                                maxLineGap=8)
        out = []
        if lines is not None:
            for l in lines[:5].reshape(-1, 4):
                out.append([float(l[0] + ox), float(l[1] + oy),
                            float(l[2] + ox), float(l[3] + oy)])
        return out
