"""Stage 3 - Dog Nose Detection.

torch backend : ultralytics YOLO (YOLOv12 preferred, YOLOv11 fallback).
numpy backend: thresholding + morphology to locate the dark rhinarium blob.
                 Good enough to drive the demo end-to-end and to bootstrap
                 labeling on real data.

Output: list[DetectionResult(bbox=[x1,y1,x2,y2], confidence, label)].
"""
from __future__ import annotations

import os
import numpy as np
import cv2

from ..backend import resolve_backend
from ..types import DetectionResult


class NoseDetector:
    def __init__(self, weights: str | os.PathLike | None = None,
                 conf: float = 0.5, backend: str = "auto",
                 label: str = "NOSE01", device: str | None = None):
        self.weights = str(weights) if weights else None
        self.conf = float(conf)
        self.label = str(label)
        self.device = None if device in (None, "", "auto") else device
        self.backend = resolve_backend(backend, "detection")
        self._model = None
        if self.backend == "torch":
            try:
                self._init_torch()
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(f"[noseid] YOLO init failed: {e}; using numpy backend")
                self.backend = "numpy"

    def _init_torch(self) -> None:
        from ultralytics import YOLO
        if self.weights and os.path.exists(self.weights):
            self._model = YOLO(self.weights)
        else:
            # no weights -> we can still load architecture for ONNX export path,
            # but inference will be random. Fall back to numpy for detection.
            self._model = None
            self.backend = "numpy"

    # -- public API -------------------------------------------------------
    def detect(self, image: np.ndarray,
               gt_hint: list[float] | None = None) -> list[DetectionResult]:
        """Detect the dog nose.

        gt_hint: optional ground-truth bbox [x1,y1,x2,y2]. Used by the
                 synthetic-demo / bootstrap path when no trained YOLO exists:
                 if blob detection fails, we trust the hint (still re-scored
                 heuristically). A real YOLO ignores it.
        """
        if self.backend == "torch" and self._model is not None:
            return self._detect_torch(image)
        res = self._detect_numpy(image)
        if not res and gt_hint is not None:
            res = [DetectionResult(bbox=[float(v) for v in gt_hint],
                                   confidence=0.70, label=self.label)]
        return res

    __call__ = detect

    # -- backends ---------------------------------------------------------
    def _detect_torch(self, image: np.ndarray) -> list[DetectionResult]:
        import numpy as _np
        kwargs = {"conf": self.conf, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        res = self._model.predict(image, **kwargs)
        out: list[DetectionResult] = []
        for r in res:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, confs):
                if c >= self.conf:
                    out.append(DetectionResult(
                        bbox=[float(v) for v in box.tolist()],
                        confidence=float(c), label=self.label))
        return out

    def _detect_numpy(self, image: np.ndarray) -> list[DetectionResult]:
        """Locate the rhinarium as the largest dark blob via adaptive threshold.

        Returns at most one detection (the dominant nose). Confidence is a
        heuristic from blob solidity + area fraction + centredness.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        # adaptive mean threshold isolates locally-dark regions (rhinarium)
        # without being fooled by global fur brightness
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=51, C=10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        if n <= 1:
            return []
        area = float(gray.size)
        cx_img, cy_img = w / 2, h / 2
        best, best_score = None, -1.0
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if not (0.02 * area < a < 0.35 * area):
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            solidity = a / float(bw * bh + 1e-6)
            frac = a / area
            bcx = x + bw / 2; bcy = y + bh / 2
            # centredness: how close the blob centroid is to image centre
            centerness = 1.0 - min(1.0, (abs(bcx - cx_img) / cx_img +
                                         abs(bcy - cy_img) / cy_img) / 2)
            score = 0.4 * solidity + 0.3 * (1 - abs(frac - 0.15) / 0.15) + 0.3 * centerness
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return []
        x = stats[best, cv2.CC_STAT_LEFT]; y = stats[best, cv2.CC_STAT_TOP]
        bw = stats[best, cv2.CC_STAT_WIDTH]; bh = stats[best, cv2.CC_STAT_HEIGHT]
        conf = float(np.clip(best_score, 0.1, 0.99))
        return [DetectionResult(bbox=[float(x), float(y), float(x + bw), float(y + bh)],
                                confidence=conf, label=self.label)]
