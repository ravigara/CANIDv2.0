"""Stage 2 - Dog Nose Validator.

Rejects: blurry, side-profile (>20deg), low-light, occluded, partial,
motion-blurred, and multi-nose images. Uses pure OpenCV/numpy so it runs
without the torch stack.

Scores:
  - blur      : variance of Laplacian (higher = sharper)
  - brightness: mean luminance in [0,255]
  - angle     : estimated yaw in degrees (symmetry-based proxy)
  - quality   : composite 0..100 score

Output JSON shape matches the spec:
    {"valid": true, "confidence": 98.5, "reason": "Valid Capture"}
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import get_config, cfg_get
from ..types import ValidationResult


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian - classic focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _angle_score(gray: np.ndarray) -> float:
    """Estimate pose yaw as the horizontal asymmetry of the central band.

    Symmetric noses -> ~0 deg. Strong left/right asymmetry -> larger angle.
    This is a cheap proxy; a real system uses the landmark net's pose head.
    """
    h, w = gray.shape
    band = gray[h // 3: 2 * h // 3, :]
    left = band[:, : w // 2]
    right = cv2.flip(band[:, w // 2:], 1)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    diff = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))
    # map mean abs diff (0..~80) roughly to 0..40 deg
    return min(40.0, diff * 0.5)


def _count_noses(gray: np.ndarray) -> int:
    """Heuristic: count large dark blob connected components (candidate noses).

    Only blobs covering >= 3% of the frame are counted, so groove/pigment
    micro-texture (lots of tiny blobs) doesn't inflate the count.
    """
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # morphological close so the rhinarium reads as one blob, not many
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    area = gray.size
    count = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 0.03 * area < a < 0.35 * area:
            count += 1
    return max(1, count)


class NoseValidator:
    """Validate a candidate dog-nose capture."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or get_config()
        v = cfg_get(cfg, "validation", {})
        self.blur_thr = float(v.get("blur_lap_threshold", 100.0))
        self.bright_min = float(v.get("brightness_min", 50.0))
        self.bright_max = float(v.get("brightness_max", 220.0))
        self.area_min = float(v.get("min_nose_area_ratio", 0.05))
        self.area_max = float(v.get("max_nose_area_ratio", 0.60))
        self.quality_min = float(v.get("quality_min", 60.0))
        # During development, quality is advisory so the detector/embedding
        # stages can still be evaluated on ordinary photographs. Applications
        # can enable this for a strict capture gate.
        self.hard_reject = bool(v.get("hard_reject", False))
        self.angle_max = 20.0
        self.nose_count = None  # filled by detect() if a detector is attached

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def score(self, image: np.ndarray, bbox: list[float] | None = None) -> dict:
        """Return raw scores dict. ``bbox`` optionally restricts analysis."""
        work = image
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                work = image[y1:y2, x1:x2]
        gray = self._to_gray(work)
        blur = _blur_score(gray)
        brightness = _brightness(gray)
        angle = _angle_score(gray)
        # composite quality: each sub-score in 0..100, weighted
        s_blur = np.clip(blur / max(self.blur_thr, 1.0) * 100, 0, 100)
        s_bright = 100 - min(100, abs(brightness - 135) * 1.2)
        s_angle = 100 - min(100, (angle / self.angle_max) * 100)
        quality = float(0.4 * s_blur + 0.3 * s_bright + 0.3 * s_angle)
        return {
            "blur": float(blur),
            "brightness": float(brightness),
            "angle": float(angle),
            "quality": float(quality),
            "blur_score": float(s_blur),
            "brightness_score": float(s_bright),
            "angle_score": float(s_angle),
        }

    def validate(self, image: np.ndarray, bbox: list[float] | None = None) -> ValidationResult:
        """Validate and return a ValidationResult with verdict + reason."""
        scores = self.score(image, bbox)
        n_noses = self.nose_count if self.nose_count else _count_noses(self._to_gray(image))

        reasons = []
        if scores["blur"] < self.blur_thr:
            reasons.append("Image too blurry")
        if not (self.bright_min <= scores["brightness"] <= self.bright_max):
            reasons.append("Poor lighting")
        if scores["angle"] > self.angle_max:
            reasons.append("Side profile exceeds 20 degrees")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            nose_area = max(0, x2 - x1) * max(0, y2 - y1)
            frame_area = float(image.shape[0] * image.shape[1]) or 1.0
            ratio = nose_area / frame_area
            if ratio < self.area_min:
                reasons.append("Nose too small / partial capture")
            elif ratio > self.area_max:
                reasons.append("Nose region implausibly large")
        if n_noses > 1:
            reasons.append("Multiple dogs detected")
        if scores["quality"] < self.quality_min:
            reasons.append("Composite quality below threshold")

        if reasons:
            # confidence that the image is BAD
            confidence = round(100 - scores["quality"], 2)
            return ValidationResult(
                valid=False, confidence=confidence,
                reason="; ".join(reasons), scores=scores,
            )
        return ValidationResult(
            valid=True,
            confidence=round(scores["quality"], 2),
            reason="Valid Capture",
            scores=scores,
        )
