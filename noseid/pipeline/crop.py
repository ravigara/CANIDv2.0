"""Shared detector-crop and anatomy normalization utilities."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import LandmarkResult


@dataclass
class NormalizedNoseCrop:
    image: np.ndarray
    masks: np.ndarray | None
    landmarks: LandmarkResult | None
    source_bbox: list[int]
    aligned: bool
    mask_used: bool
    fallback_reason: str | None = None


def padded_bbox(image_shape: tuple[int, ...], bbox: list[float] | None,
                pad: float = 0.15) -> tuple[int, int, int, int]:
    """Clip a detector box with proportional padding to image bounds."""
    h, w = image_shape[:2]
    if bbox is None:
        bbox = [0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid detector bbox: {bbox}")
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(round(x1 - pad * bw))),
        max(0, int(round(y1 - pad * bh))),
        min(w, int(round(x2 + pad * bw))),
        min(h, int(round(y2 + pad * bh))),
    )


def _valid_points(landmarks: LandmarkResult | None,
                  min_confidence: float) -> tuple[np.ndarray | None, str | None]:
    if landmarks is None:
        return None, "landmarks_missing"
    if not landmarks.valid:
        return None, "landmarks_invalid"
    if landmarks.confidence and any(
            float(landmarks.confidence.get(name, 0.0)) < min_confidence
            for name in ("left_nare", "right_nare", "philtrum")):
        return None, "landmarks_low_confidence"
    points = np.asarray([
        landmarks.left_nare, landmarks.right_nare, landmarks.philtrum,
    ], dtype=np.float32)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return None, "landmarks_non_finite"
    if np.linalg.norm(points[0] - points[1]) < 2.0:
        return None, "nares_too_close"
    return points, None


def _warp_masks(masks: np.ndarray | None, matrix: np.ndarray | None,
                crop_box: tuple[int, int, int, int], target: int) -> np.ndarray | None:
    if masks is None or masks.ndim != 3:
        return None
    x1, y1, x2, y2 = crop_box
    channels = []
    if matrix is not None:
        for c in range(masks.shape[-1]):
            channels.append(cv2.warpAffine(
                masks[..., c].astype(np.uint8), matrix, (target, target),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0))
    else:
        local = masks[y1:y2, x1:x2]
        if local.size == 0:
            return None
        h, w = local.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        padded = np.zeros((side, side, local.shape[-1]), dtype=np.uint8)
        padded[top:top + h, left:left + w] = local
        for c in range(padded.shape[-1]):
            channels.append(cv2.resize(
                padded[..., c], (target, target), interpolation=cv2.INTER_NEAREST))
    return np.stack(channels, axis=-1).astype(np.uint8)


def _transform_landmarks(landmarks: LandmarkResult | None,
                         matrix: np.ndarray | None,
                         crop_box: tuple[int, int, int, int],
                         target: int,
                         min_confidence: float) -> LandmarkResult | None:
    points, _ = _valid_points(landmarks, min_confidence)
    if points is None:
        return None
    if matrix is not None:
        out = cv2.transform(points[None, ...], matrix)[0]
    else:
        x1, y1, x2, y2 = crop_box
        side = max(y2 - y1, x2 - x1)
        top, left = (side - (y2 - y1)) // 2, (side - (x2 - x1)) // 2
        out = points.copy()
        out[:, 0] = (out[:, 0] - x1 + left) * target / side
        out[:, 1] = (out[:, 1] - y1 + top) * target / side
    return LandmarkResult(
        left_nare=out[0].astype(float).tolist(),
        right_nare=out[1].astype(float).tolist(),
        philtrum=out[2].astype(float).tolist(),
        confidence=dict(landmarks.confidence),
        valid=landmarks.valid,
    )


def normalize_nose_crop(
    image: np.ndarray,
    bbox: list[float] | None,
    landmarks: LandmarkResult | None = None,
    masks: np.ndarray | None = None,
    classes: list[str] | None = None,
    target_size: int = 224,
    pad: float = 0.15,
    min_confidence: float = 0.50,
    mask_background: bool = True,
) -> NormalizedNoseCrop:
    """Create the canonical identity crop used by training and inference."""
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty HxW or HxWxC array")
    target = int(target_size)
    if target < 16:
        raise ValueError("target_size must be at least 16")
    box = padded_bbox(image.shape, bbox, pad)
    x1, y1, x2, y2 = box

    points, reason = _valid_points(landmarks, min_confidence)
    matrix = None
    aligned = False
    if points is not None:
        destination = np.asarray([
            [0.30 * target, 0.38 * target],
            [0.70 * target, 0.38 * target],
            [0.50 * target, 0.70 * target],
        ], dtype=np.float32)
        matrix = cv2.getAffineTransform(points, destination)
        if np.isfinite(matrix).all():
            crop = cv2.warpAffine(
                image, matrix, (target, target),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            aligned = True
        else:
            matrix = None
            reason = "alignment_failed"
    if not aligned:
        local = image[y1:y2, x1:x2]
        if local.size == 0:
            raise ValueError("detector crop is empty")
        h, w = local.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        border = cv2.copyMakeBorder(
            local, top, side - h - top, left, side - w - left,
            cv2.BORDER_REFLECT_101)
        crop = cv2.resize(border, (target, target), interpolation=cv2.INTER_AREA)

    normalized_masks = _warp_masks(masks, matrix, box, target)
    normalized_landmarks = _transform_landmarks(
        landmarks, matrix, box, target, min_confidence)
    mask_used = False
    if mask_background and normalized_masks is not None and classes:
        try:
            rhinarium_id = classes.index("rhinarium")
            rhinarium = normalized_masks[..., rhinarium_id] > 0
            area_ratio = float(rhinarium.mean())
            if 0.02 <= area_ratio <= 0.90:
                expanded = cv2.dilate(
                    rhinarium.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                ).astype(bool)
                points_ok = True
                if normalized_landmarks is not None:
                    points = np.asarray([
                        normalized_landmarks.left_nare,
                        normalized_landmarks.right_nare,
                        normalized_landmarks.philtrum,
                    ], dtype=np.float32).round().astype(int)
                    points_ok = all(
                        0 <= int(x) < target and 0 <= int(y) < target
                        and bool(expanded[int(y), int(x)])
                        for x, y in points)
                if points_ok:
                    fill = cv2.GaussianBlur(
                        crop, (0, 0), sigmaX=max(2.0, target / 24.0))
                    crop = np.where(expanded[..., None], crop, fill).astype(np.uint8)
                    mask_used = True
                else:
                    reason = reason or "rhinarium_mask_misses_landmark"
        except ValueError:
            reason = reason or "rhinarium_class_missing"

    return NormalizedNoseCrop(
        image=np.ascontiguousarray(crop),
        masks=normalized_masks,
        landmarks=normalized_landmarks,
        source_bbox=list(box),
        aligned=aligned,
        mask_used=mask_used,
        fallback_reason=reason,
    )
