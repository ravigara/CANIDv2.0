"""Anatomy-aware feature crops and shared-encoder embeddings.

DoGNose stores one normalized embedding for an entire detector crop.  This
module keeps the same useful embedding properties (frozen image encoder,
unit-norm vectors, cosine matching, centroid enrollment) but applies the
encoder independently to the four anatomical regions produced by this
project:

    rhinarium, left_nare, right_nare, philtrum

The existing embedding checkpoint is deliberately reused as a shared encoder;
adding this representation does not invalidate or retrain the detector,
segmenter, landmark model, or current embedding checkpoint.  If a region is
not confidently available, it is omitted rather than replaced with a guessed
mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import cfg_get, get_config
from ..pipeline.crop import NormalizedNoseCrop, normalize_nose_crop
from ..types import EmbeddingResult, LandmarkResult


FEATURE_NAMES = ("rhinarium", "left_nare", "right_nare", "philtrum")
_LANDMARK_NAMES = {"left_nare": "left_nare", "right_nare": "right_nare",
                   "philtrum": "philtrum"}


@dataclass
class FeatureCrop:
    """One normalized crop for one anatomical region."""

    name: str
    image: np.ndarray
    mask: np.ndarray | None
    bbox: list[int]
    available: bool = True
    confidence: float = 1.0
    fallback_reason: str | None = None


@dataclass
class FeatureEmbeddingResult:
    """Embeddings and diagnostics for all available anatomical regions."""

    dog_id: str
    embeddings: dict[str, list[float]]
    feature_confidence: dict[str, float]
    features_present: list[str]
    embedding_size: int
    source: str = "unknown"
    representation: str = "noseid-v2-anatomy-features"
    crop_diagnostics: dict[str, dict] | None = None

    def to_dict(self) -> dict:
        return {
            "dog_id": self.dog_id,
            "embeddings": self.embeddings,
            "feature_confidence": self.feature_confidence,
            "features_present": self.features_present,
            "embedding_size": self.embedding_size,
            "source": self.source,
            "representation": self.representation,
            "crop_diagnostics": self.crop_diagnostics or {},
        }


def _mask_for(normalized: NormalizedNoseCrop, name: str,
              classes: list[str] | None) -> np.ndarray | None:
    if normalized.masks is None or not classes or name not in classes:
        return None
    channel = normalized.masks[..., classes.index(name)]
    return (channel > 0).astype(np.uint8)


def _landmark_for(landmarks: LandmarkResult | None,
                  name: str) -> np.ndarray | None:
    if landmarks is None or name not in _LANDMARK_NAMES:
        return None
    point = np.asarray(getattr(landmarks, _LANDMARK_NAMES[name]), dtype=np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def _confidence_for(landmarks: LandmarkResult | None, name: str) -> float:
    if landmarks is None or not landmarks.confidence:
        return 1.0
    # The rhinarium is a segmentation feature, not a landmark. Missing
    # landmark confidence must not turn a valid rhinarium crop into a zero-
    # confidence diagnostic.
    return float(landmarks.confidence.get(name, 1.0))


def _square_crop(image: np.ndarray, mask: np.ndarray | None,
                 center: np.ndarray, side: float, target: int,
                 suppress_background: bool) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """Extract a square region with reflection padding and optional masking."""
    h, w = image.shape[:2]
    side = max(8.0, float(side))
    cx, cy = float(center[0]), float(center[1])
    x1 = int(np.floor(cx - side / 2.0))
    y1 = int(np.floor(cy - side / 2.0))
    x2 = int(np.ceil(cx + side / 2.0))
    y2 = int(np.ceil(cy + side / 2.0))
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - w), max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101)
        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0)
        x1 += pad_left; x2 += pad_left
        y1 += pad_top; y2 += pad_top
    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2] if mask is not None else None
    if crop.size == 0:
        raise ValueError("anatomical feature crop is empty")

    if suppress_background and crop_mask is not None and crop_mask.any():
        kernel_size = max(3, int(round(side / 18.0)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(crop_mask, kernel)
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(1.5, side / 20.0))
        if crop.ndim == 3:
            crop = np.where(expanded[..., None] > 0, crop, blurred)
        else:
            crop = np.where(expanded > 0, crop, blurred)

    crop = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
    if crop_mask is not None:
        crop_mask = cv2.resize(crop_mask, (target, target),
                               interpolation=cv2.INTER_NEAREST)
    return (np.ascontiguousarray(crop),
            None if crop_mask is None else np.ascontiguousarray(crop_mask),
            [x1 - pad_left, y1 - pad_top, x2 - pad_left, y2 - pad_top])


def extract_feature_crops(normalized: NormalizedNoseCrop,
                          classes: list[str] | None,
                          target_size: int = 224,
                          min_mask_area: int = 12,
                          suppress_background: bool = True) -> dict[str, FeatureCrop]:
    """Build stable, region-specific crops from a normalized anatomy crop.

    Segmentation masks define the crop whenever they contain enough pixels.
    Nare and philtrum landmarks are used only as a geometric fallback; a
    missing rhinarium mask is not fabricated, because it would contaminate the
    identity representation with arbitrary background.
    """
    image = normalized.image
    h, w = image.shape[:2]
    result: dict[str, FeatureCrop] = {}
    for name in FEATURE_NAMES:
        mask = _mask_for(normalized, name, classes)
        point = _landmark_for(normalized.landmarks, name)
        confidence = _confidence_for(normalized.landmarks, name)
        fallback_reason = None
        if mask is not None and int(mask.sum()) >= min_mask_area:
            ys, xs = np.where(mask > 0)
            center = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
            extent = max(float(xs.max() - xs.min() + 1),
                         float(ys.max() - ys.min() + 1))
            # Keep contextual texture around small parts. The lower bound is
            # important for the narrow philtrum and tiny predicted nares.
            min_side = target_size * (0.72 if name == "rhinarium" else 0.24)
            side = max(extent * (1.20 if name == "rhinarium" else 2.20), min_side)
        elif name != "rhinarium" and point is not None:
            center = point
            side = target_size * 0.28
            fallback_reason = "mask_missing_or_small_landmark_fallback"
        else:
            # Preserve an explicit unavailable region. Callers can require a
            # minimum number of regions before enrollment or identification.
            result[name] = FeatureCrop(
                name=name, image=np.zeros((target_size, target_size, 3),
                                          dtype=np.uint8), mask=None,
                bbox=[0, 0, 0, 0], available=False, confidence=0.0,
                fallback_reason="mask_missing")
            continue
        crop, crop_mask, bbox = _square_crop(
            image, mask, center, side, target_size, suppress_background)
        result[name] = FeatureCrop(
            name=name, image=crop, mask=crop_mask, bbox=bbox,
            available=True, confidence=max(0.0, min(1.0, confidence)),
            fallback_reason=fallback_reason)
    return result


class FeatureEmbedder:
    """Apply the existing image embedder independently to anatomy regions."""

    def __init__(self, embedder, cfg: dict | None = None):
        self.embedder = embedder
        self.cfg = cfg or get_config()
        self.target_size = int(cfg_get(self.cfg, "embedding.feature_crop_size",
                                      cfg_get(self.cfg, "image.crop_size", 224)))
        self.min_mask_area = int(cfg_get(self.cfg, "embedding.feature_min_mask_area", 12))
        self.suppress_background = bool(cfg_get(
            self.cfg, "embedding.feature_mask_background", True))
        self.min_confidence = float(cfg_get(
            self.cfg, "landmarks.min_confidence", 0.50))
        self.representation = str(cfg_get(
            self.cfg, "matching.embedding_version",
            "noseid-v2-anatomy-features"))

    def embed(self, image: np.ndarray,
              landmarks: LandmarkResult | None = None,
              masks: np.ndarray | None = None,
              classes: list[str] | None = None,
              bbox: list[float] | None = None,
              dog_id: str | None = None) -> FeatureEmbeddingResult:
        normalized = normalize_nose_crop(
            image, bbox, landmarks, masks, classes,
            target_size=self.target_size,
            pad=float(cfg_get(self.cfg, "embedding.crop_pad", 0.15)),
            min_confidence=self.min_confidence,
            mask_background=bool(cfg_get(self.cfg, "embedding.mask_background", True)),
        )
        crops = extract_feature_crops(
            normalized, classes, target_size=self.target_size,
            min_mask_area=self.min_mask_area,
            suppress_background=self.suppress_background,
        )
        embeddings: dict[str, list[float]] = {}
        confidence: dict[str, float] = {}
        diagnostics: dict[str, dict] = {}
        source = "unknown"
        for name, feature in crops.items():
            diagnostics[name] = {
                "available": feature.available,
                "bbox": feature.bbox,
                "confidence": feature.confidence,
                "fallback_reason": feature.fallback_reason,
            }
            if not feature.available:
                continue
            embedded: EmbeddingResult = self.embedder.embed_crop(feature.image)
            embeddings[name] = embedded.embedding
            confidence[name] = feature.confidence
            source = embedded.source
        return FeatureEmbeddingResult(
            dog_id=dog_id or "?", embeddings=embeddings,
            feature_confidence=confidence,
            features_present=sorted(embeddings),
            embedding_size=self.embedder.embed_dim,
            source=source, representation=self.representation,
            crop_diagnostics=diagnostics,
        )

    __call__ = embed
