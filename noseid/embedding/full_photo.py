"""Full-frame appearance embeddings used only for coarse retrieval.

The identity model in ``embedding.model`` is trained for normalized nose
crops.  It must not be applied directly to an ordinary dog photograph.  This
module therefore provides a separate, frozen full-frame representation.  It
uses the existing learned-free multi-scale image descriptor so the new
gallery can be built without retraining or downloading another checkpoint.

This vector is deliberately a candidate-generation signal.  The final
decision still comes from the whole-nose and anatomical-region embeddings.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import cfg_get, get_config
from ..features.extractor import _l2
from ..types import EmbeddingResult


def _color_descriptor(rgb: np.ndarray) -> np.ndarray:
    """Return a compact global colour descriptor for the complete photo."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    parts = []
    for channel, bins, upper in (
            (rgb[..., 0], 12, 256), (rgb[..., 1], 12, 256),
            (rgb[..., 2], 12, 256), (hsv[..., 0], 10, 180),
            (hsv[..., 1], 9, 256), (hsv[..., 2], 9, 256)):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, upper])
        parts.append(hist.reshape(-1).astype(np.float32))
    return _l2(np.concatenate(parts))


def _texture_descriptor(rgb: np.ndarray) -> np.ndarray:
    """Fast multi-scale edge and spatial colour descriptor."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    parts = []
    for size in (32, 64, 128):
        frame = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        hist, _ = np.histogram(angle, bins=8, range=(0.0, 1.0),
                               weights=magnitude)
        parts.extend((hist / (hist.sum() + 1e-6)).tolist())
        parts.extend((float(magnitude.mean()), float(magnitude.std())))
    # A 4x4 colour layout retains rough dog/coat/background composition while
    # staying insensitive to the original image resolution.
    small = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    parts.extend((small / 255.0).reshape(-1).tolist())
    return _l2(np.asarray(parts, dtype=np.float32))


class FullPhotoEmbedder:
    """Create one normalized embedding from the entire input photograph.

    The descriptor intentionally does not crop around the nose.  It contains
    multi-scale texture/edge information from the full frame plus colour
    statistics.  It is not exposed as the final biometric identity signal;
    ``FullPhotoCascadeIndex`` uses it only to add coarse candidates.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get_config()
        self.embed_dim = int(cfg_get(
            self.cfg, "full_photo.dim", cfg_get(self.cfg, "embedding.dim", 256)))
        self.backend = str(cfg_get(
            self.cfg, "full_photo.backend", "hybrid_numpy")).lower()

    @staticmethod
    def _validate(image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("full photo must be a non-empty numpy image")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("full photo must have three color channels")
        return image

    def embed(self, image: np.ndarray,
              dog_id: str | None = None) -> EmbeddingResult:
        image = self._validate(image)
        # This deliberately avoids SIFT/Gabor and any model download: full
        # photo retrieval should be inexpensive and must not slow the trained
        # nose pipeline.
        texture = _texture_descriptor(image)
        colour = _color_descriptor(image)
        raw = np.concatenate([texture, colour]).astype(np.float32)
        if raw.shape[0] > self.embed_dim:
            raw = raw[:self.embed_dim]
        elif raw.shape[0] < self.embed_dim:
            raw = np.pad(raw, (0, self.embed_dim - raw.shape[0]))
        embedding = _l2(raw)
        return EmbeddingResult(
            dog_id=dog_id or "?", embedding=embedding.tolist(),
            embedding_size=embedding.shape[0], source="full_photo_fast")

    __call__ = embed
