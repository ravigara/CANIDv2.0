"""Shared result/dataclass types used across pipeline stages.

Plain dataclasses (not attrs/pydantic) so there are zero extra dependencies
for the core pipeline. They serialize to the JSON shapes in the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


def _to_jsonable(o: Any) -> Any:
    import numpy as np
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


@dataclass
class JSONMixin:
    def to_dict(self) -> dict:
        return _to_jsonable(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# Stage 2 ---------------------------------------------------------------
@dataclass
class ValidationResult(JSONMixin):
    valid: bool
    confidence: float
    reason: str
    scores: dict = field(default_factory=dict)   # blur, brightness, angle, quality


# Stage 3 ---------------------------------------------------------------
@dataclass
class DetectionResult(JSONMixin):
    bbox: list[float]          # [x1, y1, x2, y2]
    confidence: float
    label: str = "NOSE01"


# Stage 4 ---------------------------------------------------------------
@dataclass
class LandmarkResult(JSONMixin):
    left_nare: list[float]
    right_nare: list[float]
    philtrum: list[float]
    nose_boundary: list[list[float]] = field(default_factory=list)
    fold_lines: list[list[float]] = field(default_factory=list)
    texture_regions: list[list[float]] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    valid: bool = True


# Stage 5 ---------------------------------------------------------------
@dataclass
class SegmentationResult(JSONMixin):
    # per-class mask availability + mean confidence
    classes: list[str]
    present: dict[str, bool]
    iou: float = 0.0
    dice: float = 0.0
    # mask arrays kept out of JSON dump path (use to_dict -> lists)
    masks: Any = None   # np.ndarray HxWxC uint8, optional

    def to_dict(self) -> dict:
        import numpy as np
        d = _to_jsonable({k: v for k, v in asdict(self).items() if k != "masks"})
        d["has_masks"] = self.masks is not None
        return d


# Stage 6 ---------------------------------------------------------------
@dataclass
class FeatureVector(JSONMixin):
    deep: list[float]
    classical: list[float]
    morphological: list[float]
    fused_dim: int = 0


# Stage 7/8/9 -----------------------------------------------------------
@dataclass
class EmbeddingResult(JSONMixin):
    dog_id: str
    embedding: list[float]
    embedding_size: int
    source: str = "unknown"      # 'torch' | 'numpy'


@dataclass
class EnrollmentResult(JSONMixin):
    dog_id: str
    embedding_size: int
    num_images: int
    mean_confidence: float
    status: str = "Enrolled"


@dataclass
class IdentificationResult(JSONMixin):
    dog_id: str | None
    similarity: float
    confidence: float
    status: str                  # 'Verified' | 'Unknown Dog' | 'Rejected'
    candidates: list[dict] = field(default_factory=list)
