"""Two-stage whole-nose retrieval plus anatomy-feature reranking.

``CascadeFeatureIndex`` extends the active anatomy gallery without changing
the old ``FeatureIndex`` format.  Each dog has:

* one normalized whole-nose centroid for coarse candidate retrieval; and
* one normalized centroid per available anatomical feature for reranking.

The whole-nose search is deliberately a candidate generator, not a hard
accept/reject gate.  This prevents blur, glare, or a partial nose from
discarding the correct dog before the more specific anatomy comparison runs.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..config import cfg_get, get_config
from ..embedding.features import FEATURE_NAMES
from ..types import EnrollmentResult, IdentificationResult
from .feature_index import FeatureIndex, _normalize


def _rows(values: list[np.ndarray] | np.ndarray,
          dim: int, label: str) -> tuple[list[np.ndarray], np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = [values]
    rows = [_normalize(np.asarray(value, dtype=np.float32)) for value in values]
    if not rows:
        raise ValueError(f"{label} embeddings cannot be empty")
    if any(row.shape[0] != dim for row in rows):
        raise ValueError(f"{label} embeddings must have dimension {dim}")
    return rows, _normalize(np.mean(np.stack(rows), axis=0))


class CascadeFeatureIndex(FeatureIndex):
    """Whole-nose candidate retrieval followed by feature reranking."""

    FORMAT = "noseid-cascade-gallery-v1"

    def __init__(self, dim: int = 256, cfg: dict | None = None):
        super().__init__(dim=dim, cfg=cfg)
        self.embedding_version = str(cfg_get(
            self.cfg, "matching.embedding_version", "noseid-v3-cascade"))
        self.retrieval_k = int(cfg_get(
            self.cfg, "matching.cascade_retrieval_k", 50))
        self.whole_weight = float(cfg_get(
            self.cfg, "matching.cascade_whole_weight", 0.25))
        self.feature_weight = float(cfg_get(
            self.cfg, "matching.cascade_feature_weight", 0.75))
        if self.whole_weight < 0 or self.feature_weight < 0:
            raise ValueError("cascade weights must be non-negative")
        if self.whole_weight + self.feature_weight <= 0:
            raise ValueError("cascade weights must have a positive sum")
        self._whole_templates: dict[str, np.ndarray] = {}
        self._whole_raw: dict[str, list[np.ndarray]] = {}

    def enroll(self, dog_id: str,
               whole_embeddings: list[np.ndarray] | np.ndarray,
               feature_embeddings: dict[str, list[np.ndarray] | np.ndarray],
               metadata: dict | None = None,
               num_images: int | None = None) -> EnrollmentResult:
        """Enroll whole-nose and anatomy templates atomically."""
        dog_id = str(dog_id)
        whole_rows, whole_template = _rows(
            whole_embeddings, self.dim, "whole-nose")
        # Validate the whole template before FeatureIndex.enroll mutates its
        # feature dictionaries.  This keeps failed enrollment all-or-nothing.
        result = super().enroll(
            dog_id, feature_embeddings, metadata=metadata,
            num_images=num_images,
        )
        self._whole_templates[dog_id] = whole_template
        self._whole_raw[dog_id] = whole_rows
        whole_similarity = float(np.mean([row @ whole_template for row in whole_rows]))
        self._records[dog_id]["whole_nose_count"] = len(whole_rows)
        self._records[dog_id]["whole_nose_mean_similarity"] = round(
            whole_similarity, 6)
        self._records[dog_id]["cascade"] = {
            "retrieval_k": self.retrieval_k,
            "whole_weight": self.whole_weight,
            "feature_weight": self.feature_weight,
        }
        return result

    def _combined_score(self, whole_score: float,
                        feature_score: float) -> float:
        total = self.whole_weight + self.feature_weight
        return ((self.whole_weight * whole_score) +
                (self.feature_weight * feature_score)) / total

    def identify(self, whole_embedding: np.ndarray,
                 feature_embeddings: dict[str, np.ndarray],
                 top_k: int | None = None) -> IdentificationResult:
        """Retrieve by whole nose, then rerank with anatomy features."""
        query_whole = _normalize(whole_embedding)
        query_features = {
            name: _normalize(value)
            for name, value in feature_embeddings.items()
            if name in FEATURE_NAMES
        }
        if len(query_features) < self.min_features:
            return IdentificationResult(
                dog_id=None, similarity=0.0, confidence=0.0,
                status="Rejected", candidates=[{
                    "reason": "Insufficient anatomical features",
                    "features_available": sorted(query_features),
                }])
        if not self._whole_templates:
            return IdentificationResult(
                dog_id=None, similarity=0.0, confidence=0.0,
                status="Unknown Dog", candidates=[])

        coarse = sorted(
            ((dog_id, float(query_whole @ template))
             for dog_id, template in self._whole_templates.items()),
            key=lambda item: item[1], reverse=True,
        )
        retrieval_limit = min(
            len(coarse), max(self.retrieval_k, top_k or self.top_k))
        candidates = []
        for retrieval_rank, (dog_id, whole_score) in enumerate(
                coarse[:retrieval_limit], start=1):
            template = self._templates.get(dog_id, {})
            scored = self._score(query_features, template)
            if scored is None:
                continue
            feature_score, feature_scores = scored
            combined = self._combined_score(whole_score, feature_score)
            candidates.append({
                "dog_id": dog_id,
                "similarity": round(combined, 4),
                "confidence": round(combined, 4),
                "whole_nose_similarity": round(whole_score, 4),
                "feature_similarity": round(feature_score, 4),
                "feature_scores": feature_scores,
                "features_used": sorted(feature_scores),
                "whole_nose_retrieval_rank": retrieval_rank,
            })
        candidates.sort(key=lambda row: row["similarity"], reverse=True)
        candidates = candidates[:min(top_k or self.top_k, len(candidates))]
        if not candidates:
            return IdentificationResult(
                dog_id=None, similarity=0.0, confidence=0.0,
                status="Unknown Dog", candidates=[])

        best = candidates[0]
        similarity = float(best["similarity"])
        top2 = candidates[1]["similarity"] if len(candidates) > 1 else None
        margin = similarity - float(top2) if top2 is not None else 1.0
        best["margin"] = round(margin, 4)
        if similarity > self.verify_thr and margin >= self.margin_thr:
            status = "Verified"
            dog_id = best["dog_id"]
        elif similarity < self.reject_thr:
            status = "Unknown Dog"
            dog_id = None
        else:
            status = "Rejected"
            dog_id = None
        return IdentificationResult(
            dog_id=dog_id, similarity=round(similarity, 4),
            confidence=round(similarity, 4), status=status,
            candidates=candidates)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "dim": self.dim,
            "verify_thr": self.verify_thr,
            "reject_thr": self.reject_thr,
            "margin_thr": self.margin_thr,
            "top_k": self.top_k,
            "min_features": self.min_features,
            "embedding_version": self.embedding_version,
            "feature_weights": self.feature_weights,
            "retrieval_k": self.retrieval_k,
            "whole_weight": self.whole_weight,
            "feature_weight": self.feature_weight,
            "whole_templates": {
                dog_id: vector.tolist()
                for dog_id, vector in self._whole_templates.items()
            },
            "templates": {
                dog_id: {name: vector.tolist() for name, vector in features.items()}
                for dog_id, features in self._templates.items()
            },
            "records": self._records,
        }
        with open(os.path.join(dir_path, "cascade_meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "CascadeFeatureIndex":
        path = os.path.join(dir_path, "cascade_meta.json")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported cascade gallery format: {payload.get('format')}")
        cfg = get_config({
            "matching": {
                "verify_threshold": payload.get("verify_thr", 0.50),
                "reject_threshold": payload.get("reject_thr", 0.20),
                "margin_threshold": payload.get("margin_thr", 0.00),
                "top_k": payload.get("top_k", 5),
                "min_features": payload.get("min_features", 2),
                "embedding_version": payload.get("embedding_version",
                                               "noseid-v3-cascade"),
                "feature_weights": payload.get("feature_weights", {}),
                "cascade_retrieval_k": payload.get("retrieval_k", 50),
                "cascade_whole_weight": payload.get("whole_weight", 0.25),
                "cascade_feature_weight": payload.get("feature_weight", 0.75),
            }
        })
        index = cls(dim=int(payload["dim"]), cfg=cfg)
        index._whole_templates = {
            dog_id: _normalize(np.asarray(vector, dtype=np.float32))
            for dog_id, vector in payload.get("whole_templates", {}).items()
        }
        index._templates = {
            dog_id: {name: _normalize(np.asarray(vector, dtype=np.float32))
                     for name, vector in features.items()}
            for dog_id, features in payload.get("templates", {}).items()
        }
        index._records = payload.get("records", {})
        return index

    def delete(self, dog_id: str) -> bool:
        existed = super().delete(dog_id)
        self._whole_templates.pop(str(dog_id), None)
        self._whole_raw.pop(str(dog_id), None)
        return existed
