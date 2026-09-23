"""Persistent anatomy-specific identity gallery.

The legacy :class:`NoseIndex` stores one vector per dog.  ``FeatureIndex``
stores a centroid for each available anatomical region instead and combines
per-region cosine scores at query time.  This keeps the gallery useful when a
single region is blurred or occluded, while preventing a whole-nose template
from hiding which anatomy carried the match.

The on-disk format is JSON for portability and auditability.  The old FAISS
``meta.json``/``nose.index`` files are never overwritten, so existing galleries
remain readable as legacy artifacts during migration.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..config import cfg_get, get_config
from ..embedding.features import FEATURE_NAMES
from ..types import EnrollmentResult, IdentificationResult


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("embedding must have non-zero norm")
    return (vector / norm).astype(np.float32)


class FeatureIndex:
    """Exact cosine gallery of per-anatomical-feature templates."""

    FORMAT = "noseid-feature-gallery-v1"

    def __init__(self, dim: int = 256, cfg: dict | None = None):
        self.cfg = cfg or get_config()
        self.dim = int(dim)
        self.verify_thr = float(cfg_get(self.cfg, "matching.verify_threshold", 0.50))
        self.reject_thr = float(cfg_get(self.cfg, "matching.reject_threshold", 0.20))
        self.margin_thr = float(cfg_get(self.cfg, "matching.margin_threshold", 0.00))
        self.top_k = int(cfg_get(self.cfg, "matching.top_k", 5))
        self.min_features = int(cfg_get(self.cfg, "matching.min_features", 2))
        self.embedding_version = str(cfg_get(
            self.cfg, "matching.embedding_version", "noseid-v2-anatomy-features"))
        configured_weights = cfg_get(self.cfg, "matching.feature_weights", {}) or {}
        self.feature_weights = {
            name: float(configured_weights.get(name, 1.0))
            for name in FEATURE_NAMES
        }
        self._templates: dict[str, dict[str, np.ndarray]] = {}
        self._raw: dict[str, dict[str, list[np.ndarray]]] = {}
        self._records: dict[str, dict] = {}

    @property
    def size(self) -> int:
        return len(self._templates)

    def enroll(self, dog_id: str,
               feature_embeddings: dict[str, list[np.ndarray] | np.ndarray],
               metadata: dict | None = None,
               num_images: int | None = None) -> EnrollmentResult:
        """Average each feature independently and register the dog template."""
        dog_id = str(dog_id)
        templates: dict[str, np.ndarray] = {}
        raw: dict[str, list[np.ndarray]] = {}
        feature_similarity: dict[str, float] = {}
        for name in FEATURE_NAMES:
            values = feature_embeddings.get(name)
            if values is None:
                continue
            if isinstance(values, np.ndarray) and values.ndim == 1:
                values = [values]
            rows = [_normalize(np.asarray(value, dtype=np.float32))
                    for value in values]
            if not rows:
                continue
            if any(row.shape[0] != self.dim for row in rows):
                raise ValueError(f"{name} embeddings must have dimension {self.dim}")
            template = _normalize(np.mean(np.stack(rows), axis=0))
            templates[name] = template
            raw[name] = rows
            feature_similarity[name] = float(np.mean([row @ template for row in rows]))

        if len(templates) < self.min_features:
            raise ValueError(
                f"only {len(templates)} anatomical features available for {dog_id}; "
                f"need >= {self.min_features}")

        self._templates[dog_id] = templates
        self._raw[dog_id] = raw
        mean_similarity = float(np.mean(list(feature_similarity.values())))
        self._records[dog_id] = {
            "dog_id": dog_id,
            "embedding_version": self.embedding_version,
            "features": sorted(templates),
            "feature_counts": {name: len(rows) for name, rows in raw.items()},
            "feature_mean_similarity": {
                name: round(value, 6)
                for name, value in feature_similarity.items()
            },
            "num_images": int(num_images or max(len(rows) for rows in raw.values())),
            "metadata": dict(metadata or {}),
        }
        return EnrollmentResult(
            dog_id=dog_id,
            embedding_size=self.dim,
            num_images=int(num_images or max(len(rows) for rows in raw.values())),
            mean_confidence=round(mean_similarity, 4),
            status="Enrolled",
            embedding_version=self.embedding_version,
        )

    def _score(self, query: dict[str, np.ndarray],
               template: dict[str, np.ndarray]) -> tuple[float, dict[str, float]] | None:
        scores: dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name in query and name in template:
                scores[name] = float(_normalize(query[name]) @ template[name])
        if len(scores) < self.min_features:
            return None
        weights = np.asarray([self.feature_weights[name] for name in scores], dtype=np.float32)
        values = np.asarray(list(scores.values()), dtype=np.float32)
        aggregate = float(np.average(values, weights=weights))
        return aggregate, {name: round(value, 4) for name, value in scores.items()}

    def identify(self, feature_embeddings: dict[str, np.ndarray],
                 top_k: int | None = None) -> IdentificationResult:
        """Search with all available regions and apply threshold + margin."""
        query = {name: _normalize(value) for name, value in feature_embeddings.items()
                 if name in FEATURE_NAMES}
        if len(query) < self.min_features:
            return IdentificationResult(
                dog_id=None, similarity=0.0, confidence=0.0,
                status="Rejected", candidates=[{
                    "reason": "Insufficient anatomical features",
                    "features_available": sorted(query),
                }])
        candidates = []
        for dog_id, template in self._templates.items():
            scored = self._score(query, template)
            if scored is None:
                continue
            similarity, feature_scores = scored
            candidates.append({
                "dog_id": dog_id,
                "similarity": round(similarity, 4),
                "feature_scores": feature_scores,
                "features_used": sorted(feature_scores),
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
        # The active UI contract is top-1 identification: the highest-scoring
        # registered dog is accepted only when its confidence is strictly
        # above the configured threshold.  Margin remains in the response for
        # review, but must not turn a >50% top candidate into "not found".
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
            "templates": {
                dog_id: {name: vector.tolist() for name, vector in features.items()}
                for dog_id, features in self._templates.items()
            },
            "records": self._records,
        }
        with open(os.path.join(dir_path, "feature_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "FeatureIndex":
        path = os.path.join(dir_path, "feature_meta.json")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported feature gallery format: {payload.get('format')}")
        cfg = get_config({
            "matching": {
                "verify_threshold": payload.get("verify_thr", 0.50),
                "reject_threshold": payload.get("reject_thr", 0.20),
                "margin_threshold": payload.get("margin_thr", 0.00),
                "top_k": payload.get("top_k", 5),
                "min_features": payload.get("min_features", 2),
                "embedding_version": payload.get("embedding_version",
                                               "noseid-v2-anatomy-features"),
                "feature_weights": payload.get("feature_weights", {}),
            }
        })
        index = cls(dim=int(payload["dim"]), cfg=cfg)
        index._templates = {
            dog_id: {name: _normalize(np.asarray(vector, dtype=np.float32))
                     for name, vector in features.items()}
            for dog_id, features in payload.get("templates", {}).items()
        }
        index._records = payload.get("records", {})
        return index

    def get_record(self, dog_id: str) -> dict | None:
        record = self._records.get(str(dog_id))
        return dict(record) if record is not None else None

    def update_metadata(self, dog_id: str, metadata: dict) -> None:
        """Merge display metadata without changing any feature templates."""
        record = self._records.get(str(dog_id))
        if record is None:
            raise KeyError(dog_id)
        current = dict(record.get("metadata", {}))
        current.update(metadata)
        record["metadata"] = current

    def delete(self, dog_id: str) -> bool:
        """Remove one dog template and its enrollment metadata."""
        dog_id = str(dog_id)
        existed = dog_id in self._templates or dog_id in self._records
        self._templates.pop(dog_id, None)
        self._raw.pop(dog_id, None)
        self._records.pop(dog_id, None)
        return existed

    def list_records(self) -> list[dict]:
        """Return enrollment metadata without exposing template vectors."""
        return [dict(self._records[dog_id])
                for dog_id in sorted(self._records)]

    def __len__(self) -> int:
        return self.size
