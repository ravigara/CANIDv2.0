"""Full-photo candidate retrieval followed by nose-feature reranking."""
from __future__ import annotations

import json
import os

import numpy as np

from ..config import cfg_get, get_config
from ..embedding.features import FEATURE_NAMES
from ..types import EnrollmentResult, IdentificationResult
from .cascade_index import CascadeFeatureIndex, _normalize, _rows


class FullPhotoCascadeIndex(CascadeFeatureIndex):
    """Union full-photo and whole-nose candidates, then score anatomy.

    Full-frame appearance is intentionally not a hard gate.  The union keeps
    a correct dog if either the appearance descriptor or the nose descriptor
    is unreliable for a query photo.
    """

    FORMAT = "noseid-full-photo-cascade-gallery-v1"

    def __init__(self, dim: int = 256, cfg: dict | None = None):
        super().__init__(dim=dim, cfg=cfg)
        self.photo_weight = float(cfg_get(
            self.cfg, "matching.cascade_photo_weight", 0.15))
        self.whole_weight = float(cfg_get(
            self.cfg, "matching.cascade_whole_weight", 0.25))
        self.feature_weight = float(cfg_get(
            self.cfg, "matching.cascade_feature_weight", 0.60))
        if min(self.photo_weight, self.whole_weight, self.feature_weight) < 0:
            raise ValueError("full-photo cascade weights must be non-negative")
        if self.photo_weight + self.whole_weight + self.feature_weight <= 0:
            raise ValueError("full-photo cascade weights must have a positive sum")
        self._photo_templates: dict[str, np.ndarray] = {}
        self._photo_raw: dict[str, list[np.ndarray]] = {}

    def enroll(self, dog_id: str,
               photo_embeddings: list[np.ndarray] | np.ndarray,
               whole_embeddings: list[np.ndarray] | np.ndarray,
               feature_embeddings: dict[str, list[np.ndarray] | np.ndarray],
               metadata: dict | None = None,
               num_images: int | None = None) -> EnrollmentResult:
        photo_rows, photo_template = _rows(
            photo_embeddings, self.dim, "full-photo")
        result = super().enroll(
            dog_id, whole_embeddings, feature_embeddings,
            metadata=metadata, num_images=num_images)
        dog_id = str(dog_id)
        self._photo_templates[dog_id] = photo_template
        self._photo_raw[dog_id] = photo_rows
        self._records[dog_id]["full_photo_count"] = len(photo_rows)
        self._records[dog_id]["full_photo_mean_similarity"] = round(
            float(np.mean([row @ photo_template for row in photo_rows])), 6)
        self._records[dog_id]["full_photo_cascade"] = {
            "photo_weight": self.photo_weight,
            "whole_weight": self.whole_weight,
            "feature_weight": self.feature_weight,
            "retrieval_k": self.retrieval_k,
        }
        return result

    def enroll_full_photo_template(self, dog_id: str,
                                   photo_embeddings: list[np.ndarray] | np.ndarray,
                                   metadata: dict | None = None,
                                   num_images: int | None = None
                                   ) -> EnrollmentResult:
        """Add a photo template while reusing an existing nose template.

        This is used when migrating an already-built whole-nose cascade. It
        avoids rerunning detector/segmentation/landmarks for every image; the
        existing nose templates remain byte-for-byte available for rollback
        comparison.
        """
        dog_id = str(dog_id)
        if dog_id not in self._templates or dog_id not in self._whole_templates:
            raise ValueError(f"nose template is missing for {dog_id}")
        photo_rows, photo_template = _rows(
            photo_embeddings, self.dim, "full-photo")
        self._photo_templates[dog_id] = photo_template
        self._photo_raw[dog_id] = photo_rows
        record = self._records.setdefault(dog_id, {"dog_id": dog_id})
        if metadata:
            record["metadata"] = dict(metadata)
        record["full_photo_count"] = len(photo_rows)
        record["full_photo_mean_similarity"] = round(
            float(np.mean([row @ photo_template for row in photo_rows])), 6)
        record["full_photo_cascade"] = {
            "photo_weight": self.photo_weight,
            "whole_weight": self.whole_weight,
            "feature_weight": self.feature_weight,
            "retrieval_k": self.retrieval_k,
        }
        return EnrollmentResult(
            dog_id=dog_id, embedding_size=self.dim,
            num_images=int(num_images or len(photo_rows)),
            mean_confidence=record["full_photo_mean_similarity"],
            status="Enrolled", embedding_version=self.embedding_version,
            photos_processed=int(num_images or len(photo_rows)))

    def _combined_score(self, photo_score: float, whole_score: float,
                        feature_score: float) -> float:
        total = self.photo_weight + self.whole_weight + self.feature_weight
        return ((self.photo_weight * photo_score) +
                (self.whole_weight * whole_score) +
                (self.feature_weight * feature_score)) / total

    def identify(self, photo_embedding: np.ndarray,
                 whole_embedding: np.ndarray,
                 feature_embeddings: dict[str, np.ndarray],
                 top_k: int | None = None) -> IdentificationResult:
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
        if not self._photo_templates or not self._whole_templates:
            return IdentificationResult(
                dog_id=None, similarity=0.0, confidence=0.0,
                status="Unknown Dog", candidates=[])

        query_photo = _normalize(photo_embedding)
        query_whole = _normalize(whole_embedding)
        photo_coarse = sorted(
            ((dog_id, float(query_photo @ template))
             for dog_id, template in self._photo_templates.items()),
            key=lambda item: item[1], reverse=True)
        whole_coarse = sorted(
            ((dog_id, float(query_whole @ template))
             for dog_id, template in self._whole_templates.items()),
            key=lambda item: item[1], reverse=True)
        limit = min(len(photo_coarse), max(self.retrieval_k, top_k or self.top_k))
        whole_limit = min(len(whole_coarse), max(self.retrieval_k, top_k or self.top_k))
        photo_ranks = {dog_id: (rank, score)
                       for rank, (dog_id, score) in enumerate(photo_coarse[:limit], 1)}
        whole_ranks = {dog_id: (rank, score)
                       for rank, (dog_id, score) in enumerate(whole_coarse[:whole_limit], 1)}
        candidate_ids = set(photo_ranks) | set(whole_ranks)

        candidates = []
        for dog_id in candidate_ids:
            photo_rank, photo_score = photo_ranks.get(
                dog_id, (None, float(photo_coarse[-1][1])))
            whole_rank, whole_score = whole_ranks.get(
                dog_id, (None, float(whole_coarse[-1][1])))
            scored = self._score(query_features, self._templates.get(dog_id, {}))
            if scored is None:
                continue
            feature_score, feature_scores = scored
            combined = self._combined_score(photo_score, whole_score, feature_score)
            candidates.append({
                "dog_id": dog_id,
                "similarity": round(combined, 4),
                "confidence": round(combined, 4),
                "full_photo_similarity": round(photo_score, 4),
                "whole_nose_similarity": round(whole_score, 4),
                "feature_similarity": round(feature_score, 4),
                "feature_scores": feature_scores,
                "features_used": sorted(feature_scores),
                "full_photo_retrieval_rank": photo_rank,
                "whole_nose_retrieval_rank": whole_rank,
                "retrieval_sources": [
                    source for source, rank in (
                        ("full_photo", photo_rank), ("whole_nose", whole_rank))
                    if rank is not None
                ],
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
            status, dog_id = "Verified", best["dog_id"]
        elif similarity < self.reject_thr:
            status, dog_id = "Unknown Dog", None
        else:
            status, dog_id = "Rejected", None
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
            "photo_weight": self.photo_weight,
            "whole_weight": self.whole_weight,
            "feature_weight": self.feature_weight,
            "photo_templates": {
                dog_id: vector.tolist()
                for dog_id, vector in self._photo_templates.items()
            },
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
        with open(os.path.join(dir_path, "full_photo_cascade_meta.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "FullPhotoCascadeIndex":
        path = os.path.join(dir_path, "full_photo_cascade_meta.json")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != cls.FORMAT:
            raise ValueError(
                f"unsupported full-photo gallery format: {payload.get('format')}")
        cfg = get_config({"matching": {
            "verify_threshold": payload.get("verify_thr", 0.50),
            "reject_threshold": payload.get("reject_thr", 0.20),
            "margin_threshold": payload.get("margin_thr", 0.00),
            "top_k": payload.get("top_k", 5),
            "min_features": payload.get("min_features", 2),
            "embedding_version": payload.get(
                "embedding_version", "noseid-v4-full-photo-cascade"),
            "feature_weights": payload.get("feature_weights", {}),
            "cascade_retrieval_k": payload.get("retrieval_k", 50),
            "cascade_photo_weight": payload.get("photo_weight", 0.15),
            "cascade_whole_weight": payload.get("whole_weight", 0.25),
            "cascade_feature_weight": payload.get("feature_weight", 0.60),
        }})
        index = cls(dim=int(payload["dim"]), cfg=cfg)
        index._photo_templates = {
            dog_id: _normalize(np.asarray(vector, dtype=np.float32))
            for dog_id, vector in payload.get("photo_templates", {}).items()
        }
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
        self._photo_templates.pop(str(dog_id), None)
        self._photo_raw.pop(str(dog_id), None)
        return existed
