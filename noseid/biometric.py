"""End-to-end dog registration and identification flow.

The active production path is intentionally small:

    nose detection -> anatomical parts -> normalized crop -> embedding store

``gt_hints`` remains only as a compatibility hook for the synthetic demo and
tests. Quality scoring, synthetic generation, and evaluation are separate
tools; they are not inserted into registration or identification.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import get_config, cfg_get
from .types import (ValidationResult, DetectionResult, LandmarkResult,
                    SegmentationResult, EmbeddingResult,
                    EnrollmentResult, IdentificationResult)
from .pipeline.detection import NoseDetector
from .pipeline.landmarks import LandmarkDetector
from .pipeline.segmentation import NoseSegmenter
from .embedding import Embedder
from .embedding.features import FeatureEmbedder, FeatureEmbeddingResult
from .matching import FeatureIndex, NoseIndex
from .data.synth import CLASSES


@dataclass
class PipelineComponents:
    detector: NoseDetector
    landmarks: LandmarkDetector
    segmenter: NoseSegmenter
    embedder: Embedder
    feature_embedder: FeatureEmbedder


def build_pipeline(cfg: dict | None = None) -> PipelineComponents:
    cfg = cfg or get_config()
    det_cfg = cfg_get(cfg, "detection", {})
    lm_cfg = cfg_get(cfg, "landmarks", {})
    seg_cfg = cfg_get(cfg, "segmentation", {})
    emb_cfg = cfg_get(cfg, "embedding", {})
    embedder = Embedder(
        weights=emb_cfg.get("weights"),
        backbone=cfg_get(cfg, "backbone_default", "efficientnet_v2"),
        embed_dim=int(emb_cfg.get("dim", 256)), cfg=cfg)
    return PipelineComponents(
        detector=NoseDetector(
            weights=det_cfg.get("weights"),
            conf=float(det_cfg.get("conf_threshold", 0.5)),
            backend=det_cfg.get("backend", "auto"),
            label=det_cfg.get("class_name", "NOSE01"),
            device=det_cfg.get("device", "auto")),
        landmarks=LandmarkDetector(
            weights=lm_cfg.get("weights"),
            backend=lm_cfg.get("backend", "auto"),
            device=lm_cfg.get("device", "auto"),
            input_size=int(lm_cfg.get("input_size", 256)),
            crop_pad=float(lm_cfg.get("crop_pad", 0.15)),
            min_confidence=float(lm_cfg.get("min_confidence", 0.20))),
        segmenter=NoseSegmenter(
            weights=seg_cfg.get("weights"),
            backend=seg_cfg.get("backend", "auto"), cfg=cfg,
            device=seg_cfg.get("device", "auto")),
        embedder=embedder,
        feature_embedder=FeatureEmbedder(embedder, cfg),
    )


@dataclass
class GTHint:
    """Optional ground-truth hints for the synthetic/bootstrap path."""
    bbox: list[float] | None = None
    left_nare: list[float] | None = None
    right_nare: list[float] | None = None
    philtrum: list[float] | None = None

    @classmethod
    def from_sample(cls, sample: dict) -> "GTHint":
        return cls(bbox=sample.get("bbox"),
                   left_nare=sample.get("left_nare"),
                   right_nare=sample.get("right_nare"),
                   philtrum=sample.get("philtrum"))


class NoseNotDetectedError(RuntimeError):
    """Raised when an input image cannot enter the biometric flow."""


def _run_anatomy(pc: PipelineComponents, image: np.ndarray,
                 gt: GTHint | None = None
                 ) -> tuple[DetectionResult, LandmarkResult | None,
                            SegmentationResult, list[float]]:
    """Run detection, landmark localization, and anatomical segmentation."""
    dets = pc.detector.detect(image, gt_hint=gt.bbox if gt else None)
    det = max(dets, key=lambda item: item.confidence) if dets else None
    bbox = det.bbox if det is not None else (gt.bbox if gt else None)
    if bbox is None:
        raise NoseNotDetectedError("No dog nose detected")

    if gt and gt.left_nare and gt.right_nare and gt.philtrum:
        lm = LandmarkResult(left_nare=gt.left_nare, right_nare=gt.right_nare,
                            philtrum=gt.philtrum)
    else:
        lm = pc.landmarks.detect(image, bbox)
        if lm is not None and not lm.valid:
            lm = None
    seg = pc.segmenter.segment(image, bbox)
    return det, lm, seg, bbox


def run_to_embedding(pc: PipelineComponents, image: np.ndarray,
                     gt: GTHint | None = None,
                     dog_id: str | None = None,
                     skip_validation: bool = False
                     ) -> tuple[ValidationResult | None, DetectionResult | None,
                                LandmarkResult | None, SegmentationResult | None,
                                EmbeddingResult]:
    """Run detect -> anatomical parts -> normalized crop -> embed.

    Returns all intermediate results so callers (enroll/identify/explainability)
    can reuse them without recomputing.
    """
    det, lm, seg, bbox = _run_anatomy(pc, image, gt)
    vr = None
    emb = pc.embedder.embed(image, lm, seg.masks, seg.classes or CLASSES,
                            bbox, dog_id)
    return vr, det, lm, seg, emb


def run_to_feature_embeddings(
        pc: PipelineComponents, image: np.ndarray,
        gt: GTHint | None = None, dog_id: str | None = None
        ) -> tuple[DetectionResult, LandmarkResult | None,
                   SegmentationResult, FeatureEmbeddingResult]:
    """Run detect -> anatomy -> one shared embedding per anatomical feature."""
    det, lm, seg, bbox = _run_anatomy(pc, image, gt)
    result = pc.feature_embedder.embed(
        image, lm, seg.masks, seg.classes or CLASSES, bbox, dog_id)
    return det, lm, seg, result


def enroll_dog(pc: PipelineComponents, index: NoseIndex | FeatureIndex, dog_id: str,
               samples: list[np.ndarray], gt_hints: list[GTHint] | None = None,
               min_valid: int = 5, metadata: dict | None = None) -> EnrollmentResult:
    """Embed valid registration photos and persist one dog template."""
    embs: list[np.ndarray] = []
    feature_embs: dict[str, list[np.ndarray]] = {}
    feature_images = 0
    errors: list[dict] = []
    for i, img in enumerate(samples):
        gt = gt_hints[i] if gt_hints else None
        try:
            if isinstance(index, FeatureIndex):
                _, _, _, er = run_to_feature_embeddings(pc, img, gt, dog_id)
                if len(er.embeddings) < index.min_features:
                    raise ValueError(
                        f"only {len(er.embeddings)} anatomical features detected")
                for name, vector in er.embeddings.items():
                    feature_embs.setdefault(name, []).append(
                        np.asarray(vector, dtype=np.float32))
                feature_images += 1
            else:
                _, _, _, _, er = run_to_embedding(pc, img, gt, dog_id,
                                                  skip_validation=True)
                embs.append(np.asarray(er.embedding, dtype=np.float32))
        except (NoseNotDetectedError, ValueError, RuntimeError) as exc:
            errors.append({"photo": i + 1, "code": "PROCESSING_ERROR",
                           "message": str(exc)})
            continue
    processed = feature_images if isinstance(index, FeatureIndex) else len(embs)
    if processed < min_valid:
        raise ValueError(f"only {processed}/{len(samples)} valid images for {dog_id}; "
                         f"need >= {min_valid}")
    if isinstance(index, FeatureIndex):
        result = index.enroll(dog_id, feature_embs, metadata=metadata,
                              num_images=processed)
    else:
        result = index.enroll(dog_id, embs, metadata=metadata)
    result.embedding_version = index.embedding_version
    result.photos_processed = processed
    result.photos_failed = len(errors)
    result.photo_errors = errors
    return result


def identify(pc: PipelineComponents, index: NoseIndex | FeatureIndex, image: np.ndarray,
             gt: GTHint | None = None) -> tuple[IdentificationResult, dict]:
    """Identify one image against the persisted dog embedding templates."""
    try:
        if isinstance(index, FeatureIndex):
            det, lm, seg, er = run_to_feature_embeddings(pc, image, gt)
        else:
            vr, det, lm, seg, er = run_to_embedding(
                pc, image, gt, skip_validation=True)
    except NoseNotDetectedError:
        return IdentificationResult(
            dog_id=None, similarity=0.0, confidence=0.0,
            status="Rejected", candidates=[{"reason": "No nose detected"}]), \
            {"validation_warning": "No nose detected"}
    if isinstance(index, FeatureIndex):
        res = index.identify({name: np.asarray(vector, dtype=np.float32)
                              for name, vector in er.embeddings.items()})
        embedding_meta = er.to_dict()
    else:
        res = index.identify(np.asarray(er.embedding, dtype=np.float32))
        embedding_meta = er.to_dict()
    return res, {
        "detection": det.to_dict() if det else None,
        "parts": seg.to_dict() if seg else None,
        "embedding": embedding_meta,
    }


def registration_response(result: EnrollmentResult) -> dict:
    """Return an enrollment response compatible with the reference app."""
    return {
        "nose_print_id": result.dog_id,
        "photos_processed": result.photos_processed or result.num_images,
        "photos_failed": result.photos_failed,
        "photo_errors": result.photo_errors or None,
        "embedding_version": result.embedding_version,
    }


def identification_response(result: IdentificationResult,
                            index: NoseIndex | FeatureIndex) -> dict:
    """Return a public, reference-style identification response.

    Only registration metadata supplied to ``NoseIndex.enroll`` is returned;
    raw embeddings are never exposed.
    """
    if result.candidates and result.candidates[0].get("reason") == "No nose detected":
        return {
            "match": False,
            "matched": False,
            "status": "no_nose",
            "code": "NO_NOSE",
            "message": "No dog nose detected. Please provide a clear nose image.",
            "confidence": 0.0,
            "possible_matches": [],
            "candidates": [],
        }
    def _dog_response(dog_id: str, record: dict) -> dict:
        metadata = record.get("metadata", {}) or {}
        dog = {"dog_id": str(dog_id),
               "name": metadata.get("name") or str(dog_id)}
        for key in ("identification_chip_id", "colour", "breed", "age",
                    "blood_type", "photo_url"):
            if metadata.get(key):
                dog[key] = metadata[key]
        if metadata.get("owner"):
            dog["owner"] = dict(metadata["owner"])
        return dog

    def _candidate_response(candidate: dict) -> dict:
        """Add safe dog metadata to a ranked match candidate.

        Candidate rows are part of the UI/API contract, but vectors must
        remain private.  The matcher only returns scores and IDs; resolve the
        display metadata here so both the CLI and web client get the same
        shape.
        """
        row = dict(candidate)
        candidate_id = row.get("dog_id")
        if candidate_id:
            record = index.get_record(str(candidate_id)) or {}
            row["dog"] = _dog_response(str(candidate_id), record)
        score = row.get("similarity")
        if score is not None:
            row["confidence"] = float(score)
            row["confidence_pct"] = f"{float(score) * 100:.1f}%"
        return row

    candidates = [_candidate_response(candidate)
                  for candidate in (result.candidates or [])]
    if result.status != "Verified" or not result.dog_id:
        candidate = candidates[0] if candidates else {}
        return {
            "match": False,
            "matched": False,
            "status": "not_recognized" if result.status != "Rejected"
            else "possible_matches",
            "code": "NO_MATCH",
            "message": "This dog is not in the database yet, or the match is not confident enough.",
            "confidence": result.confidence,
            "confidence_pct": f"{result.confidence * 100:.1f}%",
            "margin": candidate.get("margin"),
            "possible_matches": candidates,
            "candidates": candidates,
        }
    record = index.get_record(result.dog_id) or {}
    dog = _dog_response(result.dog_id, record)
    return {
        "match": True,
        "matched": True,
        "status": "recognized",
        "message": "Match found",
        "confidence": result.confidence,
        "confidence_pct": f"{result.confidence * 100:.1f}%",
        "margin": (result.candidates[0].get("margin")
                   if result.candidates else None),
        "features_used": (result.candidates[0].get("features_used", [])
                           if result.candidates else []),
        "dog": dog,
        "possible_matches": candidates,
        "candidates": candidates,
    }
