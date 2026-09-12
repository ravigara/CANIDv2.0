"""Stage 9 - End-to-end biometric pipeline runner.

Wires the stages together:
  Validation -> Detection -> Landmarks -> Segmentation -> Embedding -> FAISS

For the synthetic / untrained path, ``gt_hints`` lets the demo pass known
ground-truth boxes + landmarks so detection/landmark numpy fallbacks (which
aren't trained) don't dominate the error. A production run with a trained YOLO
+ landmark net + SegFormer + EmbeddingNet leaves gt_hints=None and every stage
runs on real model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .config import get_config, cfg_get
from .types import (ValidationResult, DetectionResult, LandmarkResult,
                    SegmentationResult, EmbeddingResult,
                    EnrollmentResult, IdentificationResult)
from .pipeline.validator import NoseValidator
from .pipeline.detection import NoseDetector
from .pipeline.landmarks import LandmarkDetector
from .pipeline.segmentation import NoseSegmenter
from .embedding import Embedder
from .matching import NoseIndex
from .data.synth import CLASSES


@dataclass
class PipelineComponents:
    validator: NoseValidator
    detector: NoseDetector
    landmarks: LandmarkDetector
    segmenter: NoseSegmenter
    embedder: Embedder


def build_pipeline(cfg: dict | None = None) -> PipelineComponents:
    cfg = cfg or get_config()
    det_cfg = cfg_get(cfg, "detection", {})
    lm_cfg = cfg_get(cfg, "landmarks", {})
    seg_cfg = cfg_get(cfg, "segmentation", {})
    emb_cfg = cfg_get(cfg, "embedding", {})
    return PipelineComponents(
        validator=NoseValidator(cfg),
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
        embedder=Embedder(
            weights=emb_cfg.get("weights"),
            backbone=cfg_get(cfg, "backbone_default", "efficientnet_v2"),
            embed_dim=int(emb_cfg.get("dim", 256)), cfg=cfg),
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


def run_to_embedding(pc: PipelineComponents, image: np.ndarray,
                     gt: GTHint | None = None,
                     dog_id: str | None = None,
                     skip_validation: bool = False
                     ) -> tuple[ValidationResult | None, DetectionResult | None,
                                LandmarkResult | None, SegmentationResult | None,
                                EmbeddingResult]:
    """Run detect -> validation -> landmarks -> segment -> embed.

    Returns all intermediate results so callers (enroll/identify/explainability)
    can reuse them without recomputing.
    """
    # 1. detection (with gt fallback for synthetic path). Detection comes
    # first so quality measurements can be made on the actual nose region
    # instead of the full portrait/background.
    dets = pc.detector.detect(image, gt_hint=gt.bbox if gt else None)
    # The crop-preparation script enrolls the highest-confidence detection;
    # inference must select the same box when YOLO returns overlapping boxes.
    det = max(dets, key=lambda item: item.confidence) if dets else None
    bbox = det.bbox if det is not None else (gt.bbox if gt else None)

    # 2. validation. This remains available as a quality report even when
    # hard_reject is disabled for development inference.
    vr = None
    if not skip_validation:
        vr = pc.validator.validate(image, bbox=bbox)

    # 3. landmarks (gt fallback)
    if gt and gt.left_nare and gt.right_nare and gt.philtrum:
        lm = LandmarkResult(left_nare=gt.left_nare, right_nare=gt.right_nare,
                            philtrum=gt.philtrum)
    else:
        lm = pc.landmarks.detect(image, bbox) if bbox else None
        # Do not use low-confidence learned landmarks for alignment/features;
        # the embedder will fall back to a detector crop when landmarks are
        # unavailable. The result remains available to callers when they call
        # LandmarkDetector directly for diagnostics.
        if lm is not None and not lm.valid:
            lm = None

    # 4. segmentation
    seg = pc.segmenter.segment(image, bbox)

    # 5. embedding
    emb = pc.embedder.embed(image, lm, seg.masks, CLASSES, bbox, dog_id)
    return vr, det, lm, seg, emb


def enroll_dog(pc: PipelineComponents, index: NoseIndex, dog_id: str,
               samples: list[np.ndarray], gt_hints: list[GTHint] | None = None,
               min_valid: int = 5) -> EnrollmentResult:
    """Stage 8 enrollment: validate, embed N images, average, add to FAISS."""
    embs: list[np.ndarray] = []
    for i, img in enumerate(samples):
        gt = gt_hints[i] if gt_hints else None
        vr, det, _, _, er = run_to_embedding(
            pc, img, gt, dog_id, skip_validation=False)
        if det is None and gt is None:
            continue
        if (vr is not None and not vr.valid
                and pc.validator.hard_reject):
            continue
        embs.append(np.asarray(er.embedding, dtype=np.float32))
    if len(embs) < min_valid:
        raise ValueError(f"only {len(embs)}/{len(samples)} valid images for {dog_id}; "
                         f"need >= {min_valid}")
    return index.enroll(dog_id, embs)


def identify(pc: PipelineComponents, index: NoseIndex, image: np.ndarray,
             gt: GTHint | None = None) -> tuple[IdentificationResult, dict]:
    """Stage 9 identification: single image -> FAISS search -> verdict."""
    vr, det, lm, seg, er = run_to_embedding(pc, image, gt, skip_validation=False)
    if det is None and gt is None:
        return IdentificationResult(
            dog_id=None, similarity=0.0, confidence=0.0,
            status="Rejected", candidates=[{"reason": "No nose detected"}]), \
            {"validation": vr.to_dict() if vr else None,
             "validation_warning": "No nose detected"}
    if vr is not None and not vr.valid and pc.validator.hard_reject:
        return IdentificationResult(dog_id=None, similarity=0.0, confidence=0.0,
                                    status="Rejected",
                                    candidates=[{"reason": vr.reason}]), \
            {"validation": vr.to_dict()}
    res = index.identify(np.asarray(er.embedding, dtype=np.float32))
    return res, {
        "validation": vr.to_dict() if vr else None,
        "detection": det.to_dict() if det else None,
        "validation_warning": (
            vr.reason if vr is not None and not vr.valid else None
        ),
        "embedding": er.to_dict(),
    }
