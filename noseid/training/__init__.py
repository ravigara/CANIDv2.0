"""Training subpackage: biometric metrics + trainer (Stage 11)."""
from .metrics import (compute_biometric_metrics, far_at_threshold,
                      frr_at_threshold, compute_eer, evaluate_pairs)  # noqa: F401
from .trainer import MetricLearningTrainer  # noqa: F401

__all__ = [
    "compute_biometric_metrics", "far_at_threshold", "frr_at_threshold",
    "compute_eer", "evaluate_pairs", "MetricLearningTrainer",
]
