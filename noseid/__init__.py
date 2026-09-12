"""Dog Nose Biometric Identification System (noseid).

A metric-learning biometric pipeline for identifying individual dogs from
rhinarium (nose) patterns. Subpackages:

  noseid.data        synthetic data generation + dataset wrappers + augmentation
  noseid.pipeline    validation, detection, landmarks, segmentation
  noseid.features    hybrid deep/classical/morphological feature extraction
  noseid.embedding   embedding network + ArcFace/Triplet/Contrastive losses
  noseid.matching    FAISS enrollment + identification
  noseid.training    trainer + biometric metrics (FAR/FRR/EER/AUC)
  noseid.api         FastAPI service (Stage 13)
"""
from .config import get_config, CONFIG_PATH  # noqa: F401

__version__ = "0.1.0"
__all__ = ["get_config", "CONFIG_PATH", "__version__"]
