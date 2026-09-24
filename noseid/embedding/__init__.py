"""Embedding subpackage: biometric embedding network + metric-learning losses.

Lazy import so the loss helpers (numpy-referencable) load without torch; the
torch-only ``EmbeddingNet`` is imported on demand.
"""

def __getattr__(name):  # PEP 562
    if name in ("ArcFaceLoss", "TripletLoss", "ContrastiveLoss", "CombinedMetricLoss"):
        from .losses import ArcFaceLoss, TripletLoss, ContrastiveLoss, CombinedMetricLoss
        g = {"ArcFaceLoss": ArcFaceLoss, "TripletLoss": TripletLoss,
             "ContrastiveLoss": ContrastiveLoss, "CombinedMetricLoss": CombinedMetricLoss}
        return g[name]
    if name == "Embedder":
        from .model import Embedder
        return Embedder
    if name == "FullPhotoEmbedder":
        from .full_photo import FullPhotoEmbedder
        return FullPhotoEmbedder
    if name == "EmbeddingNet":
        from .model import EmbeddingNet  # raises if torch missing - intended
        return EmbeddingNet
    if name in ("FeatureEmbedder", "FeatureEmbeddingResult", "FEATURE_NAMES"):
        from .features import FeatureEmbedder, FeatureEmbeddingResult, FEATURE_NAMES
        return {"FeatureEmbedder": FeatureEmbedder,
                "FeatureEmbeddingResult": FeatureEmbeddingResult,
                "FEATURE_NAMES": FEATURE_NAMES}[name]
    raise AttributeError(name)

__all__ = ["ArcFaceLoss", "TripletLoss", "ContrastiveLoss", "CombinedMetricLoss",
           "EmbeddingNet", "Embedder", "FeatureEmbedder",
           "FeatureEmbeddingResult", "FEATURE_NAMES", "FullPhotoEmbedder"]
