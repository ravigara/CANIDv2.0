"""Persistent dog-template storage and vector identification."""
from .feature_index import FeatureIndex  # noqa: F401
from .cascade_index import CascadeFeatureIndex  # noqa: F401
from .full_photo_cascade_index import FullPhotoCascadeIndex  # noqa: F401
from .index import EmbeddingStore, NoseIndex  # noqa: F401

__all__ = ["EmbeddingStore", "NoseIndex", "FeatureIndex", "CascadeFeatureIndex",
           "FullPhotoCascadeIndex"]
