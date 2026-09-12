"""Pipeline subpackage: validation, detection, landmarks, segmentation.

Imports are lazy so each stage can be exercised independently (e.g. while the
torch-backed detection/segmentation modules are still being written). Use
``from noseid.pipeline.validator import NoseValidator`` etc. directly, or
import this package after all modules exist.
"""

def __getattr__(name):  # PEP 562 lazy module access
    if name == "NoseValidator":
        from .validator import NoseValidator
        return NoseValidator
    if name == "NoseDetector":
        from .detection import NoseDetector
        return NoseDetector
    if name == "LandmarkDetector":
        from .landmarks import LandmarkDetector
        return LandmarkDetector
    if name == "NoseSegmenter":
        from .segmentation import NoseSegmenter
        return NoseSegmenter
    raise AttributeError(name)

__all__ = ["NoseValidator", "NoseDetector", "LandmarkDetector", "NoseSegmenter"]
