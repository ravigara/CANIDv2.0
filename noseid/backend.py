"""Backend availability detection.

The whole system is dual-mode: when the heavy deep-learning stack (torch,
ultralytics, transformers) is installed, real models run. When it is not,
numpy/OpenCV fallbacks take over so the pipeline still executes end-to-end.

Each consumer calls ``resolve_backend('auto'|'torch'|'numpy', 'detection')``
and gets back either ``'torch'`` or ``'numpy'`` with availability known.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache

_AVAILABLE: dict[str, bool] | None = None


def _probe() -> dict[str, bool]:
    def have(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None
    return {
        "torch": have("torch") and have("torchvision"),
        "ultralytics": have("ultralytics"),
        "transformers": have("transformers"),
        "faiss": have("faiss"),
        "albumentations": have("albumentations"),
        "fastapi": have("fastapi"),
        "onnx": have("onnx") and have("onnxruntime"),
        "tensorflow": have("tensorflow"),
    }


def availability() -> dict[str, bool]:
    """Map of optional dependency -> installed (bool). Cached."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _probe()
    return _AVAILABLE


@lru_cache(maxsize=16)
def resolve_backend(preference: str, component: str) -> str:
    """Resolve a component's backend.

    preference: 'auto' picks torch if the deps for ``component`` are present,
                otherwise numpy. 'torch'/'numpy' are honored verbatim, but if
                torch is requested but unavailable we fall back to numpy with a
                warning (rather than crashing).
    component : 'detection' | 'segmentation' | 'landmarks' | 'embedding' | 'training'
    """
    avail = availability()
    needs = {
        "detection": ["torch", "ultralytics"],
        "segmentation": ["torch", "transformers"],
        "landmarks": ["torch"],
        "embedding": ["torch"],
        "training": ["torch"],
    }.get(component, ["torch"])

    torch_ok = all(avail.get(n, False) for n in needs)

    pref = (preference or "auto").lower()
    if pref == "torch" and not torch_ok:
        import warnings
        warnings.warn(
            f"[noseid] torch backend requested for '{component}' but "
            f"dependencies missing ({[n for n in needs if not avail.get(n)]}); "
            f"falling back to numpy.",
            RuntimeWarning, stacklevel=2,
        )
        return "numpy"
    if pref == "auto":
        return "torch" if torch_ok else "numpy"
    return pref


def info() -> str:
    """Human-readable status line of what's installed."""
    a = availability()
    on = [k for k, v in a.items() if v]
    off = [k for k, v in a.items() if not v]
    return (f"installed: {', '.join(on) or 'none'} | "
            f"missing: {', '.join(off) or 'none'}")
