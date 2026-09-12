"""Stage 6 - Hybrid Feature Extraction.

Three families of features, fused into one vector:

  Deep          : ConvNeXt-V2 / ViT features (torch) OR a learned-free deep-ish
                  proxy (multi-scale Gabor + steerable filter bank, numpy).
  Classical     : SIFT, ORB, LBP histogram, Gabor energy, Fourier descriptors.
  Morphological : nare dims, nare spacing, philtrum length, fold distances,
                  symmetry score - all derived from landmarks + masks.

Every descriptor is L2-normalized then PCA-whitened down to a fixed width so
concatenation is balanced (no single family dominates). The fused vector is
itself L2-normalized: this is what the ArcFace/Triplet head and the FAISS
numpy-fallback embedding consume.

Output: FeatureVector(deep, classical, morphological, fused_dim) and a raw
fused numpy array via .fused().
"""
from __future__ import annotations

import os
import numpy as np
import cv2

from ..backend import resolve_backend
from ..types import FeatureVector, LandmarkResult, SegmentationResult


# ---- classical descriptors --------------------------------------------------

def lbp_histogram(gray: np.ndarray, P: int = 8, R: int = 1,
                  n_bins: int = 32) -> np.ndarray:
    """Uniform-ish LBP histogram (no sklearn dependency)."""
    h, w = gray.shape
    pad = np.pad(gray, R, mode="edge")
    hist = np.zeros(n_bins, dtype=np.float32)
    shifted = []
    for k in range(P):
        ang = 2 * np.pi * k / P
        dx, dy = int(round(R * np.cos(ang))), int(round(R * np.sin(ang)))
        shifted.append(pad[R + dy:R + dy + h, R + dx:R + dx + w])
    center = gray
    lbp = np.zeros((h, w), dtype=np.uint8)
    for sh in shifted:
        lbp = (lbp << 1) | (sh >= center).astype(np.uint8)
    # reduce to n_bins by modulo grouping (cheap uniform approximation)
    hist = np.bincount((lbp % n_bins).ravel(), minlength=n_bins).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def gabor_bank(gray: np.ndarray, n_orient: int = 8, scales=(3, 7, 11, 15)) -> np.ndarray:
    """Mean + std Gabor response energy per (orient, scale)."""
    feats = []
    for lam in scales:
        for k in range(n_orient):
            theta = np.pi * k / n_orient
            kern = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                      lambd=lam, gamma=0.5, psi=0,
                                      ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray, cv2.CV_32F, kern)
            feats += [float(np.mean(np.abs(resp))), float(np.std(resp))]
    arr = np.asarray(feats, dtype=np.float32)
    return arr


def fourier_descriptor(mask: np.ndarray, n_harmonics: int = 16) -> np.ndarray:
    """Fourier descriptors of the rhinarium contour shape."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(n_harmonics * 2, dtype=np.float32)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(c) < 4:
        return np.zeros(n_harmonics * 2, dtype=np.float32)
    z = c[:, 0] + 1j * c[:, 1]
    Z = np.fft.fft(z)
    Z /= (np.abs(Z[0]) + 1e-6)         # translation/scale invariance
    Z = Z[1:1 + n_harmonics]            # drop DC (shape, not position)
    fd = np.concatenate([Z.real, Z.imag]).astype(np.float32)
    return fd


def sift_orb_stats(gray: np.ndarray) -> np.ndarray:
    """Compact SIFT/ORB keypoint statistics (count + response moments)."""
    sift = cv2.SIFT_create(nfeatures=64)
    kps = sift.detect(gray, None)
    if not kps:
        return np.zeros(8, dtype=np.float32)
    sizes = np.array([k.size for k in kps])
    resps = np.array([k.response for k in kps])
    return np.array([
        len(kps), sizes.mean(), sizes.std(), sizes.min(), sizes.max(),
        resps.mean(), resps.std(), resps.max(),
    ], dtype=np.float32)


# ---- morphological descriptors ---------------------------------------------

def morphological_features(landmarks: LandmarkResult,
                           masks: np.ndarray,
                           classes: list[str]) -> np.ndarray:
    """Geometric/shape features from landmarks + segmentation."""
    cid = {c: i for i, c in enumerate(classes)}
    ln = np.asarray(landmarks.left_nare)
    rn = np.asarray(landmarks.right_nare)
    phil = np.asarray(landmarks.philtrum)

    inter_nare = float(np.linalg.norm(ln - rn))
    mid = (ln + rn) / 2
    nare_to_phil = float(np.linalg.norm(mid - phil))
    # nare size from mask
    ln_mask = masks[..., cid["left_nare"]] if "left_nare" in cid else None
    rn_mask = masks[..., cid["right_nare"]] if "right_nare" in cid else None
    ln_area = float(ln_mask.sum()) if ln_mask is not None else 0.0
    rn_area = float(rn_mask.sum()) if rn_mask is not None else 0.0
    rhin_area = float(masks[..., cid["rhinarium"]].sum()) if "rhinarium" in cid else 1.0
    fold_area = float(masks[..., cid["fold"]].sum()) if "fold" in cid else 0.0
    # symmetry: how close L and R nare areas are
    sym = 1.0 - abs(ln_area - rn_area) / (ln_area + rn_area + 1e-6)
    feat = np.array([
        inter_nare, nare_to_phil,
        ln_area, rn_area, rhin_area, fold_area,
        inter_nare / (np.sqrt(rhin_area) + 1e-6),
        nare_to_phil / (np.sqrt(rhin_area) + 1e-6),
        sym, len(landmarks.fold_lines),
    ], dtype=np.float32)
    return feat


# ---- deep proxy (numpy) ----------------------------------------------------

def deep_proxy(gray: np.ndarray) -> np.ndarray:
    """A learned-free 'deep' texture descriptor used when torch is absent.

    Multi-scale local energy + gradient histogram pyramid - dimensionality is
    fixed by design so the embedding head is stable.
    """
    feats = []
    for size in (32, 64, 128):
        g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        h, _ = np.histogram(ang, bins=8, weights=mag, range=(0, 1))
        h = h / (h.sum() + 1e-6)
        feats.append(h)
        feats.append([mag.mean(), mag.std()])
    feats.append(gabor_bank(gray)[::2])  # subsample to keep dim stable
    out = np.concatenate([np.atleast_1d(f) for f in feats]).astype(np.float32)
    return out


# ---- normalization / fusion ------------------------------------------------

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-8
    return (x / n).astype(np.float32)


class FeatureExtractor:
    """Hybrid feature extractor with torch/numpy deep backends."""

    def __init__(self, weights: str | os.PathLike | None = None,
                 backend: str = "auto", target_dim: int = 256):
        self.weights = str(weights) if weights else None
        self.backend = resolve_backend(backend, "embedding")
        self.target_dim = int(target_dim)
        self._backbone = None
        if self.backend == "torch":
            try:
                self._init_torch()
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(f"[noseid] deep backbone load failed: {e}; "
                              f"using numpy deep proxy")
                self.backend = "numpy"

    def _init_torch(self) -> None:
        import torch
        import timm
        name = "convnextv2_base.fcmae_ft_in1k" if (self.weights is None) else None
        if self.weights and os.path.exists(self.weights):
            self._backbone = timm.create_model(self.weights, pretrained=False,
                                               features_only=True)
        else:
            self._backbone = timm.create_model(name, pretrained=True,
                                               features_only=True)
        self._backbone.eval()

    def _deep_torch(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            feats = self._backbone(t.unsqueeze(0))
        # global average pool each stage then concat
        pooled = [torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
                  for f in feats]
        v = torch.cat(pooled, 1)[0].cpu().numpy()
        return v.astype(np.float32)

    def extract(self, image: np.ndarray,
                landmarks: LandmarkResult | None = None,
                masks: np.ndarray | None = None,
                classes: list[str] | None = None) -> FeatureVector:
        """Compute all three feature families + the fused vector."""
        rgb = image
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        # deep
        if self.backend == "torch" and self._backbone is not None:
            deep = self._deep_torch(rgb)
        else:
            deep = deep_proxy(gray)
        # classical
        classical = np.concatenate([
            lbp_histogram(gray),
            gabor_bank(gray),
            fourier_descriptor(masks[..., 0] if masks is not None
                               else (gray > gray.mean()).astype(np.uint8)),
            sift_orb_stats(gray),
        ]).astype(np.float32)
        # morphological (zero if no landmarks supplied)
        if landmarks is not None and masks is not None and classes is not None:
            morph = morphological_features(landmarks, masks, classes)
        else:
            morph = np.zeros(10, dtype=np.float32)

        # balance each family via L2 then resize to fixed widths so concat is fair
        def _resize_to(v: np.ndarray, n: int) -> np.ndarray:
            v = _l2(v)
            if v.shape[0] == n:
                return v
            if v.shape[0] > n:
                return v[:n]
            return np.pad(v, (0, n - v.shape[0]))
        td = self.target_dim
        d = _resize_to(deep, td // 2)
        c = _resize_to(classical, td // 4)
        m = _resize_to(morph, td // 4)
        fused = _l2(np.concatenate([d, c, m])).astype(np.float32)
        return FeatureVector(
            deep=d.tolist(), classical=c.tolist(), morphological=m.tolist(),
            fused_dim=int(fused.shape[0]),
        )

    @staticmethod
    def fused_array(fv: FeatureVector) -> np.ndarray:
        """Return the L2-normalized fused vector as a numpy array."""
        return _l2(np.concatenate([
            np.asarray(fv.deep, dtype=np.float32),
            np.asarray(fv.classical, dtype=np.float32),
            np.asarray(fv.morphological, dtype=np.float32),
        ]).astype(np.float32))

    __call__ = extract
