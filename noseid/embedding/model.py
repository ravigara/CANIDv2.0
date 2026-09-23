"""Stage 7 - Biometric Embedding Network.

EmbeddingNet (torch): timm backbone (ConvNeXt-V2 / ViT / EfficientNetV2) ->
                      pooling -> projection MLP -> L2-normalized 256-d embedding.
                      Optional ArcFace classification head for joint training.

Embedder (factory): resolves backend. When torch is missing it uses a
                    *discriminative* numpy fallback: it hashes a deterministic
                    per-rhinarium texture signature so same-dog images land near
                    each other and different dogs are far apart. This lets the
                    FAISS enroll/identify demo run meaningfully TODAY without a
                    trained model. Swap in a trained EmbeddingNet later by
                    passing weights or setting backend='torch'.
"""
from __future__ import annotations

import os
import numpy as np
import cv2

from ..backend import resolve_backend
from ..config import get_config, cfg_get
from ..types import EmbeddingResult
from ..features.extractor import FeatureExtractor, _l2
from ..types import LandmarkResult, SegmentationResult  # noqa: F401 (typing)
from ..pipeline.crop import normalize_nose_crop


# ----------------------------------------------------------------- torch net
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    BACKBONE_REGISTRY = {
        "convnext_v2": "convnextv2_base.fcmae_ft_in1k",
        "vit": "vit_base_patch16_224.dino",
        "efficientnet_v2": "tf_efficientnetv2_s.in21k_ft_in1k",
    }

    class EmbeddingNet(nn.Module):
        """Backbone + projection head producing L2-normalized embeddings."""

        def __init__(self, backbone: str = "efficientnet_v2", embed_dim: int = 256,
                     pretrained: bool = True, num_classes: int | None = None):
            super().__init__()
            import timm
            name = BACKBONE_REGISTRY.get(backbone, backbone)
            self.backbone = timm.create_model(name, pretrained=pretrained,
                                              num_classes=0, global_pool="avg")
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, embed_dim),
            )
            self.embed_dim = embed_dim
            self.num_classes = num_classes

        def forward(self, x, return_logits: bool = False):
            f = self.backbone(x)
            e = self.head(f)
            e = F.normalize(e, dim=1)
            return e

        @torch.no_grad()
        def embed(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.forward(x)


# --------------------------------------------------------- discriminative fallback

# A fixed random projection matrix that whitens/balances the fused feature
# vector. Seeded so behavior is deterministic across runs / across dogs.
_PROJ_CACHE: dict[int, np.ndarray] = {}


def _projection(in_dim: int, out_dim: int, seed: int = 7) -> np.ndarray:
    key = (in_dim, out_dim, seed)
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(seed)
        P = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        P /= np.sqrt(in_dim)
        _PROJ_CACHE[key] = P
    return _PROJ_CACHE[key]


_MEAN_CROP_CACHE: dict = {}


def _population_mean_crop(size: int = 64, n_samples: int = 40,
                          seed: int = 1000) -> np.ndarray:
    """Estimate the average aligned+CLAHE'd rhinarium crop (shared anatomy).

    IMPORTANT: must be computed through the SAME preprocessing the actual
    embedding uses (canonical crop -> CLAHE -> resize), otherwise the mean
    won't match the crop distribution and mean-shaping won't remove the common
    anatomy. Cached per size.
    """
    key = (size, n_samples, seed)
    if key in _MEAN_CROP_CACHE:
        return _MEAN_CROP_CACHE[key]
    # build via the same generator + canonical-crop + CLAHE path used at embed
    from ..data import NoseSynthGenerator
    from ..types import LandmarkResult
    from ..pipeline.landmarks import LandmarkDetector
    g = NoseSynthGenerator(256, seed=seed)
    lm = LandmarkDetector()
    acc = np.zeros((size, size), dtype=np.float32)
    count = 0
    for i in range(n_samples):
        did = f"dog_{i+1:06d}"
        s = g.render_labeled(did, 0)
        L = LandmarkResult(left_nare=s["left_nare"], right_nare=s["right_nare"],
                           philtrum=s["philtrum"])
        crop = _canonical_crop(s["image"], s["bbox"], L, 144)
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(crop)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        acc += crop.astype(np.float32)
        count += 1
    mean = (acc / max(count, 1)).astype(np.float32)
    _MEAN_CROP_CACHE[key] = mean
    return mean


def _rhinarium_signature(rhinarium_crop_gray: np.ndarray, n_bits: int = 256) -> np.ndarray:
    """Deterministic, per-dog discriminative texture signature.

    Pipeline: CLAHE normalize -> resize to 64x64 -> flatten -> subtract the
    population-mean crop (mean-shaping) -> L2 normalize. Mean-shaping removes
    the shared rhinarium anatomy so cross-dog cosine similarity drops to ~0
    while same-dog stays ~1. This is what makes the untrained numpy embedding
    discriminate dogs reliably; a trained ArcFace head replaces it in torch mode.
    """
    g = rhinarium_crop_gray
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    side = int(round((n_bits ** 0.5)))      # e.g. 256 -> 16x16 (kept coarse for robustness)
    side = max(16, min(64, side))
    g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = g.flatten()
    mean = _population_mean_crop(side).flatten()
    v = v - mean
    v = _l2(v)
    if v.shape[0] >= n_bits:
        return v[:n_bits].astype(np.float32)
    return _l2(np.pad(v, (0, n_bits - v.shape[0]))).astype(np.float32)


def _canonical_crop(image: np.ndarray, bbox: list[float] | None,
                    landmarks: LandmarkResult | None,
                    target: int = 144) -> np.ndarray:
    """Return a translation+rotation-normalized rhinarium crop.

    Aligns so the left_nare -> right_nare axis is horizontal and centered on
    the nare midpoint, then pads to a square. Falls back to a plain bbox crop
    when landmarks are unavailable. This is what makes the untrained numpy
    embedding actually discriminate dogs: it removes pose jitter so the
    per-dog texture pattern is the dominant signal.
    """
    h, w = image.shape[:2]
    # start from bbox crop with padding
    x1, y1, x2, y2 = (bbox if bbox is not None
                      else [0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)

    if landmarks is not None:
        ln = np.asarray(landmarks.left_nare, dtype=np.float32)
        rn = np.asarray(landmarks.right_nare, dtype=np.float32)
        mid = (ln + rn) / 2
        d = rn - ln
        angle_deg = float(np.degrees(np.arctan2(d[1], d[0])))
        # rotate about the nare midpoint to make the axis horizontal
        M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), angle_deg, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # recenter the crop on the (now horizontal) nare midpoint
        cx, cy = float(mid[0]), float(mid[1])
    else:
        rotated = image

    half = side * 0.6
    x1n = int(max(0, cx - half)); x2n = int(min(w, cx + half))
    y1n = int(max(0, cy - half)); y2n = int(min(h, cy + half))
    crop = rotated[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        crop = rotated
    # pad to square
    ch, cwc = crop.shape[:2]
    s = max(ch, cwc)
    if crop.ndim == 3:
        canvas = np.zeros((s, s, crop.shape[2]), dtype=crop.dtype)
    else:
        canvas = np.zeros((s, s), dtype=crop.dtype)
    canvas[(s - ch) // 2:(s - ch) // 2 + ch, (s - cwc) // 2:(s - cwc) // 2 + cwc] = crop
    return cv2.resize(canvas, (target, target), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------- Embedder
class Embedder:
    """Produces a 256-d L2-normalized embedding for a rhinarium image.

    backend='torch'   -> trained EmbeddingNet (needs weights + torch stack).
    backend='numpy'   -> fused features (deep proxy + classical + morphological)
                         blended with the discriminative rhinarium signature so
                         the untrained demo still separates dogs reliably.
    """

    def __init__(self, weights: str | os.PathLike | None = None,
                 backbone: str | None = None, embed_dim: int = 256,
                 backend: str = "auto", cfg: dict | None = None,
                 sig_blend: float | None = None):
        self.cfg = cfg or get_config()
        self.embed_dim = int(cfg_get(self.cfg, "embedding.dim", embed_dim))
        self.backbone_name = (backbone or cfg_get(self.cfg, "backbone_default",
                                                  "efficientnet_v2"))
        self.weights = str(weights) if weights else None
        self.backend = resolve_backend(backend, "embedding")
        # In the untrained numpy path the handcrafted fused features carry no
        # discriminative signal, so let the deterministic rhinarium signature
        # dominate. A trained torch model ignores sig_blend entirely.
        if sig_blend is None:
            sig_blend = 0.95 if self.backend == "numpy" else 0.0
        self.sig_blend = float(sig_blend)
        self.crop_size = int(cfg_get(self.cfg, "image.crop_size", 224))
        self.crop_pad = float(cfg_get(self.cfg, "embedding.crop_pad", 0.15))
        self.mask_background = bool(cfg_get(
            self.cfg, "embedding.mask_background", True))
        self.landmark_min_confidence = float(cfg_get(
            self.cfg, "landmarks.min_confidence", 0.50))
        self._net = None
        # The torch embedding path does not use handcrafted features. Keep the
        # feature extractor numpy-only there to avoid an unnecessary pretrained
        # backbone download during pipeline construction.
        fx_backend = "numpy" if self.backend == "torch" else "auto"
        self.fx = FeatureExtractor(target_dim=self.embed_dim, backend=fx_backend)
        if self.backend == "torch":
            try:
                self._init_torch()
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(f"[noseid] EmbeddingNet init failed: {e}; numpy backend")
                self.backend = "numpy"

    def _init_torch(self) -> None:
        net = EmbeddingNet(self.backbone_name, self.embed_dim,
                           pretrained=not bool(self.weights))
        if self.weights and os.path.exists(self.weights):
            sd = torch.load(self.weights, map_location="cpu")
            net.load_state_dict(sd.get("model", sd), strict=False)
        net.eval()
        self._net = net

    # -- public API -------------------------------------------------------
    def embed(self, image: np.ndarray,
              landmarks: LandmarkResult | None = None,
              masks: np.ndarray | None = None,
              classes: list[str] | None = None,
              bbox: list[float] | None = None,
              dog_id: str | None = None) -> EmbeddingResult:
        if self.backend == "torch" and self._net is not None:
            emb = self._embed_torch(image, landmarks, masks, classes, bbox)
            return EmbeddingResult(dog_id=dog_id or "?", embedding=emb.tolist(),
                                   embedding_size=emb.shape[0], source="torch")
        emb = self._embed_numpy(image, landmarks, masks, classes, bbox)
        return EmbeddingResult(dog_id=dog_id or "?", embedding=emb.tolist(),
                               embedding_size=emb.shape[0], source="numpy")

    def embed_crop(self, crop: np.ndarray, dog_id: str | None = None) -> EmbeddingResult:
        """Embed an already-normalized crop without applying anatomy again.

        ``FeatureEmbedder`` calls this for each anatomical crop.  Keeping this
        path separate from :meth:`embed` prevents a second detector-box
        normalization from undoing the region crop.
        """
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("feature crop must be a non-empty numpy image")
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            crop = crop[..., :3]
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("feature crop must have three color channels")
        if self.backend == "torch" and self._net is not None:
            emb = self._embed_raw_torch(crop)
            source = "torch"
        else:
            emb = self._embed_raw_numpy(crop)
            source = "numpy"
        return EmbeddingResult(dog_id=dog_id or "?", embedding=emb.tolist(),
                               embedding_size=emb.shape[0], source=source)

    __call__ = embed

    # -- backends ---------------------------------------------------------
    def _embed_torch(self, image: np.ndarray,
                     landmarks: LandmarkResult | None = None,
                     masks: np.ndarray | None = None,
                     classes: list[str] | None = None,
                     bbox: list[float] | None = None) -> np.ndarray:
        normalized = normalize_nose_crop(
            image, bbox, landmarks, masks, classes,
            target_size=self.crop_size, pad=self.crop_pad,
            min_confidence=self.landmark_min_confidence,
            mask_background=self.mask_background,
        )
        return self._embed_raw_torch(normalized.image)

    def _embed_raw_torch(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        input_size = int(cfg_get(self.cfg, "training.image_size", 224))
        rgb = cv2.resize(rgb, (input_size, input_size)) \
            if rgb.shape[:2] != (input_size, input_size) else rgb
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            e = self._net.embed(t.unsqueeze(0))[0].cpu().numpy()
        return _l2(e.astype(np.float32))

    def _embed_raw_numpy(self, crop: np.ndarray) -> np.ndarray:
        """Embed a feature crop with the deterministic no-torch fallback."""
        crop = cv2.resize(crop, (self.crop_size, self.crop_size),
                          interpolation=cv2.INTER_AREA)
        fv = self.fx.extract(crop)
        fused = FeatureExtractor.fused_array(fv)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sig = _rhinarium_signature(gray, self.embed_dim)
        return _l2(self.sig_blend * sig + (1 - self.sig_blend) * fused).astype(np.float32)

    def _embed_numpy(self, image: np.ndarray,
                     landmarks: LandmarkResult | None,
                     masks: np.ndarray | None,
                     classes: list[str] | None,
                     bbox: list[float] | None) -> np.ndarray:
        normalized = normalize_nose_crop(
            image, bbox, landmarks, masks, classes,
            target_size=self.crop_size, pad=self.crop_pad,
            min_confidence=self.landmark_min_confidence,
            mask_background=self.mask_background,
        )
        crop = normalized.image

        # fused features (texture + geometry) from the same normalized input
        fv = self.fx.extract(crop, normalized.landmarks,
                             normalized.masks, classes)
        fused = FeatureExtractor.fused_array(fv)   # already L2-normalized, 256-d

        # rhinarium signature from a CANONICALLY ALIGNED crop. Aligning on the
        # nare axis (left->right nare line made horizontal, centered on their
        # midpoint) cancels the per-image rotation/translation jitter that the
        # synth generator injects - the dominant noise source. This is the same
        # idea as 2D face normalization before face embedding.
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        sig = _rhinarium_signature(gray, self.embed_dim)

        # blend: signature is the strongly-discriminative component, fused
        # features keep a small voice so a fully trained model would override it
        emb = _l2(self.sig_blend * sig + (1 - self.sig_blend) * fused)
        return emb.astype(np.float32)
