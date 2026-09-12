"""Stage 7 - Metric-learning losses: ArcFace, Triplet, Contrastive.

Implemented in pure torch so they work on CPU/GPU and are usable even outside
the main training loop (e.g. for custom fine-tuning). Each loss is also
callable in numpy for unit tests via the ``.numpy()`` static helpers.

ArcFace: additive angular margin softmax on L2-normalized embeddings + a
         learned per-class weight matrix. Pushes intra-class compactness and
         inter-class separation.
Triplet : max(0, d(a,p) - d(a,n) + margin) with hard-negative friendly default.
Contrastive: y * d^2 + (1-y) * max(margin - d, 0)^2 for pairs.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore


# ----------------------------------------------------------------------- torch
if _HAS_TORCH:

    class ArcFaceLoss(nn.Module):
        """ArcFace additive angular margin loss.

        Wraps a learnable ``W [dim, num_classes]`` classifier; embeddings must
        already be L2-normalized by the backbone head (we re-normalize here).
        """

        def __init__(self, embed_dim: int, num_classes: int,
                     scale: float = 30.0, margin: float = 0.50):
            super().__init__()
            self.scale = float(scale)
            self.margin = float(margin)
            self.W = nn.Parameter(torch.empty(num_classes, embed_dim))
            nn.init.xavier_uniform_(self.W)
            self.num_classes = int(num_classes)

        def forward(self, embeddings: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            # normalize embeddings and weights
            emb = F.normalize(embeddings, dim=1)
            w = F.normalize(self.W, dim=1)                 # [C, D]
            cos = emb @ w.t()                              # [B, C]
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            one_hot = F.one_hot(labels, self.num_classes).float()
            # add margin only to the target class
            target_logits = torch.cos(theta + self.margin)
            logits = cos * (1 - one_hot) + target_logits * one_hot
            logits *= self.scale
            return F.cross_entropy(logits, labels)

    class TripletLoss(nn.Module):
        def __init__(self, margin: float = 0.3, p: int = 2):
            super().__init__()
            self.margin = float(margin)
            self.p = int(p)

        def forward(self, anchor, positive, negative):
            d_ap = F.pairwise_distance(anchor, positive, p=self.p)
            d_an = F.pairwise_distance(anchor, negative, p=self.p)
            return F.relu(d_ap - d_an + self.margin).mean()

    class ContrastiveLoss(nn.Module):
        def __init__(self, margin: float = 1.0):
            super().__init__()
            self.margin = float(margin)

        def forward(self, e1, e2, label):   # label: 1 same, 0 different
            d = F.pairwise_distance(e1, e2)
            loss = label * d.pow(2) + (1 - label) * F.relu(self.margin - d).pow(2)
            return loss.mean()

    class CombinedMetricLoss(nn.Module):
        """Weighted sum of ArcFace + Triplet + Contrastive (spec Stage 11)."""

        def __init__(self, embed_dim: int, num_classes: int,
                     w_arcface: float = 1.0, w_triplet: float = 0.5,
                     w_contrastive: float = 0.3, arcface_margin: float = 0.5,
                     triplet_margin: float = 0.3, contrastive_margin: float = 1.0):
            super().__init__()
            self.arcface = ArcFaceLoss(embed_dim, num_classes, margin=arcface_margin)
            self.triplet = TripletLoss(margin=triplet_margin)
            self.contrastive = ContrastiveLoss(margin=contrastive_margin)
            self.w_a = float(w_arcface)
            self.w_t = float(w_triplet)
            self.w_c = float(w_contrastive)

        def forward(self, embeddings, labels, triplets=None, pairs=None):
            total = self.w_a * self.arcface(embeddings, labels)
            if triplets is not None:
                a, p, n = triplets
                total = total + self.w_t * self.triplet(a, p, n)
            if pairs is not None:
                e1, e2, y = pairs
                total = total + self.w_c * self.contrastive(e1, e2, y)
            return total


# ----------------------------------------------------------------------- numpy
# Reference numpy implementations so losses are unit-testable without torch.

def _arcface_numpy(embeddings: np.ndarray, labels: np.ndarray, W: np.ndarray,
                   scale: float = 30.0, margin: float = 0.5) -> float:
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    w = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    cos = emb @ w.T
    cos = np.clip(cos, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos)
    one_hot = np.zeros_like(cos)
    one_hot[np.arange(len(labels)), labels] = 1
    logits = cos * (1 - one_hot) + np.cos(theta + margin) * one_hot
    logits *= scale
    # cross-entropy
    m = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - m)
    loss = -np.log(exp[np.arange(len(labels)), labels] / exp.sum(axis=1) + 1e-12)
    return float(loss.mean())


def _triplet_numpy(a, p, n, margin=0.3):
    d_ap = np.linalg.norm(a - p, axis=1)
    d_an = np.linalg.norm(a - n, axis=1)
    return float(np.maximum(0, d_ap - d_an + margin).mean())


def _contrastive_numpy(e1, e2, y, margin=1.0):
    d = np.linalg.norm(e1 - e2, axis=1)
    return float((y * d ** 2 + (1 - y) * np.maximum(0, margin - d) ** 2).mean())


# expose numpy helpers on the torch classes for symmetry / testing
if _HAS_TORCH:
    ArcFaceLoss.numpy = staticmethod(_arcface_numpy)        # type: ignore
    TripletLoss.numpy = staticmethod(_triplet_numpy)        # type: ignore
    ContrastiveLoss.numpy = staticmethod(_contrastive_numpy)  # type: ignore
else:
    # minimal stand-ins so the module imports without torch
    class ArcFaceLoss:  # type: ignore
        numpy = staticmethod(_arcface_numpy)
    class TripletLoss:  # type: ignore
        numpy = staticmethod(_triplet_numpy)
    class ContrastiveLoss:  # type: ignore
        numpy = staticmethod(_contrastive_numpy)
    class CombinedMetricLoss:  # type: ignore
        pass
