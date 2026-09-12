"""Stage 11 - Biometric evaluation metrics (numpy, no torch dependency).

  FAR(thr) = fraction of impostor pairs scoring >= thr  (false accept)
  FRR(thr) = fraction of genuine pairs scoring <  thr   (false reject)
  EER      = threshold where FAR == FRR (the equal-error-rate operating point)
  ROC-AUC  = area under FPR vs TPR curve
  Accuracy = best-1 acc at the threshold maximizing accuracy

All inputs are similarity scores in [-1, 1] (cosine). Pair lists can be passed
directly, or generated via evaluate_pairs(embeddings, labels).
"""
from __future__ import annotations

import numpy as np


def evaluate_pairs(embeddings: np.ndarray, labels: np.ndarray,
                   max_pairs: int = 20000, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Build genuine/impostor similarity arrays from a set of embeddings.

    embeddings: [N, D] L2-normalized (or will be normalized here)
    labels    : [N] identity label per embedding
    Returns (genuine_sims, impostor_sims) as 1-D float arrays.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    N = len(X)
    genuine, impostor = [], []
    # all genuine pairs (capped)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        ii, jj = np.triu_indices(len(idx), k=1)
        sims = X[idx[ii]] @ X[idx[jj]].T
        genuine.extend(np.diag(sims).tolist())
    # sampled impostor pairs
    for _ in range(max_pairs):
        a, b = rng.integers(0, N, 2)
        if a == b or y[a] == y[b]:
            continue
        impostor.append(float(X[a] @ X[b]))
    return np.asarray(genuine, dtype=np.float32), np.asarray(impostor, dtype=np.float32)


def far_at_threshold(impostor_sims: np.ndarray, thr: float) -> float:
    if len(impostor_sims) == 0:
        return 0.0
    return float(np.mean(impostor_sims >= thr))


def frr_at_threshold(genuine_sims: np.ndarray, thr: float) -> float:
    if len(genuine_sims) == 0:
        return 0.0
    return float(np.mean(genuine_sims < thr))


def compute_eer(genuine_sims: np.ndarray, impostor_sims: np.ndarray
                ) -> tuple[float, float]:
    """Equal Error Rate (%) and the threshold at which FAR == FRR."""
    scores = np.concatenate([genuine_sims, impostor_sims])
    labels = np.concatenate([np.ones_like(genuine_sims),
                             np.zeros_like(impostor_sims)])
    # sweep thresholds over the observed score range
    thrs = np.linspace(float(scores.min()), float(scores.max()), 500)
    best_thr, best_diff = thrs[0], 1.0
    for t in thrs:
        far = far_at_threshold(impostor_sims, t)
        frr = frr_at_threshold(genuine_sims, t)
        if abs(far - frr) < best_diff:
            best_diff = abs(far - frr)
            best_thr = t
    eer = (far_at_threshold(impostor_sims, best_thr) +
           frr_at_threshold(genuine_sims, best_thr)) / 2
    return float(eer * 100), float(best_thr)


def _roc_auc(genuine_sims: np.ndarray, impostor_sims: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn needed)."""
    g = np.asarray(genuine_sims)
    i = np.asarray(impostor_sims)
    if len(g) == 0 or len(i) == 0:
        return 0.5
    # rank-sum style: P(genuine > impostor)
    n_g, n_i = len(g), len(i)
    # vectorized comparison (cap memory)
    if n_g * n_i > 50_000_000:
        rng = np.random.default_rng(0)
        g = g[rng.choice(n_g, 5000, replace=len(g) < 5000)]
        i = i[rng.choice(n_i, 5000, replace=len(i) < 5000)]
        n_g, n_i = len(g), len(i)
    wins = (g[:, None] > i[None, :]).sum()
    ties = (g[:, None] == i[None, :]).sum()
    return float((wins + 0.5 * ties) / (n_g * n_i))


def compute_biometric_metrics(genuine_sims: np.ndarray,
                              impostor_sims: np.ndarray,
                              threshold: float | None = None,
                              eer_threshold: bool = True) -> dict:
    """Full Stage 11 metric bundle at a chosen operating threshold.

    Spec targets: FAR < 2%, FRR < 2%, EER < 1%, Accuracy > 95%.
    """
    if threshold is None:
        eer_pct, threshold = compute_eer(genuine_sims, impostor_sims)
    elif eer_threshold:
        _, threshold = compute_eer(genuine_sims, impostor_sims)
    eer_pct, eer_thr = compute_eer(genuine_sims, impostor_sims)
    far = far_at_threshold(impostor_sims, threshold)
    frr = frr_at_threshold(genuine_sims, threshold)
    auc = _roc_auc(genuine_sims, impostor_sims)
    acc = 0.5 * ((1 - far) + (1 - frr))   # balanced accuracy at this threshold
    # precision / recall at this threshold treating genuine=positive
    tp = np.sum(genuine_sims >= threshold)
    fn = np.sum(genuine_sims < threshold)
    fp = np.sum(impostor_sims >= threshold)
    tn = np.sum(impostor_sims < threshold)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {
        "threshold": float(threshold),
        "FAR": float(far * 100),       # %
        "FRR": float(frr * 100),       # %
        "EER": float(eer_pct),         # %
        "EER_threshold": float(eer_thr),
        "accuracy": float(acc * 100),  # %
        "precision": float(prec * 100),
        "recall": float(rec * 100),
        "f1": float(f1 * 100),
        "roc_auc": float(auc),
        "n_genuine": int(len(genuine_sims)),
        "n_impostor": int(len(impostor_sims)),
    }
