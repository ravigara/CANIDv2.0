"""Stage 8/9 - FAISS-backed dog identity index.

Enrollment   : average a dog's embeddings (template embedding), add to FAISS.
Identification: 1-NN search, convert distance -> cosine similarity, apply the
                verify/reject thresholds from config.

Supports:
  - flat   : IndexFlatL2 (exact, small scale)
  - ivf    : IndexIVFFlat (approximate, 100k+ dogs)
  - hnsw   : IndexHNSWFlat (graph, fast recall)
  - inner_product / cosine: normalize vectors + use IP metric

The id<->row mapping is kept in a side dict so dog ids survive re-indexing and
so the index can be saved/loaded with metadata.
"""
from __future__ import annotations

import json
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False
    faiss = None  # type: ignore

from ..config import get_config, cfg_get
from ..types import EnrollmentResult, IdentificationResult


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return (v / n).astype(np.float32)


def _dist_to_cos(d: float, metric: str) -> float:
    """Convert a FAISS distance to cosine similarity in [-1, 1]."""
    if metric == "inner_product":
        return float(d)
    # FAISS IndexFlatL2/IndexIVFFlat return squared L2 distance. For unit
    # vectors: ||a-b||^2 = 2 - 2cos, therefore cos = 1 - d/2.
    return float(1.0 - d / 2.0)


class NoseIndex:
    """FAISS-backed gallery of enrolled dog embeddings."""

    def __init__(self, dim: int = 256, index_type: str = "flat",
                 metric: str = "l2", cfg: dict | None = None):
        if not _HAS_FAISS:
            raise ImportError("faiss is required for NoseIndex; "
                              "pip install faiss-cpu")
        self.cfg = cfg or get_config()
        self.dim = int(dim)
        self.index_type = index_type
        self.metric = metric
        self.verify_thr = float(cfg_get(self.cfg, "matching.verify_threshold", 0.90))
        self.reject_thr = float(cfg_get(self.cfg, "matching.reject_threshold", 0.70))
        self.margin_thr = float(cfg_get(self.cfg, "matching.margin_threshold", 0.08))
        self.embedding_version = str(cfg_get(
            self.cfg, "matching.embedding_version", "noseid-v1-anatomy"))
        self.top_k = int(cfg_get(self.cfg, "matching.top_k", 5))
        self.nlist = int(cfg_get(self.cfg, "faiss.nlist", 256))
        self.nprobe = int(cfg_get(self.cfg, "faiss.nprobe", 16))

        self._index = self._build_index()
        # bookkeeping
        self._id2row: dict[str, int] = {}
        self._row2id: dict[int, str] = {}
        self._templates: dict[str, np.ndarray] = {}  # averaged embeddings
        self._raw: dict[str, list[np.ndarray]] = {}  # all per-dog embeddings
        self._records: dict[str, dict] = {}

    # -- index construction ----------------------------------------------
    def _build_index(self):
        d = self.dim
        if self.metric == "inner_product":
            m = faiss.METRIC_INNER_PRODUCT
        else:
            m = faiss.METRIC_L2
        if self.index_type == "flat":
            return faiss.IndexFlatIP(d) if self.metric == "inner_product" \
                else faiss.IndexFlatL2(d)
        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(d, 32, m)
            idx.hnsw.efConstruction = 64
            return idx
        if self.index_type == "ivf":
            quantizer = (faiss.IndexFlatIP(d) if self.metric == "inner_product"
                         else faiss.IndexFlatL2(d))
            idx = faiss.IndexIVFFlat(quantizer, d, self.nlist, m)
            return idx
        # default
        return faiss.IndexFlatL2(d)

    @property
    def size(self) -> int:
        return self._index.ntotal

    # -- enrollment (Stage 8) --------------------------------------------
    def enroll(self, dog_id: str, embeddings: list[np.ndarray],
               metadata: dict | None = None) -> EnrollmentResult:
        """Average the dog's embeddings into a template and add it to the index.

        Re-enrolling an existing dog replaces its template (rebuilds the index
        if necessary so row positions stay consistent).
        """
        if not embeddings:
            raise ValueError("need at least one embedding to enroll")
        embs = _normalize(np.stack([np.asarray(e, dtype=np.float32).reshape(-1)
                                    for e in embeddings]))
        template = _normalize(embs.mean(axis=0))
        mean_conf = float(np.mean([float(embs[i] @ template)
                                   for i in range(len(embs))]))

        # remove old entry if re-enrolling
        if dog_id in self._id2row:
            self._rebuild_without(dog_id)

        self._raw[dog_id] = [e for e in embs]
        self._templates[dog_id] = template
        self._records[dog_id] = {
            "dog_id": dog_id,
            "embedding_version": self.embedding_version,
            "num_images": len(embeddings),
            "mean_similarity": round(mean_conf, 6),
            "metadata": dict(metadata or {}),
        }
        row = len(self._id2row)
        self._id2row[dog_id] = row
        self._row2id[row] = dog_id
        # rebuild from scratch (simple + correct; fine for 100k scale with flat)
        self._rebuild_index()
        return EnrollmentResult(
            dog_id=dog_id, embedding_size=self.dim,
            num_images=len(embeddings), mean_confidence=round(mean_conf, 4),
            status="Enrolled")

    def _rebuild_without(self, dog_id: str) -> None:
        self._raw.pop(dog_id, None)
        self._templates.pop(dog_id, None)
        self._records.pop(dog_id, None)

    def _rebuild_index(self) -> None:
        self._index.reset()
        if not self._templates:
            self._id2row.clear(); self._row2id.clear()
            return
        if self.index_type == "ivf" and not self._index.is_trained:
            # IVF needs training; with few points fall back to flat behavior
            vecs = np.stack(list(self._templates.values())).astype(np.float32)
            n = min(self.nlist, max(1, vecs.shape[0]))
            self._index.train(vecs[:n])
            self._index.nprobe = self.nprobe
        ids = sorted(self._templates.keys())
        self._id2row = {d: i for i, d in enumerate(ids)}
        self._row2id = {i: d for i, d in enumerate(ids)}
        mat = _normalize(np.stack([self._templates[d] for d in ids]))
        self._index.add(mat)

    # -- identification (Stage 9) ----------------------------------------
    def identify(self, embedding: np.ndarray,
                 top_k: int | None = None) -> IdentificationResult:
        """1-NN search and threshold-based decision."""
        if self.size == 0:
            return IdentificationResult(dog_id=None, similarity=0.0,
                                        confidence=0.0, status="Unknown Dog")
        q = _normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        k = min(top_k or self.top_k, self.size)
        dists, rows = self._index.search(q, k)
        sims = [_dist_to_cos(float(d), self.metric) for d in dists[0]]
        cands = []
        for row, sim in zip(rows[0], sims):
            if row < 0:
                continue
            cands.append({"dog_id": self._row2id.get(int(row), "?"),
                          "similarity": round(sim, 4)})
        if not cands:
            return IdentificationResult(dog_id=None, similarity=0.0,
                                        confidence=0.0, status="Unknown Dog")
        best = cands[0]
        sim = best["similarity"]
        # Keep confidence equal to cosine similarity in [0, 1], matching the
        # vector-search response used by the reference registration app.
        conf = round(sim, 4)
        top2_sim = cands[1]["similarity"] if len(cands) > 1 else None
        margin = (sim - top2_sim) if top2_sim is not None else 1.0
        if sim >= self.verify_thr and margin >= self.margin_thr:
            status = "Verified"
        elif sim < self.reject_thr:
            status = "Unknown Dog"
        else:
            status = "Rejected"   # ambiguous band
        return IdentificationResult(
            dog_id=best["dog_id"] if status == "Verified" else None,
            similarity=round(sim, 4), confidence=conf,
            status=status, candidates=cands)

    # -- persistence -----------------------------------------------------
    def save(self, dir_path: str) -> None:
        import os
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(dir_path, "nose.index"))
        meta = {
            "dim": self.dim, "index_type": self.index_type,
            "metric": self.metric, "verify_thr": self.verify_thr,
            "reject_thr": self.reject_thr, "margin_thr": self.margin_thr,
            "top_k": self.top_k, "embedding_version": self.embedding_version,
            "templates": {d: t.tolist() for d, t in self._templates.items()},
            "records": self._records,
        }
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(cls, dir_path: str) -> "NoseIndex":
        import os
        with open(os.path.join(dir_path, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        idx = cls(dim=meta["dim"], index_type=meta["index_type"],
                  metric=meta["metric"])
        idx.verify_thr = meta["verify_thr"]
        idx.reject_thr = meta["reject_thr"]
        idx.margin_thr = meta.get("margin_thr", 0.0)
        idx.top_k = meta["top_k"]
        idx.embedding_version = meta.get("embedding_version", idx.embedding_version)
        idx._index = faiss.read_index(os.path.join(dir_path, "nose.index"))
        idx._templates = {d: np.asarray(t, dtype=np.float32)
                          for d, t in meta["templates"].items()}
        idx._records = meta.get("records", {
            d: {"dog_id": d, "embedding_version": idx.embedding_version}
            for d in idx._templates
        })
        ids = sorted(idx._templates.keys())
        idx._id2row = {d: i for i, d in enumerate(ids)}
        idx._row2id = {i: d for i, d in enumerate(ids)}
        return idx

    def __len__(self) -> int:
        return self.size

    def get_record(self, dog_id: str) -> dict | None:
        """Return persisted registration metadata for a stored dog."""
        record = self._records.get(str(dog_id))
        return dict(record) if record is not None else None


# Application-facing name. ``NoseIndex`` remains as a compatibility alias for
# existing galleries and scripts.
EmbeddingStore = NoseIndex
