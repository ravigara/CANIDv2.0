"""Stage 11 - Metric-learning trainer.

Orchestrates training of the EmbeddingNet (torch) with the combined
ArcFace + Triplet + Contrastive loss, per the spec:

    image_size: 224, batch: 32, epochs: 100, lr: 1e-4, AdamW, weight_decay: 1e-5,
    AMP on, early stopping patience 10, loss weights arcface=1.0 triplet=0.5
    contrastive=0.3.

When torch is unavailable the trainer raises a clear error - the numpy
fallback embedding is already production-quality on synth data and needs no
training. The trainer is the path to high accuracy on REAL data.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..backend import resolve_backend, availability
from ..config import get_config, cfg_get


class MetricLearningTrainer:
    def __init__(self, train_dir: str, val_dir: str | None = None,
                 backbone: str = "efficientnet_v2", embed_dim: int = 256,
                 cfg: dict | None = None, device: str = "auto",
                 output_dir: str = "output/models", epochs: int | None = None,
                 batch_size: int | None = None, lr: float | None = None):
        if not availability().get("torch"):
            raise RuntimeError(
                "torch is required for training. Install with the cu128 wheel "
                "(see requirements.txt), or use the numpy-fallback embedding "
                "which needs no training.")
        import torch
        self.cfg = cfg or get_config()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.device = self._resolve_device(device, torch)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tr = cfg_get(self.cfg, "training", {})
        self.image_size = int(tr.get("image_size", 224))
        self.batch_size = int(batch_size if batch_size is not None
                              else tr.get("batch_size", 32))
        self.epochs = int(epochs if epochs is not None else tr.get("epochs", 100))
        self.lr = float(lr if lr is not None else tr.get("lr", 1e-4))
        self.weight_decay = float(tr.get("weight_decay", 1e-5))
        self.amp = bool(tr.get("mixed_precision", True))
        self.patience = int(tr.get("early_stopping_patience", 10))
        lw = tr.get("losses", {})
        self.loss_weights = {
            "arcface": float(lw.get("arcface", 1.0)),
            "triplet": float(lw.get("triplet", 0.5)),
            "contrastive": float(lw.get("contrastive", 0.3)),
        }

    @staticmethod
    def _resolve_device(pref: str, torch) -> str:
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if str(pref).isdigit():
            if torch.cuda.is_available():
                return f"cuda:{pref}"
            return "cpu"
        return pref

    def _build_dataloaders(self):
        import torch
        from torch.utils.data import DataLoader
        from ..data.dataset import NoseDataset, NoseTripletDataset
        from ..data.augment import train_aug, val_aug
        ds = NoseDataset(self.train_dir, "train",
                         transform=train_aug(self.image_size),
                         return_tensor=True)
        if not len(ds):
            raise ValueError(f"no training images found under {self.train_dir}")
        self.num_classes = len(ds.dogs)
        self.id_to_label = ds.id_to_label
        main = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=0, drop_last=False)
        triplet = None
        try:
            tds = NoseTripletDataset(self.train_dir, "train",
                                     transform=train_aug(self.image_size),
                                     return_tensor=True,
                                     n_triplets=len(ds) * 2)
            triplet = DataLoader(tds, batch_size=self.batch_size, shuffle=True,
                                 num_workers=0)
        except Exception:
            pass
        val = None
        if self.val_dir:
            vds = NoseDataset(self.val_dir, "val",
                              transform=val_aug(self.image_size),
                              return_tensor=True, id_to_label=self.id_to_label)
            val = DataLoader(vds, batch_size=self.batch_size, shuffle=False)
        return main, triplet, val

    def train(self, quiet: bool = False) -> dict:
        import torch
        from torch.optim import AdamW
        from ..embedding.model import EmbeddingNet
        from ..embedding.losses import CombinedMetricLoss
        from .metrics import evaluate_pairs, compute_biometric_metrics

        main_loader, triplet_loader, val_loader = self._build_dataloaders()
        net = EmbeddingNet(self.backbone, self.embed_dim,
                           pretrained=True, num_classes=self.num_classes
                           ).to(self.device)
        loss_fn = CombinedMetricLoss(
            self.embed_dim, self.num_classes,
            w_arcface=self.loss_weights["arcface"],
            w_triplet=self.loss_weights["triplet"],
            w_contrastive=self.loss_weights["contrastive"]).to(self.device)
        params = list(net.parameters()) + list(loss_fn.parameters())
        opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp and
                                           self.device == "cuda")

        best_val, best_epoch, no_improve = 1e9, -1, 0
        history = []
        triplet_iter = iter(triplet_loader) if triplet_loader else None
        for epoch in range(self.epochs):
            net.train(); ep_loss = 0.0; n = 0
            for imgs, targets in main_loader:
                imgs = imgs.to(self.device)
                labels = targets["label"].to(self.device)
                # optional triplet batch
                triplets = None
                if triplet_iter is not None:
                    try:
                        a, p, nn = next(triplet_iter)
                        triplets = (a.to(self.device), p.to(self.device),
                                    nn.to(self.device))
                    except StopIteration:
                        triplet_iter = iter(triplet_loader)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.amp and
                                             self.device == "cuda"):
                    emb = net(imgs)
                    loss = loss_fn(emb, labels,
                                   triplets=tuple(
                                       net(x) for x in triplets) if triplets else None)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                ep_loss += float(loss.item()); n += 1
            # validation (embedding separation)
            val_metrics = {}
            if val_loader is not None:
                embs_all, labels_all = [], []
                net.eval()
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = imgs.to(self.device)
                        e = net(imgs).cpu().numpy()
                        embs_all.append(e)
                        labels_all.extend(targets["label"].tolist())
                embs_all = np.concatenate(embs_all)
                gen, imp = evaluate_pairs(embs_all, np.asarray(labels_all))
                val_metrics = compute_biometric_metrics(gen, imp)
            rec = {"epoch": epoch, "loss": ep_loss / max(n, 1), "val": val_metrics}
            history.append(rec)
            if not quiet:
                print(f"[epoch {epoch:03d}] loss={rec['loss']:.4f} "
                      f"val_EER={val_metrics.get('EER','NA')}")
            # early stopping on validation EER (lower better)
            cur = val_metrics.get("EER", rec["loss"])
            if cur < best_val:
                best_val, best_epoch, no_improve = cur, epoch, 0
                torch.save({"model": net.state_dict(), "epoch": epoch,
                            "id_to_label": self.id_to_label},
                           self.output_dir / "embedding_best.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if not quiet:
                        print(f"early stop at epoch {epoch} (best {best_epoch})")
                    break
        return {"best_epoch": best_epoch, "best_val": best_val,
                "history": history,
                "weights": str(self.output_dir / "embedding_best.pt")}
