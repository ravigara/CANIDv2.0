"""Train a landmark detector (Stage 4) for left/right nare + philtrum.

Consumes COCO keypoint annotations imported by roboflow_import.py:

    dataset/keypoints/<split>/<dog>/<img>.json   {"keypoints": [[x,y,v],...]}

Architecture: a small CNN backbone + per-keypoint heatmap regression (the
MediaPipe / HeatmapMSE style). Predicts K heatmaps; argmax + soft-argmax gives
subpixel keypoint coordinates.

Usage:
    pip install torch torchvision albumentations
    python scripts/train_landmarks.py --data dataset --epochs 100

Outputs: output/models/landmarks_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noseid.pipeline.landmark_model import (  # noqa: E402
    KEYPOINT_NAMES, NUM_KPTS, build_landmark_model)



def _resolve_device(device: str, torch) -> str:
    """Normalize CLI values such as ``0`` to a valid PyTorch device."""
    value = str(device).lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
        return f"cuda:{value}"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return str(device)


def _scan_samples(data_dir: str, split: str):
    base = Path(data_dir) / split
    kbase = Path(data_dir) / "keypoints" / split
    samples = []
    if not base.exists():
        return samples
    for dog_dir in sorted(base.iterdir()):
        if not dog_dir.is_dir():
            continue
        for img in sorted(dog_dir.glob("*.[jp][pn]g")):
            kp = kbase / dog_dir.name / (img.stem + ".json")
            if kp.exists():
                samples.append((img, kp))
    return samples


def _gaussian_heatmap(cx: float, cy: float, h: int, w: int,
                      sigma: float = 2.0) -> np.ndarray:
    """Render a single 2D Gaussian centered at (cx, cy) in a HxW grid."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


class KptDataset:
    """Image + (K, H/4, W/4) heatmaps at 1/4 resolution."""

    def __init__(self, samples, image_size=256, augment=False):
        import albumentations as A
        self.samples = samples
        self.image_size = image_size
        self.augment = bool(augment)
        ops = [A.Resize(image_size, image_size)]
        if augment:
            # Horizontal flipping is performed explicitly in __getitem__ so
            # the anatomical left/right keypoint identities can be swapped.
            # Other geometric transforms are applied to image and keypoints
            # together by Albumentations.
            ops += [A.ColorJitter(p=0.3),
                    A.Rotate(limit=15, p=0.5)]
        self.transform = A.Compose(
            ops,
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import cv2
        img_path, kp_path = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        kpts = json.loads(kp_path.read_text())["keypoints"]
        kpts = np.asarray(kpts[:NUM_KPTS], dtype=np.float32)  # [K,3]

        # Mirror the image only with an explicit anatomical swap. Without
        # this, a mirrored image would retain incorrect left/right targets.
        if self.augment and np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            kpts[:, 0] = (w0 - 1) - kpts[:, 0]
            kpts[[0, 1]] = kpts[[1, 0]]

        # Albumentations now transforms coordinates for Resize/Rotate along
        # with the image. Visibility values remain attached to each point.
        out = self.transform(image=image, keypoints=kpts[:, :2].tolist())
        image = out["image"]
        h, w = image.shape[:2]
        transformed = np.asarray(out["keypoints"], dtype=np.float32)
        if transformed.shape == (NUM_KPTS, 2):
            kpts[:, :2] = transformed
        # heatmaps at 1/4 res
        hm_h, hm_w = h // 4, w // 4
        heatmaps = np.zeros((NUM_KPTS, hm_h, hm_w), dtype=np.float32)
        for k in range(min(NUM_KPTS, len(kpts))):
            x, y, v = kpts[k]
            if v > 0:
                heatmaps[k] = _gaussian_heatmap(x / 4, y / 4, hm_h, hm_w, sigma=1.5)
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        hm_t = torch.from_numpy(heatmaps)
        return img_t, hm_t


def _build_model():
    """Build the shared EfficientNet/MobileNet landmark heatmap model."""
    return build_landmark_model(pretrained=True)


def train_landmarks(data_dir: str, epochs: int = 100, batch_size: int = 16,
                    image_size: int = 256, lr: float = 1e-4, device: str = "auto",
                    output_dir: str = "output/models") -> dict:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    train_samples = _scan_samples(data_dir, "train")
    val_samples = _scan_samples(data_dir, "val") or _scan_samples(data_dir, "valid")
    if not train_samples:
        raise FileNotFoundError(
            f"no keypoint json under {data_dir}/keypoints/train. "
            f"Run roboflow_import.py with a COCO keypoint export first.")
    print(f"[*] train samples: {len(train_samples)}, val samples: {len(val_samples)}")

    dev = _resolve_device(device, torch)
    use_cuda = dev.startswith("cuda")
    print(f"[*] device: {dev}")

    model = _build_model().to(dev)
    train_ds = KptDataset(train_samples, image_size, augment=True)
    val_ds = KptDataset(val_samples or train_samples[:10], image_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    loss_fn = torch.nn.MSELoss()

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, history = 1e9, -1, []

    for epoch in range(epochs):
        model.train(); ep_loss = 0.0; n = 0
        for imgs, heatmaps in train_dl:
            imgs = imgs.to(dev); heatmaps = heatmaps.to(dev)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                pred = model(imgs)
                loss = loss_fn(pred, heatmaps)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()); n += 1
        # validation
        model.eval(); v_loss = 0.0; vn = 0
        with torch.no_grad():
            for imgs, heatmaps in val_dl:
                imgs = imgs.to(dev); heatmaps = heatmaps.to(dev)
                v_loss += float(loss_fn(model(imgs), heatmaps).item()); vn += 1
        v_loss /= max(vn, 1)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(n, 1),
                        "val_loss": v_loss})
        print(f"[epoch {epoch:03d}] train={ep_loss/max(n,1):.5f} val={v_loss:.5f}")
        if v_loss < best_loss:
            best_loss, best_epoch = v_loss, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "keypoints": KEYPOINT_NAMES},
                       out_dir / "landmarks_best.pt")
    return {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "history": history, "weights": str(out_dir / "landmarks_best.pt")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dataset dir (COCO-keypoint imported)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="output/models")
    args = p.parse_args()
    result = train_landmarks(
        data_dir=args.data, epochs=args.epochs, batch_size=args.batch_size,
        image_size=args.image_size, lr=args.lr, device=args.device,
        output_dir=args.output)
    print("\n[*] landmark training complete:")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
