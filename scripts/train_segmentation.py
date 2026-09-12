"""Train SegFormer (Stage 5) for rhinarium/nare/philtrum segmentation.

Consumes the per-class masks written by roboflow_import.py from a Roboflow
COCO-segmentation export. Layout expected:

    dataset/
      train/<dog>/<img>.jpg
      masks/train/<dog>/<img>.png        (HxWxC uint8, one channel per class)
      masks/train/<dog>/<img>.classes.txt

The real export may omit ``fold``. Background is derived automatically and
the six-channel output contract remains compatible with the runtime pipeline.

Usage:
    pip install torch torchvision transformers albumentations
    python scripts/train_segmentation.py --data dataset --epochs 100

Outputs: output/models/segmentation_best/  (HuggingFace checkpoint)
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import numpy as np

# Make the repository package importable when this file is launched directly
# from the documented ``python scripts\train_segmentation.py`` command.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noseid.data.synth import CLASSES, CLASS_ID

N_CLASSES = len(CLASSES)  # six-channel runtime contract; fold is optional
# Background is derived from the inverse of all supplied foreground masks.
# Fold is intentionally excluded from the validation aggregate when it is not
# present in the export.
EVALUATED_CLASSES = [c for c in CLASSES if c not in {"fold", "background"}]
EVALUATED_CLASSES.append("background")


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
    """Yield (image_path, mask_path, classes_path) tuples."""
    base = Path(data_dir) / split
    mbase = Path(data_dir) / "masks" / split
    samples = []
    if not base.exists():
        return samples
    for dog_dir in sorted(base.iterdir()):
        if not dog_dir.is_dir():
            continue
        for img in sorted(dog_dir.glob("*.[jp][pn]g")):
            mask = mbase / dog_dir.name / (img.stem + ".png")
            classes = mbase / dog_dir.name / (img.stem + ".classes.txt")
            if mask.exists():
                samples.append((img, mask, classes))
    return samples


class SegDataset:
    """Minimal torch Dataset: image + multi-class mask tensor."""

    def __init__(self, samples, image_size=512, augment=False):
        import albumentations as A
        self.samples = samples
        self.augment = augment
        self.image_size = image_size
        ops = [A.Resize(image_size, image_size)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Rotate(limit=15, p=0.5)]
        self.transform = A.Compose(
            ops,
            additional_targets={"mask": "mask"}) if A is not None else None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import torch
        img_path, mask_path, classes_path = self.samples[idx]
        import cv2
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask_stack = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        # Convert the imported per-channel mask to the fixed anatomical class
        # IDs.  The channel order is read from the sidecar file, not assumed
        # from Roboflow category order.  Unannotated pixels are background.
        if mask_stack.ndim == 2:
            labels = mask_stack.astype(np.uint8)
        else:
            labels = np.full(mask_stack.shape[:2], CLASS_ID["background"],
                             dtype=np.uint8)
            names = []
            if classes_path.exists():
                names = [x.strip() for x in classes_path.read_text(
                    encoding="utf-8").splitlines() if x.strip()]
            for channel, name in enumerate(names[:mask_stack.shape[-1]]):
                if name not in CLASS_ID:
                    continue
                labels[mask_stack[..., channel] > 0] = CLASS_ID[name]
        if self.transform is not None:
            out = self.transform(image=image, mask=labels)
            image, labels = out["image"], out["mask"]
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        m_t = torch.from_numpy(labels).long()
        return img_t, m_t


def train_segmentation(data_dir: str, epochs: int = 100, batch_size: int = 8,
                       image_size: int = 512, lr: float = 6e-5, device: str = "auto",
                       output_dir: str = "output/models",
                       backbone: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                       amp: bool = False, resume: bool = False) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)

    train_samples = _scan_samples(data_dir, "train")
    val_samples = _scan_samples(data_dir, "val") or _scan_samples(data_dir, "valid")
    if not train_samples:
        raise FileNotFoundError(
            f"no train masks found under {data_dir}/masks/train. "
            f"Run roboflow_import.py with a COCO segmentation export first.")
    print(f"[*] train samples: {len(train_samples)}, val samples: {len(val_samples)}")

    dev = _resolve_device(device, torch)
    use_cuda = dev.startswith("cuda")
    print(f"[*] device: {dev}")

    out_dir = Path(output_dir) / "segmentation_best"
    resume_dir = out_dir if resume and (out_dir / "config.json").exists() else None
    model_source = str(resume_dir) if resume_dir else backbone
    print(f"[*] model source: {model_source}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_source, num_labels=N_CLASSES,
        ignore_mismatched_sizes=not bool(resume_dir)).to(dev)
    # Prefer the local cache/checkpoint so a completed download does not
    # trigger an unnecessary Hugging Face HEAD request on every restart.
    try:
        proc = SegformerImageProcessor.from_pretrained(
            model_source, local_files_only=True)
    except OSError:
        proc = SegformerImageProcessor.from_pretrained(model_source)

    # normalize images using the backbone's expected mean/std
    mean = torch.tensor(proc.image_mean).view(3, 1, 1).to(dev)
    std = torch.tensor(proc.image_std).view(3, 1, 1).to(dev)

    train_ds = SegDataset(train_samples, image_size, augment=True)
    val_ds = SegDataset(val_samples or train_samples[:10], image_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = bool(amp and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice, best_epoch = 0.0, -1
    history = []

    def _dice(pred, gt, eps=1e-6):
        dices = []
        for name in EVALUATED_CLASSES:
            c = CLASS_ID[name]
            p = (pred == c); g = (gt == c)
            inter = (p & g).sum(); union = p.sum() + g.sum()
            dices.append((2 * inter + eps) / (union + eps))
        return float(np.mean(dices))

    for epoch in range(epochs):
        model.train(); ep_loss = 0.0; n = 0
        for imgs, masks in train_dl:
            imgs = ((imgs.to(dev) - mean) / std)
            masks = masks.to(dev)
            opt.zero_grad()
            amp_context = (torch.autocast(device_type="cuda")
                           if use_amp else nullcontext())
            with amp_context:
                out = model(pixel_values=imgs).logits
                out = torch.nn.functional.interpolate(
                    out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(out, masks)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            ep_loss += float(loss.item()); n += 1

        # validation Dice
        model.eval(); val_dice = 0.0; vn = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs = ((imgs.to(dev) - mean) / std)
                masks = masks.to(dev)
                logits = model(pixel_values=imgs).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(1)
                val_dice += _dice(pred.cpu().numpy(), masks.cpu().numpy()); vn += 1
        val_dice /= max(vn, 1)
        history.append({"epoch": epoch, "loss": ep_loss / max(n, 1),
                        "val_dice": val_dice})
        print(f"[epoch {epoch:03d}] loss={ep_loss/max(n,1):.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            model.save_pretrained(out_dir); proc.save_pretrained(out_dir)

    return {"best_epoch": best_epoch, "best_dice": best_dice,
            "history": history, "output_dir": str(out_dir)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dataset dir (COCO-mask imported)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="output/models")
    p.add_argument("--amp", action="store_true",
                   help="enable CUDA mixed precision (off by default for stability)")
    p.add_argument("--resume", action="store_true",
                   help="resume model weights from output/models/segmentation_best")
    args = p.parse_args()
    result = train_segmentation(
        data_dir=args.data, epochs=args.epochs, batch_size=args.batch_size,
        image_size=args.image_size, lr=args.lr, device=args.device,
        output_dir=args.output, amp=args.amp, resume=args.resume)
    print("\n[*] segmentation training complete:")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    print(f"    best Dice: {result['best_dice']:.4f}")
    print(f"    checkpoint: {result['output_dir']}")


if __name__ == "__main__":
    main()
