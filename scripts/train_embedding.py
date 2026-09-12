"""Train the ArcFace embedding network (Stage 7 + Stage 11).

This is THE critical training step that turns the pipeline into a real biometric
system (EER <1%). It needs:

  1. A per-identity image folder layout (Roboflow import or synth dataset):
       train/dog_001/*.jpg, train/dog_002/*.jpg, ...

  2. The torch stack installed:
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
       pip install timm albumentations

Usage:
    python scripts/train_embedding.py --train-dir dataset/train --val-dir dataset/val \\
        --epochs 100 --backbone efficientnet_v2 --embed-dim 256

The output: output/models/embedding_best.pt (usable via --weights flag in
the Embedder and CLI identify command).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from noseid.training import MetricLearningTrainer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", required=True, help="per-dog image folders")
    p.add_argument("--val-dir", default=None, help="validation split (optional)")
    p.add_argument("--backbone", default="efficientnet_v2",
                    choices=["convnext_v2", "vit", "efficientnet_v2"],
                    help="backbone for the embedding net")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None,
                    help="override config batch size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|0")
    p.add_argument("--output", default="output/models")
    args = p.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["training.lr"] = args.lr

    print("[*] embedding training")
    print(f"    train_dir   : {args.train_dir}")
    print(f"    val_dir     : {args.val_dir or '(none)'}")
    print(f"    backbone    : {args.backbone}")
    print(f"    embed_dim   : {args.embed_dim}")
    print(f"    epochs      : {args.epochs}")
    print(f"    device      : {args.device}")

    trainer = MetricLearningTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        device=args.device,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    result = trainer.train()
    print("\n[*] training complete:")
    print(json.dumps(result, indent=2))
    print(f"\n    weights saved to: {result['weights']}")
    print("    use with:  python -m noseid.cli identify --weights "
          f"{result['weights']} --image your_photo.jpg")


if __name__ == "__main__":
    main()
