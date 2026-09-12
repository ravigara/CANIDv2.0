"""End-to-end CLI for the Dog Nose Biometric ID System.

Subcommands:
  generate   build a synthetic dataset (Stage 1)
  enroll     enroll a dog from a folder of images into the FAISS index (Stage 8)
  identify   identify a dog from a single image (Stage 9)
  demo       one-shot: generate a tiny dataset, enroll N dogs, run identification
  evaluate   compute FAR/FRR/EER/AUC over a dataset (Stage 11)

Examples:
  python -m noseid.cli demo --dogs 12 --images 8
  python -m noseid.cli generate --out dataset --dogs 50 --images 12
  python -m noseid.cli enroll --index output/index --dog DOG_001 --dir my_images
  python -m noseid.cli identify --index output/index --image photo.jpg
  python -m noseid.cli evaluate --dataset dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2))


def cmd_demo(args) -> int:
    """Generate -> enroll -> identify end-to-end on synthetic data."""
    from .data import NoseSynthGenerator
    from .biometric import build_pipeline, GTHint, run_to_embedding, identify
    from .matching import NoseIndex

    g = NoseSynthGenerator(256, seed=args.seed)
    pc = build_pipeline()
    idx = NoseIndex(dim=256)
    dogs = [f"dog_{i:03d}" for i in range(1, args.dogs + 1)]
    print(f"[*] enrolling {args.dogs} dogs ({args.images} images each)...")
    for did in dogs:
        embs = []
        for i in range(args.images):
            s = g.render_labeled(did, i)
            _, _, _, _, er = run_to_embedding(pc, s["image"],
                                              GTHint.from_sample(s), did)
            embs.append(np.asarray(er.embedding, dtype=np.float32))
        idx.enroll(did, embs)
    print(f"[*] index size: {idx.size}")

    print("[*] identifying held-out genuine images...")
    ok = 0
    for did in dogs:
        s = g.render_labeled(did, args.images + 1)
        res, _ = identify(pc, idx, s["image"], GTHint.from_sample(s))
        ok += int(res.dog_id == did and res.status == "Verified")
    print(f"    genuine verified: {ok}/{args.dogs}")

    print("[*] identifying unknown (un-enrolled) dogs...")
    uok = 0
    for i in range(100, 100 + args.dogs):
        s = g.render_labeled(f"dog_{i:03d}", 0)
        res, _ = identify(pc, idx, s["image"], GTHint.from_sample(s))
        uok += int(res.status in ("Unknown Dog", "Rejected"))
    print(f"    unknown rejected: {uok}/{args.dogs}")

    print("[*] sample identification result:")
    s = g.render_labeled(dogs[0], args.images + 2)
    res, _ = identify(pc, idx, s["image"], GTHint.from_sample(s))
    _print_json(res.to_dict())
    return 0


def cmd_generate(args) -> int:
    from .data import generate_dataset
    m = generate_dataset(args.out, num_dogs=args.dogs,
                         images_per_dog=args.images,
                         image_size=256, seed=args.seed)
    _print_json({"status": "generated", **m})
    return 0


def _load_image_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cmd_enroll(args) -> int:
    from .biometric import build_pipeline, enroll_dog, GTHint
    from .matching import NoseIndex
    from .config import get_config
    idx_path = Path(args.index)
    if (idx_path / "meta.json").exists():
        idx = NoseIndex.load(str(idx_path))
    else:
        idx = NoseIndex(dim=256)
    overrides = {}
    if args.weights:
        overrides["embedding"] = {"weights": args.weights}
    if args.strict_quality:
        overrides["validation"] = {"hard_reject": True}
    cfg = get_config(overrides or None)
    pc = build_pipeline(cfg)
    d = Path(args.dir)
    files = sorted([*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png")])
    if not files:
        print(f"[!] no images found in {d}", file=sys.stderr); return 1
    imgs = [_load_image_rgb(str(f)) for f in files]
    res = enroll_dog(pc, idx, args.dog, imgs, min_valid=args.min_valid)
    idx.save(str(idx_path))
    _print_json(res.to_dict())
    return 0


def cmd_identify(args) -> int:
    from .biometric import build_pipeline, identify
    from .matching import NoseIndex
    from .config import get_config
    idx = NoseIndex.load(args.index)
    overrides = {}
    if args.weights:
        overrides["embedding"] = {"weights": args.weights}
    if args.strict_quality:
        overrides["validation"] = {"hard_reject": True}
    cfg = get_config(overrides or None)
    pc = build_pipeline(cfg)
    img = _load_image_rgb(args.image)
    res, meta = identify(pc, idx, img)
    output = res.to_dict()
    if meta.get("validation_warning"):
        output["validation_warning"] = meta["validation_warning"]
        output["validation_scores"] = meta["validation"].get("scores", {})
    _print_json(output)
    return 0 if res.status != "Error" else 1


def cmd_evaluate(args) -> int:
    from .biometric import build_pipeline, GTHint, run_to_embedding
    from .training import evaluate_pairs, compute_biometric_metrics
    from .data import NoseDataset, val_aug
    pc = build_pipeline()
    ds = NoseDataset(args.dataset, "val", transform=val_aug(224),
                     return_tensor=False)
    X, y = [], []
    for i in range(min(len(ds), args.max_samples)):
        img, t = ds[i]
        # use detector bbox via gt-free path; for synth use direct embed
        er = pc.embedder.embed(img, dog_id=t["dog_id"])
        X.append(np.asarray(er.embedding)); y.append(t["label"])
    X = np.stack(X); y = np.asarray(y)
    gen, imp = evaluate_pairs(X, y, max_pairs=3000)
    m = compute_biometric_metrics(gen, imp)
    _print_json(m)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="noseid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end synthetic demo")
    d.add_argument("--dogs", type=int, default=12)
    d.add_argument("--images", type=int, default=8)
    d.add_argument("--seed", type=int, default=3)
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("generate", help="generate synthetic dataset (Stage 1)")
    g.add_argument("--out", default="dataset")
    g.add_argument("--dogs", type=int, default=20)
    g.add_argument("--images", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("enroll", help="enroll a dog from images (Stage 8)")
    e.add_argument("--index", default="output/index")
    e.add_argument("--dog", required=True)
    e.add_argument("--dir", required=True)
    e.add_argument("--min-valid", type=int, default=3)
    e.add_argument("--weights", default=None,
                   help="optional embedding checkpoint")
    e.add_argument("--strict-quality", action="store_true",
                   help="reject images failing the quality gate")
    e.set_defaults(func=cmd_enroll)

    i = sub.add_parser("identify", help="identify a dog (Stage 9)")
    i.add_argument("--index", default="output/index")
    i.add_argument("--image", required=True)
    i.add_argument("--weights", default=None,
                   help="optional embedding checkpoint")
    i.add_argument("--strict-quality", action="store_true",
                   help="reject images failing the quality gate")
    i.set_defaults(func=cmd_identify)

    v = sub.add_parser("evaluate", help="compute FAR/FRR/EER/AUC (Stage 11)")
    v.add_argument("--dataset", default="dataset")
    v.add_argument("--max-samples", type=int, default=400)
    v.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
