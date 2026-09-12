"""Import a Roboflow export into the noseid training layout.

Roboflow exports (YOLO or COCO) as:
    roboflow_export/
      train/images/*.jpg     (flat, all dogs mixed)
      train/labels/*.txt     (YOLO: class_id cx cy w h per line)
      valid/images/...
      test/images/...
      data.yaml              (class names + path mappings)

This script converts to noseid's per-identity layout:
    dataset/
      train/dog_001/001.jpg   (one folder per dog)
      train/dog_002/001.jpg
      val/dog_001/001.jpg
      ...

Usage:
    python scripts/roboflow_import.py --source roboflow_export --out dataset
    python scripts/roboflow_import.py --source roboflow_export --out dataset --split val

The identity mapping is inferred from the YOLO label filenames if Roboflow
was set up with a 'dog_id' metadata field, OR you provide a CSV mapping file:
    image_name,dog_id
    img_001.jpg,dog_001
    img_002.jpg,dog_002

If no mapping is provided, images are auto-assigned unique IDs (one per N images).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# Keep exported mask channels deterministic.  ``fold`` and ``background`` are
# optional in the Roboflow project: background can be derived from the
# foreground masks, and a missing fold simply means that feature is not being
# trained.
CANONICAL_MASK_CLASSES = [
    "rhinarium", "left_nare", "right_nare", "philtrum", "fold", "background",
]


def _parse_yaml_simple(p: Path) -> dict:
    """Minimal YAML parser for Roboflow's data.yaml (no pyyaml required)."""
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


# --- COCO parsing (segmentation polygons + keypoints) -----------------------

def _detect_format(src: Path, split: str) -> str:
    """Auto-detect 'yolo' vs 'coco' export layout."""
    # COCO: a *_annotations.json or annotations.json under the split or root
    coco_candidates = [
        src / split / "_annotations.coco.json",
        src / split / "annotations.json",
        src / "_annotations.coco.json",
        src / "annotations.json",
    ]
    for c in coco_candidates:
        if c.exists():
            return "coco"
    return "yolo"


def _coco_polygon_to_mask(segments: list, h: int, w: int) -> np.ndarray:
    """Rasterize COCO polygon segmentation (list of polygons) to a binary mask."""
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    if not segments:
        return mask
    for seg in segments:
        if isinstance(seg, list):           # polygon: [x1,y1,x2,y2,...]
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        elif isinstance(seg, dict):          # RLE
            # best-effort: skip compressed RLE (needs pycocotools)
            continue
    return mask


def _load_coco(json_path: Path) -> dict:
    """Return {image_id: {file_name, width, height, masks: {class: ndarray},
                          keypoints: [[x,y,v],...]}} indexed from a COCO file."""
    import json as _json
    with open(json_path, "r", encoding="utf-8") as fh:
        coco = _json.load(fh)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    out: dict[int, dict] = {}
    for img in coco.get("images", []):
        out[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"], "height": img["height"],
            "masks": {}, "keypoints": [],
        }
    for ann in coco.get("annotations", []):
        iid = ann.get("image_id")
        if iid not in out:
            continue
        rec = out[iid]
        cat = cat_id_to_name.get(ann.get("category_id"), "object")
        h, w = rec["height"], rec["width"]
        # segmentation -> per-class mask
        if "segmentation" in ann:
            m = _coco_polygon_to_mask(ann["segmentation"], h, w)
            if cat not in rec["masks"]:
                rec["masks"][cat] = np.zeros((h, w), dtype=np.uint8)
            rec["masks"][cat] |= m
        # bbox -> also stash as a mask rectangle for detection fallback
        if "bbox" in ann and "segmentation" not in ann:
            x, y, bw, bh = ann["bbox"]
            if cat not in rec["masks"]:
                rec["masks"][cat] = np.zeros((h, w), dtype=np.uint8)
            rec["masks"][cat][int(y):int(y + bh), int(x):int(x + bw)] = 1
        # keypoints: flat list [x1,y1,v1, x2,y2,v2, ...]
        if "keypoints" in ann:
            kp = ann["keypoints"]
            rec["keypoints"] = [kp[i:i + 3] for i in range(0, len(kp), 3)]
    return out


# --- main import ------------------------------------------------------------

def import_roboflow(source: str, out: str, split: str = "train",
                    mapping_csv: str | None = None, images_per_dog: int = 0,
                    seed: int = 42) -> dict:
    src = Path(source)
    dst = Path(out)
    fmt = _detect_format(src, split)
    img_dir = src / split / "images"
    # COCO puts images directly under split/, Roboflow variant under images/
    if not img_dir.exists():
        img_dir = src / split

    if not img_dir.exists():
        raise FileNotFoundError(f"no image dir for split {split} under {src}")

    # COCO branch
    if fmt == "coco":
        return _import_coco(src, dst, split, img_dir, mapping_csv,
                            images_per_dog, seed)
    return _import_yolo(src, dst, split, img_dir, mapping_csv,
                        images_per_dog, seed)


def _resolve_dog_ids(images: list, mapping_csv: str | None,
                     images_per_dog: int) -> list[str]:
    """Decide the dog identity for each image (CSV mapping or auto-assign)."""
    if mapping_csv:
        import csv
        id_map: dict[str, str] = {}
        with open(mapping_csv, "r") as f:
            for row in csv.DictReader(f):
                id_map[row["image_name"]] = row["dog_id"]
        return [id_map.get(p.name, p.stem) for p in images]
    per = max(1, images_per_dog)
    return [f"dog_{(i // per) + 1:04d}" for i in range(len(images))]


def _import_yolo(src, dst, split, img_dir, mapping_csv, images_per_dog, seed) -> dict:
    lbl_dir = src / split / "labels"
    exts = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for e in exts:
        images.extend(sorted(img_dir.glob(e)))
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")
    dog_ids = _resolve_dog_ids(images, mapping_csv, images_per_dog)
    counts: dict[str, int] = {}
    for img_path, dog_id in zip(images, dog_ids):
        out_dir = dst / split / dog_id
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_dir / img_path.name)
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, out_dir / lbl.name)
        counts[dog_id] = counts.get(dog_id, 0) + 1
    for cand in (src / "data.yaml", src / "dataset.yaml"):
        if cand.exists():
            shutil.copy2(cand, dst / cand.name); break
    return {"split": split, "format": "yolo", "n_images": len(images),
            "n_dogs": len(counts), "output_dir": str(dst / split)}


def _import_coco(src, dst, split, img_dir, mapping_csv, images_per_dog, seed) -> dict:
    import json as _json
    # locate the COCO json
    coco_json = None
    for cand in (src / split / "_annotations.coco.json",
                 src / split / "annotations.json",
                 src / "_annotations.coco.json",
                 src / "annotations.json"):
        if cand.exists():
            coco_json = cand; break
    if coco_json is None:
        raise FileNotFoundError("COCO annotations json not found")
    records = _load_coco(coco_json)
    # build a list of (image_path, record) preserving order
    pairs = []
    for rec in records.values():
        p = img_dir / rec["file_name"]
        if p.exists():
            pairs.append((p, rec))
    images = [p for p, _ in pairs]
    dog_ids = _resolve_dog_ids(images, mapping_csv, images_per_dog)

    masks_out = dst / "masks" / split
    kpts_out = dst / "keypoints" / split
    counts: dict[str, int] = {}
    n_with_masks = n_with_kpts = 0
    for (img_path, rec), dog_id in zip(pairs, dog_ids):
        out_dir = dst / split / dog_id
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_dir / img_path.name)
        # write per-class masks (one PNG per image, HxWxC) if present
        if rec["masks"]:
            n_with_masks += 1
            # Do not use alphabetical order here.  The trainer and runtime
            # use the anatomical contract above, while Roboflow category IDs
            # and names may arrive in a different order.
            classes = [c for c in CANONICAL_MASK_CLASSES if c in rec["masks"]]
            # Preserve any unexpected category rather than silently dropping
            # it; the training loader will report/ignore unknown names.
            classes += [c for c in rec["masks"] if c not in classes]
            h, w = rec["height"], rec["width"]
            stack = np.stack([rec["masks"][c] for c in classes], axis=-1)
            mdir = masks_out / dog_id; mdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mdir / (img_path.stem + ".png")), stack)
            (mdir / (img_path.stem + ".classes.txt")).write_text("\n".join(classes))
        # write keypoints (one JSON per image) if present
        if rec["keypoints"]:
            n_with_kpts += 1
            kdir = kpts_out / dog_id; kdir.mkdir(parents=True, exist_ok=True)
            (kdir / (img_path.stem + ".json")).write_text(
                _json.dumps({"keypoints": rec["keypoints"]}))
        counts[dog_id] = counts.get(dog_id, 0) + 1
    return {"split": split, "format": "coco", "n_images": len(images),
            "n_with_masks": n_with_masks, "n_with_keypoints": n_with_kpts,
            "n_dogs": len(counts), "output_dir": str(dst / split)}

    # load identity mapping
    id_map: dict[str, str] = {}
    if mapping_csv:
        with open(mapping_csv, "r") as f:
            for row in csv.DictReader(f):
                id_map[row["image_name"]] = row["dog_id"]
    else:
        # check if Roboflow metadata JSON exists per-image (some projects export it)
        meta_dir = src / split / "images"  # Roboflow puts .json next to .jpg
        # fall back to auto-assign
        pass

    exts = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for e in exts:
        images.extend(sorted(img_dir.glob(e)))

    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")

    # assign dog_ids
    if id_map:
        dog_ids = [id_map.get(p.name, p.stem) for p in images]
    else:
        # auto-assign: images_per_dog images per identity
        if images_per_dog <= 0:
            images_per_dog = 1  # each image is its own dog (worst case)
        dog_ids = []
        for i, p in enumerate(images):
            dog_ids.append(f"dog_{(i // images_per_dog) + 1:04d}")

    counts: dict[str, int] = {}
    for img_path, dog_id in zip(images, dog_ids):
        out_dir = dst / split / dog_id
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_dir / img_path.name)
        # copy label if exists
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, out_dir / lbl.name)
        counts[dog_id] = counts.get(dog_id, 0) + 1

    # copy data.yaml to output root
    for candidate in src.glob("data.yaml"), src.glob("dataset.yaml"):
        if candidate.exists():
            shutil.copy2(candidate, dst / candidate.name)
            break

    return {"split": split, "n_images": len(images), "n_dogs": len(counts),
            "output_dir": str(dst / split)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Roboflow export root")
    parser.add_argument("--out", default="dataset", help="output dir (noseid layout)")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--mapping-csv", default=None, help="image_name->dog_id CSV")
    parser.add_argument("--images-per-dog", type=int, default=0,
                        help="auto-assign N images per identity if no mapping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = import_roboflow(args.source, args.out, args.split,
                             args.mapping_csv, args.images_per_dog, args.seed)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
