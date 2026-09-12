"""Stage 1 - Synthetic dog-nose image generator.

Why this exists: real rhinarium data is hard to source, so we generate
*visually plausible* synthetic noses whose per-dog texture is a deterministic
function of a dog id. This gives the entire pipeline (detection -> landmarks ->
segmentation -> embedding -> FAISS matching) something realistic to chew on
TODAY without any real data.

Each "dog" gets a unique seeded texture (grooves, ridges, pigment blobs);
per-image variation (lighting, angle, blur, jitter) is applied so the
metric-learning objective is non-trivial.

Also produces labels the later stages consume:
  - bbox  (YOLO)         : one NOSE01 box per image
  - mask  (per-class)    : rhinarium / L-nare / R-nare / philtrum / fold / bg
  - kpts  (landmarks)    : left_nare, right_nare, philtrum
  - meta  (json)         : blur/brightness/angle for the validator
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["rhinarium", "left_nare", "right_nare", "philtrum", "fold", "background"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _draw_grooves(canvas: np.ndarray, rng: np.random.Generator,
                  cx: float, cy: float, r: float, density: float,
                  pigment: float) -> None:
    """Curved groove/ridge lines + pigment blobs -> 'micro' texture."""
    n = int(density * 80)
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(r * 0.3, r * 0.95)
        x0 = cx + math.cos(ang) * r * 0.2
        y0 = cy + math.sin(ang) * r * 0.2
        x1 = cx + math.cos(ang) * length
        y1 = cy + math.sin(ang) * length
        thick = int(rng.integers(1, 3))
        shade = int(rng.integers(20, 90))
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)),
                 (shade, shade, max(shade - 10, 0)), thick, lineType=cv2.LINE_AA)
    for _ in range(int(8 + pigment * 12)):
        bx = cx + rng.normal(0, r * 0.4)
        by = cy + rng.normal(0, r * 0.4)
        br = int(rng.uniform(2, 7))
        col = int(rng.integers(60, 180))
        cv2.circle(canvas, (int(bx), int(by)), br, (col, max(col - 10, 0), max(col - 20, 0)), -1)


def _dog_anatomy(size: int, dog_seed: int) -> dict:
    """Deterministic per-dog anatomy: center, radii, nares, pigment."""
    rng = _rng(dog_seed)
    cx, cy = size / 2, size / 2          # canonical center for the template
    r = size * rng.uniform(0.32, 0.40)
    nare_sep = r * rng.uniform(0.28, 0.42)
    nare_r = r * rng.uniform(0.10, 0.16)
    left = (cx - nare_sep / 2, cy + rng.uniform(-0.05, 0.08) * r)
    right = (cx + nare_sep / 2, cy + rng.uniform(-0.05, 0.08) * r)
    phil = (cx, cy + r * rng.uniform(0.30, 0.45))
    return {
        "cx": cx, "cy": cy, "r": r,
        "left_nare": left, "right_nare": right, "philtrum": phil,
        "nare_r": nare_r,
        "density": float(rng.uniform(0.9, 1.3)),
        "pigment": float(rng.uniform(0.2, 1.0)),
        "base": int(rng.integers(140, 200)),
        "dog_seed": dog_seed,
    }


def _dog_texture_template(size: int, anat: dict) -> np.ndarray:
    """Render a CANONICAL per-dog rhinarium texture template, drawn once.

    This template is the dog's identity signal: a deterministic, STRONG groove +
    ridge + pigment pattern. Per-image rendering warps this template by the pose
    transform (rather than redrawing random grooves), exactly like a fingerprint
    viewed under different poses. The pattern amplitude is large relative to the
    shared anatomy so cross-dog cosine similarity stays well below 1.
    """
    rng = _rng(anat["dog_seed"])
    cx, cy, r = anat["cx"], anat["cy"], anat["r"]
    # base rhinarium tone + soft disk mask (common anatomy, low weight)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = np.clip(1.0 - (dist - r) / (r * 0.10), 0.0, 1.0)
    tmpl = np.full((size, size), anat["base"], dtype=np.float32) * disk + 30 * (1 - disk)

    ang_grid = np.arctan2(yy - cy, xx - cx)
    rid_norm = (dist / r).clip(0, 1)

    # Per-dog primary ridge field: unique (radial freq, phase) + strong depth.
    radial_freq = rng.uniform(0.10, 0.22) * size
    radial_phase = rng.uniform(0, 2 * math.pi)
    ridge_depth = rng.uniform(70.0, 95.0)            # strong grooves
    ridge = np.sin(radial_freq * rid_norm + radial_phase)
    # per-dog angular modulation (the "fingerprint" direction pattern)
    n_ang = int(rng.integers(3, 6))
    angular_mod = np.zeros_like(dist)
    for _ in range(n_ang):
        f = rng.uniform(2.0, 9.0)
        p = rng.uniform(0, 2 * math.pi)
        a = rng.uniform(0.4, 1.0)
        angular_mod += a * np.sin(f * ang_grid + p)
    angular_mod = (angular_mod - angular_mod.mean())
    tmpl -= ridge_depth * (0.5 + 0.5 * ridge) * disk * (0.6 + 0.4 * (angular_mod / (np.abs(angular_mod).max() + 1e-6)))

    # Per-dog secondary high-frequency micro-ridges (cracks/grooves).
    micro_f = rng.uniform(0.35, 0.55) * size
    micro_p = rng.uniform(0, 2 * math.pi)
    tmpl -= rng.uniform(20.0, 35.0) * np.sin(micro_f * rid_norm + micro_p) * disk

    # Per-dog macro features: distinctive dark ridge clusters at fixed positions.
    n_clusters = int(rng.integers(5, 9))
    for _ in range(n_clusters):
        cdist = rng.uniform(0.2, 0.85) * r
        cang = rng.uniform(0, 2 * math.pi)
        bx = cx + cdist * np.cos(cang)
        by = cy + cdist * np.sin(cang)
        br = int(rng.uniform(6, 12))
        shade = int(rng.integers(20, 70))
        cv2.circle(tmpl, (int(bx), int(by)), br, shade, -1)

    # Per-dog pigment blobs (fixed positions).
    n_pig = int(8 + anat["pigment"] * 10)
    for _ in range(n_pig):
        bx = cx + rng.normal(0, r * 0.45)
        by = cy + rng.normal(0, r * 0.45)
        br = int(rng.uniform(3, 8))
        shade = int(rng.integers(50, 130))
        cv2.circle(tmpl, (int(bx), int(by)), br, shade, -1)

    # nares (fixed position in template space)
    cv2.circle(tmpl, (int(anat["left_nare"][0]), int(anat["left_nare"][1])),
               int(anat["nare_r"]), 5, -1)
    cv2.circle(tmpl, (int(anat["right_nare"][0]), int(anat["right_nare"][1])),
               int(anat["nare_r"]), 5, -1)
    cv2.line(tmpl,
             (int((anat["left_nare"][0] + anat["right_nare"][0]) / 2),
              int((anat["left_nare"][1] + anat["right_nare"][1]) / 2)),
             (int(anat["philtrum"][0]), int(anat["philtrum"][1])), 15, 2)
    return np.clip(tmpl, 0, 255).astype(np.uint8)


def _render(size: int, anat: dict, tmpl: np.ndarray, img_seed: int) -> tuple[np.ndarray, dict]:
    """Render one nose image by warping the per-dog template + adding noise.

    Per-image variation (small pose/scale/lighting/blur) is added on top of a
    STABLE per-dog texture template so identity dominates over noise.
    """
    rng = _rng(img_seed)
    fur_base = int(rng.integers(110, 165))
    img = np.full((size, size, 3), fur_base, dtype=np.uint8)
    speckle = rng.normal(0, 10, (size, size, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + speckle, 0, 255).astype(np.uint8)

    scale = float(rng.uniform(0.94, 1.05))
    dx = float(rng.normal(0, size * 0.015))
    dy = float(rng.normal(0, size * 0.015))
    rot = float(rng.normal(0, 4))                       # degrees, simulates yaw
    cx = anat["cx"] + dx
    cy = anat["cy"] + dy
    r = anat["r"] * scale

    # background lighting gradient (indoor/outdoor)
    light_dir = rng.uniform(0, 2 * math.pi)
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    g = (np.sin((xs / size) * math.pi + light_dir) * 12 +
         np.cos((ys / size) * math.pi) * 12).astype(np.float32)
    img = img.astype(np.float32)
    img += g[..., None]

    # warp the per-dog template into this image's pose
    M = cv2.getRotationMatrix2D((cx, cy), rot, scale)
    M[0, 2] += dx; M[1, 2] += dy
    warped_tmpl = cv2.warpAffine(tmpl, M, (size, size),
                                 borderMode=cv2.BORDER_REPLICATE)
    # soft rhinarium mask in image space
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disk = np.clip(1.0 - (dist - r) / (r * 0.10), 0.0, 1.0)
    for c in range(3):
        img[..., c] = img[..., c] * (1 - disk) + warped_tmpl * disk
    img = np.clip(img, 0, 255).astype(np.uint8)

    # warp landmarks into image space for labels
    def _warp(p):
        return (M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2],
                M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])
    left = _warp(anat["left_nare"])
    right = _warp(anat["right_nare"])
    phil = _warp(anat["philtrum"])
    nare_r = anat["nare_r"] * scale

    beta = float(rng.uniform(-25, 25))                  # lighting condition
    img = np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)

    blur_strength = float(rng.uniform(0, 1))
    if blur_strength > 0.8:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    meta = {
        "cx": cx, "cy": cy, "r": r, "rot": rot,
        "left_nare": list(left), "right_nare": list(right), "philtrum": list(phil),
        "nare_r": nare_r, "brightness_offset": beta, "blur": blur_strength,
        "scale": scale,
    }
    return img, meta


def _build_masks(size: int, meta: dict) -> np.ndarray:
    """6-class HxWxC uint8 mask aligned with CLASSES."""
    masks = np.zeros((size, size, len(CLASSES)), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy, r = meta["cx"], meta["cy"], meta["r"]
    rhin = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < r
    masks[..., CLASS_ID["rhinarium"]] = rhin.astype(np.uint8)

    nr = meta["nare_r"]
    ln, rn, phil = meta["left_nare"], meta["right_nare"], meta["philtrum"]
    masks[..., CLASS_ID["left_nare"]] = (((xx - ln[0]) ** 2 + (yy - ln[1]) ** 2) < (nr * 1.2) ** 2).astype(np.uint8)
    masks[..., CLASS_ID["right_nare"]] = (((xx - rn[0]) ** 2 + (yy - rn[1]) ** 2) < (nr * 1.2) ** 2).astype(np.uint8)
    midx = (ln[0] + rn[0]) / 2
    ph_band = (np.abs(xx - midx) < nr * 0.6) & (yy > (ln[1] + rn[1]) / 2) & (yy < phil[1])
    masks[..., CLASS_ID["philtrum"]] = ph_band.astype(np.uint8)
    edge = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) > r * 0.7) & rhin
    masks[..., CLASS_ID["fold"]] = edge.astype(np.uint8)
    masks[..., CLASS_ID["background"]] = (~rhin).astype(np.uint8)
    return masks


def _bbox_from_meta(size: int, meta: dict) -> list[float]:
    """YOLO/XYXY box tightly around the rhinarium disk."""
    pad = meta["r"] * 1.12
    x1 = max(0.0, meta["cx"] - pad)
    y1 = max(0.0, meta["cy"] - pad)
    x2 = min(size, meta["cx"] + pad)
    y2 = min(size, meta["cy"] + pad)
    return [float(x1), float(y1), float(x2), float(y2)]


class NoseSynthGenerator:
    """Generate synthetic dog-nose images + labels for one or many dogs."""

    def __init__(self, image_size: int = 256, seed: int = 42):
        self.size = int(image_size)
        self.seed = int(seed)

    def render_image(self, dog_id: str, img_index: int) -> tuple[np.ndarray, dict]:
        """Render one image for ``dog_id``. Deterministic given dog_id + index."""
        dog_seed = (hash(dog_id) % (2**31)) ^ self.seed
        anat = _dog_anatomy(self.size, dog_seed)
        tmpl = _dog_texture_template(self.size, anat)   # stable per-dog template
        img_seed = dog_seed * 1009 + int(img_index) + 1
        img, meta = _render(self.size, anat, tmpl, img_seed)
        meta["bbox_xyxy"] = _bbox_from_meta(self.size, meta)
        meta["class_ids"] = list(range(len(CLASSES)))
        meta["classes"] = CLASSES
        return img, meta

    def render_labeled(self, dog_id: str, img_index: int) -> dict:
        """Image + masks + bbox + landmarks as a dict (for dataset writes)."""
        img, meta = self.render_image(dog_id, img_index)
        masks = _build_masks(self.size, meta)
        return {
            "image": img,
            "masks": masks,
            "bbox": meta["bbox_xyxy"],
            "left_nare": meta["left_nare"],
            "right_nare": meta["right_nare"],
            "philtrum": meta["philtrum"],
            "meta": meta,
        }


def _yolo_line(size: int, bbox: list[float], class_id: int = 0) -> str:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return f"{class_id} {(x1 + w/2)/size:.6f} {(y1 + h/2)/size:.6f} {w/size:.6f} {h/size:.6f}"


def generate_dataset(base_dir: str | Path, num_dogs: int = 20,
                     images_per_dog: int = 10, image_size: int = 256,
                     seed: int = 42, splits: dict | None = None) -> dict:
    """Write a synthetic dataset under ``base_dir`` in YOLO + COCO layout.

    Layout written:
        base_dir/
          images/{train,val,test}/<dog>/<idx>.jpg
          labels/{train,val,test}/<dog>/<idx>.txt        (YOLO detection)
          masks/{train,val,test}/<dog>/<idx>.png         (HxWxC class masks)
          coco/annotations_{split}.json                  (COCO detection + kpts)
          dataset.yaml                                   (YOLO dataset config)

    Returns a small manifest dict (counts per split).
    """
    splits = splits or {"train": 0.7, "val": 0.15, "test": 0.15}
    base = Path(base_dir)
    gen = NoseSynthGenerator(image_size=image_size, seed=seed)

    counts = {s: 0 for s in splits}
    for s in splits:
        (base / "images" / s).mkdir(parents=True, exist_ok=True)
        (base / "labels" / s).mkdir(parents=True, exist_ok=True)
        (base / "masks" / s).mkdir(parents=True, exist_ok=True)
    (base / "coco").mkdir(parents=True, exist_ok=True)

    rng = _rng(seed)
    split_names = list(splits.keys())
    split_cum = []
    acc = 0.0
    for s in split_names:
        acc += splits[s]
        split_cum.append(acc)

    coco_out = {s: {"images": [], "annotations": [], "categories": []} for s in splits}
    cat_id = 1
    coco_cats = [
        {"id": cat_id, "name": "NOSE01", "supercategory": "nose"},
    ]
    # keypoint definition: left_nare, right_nare, philtrum (all visible)
    kp_template = {"keypoints": ["left_nare", "right_nare", "philtrum"],
                   "skeleton": [[1, 2], [2, 3]]}

    img_global_id = 1
    ann_global_id = 1

    for d in range(num_dogs):
        dog_id = f"dog_{d+1:03d}"
        # assign this dog's images across splits deterministically
        for i in range(images_per_dog):
            u = rng.random()
            split = split_names[next(k for k, v in enumerate(split_cum) if u <= v)]
            sample = gen.render_labeled(dog_id, i)

            ip = base / "images" / split / dog_id
            lp = base / "labels" / split / dog_id
            mp = base / "masks" / split / dog_id
            ip.mkdir(parents=True, exist_ok=True)
            lp.mkdir(parents=True, exist_ok=True)
            mp.mkdir(parents=True, exist_ok=True)

            stem = f"{i:03d}"
            cv2.imwrite(str(ip / f"{stem}.jpg"), sample["image"],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lp / f"{stem}.txt").write_text(
                _yolo_line(image_size, sample["bbox"], class_id=0) + "\n")
            cv2.imwrite(str(mp / f"{stem}.png"), sample["masks"])

            # COCO record (detection bbox + keypoints)
            x1, y1, x2, y2 = sample["bbox"]
            w, h = x2 - x1, y2 - y1
            ln, rn, phil = sample["left_nare"], sample["right_nare"], sample["philtrum"]
            coco_out[split]["images"].append({
                "id": img_global_id, "file_name": f"{dog_id}/{stem}.jpg",
                "width": image_size, "height": image_size, "dog_id": dog_id,
            })
            coco_out[split]["annotations"].append({
                "id": ann_global_id, "image_id": img_global_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h], "area": float(w * h),
                "iscrowd": 0,
                "keypoints": [ln[0], ln[1], 2, rn[0], rn[1], 2, phil[0], phil[1], 2],
                "num_keypoints": 3,
                "dog_id": dog_id,
            })
            img_global_id += 1
            ann_global_id += 1
            counts[split] += 1

    for s, data in coco_out.items():
        data["categories"] = coco_cats
        (base / "coco" / f"annotations_{s}.json").write_text(json.dumps(data, indent=2))

    # YOLO dataset.yaml
    yolo_yaml = (
        f"path: {base.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: NOSE01\n"
    )
    (base / "dataset.yaml").write_text(yolo_yaml)
    return {"splits": counts, "num_dogs": num_dogs, "base_dir": str(base)}
