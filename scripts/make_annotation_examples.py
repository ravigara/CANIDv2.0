"""Generate visual reference images for each Roboflow annotation phase.

Creates example annotated images so you can see EXACTLY how each annotation
type should look when done in Roboflow. Saves PNGs to annotation_examples/.

Phases:
  01_raw             - the source image (no annotation)
  02_object_detection - one bounding box around the rhinarium
  03_instance_segmentation - colored polygons for each region
  04_keypoint_detection    - 3 dots at left_nare, right_nare, philtrum
  05_all_combined          - all three overlays on one image (for reference)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# matplotlib is used only for clean overlays + legend; safe to require here
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noseid.data import NoseSynthGenerator


# Roboflow-style color palette (matches typical Roboflow UI colors)
COLOR_RHINARIUM = (75, 175, 80)      # green
COLOR_LEFT_NARE = (75, 105, 255)     # red (BGR -> displays red)
COLOR_RIGHT_NARE = (42, 42, 255)     # lighter red
COLOR_PHILTRUM = (0, 165, 255)       # orange
COLOR_FOLD = (203, 92, 215)          # purple
COLOR_BG = (128, 128, 128)           # gray
COLOR_BOX = (0, 200, 255)            # yellow box


def _to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _ring_mask(cx, cy, r_in, r_out, size):
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((d >= r_in) & (d <= r_out)).astype(np.uint8)


def _draw_phase(out_dir: Path, name: str, image_bgr, title: str,
                draw_fn, legend=None):
    """Render one annotated image with a clean title + optional legend."""
    canvas = image_bgr.copy()
    canvas = draw_fn(canvas)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(_to_rgb(canvas))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    if legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=tuple(c / 255 for c in col[::-1]),  # BGR->RGB
                         edgecolor="black", label=name) for name, col in legend]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=True, fontsize=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def generate_examples(out_root: str = "annotation_examples") -> None:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    # one clear, well-lit example image per phase
    g = NoseSynthGenerator(320, seed=7)
    sample = g.render_labeled("dog_005", 2)
    img_bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
    s = sample["meta"]
    cx, cy, r = s["cx"], s["cy"], s["r"]
    ln, rn, phil = s["left_nare"], s["right_nare"], s["philtrum"]
    bbox = sample["bbox"]

    # 1. RAW -----------------------------------------------------------------
    _draw_phase(out, "01_raw", img_bgr, "Phase 1 — Source Image (no annotation)",
                lambda c: c)

    # 2. OBJECT DETECTION (one box) -----------------------------------------
    def draw_box(c):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(c, (x1, y1), (x2, y2), COLOR_BOX, 3)
        # label background
        (tw, th), _ = cv2.getTextSize("NOSE01 0.99", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(c, (x1, y1 - th - 8), (x1 + tw + 8, y1), COLOR_BOX, -1)
        cv2.putText(c, "NOSE01 0.99", (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return c
    _draw_phase(out, "02_object_detection", img_bgr,
                "Phase 2 — Object Detection (ONE box around rhinarium)",
                draw_box,
                 legend=[("NOSE01", COLOR_BOX)])

    # 3. INSTANCE SEGMENTATION (colored polygons) ---------------------------
    def draw_seg(c):
        overlay = c.copy()
        # rhinarium disk
        cv2.circle(overlay, (int(cx), int(cy)), int(r * 0.95), COLOR_RHINARIUM, -1)
        # fold ring
        cv2.circle(overlay, (int(cx), int(cy)), int(r * 0.9), COLOR_FOLD,
                   int(r * 0.18))
        # philtrum band (between nares down to philtrum point)
        midx = (ln[0] + rn[0]) / 2
        pts = np.array([[midx - r * 0.06, (ln[1] + rn[1]) / 2],
                        [midx + r * 0.06, (ln[1] + rn[1]) / 2],
                        [midx + r * 0.04, phil[1]],
                        [midx - r * 0.04, phil[1]]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], COLOR_PHILTRUM)
        # nares
        cv2.ellipse(overlay, (int(ln[0]), int(ln[1])),
                    (int(s["nare_r"]), int(s["nare_r"] * 0.8)), 0, 0, 360,
                    COLOR_LEFT_NARE, -1)
        cv2.ellipse(overlay, (int(rn[0]), int(rn[1])),
                    (int(s["nare_r"]), int(s["nare_r"] * 0.8)), 0, 0, 360,
                    COLOR_RIGHT_NARE, -1)
        cv2.addWeighted(overlay, 0.55, c, 0.45, 0, dst=c)
        # outline the rhinarium for clarity
        cv2.circle(c, (int(cx), int(cy)), int(r * 0.95), (255, 255, 255), 2)
        return c
    _draw_phase(out, "03_instance_segmentation", img_bgr,
                "Phase 3 — Instance Segmentation (polygon per region)",
                draw_seg,
                legend=[("rhinarium", COLOR_RHINARIUM),
                        ("left_nare", COLOR_LEFT_NARE),
                        ("right_nare", COLOR_RIGHT_NARE),
                        ("philtrum", COLOR_PHILTRUM),
                        ("fold", COLOR_FOLD),
                        ("background", COLOR_BG)])

    # 4. KEYPOINT DETECTION (3 dots) ----------------------------------------
    def draw_kpts(c):
        for (px, py), color, label in [
            (ln, COLOR_LEFT_NARE, "left_nare"),
            (rn, COLOR_RIGHT_NARE, "right_nare"),
            (phil, COLOR_PHILTRUM, "philtrum")]:
            cv2.circle(c, (int(px), int(py)), 8, color, -1)
            cv2.circle(c, (int(px), int(py)), 8, (255, 255, 255), 2)
            cv2.putText(c, label, (int(px) + 12, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
        # skeleton lines (COCO style)
        cv2.line(c, (int(ln[0]), int(ln[1])), (int(rn[0]), int(rn[1])),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(c, (int((ln[0] + rn[0]) / 2), int((ln[1] + rn[1]) / 2)),
                 (int(phil[0]), int(phil[1])), (255, 255, 255), 2, cv2.LINE_AA)
        return c
    _draw_phase(out, "04_keypoint_detection", img_bgr,
                "Phase 4 — Keypoint Detection (3 dots, ORDER MATTERS)",
                draw_kpts,
                legend=[("1. left_nare", COLOR_LEFT_NARE),
                        ("2. right_nare", COLOR_RIGHT_NARE),
                        ("3. philtrum", COLOR_PHILTRUM)])

    # 5. ALL COMBINED (reference) --------------------------------------------
    def draw_all(c):
        c = draw_seg(c)
        c = draw_kpts(c)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(c, (x1, y1), (x2, y2), COLOR_BOX, 3)
        return c
    _draw_phase(out, "05_all_combined", img_bgr,
                "All annotations combined (reference only — don't actually draw all)",
                draw_all,
                legend=[("box", COLOR_BOX),
                        ("rhinarium", COLOR_RHINARIUM),
                        ("left_nare", COLOR_LEFT_NARE),
                        ("right_nare", COLOR_RIGHT_NARE),
                        ("philtrum", COLOR_PHILTRUM),
                        ("fold", COLOR_FOLD)])

    # 6. ANNOTATION GUIDE CHEAT SHEET (text card) ---------------------------
    guide_text = [
        "ROBOFLOW ANNOTATION CHEAT SHEET",
        "",
        "PHASE 1 (MUST): Per-dog folders  ->  train/dog_001/*.jpg",
        "  No annotation needed. Just organize photos by dog.",
        "",
        "PHASE 2 (MUST): Object Detection  ->  ONE box per image",
        "  Class: NOSE01  |  Box: around the rhinarium (nose pad)",
        "  Export: YOLOv8  ->  scripts/train_detection.py",
        "",
        "PHASE 3 (optional): Instance Segmentation  ->  polygons",
        "  Classes: rhinarium, left_nare, right_nare,",
        "           philtrum, fold, background",
        "  Export: COCO Mask  ->  scripts/train_segmentation.py",
        "",
        "PHASE 4 (optional): Keypoint Detection  ->  3 dots",
        "  Order: 1.left_nare  2.right_nare  3.philtrum",
        "  Export: COCO Keypoints  ->  scripts/train_landmarks.py",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(guide_text), transform=ax.transAxes,
            fontsize=13, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#888"))
    fig.savefig(out / "00_cheat_sheet.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print(f"[*] generated {len(list(out.glob('*.png')))} reference images in {out}/")
    for p in sorted(out.glob("*.png")):
        print(f"    {p.name}")


if __name__ == "__main__":
    generate_examples("annotation_examples")
