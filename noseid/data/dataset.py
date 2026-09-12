"""Dataset wrappers for metric learning + detection training.

NoseDataset        : returns (image_uint8_or_tensor, label_dict). Multi-task.
NoseTripletDataset : yields (anchor, positive, negative) for triplet/contrastive.
list_dogs          : helper listing dog identity folders under a split.

Works with or without torch. When torch is installed and ``return_tensor`` is
True, images come back as float32 CHW normalized tensors; otherwise as numpy
uint8 arrays. Augmentation is applied via the albumentations transforms from
``noseid.data.augment``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def list_dogs(split_dir: str | Path) -> list[str]:
    """Return sorted identity folder names that actually contain images."""
    p = Path(split_dir)
    out = []
    for d in sorted(p.iterdir()) if p.exists() else []:
        if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.jpeg"))
                           or any(d.glob("*.png"))):
            out.append(d.name)
    return out


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_label(image_path: Path, split: str, base: Path) -> dict | None:
    """Locate the matching YOLO label + mask + (COCO-derived) keypoints."""
    dog = image_path.parent.name
    stem = image_path.stem
    # Do not put absent optional values into the target dictionary.  PyTorch's
    # default collate cannot batch dictionaries containing None, which is
    # common for direct identity images without colocated annotations.
    label = {}
    # Support both generated datasets (labels/<split>/<dog>) and the direct
    # per-dog layout produced by roboflow_import.py (label beside the image).
    candidates = [image_path.parent / f"{stem}.txt",
                  base / "labels" / split / dog / f"{stem}.txt"]
    lp = next((p for p in candidates if p.exists()), candidates[-1])
    if lp.exists():
        parts = lp.read_text().strip().split()
        if len(parts) >= 5:
            cid, cx, cy, w, h = map(float, parts[:5])
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2] if img is not None else (256, 256)
            label["bbox"] = [
                (cx - w / 2) * width, (cy - h / 2) * height,
                (cx + w / 2) * width, (cy + h / 2) * height,
            ]
    mask_candidates = [image_path.parent / f"{stem}.png",
                       base / "masks" / split / dog / f"{stem}.png"]
    mp = next((p for p in mask_candidates if p.exists()), mask_candidates[-1])
    if mp.exists():
        label["masks"] = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    return label


class NoseDataset:
    """Flat dataset over all dog identity folders of a split.

    __getitem__ returns:
        image  : np.uint8 HxWx3 (RGB) OR torch.float32 CHW (if return_tensor)
        target : {'dog_id': str, 'label': int, 'path': str, **optional labels}
    """

    def __init__(self, base_dir: str | Path, split: str = "train",
                 transform: Callable | None = None,
                 return_tensor: bool = False,
                 id_to_label: dict[str, int] | None = None):
        self.base = Path(base_dir)
        self.split = split
        self.transform = transform
        self.return_tensor = return_tensor and _HAS_TORCH
        # Accept the synthetic layout, dataset/<split>/<dog>, and a direct
        # split directory such as dataset/train/<dog>.
        candidates = [self.base / "images" / split, self.base / split, self.base]
        self.img_dir = next((p for p in candidates if list_dogs(p)), candidates[0])
        self.dogs = list_dogs(self.img_dir)
        if id_to_label is None:
            id_to_label = {d: i for i, d in enumerate(self.dogs)}
        self.id_to_label = id_to_label

        self.samples: list[tuple[Path, str]] = []
        exts = ("*.jpg", "*.jpeg", "*.png")
        for dog in self.dogs:
            for e in exts:
                for p in sorted((self.img_dir / dog).glob(e)):
                    self.samples.append((p, dog))

    def __len__(self) -> int:
        return len(self.samples)

    def _to_tensor(self, arr: np.ndarray) -> Any:
        import torch  # local import keeps module importable w/o torch
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

    def __getitem__(self, idx: int) -> tuple[Any, dict]:
        path, dog = self.samples[idx]
        img = _load_image(path)
        label = _find_label(path, self.split, self.base) or {}
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        target = {
            "dog_id": dog,
            "label": int(self.id_to_label.get(dog, -1)),
            "path": str(path),
        }
        target.update(label)
        if self.return_tensor:
            return self._to_tensor(np.ascontiguousarray(img)), target
        return img, target


class NoseTripletDataset:
    """Yield (anchor, positive, negative) image triplets.

    Used by triplet/contrastive losses. Pairs a random image of one dog with
    another image of the same dog and an image of a different dog.
    """

    def __init__(self, base_dir: str | Path, split: str = "train",
                 transform: Callable | None = None,
                 return_tensor: bool = False,
                 n_triplets: int | None = None, seed: int = 42):
        self.ds = NoseDataset(base_dir, split, transform, return_tensor)
        self.rng = random.Random(seed)
        if len(self.ds.dogs) < 2:
            raise ValueError("need >= 2 dogs for triplet mining")
        # index images by dog
        self.by_dog: dict[str, list[int]] = {d: [] for d in self.ds.dogs}
        for i, (_, dog) in enumerate(self.ds.samples):
            self.by_dog[dog].append(i)
        self.by_dog = {d: v for d, v in self.by_dog.items() if len(v) >= 2}
        self.dogs_with_pairs = list(self.by_dog.keys())
        self.n_triplets = n_triplets or (len(self.ds.samples) * 4)

    def __len__(self) -> int:
        return self.n_triplets

    def _sample(self, idx: int) -> tuple[Any, dict]:
        return self.ds[idx]

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        a_dog = self.rng.choice(self.dogs_with_pairs)
        n_dog = self.rng.choice([d for d in self.dogs_with_pairs if d != a_dog])
        ai, pi = self.rng.sample(self.by_dog[a_dog], 2)
        ni = self.rng.choice(self.by_dog[n_dog])
        a = self._sample(ai)[0]
        p = self._sample(pi)[0]
        n = self._sample(ni)[0]
        return a, p, n
