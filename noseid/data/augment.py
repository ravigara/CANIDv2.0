"""Stage 12 - Data augmentation (Albumentations).

Builds the train/val transforms from the spec:
rotation, brightness, contrast, Gaussian noise, blur, perspective, shifts,
zoom, color jitter. Validation only does resize + normalize-ready output
(returns uint8 so the dataset wrapper can tensorize).
"""
from __future__ import annotations

try:
    import albumentations as A
    _HAS_A = True
except Exception:  # pragma: no cover
    _HAS_A = False


def build_augmentation(image_size: int = 224, train: bool = True):
    """Return an albumentations Compose (or None if A not installed)."""
    if not _HAS_A:
        return None
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size),
                                scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.7),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                          hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=0, border_mode=0, p=0.3),
            A.Resize(height=image_size, width=image_size),
        ])
    return A.Compose([A.Resize(height=image_size, width=image_size)])




def train_aug(image_size: int = 224):
    return build_augmentation(image_size, train=True)


def val_aug(image_size: int = 224):
    return build_augmentation(image_size, train=False)
