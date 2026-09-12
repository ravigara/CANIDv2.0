"""Data subpackage: synthetic generation, datasets, augmentation."""
from .synth import NoseSynthGenerator, generate_dataset  # noqa: F401
from .dataset import NoseDataset, NoseTripletDataset, list_dogs  # noqa: F401
from .augment import build_augmentation, train_aug, val_aug  # noqa: F401

__all__ = [
    "NoseSynthGenerator", "generate_dataset",
    "NoseDataset", "NoseTripletDataset", "list_dogs",
    "build_augmentation", "train_aug", "val_aug",
]
