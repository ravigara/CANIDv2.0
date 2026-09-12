"""Reusable heatmap model and decoding helpers for anatomical keypoints."""
from __future__ import annotations

from typing import Iterable

import numpy as np

KEYPOINT_NAMES = ["left_nare", "right_nare", "philtrum"]
NUM_KPTS = len(KEYPOINT_NAMES)


def build_landmark_model(num_keypoints: int = NUM_KPTS,
                         pretrained: bool = False):
    """Build the same network used by ``scripts/train_landmarks.py``.

    ``pretrained`` is only useful during training. Runtime loading uses
    ``False`` so loading a local checkpoint never needs a network download.
    """
    import torch
    import torch.nn as nn

    try:
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        feat = backbone.features
        c_out = 1280
    except Exception:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feat = backbone.features
        c_out = 576

    class LandmarkNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = feat
            self.head = nn.Sequential(
                nn.Conv2d(c_out, 64, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, num_keypoints, 1))

        def forward(self, x):
            import torch.nn.functional as F
            heatmaps = self.head(self.feat(x))
            target_size = (x.shape[-2] // 4, x.shape[-1] // 4)
            return F.interpolate(heatmaps, size=target_size,
                                 mode="bilinear", align_corners=False)

    return LandmarkNet()


def load_landmark_model(weights: str, device: str = "cpu"):
    """Load a locally trained landmark checkpoint."""
    import torch

    model = build_landmark_model(pretrained=False)
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def decode_heatmaps(heatmaps, input_size: tuple[int, int] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps into input-image XY points and peak confidences.

    The model is trained with Gaussian heatmaps whose target peak is one.
    Sigmoid peak values provide a conservative confidence proxy; this is not a
    calibrated probability and must be tuned on held-out annotations.
    """
    import torch

    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    probs = torch.sigmoid(heatmaps.detach().float())
    k, hm_h, hm_w = probs.shape
    flat = probs.reshape(k, -1)
    values, indices = flat.max(dim=1)
    ys = torch.div(indices, hm_w, rounding_mode="floor").float()
    xs = (indices % hm_w).float()

    # Weighted local centroid gives a little subpixel stability without
    # changing the model's trained heatmap contract.
    points = []
    for i in range(k):
        x, y = int(xs[i]), int(ys[i])
        x0, x1 = max(0, x - 1), min(hm_w, x + 2)
        y0, y1 = max(0, y - 1), min(hm_h, y + 2)
        patch = probs[i, y0:y1, x0:x1]
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=patch.device),
            torch.arange(x0, x1, device=patch.device), indexing="ij")
        mass = patch.sum().clamp_min(1e-6)
        points.append(((patch * xx).sum() / mass,
                       (patch * yy).sum() / mass))
    points = torch.stack([torch.stack(p) for p in points])

    if input_size is not None:
        in_h, in_w = input_size
        points[:, 0] *= float(in_w) / float(hm_w)
        points[:, 1] *= float(in_h) / float(hm_h)
    return points.cpu().numpy().astype(np.float32), values.cpu().numpy().astype(np.float32)
