"""Build a FAISS identity gallery from the current embedding checkpoint.

The gallery is built from the normalized detector crops in ``identity_crops``
so it uses the same input representation used during embedding training.
This is a development gallery: it is useful for exercising enrollment and
identification, but it must not be treated as a production recognition set
until session-separated known-dog evaluation is available.

Example:
    python scripts/build_identity_gallery.py --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_identity_embeddings import (  # noqa: E402
    _collect_samples,
    _embed_samples,
    _resolve_device,
)


def build_gallery(crops: str, weights: str, output: str, backbone: str,
                  embed_dim: int, device_value: str, batch_size: int,
                  source_data: str | None = None) -> dict:
    import torch
    from noseid.config import get_config
    from noseid.embedding.model import EmbeddingNet
    from noseid.matching.index import NoseIndex

    crop_root = Path(crops).resolve()
    checkpoint = Path(weights).resolve()
    destination = Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"embedding checkpoint not found: {checkpoint}")
    if not (crop_root / "train").is_dir():
        raise FileNotFoundError(f"training crop directory not found: {crop_root / 'train'}")

    device = _resolve_device(device_value, torch)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    id_to_label = state.get("id_to_label", {})
    net = EmbeddingNet(backbone, embed_dim, pretrained=False,
                       num_classes=len(id_to_label) or None)
    net.load_state_dict(state.get("model", state), strict=False)
    net.to(device).eval()

    samples = _collect_samples(crop_root, "train")
    if not samples:
        raise RuntimeError(f"no readable training crops found under {crop_root / 'train'}")
    embeddings, labels, paths = _embed_samples(
        samples, net, torch, device, batch_size)

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    source_root = Path(source_data).resolve() if source_data else None
    source_train = source_root / "train" if source_root else None
    if source_train is not None and source_train.is_dir():
        source_ids = sorted(p.name for p in source_train.iterdir() if p.is_dir())
    else:
        source_ids = sorted(
            p.name for p in (crop_root / "train").iterdir() if p.is_dir())
    enrolled_ids = sorted(counts)
    skipped_ids = [dog_id for dog_id in source_ids if dog_id not in counts]

    cfg = get_config()
    index = NoseIndex(dim=embed_dim, cfg=cfg)
    enrollment = []
    label_array = np.asarray(labels)
    for dog_id in enrolled_ids:
        group = embeddings[label_array == dog_id]
        result = index.enroll(dog_id, [row for row in group])
        enrollment.append(result.to_dict())
    destination.mkdir(parents=True, exist_ok=True)
    index.save(str(destination))

    manifest = {
        "gallery_type": "development_identity_gallery",
        "weights": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "backbone": backbone,
        "embedding_dim": embed_dim,
        "device": str(device),
        "source_crops": str(crop_root / "train"),
        "source_data": str(source_root) if source_root else None,
        "source_images_embedded": len(paths),
        "enrolled_identities": enrolled_ids,
        "enrolled_identity_count": len(enrolled_ids),
        "images_per_identity": counts,
        "skipped_identities_without_usable_crops": skipped_ids,
        "thresholds": {
            "verify_cosine": index.verify_thr,
            "reject_cosine": index.reject_thr,
        },
        "enrollment": enrollment,
        "development_only": True,
        "warning": (
            "This gallery is built from the current training crops. The two "
            "identities without usable detector crops were skipped, and the "
            "current split has no session-separated known-dog test set."
        ),
    }
    (destination / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="identity_crops")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--output", default="output/index")
    parser.add_argument("--source-data", default="identity_data")
    parser.add_argument("--backbone", default="efficientnet_v2")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    manifest = build_gallery(
        args.crops, args.weights, args.output, args.backbone, args.embed_dim,
        args.device, args.batch_size, args.source_data,
    )
    print(json.dumps({
        "gallery": args.output,
        "enrolled_identity_count": manifest["enrolled_identity_count"],
        "source_images_embedded": manifest["source_images_embedded"],
        "skipped_identities_without_usable_crops": (
            manifest["skipped_identities_without_usable_crops"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
