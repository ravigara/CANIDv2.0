"""Run a disposable smoke test against a saved identity gallery.

This checks that the current embedding model and FAISS index agree on the
training crops, then measures how validation/test identities are handled as
unknown dogs. The latter is intentionally reported as an open-set smoke test,
not as known-dog accuracy, because those identities are absent from training.

Example:
    python scripts/test_identity_gallery.py --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_identity_embeddings import (  # noqa: E402
    SPLITS,
    _collect_samples,
    _embed_samples,
    _resolve_device,
)


def evaluate_gallery(crops: str, weights: str, index_dir: str, output: str,
                     backbone: str, embed_dim: int, device_value: str,
                     batch_size: int) -> dict:
    import torch
    from noseid.matching.index import NoseIndex
    from noseid.embedding.model import EmbeddingNet

    crop_root = Path(crops).resolve()
    checkpoint = Path(weights).resolve()
    gallery_path = Path(index_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"embedding checkpoint not found: {checkpoint}")
    if not (gallery_path / "meta.json").is_file():
        raise FileNotFoundError(f"gallery not found: {gallery_path}")

    device = _resolve_device(device_value, torch)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    id_to_label = state.get("id_to_label", {})
    net = EmbeddingNet(backbone, embed_dim, pretrained=False,
                       num_classes=len(id_to_label) or None)
    net.load_state_dict(state.get("model", state), strict=False)
    net.to(device).eval()
    index = NoseIndex.load(str(gallery_path))
    gallery_ids = sorted(index._templates)

    split_results = {}
    for split in SPLITS:
        samples = _collect_samples(crop_root, split)
        embeddings, labels, paths = _embed_samples(
            samples, net, torch, device, batch_size)
        known = split == "train"
        statuses = Counter()
        correct = 0
        rows = []
        for embedding, label, path in zip(embeddings, labels, paths):
            result = index.identify(embedding)
            statuses[result.status] += 1
            if known and result.status == "Verified" and result.dog_id == label:
                correct += 1
            rows.append({
                "image": path,
                "expected_dog_id": label,
                "returned_dog_id": result.dog_id,
                "similarity_percent": result.similarity,
                "status": result.status,
            })
        split_results[split] = {
            "samples": len(labels),
            "identities": sorted(set(labels)),
            "known_training_gallery": known,
            "status_counts": dict(sorted(statuses.items())),
            "rank1_verified_accuracy_percent": (
                float(100.0 * correct / len(labels)) if known and labels else None
            ),
            "unknown_false_accept_rate_percent": (
                float(100.0 * statuses["Verified"] / len(labels))
                if not known and labels else None
            ),
            "rows": rows,
        }

    result = {
        "gallery": str(gallery_path),
        "weights": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "device": str(device),
        "gallery_identities": gallery_ids,
        "thresholds": {
            "verify_cosine": index.verify_thr,
            "reject_cosine": index.reject_thr,
        },
        "split_results": split_results,
        "warning": (
            "Training results are in-sample. Valid/test identities are not "
            "in this gallery, so their result is an open-set rejection smoke "
            "test rather than known-dog identification accuracy."
        ),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="identity_crops")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--index", default="output/index")
    parser.add_argument("--output", default="identity_test_review/gallery_smoke.json")
    parser.add_argument("--backbone", default="efficientnet_v2")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = evaluate_gallery(
        args.crops, args.weights, args.index, args.output, args.backbone,
        args.embed_dim, args.device, args.batch_size,
    )
    print(json.dumps({
        "output": args.output,
        "gallery_identities": len(result["gallery_identities"]),
        "splits": {
            split: {
                key: value for key, value in values.items()
                if key != "rows"
            }
            for split, values in result["split_results"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
