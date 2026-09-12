"""Evaluate a trained embedding checkpoint on normalized identity crops.

This evaluator is intentionally conservative. It reports training-set pair
separation and open-set rejection for dogs absent from the training gallery.
It only reports known-dog identification accuracy when a split contains dog
IDs also present in the training split.

Example:
    python scripts/evaluate_identity_embeddings.py --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def _resolve_device(value: str, torch):
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return torch.device(f"cuda:{value}")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return torch.device(value)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def _collect_samples(root: Path, split: str) -> list[tuple[Path, str]]:
    split_dir = root / split
    samples = []
    if not split_dir.is_dir():
        return samples
    for dog_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(dog_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, dog_dir.name))
    return samples


def _load_batch(paths: list[Path], torch, device):
    batch = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable crop: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        batch.append(tensor.float() / 255.0)
    values = torch.stack(batch).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (values - mean) / std


def _embed_samples(samples, net, torch, device, batch_size: int):
    embeddings = []
    labels = []
    paths = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            group = samples[start:start + batch_size]
            tensor = _load_batch([p for p, _ in group], torch, device)
            output = net.embed(tensor).detach().cpu().numpy()
            embeddings.append(output)
            labels.extend(dog for _, dog in group)
            paths.extend(str(path) for path, _ in group)
    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), labels, paths
    return _normalize_rows(np.concatenate(embeddings)), labels, paths


def _percentiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(values.max()),
    }


def _gallery(train_embeddings: np.ndarray, train_labels: list[str]):
    labels = sorted(set(train_labels))
    templates = []
    for label in labels:
        group = train_embeddings[np.asarray(train_labels) == label]
        templates.append(_normalize_rows(group.mean(axis=0, keepdims=True))[0])
    return labels, _normalize_rows(np.stack(templates))


def _gallery_results(embeddings, labels, gallery_labels, templates,
                     verify_threshold: float, reject_threshold: float) -> dict:
    if len(embeddings) == 0:
        return {"samples": 0}
    scores = embeddings @ templates.T
    best_indices = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(scores)), best_indices]
    predicted = [gallery_labels[i] for i in best_indices]
    known = np.asarray([label in gallery_labels for label in labels])
    known_count = int(known.sum())
    known_correct = int(sum(
        bool(is_known) and predicted[i] == labels[i]
        for i, is_known in enumerate(known)
    ))
    unknown = ~known
    unknown_scores = best_scores[unknown]
    return {
        "samples": int(len(labels)),
        "known_identity_samples": known_count,
        "known_rank1_accuracy_percent": (
            float(100.0 * known_correct / known_count) if known_count else None
        ),
        "unknown_samples": int(unknown.sum()),
        "unknown_rejection_rate_percent": (
            float(100.0 * np.mean(unknown_scores < verify_threshold))
            if len(unknown_scores) else None
        ),
        "unknown_false_accept_rate_percent": (
            float(100.0 * np.mean(unknown_scores >= verify_threshold))
            if len(unknown_scores) else None
        ),
        "similarity_to_training_gallery": _percentiles(best_scores),
        "unknown_similarity_to_training_gallery": _percentiles(unknown_scores),
        "thresholds": {
            "verify_cosine": verify_threshold,
            "reject_cosine": reject_threshold,
        },
    }


def evaluate(data: str, weights: str, output: str, backbone: str,
             embed_dim: int, device_value: str, batch_size: int,
             verify_threshold: float, reject_threshold: float) -> dict:
    import torch
    from noseid.embedding.model import EmbeddingNet
    from noseid.training.metrics import compute_biometric_metrics, evaluate_pairs

    root = Path(data).resolve()
    checkpoint = Path(weights).resolve()
    device = _resolve_device(device_value, torch)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    id_to_label = state.get("id_to_label", {})
    net = EmbeddingNet(backbone, embed_dim, pretrained=False,
                       num_classes=len(id_to_label) or None)
    net.load_state_dict(state.get("model", state), strict=False)
    net.to(device).eval()

    split_results = {}
    embedded = {}
    for split in SPLITS:
        samples = _collect_samples(root, split)
        embs, labels, paths = _embed_samples(samples, net, torch, device,
                                              batch_size)
        embedded[split] = (embs, labels, paths)
        split_results[split] = {
            "samples": len(samples),
            "identities": sorted(set(labels)),
            "embeddings": int(len(embs)),
        }

    train_embeddings, train_labels, _ = embedded["train"]
    gallery_labels, templates = _gallery(train_embeddings, train_labels)
    genuine, impostor = evaluate_pairs(train_embeddings,
                                       np.asarray(train_labels),
                                       max_pairs=5000)
    train_metrics = compute_biometric_metrics(genuine, impostor)
    result = {
        "data": str(root),
        "weights": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "device": str(device),
        "gallery_identities": gallery_labels,
        "split_results": split_results,
        "training_pair_metrics": train_metrics,
        "warning": (
            "Training pair metrics are in-sample. Valid/test identities are "
            "reported as unknown when absent from the training gallery. "
            "Known-dog repeat-session accuracy requires overlapping dog IDs "
            "across session-separated splits."
        ),
        "open_set": {
            split: _gallery_results(
                embedded[split][0], embedded[split][1], gallery_labels,
                templates, verify_threshold, reject_threshold)
            for split in ("valid", "test")
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from noseid.config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="identity_crops")
    parser.add_argument("--weights", default="output/models/embedding_best.pt")
    parser.add_argument("--output", default="identity_test_review/identity_evaluation.json")
    parser.add_argument("--backbone", default="efficientnet_v2")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verify-threshold", type=float,
                        default=float(cfg["matching"]["verify_threshold"]))
    parser.add_argument("--reject-threshold", type=float,
                        default=float(cfg["matching"]["reject_threshold"]))
    args = parser.parse_args()
    result = evaluate(
        args.data, args.weights, args.output, args.backbone, args.embed_dim,
        args.device, args.batch_size, args.verify_threshold,
        args.reject_threshold,
    )
    print(json.dumps({
        "output": result["output"] if "output" in result else args.output,
        "checkpoint_epoch": result["checkpoint_epoch"],
        "gallery_identities": len(result["gallery_identities"]),
        "training_pair_metrics": result["training_pair_metrics"],
        "open_set": result["open_set"],
    }, indent=2))


if __name__ == "__main__":
    main()
