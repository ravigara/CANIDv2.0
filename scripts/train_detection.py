"""Train a YOLO dog-nose detector from a Roboflow YOLO export.

Reads the data.yaml that Roboflow provides and launches ultralytics training.
Exports the best weights as ONNX for Stage 3 deployment.

Usage:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install ultralytics
    python scripts/train_detection.py --data detection_export/data.yaml --epochs 100 --model yolov8s.pt
    python scripts/train_detection.py --data detection_export/data.yaml --model yolov8s.pt --export-onnx

The output weights go to output/models/detection_best.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(value: str | Path) -> Path:
    """Resolve relative training paths from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_detection(data_yaml: str, model: str = "yolov8s.pt",
                   epochs: int = 100, image_size: int = 640,
                   batch_size: int = 16, device: str = "0",
                   output_dir: str = "output/models",
                   export_onnx: bool = False, export_tflite: bool = False) -> dict:
    """Train and optionally export a YOLO detection model.

    Returns a dict with paths to saved artifacts.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed. Run:\n"
              "    pip install ultralytics\n"
              "Exiting.", file=sys.stderr)
        sys.exit(1)

    out = _project_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_path = _project_path(data_yaml)
    model_path = _project_path(model)
    model_source = str(model_path) if model_path.exists() else model

    print(f"[*] loading model: {model_source}")
    yolo = YOLO(model_source)
    print(f"[*] training on: {data_path}")
    print(f"    epochs={epochs}, imgsz={image_size}, batch={batch_size}, device={device}")
    results = yolo.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(out),
        name="detection",
        exist_ok=True,
        patience=15,
        save=True,
        save_period=10,
        pretrained=True,
        verbose=True,
    )

    best = out / "detection" / "weights" / "best.pt"
    last = out / "detection" / "weights" / "last.pt"

    artifacts = {
        "best_weights": str(best) if best.exists() else None,
        "last_weights": str(last) if last.exists() else None,
        "results_dir": str(out / "detection"),
    }

    if export_onnx and best.exists():
        print("[*] exporting to ONNX...")
        yolo_export = YOLO(str(best))
        yolo_export.export(format="onnx", imgsz=image_size, simplify=True)
        onnx_path = best.with_suffix(".onnx")
        artifacts["onnx"] = str(onnx_path) if onnx_path.exists() else None

    if export_tflite and best.exists():
        print("[*] exporting to TFLite...")
        yolo_export = YOLO(str(best))
        yolo_export.export(format="tflite", imgsz=image_size)
        tflite_path = best.with_suffix(".tflite")
        artifacts["tflite"] = str(tflite_path) if tflite_path.exists() else None

    return artifacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to Roboflow data.yaml")
    p.add_argument("--model", default="yolov8s.pt",
                   help="base model checkpoint, for example yolov8s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="0", help="cuda device id or 'cpu'")
    p.add_argument("--output", default="output/models")
    p.add_argument("--export-onnx", action="store_true")
    p.add_argument("--export-tflite", action="store_true")
    args = p.parse_args()
    result = train_detection(
        data_yaml=args.data, model=args.model, epochs=args.epochs,
        image_size=args.image_size, batch_size=args.batch_size,
        device=args.device, output_dir=args.output,
        export_onnx=args.export_onnx, export_tflite=args.export_tflite,
    )
    import json
    print("\n[*] training complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
