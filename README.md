# Dog Nose Biometric Identifier

A research-stage computer-vision pipeline for detecting a dog's nose,
segmenting its anatomy, locating stable landmarks, and preparing normalized
crops for future dog-identity matching.

> This project is not a production biometric system. Detection, segmentation,
> and landmark accuracy do not establish identity accuracy. Real deployment
> requires consent, privacy controls, security review, open-set evaluation,
> and a secondary official identifier.

## Pipeline

The project keeps four learning problems separate:

1. **Detection** finds the nose pad using the single `NOSE01` class.
2. **Anatomical segmentation** predicts the rhinarium, nares, philtrum, and
   background.
3. **Landmarks** predicts `left_nare`, `right_nare`, and `philtrum` points for
   geometric alignment.
4. **Identity embedding** and FAISS matching use the current development
   checkpoint and gallery; session-separated evaluation is still required
   before any biometric claim.

The runtime supports trained Torch models and NumPy/OpenCV fallbacks where
available. The CLI provides synthetic demonstration, enrollment,
identification, and evaluation commands.

## Project status

- Detection is trained and integrated with the runtime.
- Anatomy segmentation is trained and usable as a development baseline; the
  two nare classes remain the main quality limitation.
- The three-point landmark model is trained and integrated, with an OpenCV
  heuristic fallback.
- Identity embedding training is complete for the current supplied images;
  the development gallery is enrolled and the identification smoke test runs.
- The current identity split has unseen dogs in validation/test rather than
  repeat sessions of training dogs, so generalization is not yet measured.

Datasets, photographs, generated reports, and model binaries remain local
artifacts. They are excluded from GitHub by `.gitignore`. Restore model files
under the paths in `config/default.yaml` when running trained inference.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch and torchvision build
before installing the remaining requirements. CPU-only environments can run
the synthetic fallback demo and tests.

## Quick start

Run the synthetic end-to-end demo without repository datasets:

```bash
python -m noseid.cli demo --dogs 12 --images 8
```

Run the required checks:

```bash
python -m compileall -q noseid scripts tests
pytest -q
```

## Trained inference

Place externally managed checkpoints at the configured paths:

```text
output/models/detection_best.pt
output/models/segmentation_retrained/segmentation_best/
output/models/landmarks_best.pt
output/models/embedding_best.pt
```

Test one image with the detector, landmark model, and anatomy segmenter:

```bash
python scripts/test_unseen.py --image path/to/image.jpg --device 0
```

The script writes crops, overlays, and a JSON report to
`output/results/unseen_model/`; generated output is ignored by Git.

## Training workflow

Keep source data outside the repository or restore it into the gitignored
paths. Do not commit photographs, annotations, identity labels, or weights.

### Detection

Prepare a YOLO export with exactly one class, `NOSE01`:

```bash
python scripts/train_detection.py \
  --data path/to/detection_export/data.yaml \
  --model path/to/yolov8s.pt \
  --epochs 100 --image-size 640 --batch-size 16 \
  --device 0 --output output/models
```

### Anatomy segmentation

Prepare a COCO instance-segmentation export with `rhinarium`, `left_nare`,
`right_nare`, and `philtrum`, then import its splits and train:

```bash
python scripts/roboflow_import.py --source path/to/segmentation_export \
  --out anatomy_data_retrained --split train
python scripts/roboflow_import.py --source path/to/segmentation_export \
  --out anatomy_data_retrained --split valid
python scripts/roboflow_import.py --source path/to/segmentation_export \
  --out anatomy_data_retrained --split test
python scripts/train_segmentation.py --data anatomy_data_retrained \
  --epochs 100 --batch-size 8 --image-size 512 --device 0 \
  --output output/models/segmentation_retrained
```

### Landmarks

Prepare a separate COCO Keypoints export with one `nose` object and the point
order `left_nare`, `right_nare`, `philtrum`:

```bash
python scripts/roboflow_import.py --source path/to/keypoint_export \
  --out keypoint_data --split train
python scripts/roboflow_import.py --source path/to/keypoint_export \
  --out keypoint_data --split valid
python scripts/roboflow_import.py --source path/to/keypoint_export \
  --out keypoint_data --split test
python scripts/train_landmarks.py --data keypoint_data --epochs 100 \
  --batch-size 16 --image-size 256 --device 0 --output output/models
python scripts/evaluate_landmarks.py --data keypoint_data --split test
```

Manually review held-out overlays and confidence behavior before using
landmarks for identity alignment.

### Identity embeddings

Use opaque dog IDs, session-separated train/validation/test splits, multiple
captures per dog, and dogs absent from training for unknown-dog testing.
Validate the dataset and review normalized crops before training:

```bash
python scripts/validate_identity_dataset.py --data identity_data
python scripts/prepare_identity_crops.py \
  --weights output/models/detection_best.pt \
  --source identity_data --out identity_crops --device 0
python scripts/train_embedding.py \
  --train-dir identity_crops/train --val-dir identity_crops/valid \
  --backbone efficientnet_v2 --embed-dim 256 \
  --epochs 100 --batch-size 32 --device 0 --output output/models
```

The completed current run produced `output/models/embedding_best.pt`. Build
the FAISS gallery and run its disposable smoke test with:

```bash
python scripts/build_identity_gallery.py --device 0
python scripts/test_identity_gallery.py --device 0
```

The gallery artifacts are under `output/index/`; the smoke JSON is under
`identity_test_review/`. Evaluate rank-1/rank-5 accuracy, genuine/impostor
distributions, FAR, FRR, EER, unknown rejection, and latency on locked
session-separated test data before calling the system biometric-ready.

During current development, image-quality checks are reported as warnings so
the identity model can still be tested on ordinary photographs. Images with
no detected nose are still rejected. Use `--strict-quality` with `identify` or
`enroll` to restore hard quality rejection.

## Repository layout

```text
noseid/                  Runtime package and model components
  data/                  Synthetic data and dataset helpers
  embedding/             Embedding networks and metric-learning losses
  features/              Learned and classical feature extraction
  matching/              FAISS enrollment and search
  pipeline/              Validation, detection, landmarks, segmentation, crops
  training/              Training loop and biometric metrics
config/default.yaml      Runtime paths and thresholds
scripts/                 Import, training, evaluation, and preparation tools
tests/                   Regression tests
requirements.txt         Python dependencies
ANNOTATION_GUIDE.md      Data and annotation contract
```

Datasets, exports, images, generated reports, experiment runs, galleries, and
model binaries are excluded by `.gitignore`.

## License

No license has been selected yet. Add a license before distributing the
project or accepting external contributions.
