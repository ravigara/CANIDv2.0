# Dog Nose Biometric Identifier

A research-stage dog registration and identification pipeline. It detects the
nose, creates a coarse embedding from the complete dog photograph, stores one
normalized whole-nose embedding, and stores separate embeddings for each
detected nose feature for final reranking.

> This project is not a production biometric system. Detection, segmentation,
> and landmark accuracy do not establish identity accuracy. Real deployment
> requires consent, privacy controls, security review, open-set evaluation,
> and a secondary official identifier.

## Active pipeline

The registration and identification path is intentionally limited to:

1. **Full-photo retrieval** creates a lightweight whole-frame appearance
   vector and retrieves a broad candidate union. This is only a coarse signal.
2. **Nose detection** finds the single `NOSE01` object.
3. **Parts identification** runs anatomical segmentation and the three
   landmarks: `left_nare`, `right_nare`, and `philtrum`.
4. **Embedding extraction** creates one whole-nose embedding plus separate
   L2-normalized embeddings for `rhinarium`, `left_nare`, `right_nare`, and
   `philtrum` with the existing embedding checkpoint reused as a shared
   encoder.
5. **Cascade matching** unions full-photo and whole-nose candidates, then
   reranks them with weighted anatomical-feature similarity. The default final
   weights are 0.15 full photo, 0.25 whole nose, and 0.60 anatomy; the final
   candidate must score strictly above 50%.
6. **Template storage** averages full-photo, whole-nose, and each feature
   independently into a dog template and stores the active gallery in
   `output/index/full_photo_cascade_meta.json`.

Dataset generation, training, quality analysis, and biometric evaluation are
separate development tools; they are not inserted into the active flow.

The runtime supports trained Torch models and NumPy/OpenCV fallbacks where
available. The CLI provides synthetic demonstration, enrollment,
identification, and evaluation commands.

## Project status

- Detection is trained and integrated with the runtime.
- Anatomy segmentation is trained and usable as a development baseline; the
  two nare classes remain the main quality limitation.
- The three-point landmark model is trained and integrated, with an OpenCV
  heuristic fallback.
- Identity embeddings are stored as versioned whole-nose plus per-feature
  centroid templates; missing regions are omitted rather than fabricated. The
  current gallery remains a development artifact.
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

Run the development-only synthetic demo:

```bash
python -m noseid.cli demo --dogs 12 --images 8
```

Register a dog from several images. The detector and anatomical models run on
every image; averaged whole-nose and per-feature templates are persisted:

```bash
python -m noseid.cli enroll --index output/index --dog DOG_001 \
  --dir my_images --name Rex --breed GSD
python -m noseid.cli identify --index output/index --image photo.jpg
```

Enrollment stores full-photo cascade templates and metadata in
`output/index/full_photo_cascade_meta.json`. Identification returns a
reference-style response:

```json
{
  "match": true,
  "matched": true,
  "message": "Match found",
  "confidence": 0.8735,
  "confidence_pct": "87.4%",
  "margin": 0.1421,
  "features_used": ["left_nare", "philtrum", "rhinarium", "right_nare"],
  "dog": {"dog_id": "DOG_001", "name": "Rex", "breed": "GSD"}
}
```

## Browser application

The local web app uses full-photo candidate retrieval, whole-nose retrieval,
and anatomy-feature reranking from the active gallery.
It supports multi-photo registration and single-photo identification. A
recognized result includes the registered dog name and profile photo; an
unrecognized result includes ranked possible matches with confidence scores.
The Registered dogs section lists every profile; clicking one opens all photos
stored for that registration. Registration captures the dog name, optional
identification chip ID, colour, breed, age, blood type, and owner name/contact
details. A profile can be deleted from its detail view after confirmation. The
same profile view also accepts one or more current photos: the photos are
appended to the local registry and the selected dog's full-photo, whole-nose,
and anatomy templates are rebuilt from the retained historical and new images.

Start it from the repository root:

```powershell
python run_frontend.py
```

Open <http://127.0.0.1:8000>. The first request loads the existing detector,
segmentation, landmark, and embedding checkpoints. Registration saves the
all submitted photos under the local registry and stores full-photo,
whole-nose, and per-feature templates in
`output/index/full_photo_cascade_meta.json`; it does not retrain anything.
Set `NOSEID_GALLERY_MODE=cascade` to use the previous whole-nose
cascade, or `NOSEID_GALLERY_MODE=feature` to use the earlier anatomy-only
gallery without deleting the new artifacts.
The service endpoints are `POST /api/register`, `POST /api/identify`,
`GET /api/dogs`, `GET /api/dogs/{dog_id}`, `DELETE /api/dogs/{dog_id}`,
`POST /api/dogs/{dog_id}/images`, `POST /api/dogs/bulk-delete`, and
`GET /api/health`. The Registered Dogs section supports selecting several
profiles and deleting them in one action. Updating images does not retrain a
model and does not modify the preserved `registration/` source folders.

Run the required checks:

```bash
python -m compileall -q noseid scripts tests
pytest -q
```

## Trained inference

Place externally managed checkpoints at the configured paths. The existing
upstream model artifacts are reused; switching to feature templates does not
require retraining them:

```text
output/models/detection_best.pt
output/models/segmentation_retrained/segmentation_best/
output/models/landmarks_best.pt
output/models/embedding_best.pt
```

Test one image with the detector, anatomy models, and feature embedder:

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
Validate the dataset and review anatomy-specific features before enrollment:

```bash
python scripts/validate_identity_dataset.py --data identity_data
python scripts/build_cascade_gallery.py \
  --identity-source identity_data \
  --registration-source registration \
  --weights output/models/embedding_best.pt \
  --output output/index --device 0
python scripts/build_full_photo_gallery.py \
  --identity-source identity_data \
  --registration-source registration \
  --weights output/models/embedding_best.pt \
  --output output/index --device 0
python scripts/test_identity_gallery.py \
  --source-data identity_data \
  --weights output/models/embedding_best.pt \
  --index output/index --device 0

# Register every dog folder in one model-loaded run:
python scripts/register_directory.py \
  --source registration --index output/index --min-valid 3
```

The previous whole-nose cascade remains available at
`output/index/cascade_meta.json`; the new full-photo gallery is reversible.
The previous anatomy-only gallery remains available at
`output/index/feature_meta.json`. Use `--feature-only` with the CLI, or set
`NOSEID_GALLERY_MODE=feature` for the web service, to compare or roll back to
that path:

```powershell
$env:NOSEID_GALLERY_MODE = "feature"
python run_frontend.py
```

The completed current run produced `output/models/embedding_best.pt`; it is
reused as a shared encoder and does not need to be retrained.

The gallery artifacts are under `output/index/`; the smoke JSON is under
`identity_test_review/`. Evaluate rank-1/rank-5 accuracy, genuine/impostor
distributions, FAR, FRR, EER, unknown rejection, and latency on locked
session-separated test data before calling the system biometric-ready.

During current development, anatomy availability and quality remain review
signals. Images with no detected nose are rejected, and `identify --debug`
includes the per-feature diagnostics for manual review.

## Repository layout

```text
noseid/                  Runtime package and model components
  api/                  FastAPI service for the browser app
  data/                  Synthetic data and dataset helpers
  embedding/             Shared embedding network, feature crops, and losses
  features/              Learned and classical feature extraction
  matching/              Legacy FAISS, feature-only, and cascade enrollment/search
  pipeline/              Validation, detection, landmarks, segmentation, crops
  training/              Training loop and biometric metrics
config/default.yaml      Runtime paths and thresholds
scripts/                 Import, training, evaluation, and preparation tools
frontend/                No-build registration and identification UI
run_frontend.py          Local Uvicorn entry point
tests/                   Regression tests
requirements.txt         Python dependencies
ANNOTATION_GUIDE.md      Data and annotation contract
```

Datasets, exports, images, generated reports, experiment runs, galleries, and
model binaries are excluded by `.gitignore`.

## License

No license has been selected yet. Add a license before distributing the
project or accepting external contributions.
