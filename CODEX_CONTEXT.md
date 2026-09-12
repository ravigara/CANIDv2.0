# Codex Project Handoff

Last updated: 2026-09-06

This is the durable handoff for the dog-nose biometric identifier. Read this
file and `ANNOTATION_GUIDE.md` before changing the repository. Paths below are
relative to `E:\biometric_identifier` unless stated otherwise.

## Project objective

Build the system in these separate phases:

1. Detect a dog nose in an ordinary image or camera frame.
2. Segment the rhinarium and anatomical regions.
3. Predict stable anatomical keypoints.
4. Extract an identity embedding from a normalized nose crop.
5. Match the embedding against enrolled dogs and reject unknown dogs.
6. Add registry, consent, audit, enrollment, identification, and application
   services.

Detection, anatomy, keypoints, and identity are separate learning problems.
No detector or anatomy metric establishes biometric identity accuracy.

## Hardware and environment

- Windows workspace: `E:\biometric_identifier`
- GPU: NVIDIA RTX 5050 Laptop GPU
- PyTorch: CUDA is available (`torch 2.11.0+cu128` was verified)
- GPU training device: `cuda:0` / CLI value `0`
- Main dependencies verified: torch, torchvision, ultralytics, transformers,
  albumentations, FAISS, OpenCV

## Current verified state

### Detection

The current `roboflow_export` is a fresh one-class detection export, not the
old 22-class export:

```yaml
nc: 1
names: ['NOSE01']
```

Verified image/label counts:

```text
train: 829 images, 829 labels
valid: 220 images, 220 labels
test: 109 images, 109 labels
```

Every detection label uses class index `0`.

Artifacts:

```text
output/models/detection/weights/best.pt
output/models/detection/weights/last.pt
output/models/detection/results.csv
output/models/detection_best.pt
```

The selected detector was the best checkpoint, not necessarily the last
epoch. The results file contains 58 completed epochs. The strongest recorded
row was epoch 43:

```text
precision: 0.96703
recall:    0.90834
mAP50:     0.95103
mAP50-95:  0.58768
```

The detector is connected in `config/default.yaml`:

```yaml
detection:
  backend: torch
  class_name: NOSE01
  weights: output/models/detection_best.pt
  conf_threshold: 0.5
```

These metrics are a detector acceptance baseline. Continue checking false
positives, missed noses, bounding-box tightness, unseen breeds, poor lighting,
side angles, and target-device speed.

### Segmentation export and imported data

The current COCO segmentation export is in `segmentation_export`:

```text
train: 210 images, 840 annotations
valid: 58 images, 232 annotations
test: 29 images, 115 annotations
```

Active annotated classes are:

```text
rhinarium
left_nare
right_nare
philtrum
```

Roboflow also emits an unused `canid02` category with zero annotations. There
is no `fold` class and no explicit `background` class. This is intentional:
fold is optional, and background is derived as the inverse of the supplied
foreground masks.

One test image has no `left_nare` annotation because only one nostril is
visible. It should remain unlabeled rather than receiving a guessed mask.

The clean imported dataset is:

```text
anatomy_data/train: 210 images and 210 masks
anatomy_data/valid: 58 images and 58 masks
anatomy_data/test: 29 images and 29 masks
```

The earlier partial import was preserved at:

```text
anatomy_data_partial_backup_20260730
```

Do not mix that backup into training.

The latest re-import from the current `segmentation_export` was kept separate
at:

```text
anatomy_data_retrained_20260730/
```

It contains 210/58/29 images and 210/58/29 imported mask PNGs for
train/valid/test. This is the dataset used for the current retrained
checkpoint. The older `anatomy_data` remains preserved for comparison.

Important semantic check: the COCO structure can verify files and polygons but
cannot prove whether `left_nare` and `right_nare` were named from the dog's
anatomical perspective. The project contract is dog's perspective: in a
front-facing image, viewer-left is usually the dog's right nare. This must be
visually verified before using left/right anatomy for alignment. If an export
used viewer perspective, correct or consistently remap it and retrain before
claiming anatomical left/right correctness.

### Segmentation model

The trained checkpoint is:

```text
output/models/segmentation_retrained/segmentation_best/
  config.json
  model.safetensors
  preprocessor_config.json
```

It is a six-output SegFormer-B0 checkpoint compatible with the runtime class
contract. The fold channel is not supervised and is excluded from the
reported aggregate metric. The runtime config now points to it:

```yaml
segmentation:
  backend: torch
  num_classes: 6
  weights: output/models/segmentation_retrained/segmentation_best
```

Measured checkpoint performance, calculated with the saved model on the
imported masks:

```text
                  valid Dice     test Dice
rhinarium           0.9065        0.9188
left_nare           0.5812        0.4503
right_nare          0.5412        0.4586
philtrum            0.7752        0.7502
background          0.9393        0.9402
mean                0.7487        0.7036
```

The user accepted this checkpoint as the current working model and may
improve it later. The nares are the main improvement opportunity; these scores
are not a production biometric gate.

The retrained checkpoint was loaded successfully and tested through
`scripts/test_unseen.py` on `unseen/dimg.jpeg`. Detection produced a box with
confidence approximately 0.83 and the segmentation runtime returned
rhinarium, both nares, philtrum, and background masks. This is a smoke test,
not a replacement for held-out per-class evaluation.

### Unseen-image testing

`scripts/test_unseen.py` was added. It runs the detector on the original
image, crops the detected nose with padding, runs SegFormer on the crop, and
saves:

```text
output/results/unseen_model/<stem>_crop.jpg
output/results/unseen_model/<stem>_overlay.jpg
output/results/unseen_model/<stem>_report.json
```

The script default now uses the retrained checkpoint above. Run it with:

```powershell
python scripts\test_unseen.py --image unseen\your_image.jpg --device 0
```

If detection misses the nose, try `--conf 0.30`. This is qualitative testing;
it is not a substitute for a labeled held-out evaluation set.

### Runtime integration status and known gaps

The detector, segmentation, and landmark checkpoints are now connected to the
main pipeline through `config/default.yaml`:

```yaml
landmarks:
  backend: torch
  weights: output/models/landmarks_best.pt
  device: 0
  input_size: 256
  crop_pad: 0.15
  min_confidence: 0.50
```

`noseid/pipeline/landmark_model.py` is the reusable network/decoder module.
`noseid/pipeline/landmarks.py` loads the checkpoint, runs inference on the
padded detector crop, decodes heatmaps, maps points to original-image
coordinates, and exposes confidence plus validity. The OpenCV implementation
remains the fallback when auto mode has no checkpoint or model dependencies
are unavailable. Low-confidence learned landmarks are excluded from embedding
alignment so the embedder can fall back to a detector crop.

`scripts/test_unseen.py` now covers detector, padded-crop segmentation, and
landmarks, and writes landmark coordinates/confidence into its report. The
Torch and NumPy segmentation paths now both operate on the padded detector
crop while returning source-image-sized masks. A no-ground-truth full pipeline
smoke test passed on `unseen/dimg.jpeg`.

On 2026-08-03, representative unseen overlays were reviewed and the shared
detector-crop normalization path was hardened. Identity inference now uses a
canonical square crop aligned from reliable landmarks, with conservative
rhinarium-mask suppression only when the mask contains all retained landmark
points; otherwise it falls back to the aligned unmasked crop. The same
behavior is used by the learned and NumPy embedding paths. `test_unseen.py`
also writes a normalized crop and preprocessing metadata.

The new `scripts/evaluate_segmentation.py` evaluator measured the current
checkpoint on all 29 test images. Per-image mean Dice was 0.9128 for
`rhinarium`, 0.4479 for `left_nare`, 0.3563 for `right_nare`, 0.6908 for
`philtrum`, and 0.9386 for `background`; the evaluated aggregate was 0.6693.
This formally accepts the checkpoint for continued development, not for a
biometric production gate. The nares remain the main segmentation weakness.

Remaining work is quality improvement, not basic wiring: better
missing/occluded-point coverage and threshold tuning. Representative unseen
overlays and the confidence sweep have now been checked; the keypoint
confidence threshold remains 0.50 because 0.55 sharply reduces coverage. Do
not claim biometric readiness from this initial keypoint integration.

## Code changes completed

- Detection defaults and result labels use `NOSE01`.
- Detection training resolves paths from the repository root.
- Detection training uses the available `yolov8s.pt` default.
- `prepare_identity_crops.py` creates padded, split-preserving crops and a
  crop manifest.
- Dataset loading supports direct per-dog folders, JPEG files, colocated
  labels, real image dimensions, and multiple layouts.
- Embedding trainer fixes include loader naming, labels, validation IDs,
  configurable training values, and no-image errors.
- Local embedding weights load without unnecessary model downloads.
- CLI enrollment and identification accept embedding weights.
- `roboflow_import.py` writes mask channels in canonical anatomical order.
- `train_segmentation.py` reads `.classes.txt` instead of assuming Roboflow
  channel order, derives background, and ignores the optional fold in the
  validation aggregate.
- Segmentation CLI device `0` is normalized to `cuda:0`.
- Segmentation AMP is disabled by default for stability; `--amp` is optional.
- Segmentation supports `--resume` from the checkpoint in the selected output directory.
- Segmentation processor loading prefers local cached/checkpoint files.
- `test_unseen.py` provides detector-plus-segmentation-plus-landmark inference.
- `test_unseen.py` now also runs the trained landmark model and draws the
  three predicted points on the overlay.
- `noseid/pipeline/landmark_model.py` contains the reusable landmark network,
  checkpoint loader, and heatmap decoder.
- `noseid/pipeline/landmarks.py` supports Torch checkpoint inference on a
  padded detector crop, original-image coordinate mapping, confidence, and
  heuristic fallback.
- `noseid/pipeline/segmentation.py` applies the padded detector crop for both
  Torch and NumPy segmentation while preserving source-image mask dimensions.
- `noseid/pipeline/crop.py` provides shared canonical detector-crop
  normalization, landmark alignment, mask transformation, and fallbacks.
- `scripts/evaluate_landmarks.py` reports held-out pixel error and PCK.
- `scripts/evaluate_segmentation.py` reports held-out anatomy IoU/Dice.
- `scripts/prepare_identity_crops.py` prepares anatomy-aware normalized crops
  and records per-image detection, alignment, and mask metadata.
- FAISS L2 distance conversion correctly maps squared distance to cosine
  similarity for identity thresholds.
- `tests/test_pipeline.py` covers the shared landmark model output shape.
- `config/default.yaml` uses `output/models/segmentation_retrained/segmentation_best`.
- `test_unseen.py` uses the same retrained segmentation checkpoint by default.
- Keypoint COCO export validation confirmed 288/82/40 images for
  train/valid/test, with one `nose` category and keypoints in the required
  order `left_nare`, `right_nare`, `philtrum`.
- Keypoint import validation confirmed matching image/JSON pairs in
  `keypoint_data` and 1,230 visible keypoints.
- `scripts/train_landmarks.py` now applies geometric transforms to keypoint
  coordinates, swaps left/right targets during horizontal flips, produces
  heatmaps at one-quarter input resolution, and uses the current Torch AMP API.
- `scripts/build_identity_metadata.py` creates provisional per-image identity
  metadata with measured quality values.
- `scripts/test_identity_dataset.py` creates disposable detector, landmark,
  segmentation, and normalized-crop review artifacts.
- `scripts/evaluate_identity_embeddings.py` evaluates the trained identity
  checkpoint with explicit in-sample and open-set warnings.
- `scripts/build_identity_gallery.py` builds the current 16-identity FAISS
  gallery from usable training crops.
- `scripts/test_identity_gallery.py` runs a disposable gallery smoke test.
- `config/default.yaml` now points the embedding runtime at
  `output/models/embedding_best.pt`.
- Runtime identity inference now selects the highest-confidence detector box,
  reports quality failures as warnings by default, and retains
  `--strict-quality` for hard rejection. Images with no detected nose remain
  rejected rather than being embedded as full-frame photos.

## Verification completed

The following checks pass:

```text
python -m compileall -q noseid scripts tests
pytest -q
24 passed, 4 warnings
python scripts/train_segmentation.py --help
python scripts/test_unseen.py --help
segmentation device probe: 0 -> cuda:0
segmentation checkpoint resume-load probe: passed
keypoint export/import structure probe: passed
keypoint augmented dataset probe: passed
keypoint model output shape probe: passed (`[batch, 3, 64, 64]`)
CUDA keypoint training batch probe: passed
held-out keypoint evaluation: passed (40 images, 120 points)
unseen detector-plus-retrained-segmentation-landmark smoke test: passed
full pipeline without gt_hints crop probe: passed
held-out segmentation evaluation: passed (29 images; development baseline)
held-out landmark confidence sweep: passed (40 test, 82 valid images)
anatomy-aware normalized crop smoke tests: passed
```

The latest pytest warnings are one Albumentations version-check network
warning caused by the restricted environment, plus three FAISS/SWIG
dependency deprecations for `SwigPyPacked`, `SwigPyObject`, and
`swigvarlink`. They are not test or training failures.

## Current keypoint phase status

The separate Roboflow Keypoint Detection project has been created and
exported. Do not add keypoints to the segmentation class list. The project
uses one overall object category, `nose`, with exactly this keypoint order:

```text
1. left_nare
2. right_nare
3. philtrum
```

The current export is in `keypoint_export` and was structurally verified:

```text
train: 288 images, 288 keypoint annotations
valid: 82 images, 82 keypoint annotations
test: 40 images, 40 keypoint annotations
```

Every current annotation contains all three points with visibility `2`. This
is enough to train visible-landmark localization, but it does not yet provide
strong evidence for invisible/occluded-point rejection. Future data should
include carefully marked unavailable points without guessed coordinates.

The imported data is in `keypoint_data`. The importer also creates `masks/`
files when the COCO keypoint export includes an object segmentation field;
these are not used by `train_landmarks.py` and must not be confused with the
anatomical segmentation dataset.

The training checkpoint is:

```text
output/models/landmarks_best.pt
```

Its saved metadata reports epoch 99 and the required keypoint order. The
training script was fixed before training because the original augmentation
path changed images without changing keypoint coordinates, and the original
network output was 32x32 while targets were 64x64.

The exact import commands, for rebuilding the data from the export, are:

```powershell
python scripts\roboflow_import.py --source keypoint_export --out keypoint_data --split train
python scripts\roboflow_import.py --source keypoint_export --out keypoint_data --split valid
python scripts\roboflow_import.py --source keypoint_export --out keypoint_data --split test
```

Verify JSON files under:

```text
keypoint_data/keypoints/train/
keypoint_data/keypoints/valid/
keypoint_data/keypoints/test/
```

Train with:

```powershell
python scripts\train_landmarks.py `
  --data keypoint_data `
  --epochs 100 `
  --batch-size 16 `
  --image-size 256 `
  --device 0 `
  --output output\models
```

Initial held-out evaluation on `keypoint_data/test` produced this baseline
using the same square-resize preprocessing as training:

```text
                         mean px   PCK@5%   PCK@10%
left_nare                 17.54      0.500     0.900
right_nare                13.40      0.575     0.925
philtrum                  15.69      0.325     0.850
overall                   15.54      0.467     0.892
```

These are development metrics on 40 images, not a production gate. The
confidence sweep shows that the current 0.50 threshold retains all annotated
test/validation points, while 0.55 retains only about half of test points.
Add occluded/unavailable examples before relying on rejection behavior.

## Identity phase after keypoints

Real per-dog photographs have now been supplied under `identity_data` and the
identity workflow has been continued with the current detector-crop pipeline.
Do not use detection or anatomical class names as dog IDs.

The source identity folders are preserved at:

```text
identity_data/
  identitydataset/dog1/*.jpg ... dog26/*.jpg
```

The prepared working split is:

```text
identity_data/train: 187 images, dog_0001..dog_0018
identity_data/valid: 57 images, dog_0019..dog_0022
identity_data/test: 41 images, dog_0023..dog_0026
```

This is a source-burst-preserving split. Validation/test dogs are unseen
identities, not repeat sessions of the training dogs, so it supports open-set
smoke testing but not known-dog generalization accuracy. Metadata is in
`identity_data/identity_metadata.csv`; quality, camera, and session fields
remain provisional where the source data did not provide them.

Validate and review the current data with:

```powershell
python scripts\validate_identity_dataset.py --data identity_data
python scripts\test_identity_dataset.py --device 0
```

Create detector crops:

```powershell
python scripts\prepare_identity_crops.py `
  --weights output\models\detection_best.pt `
  --source identity_data `
  --out identity_crops `
  --conf 0.50 `
  --pad 0.15 `
  --device 0
```

Review `identity_crops/crop_manifest.json`, then train embeddings only after
the crops and identity split are trusted:

```powershell
python scripts\train_embedding.py `
  --train-dir identity_crops\train `
  --val-dir identity_crops\valid `
  --backbone efficientnet_v2 `
  --embed-dim 256 `
  --epochs 100 `
  --batch-size 32 `
  --device 0 `
  --output output\models
```

The current embedding checkpoint is:

```text
output/models/embedding_best.pt
```

It was trained from 149 usable training crops for 16 identities. Two training
dogs (`dog_0003`, `dog_0004`) had no usable detector crops and were not
enrolled. The checkpoint's best epoch was 71; the initial batch-size-32 run
hit GPU memory limits, so the completed run used batch size 8.

The current development FAISS gallery is:

```text
output/index/nose.index
output/index/meta.json
output/index/gallery_manifest.json
```

Build or rebuild it with:

```powershell
python scripts\build_identity_gallery.py --device 0
python scripts\test_identity_gallery.py --device 0
```

The smoke results are written to
`identity_test_review/gallery_smoke.json`. All 149 training crops currently
verify against the gallery. The unseen valid/test identities are intentionally
reported as open-set checks; their current false-accept rates are about 71.7%
and 56.8%, respectively. This is a development workflow result, not a
biometric readiness claim. A useful next dataset improvement is overlapping
known dogs across session-separated splits so rank-1/rank-5, FAR, FRR, EER,
repeatability, and latency can be measured honestly.

## Safety and preservation rules

- Preserve user data and existing changes.
- Treat old 22-class exports as legacy unless verified; the current
  `roboflow_export` has been verified as one-class.
- Do not delete `anatomy_data_partial_backup_20260730` until the new import is
  intentionally archived elsewhere.
- Do not delete base checkpoints, source datasets, model artifacts, or any
  future `output/index` gallery without explicit intent.
- Keep detection, segmentation, keypoints, and identity labels separate.
- Use held-out sessions and unknown dogs for biometric evaluation.
- A real deployment needs consent, access control, encryption, audit logs, and
  a secondary official identifier such as a microchip record.

## Useful files

- `README.md`: concise project overview and commands.
- `ANNOTATION_GUIDE.md`: detailed annotation and phase instructions.
- `config/default.yaml`: runtime paths and thresholds.
- `scripts/train_detection.py`: YOLO detector training.
- `scripts/train_segmentation.py`: SegFormer anatomy training.
- `scripts/test_unseen.py`: detector plus segmentation test on one image.
- `scripts/evaluate_segmentation.py`: held-out segmentation IoU/Dice evaluator.
- `scripts/validate_identity_dataset.py`: identity split/leakage preflight.
- `scripts/roboflow_import.py`: COCO/YOLO import and mask/keypoint conversion.
- `scripts/train_landmarks.py`: keypoint training.
- `scripts/evaluate_landmarks.py`: held-out keypoint pixel-error/PCK evaluation.
- `scripts/prepare_identity_crops.py`: detector crop preparation.
- `scripts/train_embedding.py`: identity embedding training.
- `scripts/evaluate_identity_embeddings.py`: checkpoint pair/open-set evaluation.
- `scripts/build_identity_gallery.py`: build the current FAISS identity gallery.
- `scripts/test_identity_gallery.py`: disposable gallery identification smoke test.
- `noseid/biometric.py`: pipeline construction and orchestration.
- `noseid/pipeline/detection.py`: detector runtime wrapper.
- `noseid/pipeline/segmentation.py`: anatomy runtime wrapper.
- `noseid/pipeline/crop.py`: shared canonical identity-crop preprocessor.
- `noseid/pipeline/landmarks.py`: trained Torch landmark runtime with heuristic
  fallback.
- `noseid/training/trainer.py`: identity metric-learning trainer.
- `tests/test_pipeline.py`: regression tests.
