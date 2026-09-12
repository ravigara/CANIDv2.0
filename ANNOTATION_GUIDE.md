# Dog Nose Biometric Identifier: Annotation and Build Guide

This guide is the implementation plan for the complete system. The project is
not only a detection project. It has four model responsibilities:

1. **Object detection** finds the dog nose in an ordinary photograph.
2. **Instance segmentation** isolates the anatomical regions of the nose.
3. **Keypoint detection** finds stable landmarks for geometric alignment.
4. **Identity embedding** learns which nose belongs to which registered dog.

The first three models prepare the image. The fourth model performs biometric
identification. A registry database and application are built only after the
models pass open-set evaluation.

## Current verified project state — 2026-08-03

Detection is complete and frozen. `roboflow_export/data.yaml` contains exactly
one class, `NOSE01`, and all label files use class index `0`. The verified
split counts are 829 train, 220 valid, and 109 test images. The selected
checkpoint is `output/models/detection_best.pt`; the original training files
remain under `output/models/detection/weights/`.

Anatomical segmentation is also trained. The current COCO export contains
210 train, 58 valid, and 29 test images with these annotated classes:

```text
rhinarium, left_nare, right_nare, philtrum
```

`fold` was intentionally omitted. Background is derived automatically. The
clean imported data is in `anatomy_data`; the earlier incomplete import is
preserved at `anatomy_data_partial_backup_20260730` and must not be mixed into
training.

The latest export was imported separately into
`anatomy_data_retrained_20260730`. It contains 210/58/29 images and matching
masks for train/valid/test. The previous `anatomy_data` and the partial backup
remain preserved. The COCO export includes an unused `canid02` category with
zero annotations; the active classes are `rhinarium`, `left_nare`,
`right_nare`, and `philtrum`.

The segmentation file structure and polygons were verified. The remaining
semantic check is left/right naming: the project uses the dog's anatomical
perspective, not the viewer's perspective. If any segmentation export used
viewer-based left/right names, correct or remap that data before relying on
left/right anatomical alignment.

The trained SegFormer checkpoint is:

```text
output/models/segmentation_retrained/segmentation_best/
  config.json
  model.safetensors
  preprocessor_config.json
```

Its current measured Dice scores are:

```text
                  valid Dice     test Dice
rhinarium           0.9065        0.9188
left_nare           0.5812        0.4503
right_nare          0.5412        0.4586
philtrum            0.7752        0.7502
background          0.9393        0.9402
mean                0.7487        0.7036
```

The user is satisfied with this working baseline and may improve the nares
later. The runtime configuration points to both trained detector and
segmentation checkpoints. `scripts/test_unseen.py` tests one image by running
detection on the original image, applying landmarks and segmentation to the
padded detector crop, and writing an overlay/report.

The separate keypoint project is now annotated, imported, trained, and wired
into the Torch landmark runtime. Representative unseen overlays, confidence
handling, and the shared detector-crop normalization path have been checked.
Identity training must still wait for trusted per-dog data and
occlusion-aware landmark validation.

The current keypoint export contains 288 train, 82 valid, and 40 test images.
It uses one object category, `nose`, with exactly these keypoints in order:

```text
left_nare, right_nare, philtrum
```

All current keypoints are visible (`visibility=2`). The trained checkpoint is
`output/models/landmarks_best.pt`, saved at epoch 99. It is now integrated into
the Torch landmark runtime. The OpenCV heuristic remains an explicit fallback
when a checkpoint or Torch dependencies are unavailable.

---

## 1. Current Project Status

Already present:

- Synthetic nose generator with detection, mask, and keypoint labels.
- Roboflow YOLO export under `roboflow_export/`.
- Training entry points:
  - `scripts/train_detection.py`
  - `scripts/train_segmentation.py`
  - `scripts/train_landmarks.py`
  - `scripts/train_embedding.py`
- OpenCV fallback pipeline, FAISS index, CLI, and unit tests.
- Real dog photographs with train/valid/test splits in the Roboflow export.

Important corrections to the old workflow:

- The detector must have **one class: `NOSE01`**. Any old 22-class export is
  legacy and must not be used. The current `roboflow_export` was verified as
  the fresh one-class dataset.
- Dog identity is not a detection class. Identity labels belong to the
  embedding dataset and registry.
- Segmentation and keypoints should run on the detector crop, not on the
  entire original photograph.
- Bounding boxes alone cannot identify a dog. Identity training requires
  several images grouped by dog ID.
- The embedding dataset loader now accepts both generated datasets and direct
  per-dog folders. The embedding trainer also accepts its CLI training values.
- The trained landmark checkpoint has a development inference adapter in
  `noseid/pipeline/landmarks.py`. It runs on the padded detector crop and
  falls back to the heuristic implementation when the learned model is not
  available. It still requires manual acceptance and threshold tuning before
  it can be treated as production-quality.

Run the baseline tests after every code change:

```bash
pytest tests/ -v
python -m compileall -q noseid scripts tests
```

---

## 2. Lock the Data and Label Contract

Do not mix the three annotation projects.

### 2.1 Detection labels

Use exactly one class:

```text
0: NOSE01
```

Draw one tight bounding box around the complete rhinarium/nose pad. Include
the full visible nose pad, both nares when visible, and a small amount of
boundary context. Do not label eyes, muzzle, head, or dog identity.

Expected Roboflow export:

```text
detection_export/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

Check `data.yaml` before training:

```yaml
nc: 1
names: [NOSE01]
```

If a future export contains 22 classes, create a new one-class Roboflow
project or remap and validate the labels before training.

### 2.2 Segmentation labels

Create a separate Roboflow **Instance Segmentation** project using close-up
nose images. Use these classes and spelling exactly:

```text
rhinarium
left_nare
right_nare
philtrum
fold
background
```

`fold` is optional. If it cannot be annotated consistently, omit it from the
Roboflow project; the importer derives background from the supplied foreground
masks and the trainer ignores the unannotated fold channel.

Recommended annotation rules:

- `rhinarium`: the complete nose pad.
- `left_nare`: the left nostril from the dog's perspective.
- `right_nare`: the right nostril from the dog's perspective.
- `philtrum`: the central groove below/between the nares.
- `fold`: visible outer folds or wrinkle ring around the nose pad.
- `background`: only if the exporter requires an explicit background mask;
  otherwise it is the inverse of `rhinarium`.
- Keep masks pixel-accurate and do not include surrounding fur in `rhinarium`.
- Use the same class order in every export.

Export as **COCO Instance Segmentation/COCO Mask**. Keep the COCO JSON files;
they are needed by `scripts/roboflow_import.py`.

### 2.3 Keypoint labels

Create a separate Roboflow **Keypoint Detection** project. Place exactly three
keypoints in this order:

```text
1. left_nare
2. right_nare
3. philtrum
```

Place each point at the anatomical center. Use front-facing and moderate-angle
images, but exclude images where a point is genuinely invisible rather than
guessing its location.

Export as **COCO Keypoints**.

### 2.4 Identity labels

Identity is not annotated in the detection, segmentation, or keypoint class
list. Maintain a separate folder structure:

```text
identity_data/
  train/
    dog_0001/
      image_001.jpg
      image_002.jpg
    dog_0002/
  valid/
    dog_0001/
    dog_0002/
  test/
    dog_0001/
    dog_0002/
```

Use real dog IDs, not arbitrary groups of sorted filenames. Every dog should
have images from multiple sessions, phones, lighting conditions, and small
pose variations.

---

## 3. Image Collection Rules

For every dog, collect at least 10-20 images initially. A stronger dataset
uses 20-50 images over multiple days.

Capture conditions:

- Nose fills a useful part of the frame.
- Focus is sharp and both nares are visible where possible.
- Use diffuse lighting; avoid strong reflections and deep shadows.
- Include small yaw, pitch, distance, and scale changes.
- Include different phones/cameras if the application will use phones.
- Include clean, slightly wet, and naturally varied noses.
- Do not put images from different dogs in one identity folder.
- Keep a record of dog ID, capture session, camera, and conditions.

Do not randomly split near-identical frames into train and test. Split by
capture session so the test measures generalization rather than memorization.

Include negative/quality cases:

- Blurry images
- Side profiles
- Occluded noses
- Multiple dogs
- Very dark or overexposed images
- Dirty, injured, or partially visible noses
- Images containing no dog nose

These are needed for rejection behavior, not for identity enrollment.

---

## 4. Phase 1 - Train Object Detection

### 4.1 Prepare the export

Create the one-class detection project in Roboflow, annotate the boxes, then
export into `detection_export/`.

Do not use the current `dataset/data.yaml` until it has been replaced or
remapped to one class. It currently describes 22 classes.

Verify image/label pairing:

```bash
python -c "from pathlib import Path; p=Path('detection_export'); print(len(list((p/'train/images').glob('*'))), len(list((p/'train/labels').glob('*.txt'))))"
```

### 4.2 Train with the existing detection trainer

Use `scripts/train_detection.py`:

```bash
python scripts/train_detection.py \
  --data detection_export/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --image-size 640 \
  --batch-size 16 \
  --device 0 \
  --output output/models \
  --export-onnx
```

`yolov8s.pt` is only a pretrained COCO starting point. The output trained
weights are the detector, not a dog identity model.

Expected artifacts:

```text
output/models/detection/weights/best.pt
output/models/detection/weights/last.pt
output/models/detection/weights/best.onnx   # when export succeeds
```

### 4.3 Detection acceptance gate

Do not continue because training completed. Check:

- Validation mAP50 and mAP50-95
- Precision and recall
- False detections on ordinary dog photographs
- Misses on side angles and different breeds
- Bounding-box tightness around the nose
- Inference speed on the target phone/device

Save the selected checkpoint as:

```text
output/models/detection_best.pt
```

Set its path in `config/default.yaml`:

```yaml
detection:
  backend: torch
  weights: output/models/detection_best.pt
  conf_threshold: 0.50
```

`build_pipeline()` now reads this configuration. Run the unit tests and a
manual prediction before starting anatomy training.

---

## 5. Phase 2 - Import and Train Segmentation

### 5.1 Export and import the COCO masks

Export the segmentation project into `segmentation_export/`. Import each split
with the existing script:

```bash
python scripts/roboflow_import.py \
  --source segmentation_export \
  --out anatomy_data_retrained_20260730 \
  --split train

python scripts/roboflow_import.py \
  --source segmentation_export \
  --out anatomy_data_retrained_20260730 \
  --split valid

python scripts/roboflow_import.py \
  --source segmentation_export \
  --out anatomy_data_retrained_20260730 \
  --split test
```

Inspect that masks exist under:

```text
anatomy_data_retrained_20260730/masks/train/
anatomy_data_retrained_20260730/masks/valid/
anatomy_data_retrained_20260730/masks/test/
```

Check that the mask class names and channel order match the six classes in
`noseid/data/synth.py` and `scripts/train_segmentation.py`.

### 5.2 Train with the existing segmentation trainer

```bash
python scripts/train_segmentation.py \
  --data anatomy_data_retrained_20260730 \
  --epochs 100 \
  --batch-size 8 \
  --image-size 512 \
  --device 0 \
  --output output/models/segmentation_retrained
```

Expected output:

```text
output/models/segmentation_retrained/segmentation_best/
  config.json
  preprocessor_config.json
  model.safetensors or pytorch_model.bin
```

Set the local checkpoint path:

```yaml
segmentation:
  backend: torch
  weights: output/models/segmentation_retrained/segmentation_best
  num_classes: 6
```

### 5.3 Segmentation acceptance gate

Evaluate mean and per-class Dice/IoU. Pay special attention to `rhinarium`,
the two nares, and `philtrum`; a good background score alone is insufficient.
Inspect overlays on unseen dogs and unseen capture sessions.

For qualitative testing, use the repository inference script:

```powershell
python scripts\test_unseen.py --image unseen\your_image.jpg --device 0
```

It saves the detector crop, segmentation overlay, and JSON report under
`output/results/unseen_model/`. The current baseline is accepted for continued
development, but the nares are the main future improvement target.

---

## 6. Phase 3 - Keypoint Training Status and Integration

### 6.1 Export and import COCO keypoints

The keypoint project was exported into `keypoint_export/` as COCO Keypoints.
The verified split counts are 288 train, 82 valid, and 40 test images. Every
image has one `nose` annotation and the keypoint order is:

```text
1. left_nare
2. right_nare
3. philtrum
```

The data was imported into `keypoint_data` with:

```bash
python scripts/roboflow_import.py \
  --source keypoint_export \
  --out keypoint_data \
  --split train

python scripts/roboflow_import.py \
  --source keypoint_export \
  --out keypoint_data \
  --split valid

python scripts/roboflow_import.py \
  --source keypoint_export \
  --out keypoint_data \
  --split test
```

Verify files exist under:

```text
keypoint_data/keypoints/train/
keypoint_data/keypoints/valid/
keypoint_data/keypoints/test/
```

### 6.2 Train with the corrected keypoint trainer

```bash
python scripts/train_landmarks.py \
  --data keypoint_data \
  --epochs 100 \
  --batch-size 16 \
  --image-size 256 \
  --device 0 \
  --output output/models
```

Expected output:

```text
output/models/landmarks_best.pt
```

### 6.3 Required integration work

The training script now correctly transforms keypoints during rotation,
resizing, and horizontal flipping. Horizontal flipping also swaps the
left/right nare targets. The network explicitly outputs 1/4-resolution
heatmaps, matching the 64x64 targets generated from 256x256 images.

The checkpoint has been verified with a CUDA training batch, held-out
evaluation, and the existing regression tests. The runtime now loads the
checkpoint from configuration, runs on the padded detector crop, decodes the
three heatmaps, maps points back to original-image coordinates, and exposes
confidence plus validity. Low-confidence points are excluded from embedding
alignment so the embedder can fall back to a detector crop.

Before treating the model as production-quality:

1. Inspect overlays on representative held-out and unseen images.
2. Add occluded/unavailable examples and test rejection behavior.
3. Tune the confidence threshold on a validation split.
4. Expand tests comparing predicted points to ground truth.
5. Test the detector-crop → landmarks → segmentation chain across poses and
   lighting without ground-truth hints.

Initial held-out results on 40 test images were:

```text
                         mean px   PCK@5%   PCK@10%
left_nare                 17.54      0.500     0.900
right_nare                13.40      0.575     0.925
philtrum                  15.69      0.325     0.850
overall                   15.54      0.467     0.892
```

These are development baselines, not biometric or production acceptance
metrics. Do not claim keypoint accuracy from training loss alone.

---

## 7. Phase 4 - Build the Identity Dataset

Detection, segmentation, and keypoint labels do not provide the identity
learning target. Prepare `identity_data/` with real dog IDs as described in
Section 2.4.

Recommended minimum for an initial model:

```text
50+ dogs
10+ images per dog
multiple sessions per dog
```

For registry use, plan for hundreds or thousands of dogs and multiple images
per enrollment.

Use `scripts/roboflow_import.py` only if it is given a trustworthy CSV mapping:

```text
image_name,dog_id
photo_001.jpg,dog_0001
photo_002.jpg,dog_0001
photo_003.jpg,dog_0002
```

Do not use `--images-per-dog` as an identity mapping for production data. It
groups sorted files by position and can assign the wrong dog ID.

Example import for a trusted detection export:

```bash
python scripts/roboflow_import.py \
  --source detection_export \
  --out identity_data \
  --split train \
  --mapping-csv identity_mapping.csv
```

Review every identity folder manually before training.

---

## 8. Phase 5 - Train the Identity Embedding Model

Use `scripts/train_embedding.py`. This is the training step that makes the
system identify individuals rather than merely locate noses.

The current embedding trainer reads the files in each identity folder directly;
it does not run detection, segmentation, or keypoint alignment internally.
Therefore, do not give it full dog-face photographs for the final biometric
model. First create detector crops:

```bash
python scripts/prepare_identity_crops.py \
  --weights output/models/detection_best.pt \
  --source identity_data \
  --out identity_crops \
  --conf 0.50 \
  --pad 0.15 \
  --device 0
```

Review `identity_crops/crop_manifest.json` and manually inspect random crops.
The current crop generator uses the trained
segmentation and keypoint models to remove background and canonically align the
nose when their confidence and mask checks pass, and records safe fallbacks in
`crop_manifest.json`. Detector-only crops remain available only through the
explicit `--detector-only` baseline option.

```bash
python scripts/train_embedding.py \
  --train-dir identity_crops/train \
  --val-dir identity_crops/valid \
  --backbone efficientnet_v2 \
  --embed-dim 256 \
  --epochs 100 \
  --batch-size 32 \
  --device 0 \
  --output output/models
```

Expected output:

```text
output/models/embedding_best.pt
```

The trainer uses ArcFace as the main loss and optional Triplet/Contrastive
losses from `noseid/embedding/losses.py`.

The identity model should consume the aligned nose crop produced by detection,
segmentation, and keypoints. Do not train it on full dog photographs if the
runtime will identify from nose patterns.

### 8.1 Biometric evaluation

Evaluate on dogs and sessions not used for training. Measure:

- Rank-1 and Rank-5 identification accuracy
- Genuine similarity distribution
- Impostor similarity distribution
- FAR: false accepts
- FRR: false rejects
- EER: equal error rate
- Unknown-dog rejection rate
- Inference latency

The model must support open-set behavior. A dog absent from the registry must
not be forced into the closest registered identity.

Do not use synthetic demo accuracy as evidence of real biometric performance.

---

## 9. Phase 6 - Build Enrollment and Matching

Use `noseid/matching/index.py` for the first gallery implementation.

Enrollment flow:

1. Capture several images of one dog.
2. Validate quality.
3. Detect, segment, align, and embed each image.
4. Reject unusable images.
5. Average or store multiple normalized embeddings.
6. Check for an existing close match to prevent duplicate registration.
7. Store the dog metadata and embedding template.

Identification flow:

1. Capture several frames.
2. Keep only high-quality frames.
3. Average their embeddings.
4. Search FAISS.
5. Return details only above the verified threshold.
6. Return `Unknown Dog` or `Ambiguous` otherwise.

Tune thresholds on a held-out validation set. Do not keep the default values
without measuring FAR and FRR on real data.

---

## 10. Phase 7 - Registry API and Application

The repository currently has no implemented `noseid/api` service. Add a backend
with endpoints similar to:

```text
POST /enroll/start
POST /enroll/{request_id}/images
POST /identify
GET  /dogs/{registry_id}
PATCH /dogs/{registry_id}
```

Store registry data separately from model artifacts. Protect embeddings and
owner information with authentication, authorization, encryption, and audit
logs.

The mobile/realtime experience should show explicit states:

```text
Searching
Hold still
Poor quality - try again
Registered dog found
Possible match - verify
Dog not found
Enrollment required
```

An unmatched street-dog scan should create a provisional enrollment request,
not silently create a permanent registry identity. Require several captures
and the required registry/microchip/owner information.

---

## 11. Final Production Gates

Before public use, validate the complete system on a locked test set:

- New dogs never seen during training
- New sessions from known dogs
- Different phones and cameras
- Different breeds, ages, colors, and nose shapes
- Dirt, moisture, glare, blur, and partial occlusion
- Multiple dogs in one frame
- Near-duplicate and duplicate enrollment attempts
- Unknown-dog rejection
- Offline and poor-network behavior
- Mobile inference latency and battery usage

Use microchip or another official identifier as a secondary verification method
until the nose system has strong independently measured reliability. Nose
biometrics should initially be treated as a complementary identifier rather
than the sole legal proof of ownership or registration.

---

## 12. Correct Build Order Summary

```text
1. Re-export detection as one-class NOSE01
2. Train and validate detection
3. Annotate/export/import segmentation masks
4. Train and validate segmentation
5. Annotate/export/import three keypoints — complete
6. Train keypoint model — complete
7. Initial keypoint evaluation and runtime integration — complete
8. Occlusion-aware keypoint validation and trusted identity-data preparation — next
9. Prepare trusted per-dog identity folders
10. Create and review detector crops
11. Train the ArcFace/metric embedding model
12. Evaluate open-set biometric performance
13. Build FAISS enrollment and search
14. Build registry API and database
15. Build realtime camera application
16. Run production, security, and field validation
```

The immediate next action is to add trusted per-dog identity data and
occluded/unavailable keypoint examples. Do not begin identity embedding
training until keypoint predictions, coordinate mapping, confidence handling,
and the complete crop-based anatomy preprocessor have been accepted across
representative images.
