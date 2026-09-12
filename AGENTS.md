# Repository Handoff

Before working in this repository, read CODEX_CONTEXT.md and
ANNOTATION_GUIDE.md. CODEX_CONTEXT.md records the current project state,
decisions, completed work, verification results, and next steps for the dog
nose biometric identifier.

Preserve user data and existing changes. Treat any checked-in 22-class
detection export as legacy until the fresh one-class NOSE01 Roboflow export has
been verified. Keep detection, anatomical segmentation, keypoints, and dog
identity labels as separate tasks.

The current completed model artifacts are:

- Detection: `output/models/detection_best.pt`
- Segmentation: `output/models/segmentation_retrained/segmentation_best`
- Keypoints: `output/models/landmarks_best.pt`

The keypoint model is trained and integrated into
`noseid/pipeline/landmarks.py`; the OpenCV heuristic remains the fallback. The
next authorized workflow step is manual held-out overlay acceptance and
confidence/crop-pipeline hardening. Do not begin identity embedding training
until the keypoint adapter and complete detector-crop anatomy pipeline have
been validated across representative images.

When changing code, run:

```powershell
python -m compileall -q noseid scripts tests
pytest -q
```
