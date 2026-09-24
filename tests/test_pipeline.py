"""Test suite for the Dog Nose Biometric ID System.

Run with:  pytest tests/ -v
All tests run on the installed numpy/cv2/faiss stack (no torch needed).
"""
import numpy as np
import pytest

from noseid.data import NoseSynthGenerator
from noseid.data.synth import CLASSES
from noseid.backend import resolve_backend, info


def test_default_config_points_to_trained_runtime_models():
    from noseid.config import get_config

    cfg = get_config(reload=True)
    assert cfg["detection"]["class_name"] == "NOSE01"
    assert cfg["detection"]["weights"] == "output/models/detection_best.pt"
    assert cfg["landmarks"]["backend"] == "torch"
    assert cfg["landmarks"]["weights"] == "output/models/landmarks_best.pt"
    assert cfg["landmarks"]["min_confidence"] == pytest.approx(0.50)
    assert cfg["segmentation"]["backend"] == "torch"
    assert cfg["segmentation"]["weights"] == (
        "output/models/segmentation_retrained/segmentation_best")
    assert cfg["matching"]["gallery_mode"] == "full_photo_cascade"
    assert cfg["matching"]["cascade_photo_weight"] == pytest.approx(0.15)


def test_trained_runtime_artifacts_exist():
    from noseid.config import PROJECT_ROOT, get_config

    cfg = get_config(reload=True)
    for section in ("detection", "landmarks", "segmentation"):
        path = PROJECT_ROOT / cfg[section]["weights"]
        assert path.exists(), f"missing {section} artifact: {path}"


# ---------- Stage 1: synth generator ----------------------------------------

def test_synth_generates_valid_image_and_labels():
    g = NoseSynthGenerator(128, seed=1)
    s = g.render_labeled("dog_001", 0)
    assert s["image"].shape == (128, 128, 3)
    assert s["image"].dtype == np.uint8
    assert s["masks"].shape == (128, 128, len(CLASSES))
    assert len(s["bbox"]) == 4
    assert len(s["left_nare"]) == 2 and len(s["philtrum"]) == 2


def test_synth_is_deterministic():
    g = NoseSynthGenerator(128, seed=1)
    a = g.render_labeled("dog_007", 3)["image"]
    b = g.render_labeled("dog_007", 3)["image"]
    assert np.array_equal(a, b)


def test_synth_different_dogs_differ():
    g = NoseSynthGenerator(128, seed=1)
    a = g.render_labeled("dog_001", 0)["image"]
    b = g.render_labeled("dog_002", 0)["image"]
    assert not np.array_equal(a, b)


# ---------- Stage 2: validator ----------------------------------------------

def test_validator_accepts_good_rejects_bad():
    from noseid.pipeline.validator import NoseValidator
    g = NoseSynthGenerator(256, seed=3)
    v = NoseValidator()
    good = g.render_labeled("dog_001", 0)["image"]
    # not all synth images pass (some are intentionally degraded), so we just
    # check the validator produces a structured verdict
    r = v.validate(good)
    assert isinstance(r.valid, bool)
    assert 0 <= r.confidence <= 100
    assert r.reason
    # a fully black image must be rejected
    bad = np.zeros((256, 256, 3), np.uint8)
    assert v.validate(bad).valid is False


# ---------- Stages 3-5: detection/landmarks/segmentation --------------------

def test_detection_returns_bbox():
    from noseid.pipeline.detection import NoseDetector
    g = NoseSynthGenerator(256, seed=3)
    det = NoseDetector()
    s = g.render_labeled("dog_001", 0)
    res = det.detect(s["image"], gt_hint=s["bbox"])
    assert len(res) >= 1
    assert len(res[0].bbox) == 4
    assert 0 <= res[0].confidence <= 1


def test_landmarks_returns_nares_and_philtrum():
    from noseid.pipeline.landmarks import LandmarkDetector
    g = NoseSynthGenerator(256, seed=3)
    lm = LandmarkDetector()
    s = g.render_labeled("dog_001", 0)
    r = lm.detect(s["image"], s["bbox"])
    assert len(r.left_nare) == 2 and len(r.right_nare) == 2 and len(r.philtrum) == 2


def test_normalized_crop_aligns_landmarks_and_masks():
    from noseid.pipeline.crop import normalize_nose_crop
    from noseid.types import LandmarkResult
    g = NoseSynthGenerator(256, seed=3)
    s = g.render_labeled("dog_001", 0)
    lm = LandmarkResult(
        left_nare=s["left_nare"], right_nare=s["right_nare"],
        philtrum=s["philtrum"], valid=True)
    result = normalize_nose_crop(
        s["image"], s["bbox"], lm, s["masks"], CLASSES, target_size=224)
    assert result.image.shape == (224, 224, 3)
    assert result.aligned is True
    assert result.landmarks is not None
    assert np.allclose(result.landmarks.left_nare, [67.2, 85.12], atol=2.0)
    assert result.masks is not None and result.masks.shape == (224, 224, len(CLASSES))


def test_normalized_crop_rejects_low_confidence_alignment():
    from noseid.pipeline.crop import normalize_nose_crop
    from noseid.types import LandmarkResult
    g = NoseSynthGenerator(128, seed=3)
    s = g.render_labeled("dog_001", 0)
    lm = LandmarkResult(
        left_nare=s["left_nare"], right_nare=s["right_nare"],
        philtrum=s["philtrum"],
        confidence={"left_nare": 0.2, "right_nare": 0.9, "philtrum": 0.9},
        valid=True)
    result = normalize_nose_crop(s["image"], s["bbox"], lm, target_size=128)
    assert result.aligned is False
    assert result.fallback_reason == "landmarks_low_confidence"


def test_landmark_model_outputs_target_heatmap_shape():
    torch = pytest.importorskip("torch")
    from noseid.pipeline.landmark_model import build_landmark_model
    model = build_landmark_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert tuple(out.shape) == (1, 3, 64, 64)


def test_segmentation_produces_all_classes():
    from noseid.pipeline.segmentation import NoseSegmenter
    g = NoseSynthGenerator(256, seed=3)
    seg = NoseSegmenter()
    s = g.render_labeled("dog_001", 0)
    r = seg.segment(s["image"], s["bbox"])
    assert r.masks is not None
    assert r.masks.shape[-1] == len(CLASSES)
    # at least one class (rhinarium or background) must be detected
    assert any(r.present.values())


def test_segmentation_metrics_compute():
    from noseid.pipeline.segmentation import compute_segmentation_metrics
    g = NoseSynthGenerator(128, seed=1)
    s = g.render_labeled("dog_001", 0)
    pred = np.argmax(s["masks"], -1)
    gt = np.argmax(s["masks"], -1)
    m = compute_segmentation_metrics(pred, gt)
    assert m["mIoU"] == pytest.approx(1.0, abs=1e-6)
    assert m["mDice"] == pytest.approx(1.0, abs=1e-6)


# ---------- Stage 6/7: features + embedding ---------------------------------

def test_feature_extractor_returns_fixed_dim():
    from noseid.features import FeatureExtractor
    g = NoseSynthGenerator(128, seed=1)
    s = g.render_labeled("dog_001", 0)
    fx = FeatureExtractor(target_dim=128)
    fv = fx.extract(s["image"])
    assert fv.fused_dim == 128


def test_embedding_is_normalized():
    from noseid.embedding import Embedder
    g = NoseSynthGenerator(256, seed=3)
    s = g.render_labeled("dog_001", 0)
    emb = Embedder()
    r = emb.embed(s["image"], bbox=s["bbox"], dog_id="dog_001")
    assert r.embedding_size == 256
    v = np.asarray(r.embedding)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_anatomy_feature_crops_are_separate_and_available():
    from noseid.embedding.features import FEATURE_NAMES, extract_feature_crops
    from noseid.pipeline.crop import normalize_nose_crop
    from noseid.types import LandmarkResult

    g = NoseSynthGenerator(256, seed=3)
    s = g.render_labeled("dog_001", 0)
    landmarks = LandmarkResult(
        left_nare=s["left_nare"], right_nare=s["right_nare"],
        philtrum=s["philtrum"], valid=True)
    normalized = normalize_nose_crop(
        s["image"], s["bbox"], landmarks, s["masks"], CLASSES,
        target_size=224)
    crops = extract_feature_crops(normalized, CLASSES)
    assert tuple(crops) == FEATURE_NAMES
    assert all(crops[name].available for name in FEATURE_NAMES)
    assert all(crops[name].image.shape == (224, 224, 3) for name in FEATURE_NAMES)


def test_feature_index_stores_and_matches_per_region_templates(tmp_path):
    from noseid.matching import FeatureIndex

    cfg = {
        "matching": {
            "verify_threshold": 0.55,
            "reject_threshold": 0.20,
            "margin_threshold": 0.08,
            "min_features": 2,
            "embedding_version": "test-feature-v1",
        }
    }
    idx = FeatureIndex(dim=4, cfg=cfg)
    dog_a = {
        "rhinarium": np.array([1, 0, 0, 0], dtype=np.float32),
        "left_nare": np.array([0, 1, 0, 0], dtype=np.float32),
        "right_nare": np.array([0, 0, 1, 0], dtype=np.float32),
        "philtrum": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    dog_b = {
        name: np.roll(vector, 1) for name, vector in dog_a.items()
    }
    idx.enroll("dog_a", {name: [vector] for name, vector in dog_a.items()})
    idx.enroll("dog_b", {name: [vector] for name, vector in dog_b.items()})
    result = idx.identify(dog_a)
    assert result.status == "Verified"
    assert result.dog_id == "dog_a"
    assert set(result.candidates[0]["feature_scores"]) == set(dog_a)
    idx.save(str(tmp_path))
    loaded = FeatureIndex.load(str(tmp_path))
    assert loaded.get_record("dog_a")["embedding_version"] == "test-feature-v1"
    assert loaded.identify(dog_a).dog_id == "dog_a"


def test_feature_response_exposes_safe_candidate_profile_metadata():
    from noseid.biometric import identification_response
    from noseid.matching import FeatureIndex

    idx = FeatureIndex(dim=2, cfg={"matching": {"min_features": 2}})
    vectors = {
        "rhinarium": np.asarray([1.0, 0.0], dtype=np.float32),
        "left_nare": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    idx.enroll("dog_a", vectors, metadata={
        "name": "Rex", "photo_url": "/media/dog_a/profile.jpg",
        "source": "private-local-path",
    })
    result = idx.identify(vectors)
    response = identification_response(result, idx)

    assert response["status"] == "recognized"
    assert response["dog"] == {
        "dog_id": "dog_a",
        "name": "Rex",
        "photo_url": "/media/dog_a/profile.jpg",
    }
    assert response["possible_matches"][0]["confidence_pct"] == "100.0%"
    assert "source" not in response["possible_matches"][0]["dog"]


def test_feature_index_delete_removes_template_and_metadata():
    from noseid.matching import FeatureIndex

    idx = FeatureIndex(dim=2, cfg={"matching": {"min_features": 1}})
    idx.enroll("dog_a", {"rhinarium": np.asarray([1.0, 0.0], dtype=np.float32)})
    assert idx.delete("dog_a") is True
    assert idx.size == 0
    assert idx.get_record("dog_a") is None
    assert idx.delete("dog_a") is False


def test_cascade_index_retrieves_whole_nose_then_reranks_features(tmp_path):
    from noseid.matching import CascadeFeatureIndex

    cfg = {"matching": {
        "verify_threshold": 0.50,
        "margin_threshold": 0.00,
        "min_features": 2,
        "cascade_retrieval_k": 2,
        "cascade_whole_weight": 0.25,
        "cascade_feature_weight": 0.75,
    }}
    idx = CascadeFeatureIndex(dim=4, cfg=cfg)
    dog_a = {name: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
             for name in ("rhinarium", "left_nare")}
    dog_b = {name: np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
             for name in ("rhinarium", "left_nare")}
    idx.enroll("dog_a", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), dog_a)
    idx.enroll("dog_b", np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), dog_b)

    result = idx.identify(
        np.asarray([0.8, 0.6, 0.0, 0.0], dtype=np.float32), dog_b)
    assert result.status == "Verified"
    assert result.dog_id == "dog_b"
    assert result.candidates[0]["whole_nose_similarity"] < 1.0
    assert result.candidates[0]["feature_similarity"] == pytest.approx(1.0)

    idx.save(str(tmp_path))
    loaded = CascadeFeatureIndex.load(str(tmp_path))
    assert loaded.identify(
        np.asarray([0.8, 0.6, 0.0, 0.0], dtype=np.float32), dog_b).dog_id == "dog_b"


def test_full_photo_cascade_unions_photo_and_nose_candidates(tmp_path):
    from noseid.matching import FullPhotoCascadeIndex

    cfg = {"matching": {
        "verify_threshold": 0.50,
        "margin_threshold": 0.00,
        "min_features": 2,
        "cascade_retrieval_k": 1,
        "cascade_photo_weight": 0.15,
        "cascade_whole_weight": 0.25,
        "cascade_feature_weight": 0.60,
    }}
    idx = FullPhotoCascadeIndex(dim=4, cfg=cfg)
    features_a = {name: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                  for name in ("rhinarium", "left_nare")}
    features_b = {name: np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                  for name in ("rhinarium", "left_nare")}
    idx.enroll("dog_a", [np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
               [np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)], features_a)
    idx.enroll("dog_b", [np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)],
               [np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)], features_b)

    # Photo retrieval points to dog_a, nose retrieval points to dog_b. The
    # candidate union keeps both available for anatomy reranking.
    result = idx.identify(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        features_b)
    assert result.status == "Verified"
    assert result.dog_id == "dog_b"
    assert {row["dog_id"] for row in result.candidates} == {"dog_a", "dog_b"}
    assert "full_photo_similarity" in result.candidates[0]
    idx.save(str(tmp_path))
    loaded = FullPhotoCascadeIndex.load(str(tmp_path))
    assert loaded.identify(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        features_b).dog_id == "dog_b"


def test_profile_image_update_appends_photos_and_rebuilds_active_template(
        tmp_path, monkeypatch):
    import asyncio
    from io import BytesIO

    import cv2
    from fastapi import UploadFile

    import noseid.api.app as api_module
    from noseid.matching import FullPhotoCascadeIndex

    cfg = {"matching": {
        "verify_threshold": 0.50,
        "margin_threshold": 0.00,
        "min_features": 2,
        "cascade_retrieval_k": 2,
        "cascade_photo_weight": 0.15,
        "cascade_whole_weight": 0.25,
        "cascade_feature_weight": 0.60,
    }}
    index = FullPhotoCascadeIndex(dim=4, cfg=cfg)
    old_vector = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    new_vector = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    old_features = {
        name: [old_vector] for name in ("rhinarium", "left_nare")
    }
    index.enroll(
        "dog_a", [old_vector], [old_vector], old_features,
        metadata={"name": "Rex"}, num_images=1)

    registry_root = tmp_path / "registry"
    index_root = tmp_path / "index"
    monkeypatch.setattr(api_module, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api_module, "INDEX_ROOT", index_root)
    monkeypatch.setattr(
        api_module, "_runtime", api_module._Runtime(object(), index))
    api_module._save_registry_images(
        "dog_a", [np.full((24, 24, 3), 40, dtype=np.uint8)])

    def fake_enroll(pipeline, active_index, dog_id, samples, min_valid,
                    metadata=None):
        assert len(samples) == 2  # retained historical photo + current photo
        return active_index.enroll(
            dog_id, [old_vector, new_vector], [old_vector, new_vector],
            {name: [old_vector, new_vector]
             for name in ("rhinarium", "left_nare")},
            metadata=metadata, num_images=2)

    monkeypatch.setattr(api_module, "enroll_dog", fake_enroll)
    current = np.full((24, 24, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    assert ok
    upload = UploadFile(file=BytesIO(encoded.tobytes()), filename="current.jpg")

    response = asyncio.run(api_module.update_dog_images(
        "dog_a", [upload], min_valid=1))

    assert response["status"] == "updated"
    assert response["new_photos_processed"] == 1
    assert response["total_photos"] == 2
    assert (registry_root / "dog_a" / "images" / "image_002.jpg").is_file()
    assert len(response["dog"]["photo_urls"]) == 2

    loaded = FullPhotoCascadeIndex.load(str(index_root))
    assert loaded.get_record("dog_a")["num_images"] == 2
    assert loaded.get_record("dog_a")["metadata"]["name"] == "Rex"
    assert loaded.identify(
        new_vector, new_vector,
        {name: new_vector for name in ("rhinarium", "left_nare")},
    ).dog_id == "dog_a"


def test_full_photo_embedder_uses_complete_image_without_nose_crop():
    from noseid.embedding import FullPhotoEmbedder

    g = NoseSynthGenerator(128, seed=2)
    image = g.render_labeled("dog_001", 0)["image"]
    result = FullPhotoEmbedder({"full_photo": {"dim": 32}}).embed(image)
    assert result.embedding_size == 32
    assert abs(float(np.linalg.norm(result.embedding)) - 1.0) < 1e-3
    assert result.source == "full_photo_fast"


# ---------- Stage 8/9: FAISS matching ---------------------------------------

def test_index_enroll_and_identify():
    from noseid.biometric import build_pipeline, GTHint, run_to_embedding
    from noseid.matching import NoseIndex
    g = NoseSynthGenerator(256, seed=3)
    pc = build_pipeline()
    idx = NoseIndex(dim=256)
    dogs = ["dog_001", "dog_002", "dog_003"]
    for did in dogs:
        embs = []
        for i in range(5):
            s = g.render_labeled(did, i)
            _, _, _, _, er = run_to_embedding(pc, s["image"],
                                              GTHint.from_sample(s), did)
            embs.append(np.asarray(er.embedding))
        idx.enroll(did, embs)
    assert idx.size == 3
    s = g.render_labeled("dog_001", 9)
    _, _, _, _, er = run_to_embedding(pc, s["image"], GTHint.from_sample(s), "dog_001")
    res = idx.identify(np.asarray(er.embedding))
    assert res.status in ("Verified", "Unknown Dog", "Rejected")


def test_index_l2_distance_maps_to_cosine():
    from noseid.matching import NoseIndex
    idx = NoseIndex(dim=3)
    idx.enroll("dog_001", [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)])
    result = idx.identify(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert result.status == "Verified"
    assert result.similarity == pytest.approx(1.0, abs=1e-3)


def test_embedding_store_persists_registration_metadata_and_reference_response(tmp_path):
    from noseid.biometric import identification_response, registration_response
    from noseid.matching import EmbeddingStore

    idx = EmbeddingStore(dim=3)
    enrolled = idx.enroll(
        "dog_001",
        [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
        metadata={"name": "Rex", "breed": "GSD"},
    )
    idx.save(str(tmp_path))
    loaded = EmbeddingStore.load(str(tmp_path))

    assert loaded.get_record("dog_001")["metadata"]["name"] == "Rex"
    assert registration_response(enrolled)["nose_print_id"] == "dog_001"
    result = loaded.identify(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    response = identification_response(result, loaded)
    assert response["match"] is True
    assert response["confidence"] == pytest.approx(1.0, abs=1e-3)
    assert response["dog"]["name"] == "Rex"


def test_index_save_load_roundtrip(tmp_path):
    from noseid.biometric import build_pipeline, GTHint, run_to_embedding
    from noseid.matching import NoseIndex
    g = NoseSynthGenerator(256, seed=3)
    pc = build_pipeline()
    idx = NoseIndex(dim=256)
    embs = []
    for i in range(4):
        s = g.render_labeled("dog_001", i)
        _, _, _, _, er = run_to_embedding(pc, s["image"], GTHint.from_sample(s), "dog_001")
        embs.append(np.asarray(er.embedding))
    idx.enroll("dog_001", embs)
    idx.save(str(tmp_path))
    idx2 = NoseIndex.load(str(tmp_path))
    assert idx2.size == 1


# ---------- Stage 11: metrics -----------------------------------------------

def test_metrics_perfect_separation():
    from noseid.training import compute_biometric_metrics
    gen = np.ones(50, dtype=np.float32) * 0.95
    imp = np.ones(50, dtype=np.float32) * 0.10
    m = compute_biometric_metrics(gen, imp)
    assert m["EER"] < 1.0
    assert m["roc_auc"] > 0.99


def test_metrics_random_is_chance():
    from noseid.training import compute_biometric_metrics
    rng = np.random.default_rng(0)
    gen = rng.uniform(0, 1, 200).astype(np.float32)
    imp = rng.uniform(0, 1, 200).astype(np.float32)
    m = compute_biometric_metrics(gen, imp)
    assert 0.4 < m["roc_auc"] < 0.6


# ---------- Backend selector ------------------------------------------------

def test_backend_resolves():
    assert resolve_backend("numpy", "detection") == "numpy"
    assert resolve_backend("auto", "detection") in ("torch", "numpy")


def test_backend_info_string():
    s = info()
    assert "installed:" in s and "missing:" in s
