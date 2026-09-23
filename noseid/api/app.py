"""FastAPI service for the anatomy-feature dog gallery.

The service is deliberately thin: all biometric work stays in
``noseid.biometric`` and all persisted identity templates stay in
``FeatureIndex``.  This module only handles uploads, registry metadata, and a
browser-friendly response shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import threading
from pathlib import Path
from time import perf_counter

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from ..biometric import (build_pipeline, enroll_dog, identification_response,
                         identify, registration_response)
from ..config import PROJECT_ROOT, get_config
from ..matching import FeatureIndex


FRONTEND_ROOT = PROJECT_ROOT / "frontend"
INDEX_ROOT = PROJECT_ROOT / "output" / "index"
REGISTRY_ROOT = PROJECT_ROOT / "output" / "registry"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass
class _Runtime:
    pipeline: object
    index: FeatureIndex


_runtime: _Runtime | None = None
_runtime_lock = threading.Lock()


def _runtime_or_error() -> _Runtime:
    """Load the models and active gallery once, on the first real request."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        gallery = INDEX_ROOT / "feature_meta.json"
        if not gallery.exists():
            raise HTTPException(
                status_code=503,
                detail=("The anatomy-feature gallery is not available. Build "
                        "output/index/feature_meta.json first."),
            )
        try:
            cfg = get_config()
            _runtime = _Runtime(
                pipeline=build_pipeline(cfg),
                index=FeatureIndex.load(str(INDEX_ROOT)),
            )
            _backfill_legacy_registration_profiles(_runtime.index)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Biometric models could not be loaded: {exc}",
            ) from exc
    return _runtime


def _safe_dog_id(value: str | None, index: FeatureIndex) -> str:
    value = (value or "").strip()
    if value:
        if not _ID_RE.fullmatch(value):
            raise HTTPException(
                status_code=422,
                detail="dog_id may contain only letters, numbers, '_' and '-'.",
            )
        return value
    number = index.size + 1
    while index.get_record(f"dog_{number:04d}") is not None:
        number += 1
    return f"dog_{number:04d}"


async def _decode_upload(upload: UploadFile) -> tuple[np.ndarray, bytes]:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=422, detail="An uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Each image must be 12 MB or smaller.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="One upload is not a readable image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), data


def _write_registry_jpeg(target: Path, image: np.ndarray) -> None:
    """Write an RGB image in a browser-friendly JPEG format."""
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save a registry photo.")
    target.write_bytes(encoded.tobytes())


def _save_registry_images(index_id: str, images: list[np.ndarray]) -> list[str]:
    """Persist all enrollment images and return their public URLs."""
    target_dir = REGISTRY_ROOT / index_id
    image_dir = target_dir / "images"
    urls = []
    for number, image in enumerate(images, start=1):
        filename = f"image_{number:03d}.jpg"
        _write_registry_jpeg(image_dir / filename, image)
        urls.append(f"/media/{index_id}/images/{filename}")
    if images:
        _write_registry_jpeg(target_dir / "profile.jpg", images[0])
    return urls


def _backfill_legacy_registration_profiles(index: FeatureIndex) -> None:
    """Give old folder enrollments a display name and profile photo.

    The batch registration command predates the browser metadata contract and
    recorded only ``source: registration``. This one-time migration copies all
    existing photos; it does not rerun any model or alter a feature template.
    """
    changed = False
    for record in index.list_records():
        dog_id = str(record.get("dog_id"))
        metadata = record.get("metadata", {}) or {}
        if metadata.get("photo_urls") or metadata.get("source") != "registration":
            continue
        dog_dir = PROJECT_ROOT / "registration" / dog_id
        photos = ([path for path in sorted(dog_dir.iterdir())
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
                  if dog_dir.is_dir() else [])
        if not photos:
            continue
        images = []
        for photo in photos:
            image_bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not images:
            continue
        image_urls = _save_registry_images(dog_id, images)
        index.update_metadata(dog_id, {
            "name": dog_id,
            "photo_url": image_urls[0],
            "photo_urls": image_urls,
        })
        changed = True
    if changed:
        index.save(str(INDEX_ROOT))


def _public_record(record: dict) -> dict:
    metadata = dict(record.get("metadata", {}))
    image_urls = list(metadata.get("photo_urls", []) or [])
    if not image_urls and metadata.get("photo_url"):
        image_urls = [metadata["photo_url"]]
    return {
        "dog_id": record.get("dog_id"),
        "name": metadata.get("name") or record.get("dog_id"),
        "identification_chip_id": metadata.get("identification_chip_id"),
        "colour": metadata.get("colour"),
        "breed": metadata.get("breed"),
        "age": metadata.get("age"),
        "blood_type": metadata.get("blood_type"),
        "owner": metadata.get("owner") or {},
        "photo_url": metadata.get("photo_url") or (image_urls[0] if image_urls else None),
        "photo_urls": image_urls,
        "image_count": len(image_urls),
        "num_images": record.get("num_images", 0),
        "features": record.get("features", []),
        "embedding_version": record.get("embedding_version"),
    }


app = FastAPI(
    title="Dog Nose Biometric Identifier",
    version="0.2.0",
    description="Registration and identification using anatomy-specific nose embeddings.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="static")
app.mount("/media", StaticFiles(directory=str(REGISTRY_ROOT), check_dir=False), name="media")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict:
    runtime = _runtime_or_error()
    return {
        "status": "ok",
        "gallery_size": runtime.index.size,
        "embedding_version": runtime.index.embedding_version,
        "features": ["rhinarium", "left_nare", "right_nare", "philtrum"],
    }


@app.get("/api/dogs")
def dogs() -> dict:
    runtime = _runtime_or_error()
    records = [_public_record(record) for record in runtime.index.list_records()]
    return {"dogs": records, "count": len(records)}


@app.get("/api/dogs/{dog_id}")
def dog(dog_id: str) -> dict:
    runtime = _runtime_or_error()
    record = runtime.index.get_record(dog_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Registered dog not found.")
    return {"dog": _public_record(record)}


@app.delete("/api/dogs/{dog_id}")
def delete_dog(dog_id: str) -> dict:
    """Delete an active dog profile and its locally stored registry photos."""
    runtime = _runtime_or_error()
    if runtime.index.get_record(dog_id) is None:
        raise HTTPException(status_code=404, detail="Registered dog not found.")
    if not _ID_RE.fullmatch(dog_id):
        raise HTTPException(status_code=422, detail="Invalid dog ID.")

    # The ID has been validated and the resolved path is required to remain a
    # direct child of output/registry before any recursive deletion occurs.
    registry_root = REGISTRY_ROOT.resolve()
    target = (REGISTRY_ROOT / dog_id).resolve()
    if target.parent != registry_root:
        raise HTTPException(status_code=422, detail="Invalid registry path.")
    deleted = runtime.index.delete(dog_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Registered dog not found.")
    runtime.index.save(str(INDEX_ROOT))
    if target.is_dir():
        shutil.rmtree(target)
    return {
        "status": "deleted",
        "dog_id": dog_id,
        "message": "The active dog profile and registry photos were deleted.",
    }


@app.post("/api/register")
async def register(
    files: list[UploadFile] = File(..., description="Three or more nose photos"),
    name: str = Form(...),
    dog_id: str | None = Form(default=None),
    identification_chip_id: str | None = Form(default=None),
    colour: str | None = Form(default=None),
    breed: str | None = Form(default=None),
    age: str | None = Form(default=None),
    blood_type: str | None = Form(default=None),
    owner_name: str | None = Form(default=None),
    owner_phone: str | None = Form(default=None),
    owner_email: str | None = Form(default=None),
    owner_address: str | None = Form(default=None),
    min_valid: int = Form(default=3),
) -> dict:
    if not files:
        raise HTTPException(status_code=422, detail="Upload at least three photos.")
    if not 1 <= min_valid <= 50:
        raise HTTPException(status_code=422, detail="min_valid must be between 1 and 50.")
    display_name = name.strip()
    if not display_name:
        raise HTTPException(status_code=422, detail="A dog name is required.")

    runtime = _runtime_or_error()
    resolved_id = _safe_dog_id(dog_id, runtime.index)
    images: list[np.ndarray] = []
    for upload in files:
        image, _ = await _decode_upload(upload)
        images.append(image)
    photo_urls = [
        f"/media/{resolved_id}/images/image_{number:03d}.jpg"
        for number in range(1, len(images) + 1)
    ]
    owner = {
        "name": (owner_name or "").strip(),
        "phone": (owner_phone or "").strip(),
        "email": (owner_email or "").strip(),
        "address": (owner_address or "").strip(),
    }
    owner = {key: value for key, value in owner.items() if value}
    metadata = {
        "name": display_name,
        "identification_chip_id": (identification_chip_id or "").strip() or None,
        "colour": (colour or "").strip() or None,
        "breed": (breed or "").strip() or None,
        "age": (age or "").strip() or None,
        "blood_type": (blood_type or "").strip() or None,
        "owner": owner,
        "photo_url": photo_urls[0],
        "photo_urls": photo_urls,
    }
    try:
        enrolled = enroll_dog(
            runtime.pipeline, runtime.index, resolved_id, images,
            min_valid=min_valid, metadata=metadata,
        )
        _save_registry_images(resolved_id, images)
        runtime.index.save(str(INDEX_ROOT))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}") from exc

    response = registration_response(enrolled)
    response.update({
        "status": "registered",
        "message": f"{display_name} is registered.",
        "dog": {"dog_id": resolved_id, **metadata},
        "gallery_size": runtime.index.size,
    })
    return response


@app.post("/api/identify")
async def identify_upload(
    file: UploadFile = File(...),
    debug: bool = Query(default=False),
) -> dict:
    runtime = _runtime_or_error()
    image, _ = await _decode_upload(file)
    started = perf_counter()
    try:
        result, diagnostics = identify(runtime.pipeline, runtime.index, image)
        response = identification_response(result, runtime.index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Identification failed: {exc}") from exc
    response["processing_ms"] = round((perf_counter() - started) * 1000, 1)
    if debug:
        response["pipeline"] = diagnostics
    return response
