"""Stage direct nose images for manual identity assignment.

The detection, anatomy, and keypoint exports do not provide a trustworthy
per-dog identity label.  This utility therefore keeps the first step
explicitly unassigned::

    identity_data/_review/unassigned/direct_closeups/<source>/<split>/*.jpg

It writes a manifest and a CSV template.  After the CSV is filled with real
dog IDs and target splits, run the same command with ``--mapping-csv``.  The
images are then copied (never moved) into the normal identity layout::

    identity_data/train/dog_0001/*.jpg

The default sources are the segmentation and keypoint exports because they
contain close-up nose photographs.  Source files and existing destination
files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_SOURCES = ("segmentation_export", "keypoint_export")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_info(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _copy_without_overwrite(source: Path, destination: Path,
                            dry_run: bool = False) -> tuple[str, Path]:
    """Copy a file without overwriting; return status and actual path."""
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) == _sha256(source):
            return "already_present", destination
        candidate = destination.with_name(
            f"{destination.stem}__{_sha256(source)[:8]}{destination.suffix}")
        if candidate.exists() and _sha256(candidate) == _sha256(source):
            return "already_present", candidate
        destination = candidate
    if dry_run:
        return "would_copy", destination
    shutil.copy2(source, destination)
    return "copied", destination


def _source_images(source: Path) -> list[tuple[str, Path]]:
    """Return (split, image) pairs for common Roboflow export layouts."""
    pairs: list[tuple[str, Path]] = []
    found_split = False
    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        found_split = True
        image_dir = split_dir / "images"
        if not image_dir.is_dir():
            image_dir = split_dir
        pairs.extend(
            (split, path)
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not found_split:
        pairs.extend(
            ("unsplit", path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return pairs


def _write_template(path: Path, rows: list[dict]) -> str:
    if path.exists():
        return "preserved_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["staged_path", "dog_id", "target_split", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "staged_path": row["staged_path"],
                "dog_id": "",
                "target_split": "",
                "notes": "",
            })
    return "created"


def stage_images(sources: list[str], out: str, dry_run: bool = False) -> dict:
    """Copy unique direct images into the review area and write metadata."""
    output = Path(out)
    review = output / "_review" / "unassigned" / "direct_closeups"
    manifest_path = output / "_review" / "unassigned" / "direct_closeups_manifest.json"
    template_path = output / "_review" / "unassigned" / "identity_mapping_template.csv"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_hashes: dict[str, str] = {}
    errors: list[str] = []
    copied = already_present = duplicates = 0

    for source_arg in sources:
        source = Path(source_arg)
        if not source.is_dir():
            raise FileNotFoundError(f"source does not exist: {source}")
        source_name = source.name
        for split, image_path in _source_images(source):
            try:
                width, height = _image_info(image_path)
                digest = _sha256(image_path)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                continue
            if digest in seen_hashes:
                duplicates += 1
                records.append({
                    "source": source_name,
                    "source_path": str(image_path.resolve()),
                    "source_split": split,
                    "sha256": digest,
                    "status": "duplicate_exact",
                    "duplicate_of": seen_hashes[digest],
                })
                continue

            destination = review / source_name / split / image_path.name
            status, actual_destination = _copy_without_overwrite(
                image_path, destination, dry_run=dry_run)
            staged_path = _relative(actual_destination, output)
            seen_hashes[digest] = staged_path
            if status in {"copied", "would_copy"}:
                copied += 1
            else:
                already_present += 1
            records.append({
                "source": source_name,
                "source_path": str(image_path.resolve()),
                "source_split": split,
                "staged_path": staged_path,
                "sha256": digest,
                "width": width,
                "height": height,
                "status": status,
            })

    unique_records = [row for row in records if "staged_path" in row]
    manifest = {
        "purpose": "manual staging before trusted per-dog identity assignment",
        "sources": [str(Path(item).resolve()) for item in sources],
        "image_count": len(unique_records),
        "copied": copied,
        "already_present": already_present,
        "duplicate_exact": duplicates,
        "errors": errors,
        "records": records,
    }
    if dry_run:
        template_status = "not_written_dry_run"
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        template_status = _write_template(template_path, unique_records)
    return {
        "manifest": str(manifest_path),
        "mapping_template": str(template_path),
        "image_count": len(unique_records),
        "copied": copied,
        "already_present": already_present,
        "duplicate_exact": duplicates,
        "errors": len(errors),
        "template": template_status,
    }


def promote_mapping(mapping_csv: str, out: str) -> dict:
    """Copy manually mapped staged files into train/valid/test dog folders."""
    output = Path(out).resolve()
    manifest_path = output / "_review" / "unassigned" / "direct_closeups_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"staging manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = {
        row["staged_path"]: row
        for row in manifest["records"]
        if "staged_path" in row
    }
    mapping_path = Path(mapping_csv)
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping CSV does not exist: {mapping_path}")

    rows = []
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"staged_path", "dog_id", "target_split"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("mapping CSV needs staged_path,dog_id,target_split columns")
        rows = list(reader)

    seen_assignments: dict[str, tuple[str, str]] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    output_records = []
    copied = already_present = 0
    for row_number, row in enumerate(rows, start=2):
        staged_key = (row.get("staged_path") or "").strip().replace("\\", "/")
        dog_id = (row.get("dog_id") or "").strip()
        target_split = (row.get("target_split") or "").strip().lower()
        if not staged_key or not dog_id or not target_split:
            continue
        if staged_key not in known:
            raise ValueError(f"row {row_number}: staged_path is not in the manifest: {staged_key}")
        if target_split not in SPLITS:
            raise ValueError(f"row {row_number}: target_split must be train, valid, or test")
        if not SAFE_ID.fullmatch(dog_id):
            raise ValueError(f"row {row_number}: unsafe dog_id: {dog_id!r}")
        staged = (output / staged_key).resolve()
        review_root = (output / "_review" / "unassigned" / "direct_closeups").resolve()
        if review_root not in staged.parents:
            raise ValueError(f"row {row_number}: staged_path escapes the review directory")
        if not staged.is_file():
            raise FileNotFoundError(f"row {row_number}: staged image missing: {staged}")

        assignment = (dog_id, target_split)
        if staged_key in seen_assignments and seen_assignments[staged_key] != assignment:
            raise ValueError(f"row {row_number}: image assigned more than once: {staged_key}")
        seen_assignments[staged_key] = assignment
        digest = _sha256(staged)
        if digest in seen_hashes and seen_hashes[digest] != assignment:
            raise ValueError(
                f"row {row_number}: exact duplicate assigned across identities/splits: {staged_key}")
        seen_hashes[digest] = assignment

        destination = output / target_split / dog_id / staged.name
        status, actual_destination = _copy_without_overwrite(staged, destination)
        if status == "copied":
            copied += 1
        else:
            already_present += 1
        output_records.append({
            "staged_path": staged_key,
            "dog_id": dog_id,
            "target_split": target_split,
            "destination": _relative(actual_destination, output),
            "sha256": digest,
            "status": status,
        })

    result = {
        "mapping_csv": str(mapping_path.resolve()),
        "copied": copied,
        "already_present": already_present,
        "assigned_images": len(output_records),
        "records": output_records,
    }
    result_path = output / "_review" / "unassigned" / "mapped_identity_manifest.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=list(DEFAULT_SOURCES),
                        help="Roboflow export roots; defaults to direct nose exports")
    parser.add_argument("--out", default="identity_data",
                        help="identity-data root (default: identity_data)")
    parser.add_argument("--mapping-csv", default=None,
                        help="filled mapping CSV; promotes staged images into dog folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="inspect and count files without copying or writing metadata")
    args = parser.parse_args()

    if args.dry_run and args.mapping_csv:
        parser.error("--dry-run cannot be combined with --mapping-csv")
    result = {"staging": stage_images(args.source, args.out, dry_run=args.dry_run)}
    if args.mapping_csv:
        result["promotion"] = promote_mapping(args.mapping_csv, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
