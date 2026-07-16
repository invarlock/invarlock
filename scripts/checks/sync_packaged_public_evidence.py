#!/usr/bin/env python3
"""Build or check the compact public-evidence index shipped in wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "public_evidence"
PACKAGED_ROOT = REPO_ROOT / "src" / "invarlock" / "_data" / "public_evidence"
INDEX_FILENAME = "evidence_index.json"
INDEX_FORMAT_VERSION = "invarlock/public-evidence-index-v1"
META_FORMAT_VERSION = "invarlock/public-evidence-meta-v1"
EVIDENCE_DIRNAME = "evidence"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"{item}: symlinks are not allowed in public evidence")
        if item.is_file():
            files.append(item)
    return sorted(files)


def _artifact_summary(path: Path, *, source_root: Path) -> dict[str, Any]:
    logical = PurePosixPath("public_evidence", path.relative_to(source_root)).as_posix()
    if path.is_file() and not path.is_symlink():
        return {
            "kind": "file",
            "path": logical,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if path.is_dir() and not path.is_symlink():
        files = _regular_files(path)
        return {
            "kind": "directory",
            "path": logical,
            "file_count": len(files),
            "size_bytes": sum(item.stat().st_size for item in files),
            "control_hashes": {
                relative: _sha256(path / relative)
                for relative in (
                    "manifest.json",
                    "manifest.signature.json",
                    "checksums.sha256",
                )
                if (path / relative).is_file()
            },
        }
    raise ValueError(f"{path}: public evidence artifact is missing or unsafe")


def _empty_index() -> dict[str, Any]:
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "status": "not_created",
        "status_label": "Evidence not yet created",
        "carrier_policy": {"installed_wheel": "compact_index_only"},
        "evidence_count": 0,
        "evidence_file_count": 0,
        "evidence_size_bytes": 0,
        "entries": [],
    }


def _validate_index(path: Path, value: dict[str, Any]) -> None:
    if value.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported public-evidence index format")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: entries must be a list")
    if value.get("evidence_count") != len(entries):
        raise ValueError(f"{path}: evidence_count must match entries")
    for field in ("evidence_file_count", "evidence_size_bytes"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{path}: {field} must be a non-negative integer")
    if not entries and (
        value.get("status") != "not_created"
        or value.get("status_label") != "Evidence not yet created"
    ):
        raise ValueError(f"{path}: empty index must say Evidence not yet created")
    if entries and (
        value.get("status") != "available"
        or value.get("status_label") != "Evidence available"
    ):
        raise ValueError(f"{path}: non-empty index must say evidence is available")
    slugs = [entry.get("slug") for entry in entries if isinstance(entry, dict)]
    if (
        len(slugs) != len(entries)
        or any(not isinstance(slug, str) or not slug for slug in slugs)
        or len(set(slugs)) != len(slugs)
    ):
        raise ValueError(f"{path}: entries must have unique slugs")


def _external_entries(
    source_root: Path, *, replacing_slugs: set[str]
) -> list[dict[str, Any]]:
    source_index = source_root / INDEX_FILENAME
    if not source_index.is_file():
        return []
    value = _read_object(source_index)
    _validate_index(source_index, value)
    entries = value["entries"]
    assert isinstance(entries, list)
    preserved: list[dict[str, Any]] = []
    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        slug = raw_entry.get("slug")
        if slug in replacing_slugs:
            continue
        artifacts = raw_entry.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "evidence_pack",
            "verification_receipt",
        }:
            raise ValueError(
                f"{source_index}: external entry {slug!r} has invalid artifacts"
            )
        for role, summary in artifacts.items():
            if not isinstance(summary, dict) or not isinstance(
                summary.get("external_asset"), dict
            ):
                raise ValueError(
                    f"{source_index}: non-local {role} for {slug!r} "
                    "must name an external_asset"
                )
            logical = summary.get("path")
            if isinstance(logical, str) and (source_root.parent / logical).exists():
                raise ValueError(
                    f"{source_index}: local artifact for {slug!r} requires "
                    "evidence.meta.json"
                )
        preserved.append(raw_entry)
    return preserved


def _artifact_totals(entries: list[dict[str, Any]]) -> tuple[int, int]:
    file_count = 0
    size_bytes = 0
    for entry in entries:
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("public evidence entry artifacts must be an object")
        for summary in artifacts.values():
            if not isinstance(summary, dict):
                raise ValueError("public evidence artifact summary must be an object")
            kind = summary.get("kind")
            count = 1 if kind == "file" else summary.get("file_count")
            size = summary.get("size_bytes")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ValueError("public evidence artifact totals are invalid")
            file_count += count
            size_bytes += size
    return file_count, size_bytes


def _validate_metadata(path: Path, value: dict[str, Any]) -> tuple[dict[str, str], str]:
    if value.get("format_version") != META_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported metadata format")
    if set(value) != {"format_version", "summary", "artifact_paths"}:
        raise ValueError(f"{path}: metadata fields are not closed")
    summary = value.get("summary")
    if (
        not isinstance(summary, str)
        or not summary
        or summary != summary.strip()
        or len(summary) > 512
        or any(ord(character) < 32 for character in summary)
    ):
        raise ValueError(f"{path}: summary must be concise plain text")
    artifact_paths = value.get("artifact_paths")
    if not isinstance(artifact_paths, dict) or set(artifact_paths) != {
        "evidence_pack",
        "verification_receipt",
    }:
        raise ValueError(f"{path}: artifact_paths must name only the pack and receipt")
    normalized: dict[str, str] = {}
    for role, relative in artifact_paths.items():
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or len(PurePosixPath(relative).parts) != 1
            or PurePosixPath(relative).name != relative
            or relative in {".", "..", "evidence.meta.json"}
        ):
            raise ValueError(f"{path}: invalid direct-child {role} path")
        normalized[role] = relative
    return normalized, summary


def build_public_evidence_index(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    evidence_root = source_root / EVIDENCE_DIRNAME
    metadata_files = sorted(evidence_root.glob("*/evidence.meta.json"))
    local_slugs = {metadata_path.parent.name for metadata_path in metadata_files}
    entries = _external_entries(source_root, replacing_slugs=local_slugs)
    for metadata_path in metadata_files:
        metadata = _read_object(metadata_path)
        artifact_paths, summary = _validate_metadata(metadata_path, metadata)
        root = metadata_path.parent
        artifacts: dict[str, Any] = {}
        for role, relative in artifact_paths.items():
            candidate = root / relative
            if candidate.resolve().parent != root.resolve():
                raise ValueError(
                    f"{metadata_path}: {role} must be a direct entry child"
                )
            artifacts[role] = _artifact_summary(candidate, source_root=source_root)
        entries.append(
            {
                "slug": root.name,
                "path": PurePosixPath(
                    "public_evidence", root.relative_to(source_root)
                ).as_posix(),
                "evidence_class": "signed_evidence_pack",
                "summary": summary,
                "artifacts": artifacts,
            }
        )

    if not entries:
        return _empty_index()
    entries.sort(key=lambda entry: str(entry["slug"]))
    file_count, size_bytes = _artifact_totals(entries)
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "status": "available",
        "status_label": "Evidence available",
        "carrier_policy": {"installed_wheel": "compact_index_only"},
        "evidence_count": len(entries),
        "evidence_file_count": file_count,
        "evidence_size_bytes": size_bytes,
        "entries": entries,
    }


def _encoded(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sync(*, source_root: Path, packaged_root: Path, write: bool) -> list[str]:
    errors: list[str] = []
    expected = _encoded(build_public_evidence_index(source_root))
    source_destination = source_root / INDEX_FILENAME
    destination = packaged_root / INDEX_FILENAME
    obsolete_packaged_tree = packaged_root / EVIDENCE_DIRNAME
    if obsolete_packaged_tree.exists():
        if write:
            shutil.rmtree(obsolete_packaged_tree)
        else:
            errors.append("full packaged public evidence tree must be removed")
    if write:
        source_root.mkdir(parents=True, exist_ok=True)
        source_destination.write_bytes(expected)
        packaged_root.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(expected)
    else:
        if (
            not source_destination.is_file()
            or source_destination.read_bytes() != expected
        ):
            errors.append("source public evidence index is out of sync")
        if not destination.is_file() or destination.read_bytes() != expected:
            errors.append("packaged public evidence index is out of sync")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--packaged-root", type=Path, default=PACKAGED_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        errors = sync(
            source_root=args.source_root,
            packaged_root=args.packaged_root,
            write=args.write,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(error)
        return 1
    print(
        "Public evidence index written."
        if args.write
        else "Public evidence index is in sync."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
