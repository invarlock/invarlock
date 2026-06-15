#!/usr/bin/env python3
"""Build or check the compact public-evidence index shipped in wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "public_evidence"
SUPPORT_MATRIX = REPO_ROOT / "contracts" / "support_matrix.json"
PACKAGED_ROOT = REPO_ROOT / "src" / "invarlock" / "_data" / "public_evidence"
INDEX_FILENAME = "published_basis_index.json"
INDEX_FORMAT_VERSION = "public-evidence-index-v1"

_DIRECTORY_CONTROL_FILES = (
    "manifest.json",
    "manifest.signature.json",
    "checksums.sha256",
    "results/final_verdict.json",
    "summary.json",
    "manifest.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _logical_path(path: Path, *, source_root: Path) -> str:
    return PurePosixPath("public_evidence", path.relative_to(source_root)).as_posix()


def _directory_summary(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    control_hashes: dict[str, str] = {}
    for rel in _DIRECTORY_CONTROL_FILES:
        control_path = path / rel
        if control_path.is_file():
            control_hashes[PurePosixPath(rel).as_posix()] = _sha256_file(control_path)
    return {
        "kind": "directory",
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
        "control_hashes": control_hashes,
    }


def _artifact_summary(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        return _directory_summary(path)
    return {"kind": "missing"}


def _published_lane_map(support_matrix_path: Path) -> dict[str, list[str]]:
    payload = _load_json_object(support_matrix_path)
    lanes_by_slug: dict[str, list[str]] = {}
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        if lane.get("support_tier") != "published_basis":
            continue
        lane_id = lane.get("lane_id")
        evidence = lane.get("evidence")
        if not isinstance(lane_id, str) or not isinstance(evidence, dict):
            continue
        for value in evidence.values():
            if not isinstance(value, str):
                continue
            parts = PurePosixPath(value).parts
            if len(parts) < 3 or parts[:2] != ("public_evidence", "published_basis"):
                continue
            lanes_by_slug.setdefault(parts[2], []).append(lane_id)
    return {slug: sorted(set(lanes)) for slug, lanes in lanes_by_slug.items()}


def build_public_evidence_index(
    *,
    source_root: Path = SOURCE_ROOT,
    support_matrix_path: Path = SUPPORT_MATRIX,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    published_root = source_root / "published_basis"
    if not published_root.is_dir():
        raise FileNotFoundError(
            f"published basis directory not found: {published_root}"
        )
    if not support_matrix_path.is_file():
        raise FileNotFoundError(f"support matrix not found: {support_matrix_path}")

    lanes_by_slug = _published_lane_map(support_matrix_path)
    entries: list[dict[str, Any]] = []
    total_size = 0
    total_files = 0

    for meta_path in sorted(published_root.glob("*/evidence.meta.json")):
        artifact_dir = meta_path.parent
        metadata = _load_json_object(meta_path)
        artifact_paths = metadata.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}

        artifacts: dict[str, Any] = {}
        for key, rel_path in sorted(artifact_paths.items()):
            if not isinstance(key, str) or not isinstance(rel_path, str):
                continue
            artifact_path = artifact_dir / rel_path
            summary = _artifact_summary(artifact_path)
            summary["path"] = _logical_path(artifact_path, source_root=source_root)
            artifacts[key] = summary
            if summary.get("kind") in {"file", "directory"}:
                total_size += int(summary.get("size_bytes") or 0)
                total_files += int(summary.get("file_count") or 1)

        entry: dict[str, Any] = {
            "slug": artifact_dir.name,
            "path": _logical_path(artifact_dir, source_root=source_root),
            "lanes": lanes_by_slug.get(artifact_dir.name, []),
            "evidence_class": metadata.get("evidence_class"),
            "summary": metadata.get("summary"),
            "artifacts": artifacts,
        }
        expected_fingerprint = metadata.get("expected_fingerprint")
        if isinstance(expected_fingerprint, str):
            entry["expected_fingerprint"] = expected_fingerprint
        entries.append(entry)

    return {
        "format_version": INDEX_FORMAT_VERSION,
        "carrier_policy": {
            "source_repository": "full_public_evidence_artifacts",
            "installed_wheel": "compact_index_only",
        },
        "source_root": "public_evidence",
        "published_basis_count": len(entries),
        "published_basis_file_count": total_files,
        "published_basis_size_bytes": total_size,
        "entries": entries,
    }


def _read_index(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json_object(path)


def check_packaged_public_evidence(
    *,
    source_root: Path = SOURCE_ROOT,
    support_matrix_path: Path = SUPPORT_MATRIX,
    packaged_root: Path = PACKAGED_ROOT,
) -> list[str]:
    errors: list[str] = []
    index_path = packaged_root / INDEX_FILENAME
    legacy_tree = packaged_root / "published_basis"
    if legacy_tree.exists():
        errors.append(
            f"legacy packaged public evidence tree must be removed: {legacy_tree}"
        )
    try:
        expected = build_public_evidence_index(
            source_root=source_root,
            support_matrix_path=support_matrix_path,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    try:
        actual = _read_index(index_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    if actual is None:
        errors.append(f"missing packaged public evidence index: {index_path}")
    elif actual != expected:
        errors.append(f"out-of-sync packaged public evidence index: {index_path}")
    return errors


def sync_packaged_public_evidence(
    *,
    source_root: Path = SOURCE_ROOT,
    support_matrix_path: Path = SUPPORT_MATRIX,
    packaged_root: Path = PACKAGED_ROOT,
) -> tuple[bool, bool]:
    index = build_public_evidence_index(
        source_root=source_root,
        support_matrix_path=support_matrix_path,
    )
    packaged_root.mkdir(parents=True, exist_ok=True)
    index_path = packaged_root / INDEX_FILENAME
    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    updated = (
        not index_path.is_file() or index_path.read_text(encoding="utf-8") != content
    )
    if updated:
        index_path.write_text(content, encoding="utf-8")

    legacy_tree = packaged_root / "published_basis"
    removed_legacy_tree = legacy_tree.exists()
    if removed_legacy_tree:
        shutil.rmtree(legacy_tree)
    return updated, removed_legacy_tree


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write the compact packaged public-evidence index and remove the "
            "legacy full packaged public-evidence tree."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the packaged index matches public_evidence/.",
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--support-matrix", default=str(SUPPORT_MATRIX))
    parser.add_argument("--packaged-root", default=str(PACKAGED_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_root = Path(args.source_root)
    support_matrix_path = Path(args.support_matrix)
    packaged_root = Path(args.packaged_root)
    write_mode = args.write
    check_mode = args.check or not args.write

    if write_mode:
        try:
            updated, removed_legacy_tree = sync_packaged_public_evidence(
                source_root=source_root,
                support_matrix_path=support_matrix_path,
                packaged_root=packaged_root,
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            "Synchronized packaged public evidence index "
            f"(updated={updated}, removed_legacy_tree={removed_legacy_tree})."
        )

    if check_mode:
        errors = check_packaged_public_evidence(
            source_root=source_root,
            support_matrix_path=support_matrix_path,
            packaged_root=packaged_root,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("Packaged public evidence index is in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
