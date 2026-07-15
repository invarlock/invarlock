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

from invarlock.evidence_pack_json import load_json_object, read_regular_file_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "public_evidence"
SUPPORT_MATRIX = REPO_ROOT / "contracts" / "support_matrix.json"
PACKAGED_ROOT = REPO_ROOT / "src" / "invarlock" / "_data" / "public_evidence"
INDEX_FILENAME = "catalog_evidence_index.json"
INDEX_FORMAT_VERSION = "public-evidence-index-v2"
CURRENT_EVIDENCE_ROOT_NAME = "catalog_evidence"
LEGACY_EVIDENCE_ROOT_NAME = "published_basis"
EVIDENCE_ROOT_NAMES = frozenset({CURRENT_EVIDENCE_ROOT_NAME, LEGACY_EVIDENCE_ROOT_NAME})
SOURCE_REPOSITORY_FULL = "full_public_evidence_artifacts"
SOURCE_REPOSITORY_EXTERNAL = "compact_index_and_external_assets"

_DIRECTORY_CONTROL_FILES = (
    "manifest.json",
    "manifest.signature.json",
    "checksums.sha256",
    "results/final_verdict.json",
    "summary.json",
    "guard_value_manifest.json",
    "guard_value_summary.json",
    "artifact_package.json",
    "checkpoint_refs.json",
    "artifact_package/state/scenarios.json",
    "artifact_package/reports/final_verdict.json",
    "manifest.json",
)
_SUPPORT_MATRIX_ARTIFACT_KEYS = {
    "evaluation_report_fixture": "evaluation_report",
    "runtime_manifest_fixture": "runtime_manifest",
    "evidence_pack_recipe": "evidence_pack_recipe",
    "evidence_pack_fixture": "evidence_pack",
    "artifact_package": "artifact_package",
    "guard_value_demo": "guard_value_demo",
}


def _sha256_file(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            read_regular_file_bytes(path, label="public evidence artifact")
        ).hexdigest()
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    return load_json_object(path, label="public evidence object")


def _validate_public_evidence_index(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(f"{path}: format_version must be {INDEX_FORMAT_VERSION}")
    carrier_policy = payload.get("carrier_policy")
    if not isinstance(carrier_policy, dict):
        raise ValueError(f"{path}: carrier_policy must be an object")
    if carrier_policy.get("installed_wheel") != "compact_index_only":
        raise ValueError(f"{path}: installed_wheel carrier policy invalid")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: entries must be a list")
    if payload.get("catalog_evidence_count") != len(entries):
        raise ValueError(f"{path}: catalog_evidence_count must match entries")
    for field in ("catalog_evidence_file_count", "catalog_evidence_size_bytes"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{path}: {field} must be a non-negative integer")
    if not entries and (
        payload.get("status") != "not_created"
        or payload.get("status_label") != "Evidence not yet created"
    ):
        raise ValueError(f"{path}: empty index must declare not_created status")


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


def _unique_file_totals(path: Path) -> tuple[int, int]:
    """Count each logical carrier file once, independent of artifact roles."""

    files = sorted(item for item in path.rglob("*") if item.is_file())
    return len(files), sum(item.stat().st_size for item in files)


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


def _maintained_lane_map(support_matrix_path: Path) -> dict[str, list[str]]:
    payload = _load_json_object(support_matrix_path)
    lanes_by_slug: dict[str, list[str]] = {}
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        if lane.get("support_tier") != "maintained_catalog":
            continue
        lane_id = lane.get("lane_id")
        evidence = lane.get("evidence")
        if not isinstance(lane_id, str) or not isinstance(evidence, dict):
            continue
        for value in evidence.values():
            if not isinstance(value, str):
                continue
            parts = PurePosixPath(value).parts
            if (
                len(parts) < 3
                or parts[0] != "public_evidence"
                or parts[1] not in EVIDENCE_ROOT_NAMES
            ):
                continue
            lanes_by_slug.setdefault(parts[2], []).append(lane_id)
    return {slug: sorted(set(lanes)) for slug, lanes in lanes_by_slug.items()}


def _maintained_support_artifacts(
    support_matrix_path: Path,
) -> dict[str, dict[str, str]]:
    payload = _load_json_object(support_matrix_path)
    artifacts_by_slug: dict[str, dict[str, str]] = {}
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        if lane.get("support_tier") != "maintained_catalog":
            continue
        evidence = lane.get("evidence")
        if not isinstance(evidence, dict):
            continue
        for key, value in evidence.items():
            artifact_key = _SUPPORT_MATRIX_ARTIFACT_KEYS.get(key)
            if artifact_key is None or not isinstance(value, str):
                continue
            parts = PurePosixPath(value).parts
            if (
                len(parts) < 3
                or parts[0] != "public_evidence"
                or parts[1] not in EVIDENCE_ROOT_NAMES
            ):
                continue
            artifacts_by_slug.setdefault(parts[2], {}).setdefault(
                artifact_key,
                value,
            )
    return artifacts_by_slug


def _attach_external_asset(
    index: dict[str, Any],
    *,
    url: str,
    sha256: str,
    size_bytes: int,
    archive_root: str,
) -> dict[str, Any]:
    carrier_policy = index.setdefault("carrier_policy", {})
    if isinstance(carrier_policy, dict):
        carrier_policy["source_repository"] = SOURCE_REPOSITORY_EXTERNAL

    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for summary in artifacts.values():
            if not isinstance(summary, dict):
                continue
            if summary.get("kind") not in {"file", "directory"}:
                continue
            artifact_path = summary.get("path")
            if not isinstance(artifact_path, str):
                continue
            summary["external_asset"] = {
                "url": url,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "archive_root": archive_root,
                "archive_path": artifact_path,
            }
    return index


def build_public_evidence_index(
    *,
    source_root: Path = SOURCE_ROOT,
    support_matrix_path: Path = SUPPORT_MATRIX,
    source_index_path: Path | None = None,
    external_asset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if source_index_path is None:
        source_index_path = source_root / INDEX_FILENAME
    catalog_evidence_root = source_root / CURRENT_EVIDENCE_ROOT_NAME
    has_local_entries = catalog_evidence_root.is_dir() and any(
        catalog_evidence_root.glob("*/evidence.meta.json")
    )
    if not has_local_entries and source_index_path.is_file():
        index = _load_json_object(source_index_path)
        _validate_public_evidence_index(source_index_path, index)
        return index
    if not has_local_entries:
        raise FileNotFoundError(
            f"catalog evidence directory or source index not found: "
            f"{catalog_evidence_root}, {source_index_path}"
        )
    if not support_matrix_path.is_file():
        raise FileNotFoundError(f"support matrix not found: {support_matrix_path}")

    lanes_by_slug = _maintained_lane_map(support_matrix_path)
    support_artifacts_by_slug = _maintained_support_artifacts(support_matrix_path)
    entries: list[dict[str, Any]] = []
    total_files, total_size = _unique_file_totals(catalog_evidence_root)

    for meta_path in sorted(catalog_evidence_root.glob("*/evidence.meta.json")):
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
        for key, logical_path in sorted(
            support_artifacts_by_slug.get(artifact_dir.name, {}).items()
        ):
            if key in artifacts:
                continue
            artifact_path = source_root / PurePosixPath(logical_path).relative_to(
                "public_evidence"
            )
            summary = _artifact_summary(artifact_path)
            summary["path"] = logical_path
            artifacts[key] = summary

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

    index = {
        "format_version": INDEX_FORMAT_VERSION,
        "carrier_policy": {
            "source_repository": SOURCE_REPOSITORY_FULL,
            "installed_wheel": "compact_index_only",
        },
        "source_root": "public_evidence",
        "catalog_evidence_count": len(entries),
        "catalog_evidence_file_count": total_files,
        "catalog_evidence_size_bytes": total_size,
        "entries": entries,
    }
    if external_asset is not None:
        index = _attach_external_asset(index, **external_asset)
    return index


def _read_index(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json_object(path)


def check_packaged_public_evidence(
    *,
    source_root: Path = SOURCE_ROOT,
    support_matrix_path: Path = SUPPORT_MATRIX,
    packaged_root: Path = PACKAGED_ROOT,
    source_index_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    index_path = packaged_root / INDEX_FILENAME
    full_evidence_trees = [
        packaged_root / root_name for root_name in sorted(EVIDENCE_ROOT_NAMES)
    ]
    for full_tree in full_evidence_trees:
        if not full_tree.exists():
            continue
        errors.append(
            f"full packaged public evidence tree must be removed: {full_tree}"
        )
    try:
        expected = build_public_evidence_index(
            source_root=source_root,
            support_matrix_path=support_matrix_path,
            source_index_path=source_index_path,
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
    source_index_path: Path | None = None,
    external_asset: dict[str, Any] | None = None,
    write_source_index: bool = False,
) -> tuple[bool, bool, bool]:
    if source_index_path is None:
        source_index_path = source_root / INDEX_FILENAME
    index = build_public_evidence_index(
        source_root=source_root,
        support_matrix_path=support_matrix_path,
        source_index_path=source_index_path,
        external_asset=external_asset,
    )
    source_index_updated = False
    if write_source_index:
        source_index_path.parent.mkdir(parents=True, exist_ok=True)
        source_content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        source_index_updated = (
            not source_index_path.is_file()
            or source_index_path.read_text(encoding="utf-8") != source_content
        )
        if source_index_updated:
            source_index_path.write_text(source_content, encoding="utf-8")

    packaged_root.mkdir(parents=True, exist_ok=True)
    index_path = packaged_root / INDEX_FILENAME
    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    updated = (
        not index_path.is_file() or index_path.read_text(encoding="utf-8") != content
    )
    if updated:
        index_path.write_text(content, encoding="utf-8")

    full_evidence_trees = [
        packaged_root / root_name for root_name in sorted(EVIDENCE_ROOT_NAMES)
    ]
    removed_full_tree = any(path.exists() for path in full_evidence_trees)
    for full_tree in full_evidence_trees:
        if full_tree.exists():
            shutil.rmtree(full_tree)
    return updated, removed_full_tree, source_index_updated


def _external_asset_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    values = {
        "url": args.external_asset_url,
        "sha256": args.external_asset_sha256,
        "size_bytes": args.external_asset_size_bytes,
        "archive_root": args.external_asset_archive_root,
    }
    provided = [
        args.external_asset_url is not None,
        args.external_asset_sha256 is not None,
        args.external_asset_size_bytes is not None,
    ]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError("external asset options must be provided together")
    if not str(values["url"]).startswith(("https://", "http://")):
        raise ValueError("external asset URL must be absolute HTTP(S)")
    if not isinstance(values["sha256"], str) or not values["sha256"].startswith(
        "sha256:"
    ):
        raise ValueError("external asset sha256 must start with sha256:")
    if not isinstance(values["size_bytes"], int) or values["size_bytes"] <= 0:
        raise ValueError("external asset size must be positive")
    return values


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
    parser.add_argument(
        "--source-index",
        default=None,
        help=(
            "Source-tree compact index used when public_evidence/catalog_evidence "
            "is externalized. Defaults to public_evidence/catalog_evidence_index.json."
        ),
    )
    parser.add_argument(
        "--write-source-index",
        action="store_true",
        help="Also write the source-tree compact index.",
    )
    parser.add_argument("--external-asset-url", default=None)
    parser.add_argument("--external-asset-sha256", default=None)
    parser.add_argument("--external-asset-size-bytes", type=int, default=None)
    parser.add_argument(
        "--external-asset-archive-root",
        default="public_evidence/catalog_evidence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_root = Path(args.source_root)
    support_matrix_path = Path(args.support_matrix)
    packaged_root = Path(args.packaged_root)
    source_index_path = (
        Path(args.source_index)
        if args.source_index is not None
        else source_root / INDEX_FILENAME
    )
    write_mode = args.write
    check_mode = args.check or not args.write
    try:
        external_asset = _external_asset_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if write_mode:
        try:
            (
                updated,
                removed_full_tree,
                source_index_updated,
            ) = sync_packaged_public_evidence(
                source_root=source_root,
                support_matrix_path=support_matrix_path,
                packaged_root=packaged_root,
                source_index_path=source_index_path,
                external_asset=external_asset,
                write_source_index=args.write_source_index,
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            "Synchronized packaged public evidence index "
            f"(updated={updated}, removed_full_tree={removed_full_tree}, "
            f"source_index_updated={source_index_updated})."
        )

    if check_mode:
        errors = check_packaged_public_evidence(
            source_root=source_root,
            support_matrix_path=support_matrix_path,
            packaged_root=packaged_root,
            source_index_path=source_index_path,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("Packaged public evidence index is in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
