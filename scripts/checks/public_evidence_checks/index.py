"""Packaged public-evidence index and external-asset validation."""

from __future__ import annotations

import hashlib
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from scripts.checks.public_evidence_checks.common import (
    PACKAGED_PUBLIC_EVIDENCE_INDEX,
    PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
    PUBLIC_EVIDENCE_ROOT,
    REPO_ROOT,
    _directory_counts,
    _load_json,
    _relative,
    _resolve_public_evidence_path,
    _sha256_file,
)


def _check_packaged_public_evidence_index(
    errors: list[str],
    public_evidence_root: Path,
    *,
    index_path: Path = PACKAGED_PUBLIC_EVIDENCE_INDEX,
    fetch_external_assets: bool = False,
) -> None:
    if not index_path.is_file():
        if public_evidence_root == PUBLIC_EVIDENCE_ROOT.resolve():
            errors.append(
                f"{_relative(index_path)}: packaged public evidence index missing"
            )
        return

    index, error = _load_json(index_path)
    if error:
        errors.append(error)
        return
    assert index is not None
    if index.get("format_version") != PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION:
        errors.append(
            f"{_relative(index_path)}: format_version must be "
            f"{PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION}"
        )
    carrier_policy = index.get("carrier_policy")
    if not isinstance(carrier_policy, dict):
        errors.append(f"{_relative(index_path)}: carrier_policy must be object")
    elif carrier_policy.get("installed_wheel") != "compact_index_only":
        errors.append(
            f"{_relative(index_path)}: installed_wheel carrier policy must be "
            "compact_index_only"
        )

    entries = index.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{_relative(index_path)}: entries must be a list")
        return
    if index.get("published_basis_count") != len(entries):
        errors.append(
            f"{_relative(index_path)}: published_basis_count must match entries"
        )
    if not entries:
        if index.get("status") != "not_created":
            errors.append(
                f"{_relative(index_path)}: an empty index must use status not_created"
            )
        if index.get("status_label") != "Evidence not yet created":
            errors.append(
                f"{_relative(index_path)}: an empty index has an invalid status label"
            )
        return

    external_assets: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(
                f"{_relative(index_path)}: entries[{entry_index}] must be object"
            )
            continue
        slug = entry.get("slug")
        entry_path = entry.get("path")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(index_path)}: entries[{entry_index}].slug required"
            )
        if (
            not isinstance(entry_path, str)
            or not entry_path.startswith("public_evidence/published_basis/")
            or ".." in Path(entry_path).parts
        ):
            errors.append(
                f"{_relative(index_path)}: entries[{entry_index}].path invalid"
            )
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            errors.append(
                f"{_relative(index_path)}: entries[{entry_index}].artifacts "
                "must be non-empty"
            )
            continue
        for artifact_name, summary in sorted(artifacts.items()):
            if not isinstance(artifact_name, str) or not isinstance(summary, dict):
                errors.append(
                    f"{_relative(index_path)}: entries[{entry_index}].artifacts "
                    "must map names to objects"
                )
                continue
            artifact_path_raw = summary.get("path")
            kind = summary.get("kind")
            if (
                not isinstance(artifact_path_raw, str)
                or not artifact_path_raw.startswith("public_evidence/published_basis/")
                or ".." in Path(artifact_path_raw).parts
            ):
                errors.append(
                    f"{_relative(index_path)}: {slug}.{artifact_name}.path invalid"
                )
                continue
            artifact_path = _resolve_public_evidence_path(
                public_evidence_root,
                artifact_path_raw,
            )
            if kind == "file":
                if not artifact_path.is_file():
                    _check_external_artifact_reference(
                        errors, index_path, slug, artifact_name, summary
                    )
                    _collect_external_asset(external_assets, summary)
                    continue
                expected_size = summary.get("size_bytes")
                if expected_size != artifact_path.stat().st_size:
                    errors.append(
                        f"{_relative(index_path)}: {slug}.{artifact_name} "
                        "size_bytes mismatch"
                    )
                expected_sha = summary.get("sha256")
                if expected_sha != _sha256_file(artifact_path):
                    errors.append(
                        f"{_relative(index_path)}: {slug}.{artifact_name} "
                        "sha256 mismatch"
                    )
            elif kind == "directory":
                if not artifact_path.is_dir():
                    _check_external_artifact_reference(
                        errors, index_path, slug, artifact_name, summary
                    )
                    _collect_external_asset(external_assets, summary)
                    continue
                expected_count = summary.get("file_count")
                expected_size = summary.get("size_bytes")
                file_count, size_bytes = _directory_counts(artifact_path)
                if expected_count != file_count:
                    errors.append(
                        f"{_relative(index_path)}: {slug}.{artifact_name} "
                        "file_count mismatch"
                    )
                if expected_size != size_bytes:
                    errors.append(
                        f"{_relative(index_path)}: {slug}.{artifact_name} "
                        "size_bytes mismatch"
                    )
                control_hashes = summary.get("control_hashes")
                if isinstance(control_hashes, dict):
                    for rel_path, expected_hash in sorted(control_hashes.items()):
                        if not isinstance(rel_path, str) or not isinstance(
                            expected_hash, str
                        ):
                            errors.append(
                                f"{_relative(index_path)}: {slug}.{artifact_name} "
                                "control_hashes must map strings to strings"
                            )
                            continue
                        control_path = artifact_path / rel_path
                        if not control_path.is_file():
                            errors.append(
                                f"{_relative(index_path)}: {slug}.{artifact_name} "
                                f"control hash file missing {rel_path!r}"
                            )
                        elif expected_hash != _sha256_file(control_path):
                            errors.append(
                                f"{_relative(index_path)}: {slug}.{artifact_name} "
                                f"control hash mismatch for {rel_path!r}"
                            )
            else:
                errors.append(
                    f"{_relative(index_path)}: {slug}.{artifact_name}.kind invalid"
                )
    if fetch_external_assets:
        _check_external_asset_downloads(errors, index_path, external_assets)


def _check_external_artifact_reference(
    errors: list[str],
    index_path: Path,
    slug: Any,
    artifact_name: str,
    summary: dict[str, Any],
) -> None:
    external = summary.get("external_asset")
    if not isinstance(external, dict):
        errors.append(
            f"{_relative(index_path)}: {slug}.{artifact_name} missing local "
            "artifact and external_asset reference"
        )
        return
    url = external.get("url")
    sha256 = external.get("sha256")
    size_bytes = external.get("size_bytes")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        errors.append(
            f"{_relative(index_path)}: {slug}.{artifact_name}.external_asset.url "
            "must be absolute HTTP(S)"
        )
    if not isinstance(sha256, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", sha256):
        errors.append(
            f"{_relative(index_path)}: {slug}.{artifact_name}.external_asset.sha256 "
            "invalid"
        )
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        errors.append(
            f"{_relative(index_path)}: {slug}.{artifact_name}.external_asset."
            "size_bytes invalid"
        )
    archive_path = external.get("archive_path")
    if archive_path is not None and (
        not isinstance(archive_path, str)
        or archive_path.startswith("/")
        or ".." in Path(archive_path).parts
    ):
        errors.append(
            f"{_relative(index_path)}: {slug}.{artifact_name}.external_asset."
            "archive_path invalid"
        )


def _collect_external_asset(
    external_assets: dict[tuple[str, str, int], dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    external = summary.get("external_asset")
    if not isinstance(external, dict):
        return
    url = external.get("url")
    sha256 = external.get("sha256")
    size_bytes = external.get("size_bytes")
    if isinstance(url, str) and isinstance(sha256, str) and isinstance(size_bytes, int):
        external_assets.setdefault((url, sha256, size_bytes), external)


def _check_external_asset_downloads(
    errors: list[str],
    index_path: Path,
    external_assets: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    for url, expected_sha, expected_size in sorted(external_assets):
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                tempfile.NamedTemporaryFile() as handle,
            ):
                digest = hashlib.sha256()
                total = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        except OSError as exc:
            errors.append(
                f"{_relative(index_path)}: external asset download failed {url}: {exc}"
            )
            continue
        actual_sha = "sha256:" + digest.hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(index_path)}: external asset sha256 mismatch {url}"
            )
        if total != expected_size:
            errors.append(
                f"{_relative(index_path)}: external asset size mismatch {url}"
            )


def _indexed_public_evidence_paths() -> set[str]:
    if not PACKAGED_PUBLIC_EVIDENCE_INDEX.is_file():
        return set()
    index, error = _load_json(PACKAGED_PUBLIC_EVIDENCE_INDEX)
    if error or index is None:
        return set()
    paths: set[str] = set()
    entries = index.get("entries")
    if not isinstance(entries, list):
        return paths
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for summary in artifacts.values():
            if not isinstance(summary, dict):
                continue
            path = summary.get("path")
            if isinstance(path, str):
                paths.add(path)
    return paths


def _public_evidence_file_exists_or_indexed(rel_path: str) -> bool:
    if (REPO_ROOT / rel_path).is_file():
        return True
    return rel_path in _indexed_public_evidence_paths()
