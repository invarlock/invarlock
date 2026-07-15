#!/usr/bin/env python3
"""Stage, verify, and upload an immutable runtime-evidence release asset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from scripts.release import runtime_release_evidence
else:
    runtime_release_evidence = importlib.import_module(
        "scripts.release.runtime_release_evidence"
        if __package__
        else "runtime_release_evidence"
    )


HANDOFF_FORMAT: Final = "invarlock/runtime-release-asset-handoff-v1"
MAX_ASSET_BYTES: Final = 4 * 1024 * 1024
MAX_DIGEST_FILE_BYTES: Final = 256
SUPPORTED_PROVIDERS: Final = ("llama_cpp", "tensorrt_llm")

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_RELEASE_TAG = re.compile(
    r"^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?$"
)
_REPOSITORY_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?"
_REPOSITORY = re.compile(rf"^{_REPOSITORY_COMPONENT}/{_REPOSITORY_COMPONENT}$")


class RuntimeReleaseAssetHandoffError(RuntimeError):
    """Raised when the release-asset handoff is unsafe or inconsistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_release_tag(release_tag: str) -> None:
    if _RELEASE_TAG.fullmatch(release_tag) is None:
        raise RuntimeReleaseAssetHandoffError(
            "release tag must be a lowercase v-prefixed semantic version"
        )


def _require_digest(digest: str) -> None:
    if _SHA256.fullmatch(digest) is None:
        raise RuntimeReleaseAssetHandoffError(
            "expected asset digest must be lowercase sha256"
        )


def _asset_filename(*, release_tag: str, source_commit: str, asset_sha256: str) -> str:
    _require_release_tag(release_tag)
    _require_digest(asset_sha256)
    if _GIT_COMMIT.fullmatch(source_commit) is None:
        raise RuntimeReleaseAssetHandoffError(
            "evidence source commit must be a full lowercase commit"
        )
    return (
        f"invarlock-{release_tag}-runtime-evidence-source-{source_commit[:12]}-"
        f"{asset_sha256}.tar.gz"
    )


def _digest_filename(asset_filename: str) -> str:
    return f"{asset_filename}.sha256"


def _read_regular_file(path: Path, *, label: str, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeReleaseAssetHandoffError(
            f"{label} is not safely readable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise RuntimeReleaseAssetHandoffError(
                f"{label} must be a bounded regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise RuntimeReleaseAssetHandoffError(f"{label} exceeds the byte limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeReleaseAssetHandoffError(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as exc:
        raise RuntimeReleaseAssetHandoffError(f"{label} could not be read") from exc
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise RuntimeReleaseAssetHandoffError(
            f"{label} already exists or cannot be created safely"
        ) from exc
    completed = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeReleaseAssetHandoffError(f"{label} could not be written")
            view = view[written:]
        os.fsync(descriptor)
        completed = True
    except OSError as exc:
        raise RuntimeReleaseAssetHandoffError(f"{label} could not be written") from exc
    finally:
        os.close(descriptor)
        if not completed:
            path.unlink(missing_ok=True)


def _summary(
    *,
    release_tag: str,
    asset_filename: str,
    asset_sha256: str,
    source_commit: str,
    source_archive_sha256: str,
    asset_size: int,
    qualification_count: int,
    behavioral_claim_count: int,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "format_version": HANDOFF_FORMAT,
        "status": status,
        "release_tag": release_tag,
        "asset_filename": asset_filename,
        "digest_filename": _digest_filename(asset_filename),
        "asset_sha256": asset_sha256,
        "asset_size": asset_size,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "qualification_count": qualification_count,
        "behavioral_claim_count": behavioral_claim_count,
    }


def verify_handoff(
    *,
    asset: Path,
    digest_file: Path,
    release_tag: str,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
    expected_asset_sha256: str,
    expected_providers: frozenset[str],
    expected_qualifications: frozenset[str],
    require_behavioral_claim: bool,
) -> dict[str, object]:
    """Verify the exact bytes, immutable name, sidecar, and evidence closure."""

    expected_filename = _asset_filename(
        release_tag=release_tag,
        source_commit=expected_source_commit,
        asset_sha256=expected_asset_sha256,
    )
    if asset.name != expected_filename:
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence asset does not use its immutable canonical name"
        )
    expected_digest_filename = _digest_filename(expected_filename)
    if digest_file.name != expected_digest_filename:
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence digest file does not use its canonical name"
        )
    asset_payload = _read_regular_file(
        asset, label="runtime evidence asset", limit=MAX_ASSET_BYTES
    )
    observed_digest = hashlib.sha256(asset_payload).hexdigest()
    if observed_digest != expected_asset_sha256:
        raise RuntimeReleaseAssetHandoffError("runtime evidence asset digest changed")
    digest_payload = _read_regular_file(
        digest_file,
        label="runtime evidence digest file",
        limit=MAX_DIGEST_FILE_BYTES,
    )
    expected_digest_payload = f"{expected_asset_sha256}  {expected_filename}\n".encode(
        "ascii"
    )
    if digest_payload != expected_digest_payload:
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence digest file is not canonical"
        )
    validation = runtime_release_evidence.validate_asset(
        asset,
        expected_source_commit=expected_source_commit,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_providers=expected_providers,
        expected_qualifications=expected_qualifications,
        require_behavioral_claim=require_behavioral_claim,
        expected_asset_sha256=expected_asset_sha256,
    )
    qualification_count = validation.get("qualification_count")
    behavioral_claim_count = validation.get("behavioral_claim_count")
    if not isinstance(qualification_count, int) or not isinstance(
        behavioral_claim_count, int
    ):
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence validation omitted receipt counts"
        )
    return _summary(
        release_tag=release_tag,
        asset_filename=expected_filename,
        asset_sha256=expected_asset_sha256,
        source_commit=expected_source_commit,
        source_archive_sha256=expected_source_archive_sha256,
        asset_size=len(asset_payload),
        qualification_count=qualification_count,
        behavioral_claim_count=behavioral_claim_count,
    )


def stage_handoff(
    *,
    source_asset: Path,
    output_dir: Path,
    release_tag: str,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
    expected_asset_sha256: str,
    expected_providers: frozenset[str],
    expected_qualifications: frozenset[str],
    require_behavioral_claim: bool,
) -> dict[str, object]:
    """Copy verified bytes into a non-overwriting, digest-bound handoff pair."""

    expected_filename = _asset_filename(
        release_tag=release_tag,
        source_commit=expected_source_commit,
        asset_sha256=expected_asset_sha256,
    )
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeReleaseAssetHandoffError(
            "output directory must be an existing non-symlink directory"
        )
    source_payload = _read_regular_file(
        source_asset, label="runtime evidence source asset", limit=MAX_ASSET_BYTES
    )
    if hashlib.sha256(source_payload).hexdigest() != expected_asset_sha256:
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence source asset digest does not match"
        )
    asset = output_dir / expected_filename
    digest_file = output_dir / _digest_filename(expected_filename)
    created_asset = False
    created_digest = False
    try:
        _write_exclusive(asset, source_payload, label="staged runtime evidence asset")
        created_asset = True
        digest_payload = f"{expected_asset_sha256}  {expected_filename}\n".encode(
            "ascii"
        )
        _write_exclusive(
            digest_file, digest_payload, label="staged runtime evidence digest file"
        )
        created_digest = True
        return verify_handoff(
            asset=asset,
            digest_file=digest_file,
            release_tag=release_tag,
            expected_source_commit=expected_source_commit,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_asset_sha256=expected_asset_sha256,
            expected_providers=expected_providers,
            expected_qualifications=expected_qualifications,
            require_behavioral_claim=require_behavioral_claim,
        )
    except Exception:
        if created_digest:
            digest_file.unlink(missing_ok=True)
        if created_asset:
            asset.unlink(missing_ok=True)
        raise


def _run_gh(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["gh", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeReleaseAssetHandoffError(
            "GitHub release command could not be executed"
        ) from exc


def _parse_gh_object(
    completed: subprocess.CompletedProcess[str], *, label: str
) -> dict[str, object]:
    if completed.returncode != 0:
        raise RuntimeReleaseAssetHandoffError(f"{label} could not be verified")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeReleaseAssetHandoffError(
            f"{label} returned invalid metadata"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeReleaseAssetHandoffError(f"{label} returned invalid metadata")
    return value


def _resolve_remote_tag_commit(*, repository: str, release_tag: str) -> str:
    record = _parse_gh_object(
        _run_gh(["api", f"repos/{repository}/git/ref/tags/{release_tag}"]),
        label="the remote release tag",
    )
    for _ in range(8):
        target = record.get("object")
        if not isinstance(target, dict):
            break
        target_type = target.get("type")
        target_sha = target.get("sha")
        if not isinstance(target_sha, str) or _GIT_COMMIT.fullmatch(target_sha) is None:
            break
        if target_type == "commit":
            return target_sha
        if target_type != "tag":
            break
        record = _parse_gh_object(
            _run_gh(["api", f"repos/{repository}/git/tags/{target_sha}"]),
            label="the remote annotated release tag",
        )
    raise RuntimeReleaseAssetHandoffError(
        "the remote release tag does not resolve to one commit"
    )


def _release_record(*, repository: str, release_tag: str) -> dict[str, object]:
    return _parse_gh_object(
        _run_gh(
            [
                "release",
                "view",
                release_tag,
                "--repo",
                repository,
                "--json",
                "tagName,isDraft,assets",
            ]
        ),
        label="the existing GitHub Release",
    )


def _release_assets(record: dict[str, object]) -> list[dict[str, object]]:
    assets = record.get("assets")
    if not isinstance(assets, list) or any(
        not isinstance(item, dict) for item in assets
    ):
        raise RuntimeReleaseAssetHandoffError(
            "the existing GitHub Release returned invalid asset metadata"
        )
    return assets


def upload_handoff(
    *,
    asset: Path,
    digest_file: Path,
    repository: str,
    release_tag: str,
    expected_release_commit: str,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
    expected_asset_sha256: str,
    expected_providers: frozenset[str],
    expected_qualifications: frozenset[str],
    require_behavioral_claim: bool,
) -> dict[str, object]:
    """Attach a verified handoff pair to an existing published GitHub Release."""

    if _REPOSITORY.fullmatch(repository) is None:
        raise RuntimeReleaseAssetHandoffError(
            "repository must use the public OWNER/NAME form"
        )
    summary = verify_handoff(
        asset=asset,
        digest_file=digest_file,
        release_tag=release_tag,
        expected_source_commit=expected_source_commit,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_asset_sha256=expected_asset_sha256,
        expected_providers=expected_providers,
        expected_qualifications=expected_qualifications,
        require_behavioral_claim=require_behavioral_claim,
    )
    if _GIT_COMMIT.fullmatch(expected_release_commit) is None:
        raise RuntimeReleaseAssetHandoffError(
            "release commit must be a full lowercase commit"
        )
    if (
        _resolve_remote_tag_commit(repository=repository, release_tag=release_tag)
        != expected_release_commit
    ):
        raise RuntimeReleaseAssetHandoffError(
            "the remote release tag does not match the expected release commit"
        )
    summary["release_commit"] = expected_release_commit
    release_record = _release_record(repository=repository, release_tag=release_tag)
    if (
        not isinstance(release_record, dict)
        or release_record.get("tagName") != release_tag
        or release_record.get("isDraft") is not False
    ):
        raise RuntimeReleaseAssetHandoffError(
            "runtime evidence requires an existing published GitHub Release"
        )
    filenames = {summary["asset_filename"], summary["digest_filename"]}
    if any(item.get("name") in filenames for item in _release_assets(release_record)):
        raise RuntimeReleaseAssetHandoffError(
            "the immutable runtime evidence release assets already exist"
        )
    uploaded = _run_gh(
        [
            "release",
            "upload",
            release_tag,
            str(asset),
            str(digest_file),
            "--repo",
            repository,
        ]
    )
    if uploaded.returncode != 0:
        raise RuntimeReleaseAssetHandoffError(
            "GitHub release asset upload failed without replacing existing assets"
        )
    published = _release_record(repository=repository, release_tag=release_tag)
    published_assets = {
        item.get("name"): item
        for item in _release_assets(published)
        if isinstance(item.get("name"), str)
    }
    asset_record = published_assets.get(summary["asset_filename"])
    digest_record = published_assets.get(summary["digest_filename"])
    digest_payload = _read_regular_file(
        digest_file,
        label="runtime evidence digest file",
        limit=MAX_DIGEST_FILE_BYTES,
    )
    if (
        asset_record is None
        or digest_record is None
        or asset_record.get("size") != summary["asset_size"]
        or digest_record.get("size") != len(digest_payload)
    ):
        raise RuntimeReleaseAssetHandoffError(
            "uploaded runtime evidence asset metadata does not match the handoff"
        )
    remote_digest = asset_record.get("digest")
    if remote_digest != f"sha256:{expected_asset_sha256}":
        raise RuntimeReleaseAssetHandoffError(
            "uploaded runtime evidence asset digest does not match the handoff"
        )
    expected_digest_file_sha256 = hashlib.sha256(digest_payload).hexdigest()
    remote_digest_file_digest = digest_record.get("digest")
    if remote_digest_file_digest != f"sha256:{expected_digest_file_sha256}":
        raise RuntimeReleaseAssetHandoffError(
            "uploaded runtime evidence digest file does not match the handoff"
        )
    summary["status"] = "uploaded"
    return summary


def _add_verification_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-asset-sha256", required=True)
    parser.add_argument(
        "--expected-provider",
        action="append",
        choices=SUPPORTED_PROVIDERS,
        required=True,
    )
    parser.add_argument(
        "--expected-qualification",
        action="append",
        required=True,
        metavar="PROVIDER[:NAME]",
    )
    parser.add_argument("--require-behavioral-claim", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="Stage an immutable asset handoff")
    stage.add_argument("--asset", required=True, type=Path)
    stage.add_argument("--output-dir", required=True, type=Path)
    _add_verification_arguments(stage)
    verify = commands.add_parser("verify", help="Verify an immutable asset handoff")
    verify.add_argument("--asset", required=True, type=Path)
    verify.add_argument("--digest-file", required=True, type=Path)
    _add_verification_arguments(verify)
    upload = commands.add_parser(
        "upload", help="Upload to an existing published GitHub Release"
    )
    upload.add_argument("--asset", required=True, type=Path)
    upload.add_argument("--digest-file", required=True, type=Path)
    upload.add_argument("--repository", required=True)
    upload.add_argument("--expected-release-commit", required=True)
    _add_verification_arguments(upload)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    providers = frozenset(args.expected_provider)
    qualifications = frozenset(args.expected_qualification)
    try:
        if args.command == "stage":
            result = stage_handoff(
                source_asset=args.asset,
                output_dir=args.output_dir,
                release_tag=args.release_tag,
                expected_source_commit=args.expected_source_commit,
                expected_source_archive_sha256=args.expected_source_archive_sha256,
                expected_asset_sha256=args.expected_asset_sha256,
                expected_providers=providers,
                expected_qualifications=qualifications,
                require_behavioral_claim=args.require_behavioral_claim,
            )
        elif args.command == "verify":
            result = verify_handoff(
                asset=args.asset,
                digest_file=args.digest_file,
                release_tag=args.release_tag,
                expected_source_commit=args.expected_source_commit,
                expected_source_archive_sha256=args.expected_source_archive_sha256,
                expected_asset_sha256=args.expected_asset_sha256,
                expected_providers=providers,
                expected_qualifications=qualifications,
                require_behavioral_claim=args.require_behavioral_claim,
            )
        else:
            result = upload_handoff(
                asset=args.asset,
                digest_file=args.digest_file,
                repository=args.repository,
                release_tag=args.release_tag,
                expected_release_commit=args.expected_release_commit,
                expected_source_commit=args.expected_source_commit,
                expected_source_archive_sha256=args.expected_source_archive_sha256,
                expected_asset_sha256=args.expected_asset_sha256,
                expected_providers=providers,
                expected_qualifications=qualifications,
                require_behavioral_claim=args.require_behavioral_claim,
            )
    except (
        RuntimeReleaseAssetHandoffError,
        runtime_release_evidence.RuntimeReleaseEvidenceError,
    ) as exc:
        parser.error(str(exc))
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
