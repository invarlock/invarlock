#!/usr/bin/env python3
"""Authenticate and verify one exact tagged release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlsplit

WORKFLOW_PATH = ".github/workflows/release.yml"
MAX_JSON_BYTES = 1024 * 1024
_MAX_LEDGER_BYTES = 32 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_TAG = re.compile(
    r"^v[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z][0-9A-Za-z.-]*)?$"
)
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z][0-9A-Za-z.-]*)?$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RUN_ID = re.compile(r"^[1-9][0-9]*$")


class CandidateError(RuntimeError):
    """A tagged-candidate identity or integrity check failed closed."""


class UrlResponse(Protocol):
    """Bounded response surface used by the authenticated API reader."""

    def __enter__(self) -> UrlResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None: ...

    def read(self, amount: int = -1) -> bytes: ...


UrlOpener = Callable[..., UrlResponse]


def _validated_release_sha(value: str) -> str:
    if _RELEASE_SHA.fullmatch(value) is None:
        raise CandidateError("release SHA is malformed")
    return value


def _validated_release_tag(value: str) -> str:
    if _RELEASE_TAG.fullmatch(value) is None:
        raise CandidateError("release tag is malformed")
    return value


def _validated_run_id(value: str) -> int:
    if _RUN_ID.fullmatch(value) is None:
        raise CandidateError("workflow run ID is malformed")
    return int(value)


def _validated_digest(value: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise CandidateError("distribution ledger digest is malformed")
    return value


def _strict_json_object(raw: bytes) -> dict[str, object]:
    def reject_duplicate_keys(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid constant: {value}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CandidateError("tagged workflow run metadata is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise CandidateError("tagged workflow run metadata must be a JSON object")
    return parsed


def authenticate_tagged_run(
    *,
    release_sha: str,
    release_tag: str,
    candidate_run_id: str,
    repository: str,
    api_url: str,
    token: str,
    opener: UrlOpener | None = None,
) -> int:
    """Authenticate one successful tag-push run at the exact release commit."""

    expected_sha = _validated_release_sha(release_sha)
    expected_tag = _validated_release_tag(release_tag)
    expected_run_id = _validated_run_id(candidate_run_id)
    if _REPOSITORY.fullmatch(repository) is None:
        raise CandidateError("repository identity is malformed")
    parsed_api = urlsplit(api_url)
    if (
        parsed_api.scheme != "https"
        or not parsed_api.netloc
        or parsed_api.username is not None
        or parsed_api.password is not None
        or parsed_api.query
        or parsed_api.fragment
    ):
        raise CandidateError("GitHub API URL is malformed")
    if not token:
        raise CandidateError("workflow run authentication is unavailable")

    selected_opener = opener if opener is not None else urllib.request.urlopen
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/repos/{repository}/actions/runs/{expected_run_id}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with selected_opener(request, timeout=30) as response:
            raw_run = response.read(MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise CandidateError("unable to authenticate the tagged workflow run") from exc
    if len(raw_run) > MAX_JSON_BYTES:
        raise CandidateError("tagged workflow run metadata is too large")
    run = _strict_json_object(raw_run)
    workflow_path = str(run.get("path", "")).split("@", 1)[0]
    if (
        run.get("id") != expected_run_id
        or run.get("event") != "push"
        or run.get("conclusion") != "success"
        or run.get("head_sha") != expected_sha
        or run.get("head_branch") != expected_tag
        or workflow_path != WORKFLOW_PATH
    ):
        raise CandidateError(
            "candidate did not come from a successful exact-tag release run"
        )
    return expected_run_id


def expected_distribution_paths(version: str) -> set[str]:
    """Return the closed coordinated-distribution filename set for a version."""

    if _VERSION.fullmatch(version) is None:
        raise CandidateError("release version is malformed")
    projects = (
        ("", "invarlock"),
        ("addins/", "invarlock_diagnostics"),
        ("addins/", "invarlock_runtime_gguf"),
        ("addins/", "invarlock_runtime_hf_vision_text"),
        ("addins/", "invarlock_runtime_tensorrt_llm"),
    )
    return {
        path
        for directory, project in projects
        for path in (
            f"{directory}{project}-{version}-py3-none-any.whl",
            f"{directory}{project}-{version}.tar.gz",
        )
    }


def _regular_file_bytes(path: Path, *, label: str, size_limit: int) -> bytes:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > size_limit:
            raise CandidateError(f"{label} must be one bounded regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise CandidateError(f"{label} is unavailable") from exc
    if len(raw) != metadata.st_size:
        raise CandidateError(f"{label} changed while it was read")
    return raw


def _hash_regular_file(path: Path) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise CandidateError(f"distribution must be a regular file: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            observed = os.fstat(handle.fileno())
    except OSError as exc:
        raise CandidateError(f"distribution is unavailable: {path.name}") from exc
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    ):
        raise CandidateError(f"distribution changed while it was read: {path.name}")
    return digest.hexdigest()


def verify_distribution_ledger(dist_dir: Path, release_tag: str) -> str:
    """Verify the closed ten-file archive set and return its ledger digest."""

    version = _validated_release_tag(release_tag).removeprefix("v")
    expected_paths = expected_distribution_paths(version)
    ledger_path = dist_dir / "SHA256SUMS"
    raw_ledger = _regular_file_bytes(
        ledger_path,
        label="distribution digest ledger",
        size_limit=_MAX_LEDGER_BYTES,
    )
    try:
        lines = raw_ledger.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CandidateError("distribution digest ledger is malformed") from exc
    entries: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if not separator or _DIGEST.fullmatch(digest) is None or not relative:
            raise CandidateError("distribution digest ledger is malformed")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or "." in parsed.parts
            or "\\" in relative
            or str(parsed) != relative
        ):
            raise CandidateError("distribution digest ledger contains an unsafe path")
        if relative in entries:
            raise CandidateError("distribution digest ledger contains a duplicate path")
        entries[relative] = digest
    if set(entries) != expected_paths:
        raise CandidateError("distribution digest ledger has the wrong file set")

    actual_files = {
        path.relative_to(dist_dir).as_posix()
        for path in dist_dir.rglob("*")
        if not path.is_dir()
    }
    if actual_files != expected_paths | {"SHA256SUMS"}:
        raise CandidateError("candidate artifact has the wrong file set")
    for relative, expected_digest in sorted(entries.items()):
        if _hash_regular_file(dist_dir / relative) != expected_digest:
            raise CandidateError(f"distribution digest mismatch: {relative}")
    return hashlib.sha256(raw_ledger).hexdigest()


def write_run_output(path: Path, run_id: int) -> None:
    """Append one validated artifact-run output for later workflow steps."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"artifact_run_id={run_id}\n")


def write_ledger_output(path: Path, digest: str) -> None:
    """Append one validated ledger output for later workflow steps."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"dist_ledger_sha256={_validated_digest(digest)}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authenticate = subparsers.add_parser("authenticate")
    authenticate.add_argument("--release-sha", required=True)
    authenticate.add_argument("--release-tag", required=True)
    authenticate.add_argument("--candidate-run-id", required=True)
    authenticate.add_argument("--repository", required=True)
    authenticate.add_argument("--api-url", required=True)
    authenticate.add_argument("--github-output", type=Path, required=True)

    ledger = subparsers.add_parser("verify-ledger")
    ledger.add_argument("--dist-dir", type=Path, required=True)
    ledger.add_argument("--release-tag", required=True)
    ledger.add_argument("--github-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate the run identity or verify its downloaded artifact ledger."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "authenticate":
            run_id = authenticate_tagged_run(
                release_sha=args.release_sha,
                release_tag=args.release_tag,
                candidate_run_id=args.candidate_run_id,
                repository=args.repository,
                api_url=args.api_url,
                token=os.environ.get("GITHUB_TOKEN", ""),
            )
            write_run_output(args.github_output, run_id)
        else:
            digest = verify_distribution_ledger(args.dist_dir, args.release_tag)
            write_ledger_output(args.github_output, digest)
        return 0
    except (CandidateError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
