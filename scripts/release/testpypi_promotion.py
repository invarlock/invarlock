#!/usr/bin/env python3
"""Record and authenticate one exact TestPyPI promotion candidate."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

FORMAT_VERSION = "invarlock/testpypi-promotion-v1"
WORKFLOW_PATH = ".github/workflows/release.yml"
_MANIFEST_FIELDS = {
    "dist_ledger_sha256",
    "format_version",
    "release_sha",
    "release_tag",
    "source_run_id",
    "target",
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_TAG = re.compile(
    r"^v[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z][0-9A-Za-z.-]*)?$"
)
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RUN_ID = re.compile(r"^[1-9][0-9]*$")
_MAX_JSON_BYTES = 1024 * 1024


class PromotionError(RuntimeError):
    """One fail-closed promotion-contract or authentication failure."""


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


@dataclass(frozen=True)
class PromotionAuthorization:
    """Authenticated artifact run and distribution-ledger identity."""

    artifact_run_id: int
    dist_ledger_sha256: str


def _validated_digest(value: str, *, label: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise PromotionError(f"{label} is malformed")
    return value


def _validated_release_sha(value: str) -> str:
    if _RELEASE_SHA.fullmatch(value) is None:
        raise PromotionError("release SHA is malformed")
    return value


def _validated_release_tag(value: str) -> str:
    if _RELEASE_TAG.fullmatch(value) is None:
        raise PromotionError("release tag is malformed")
    return value


def _validated_run_id(value: str) -> int:
    if _RUN_ID.fullmatch(value) is None:
        raise PromotionError("workflow run ID is malformed")
    return int(value)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(
        items: list[tuple[str, object]],
    ) -> dict[str, object]:
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
        raise PromotionError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return parsed


def _read_bounded_regular(path: Path, *, label: str) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise PromotionError(f"{label} must be one bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_JSON_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_JSON_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise PromotionError(f"{label} is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_JSON_BYTES or len(raw) != before.st_size:
        raise PromotionError(f"{label} changed or exceeded its size limit")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise PromotionError(f"{label} changed while it was read")
    return raw


def _write_new_regular(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PromotionError("promotion output could not be created safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def record_promotion(
    *,
    output: Path,
    release_sha: str,
    release_tag: str,
    dist_ledger_sha256: str,
    source_run_id: str,
) -> dict[str, object]:
    """Create one closed canonical promotion manifest without overwriting files."""

    payload: dict[str, object] = {
        "dist_ledger_sha256": _validated_digest(
            dist_ledger_sha256, label="distribution ledger digest"
        ),
        "format_version": FORMAT_VERSION,
        "release_sha": _validated_release_sha(release_sha),
        "release_tag": _validated_release_tag(release_tag),
        "source_run_id": _validated_run_id(source_run_id),
        "target": "testpypi",
    }
    _write_new_regular(output, _canonical_json(payload))
    return payload


def load_promotion(path: Path) -> dict[str, object]:
    """Load one closed strict promotion manifest."""

    manifest = _strict_json_object(
        _read_bounded_regular(path, label="TestPyPI promotion authorization"),
        label="TestPyPI promotion authorization",
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise PromotionError("TestPyPI promotion authorization contract is invalid")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise PromotionError("TestPyPI promotion authorization version is unsupported")
    return manifest


def current_candidate(
    *, candidate_run_id: str, dist_ledger_sha256: str
) -> PromotionAuthorization:
    """Bind the current fully gated run as the TestPyPI candidate."""

    return PromotionAuthorization(
        artifact_run_id=_validated_run_id(candidate_run_id),
        dist_ledger_sha256=_validated_digest(
            dist_ledger_sha256, label="build distribution ledger digest"
        ),
    )


def authorize_promotion(
    *,
    manifest_path: Path,
    release_sha: str,
    release_tag: str,
    candidate_run_id: str,
    repository: str,
    api_url: str,
    token: str,
    opener: UrlOpener = urllib.request.urlopen,
) -> PromotionAuthorization:
    """Authenticate manifest bindings and the originating GitHub workflow run."""

    expected_sha = _validated_release_sha(release_sha)
    expected_tag = _validated_release_tag(release_tag)
    expected_run_id = _validated_run_id(candidate_run_id)
    if _REPOSITORY.fullmatch(repository) is None:
        raise PromotionError("repository identity is malformed")
    parsed_api = urlsplit(api_url)
    if (
        parsed_api.scheme != "https"
        or not parsed_api.netloc
        or parsed_api.username is not None
        or parsed_api.password is not None
        or parsed_api.query
        or parsed_api.fragment
    ):
        raise PromotionError("GitHub API URL is malformed")
    if not token:
        raise PromotionError("workflow run authentication is unavailable")

    manifest = load_promotion(manifest_path)
    if manifest.get("release_sha") != expected_sha:
        raise PromotionError("TestPyPI promotion commit differs from this release")
    if manifest.get("release_tag") != expected_tag:
        raise PromotionError("TestPyPI promotion tag differs from this release")
    if manifest.get("source_run_id") != expected_run_id:
        raise PromotionError("TestPyPI promotion run identity is inconsistent")
    if manifest.get("target") != "testpypi":
        raise PromotionError("production requires a TestPyPI promotion")
    digest = manifest.get("dist_ledger_sha256")
    if not isinstance(digest, str):
        raise PromotionError("TestPyPI promotion ledger digest is malformed")
    digest = _validated_digest(digest, label="TestPyPI promotion ledger digest")

    request = urllib.request.Request(
        (f"{api_url.rstrip('/')}/repos/{repository}/actions/runs/{expected_run_id}"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with opener(request, timeout=30) as response:
            raw_run = response.read(_MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise PromotionError(
            "unable to authenticate the TestPyPI workflow run"
        ) from exc
    if len(raw_run) > _MAX_JSON_BYTES:
        raise PromotionError("TestPyPI workflow run metadata is too large")
    run = _strict_json_object(raw_run, label="TestPyPI workflow run metadata")
    workflow_path = str(run.get("path", "")).split("@", 1)[0]
    if (
        run.get("id") != expected_run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("conclusion") != "success"
        or run.get("head_sha") != expected_sha
        or workflow_path != WORKFLOW_PATH
    ):
        raise PromotionError(
            "TestPyPI promotion did not come from a successful exact-commit release run"
        )
    return PromotionAuthorization(
        artifact_run_id=expected_run_id,
        dist_ledger_sha256=digest,
    )


def write_github_outputs(path: Path, authorization: PromotionAuthorization) -> None:
    """Append validated scalar outputs for later workflow steps."""

    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"artifact_run_id={authorization.artifact_run_id}\n")
            handle.write(f"dist_ledger_sha256={authorization.dist_ledger_sha256}\n")
    except OSError as exc:
        raise PromotionError("workflow output is unavailable") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--release-sha", required=True)
    record.add_argument("--release-tag", required=True)
    record.add_argument("--dist-ledger-sha256", required=True)
    record.add_argument("--source-run-id", required=True)

    current = subparsers.add_parser("current")
    current.add_argument("--candidate-run-id", required=True)
    current.add_argument("--dist-ledger-sha256", required=True)
    current.add_argument("--github-output", type=Path, required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--manifest", type=Path, required=True)
    authorize.add_argument("--release-sha", required=True)
    authorize.add_argument("--release-tag", required=True)
    authorize.add_argument("--candidate-run-id", required=True)
    authorize.add_argument("--repository", required=True)
    authorize.add_argument("--api-url", required=True)
    authorize.add_argument("--github-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one promotion recording or authentication command."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "record":
            record_promotion(
                output=args.output,
                release_sha=args.release_sha,
                release_tag=args.release_tag,
                dist_ledger_sha256=args.dist_ledger_sha256,
                source_run_id=args.source_run_id,
            )
            return 0
        if args.command == "current":
            authorization = current_candidate(
                candidate_run_id=args.candidate_run_id,
                dist_ledger_sha256=args.dist_ledger_sha256,
            )
        else:
            authorization = authorize_promotion(
                manifest_path=args.manifest,
                release_sha=args.release_sha,
                release_tag=args.release_tag,
                candidate_run_id=args.candidate_run_id,
                repository=args.repository,
                api_url=args.api_url,
                token=os.environ.get("GITHUB_TOKEN", ""),
            )
        write_github_outputs(args.github_output, authorization)
        return 0
    except PromotionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
