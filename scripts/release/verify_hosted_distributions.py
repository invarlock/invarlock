#!/usr/bin/env python3
"""Verify that hosted release files are byte-identical to the build ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

PROJECTS = (
    "invarlock",
    "invarlock-diagnostics",
    "invarlock-runtime-gguf",
    "invarlock-runtime-hf-vision-text",
    "invarlock-runtime-tensorrt-llm",
)
API_ROOTS = {
    "pypi": "https://pypi.org/pypi",
    "testpypi": "https://test.pypi.org/pypi",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*")
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_DISTRIBUTION_BYTES = 1024 * 1024 * 1024


class HostedDistributionVerificationError(RuntimeError):
    """Raised when hosted release files do not match the build ledger."""


def _normalized_project_prefix(project: str) -> str:
    return re.sub(r"[-_.]+", "_", project).lower() + "-"


def _parse_build_ledger(
    ledger_path: Path, *, expected_ledger_sha256: str
) -> dict[str, dict[str, str]]:
    if _SHA256_PATTERN.fullmatch(expected_ledger_sha256) is None:
        raise HostedDistributionVerificationError(
            "build distribution ledger digest is malformed"
        )
    try:
        ledger_bytes = ledger_path.read_bytes()
    except OSError as exc:
        raise HostedDistributionVerificationError(
            f"cannot read distribution digest ledger: {ledger_path}"
        ) from exc
    if hashlib.sha256(ledger_bytes).hexdigest() != expected_ledger_sha256:
        raise HostedDistributionVerificationError(
            "distribution digest ledger changed after build"
        )

    expected: dict[str, str] = {}
    try:
        lines = ledger_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HostedDistributionVerificationError(
            "distribution digest ledger is not UTF-8"
        ) from exc
    for line in lines:
        digest, separator, relative = line.partition("  ")
        relative_path = PurePosixPath(relative)
        if (
            not separator
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.name
        ):
            raise HostedDistributionVerificationError(
                "distribution digest ledger is malformed"
            )
        name = relative_path.name
        if name in expected:
            raise HostedDistributionVerificationError(
                f"duplicate distribution filename in ledger: {name}"
            )
        expected[name] = digest
    if len(expected) != len(PROJECTS) * 2:
        raise HostedDistributionVerificationError(
            f"expected {len(PROJECTS) * 2} first-party distribution files, "
            f"found {len(expected)}"
        )

    expected_by_project: dict[str, dict[str, str]] = {}
    assigned: set[str] = set()
    for project in PROJECTS:
        prefix = _normalized_project_prefix(project)
        project_files = {
            name: digest
            for name, digest in expected.items()
            if name.lower().startswith(prefix)
        }
        wheels = [name for name in project_files if name.endswith(".whl")]
        sdists = [name for name in project_files if name.endswith(".tar.gz")]
        if len(project_files) != 2 or len(wheels) != 1 or len(sdists) != 1:
            raise HostedDistributionVerificationError(
                f"expected one wheel and one source archive for {project}, "
                f"found {sorted(project_files)}"
            )
        if set(project_files) & assigned:
            raise HostedDistributionVerificationError(
                f"distribution filename assigned twice: {project}"
            )
        assigned.update(project_files)
        expected_by_project[project] = project_files
    if assigned != set(expected):
        raise HostedDistributionVerificationError(
            "distribution ledger contains files outside the release project set"
        )
    return expected_by_project


def _open_url(url: str, *, timeout: float) -> BinaryIO:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "invarlock-release-verifier/1"},
    )
    return cast(
        BinaryIO,
        urllib.request.urlopen(request, timeout=timeout),  # noqa: S310
    )


def _verify_project(
    *,
    api_root: str,
    project: str,
    version: str,
    expected: Mapping[str, str],
    timeout: float,
    wheel_destination: Path | None = None,
) -> None:
    metadata_url = f"{api_root}/{project}/{version}/json"
    with _open_url(metadata_url, timeout=timeout) as response:
        metadata_bytes = response.read(_MAX_METADATA_BYTES + 1)
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise HostedDistributionVerificationError(
            f"hosted metadata is too large for {project}"
        )
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict):
        raise HostedDistributionVerificationError(
            f"hosted metadata is malformed for {project}"
        )
    urls = metadata.get("urls")
    if not isinstance(urls, list):
        raise HostedDistributionVerificationError(
            f"hosted metadata has no URL list for {project}"
        )
    hosted: dict[str, Mapping[str, object]] = {}
    for entry in urls:
        if not isinstance(entry, dict):
            raise HostedDistributionVerificationError(
                f"hosted metadata has malformed files for {project}"
            )
        filename = entry.get("filename")
        if not isinstance(filename, str) or filename in hosted:
            raise HostedDistributionVerificationError(
                f"hosted metadata has malformed or duplicate files for {project}"
            )
        hosted[filename] = entry
    if set(hosted) != set(expected):
        raise HostedDistributionVerificationError(
            f"hosted filename set differs for {project}: "
            f"expected={sorted(expected)} observed={sorted(hosted)}"
        )

    for name, expected_digest in sorted(expected.items()):
        entry = hosted[name]
        digests = entry.get("digests")
        declared_digest = digests.get("sha256") if isinstance(digests, dict) else None
        if declared_digest != expected_digest:
            raise HostedDistributionVerificationError(
                f"hosted metadata digest differs for {name}"
            )
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise HostedDistributionVerificationError(
                f"hosted download URL is invalid for {name}"
            )
        observed = hashlib.sha256()
        total = 0
        destination = (
            wheel_destination / name
            if wheel_destination is not None and name.endswith(".whl")
            else None
        )
        handle = destination.open("xb") if destination is not None else None
        try:
            if handle is not None:
                os.chmod(destination, 0o600)
            with _open_url(url, timeout=timeout) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_DISTRIBUTION_BYTES:
                        raise HostedDistributionVerificationError(
                            f"hosted artifact is too large for {name}"
                        )
                    observed.update(chunk)
                    if handle is not None:
                        handle.write(chunk)
        except BaseException:
            if handle is not None:
                handle.close()
                destination.unlink(missing_ok=True)
            raise
        else:
            if handle is not None:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    handle.close()
        if observed.hexdigest() != expected_digest:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise HostedDistributionVerificationError(
                f"hosted artifact bytes differ for {name}"
            )


def _wheelhouse_destination(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if os.path.lexists(absolute):
        raise HostedDistributionVerificationError(
            f"wheel destination already exists: {absolute}"
        )
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise HostedDistributionVerificationError(
            f"wheel destination parent is unavailable: {absolute.parent}"
        ) from exc
    if parent.is_symlink() or not parent.is_dir():
        raise HostedDistributionVerificationError(
            f"wheel destination parent is not a real directory: {absolute.parent}"
        )
    return parent / absolute.name


def _publish_wheelhouse(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise HostedDistributionVerificationError(
            f"wheel destination already exists: {destination}"
        )
    created = False
    try:
        destination.mkdir(mode=0o700)
        created = True
        for wheel in sorted(source.iterdir()):
            if not wheel.is_file() or wheel.suffix != ".whl":
                raise HostedDistributionVerificationError(
                    "verified wheel staging contains an unexpected entry"
                )
            target = destination / wheel.name
            os.link(wheel, target, follow_symlinks=False)
            target.chmod(0o600)
    except (HostedDistributionVerificationError, OSError) as exc:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        if isinstance(exc, HostedDistributionVerificationError):
            raise
        raise HostedDistributionVerificationError(
            "could not publish the verified wheel destination"
        ) from exc


def verify_hosted_distributions(
    *,
    ledger_path: Path,
    expected_ledger_sha256: str,
    target: str,
    version: str,
    attempts: int = 12,
    retry_delay: float = 10.0,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    wheelhouse: Path | None = None,
    projects: Sequence[str] | None = None,
) -> None:
    """Verify hosted metadata and bytes against one authenticated build ledger."""
    if target not in API_ROOTS:
        raise HostedDistributionVerificationError(
            f"unsupported publish target: {target}"
        )
    normalized_version = version.removeprefix("v")
    if _VERSION_PATTERN.fullmatch(normalized_version) is None:
        raise HostedDistributionVerificationError("release version is malformed")
    if attempts < 1 or retry_delay < 0 or timeout <= 0:
        raise HostedDistributionVerificationError("retry configuration is invalid")
    selected_projects = PROJECTS if projects is None else tuple(projects)
    if (
        not selected_projects
        or len(set(selected_projects)) != len(selected_projects)
        or not set(selected_projects).issubset(PROJECTS)
    ):
        raise HostedDistributionVerificationError("hosted project selection is invalid")
    expected = _parse_build_ledger(
        ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    wheel_destination = (
        _wheelhouse_destination(wheelhouse) if wheelhouse is not None else None
    )

    for attempt in range(1, attempts + 1):
        temporary: Path | None = None
        try:
            if wheel_destination is not None:
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=".invarlock-hosted-wheels-",
                        dir=wheel_destination.parent,
                    )
                )
            for project in selected_projects:
                _verify_project(
                    api_root=API_ROOTS[target],
                    project=project,
                    version=normalized_version,
                    expected=expected[project],
                    timeout=timeout,
                    wheel_destination=temporary,
                )
            if temporary is not None and wheel_destination is not None:
                _publish_wheelhouse(temporary, wheel_destination)
            return
        except (
            HostedDistributionVerificationError,
            OSError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            if attempt == attempts:
                raise HostedDistributionVerificationError(
                    f"hosted release verification failed after {attempts} attempts: {exc}"
                ) from exc
            print(
                "hosted release is not yet byte-identical "
                f"(attempt {attempt}/{attempts}): {exc}"
            )
            sleep(retry_delay)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--target", choices=tuple(API_ROOTS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--project", action="append", choices=PROJECTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verify_hosted_distributions(
            ledger_path=args.ledger,
            expected_ledger_sha256=args.expected_ledger_sha256,
            target=args.target,
            version=args.version,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
            timeout=args.timeout,
            wheelhouse=args.wheelhouse,
            projects=args.project,
        )
    except HostedDistributionVerificationError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"{args.target} release {args.version.removeprefix('v')} matches build ledger"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
