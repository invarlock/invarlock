#!/usr/bin/env python3
"""Validate first-party distributions against their exact source trees."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.release.release_distribution_validation import (
        DistributionValidationSpec,
        ReleasePreflightError,
        read_distribution_project,
        validate_distribution_pair,
    )
except ImportError:  # pragma: no cover - direct script execution
    from release_distribution_validation import (  # type: ignore[import-not-found, no-redef]
        DistributionValidationSpec,
        ReleasePreflightError,
        read_distribution_project,
        validate_distribution_pair,
    )

_ADDIN_PROJECTS = {
    "diagnostics": "invarlock_addins/diagnostics",
    "gguf": "invarlock_addins/gguf",
    "multimodal": "invarlock_addins/multimodal",
    "tensorrt_llm": "invarlock_addins/tensorrt_llm",
}


@dataclass(frozen=True)
class FirstPartyDistribution:
    project: str
    distribution: str
    version: str
    wheel: str
    sdist: str


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _real_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical.is_symlink():
        raise ReleasePreflightError(f"{label} must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ReleasePreflightError(f"{label} is missing") from exc
    if resolved != lexical or not resolved.is_dir():
        raise ReleasePreflightError(f"{label} must be one real directory")
    return resolved


def _contained_distribution_directory(
    repo_root: Path,
    value: Path,
    *,
    label: str = "first-party distribution directory",
) -> Path:
    candidate = value if value.is_absolute() else repo_root / value
    resolved = _real_directory(candidate, label=label)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ReleasePreflightError(f"{label} must remain inside the checkout") from exc
    return resolved


def _validate_artifact_directory(
    dist_dir: Path, *, expected_pairs: int, label: str
) -> Path:
    resolved = _real_directory(dist_dir, label=label)
    entries = sorted(resolved.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in entries):
        raise ReleasePreflightError(f"{label} must not contain symbolic links")
    wheels = [path for path in entries if path.name.endswith(".whl")]
    sdists = [path for path in entries if path.name.endswith(".tar.gz")]
    if len(wheels) != expected_pairs or len(sdists) != expected_pairs:
        raise ReleasePreflightError(
            f"{label} must contain {expected_pairs} wheel/sdist pair(s)"
        )
    return resolved


def _artifact_pair(*, dist_dir: Path, distribution_name: str) -> tuple[Path, Path]:
    normalized = _normalized_distribution(distribution_name)
    wheel_prefix = normalized.replace("-", "_") + "-"
    wheels = [
        path
        for path in dist_dir.iterdir()
        if path.name.endswith(".whl") and path.name.startswith(wheel_prefix)
    ]
    sdists = [
        path
        for path in dist_dir.iterdir()
        if path.name.endswith(".tar.gz")
        and _normalized_distribution(path.name.removesuffix(".tar.gz")).startswith(
            normalized + "-"
        )
    ]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleasePreflightError(f"{distribution_name} artifact pair is ambiguous")
    return wheels[0], sdists[0]


def validate_first_party_addin_distributions(
    *, repo_root: Path, expected_version: str, dist_dir: Path
) -> list[FirstPartyDistribution]:
    dist_dir = _validate_artifact_directory(
        dist_dir,
        expected_pairs=len(_ADDIN_PROJECTS),
        label="first-party add-in distribution directory",
    )
    results: list[FirstPartyDistribution] = []
    for project, package_path in _ADDIN_PROJECTS.items():
        project_root = repo_root / "addins" / project
        name, version = read_distribution_project(project_root)
        if version != expected_version:
            raise ReleasePreflightError(
                f"{project} add-in version does not match the release"
            )
        wheel, sdist = _artifact_pair(dist_dir=dist_dir, distribution_name=name)
        validate_distribution_pair(
            DistributionValidationSpec(
                project_root=project_root,
                distribution_name=name,
                version=version,
                package_path=package_path,
            ),
            wheel=wheel,
            sdist=sdist,
        )
        results.append(
            FirstPartyDistribution(
                project=project,
                distribution=name,
                version=version,
                wheel=wheel.name,
                sdist=sdist.name,
            )
        )
    return results


def validate_first_party_distributions(
    *,
    repo_root: Path,
    expected_version: str,
    core_dist_dir: Path,
    addin_dist_dir: Path,
) -> list[FirstPartyDistribution]:
    """Validate the core and every maintained first-party add-in pair."""

    core_dist_dir = _validate_artifact_directory(
        core_dist_dir,
        expected_pairs=1,
        label="core distribution directory",
    )
    core_name, core_version = read_distribution_project(repo_root)
    if core_version != expected_version:
        raise ReleasePreflightError("core version does not match the release")
    wheel, sdist = _artifact_pair(dist_dir=core_dist_dir, distribution_name=core_name)
    validate_distribution_pair(
        DistributionValidationSpec(
            project_root=repo_root,
            distribution_name=core_name,
            version=core_version,
            package_path="invarlock",
        ),
        wheel=wheel,
        sdist=sdist,
    )
    results = [
        FirstPartyDistribution(
            project="core",
            distribution=core_name,
            version=core_version,
            wheel=wheel.name,
            sdist=sdist.name,
        )
    ]
    results.extend(
        validate_first_party_addin_distributions(
            repo_root=repo_root,
            expected_version=expected_version,
            dist_dir=addin_dist_dir,
        )
    )
    if len({result.distribution for result in results}) != len(results):
        raise ReleasePreflightError("first-party distribution names must be unique")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-version")
    parser.add_argument("--core-dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--addin-dist-dir",
        "--dist-dir",
        dest="addin_dist_dir",
        type=Path,
        default=Path("dist/addins"),
    )
    arguments = parser.parse_args(argv)
    repo_root = _real_directory(arguments.repo_root, label="release checkout")
    try:
        core_dist_dir = _contained_distribution_directory(
            repo_root,
            arguments.core_dist_dir,
            label="core distribution directory",
        )
        addin_dist_dir = _contained_distribution_directory(
            repo_root,
            arguments.addin_dist_dir,
            label="first-party add-in distribution directory",
        )
        expected_version = (
            arguments.expected_version or read_distribution_project(repo_root)[1]
        )
        results = validate_first_party_distributions(
            repo_root=repo_root,
            expected_version=expected_version,
            core_dist_dir=core_dist_dir,
            addin_dist_dir=addin_dist_dir,
        )
    except ReleasePreflightError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "format_version": "invarlock/distribution-validation-v1",
                "ok": True,
                "distributions": [result.__dict__ for result in results],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
