"""Deterministic file helpers shared by public evidence-pack scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationSummary:
    clean_reports: int
    error_injection_reports: int
    expected_failure_reports: int
    failed_reports: int
    policy_profile: str
    report_assurance: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_paths(
    output_root: Path,
    *,
    patterns: Sequence[str],
    exclude_names: Set[str] = frozenset(),
) -> list[str]:
    relpaths: set[Path] = set()
    for pattern in patterns:
        relpaths.update(
            path.relative_to(output_root)
            for path in output_root.glob(pattern)
            if path.is_file() and path.name not in exclude_names
        )
    return [relpath.as_posix() for relpath in sorted(relpaths)]


def write_verification_summary(
    path: Path,
    *,
    summary: VerificationSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, summary.to_payload())
