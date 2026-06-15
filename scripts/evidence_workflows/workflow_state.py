"""Typed run/result/artifact helpers for evidence workflow scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class WorkflowRunMetadata:
    suite: str
    execution_mode: str
    shard_index: int = 0
    shard_count: int = 1


@dataclass(frozen=True)
class WorkflowLaneResult:
    slug: str
    lane_id: str
    model_id: str
    preset: str
    evaluate_exit: int
    verify_exit: int | None
    report_path: str
    verify_path: str | None
    status: str = "failed"
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_summary_entry(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class WorkflowPhaseResult:
    name: str
    returncode: int | None
    status: str
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_summary_entry(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class WorkflowLaneRunState:
    slug: str
    lane_id: str
    model_id: str
    preset: str
    report_path: str
    verify_path: str | None
    phases: tuple[WorkflowPhaseResult, ...]

    @property
    def status(self) -> str:
        if not self.phases:
            return "failed"
        if all(phase.ok for phase in self.phases):
            if any(phase.status == "skipped" for phase in self.phases):
                return "skipped"
            return "ok"
        return "failed"

    @property
    def detail(self) -> str | None:
        for phase in reversed(self.phases):
            if phase.detail:
                return phase.detail
        return None

    def phase_returncode(self, name: str) -> int | None:
        for phase in reversed(self.phases):
            if phase.name == name:
                return phase.returncode
        return None

    def to_lane_result(self) -> WorkflowLaneResult:
        return WorkflowLaneResult(
            slug=self.slug,
            lane_id=self.lane_id,
            model_id=self.model_id,
            preset=self.preset,
            evaluate_exit=self.phase_returncode("evaluate")
            if self.phase_returncode("evaluate") is not None
            else self._first_nonzero_returncode(),
            verify_exit=self.phase_returncode("verify"),
            report_path=self.report_path,
            verify_path=self.verify_path,
            status=self.status,
            detail=self.detail,
        )

    def to_summary_entry(self) -> dict[str, object]:
        lane_result = self.to_lane_result().to_summary_entry()
        lane_result["phases"] = [phase.to_summary_entry() for phase in self.phases]
        return lane_result

    def _first_nonzero_returncode(self) -> int:
        for phase in self.phases:
            if phase.returncode not in {None, 0}:
                return int(phase.returncode)
        return 0


@dataclass(frozen=True)
class WorkflowArtifact:
    path: str
    bytes: int
    sha256: str

    def to_manifest_entry(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowVerificationSummary:
    clean_reports: int
    error_injection_reports: int
    expected_failure_reports: int
    failed_reports: int
    policy_profile: str

    def to_summary_payload(self) -> dict[str, object]:
        return asdict(self)


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


def capture_artifacts(
    output_root: Path,
    *,
    patterns: Sequence[str],
) -> list[dict[str, object]]:
    artifacts: list[WorkflowArtifact] = []
    for relpath in collect_artifact_paths(output_root, patterns=patterns):
        path = output_root / relpath
        artifacts.append(
            WorkflowArtifact(
                path=relpath,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return [artifact.to_manifest_entry() for artifact in artifacts]


def write_summary_files(
    output_root: Path,
    *,
    metadata: WorkflowRunMetadata,
    results: Sequence[WorkflowLaneResult],
) -> None:
    summary_tsv = output_root / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "slug\tlane_id\tstatus\tdetail\tevaluate_exit\tverify_exit\treport\n"
        )
        for result in results:
            verify_exit = (
                "NA" if result.verify_exit is None else str(result.verify_exit)
            )
            handle.write(
                f"{result.slug}\t{result.lane_id}\t{result.status}\t"
                f"{result.detail or ''}\t{result.evaluate_exit}\t"
                f"{verify_exit}\t{result.report_path}\n"
            )

    payload = {
        "suite": metadata.suite,
        "execution_mode": metadata.execution_mode,
        "shard_index": metadata.shard_index,
        "shard_count": metadata.shard_count,
        "ok": all(result.ok for result in results),
        "results": [result.to_summary_entry() for result in results],
    }
    write_json(output_root / "summary.json", payload)


def write_artifact_manifest(
    output_root: Path,
    *,
    schema: str,
    metadata: WorkflowRunMetadata,
    results: Sequence[WorkflowLaneResult],
    artifact_patterns: Sequence[str],
) -> None:
    payload = {
        "schema": schema,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": metadata.suite,
        "execution_mode": metadata.execution_mode,
        "shard_index": metadata.shard_index,
        "shard_count": metadata.shard_count,
        "ok": all(result.ok for result in results),
        "lane_results": [result.to_summary_entry() for result in results],
        "files": capture_artifacts(output_root, patterns=artifact_patterns),
    }
    write_json(output_root / "artifact_manifest.json", payload)


def write_verification_summary(
    path: Path,
    *,
    summary: WorkflowVerificationSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, summary.to_summary_payload())
