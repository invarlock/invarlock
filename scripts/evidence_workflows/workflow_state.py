"""Typed run/result/artifact helpers for evidence workflow scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
class WorkflowArtifact:
    path: str
    bytes: int
    sha256: str

    def to_manifest_entry(self) -> dict[str, object]:
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


def capture_artifacts(
    output_root: Path,
    *,
    patterns: Sequence[str],
) -> list[dict[str, object]]:
    relpaths: set[Path] = set()
    for pattern in patterns:
        relpaths.update(
            path.relative_to(output_root)
            for path in output_root.glob(pattern)
            if path.is_file()
        )

    artifacts: list[WorkflowArtifact] = []
    for relpath in sorted(relpaths):
        path = output_root / relpath
        artifacts.append(
            WorkflowArtifact(
                path=relpath.as_posix(),
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
