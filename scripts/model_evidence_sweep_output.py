from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Sequence


class SummaryResult(Protocol):
    slug: str
    lane_id: str
    status: str
    detail: str | None
    evaluate_exit: int
    verify_exit: int | None
    report_path: str

    @property
    def ok(self) -> bool: ...

    def to_summary_entry(self) -> dict[str, object]: ...


class EvidenceSpec(Protocol):
    def to_manifest_entry(self) -> dict[str, str]: ...


def write_summary(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    shard_index: int,
    shard_count: int,
    results: Sequence[SummaryResult],
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
        "suite": suite,
        "execution_mode": execution_mode,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "ok": all(result.ok for result in results),
        "results": [result.to_summary_entry() for result in results],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    specs: Sequence[EvidenceSpec],
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "execution_mode": execution_mode,
        "lanes": [spec.to_manifest_entry() for spec in specs],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
