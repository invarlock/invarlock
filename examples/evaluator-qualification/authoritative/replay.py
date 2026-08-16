#!/usr/bin/env python3
"""Replay qualified records through InvarLock's strict runtime-import boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT.parent
REPO = EXAMPLE.parents[1]

sys.path.insert(0, str(REPO / "src"))

from invarlock.core.runtime_provider import (  # noqa: E402
    load_runtime_behavioral_schedule,
)
from invarlock.evaluator_qualification import (  # noqa: E402
    qualify_evaluator_export,
)
from invarlock.evidence_pack_contract import canonical_json_bytes  # noqa: E402
from invarlock.runtime_import_authoring import (  # noqa: E402
    load_external_scoring_records_jsonl,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _record_bytes(result: Any) -> bytes:
    return b"".join(
        canonical_json_bytes(
            {
                "input_sha256": record.input_sha256,
                "output_sha256": record.output_sha256,
                "output_text": record.output_text,
                "record_id": record.record_id,
                "status": record.status,
            }
        )
        for record in result.runtime_records()
    )


def replay(profile_id: str, *, write: bool) -> dict[str, object]:
    artifact = ROOT / "artifacts" / profile_id
    result = qualify_evaluator_export(
        profile_path=artifact / "profile.json",
        schedule_path=ROOT / "schedule.json",
        export_path=artifact / "export.json",
        raw_output_path=artifact / "upstream-output.json",
    )
    if result.authority != "verdict_authority":
        raise ValueError(
            f"{profile_id}: independently replayable import requires verdict authority"
        )
    cases = _load(ROOT / "cases.json")
    source_evaluation = cases.get("source_evaluation")
    raw = _load(artifact / "upstream-output.json")
    if (
        not isinstance(source_evaluation, dict)
        or raw.get("source_evaluation") != source_evaluation
    ):
        raise ValueError(f"{profile_id}: source model execution is not bound")
    records_bytes = _record_bytes(result)
    records_path = artifact / "runtime-import-records.jsonl"
    if write:
        records_path.write_bytes(records_bytes)
    elif records_path.read_bytes() != records_bytes:
        raise ValueError(f"{profile_id}: retained runtime import records are stale")
    schedule = load_runtime_behavioral_schedule(ROOT / "runtime-schedule.json")
    replayed = load_external_scoring_records_jsonl(records_path, schedule=schedule)
    if replayed != result.runtime_records():
        raise ValueError(
            f"{profile_id}: runtime import replay changed qualified records"
        )
    result_bytes = canonical_json_bytes(result.as_dict())
    replay_document: dict[str, object] = {
        "bindings": {
            "qualification_result_sha256": _digest(result_bytes),
            "runtime_import_records_sha256": _digest(records_bytes),
            "runtime_schedule_sha256": _digest(
                (ROOT / "runtime-schedule.json").read_bytes()
            ),
            "replay_runner_sha256": _digest(Path(__file__).read_bytes()),
            "source_evaluation_sha256": _digest(
                canonical_json_bytes(source_evaluation)
            ),
        },
        "format": "invarlock/evaluator-authoritative-import-replay-v1",
        "mean_score": result.mean_score,
        "profile_id": profile_id,
        "record_count": len(replayed),
        "records_sha256": result.records_sha256,
        "source_kind": source_evaluation["kind"],
    }
    replay_bytes = canonical_json_bytes(replay_document)
    replay_path = artifact / "import-replay.json"
    if write:
        replay_path.write_bytes(replay_bytes)
    elif replay_path.read_bytes() != replay_bytes:
        raise ValueError(f"{profile_id}: retained import replay is stale")
    return replay_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "verify"))
    parser.add_argument("profile_id")
    args = parser.parse_args()
    replay(args.profile_id, write=args.command == "write")
    print(f"{args.command} independently replayable import {args.profile_id}")


if __name__ == "__main__":
    main()
