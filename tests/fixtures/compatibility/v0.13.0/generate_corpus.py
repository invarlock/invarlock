#!/usr/bin/env python3
"""Generate the v0.13 compatibility inventory from the committed handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_ROOT.parents[3]
GOLDEN = REPO_ROOT / "examples/acceptance-handoff/golden"
CORPUS = FIXTURE_ROOT / "corpus.json"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate() -> bytes:
    evidence = GOLDEN / "evidence"
    receipt_path = GOLDEN / "verification.receipt.json"
    report_path = evidence / "reports/evaluation.report.json"
    schedule_path = evidence / "schedule/runtime-behavioral-schedule.json"
    manifest_path = evidence / "manifest.json"
    anchors = _object(GOLDEN / "technical-anchors.json")
    receipt = _object(receipt_path)
    report = _object(report_path)
    manifest = _object(manifest_path)
    document = {
        "format": "invarlock/compatibility-corpus-v1",
        "contract_release": "0.13.0",
        "cases": [
            {
                "name": "acceptance-handoff-accepted-v013",
                "evidence": "examples/acceptance-handoff/golden/evidence",
                "receipt": (
                    "examples/acceptance-handoff/golden/verification.receipt.json"
                ),
                "policy": ("examples/acceptance-handoff/golden/evaluated-policy.json"),
                "sha256": {
                    "manifest": _sha256(manifest_path),
                    "receipt": _sha256(receipt_path),
                    "report": _sha256(report_path),
                    "schedule": _sha256(schedule_path),
                },
                "anchors": {
                    "artifact_digests": anchors["artifact_digests"],
                    "runtime_digests": anchors["runtime_digests"],
                    "schedule_digest": anchors["schedule_digest"],
                    "evidence_signer_fingerprint": (
                        anchors["evidence_signer_fingerprint"]
                    ),
                    "verifier_identity": anchors["verifier_identity"],
                    "verifier_fingerprint": anchors["verifier_fingerprint"],
                },
                "expected": {
                    "pack_format": manifest["format"],
                    "receipt_format": receipt["statement"]["format"],
                    "report_format": report["format"],
                    "metric": report["metric"],
                    "record_count": report["record_count"],
                    "technical_verdict": report["verdict"],
                },
            }
        ],
    }
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = generate()
    if args.check:
        if CORPUS.read_bytes() != payload:
            raise SystemExit("v0.13 compatibility corpus inventory is stale")
        return
    CORPUS.write_bytes(payload)


if __name__ == "__main__":
    main()
