#!/usr/bin/env python3
"""
pilot_assemble_artifact.py
=========================

Assemble a verifier-carrying artifact (S1) from:
- an InvarLock evaluation report (`evaluation.report.json`)
- one or more verifier trace JSON files (`verifier_trace.v1`)

This is used in Step 3 pilots to prove the end-to-end artifact pipeline works
before scaling experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="Path to InvarLock evaluation.report.json",
    )
    p.add_argument(
        "--verifier-trace",
        type=Path,
        action="append",
        default=[],
        help="Path(s) to verifier_trace.v1 JSON (repeatable).",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--embed-evaluation-report",
        action="store_true",
        help="Embed the evaluation report JSON inside the artifact (portable but larger).",
    )
    p.add_argument(
        "--verify-json",
        type=Path,
        help="Optional path to `invarlock verify --json` output to embed under guard_evidence.invarlock.verify.verify_json.",
    )
    p.add_argument("--invarlock-version", type=str, default="unknown")
    p.add_argument("--git-commit", type=str, default="unknown")
    args = p.parse_args(argv)

    eval_bytes = args.evaluation_report.read_bytes()
    eval_sha = _sha256_hex(eval_bytes)
    eval_obj = (
        _read_json(args.evaluation_report) if args.embed_evaluation_report else None
    )

    traces: list[dict[str, Any]] = []
    for tpath in args.verifier_trace:
        traces.append(_read_json(tpath))

    verify_payload = None
    if args.verify_json is not None:
        verify_payload = _read_json(args.verify_json)

    artifact: dict[str, Any] = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "path": str(args.evaluation_report),
                    "sha256": eval_sha,
                }
            }
        },
        "verifier_traces": traces,
        "provenance": {
            "created_at": _utc_now_iso(),
            "tooling": {
                "invarlock_version": args.invarlock_version,
                "schema_verify_version": "0.1.0",
                "git_commit": args.git_commit,
            },
        },
    }
    if eval_obj is not None:
        artifact["guard_evidence"]["invarlock"]["evaluation_report"]["embedded"] = (
            eval_obj
        )

    if verify_payload is not None:
        artifact["guard_evidence"]["invarlock"]["verify"] = {
            "profile": str(verify_payload.get("profile", "ci")),
            "ok": bool(verify_payload.get("ok", False)),
            "errors": verify_payload.get("errors", []),
            "verify_json": verify_payload,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
