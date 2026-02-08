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

_ALLOWED_VERIFY_PROFILES = {"dev", "ci", "release"}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_verify_payload(
    verify_payload: dict[str, Any], verify_profile: str | None
) -> dict[str, Any]:
    """
    Normalize `invarlock verify --json` outputs into the artifact schema shape:
      {profile: dev|ci|release, ok: bool, errors: list[str], verify_json: object}

    InvarLock emits a verify-v1 envelope (format_version=verify-v1) that does not
    include the profile string, so callers should pass `--verify-profile` when
    embedding it.
    """

    def pick_profile(*candidates: object) -> str:
        for c in candidates:
            if isinstance(c, str) and c in _ALLOWED_VERIFY_PROFILES:
                return c
        return "ci"

    if verify_payload.get("format_version") == "verify-v1":
        summary = verify_payload.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        ok = bool(summary.get("ok", False))
        profile = pick_profile(verify_profile)

        errors: list[str] = []
        results = verify_payload.get("results")
        if isinstance(results, list):
            for r in results:
                if not isinstance(r, dict):
                    continue
                if bool(r.get("ok", True)):
                    continue
                rid = r.get("id") or r.get("kind") or "check"
                reason = r.get("reason") or "failed"
                errors.append(f"{rid}:{reason}")

        if not ok and not errors:
            errors.append(str(summary.get("reason", "failed")))

        return {
            "profile": profile,
            "ok": ok,
            "errors": errors,
            "verify_json": verify_payload,
        }

    # Legacy (pre verify-v1): {"profile": "...", "ok": bool, "errors": [...]}
    profile = pick_profile(verify_payload.get("profile"), verify_profile)
    ok = bool(verify_payload.get("ok", False))
    raw_errors = verify_payload.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = [raw_errors]
    errors = [str(e) for e in raw_errors]
    return {
        "profile": profile,
        "ok": ok,
        "errors": errors,
        "verify_json": verify_payload,
    }


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
    p.add_argument(
        "--verify-profile",
        type=str,
        choices=sorted(_ALLOWED_VERIFY_PROFILES),
        help=(
            "Profile used when producing --verify-json (dev|ci|release). "
            "Required when --verify-json is a verify-v1 envelope (no profile field)."
        ),
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
        raw_verify_payload = _read_json(args.verify_json)
        if not isinstance(raw_verify_payload, dict):
            raise TypeError("--verify-json must contain a JSON object at top-level")
        verify_payload = _normalize_verify_payload(
            raw_verify_payload, args.verify_profile
        )

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
        artifact["guard_evidence"]["invarlock"]["verify"] = verify_payload

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
