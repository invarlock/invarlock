#!/usr/bin/env python3
"""
cases_from_harness.py
=====================

Convert external verifier harness outputs (HumanEval/MBPP/etc.) into the
normalized `cases.jsonl` format consumed by verifier_trace_from_cases.py.

This tool is intentionally dependency-free and format-flexible: callers can map
their harness fields via CLI options.

Output rows match verifier_trace_from_cases.py expectations:
  - id (str)
  - verdict (pass/fail/error/timeout/skipped)
  - attempt_id (optional int)
  - output_sha256 (optional sha256 hex)
  - stderr_sha256 (optional sha256 hex)
  - wall_time_s (optional float)
  - error_type (optional str)
  - failing_test_ids (optional list[str])
  - message_excerpt (optional str, truncated)

Note: this is *not* a verifier; it does not execute code. It only normalizes and
hashes harness-emitted outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

VERDICT_ENUM = {"pass", "fail", "error", "timeout", "skipped"}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}")
            out.append(obj)
    return out


def _extract_error_type(stderr: str) -> str | None:
    for line in reversed(str(stderr).splitlines()):
        line = line.strip()
        if not line:
            continue
        if ":" in line and line[0].isalpha():
            return line.split(":", 1)[0].strip()
    return None


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _infer_verdict(
    rec: dict[str, Any],
    *,
    verdict_field: str,
    passed_field: str,
    status_field: str,
    strict: bool,
) -> str:
    verdict_raw = rec.get(verdict_field)
    if isinstance(verdict_raw, str) and verdict_raw:
        v = verdict_raw.strip().lower()
        if v in VERDICT_ENUM:
            return v
        if v in {"passed", "ok", "success"}:
            return "pass"
        if v in {"failed", "wrong"}:
            return "fail"
        if strict:
            raise ValueError(
                f"Unknown verdict={verdict_raw!r} in record id={rec.get('id')!r}"
            )
        return "error"

    passed_raw = rec.get(passed_field)
    passed = _coerce_bool(passed_raw)
    if passed is not None:
        return "pass" if passed else "fail"

    status_raw = rec.get(status_field)
    if isinstance(status_raw, str) and status_raw:
        s = status_raw.strip().lower()
        if s in VERDICT_ENUM:
            return s
        if s in {"passed", "ok", "success"}:
            return "pass"
        if s in {"failed", "wrong"}:
            return "fail"
        if "timeout" in s:
            return "timeout"
        if "error" in s or "exception" in s:
            return "error"

    if strict:
        raise ValueError(
            f"Unable to infer verdict (fields {verdict_field!r}/{passed_field!r}/{status_field!r})"
        )
    return "error"


def _first_str(rec: dict[str, Any], fields: list[str]) -> str | None:
    for f in fields:
        v = rec.get(f)
        if isinstance(v, str) and v:
            return v
    return None


def normalize_record(
    rec: dict[str, Any],
    *,
    id_fields: list[str],
    attempt_fields: list[str],
    completion_fields: list[str],
    stderr_fields: list[str],
    wall_time_fields: list[str],
    error_type_fields: list[str],
    failing_tests_fields: list[str],
    message_fields: list[str],
    verdict_field: str,
    passed_field: str,
    status_field: str,
    max_message_chars: int,
    include_output_text: bool,
    strict: bool,
) -> dict[str, Any]:
    rid = _first_str(rec, id_fields)
    if not rid:
        raise ValueError("record missing id/task_id")

    attempt_raw: Any = None
    for f in attempt_fields:
        if f in rec:
            attempt_raw = rec.get(f)
            break
    attempt_id: int | None = None
    if attempt_raw is not None:
        try:
            attempt_id = int(attempt_raw)
        except Exception:
            attempt_id = None

    completion = _first_str(rec, completion_fields)
    stderr = _first_str(rec, stderr_fields)
    message = _first_str(rec, message_fields)

    wall_time: float | None = None
    for f in wall_time_fields:
        if f not in rec:
            continue
        try:
            wall_time = float(rec.get(f))
        except Exception:
            wall_time = None
        break

    error_type = _first_str(rec, error_type_fields)
    if error_type is None and stderr:
        error_type = _extract_error_type(stderr)

    verdict = _infer_verdict(
        rec,
        verdict_field=verdict_field,
        passed_field=passed_field,
        status_field=status_field,
        strict=strict,
    )

    out: dict[str, Any] = {"id": rid, "verdict": verdict}
    if attempt_id is not None:
        out["attempt_id"] = attempt_id
    if wall_time is not None:
        out["wall_time_s"] = wall_time
    if error_type is not None:
        out["error_type"] = str(error_type)

    if completion is not None:
        out["output_sha256"] = _sha256_hex(completion.encode("utf-8"))
        if include_output_text:
            out["output"] = completion

    if stderr is not None:
        out["stderr_sha256"] = _sha256_hex(stderr.encode("utf-8"))
        if message is None:
            message = stderr

    failing = None
    for f in failing_tests_fields:
        if f in rec:
            failing = rec.get(f)
            break
    if isinstance(failing, list) and all(isinstance(x, str) for x in failing):
        out["failing_test_ids"] = failing

    if isinstance(message, str) and message:
        out["message_excerpt"] = message[:max_message_chars]

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="Input JSONL."
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output normalized cases JSONL (for verifier_trace_from_cases.py).",
    )

    p.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Primary id field (fallbacks are built-in).",
    )
    p.add_argument(
        "--attempt-field",
        type=str,
        default="attempt_id",
        help="Primary attempt id field (fallbacks are built-in).",
    )
    p.add_argument(
        "--completion-field",
        type=str,
        default="completion",
        help="Primary completion/output field (fallbacks are built-in).",
    )
    p.add_argument(
        "--stderr-field",
        type=str,
        default="stderr",
        help="Primary stderr/error-log field (fallbacks are built-in).",
    )
    p.add_argument(
        "--verdict-field",
        type=str,
        default="verdict",
        help="Field containing verdict strings (pass/fail/timeout/etc.).",
    )
    p.add_argument(
        "--passed-field",
        type=str,
        default="passed",
        help="Field containing boolean-ish pass/fail (used if verdict-field missing).",
    )
    p.add_argument(
        "--status-field",
        type=str,
        default="status",
        help="Field containing status strings (used if verdict/passed missing).",
    )
    p.add_argument("--wall-time-field", type=str, default="wall_time_s")
    p.add_argument("--error-type-field", type=str, default="error_type")
    p.add_argument("--failing-tests-field", type=str, default="failing_test_ids")
    p.add_argument("--message-field", type=str, default="message_excerpt")

    p.add_argument("--max-message-chars", type=int, default=400)
    p.add_argument(
        "--include-output-text",
        action="store_true",
        help="If set, include raw output text in normalized cases (in addition to its sha256).",
    )
    p.add_argument(
        "--non-strict",
        action="store_true",
        help="If set, convert unknown/missing verdicts to verdict=error instead of failing.",
    )
    args = p.parse_args(argv)

    strict = not bool(args.non_strict)

    rows = _read_jsonl(args.in_path)
    if not rows:
        print("No records found.", file=sys.stderr)
        return 2

    id_fields = [args.id_field, "task_id", "problem_id"]
    attempt_fields = [args.attempt_field, "completion_id", "sample_id", "generation_id"]
    completion_fields = [args.completion_field, "output", "generated", "response"]
    stderr_fields = [args.stderr_field, "error", "exception", "traceback"]
    wall_time_fields = [args.wall_time_field, "time_s", "elapsed_s"]
    error_type_fields = [args.error_type_field]
    failing_tests_fields = [args.failing_tests_field, "failed_tests", "failed_test_ids"]
    message_fields = [args.message_field, "message", "error_message", "stderr"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in rows:
            norm = normalize_record(
                rec,
                id_fields=id_fields,
                attempt_fields=attempt_fields,
                completion_fields=completion_fields,
                stderr_fields=stderr_fields,
                wall_time_fields=wall_time_fields,
                error_type_fields=error_type_fields,
                failing_tests_fields=failing_tests_fields,
                message_fields=message_fields,
                verdict_field=str(args.verdict_field),
                passed_field=str(args.passed_field),
                status_field=str(args.status_field),
                max_message_chars=int(args.max_message_chars),
                include_output_text=bool(args.include_output_text),
                strict=strict,
            )
            f.write(json.dumps(norm, ensure_ascii=True) + "\n")

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
