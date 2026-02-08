#!/usr/bin/env python3
"""
run_code_tests_verifier.py
==========================

Minimal, deterministic(ish) code-execution verifier harness for pilot runs.

Inputs:
- tasks JSONL: each line at least {"id": "...", "prompt": "...", "tests": "..."}
- completions JSONL: each line at least {"id": "...", "completion": "..."} and
  optionally {"attempt_id": 0}

Outputs:
- cases JSONL: normalized per-attempt verdicts usable by verifier_trace_from_cases.py

Notes:
- This is intended as a *pluggable* harness for the VerifAI-2 F4/S1 pipeline.
- It is not a security sandbox; in paper runs, execute verifiers inside a real
  container/sandbox and record the contract fields accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


def _make_exec_wrapper(*, cpu_s: int, mem_mb: int) -> str:
    # Best-effort resource limiting. Some limits may be ignored depending on OS.
    return f"""
import sys
import os

try:
    import resource
    # CPU seconds
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_s}, {cpu_s}))
    # Address space / virtual memory
    mem_bytes = int({mem_mb}) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except Exception:
    pass

# Avoid accidental network access in common cases (not a real sandbox).
try:
    import socket  # noqa: F401
    socket.socket = None  # type: ignore[assignment]
except Exception:
    pass

code = sys.stdin.read()
g = {{}}
exec(compile(code, "<verifier>", "exec"), g, g)
""".lstrip()


def _extract_error_type(stderr: str) -> str | None:
    # Very lightweight: try to pull `ValueError:` from the last traceback line.
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line:
            continue
        if ":" in line and line[0].isalpha():
            return line.split(":", 1)[0].strip()
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--completions", type=Path, required=True)
    p.add_argument("--out-cases", type=Path, required=True)

    p.add_argument("--task-id-field", type=str, default="id")
    p.add_argument("--task-prompt-field", type=str, default="prompt")
    p.add_argument("--task-tests-field", type=str, default="tests")

    p.add_argument("--completion-id-field", type=str, default="id")
    p.add_argument("--completion-text-field", type=str, default="completion")
    p.add_argument("--completion-attempt-field", type=str, default="attempt_id")

    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--cpu-limit-s", type=int, default=10)
    p.add_argument("--mem-limit-mb", type=int, default=2048)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--max-stderr-chars", type=int, default=400)
    args = p.parse_args(argv)

    tasks = _read_jsonl(args.tasks)
    by_id: dict[str, dict[str, Any]] = {}
    for t in tasks:
        tid = t.get(args.task_id_field)
        if not isinstance(tid, str) or not tid:
            raise ValueError(f"Task missing id field {args.task_id_field!r}: {t!r}")
        by_id[tid] = t

    completions = _read_jsonl(args.completions)
    if not completions:
        print("No completions found.", file=sys.stderr)
        return 2

    wrapper = _make_exec_wrapper(
        cpu_s=int(args.cpu_limit_s), mem_mb=int(args.mem_limit_mb)
    )

    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    with args.out_cases.open("w", encoding="utf-8") as out:
        for c in completions:
            cid = c.get(args.completion_id_field)
            if not isinstance(cid, str) or not cid:
                raise ValueError(
                    f"Completion missing id field {args.completion_id_field!r}: {c!r}"
                )
            task = by_id.get(cid)
            if task is None:
                rec = {
                    "id": cid,
                    "verdict": "error",
                    "error_type": "missing_task",
                }
                out.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            prompt = task.get(args.task_prompt_field)
            tests = task.get(args.task_tests_field)
            completion = c.get(args.completion_text_field)
            if not (
                isinstance(prompt, str)
                and isinstance(tests, str)
                and isinstance(completion, str)
            ):
                raise ValueError(f"Task/completion fields missing for id={cid}")

            attempt_id = c.get(args.completion_attempt_field)
            try:
                attempt_id_int = int(attempt_id) if attempt_id is not None else 0
            except Exception:
                attempt_id_int = 0

            program = f"{prompt}\n{completion}\n\n{tests}\n"
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [str(args.python), "-I", "-c", wrapper],
                    input=program,
                    text=True,
                    capture_output=True,
                    timeout=float(args.timeout_s),
                    env={
                        **os.environ,
                        "PYTHONHASHSEED": "0",
                    },
                )
                wall = time.perf_counter() - start
            except subprocess.TimeoutExpired as exc:
                wall = time.perf_counter() - start
                msg = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                msg = msg[: int(args.max_stderr_chars)]
                rec = {
                    "id": cid,
                    "attempt_id": attempt_id_int,
                    "verdict": "timeout",
                    "wall_time_s": wall,
                    "output_sha256": _sha256_hex(completion.encode("utf-8")),
                }
                if msg:
                    rec["stderr_sha256"] = _sha256_hex(msg.encode("utf-8"))
                    rec["message_excerpt"] = msg
                out.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            stderr = (proc.stderr or "") if isinstance(proc.stderr, str) else ""
            stderr_excerpt = stderr.strip().replace("\r\n", "\n")[
                : int(args.max_stderr_chars)
            ]
            ok = proc.returncode == 0
            verdict = "pass" if ok else "fail"

            rec = {
                "id": cid,
                "attempt_id": attempt_id_int,
                "verdict": verdict,
                "wall_time_s": wall,
                "output_sha256": _sha256_hex(completion.encode("utf-8")),
            }
            if not ok:
                et = _extract_error_type(stderr_excerpt)
                if et:
                    rec["error_type"] = et
                if stderr_excerpt:
                    rec["stderr_sha256"] = _sha256_hex(stderr_excerpt.encode("utf-8"))
                    rec["message_excerpt"] = stderr_excerpt
            out.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(args.out_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
