#!/usr/bin/env python3
"""Run multiple evidence-pack edit evaluations in one Python process."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from invarlock.cli.commands.evaluate import evaluate_command


@contextmanager
def _scoped_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _scoped_env(updates: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_entries(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Expected entries list in {path}")
    out: list[dict[str, Any]] = []
    for idx, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Entry {idx} must be an object")
        out.append(raw_entry)
    return out


def _entry_str(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Entry missing non-empty string field: {key}")
    return value


def _entry_bool(entry: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = entry.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _read_timing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    report_out = Path(_entry_str(entry, "report_out"))
    timing_json = Path(entry.get("timing_json") or report_out / "evaluate_timing.json")
    work_dir = Path(entry.get("work_dir") or report_out / ".workdir")
    started = time.perf_counter()
    exit_code = 0
    error: str | None = None

    env_updates = {
        "INVARLOCK_CONFIG_ROOT": _entry_str(entry, "config_root"),
        "INVARLOCK_STORE_EVAL_WINDOWS": "1",
    }
    if _entry_bool(entry, "allow_remote_code"):
        env_updates["INVARLOCK_ALLOW_REMOTE_CODE"] = "1"

    with _scoped_env(env_updates), _scoped_cwd(work_dir):
        try:
            evaluate_command(
                baseline=_entry_str(entry, "baseline"),
                subject=_entry_str(entry, "subject"),
                baseline_report=entry.get("baseline_report"),
                baseline_adapter=str(entry.get("baseline_adapter") or "auto"),
                subject_adapter=str(entry.get("subject_adapter") or "auto"),
                device=entry.get("device"),
                profile=str(entry.get("profile") or "ci"),
                tier=str(entry.get("tier") or "balanced"),
                preset=entry.get("preset"),
                out=_entry_str(entry, "out"),
                report_out=str(report_out),
                edit_label=entry.get("edit_label"),
                quiet=False,
                verbose=False,
                banner=False,
                style=str(entry.get("style") or "audit"),
                timing=False,
                timing_json=str(timing_json),
                progress=False,
                execution_mode=str(entry.get("execution_mode") or "container"),
                allow_network=_entry_bool(entry, "allow_network"),
                allow_host_execution=_entry_bool(entry, "allow_host_execution"),
                allow_third_party_plugins=_entry_bool(
                    entry, "allow_third_party_plugins"
                ),
                allow_remote_code=_entry_bool(entry, "allow_remote_code"),
                assurance=str(entry.get("assurance") or "strict"),
                defer_report_rendering=_entry_bool(entry, "defer_report_rendering"),
                no_color=True,
            )
        except typer.Exit as exc:
            exit_code = int(exc.exit_code or 1)
            error = f"typer_exit:{exit_code}"
        except SystemExit as exc:
            exit_code = int(exc.code or 1) if isinstance(exc.code, int) else 1
            error = f"system_exit:{exit_code}"
        except Exception as exc:  # noqa: BLE001 - group runner records per-entry errors
            exit_code = 1
            error = f"{type(exc).__name__}: {exc}"

    report_path = report_out / "evaluation.report.json"
    if exit_code != 0 and report_path.is_file():
        # Match the shell harness: a structured report is sufficient for pack
        # verdict compilation even when the underlying command exited non-zero.
        error = f"{error}; report_written"
        exit_code = 0

    duration = max(0.0, time.perf_counter() - started)
    timing_payload = _read_timing(timing_json)
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "error": error,
        "edit_spec": entry.get("edit_spec"),
        "version": entry.get("version"),
        "run": entry.get("run"),
        "subject": entry.get("subject"),
        "report": str(report_path),
        "timing_json": str(timing_json),
        "wall_seconds": duration,
        "evaluate_timing": timing_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-json", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--timing-jsonl")
    args = parser.parse_args()

    entries_path = Path(args.entries_json)
    summary_path = Path(args.summary_out)
    timing_jsonl = Path(args.timing_jsonl) if args.timing_jsonl else None
    entries = _load_entries(entries_path)

    group_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for entry in entries:
        result = _run_entry(entry)
        results.append(result)
        _append_jsonl(timing_jsonl, {"kind": "evaluate_group_entry", **result})
        if not result["ok"]:
            break

    total_seconds = max(0.0, time.perf_counter() - group_start)
    ok = all(bool(result.get("ok")) for result in results) and len(results) == len(
        entries
    )
    summary = {
        "schema": "invarlock/evidence-pack-evaluate-group-summary-v1",
        "ok": ok,
        "entry_count": len(entries),
        "completed_entries": len(results),
        "single_process_invocations": 1,
        "avoided_cli_process_invocations": max(0, len(entries) - 1),
        "total_wall_seconds": total_seconds,
        "entries": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
