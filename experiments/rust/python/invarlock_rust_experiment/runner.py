"""Bounded paired comparison with one retained record for every attempted process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .fixtures import write_json

CANDIDATES = ("reference", "optimized", "rust")
BOOTSTRAP = "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.worker import main;raise SystemExit(main())"


def balanced_order(rounds, seed):
    rng = random.Random(seed)
    base = list(CANDIDATES)
    rng.shuffle(base)
    result = []
    for round_number in range(rounds):
        shift = round_number % len(base)
        result.append(base[shift:] + base[:shift])
    return result


def classify(exit_code, timed_out, payload, matches):
    if timed_out:
        return "timeout"
    if exit_code < 0:
        return "crash"
    if exit_code != 0:
        return "nonzero_exit"
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return "invalid_output"
    return "valid" if matches else "wrong_output"


def run_process(command, directory, timeout):
    """Capture per-process resource usage via wait4, avoiding cumulative RSS deltas."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timed_out = False
    with (
        (directory / "stdout.txt").open("wb") as stdout,
        (directory / "stderr.txt").open("wb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                cwd=directory,
                start_new_session=True,
            )
        except OSError as exc:
            record = {
                "command": command,
                "exit_code": None,
                "timed_out": False,
                "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": None,
                "system_cpu_seconds": None,
                "max_rss_raw": None,
                "max_rss_raw_unit": None,
                "max_rss_bytes": None,
                "platform": sys.platform,
                "payload": None,
                "output_comparison": "not_started",
                "status": "launch_error",
                "error": str(exc),
            }
            write_json(directory / "sample.json", record)
            return record
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - started > timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                _, status, usage = os.wait4(process.pid, 0)
                break
            time.sleep(0.01)
        exit_code = os.waitstatus_to_exitcode(status)
        process.returncode = exit_code
    wall = time.monotonic() - started
    try:
        payload = json.loads((directory / "stdout.txt").read_text())
    except (ValueError, UnicodeError):
        payload = None
    # Darwin ru_maxrss is bytes; Linux ru_maxrss is KiB. Retain both the raw unit and bytes.
    rss_factor = (
        1
        if sys.platform == "darwin"
        else 1024
        if sys.platform.startswith("linux")
        else None
    )
    record = {
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_seconds": wall,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_raw": usage.ru_maxrss,
        "max_rss_raw_unit": "bytes"
        if rss_factor == 1
        else "KiB"
        if rss_factor == 1024
        else "unknown",
        "max_rss_bytes": usage.ru_maxrss * rss_factor if rss_factor else None,
        "platform": sys.platform,
        "payload": payload,
        "output_comparison": "pending",
        "status": classify(exit_code, timed_out, payload, True),
    }
    write_json(directory / "sample.json", record)
    return record


def command(
    python,
    experiment_path,
    stage,
    fixture,
    output,
    candidate="reference",
    count=None,
    metadata=None,
    scenario=None,
):
    args = [
        str(python),
        "-I",
        "-c",
        BOOTSTRAP,
        str(experiment_path),
        "--stage",
        stage,
        "--fixture",
        str(fixture),
        "--output",
        str(output),
        "--candidate",
        candidate,
    ]
    if count is not None:
        args += ["--count", str(count)]
    if metadata is not None:
        args += ["--metadata", str(metadata)]
    if scenario is not None:
        args += ["--scenario", scenario]
    return args


def verify_metadata(path):
    value = json.loads(Path(path).read_bytes())
    for key in (
        "source_commit",
        "source_tree_sha256",
        "dependency_lock",
        "root_wheel",
        "native_wheel",
        "native_cargo_lock",
    ):
        if key not in value:
            raise ValueError(f"metadata requires {key}")
    for key in ("dependency_lock", "root_wheel", "native_wheel", "native_cargo_lock"):
        item = value[key]
        raw = Path(item["path"]).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["sha256"].removeprefix("sha256:"):
            raise ValueError(f"metadata {key} file hash mismatch")
    return value


def execute_matrix(args):
    args.python = args.python.absolute()
    raw = args.matrix.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.matrix_sha256.removeprefix("sha256:"):
        raise ValueError("frozen matrix hash mismatch")
    matrix = json.loads(raw)
    if set(matrix) != {"counts", "rounds", "seed", "timeout_seconds"}:
        raise ValueError("frozen matrix fields differ from supported protocol")
    if matrix["counts"] not in ([1000, 5000], [12000, 50000]):
        raise ValueError(
            "only declared initial or separately approved large matrix is allowed"
        )
    if matrix["rounds"] != (5 if matrix["counts"] == [1000, 5000] else 3):
        raise ValueError("round count differs from declared matrix")
    metadata = verify_metadata(args.metadata)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "metadata.json", metadata)
    write_json(root / "matrix.json", matrix)
    experiment_path = Path(__file__).resolve().parents[1]
    protocol_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    write_json(
        root / "run-manifest.json",
        {
            "matrix_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "harness_sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in protocol_files
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "python": str(args.python),
            "run_created_unix_seconds": time.time(),
        },
    )
    samples = []
    for count in matrix["counts"]:
        fixture = root / f"fixture-{count}"
        fixture_record = run_process(
            command(
                args.python,
                experiment_path,
                "fixture",
                fixture,
                root / f"fixture-build-{count}",
                count=count,
                metadata=args.metadata.resolve(),
            ),
            root / f"fixture-process-{count}",
            matrix["timeout_seconds"],
        )
        write_json(root / f"fixture-status-{count}.json", fixture_record)
        if fixture_record["status"] != "valid":
            write_json(
                root / "stopped.json", {"reason": "fixture_failed", "count": count}
            )
            return 1
        for stage in ("kernel", "verify"):
            pending = []
            halt = False
            for round_number, order in enumerate(
                balanced_order(matrix["rounds"], matrix["seed"] + count)
            ):
                for position, candidate in enumerate(order):
                    sample_root = (
                        root
                        / f"n{count}-{stage}-r{round_number}-{position}-{candidate}"
                    )
                    record = run_process(
                        command(
                            args.python,
                            experiment_path,
                            stage,
                            fixture,
                            sample_root / "artifacts",
                            candidate,
                            metadata=args.metadata.resolve(),
                        ),
                        sample_root,
                        matrix["timeout_seconds"],
                    )
                    record.update(
                        count=count,
                        stage=stage,
                        round=round_number,
                        position=position,
                        candidate=candidate,
                    )
                    record["execution_status"] = record["status"]
                    pending.append((sample_root, record))
                    completed_reference = [
                        r["payload"]["result_sha256"]
                        for _, r in pending
                        if r["candidate"] == "reference" and r["status"] == "valid"
                    ]
                    if completed_reference:
                        expected_so_far = completed_reference[0]
                        for _, previous in pending:
                            if (
                                previous["status"] == "valid"
                                and previous["payload"]["result_sha256"]
                                != expected_so_far
                            ):
                                previous["status"] = "wrong_output"
                    if any(r["status"] != "valid" for _, r in pending):
                        halt = True
                        break
                if halt:
                    break
            assessment = assess_group(
                [record for _, record in pending], matrix["rounds"]
            )
            for sample_root, record in pending:
                write_json(sample_root / "assessment.json", record)
                samples.append(record)
            write_json(root / f"assessment-n{count}-{stage}.json", assessment)
            if any(record["status"] != "valid" for _, record in pending):
                write_json(
                    root / "stopped.json",
                    {"reason": "candidate_failure", "count": count, "stage": stage},
                )
                write_json(root / "summary.json", summarize(samples))
                return 1
    write_json(root / "summary.json", summarize(samples))
    return 0


def assess_group(records, requested_rounds):
    """Keep observed reference consistency separate from requested run completion."""
    references = [record for record in records if record["candidate"] == "reference"]
    successful = [
        record
        for record in references
        if record.get("execution_status", record["status"]) == "valid"
    ]
    hashes = {record["payload"]["result_sha256"] for record in successful}
    failed = any(
        record.get("execution_status", record["status"]) != "valid"
        for record in references
    )
    consistent = bool(successful) and len(hashes) == 1 and not failed
    state = (
        "consistent"
        if consistent
        else "failed"
        if failed
        else "inconsistent"
        if hashes
        else "not_observed"
    )
    expected = next(iter(hashes)) if consistent else None
    for record in records:
        execution = record.setdefault("execution_status", record["status"])
        if execution != "valid":
            record["status"] = execution
            record["output_comparison"] = "not_available"
        elif consistent:
            matched = record["payload"]["result_sha256"] == expected
            record["status"] = "valid" if matched else "wrong_output"
            record["output_comparison"] = "matched" if matched else "mismatched"
        else:
            record["status"] = (
                "unevaluable_missing_reference"
                if state == "not_observed"
                else "unevaluable_reference_failure"
            )
            record["output_comparison"] = "unevaluable"
    return {
        "reference_ok": consistent,
        "reference_state": state,
        "reference_attempted": len(references),
        "reference_successful": len(successful),
        "reference_complete": len(references) == requested_rounds,
        "requested_rounds": requested_rounds,
        "matrix_complete": len(records) == requested_rounds * len(CANDIDATES),
        "expected_result_sha256": expected,
        "samples": records,
    }


def operation_seconds(sample):
    return (sample.get("payload") or {}).get("operation_wall_seconds")


def summarize(samples):
    result = {
        "attempted": len(samples),
        "executed_successfully": sum(
            s.get("execution_status", s["status"]) == "valid" for s in samples
        ),
        "valid": sum(s["status"] == "valid" for s in samples),
        "failures": sum(
            s["status"]
            not in {
                "valid",
                "unevaluable_missing_reference",
                "unevaluable_reference_failure",
            }
            for s in samples
        ),
        "unevaluable": sum(s["status"].startswith("unevaluable_") for s in samples),
        "resource_scope": "whole worker process including imports, identity checks and output serialization; CPU and RSS are not kernel-only",
        "groups": [],
        "paired_ratios": [],
    }
    for count in sorted({s["count"] for s in samples}):
        for stage in ("kernel", "verify"):
            primary = "kernel_operation" if stage == "kernel" else "whole_fresh_process"
            for candidate in CANDIDATES:
                group = [
                    s
                    for s in samples
                    if (s["count"], s["stage"], s["candidate"])
                    == (count, stage, candidate)
                ]
                valid = [s for s in group if s["status"] == "valid"]
                process_times = [s["wall_seconds"] for s in valid]
                operations = [
                    operation_seconds(s)
                    for s in valid
                    if operation_seconds(s) is not None
                ]
                result["groups"].append(
                    {
                        "count": count,
                        "stage": stage,
                        "candidate": candidate,
                        "attempted": len(group),
                        "valid": len(valid),
                        "statuses": [s["status"] for s in group],
                        "primary_time_scope": primary,
                        "process_wall_seconds": process_times,
                        "median_process_wall_seconds": statistics.median(process_times)
                        if process_times
                        else None,
                        "min_process_wall_seconds": min(process_times, default=None),
                        "max_process_wall_seconds": max(process_times, default=None),
                        "operation_wall_seconds": operations,
                        "median_operation_wall_seconds": statistics.median(operations)
                        if operations
                        else None,
                        "min_operation_wall_seconds": min(operations, default=None),
                        "max_operation_wall_seconds": max(operations, default=None),
                    }
                )
            reference = {
                s["round"]: s
                for s in samples
                if s["count"] == count
                and s["stage"] == stage
                and s["candidate"] == "reference"
                and s["status"] == "valid"
            }
            for candidate in ("optimized", "rust"):
                pairs = [
                    (reference[s["round"]], s)
                    for s in samples
                    if s["count"] == count
                    and s["stage"] == stage
                    and s["candidate"] == candidate
                    and s["status"] == "valid"
                    and s["round"] in reference
                ]
                process_ratios = [
                    left["wall_seconds"] / right["wall_seconds"]
                    for left, right in pairs
                ]
                operation_ratios = [
                    operation_seconds(left) / operation_seconds(right)
                    for left, right in pairs
                    if operation_seconds(left) is not None
                    and operation_seconds(right) not in (None, 0)
                ]
                ratios = operation_ratios if stage == "kernel" else process_ratios
                result["paired_ratios"].append(
                    {
                        "count": count,
                        "stage": stage,
                        "candidate": candidate,
                        "valid_pairs": len(pairs),
                        "operation_timed_pairs": len(operation_ratios),
                        "primary_time_scope": primary,
                        "reference_over_candidate_process_wall_ratios": process_ratios,
                        "reference_over_candidate_operation_wall_ratios": operation_ratios,
                        "median_ratio": statistics.median(ratios) if ratios else None,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen-matrix", action="store_true", required=True)
    return execute_matrix(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
