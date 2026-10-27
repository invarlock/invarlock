"""Fresh optimization comparison; retain the original expanded measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import full_runner
from .fixtures import write_json
from .runner import classify
from .worker import sha_file

CANDIDATES = (
    "reference",
    "rust-standalone-before",
    "rust-ffi",
    "rust-standalone",
)
WORKLOADS = (
    "captured-nll-1000",
    "captured-nll-5000",
    "captured-nll-12000",
    "captured-nll-50000",
    "captured-regression",
    "native-pack",
    "judge-pack",
    "evidence-set",
)
HARNESS_FILES = (
    "__init__.py",
    "optimization_runner.py",
    "full_runner.py",
    "full_worker.py",
    "runner.py",
    "worker.py",
    "adapters.py",
    "fixtures.py",
)


def run_process(command, directory, timeout):
    """Retain wait4 measurements, rejecting non-finite JSON before record writing."""
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
    parse_error = None
    try:
        payload = json.loads((directory / "stdout.txt").read_text())
        full_runner.canonical(payload)
    except (ValueError, UnicodeError) as error:
        payload = None
        parse_error = str(error)
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
    if parse_error is not None:
        record["validation_error"] = parse_error
        record["payload_retained_in"] = "stdout.txt"
    write_json(directory / "sample.json", record)
    return record


def balanced_order(rounds, seed):
    base = list(CANDIDATES)
    random.Random(seed).shuffle(base)
    return [base[index % 4 :] + base[: index % 4] for index in range(rounds)]


def bind_inputs(metadata_path, fixtures_path, matrix_path):
    fixtures = json.loads(fixtures_path.read_bytes())
    metadata = full_runner.bind_inputs(metadata_path, fixtures)
    for name, supplied in (
        ("fixtures_manifest", fixtures_path),
        ("matrix", matrix_path),
    ):
        item = metadata[name]
        expected = item["sha256"].removeprefix("sha256:")
        if sha_file(supplied) != expected or sha_file(item["path"]) != expected:
            raise ValueError(f"{name} identity changed")
    for name in ("standalone_before", "standalone", "offline_cli", "control_cli"):
        if name == "control_cli" and name not in metadata:
            continue
        item = metadata[name]
        if sha_file(item["path"]) != item["sha256"].removeprefix("sha256:"):
            raise ValueError(f"{name} binary changed")
    sources = metadata["harness_sources"]
    for name in HARNESS_FILES:
        path = str(Path(__file__).resolve().with_name(name))
        if path not in sources or sha_file(path) != sources[path]:
            raise ValueError(f"imported harness is not frozen: {path}")
    return metadata, fixtures


def sample(candidate, fixture, directory, args, timeout):
    inputs = directory.parent / (directory.name + "-inputs")
    inputs.mkdir()
    outputs = inputs / "outputs"
    outputs.mkdir()
    request = full_runner.materialize(fixture["request"], outputs)
    request_path = inputs / "request.json"
    write_json(request_path, request)
    artifact = None
    if candidate in ("rust-standalone-before", "rust-standalone"):
        key = "standalone_before" if candidate.endswith("-before") else "standalone"
        artifact = args.bound_metadata[key]
        if sha_file(artifact["path"]) != artifact["sha256"].removeprefix("sha256:"):
            raise ValueError(f"{key} binary changed before dispatch")
        command = [artifact["path"], str(request_path)]
    elif candidate in ("reference", "rust-ffi"):
        command = [
            str(args.python),
            "-I",
            "-c",
            full_runner.BOOTSTRAP,
            str(Path(__file__).resolve().parents[1]),
            "--candidate",
            candidate,
            "--request",
            str(request_path),
            "--metadata",
            str(args.metadata),
        ]
    else:
        raise ValueError(f"unknown candidate: {candidate}")
    request_sha256 = sha_file(request_path)
    row = run_process(command, directory, timeout)
    row.update(
        candidate=candidate,
        workload=fixture["name"],
        request_sha256=request_sha256,
        output_directory=str(outputs),
        executable_artifact=artifact,
    )
    return row


def validate_sample(row, fixture, expected_identity):
    """Validate measured outputs independently before considering exact equality."""
    if row["status"] != "valid":
        return expected_identity
    try:
        payload = row["payload"]
        elapsed = payload["operation_wall_seconds"]
        if isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid operation duration")
        native_count = (
            0 if row["candidate"] == "reference" else len(fixture["request"]["steps"])
        )
        if (
            type(payload["native_transactions"]) is not int
            or payload["native_transactions"] != native_count
        ):
            raise ValueError("native transaction count differs from the request")
        if len(payload["result"]["outputs"]) != len(fixture["request"]["steps"]):
            raise ValueError("output count differs from the request")
        result = full_runner.comparison_result(row)
        if result is None:
            raise ValueError("malformed transaction output")
        full_runner.canonical(result)
        full_runner.canonical(payload)
        output = Path(row["output_directory"])
        request = full_runner.materialize(fixture["request"], output)
        observed = {}
        for label, path in request["files"].items():
            selected = Path(path)
            if selected.parent != output:
                raise ValueError("output file is outside the declared directory")
            observed[label] = "sha256:" + sha_file(selected)
        row["observed_output_files"] = observed
        if observed != result["files"]:
            raise ValueError("retained output file hashes differ from the worker")
        if row["candidate"] in ("reference", "rust-ffi"):
            identity = {
                key: value
                for key, value in payload["identity"].items()
                if key != "check_wall_seconds"
            }
            if not identity or not identity.get("wheel_bindings"):
                raise ValueError("installed runtime identity is missing")
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("installed runtime identity changed")
            expected_identity = identity
        row["status"] = "pending_comparison"
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        OSError,
    ) as error:
        row["status"] = "invalid_output"
        row["validation_error"] = str(error)
        try:
            full_runner.canonical(row.get("payload"))
        except (TypeError, ValueError, UnicodeError):
            row["payload"] = None
            row["payload_retained_in"] = "stdout.txt"
    return expected_identity


def compare_pending(records, directories):
    reference = next((row for row in records if row["candidate"] == "reference"), None)
    expected = (
        full_runner.comparison_result(reference)
        if reference is not None
        and reference["status"] in ("valid", "pending_comparison")
        else None
    )
    for row, directory in zip(records, directories, strict=True):
        if row["status"] in ("valid", "pending_comparison"):
            if expected is None:
                row["output_comparison"] = "reference_unavailable"
            else:
                actual = full_runner.comparison_result(row)
                matches = actual is not None and full_runner.canonical(
                    actual
                ) == full_runner.canonical(expected)
                row["output_comparison"] = (
                    "exact_except_declared_report_destinations"
                    if matches
                    else "different_or_unavailable"
                )
                row["status"] = "valid" if matches else "wrong_output"
        else:
            row["output_comparison"] = "different_or_unavailable"
        target = directory / "sample.json"
        temporary = directory / "sample.pending.json"
        temporary.write_bytes(full_runner.canonical(row) + b"\n")
        temporary.replace(target)
    return all(row["status"] in ("valid", "pending_comparison") for row in records)


def execute(args):
    args.output = args.output.resolve()
    args.metadata = args.metadata.resolve()
    args.python = args.python.absolute()
    raw = args.matrix.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.matrix_sha256.removeprefix("sha256:"):
        raise ValueError("frozen matrix hash mismatch")
    matrix = json.loads(raw)
    if matrix != {
        "candidates": list(CANDIDATES),
        "rounds": 4,
        "seed": 4127,
        "timeout_seconds": 300,
        "total_budget_seconds": 7200,
        "workloads": list(WORKLOADS),
    }:
        raise ValueError("matrix differs from the declared optimization protocol")
    metadata, fixtures = bind_inputs(args.metadata, args.fixtures, args.matrix)
    if [row["name"] for row in fixtures["workloads"]] != matrix["workloads"]:
        raise ValueError("fixture workloads do not match the frozen matrix")
    args.bound_metadata = metadata
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "format": "invarlock/optimization-comparison-v1",
            "metadata": metadata,
            "metadata_sha256": sha_file(args.metadata),
            "matrix": matrix,
            "matrix_sha256": hashlib.sha256(raw).hexdigest(),
            "fixtures_sha256": sha_file(args.fixtures),
            "platform": platform.platform(),
            "python": str(args.python),
            "qualification_only": args.qualification_only,
            "created_unix_seconds": time.time(),
        },
    )
    deadline = time.monotonic() + matrix["total_budget_seconds"]
    records, qualifications = [], []
    expected_identity = None
    stop = None
    validation_error = None
    phases = (
        ("qualification",)
        if args.qualification_only
        else ("qualification", "measurement")
    )
    for phase in phases:
        for index, fixture in enumerate(fixtures["workloads"]):
            rounds = 1 if phase == "qualification" else matrix["rounds"]
            for round_index, order in enumerate(
                balanced_order(rounds, matrix["seed"] + index)
            ):
                if time.monotonic() + 4 * matrix["timeout_seconds"] > deadline:
                    stop = "insufficient_remaining_declared_budget"
                    break
                pending, directories = [], []
                for position, candidate in enumerate(order):
                    directory = (
                        args.output
                        / f"{phase}-{fixture['name']}-{round_index}-{position}-{candidate}"
                    )
                    try:
                        row = sample(
                            candidate,
                            fixture,
                            directory,
                            args,
                            matrix["timeout_seconds"],
                        )
                    except (OSError, ValueError) as error:
                        # No process was dispatched; retain the failed preparation separately.
                        stop = "pre_dispatch_failed"
                        validation_error = str(error)
                        write_json(
                            args.output / "pre-dispatch-failure.json",
                            {
                                "phase": phase,
                                "workload": fixture["name"],
                                "round": round_index,
                                "position": position,
                                "candidate": candidate,
                                "error": str(error),
                            },
                        )
                        break
                    row.update(phase=phase, round=round_index, position=position)
                    expected_identity = validate_sample(row, fixture, expected_identity)
                    pending.append(row)
                    directories.append(directory)
                    if not compare_pending(pending, directories):
                        stop = f"{phase}_failed"
                        break
                (qualifications if phase == "qualification" else records).extend(
                    pending
                )
                if stop:
                    break
            if stop:
                break
        if stop:
            break
        try:
            rebound, _ = bind_inputs(args.metadata, args.fixtures, args.matrix)
            if rebound != metadata:
                raise ValueError("metadata changed during execution")
        except (OSError, KeyError, ValueError) as error:
            stop = "artifact_identity_changed"
            validation_error = str(error)
            break
    qualification_valid = sum(row["status"] == "valid" for row in qualifications)
    timing_valid = sum(row["status"] == "valid" for row in records)
    qualification_complete = (
        qualification_valid == 32 and stop != "artifact_identity_changed"
    )
    comparison_complete = (
        qualification_complete
        and stop is None
        and not args.qualification_only
        and timing_valid == 128
    )
    report = {
        "complete": comparison_complete,
        "comparison_complete": comparison_complete,
        "qualification_complete": qualification_complete,
        "qualification_only": args.qualification_only,
        "stop_reason": stop,
        "validation_error": validation_error,
        "qualification_planned": 32,
        "qualification_attempted": len(qualifications),
        "qualification_valid": qualification_valid,
        "timing_planned": 128,
        "timing_attempted": len(records),
        "timing_valid": timing_valid,
        "retries": 0,
        "exclusions": 0,
        "python_identity": expected_identity,
        "summary": full_runner.summarize(records),
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report))
    return (
        0
        if (qualification_complete if args.qualification_only else comparison_complete)
        else 1
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("python", "metadata", "matrix", "fixtures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--qualification-only", action="store_true")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
