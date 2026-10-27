"""Five-candidate comparison using fresh-exec Linux measurement v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
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
    "rust-ffi-before",
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
    "comparison_v2.py",
    "bounded_worker.py",
    "full_runner.py",
    "full_worker.py",
    "runner.py",
    "worker.py",
    "adapters.py",
    "fixtures.py",
)


def measurement_failure(
    command, directory, supervisor, error, supervisor_exit_code=None
):
    """A dispatched supervisor remains an attempted sample even if capture fails."""
    record = {
        "command": command,
        "exit_code": None,
        "timed_out": False,
        "payload": None,
        "status": "measurement_error",
        "output_comparison": "unavailable",
        "measurement_supervisor": supervisor,
        "supervisor_exit_code": supervisor_exit_code,
        "attempt_dispatched": True,
        "validation_error": str(error),
    }
    try:
        write_json(directory / "sample.json", record)
    except (OSError, ValueError) as capture_error:
        # The enclosing summary still counts the attempt when this sample's
        # storage is unavailable. Never relabel a dispatched process as absent.
        record["sample_record_write_error"] = str(capture_error)
    return record


def run_process(command, directory, timeout, supervisor):
    """Measure the real child from a freshly exec'd small Linux supervisor."""
    if not sys.platform.startswith("linux"):
        raise ValueError("measurement v2 requires Linux")
    if sha_file(supervisor["path"]) != supervisor["sha256"].removeprefix("sha256:"):
        raise ValueError("measurement supervisor changed before dispatch")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metrics = directory / "process-metrics.json"
    with (
        (directory / "stdout.txt").open("wb") as stdout,
        (directory / "stderr.txt").open("wb") as stderr,
    ):
        # Failures before Popen returns have no dispatched supervisor. Every
        # failure after it returns is converted into an attempted sample.
        process = subprocess.Popen(
            [supervisor["path"], str(metrics), str(timeout), "--", *command],
            cwd=directory,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.wait()
        except OSError as error:
            return measurement_failure(
                command, directory, supervisor, error, process.returncode
            )
    try:
        if process.returncode or not metrics.is_file():
            raise ValueError(
                "supervisor failed or did not produce measurement metadata"
            )
        record = json.loads(metrics.read_bytes())
        if record["format"] != "invarlock/linux-process-measurement-v2":
            raise ValueError("unexpected measurement format")
        for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds"):
            if (
                isinstance(record[key], bool)
                or not math.isfinite(record[key])
                or record[key] < 0
            ):
                raise ValueError("invalid measured resource value")
        for key in (
            "exit_code",
            "wait_status",
            "exec_errno",
            "max_rss_raw",
            "max_rss_bytes",
        ):
            if type(record[key]) is not int:
                raise ValueError("measurement integer field has invalid type")
        if type(record["timed_out"]) is not bool:
            raise ValueError("measurement timeout flag has invalid type")
        if record["exec_errno"] < 0 or record["max_rss_raw"] < 0:
            raise ValueError("negative errno or RSS")
        if (
            record["max_rss_raw_unit"] != "KiB"
            or record["max_rss_bytes"] != record["max_rss_raw"] * 1024
        ):
            raise ValueError("measurement RSS unit conversion differs")
        if os.waitstatus_to_exitcode(record["wait_status"]) != record["exit_code"]:
            raise ValueError("measurement exit code differs from wait status")
        record.update(
            command=command,
            platform=sys.platform,
            payload=None,
            output_comparison="pending",
            measurement_supervisor=supervisor,
            attempt_dispatched=True,
        )
        try:
            record["payload"] = json.loads((directory / "stdout.txt").read_bytes())
            full_runner.canonical(record["payload"])
        except (ValueError, UnicodeError) as error:
            record.update(
                payload=None,
                validation_error=str(error),
                payload_retained_in="stdout.txt",
            )
        record["status"] = (
            "launch_error"
            if record["exec_errno"]
            else classify(
                record["exit_code"], record["timed_out"], record["payload"], True
            )
        )
        write_json(directory / "sample.json", record)
        return record
    except (OSError, ValueError, KeyError, TypeError) as error:
        return measurement_failure(
            command, directory, supervisor, error, process.returncode
        )


def balanced_order(rounds, seed):
    base = list(CANDIDATES)
    random.Random(seed).shuffle(base)
    return [
        base[index % len(base) :] + base[: index % len(base)] for index in range(rounds)
    ]


def bind_inputs(metadata_path, fixtures_path, matrix_path):
    fixtures = json.loads(fixtures_path.read_bytes())
    metadata = full_runner.bind_inputs(metadata_path, fixtures)
    for fixture in fixtures["workloads"]:
        name = fixture["name"]
        expected = (
            ["verify-captured", "verify-captured-receipt", "report"]
            if name.startswith("captured-")
            else ["verify-native", "verify-native-receipt", "report"]
            if name == "native-pack"
            else [
                "verify-judge",
                "write-judge-receipt",
                "verify-judge-receipt",
                "report",
            ]
            if name == "judge-pack"
            else ["verify-evidence-set", "verify-stored-evidence-set", "report"]
            if name == "evidence-set"
            else None
        )
        if (
            name not in WORKLOADS
            or [step["operation"] for step in fixture["request"]["steps"]] != expected
        ):
            raise ValueError(
                "fixture operations differ from independently audited domain contract"
            )
    for name, supplied in (
        ("fixtures_manifest", fixtures_path),
        ("matrix", matrix_path),
    ):
        item = metadata[name]
        expected = item["sha256"].removeprefix("sha256:")
        if sha_file(supplied) != expected or sha_file(item["path"]) != expected:
            raise ValueError(f"{name} identity changed")
    for name in (
        "standalone_before",
        "standalone",
        "offline_cli",
        "control_cli",
        "measurement_supervisor",
        "measurement_source",
        "before_runtime_metadata",
        "harness_revision",
    ):
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
    before_path = Path(metadata["before_runtime_metadata"]["path"])
    before = full_runner.bind_inputs(before_path, fixtures)
    if (
        before["root_wheel"] != metadata["root_wheel"]
        or before["source_commit"] != metadata["source_commit"]
    ):
        raise ValueError("Python baseline differs between runtime environments")
    if before["native_wheel"]["sha256"] == metadata["native_wheel"]["sha256"]:
        raise ValueError("before and after native wheels must be independently bound")
    for candidate in ("reference", "rust-ffi-before", "rust-ffi"):
        runtime = metadata["runtimes"][candidate]
        if not Path(runtime["python"]).is_file():
            raise ValueError("runtime Python missing")
        expected_meta = before_path if candidate == "rust-ffi-before" else metadata_path
        if Path(runtime["metadata"]) != expected_meta:
            raise ValueError("runtime metadata binding differs")
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
    elif candidate in ("reference", "rust-ffi-before", "rust-ffi"):
        command = [
            args.bound_metadata["runtimes"][candidate]["python"],
            "-I",
            "-c",
            "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.bounded_worker import main;raise SystemExit(main())",
            str(Path(__file__).resolve().parents[1]),
            "--candidate",
            "rust-ffi" if candidate == "rust-ffi-before" else candidate,
            "--request",
            str(request_path),
            "--metadata",
            args.bound_metadata["runtimes"][candidate]["metadata"],
        ]
    else:
        raise ValueError(f"unknown candidate: {candidate}")
    request_sha256 = sha_file(request_path)
    row = run_process(
        command, directory, timeout, args.bound_metadata["measurement_supervisor"]
    )
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
        statuses = [item["status"] for item in payload["result"]["outputs"]]
        expected_statuses = (
            [7, 0, 0]
            if fixture["name"] == "captured-regression"
            else [0] * (4 if fixture["name"] == "judge-pack" else 3)
        )
        if (
            any(type(value) is not int for value in statuses)
            or statuses != expected_statuses
        ):
            raise ValueError(
                "domain statuses differ from the independently audited fixture contract"
            )
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
        if row["candidate"] in ("reference", "rust-ffi-before", "rust-ffi"):
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
        "rounds": 5,
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
            "format": "invarlock/five-candidate-comparison-v2",
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
    expected_identities = {}
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
                if (
                    time.monotonic() + len(CANDIDATES) * matrix["timeout_seconds"]
                    > deadline
                ):
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
                    expected_identities[candidate] = validate_sample(
                        row, fixture, expected_identities.get(candidate)
                    )
                    pending.append(row)
                    directories.append(directory)
                    try:
                        comparison_valid = compare_pending(pending, directories)
                    except (OSError, ValueError) as error:
                        row["status"] = "measurement_error"
                        row["validation_error"] = str(error)
                        stop = "sample_record_write_failed"
                        validation_error = str(error)
                        break
                    if not comparison_valid:
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
        qualification_valid == 40 and stop != "artifact_identity_changed"
    )
    comparison_complete = (
        qualification_complete
        and stop is None
        and not args.qualification_only
        and timing_valid == 200
    )
    report = {
        "complete": comparison_complete,
        "comparison_complete": comparison_complete,
        "qualification_complete": qualification_complete,
        "qualification_only": args.qualification_only,
        "stop_reason": stop,
        "validation_error": validation_error,
        "qualification_planned": 40,
        "qualification_attempted": len(qualifications),
        "qualification_valid": qualification_valid,
        "timing_planned": 200,
        "timing_attempted": len(records),
        "timing_valid": timing_valid,
        "retries": 0,
        "exclusions": 0,
        "python_identities": expected_identities,
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
