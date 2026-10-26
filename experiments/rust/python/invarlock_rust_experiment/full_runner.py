"""Frozen whole-transaction comparison with retained failures and exact outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

from .fixtures import write_json
from .runner import classify, run_process, verify_metadata
from .worker import sha_file

CANDIDATES = ("reference", "optimized", "rust-ffi", "rust-standalone")
BOOTSTRAP = "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.full_worker import main;raise SystemExit(main())"


def balanced_order(rounds, seed):
    base = list(CANDIDATES)
    random.Random(seed).shuffle(base)
    return [base[index % 4 :] + base[: index % 4] for index in range(rounds)]


def materialize(value, output):
    if isinstance(value, str):
        return value.replace("$OUTPUT", str(output))
    if isinstance(value, list):
        return [materialize(item, output) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item, output) for key, item in value.items()}
    return value


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def bind_inputs(metadata, fixtures):
    verify_metadata(metadata)
    meta = json.loads(metadata.read_bytes())
    for name in ("standalone", "offline_cli"):
        if sha_file(meta[name]["path"]) != meta[name]["sha256"].removeprefix("sha256:"):
            raise ValueError(f"{name} binary changed")
    for path, expected in meta.get("harness_sources", {}).items():
        if sha_file(path) != expected:
            raise ValueError(f"harness source changed: {path}")
    for fixture in fixtures["workloads"]:
        for path, expected in fixture["files"].items():
            if sha_file(path) != expected:
                raise ValueError(f"fixture changed: {path}")
    return meta


def sample(candidate, fixture, directory, args, timeout):
    inputs = directory.parent / (directory.name + "-inputs")
    inputs.mkdir()
    outputs = inputs / "outputs"
    outputs.mkdir()
    request = materialize(fixture["request"], outputs)
    request_path = inputs / "request.json"
    write_json(request_path, request)
    if candidate == "rust-standalone":
        command = [str(args.standalone), str(request_path)]
    else:
        command = [
            str(args.python),
            "-I",
            "-c",
            BOOTSTRAP,
            str(Path(__file__).resolve().parents[1]),
            "--candidate",
            candidate,
            "--request",
            str(request_path),
            "--metadata",
            str(args.metadata),
        ]
    record = run_process(command, directory, timeout)
    record.update(
        candidate=candidate,
        workload=fixture["name"],
        request_sha256=sha_file(request_path),
        output_directory=str(outputs),
    )
    return record


def comparison_result(record):
    payload = record.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        return None
    result = json.loads(json.dumps(payload["result"]))
    if not isinstance(result.get("outputs"), list) or not isinstance(
        result.get("files"), dict
    ):
        return None
    directory = Path(record["output_directory"])
    for outcome in result["outputs"]:
        if not isinstance(outcome, dict) or not isinstance(
            outcome.get("payload"), dict
        ):
            return None
        report = outcome.get("payload", {}).get("report")
        if isinstance(report, dict):
            for field in ("requested_outputs", "written_outputs"):
                for name, path in report.get(field, {}).items():
                    selected = Path(path)
                    if selected.parent != directory:
                        return None
                    report[field][name] = "$OUTPUT/" + selected.name
    return result


def check_round(records, directories):
    reference = next(row for row in records if row["candidate"] == "reference")
    expected = comparison_result(reference)
    for row, directory in zip(records, directories, strict=True):
        payload = row.get("payload")
        actual = comparison_result(row)
        matches = (
            expected is not None
            and actual is not None
            and canonical(actual) == canonical(expected)
        )
        row["output_comparison"] = (
            "exact_except_declared_report_destinations"
            if matches
            else "different_or_unavailable"
        )
        if row["exit_code"] is not None:
            row["status"] = classify(
                row["exit_code"], row["timed_out"], payload, matches
            )
        path = directory / "sample.json"
        path.write_bytes(canonical(row) + b"\n")
    return all(row["status"] == "valid" for row in records)


def summarize(records):
    groups = {}
    for row in records:
        key = (row["workload"], row["candidate"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (workload, candidate), rows in sorted(groups.items()):
        valid = [row for row in rows if row["status"] == "valid"]
        item = {
            "workload": workload,
            "candidate": candidate,
            "attempted": len(rows),
            "valid": len(valid),
            "failures": [row["status"] for row in rows if row["status"] != "valid"],
        }
        if valid:
            for metric in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "max_rss_bytes",
            ):
                values = [row[metric] for row in valid]
                item[metric] = {
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            values = [row["payload"]["operation_wall_seconds"] for row in valid]
            item["operation_wall_seconds"] = {
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        summary.append(item)
    return summary


def execute(args):
    args.output = args.output.resolve()
    args.metadata = args.metadata.resolve()
    args.python = args.python.absolute()
    raw = args.matrix.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.matrix_sha256.removeprefix("sha256:"):
        raise ValueError("frozen matrix hash mismatch")
    matrix = json.loads(raw)
    if matrix["candidates"] != list(CANDIDATES) or matrix["rounds"] != 4:
        raise ValueError(
            "expanded protocol requires four candidates and four balanced rounds"
        )
    fixtures = json.loads(args.fixtures.read_bytes())
    if [row["name"] for row in fixtures["workloads"]] != matrix["workloads"]:
        raise ValueError("fixture workloads do not match the frozen matrix")
    metadata = bind_inputs(args.metadata, fixtures)
    args.standalone = Path(metadata["standalone"]["path"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "metadata": metadata,
            "matrix": matrix,
            "matrix_sha256": hashlib.sha256(raw).hexdigest(),
            "fixtures_sha256": sha_file(args.fixtures),
            "harness_sources": {
                p.name: sha_file(p) for p in Path(__file__).parent.glob("*.py")
            },
            "platform": platform.platform(),
            "created_unix_seconds": time.time(),
        },
    )
    deadline = time.monotonic() + matrix["total_budget_seconds"]
    records = []
    qualifications = []
    stop = None
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
                pending, directories = [], []
                if time.monotonic() + 4 * matrix["timeout_seconds"] > deadline:
                    stop = "insufficient_remaining_declared_budget"
                    break
                for position, candidate in enumerate(order):
                    directory = (
                        args.output
                        / f"{phase}-{fixture['name']}-{round_index}-{position}-{candidate}"
                    )
                    row = sample(
                        candidate, fixture, directory, args, matrix["timeout_seconds"]
                    )
                    row.update(phase=phase, round=round_index, position=position)
                    pending.append(row)
                    directories.append(directory)
                good = check_round(pending, directories)
                (qualifications if phase == "qualification" else records).extend(
                    pending
                )
                if not good:
                    stop = f"{phase}_failed"
                    break
            if stop:
                break
        if stop:
            break
        bind_inputs(args.metadata, fixtures)
    report = {
        "complete": stop is None,
        "stop_reason": stop,
        "qualification_attempted": len(qualifications),
        "qualification_valid": sum(row["status"] == "valid" for row in qualifications),
        "timing_attempted": len(records),
        "timing_valid": sum(row["status"] == "valid" for row in records),
        "retries": 0,
        "exclusions": 0,
        "summary": summarize(records),
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report))
    return 0 if stop is None else 1


def main():
    parser = argparse.ArgumentParser()
    for name in ("python", "metadata", "matrix", "fixtures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--qualification-only", action="store_true")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
