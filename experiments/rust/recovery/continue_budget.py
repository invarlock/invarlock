"""Continue only unmeasured balanced rounds after the declared budget guard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def remaining_rounds(matrix, rows, order):
    expected = [
        (name, round_index) for name in matrix["workloads"] for round_index in range(4)
    ]
    groups = {}
    for row in rows:
        key = (row["workload"], row["round"])
        if key not in expected or row["status"] != "valid":
            raise ValueError("unknown or unsuccessful measured attempt")
        groups.setdefault(key, []).append(row)
    actual = [key for key in expected if key in groups]
    if actual != expected[: len(actual)]:
        raise ValueError("measured rounds are not a contiguous matrix prefix")
    for name, round_index in actual:
        observed = sorted(groups[name, round_index], key=lambda row: row["position"])
        planned = order(4, matrix["seed"] + matrix["workloads"].index(name))[
            round_index
        ]
        if (
            len(observed) != 4
            or [row["position"] for row in observed] != list(range(4))
            or [row["candidate"] for row in observed] != planned
        ):
            raise ValueError("partial, duplicate, or reordered measured round")
    return expected[len(actual) :]


def stable_identity(identity):
    return {
        key: value for key, value in identity.items() if key != "check_wall_seconds"
    }


def validate_python(records, python):
    selected = [row for row in records if row["candidate"] != "rust-standalone"]
    if not selected or any(row["command"][0] != str(python) for row in selected):
        raise ValueError("continuation Python differs from the qualified executable")
    expected = stable_identity(selected[0]["payload"]["identity"])
    if any(stable_identity(row["payload"]["identity"]) != expected for row in selected):
        raise ValueError("retained runtime identities differ")
    return expected


def recorded_attempts(prior):
    paths = []
    for directory in prior.iterdir():
        if not directory.is_dir():
            continue
        if not directory.name.startswith(("qualification-", "measurement-")):
            raise ValueError("unexpected directory in retained run")
        attempt = directory.parent / directory.name.removesuffix("-inputs")
        inputs = attempt.parent / (attempt.name + "-inputs")
        if (
            not (attempt / "sample.json").is_file()
            or not (inputs / "request.json").is_file()
        ):
            raise ValueError("unrecorded or interrupted attempt forbids continuation")
        if directory == attempt:
            paths.append(attempt / "sample.json")
    return paths


def validate_runner(module_path, sources, sha_file):
    path = str(Path(module_path).resolve())
    if path not in sources or sha_file(path) != sources[path]:
        raise ValueError("imported runner differs from frozen harness")


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "experiment",
        "prior",
        "metadata",
        "fixtures",
        "python",
        "rule",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.experiment.resolve()))
    from invarlock_rust_experiment import full_runner
    from invarlock_rust_experiment.fixtures import write_json
    from invarlock_rust_experiment.full_runner import (
        CANDIDATES,
        balanced_order,
        bind_inputs,
        canonical,
        check_round,
        comparison_result,
        materialize,
        sample,
        summarize,
    )
    from invarlock_rust_experiment.worker import sha_file

    prior = json.loads((args.prior / "summary.json").read_bytes())
    manifest = json.loads((args.prior / "manifest.json").read_bytes())
    matrix = manifest["matrix"]
    runner_hash = sha_file(__file__)
    rule = json.loads(args.rule.read_bytes())
    if (
        prior["complete"]
        or prior["stop_reason"] != "insufficient_remaining_declared_budget"
    ):
        raise ValueError("only the declared budget guard permits continuation")
    if matrix["candidates"] != list(CANDIDATES) or matrix["rounds"] != 4:
        raise ValueError("unexpected original matrix")
    if (
        rule["per_process_timeout_seconds"] != matrix["timeout_seconds"]
        or rule["planned_timing_attempts_unchanged"] != 128
    ):
        raise ValueError("continuation changed the attempt or timeout contract")
    args.metadata = args.metadata.resolve()
    args.python = args.python.absolute()
    fixtures = json.loads(args.fixtures.read_bytes())
    metadata = bind_inputs(args.metadata, fixtures)
    validate_runner(full_runner.__file__, metadata["harness_sources"], sha_file)
    frozen_matrix = Path(metadata["matrix"]["path"])
    if (
        json.loads(frozen_matrix.read_bytes()) != matrix
        or sha_file(frozen_matrix) != manifest["matrix_sha256"]
        or sha_file(frozen_matrix) != metadata["matrix"]["sha256"]
    ):
        raise ValueError("original matrix changed")
    if (
        metadata != manifest["metadata"]
        or sha_file(args.fixtures) != manifest["fixtures_sha256"]
    ):
        raise ValueError("artifacts or fixtures changed")
    records = []
    for path in recorded_attempts(args.prior):
        row = json.loads(path.read_bytes())
        if row.get("phase") not in ("qualification", "measurement"):
            raise ValueError("unknown original attempt phase")
        if (
            row.get("status") != "valid"
            or row.get("output_comparison")
            != "exact_except_declared_report_destinations"
        ):
            raise ValueError(
                "a failed, pending or interrupted attempt forbids continuation"
            )
        if json.loads((path.parent / "stdout.txt").read_bytes()) != row["payload"]:
            raise ValueError("retained stdout no longer matches its recorded result")
        request_path = (
            path.parent.parent / (path.parent.name + "-inputs") / "request.json"
        )
        if sha_file(request_path) != row["request_sha256"]:
            raise ValueError("retained request changed")
        request = json.loads(request_path.read_bytes())
        fixture = next(f for f in fixtures["workloads"] if f["name"] == row["workload"])
        if request != materialize(fixture["request"], Path(row["output_directory"])):
            raise ValueError("retained request differs from the frozen fixture")
        for label, file in request["files"].items():
            if "sha256:" + sha_file(file) != row["payload"]["result"]["files"][label]:
                raise ValueError("retained output file changed")
        expected_name = f"{row['phase']}-{row['workload']}-{row['round']}-{row['position']}-{row['candidate']}"
        if path.parent.name != expected_name:
            raise ValueError("retained attempt directory differs from its record")
        records.append(row)
    qualifications = [r for r in records if r.get("phase") == "qualification"]
    measured = [r for r in records if r.get("phase") == "measurement"]
    if len(qualifications) != 32 or len(measured) != prior["timing_attempted"]:
        raise ValueError("original qualification or measurement denominator changed")
    for workload in matrix["workloads"]:
        group = [r for r in qualifications if r["workload"] == workload]
        observed = sorted(group, key=lambda row: row["position"])
        planned = balanced_order(
            1, matrix["seed"] + matrix["workloads"].index(workload)
        )[0]
        if (
            len(group) != 4
            or any(r["round"] != 0 for r in group)
            or [r["position"] for r in observed] != list(range(4))
            or [r["candidate"] for r in observed] != planned
        ):
            raise ValueError("qualification coverage or order changed")
    expected_identity = validate_python(records, args.python)
    identity_code = (
        "import sys,json;sys.path.insert(0,sys.argv.pop(1));"
        "from invarlock_rust_experiment.worker import runtime_identity,check_wheel_binding;"
        "facts=runtime_identity();facts['wheel_bindings']=check_wheel_binding(sys.argv[1]);"
        "print(json.dumps(facts))"
    )
    identity_process = subprocess.run(
        [
            str(args.python),
            "-I",
            "-c",
            identity_code,
            str(args.experiment),
            str(args.metadata),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if json.loads(identity_process.stdout) != expected_identity:
        raise ValueError("current runtime identity differs from retained qualification")
    pending = remaining_rounds(matrix, measured, balanced_order)
    # Recheck all retained round outputs without rewriting the original records.
    for phase, selected in (
        ("qualification", qualifications),
        ("measurement", measured),
    ):
        for key in {(r["workload"], r["round"]) for r in selected}:
            group = [r for r in selected if (r["workload"], r["round"]) == key]
            reference = next(r for r in group if r["candidate"] == "reference")
            expected = comparison_result(reference)
            if expected is None or any(
                canonical(comparison_result(r)) != canonical(expected) for r in group
            ):
                raise ValueError(f"retained {phase} outputs differ")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.standalone = Path(metadata["standalone"]["path"])
    write_json(
        args.output / "manifest.json",
        {
            "format": "invarlock/budget-continuation-v1",
            "parent": str(args.prior.resolve()),
            "parent_manifest_sha256": sha_file(args.prior / "manifest.json"),
            "parent_summary_sha256": sha_file(args.prior / "summary.json"),
            "rule_sha256": sha_file(args.rule),
            "runner_sha256": runner_hash,
            "metadata": metadata,
            "python_identity": expected_identity,
            "matrix": matrix,
            "fixtures_sha256": sha_file(args.fixtures),
            "unmeasured_rounds": pending,
            "qualification_reused": 32,
            "budget_seconds": rule["continuation_limit_seconds"],
            "created_unix_seconds": time.time(),
        },
    )
    deadline = time.monotonic() + rule["continuation_limit_seconds"]
    added = []
    stop = None
    for workload, round_index in pending:
        if time.monotonic() + 4 * matrix["timeout_seconds"] > deadline:
            stop = "insufficient_remaining_declared_budget"
            break
        fixture = next(f for f in fixtures["workloads"] if f["name"] == workload)
        order = balanced_order(4, matrix["seed"] + matrix["workloads"].index(workload))[
            round_index
        ]
        group, directories = [], []
        for position, candidate in enumerate(order):
            directory = (
                args.output
                / f"measurement-{workload}-{round_index}-{position}-{candidate}"
            )
            row = sample(candidate, fixture, directory, args, matrix["timeout_seconds"])
            row.update(phase="measurement", round=round_index, position=position)
            group.append(row)
            directories.append(directory)
        good = check_round(group, directories)
        added.extend(group)
        if any(
            row["candidate"] != "rust-standalone"
            and stable_identity(row.get("payload", {}).get("identity", {}))
            != expected_identity
            for row in group
        ):
            stop = "runtime_identity_changed"
            break
        if not good:
            stop = "measurement_failed"
            break
    bind_inputs(args.metadata, fixtures)
    if sha_file(__file__) != runner_hash:
        raise ValueError("continuation runner changed during measurement")
    combined = measured + added
    complete = (
        stop is None
        and len(combined) == 128
        and not remaining_rounds(matrix, combined, balanced_order)
    )
    report = {
        "complete": complete,
        "stop_reason": stop,
        "qualification_attempted": 32,
        "qualification_valid": 32,
        "qualification_reused_not_repeated": 32,
        "timing_attempted": len(combined),
        "timing_valid": sum(r["status"] == "valid" for r in combined),
        "chunk_timing_attempts": [len(measured), len(added)],
        "retries": 0,
        "exclusions": 0,
        "summary": summarize(combined),
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
