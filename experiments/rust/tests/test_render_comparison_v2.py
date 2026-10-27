"""Validate the new five-candidate result path using tiny retained artifacts."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "render_comparison_v2", Path(__file__).parents[1] / "render_comparison_v2.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report.canonical(value) + b"\n")


def fixture_tree(root):
    original = Path("/retained")
    run = root / "runs/primary"
    run.mkdir(parents=True)

    def artifact(name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": str(original / name), "sha256": report.sha(path)}

    env = {"machine": "aarch64", "platform": "Linux-test", "work_filesystem": "ext4"}
    metadata = {
        "source_commit": "a" * 40,
        "before_source": {"commit": "b" * 40, "native_tree": "d" * 40},
        "after_source": {"commit": "c" * 40, "native_tree": "e" * 40},
        "harness_sources": {},
        "supervisor_compiler": "test compiler",
    }
    for name in (
        "root_wheel",
        "native_wheel",
        "standalone_before",
        "standalone",
        "offline_cli",
        "measurement_supervisor",
        "measurement_source",
        "harness_revision",
    ):
        metadata[name] = artifact("artifacts/" + name, name.encode())
    metadata["environment"] = artifact("environment.json", report.canonical(env))
    matrix = {
        "candidates": list(report.CANDIDATES),
        "rounds": 5,
        "seed": 4127,
        "timeout_seconds": 300,
        "total_budget_seconds": 7200,
        "workloads": list(report.WORKLOADS),
    }
    metadata["matrix"] = artifact("inputs/matrix.json", report.canonical(matrix))
    fixtures = {"workloads": []}
    for workload in report.WORKLOADS:
        request = {
            "steps": [
                {"operation": op, "config": {}}
                for op in report.expected_operations(workload)
            ],
            "files": {"receipt": "$OUTPUT/receipt.json"},
        }
        fixtures["workloads"].append(
            {"name": workload, "request": request, "files": {}}
        )
    metadata["fixtures_manifest"] = artifact(
        "fixtures.json", report.canonical(fixtures)
    )
    files = {}
    for name in ("before", "after"):
        descriptor = artifact("inputs/" + name + ".tar", name.encode())
        files[name + ".tar"] = descriptor["sha256"]
    bundle = {
        "before": metadata["before_source"],
        "after": metadata["after_source"],
        "files": files,
    }
    metadata["input_bundle"] = artifact("inputs/bundle.json", report.canonical(bundle))
    before = copy.deepcopy(metadata)
    before["native_wheel"] = artifact("artifacts/native_wheel_before", b"before wheel")
    before["standalone"] = metadata["standalone_before"]
    metadata["before_runtime_metadata"] = artifact(
        "before-runtime-metadata.json", report.canonical(before)
    )
    metadata["runtimes"] = {
        candidate: {
            "python": "/retained/"
            + ("before-venv" if candidate.endswith("-before") else "venv")
            + "/bin/python",
            "metadata": "/retained/"
            + (
                "before-runtime-metadata.json"
                if candidate.endswith("-before")
                else "metadata.json"
            ),
        }
        for candidate in ("reference", "rust-ffi-before", "rust-ffi")
    }
    write(root / "metadata.json", metadata)
    manifest = {
        "format": "invarlock/five-candidate-comparison-v2",
        "metadata": metadata,
        "metadata_sha256": report.sha(root / "metadata.json"),
        "matrix": matrix,
        "matrix_sha256": metadata["matrix"]["sha256"],
        "fixtures_sha256": metadata["fixtures_manifest"]["sha256"],
        "python": "/retained/venv/bin/python",
        "qualification_only": False,
    }
    write(run / "manifest.json", manifest)
    identities, groups = {}, {}
    for phase in ("qualification", "measurement"):
        for wi, fixture in enumerate(fixtures["workloads"]):
            for round_index in range(1 if phase == "qualification" else 5):
                for position, candidate in enumerate(report.order(round_index, wi)):
                    name = f"{phase}-{fixture['name']}-{round_index}-{position}-{candidate}"
                    directory = run / name
                    output = "/retained/runs/primary/" + name + "-inputs/outputs"
                    inp = run / (name + "-inputs")
                    request = report.materialize(fixture["request"], output)
                    write(inp / "request.json", request)
                    write(inp / "outputs/receipt.json", {"constant": True})
                    hashes = {
                        "receipt": "sha256:" + report.sha(inp / "outputs/receipt.json")
                    }
                    statuses = (
                        [7, 0, 0]
                        if fixture["name"] == "captured-regression"
                        else [0] * len(request["steps"])
                    )
                    payload = {
                        "ok": True,
                        "native_transactions": 0
                        if candidate == "reference"
                        else len(request["steps"]),
                        "operation_wall_seconds": 0.1,
                        "result": {
                            "files": hashes,
                            "outputs": [
                                {"status": status, "payload": {"value": 1}}
                                for status in statuses
                            ],
                        },
                    }
                    if candidate in metadata["runtimes"]:
                        runtime = metadata["runtimes"][candidate]
                        runtime_metadata = (
                            before if candidate == "rust-ffi-before" else metadata
                        )
                        identity = {
                            "python": "3.12.13 test",
                            "native_version": "test",
                            "executable": runtime["python"],
                            "wheel_bindings": {
                                key: {
                                    "sha256": runtime_metadata[key]["sha256"],
                                    "installed_files_checked": 1,
                                }
                                for key in ("root_wheel", "native_wheel")
                            },
                        }
                        identities[candidate] = identity
                        payload["identity"] = {**identity, "check_wall_seconds": 0.01}
                        command = [
                            runtime["python"],
                            "-I",
                            "-c",
                            "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.bounded_worker import main;raise SystemExit(main())",
                            "/retained/inputs/harness",
                            "--candidate",
                            "rust-ffi" if candidate == "rust-ffi-before" else candidate,
                            "--request",
                            "/retained/runs/primary/" + name + "-inputs/request.json",
                            "--metadata",
                            runtime["metadata"],
                        ]
                        executable = None
                    else:
                        executable = metadata[
                            "standalone_before"
                            if candidate.endswith("-before")
                            else "standalone"
                        ]
                        command = [
                            executable["path"],
                            "/retained/runs/primary/" + name + "-inputs/request.json",
                        ]
                    factor = report.CANDIDATES.index(candidate) + 1
                    raw = {
                        "format": "invarlock/linux-process-measurement-v2",
                        "exit_code": 0,
                        "wait_status": 0,
                        "timed_out": False,
                        "exec_errno": 0,
                        "wall_seconds": factor + round_index / 10,
                        "user_cpu_seconds": (10 if round_index < 2 else 1) * factor,
                        "system_cpu_seconds": (10 if round_index in (2, 3) else 1)
                        * factor,
                        "max_rss_raw": factor * 100,
                        "max_rss_raw_unit": "KiB",
                        "max_rss_bytes": factor * 102400,
                    }
                    row = {
                        **raw,
                        "attempt_dispatched": True,
                        "phase": phase,
                        "round": round_index,
                        "position": position,
                        "candidate": candidate,
                        "workload": fixture["name"],
                        "status": "valid",
                        "output_comparison": "exact_except_declared_report_destinations",
                        "measurement_supervisor": metadata["measurement_supervisor"],
                        "platform": "linux",
                        "command": command,
                        "executable_artifact": executable,
                        "payload": payload,
                        "output_directory": output,
                        "request_sha256": report.sha(inp / "request.json"),
                        "observed_output_files": hashes,
                    }
                    write(directory / "sample.json", row)
                    write(directory / "process-metrics.json", raw)
                    write(directory / "stdout.txt", payload)
                    (directory / "stderr.txt").write_bytes(b"")
                    if phase == "measurement":
                        groups.setdefault((fixture["name"], candidate), []).append(
                            {**raw, "operation_wall_seconds": 0.1}
                        )
    summary = []
    for (workload, candidate), rows in sorted(groups.items()):
        item = {
            "workload": workload,
            "candidate": candidate,
            "attempted": 5,
            "valid": 5,
            "failures": [],
        }
        for key in report.METRICS:
            vals = [r[key] for r in rows]
            item[key] = {
                "median": report.statistics.median(vals),
                "minimum": min(vals),
                "maximum": max(vals),
            }
        summary.append(item)
    write(
        run / "summary.json",
        {
            "complete": True,
            "comparison_complete": True,
            "qualification_complete": True,
            "qualification_only": False,
            "stop_reason": None,
            "validation_error": None,
            "qualification_planned": 40,
            "qualification_attempted": 40,
            "qualification_valid": 40,
            "timing_planned": 200,
            "timing_attempted": 200,
            "timing_valid": 200,
            "retries": 0,
            "exclusions": 0,
            "python_identities": {c: identities.get(c) for c in report.CANDIDATES},
            "summary": summary,
        },
    )
    return run


@pytest.fixture
def retained(tmp_path):
    return tmp_path, fixture_tree(tmp_path)


def validate(root, run):
    return report.validate(
        run,
        root / "metadata.json",
        root / "environment.json",
        expected_python="a" * 40,
        expected_before="b" * 40,
        expected_after="c" * 40,
    )


def test_complete_path_has_five_candidates_and_separate_before_after(retained):
    root, run = retained
    result = validate(root, run)
    assert result["qualification_valid"] == 40
    assert result["timing_valid"] == 200
    assert len(result["samples"]) == 240
    assert len(result["summary"]) == 40
    assert len(result["derived"]["versus_python"]) == 32
    assert len(result["derived"]["before_after"]) == 16
    cells = {(r["workload"], r["candidate"]): r for r in result["summary"]}
    # CPU component medians are 1 + 1; median of per-attempt totals is 11.
    assert cells["native-pack", "reference"]["total_cpu_seconds"] == {
        "median": 11,
        "minimum": 2,
        "maximum": 11,
    }
    modes = {
        r["mode"]: r
        for r in result["derived"]["before_after"]
        if r["workload"] == "native-pack"
    }
    assert (
        modes["ffi"]["metrics"]["wall_seconds"]["after_over_before_ratio"] == 4.2 / 3.2
    )
    assert (
        modes["standalone"]["metrics"]["wall_seconds"]["after_over_before_ratio"]
        == 5.2 / 2.2
    )
    assert all(
        s["domain_statuses"] == [7, 0, 0]
        for s in result["samples"]
        if s["workload"] == "captured-regression"
    )
    report.canonical(result)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "No such file"),
        ("orphan", "orphaned"),
        ("raw", "raw process metrics"),
        ("rss", "RSS unit"),
        ("statuses", "domain statuses"),
        ("payload", "exact outputs differ"),
        ("summary", "recomputed sample"),
        ("source", "unexpected final"),
        ("wheel", "before native wheel bytes"),
        ("supervisor", "supervisor identity"),
        ("identity", "installed runtime identity"),
        ("count", "timing_valid"),
    ],
)
def test_retained_inconsistencies_fail_closed(retained, mutation, message):
    root, run = retained
    path = next(run.glob("measurement-native-pack-*-rust-ffi/sample.json"))
    row = report.read(path)
    if mutation == "missing":
        path.unlink()
    elif mutation == "orphan":
        (run / "orphan-attempt").mkdir()
    elif mutation in ("raw", "rss"):
        row["max_rss_bytes"] += 1
        write(path, row)
        if mutation == "rss":
            raw = report.read(path.parent / "process-metrics.json")
            raw["max_rss_bytes"] += 1
            write(path.parent / "process-metrics.json", raw)
    elif mutation in ("statuses", "payload"):
        if mutation == "statuses":
            row["payload"]["result"]["outputs"][0]["status"] = 7
        else:
            row["payload"]["result"]["outputs"][0]["payload"]["value"] = 2
        write(path, row)
        write(path.parent / "stdout.txt", row["payload"])
    elif mutation in ("summary", "count"):
        p = run / "summary.json"
        x = report.read(p)
        if mutation == "summary":
            x["summary"][0]["wall_seconds"]["median"] += 1
        else:
            x["timing_valid"] = 199
        write(p, x)
    elif mutation == "source":
        with pytest.raises(ValueError, match=message):
            report.validate(
                run,
                root / "metadata.json",
                root / "environment.json",
                expected_python="a" * 40,
                expected_before="b" * 40,
                expected_after="f" * 40,
            )
        return
    elif mutation == "wheel":
        (root / "artifacts/native_wheel_before").write_bytes(b"wrong")
    elif mutation == "supervisor":
        row["measurement_supervisor"]["sha256"] = "0" * 64
        write(path, row)
    elif mutation == "identity":
        row["payload"]["identity"]["native_version"] = "different"
        write(path, row)
        write(path.parent / "stdout.txt", row["payload"])
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        validate(root, run)


def test_order_balances_all_five_positions():
    for wi in range(8):
        rounds = [report.order(r, wi) for r in range(5)]
        assert all(
            {row[p] for row in rounds} == set(report.CANDIDATES) for p in range(5)
        )


def test_zero_denominator_is_unavailable_not_infinite():
    result = report.median_change(1, 0)
    assert result["after_over_before_ratio"] is None
    assert result["ratio_unavailable_reason"] == "zero_before_median"
    assert result["observed_median_status"] == "loss"
    report.canonical(result)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_invalid_json_evidence_is_rejected(tmp_path, raw):
    p = tmp_path / "invalid.json"
    p.write_text(raw)
    with pytest.raises(ValueError):
        report.read(p)


def test_process_metrics_reject_false_dispatch_and_boolean_exit():
    raw = {
        "format": "invarlock/linux-process-measurement-v2",
        "exit_code": 0,
        "wait_status": 0,
        "timed_out": False,
        "exec_errno": 0,
        "wall_seconds": 1.0,
        "user_cpu_seconds": 0.5,
        "system_cpu_seconds": 0.1,
        "max_rss_raw": 100,
        "max_rss_raw_unit": "KiB",
        "max_rss_bytes": 102400,
    }
    row = {**raw, "attempt_dispatched": False, "measurement_supervisor": {}}
    with pytest.raises(ValueError, match="dispatch"):
        report.validate_process_metrics(row, raw, {})
    row["attempt_dispatched"] = True
    row["exit_code"] = raw["exit_code"] = False
    with pytest.raises(ValueError, match="exit"):
        report.validate_process_metrics(row, raw, {})
