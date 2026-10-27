"""Five-candidate dispatch, identity, and outcome admission boundaries."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from invarlock_rust_experiment import comparison_v2 as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(name):
    return {
        "name": name,
        "files": {},
        "request": {
            "steps": [
                {"operation": "report", "config": {}}
                for _ in range(4 if name == "judge-pack" else 3)
            ],
            "files": {"receipt": "$OUTPUT/receipt.json"},
        },
    }


def valid_row(candidate, item, directory):
    directory.mkdir(parents=True)
    out = directory.parent / (directory.name + "-files")
    out.mkdir()
    (out / "receipt.json").write_text("same")
    statuses = [0] * len(item["request"]["steps"])
    if item["name"] == "captured-regression":
        statuses[0] = 7
    return {
        "candidate": candidate,
        "workload": item["name"],
        "status": "valid",
        "exit_code": 0,
        "timed_out": False,
        "output_directory": str(out),
        "wall_seconds": 1.0,
        "user_cpu_seconds": 0.5,
        "system_cpu_seconds": 0.1,
        "max_rss_bytes": 1024,
        "payload": {
            "ok": True,
            "operation_wall_seconds": 0.2,
            "native_transactions": 0 if candidate == "reference" else len(statuses),
            "identity": {
                "executable": candidate,
                "wheel_bindings": {"native_wheel": candidate},
                "check_wall_seconds": 0.1,
            },
            "result": {
                "outputs": [
                    {"status": value, "payload": {"value": 1}} for value in statuses
                ],
                "files": {"receipt": "sha256:" + runner.sha_file(out / "receipt.json")},
            },
        },
    }


def setup(tmp_path, monkeypatch, qualification_only=False):
    matrix = json.loads(
        (Path(__file__).parents[1] / "matrix-comparison-v2.json").read_text()
    )
    fixtures = {"workloads": [fixture(name) for name in runner.WORKLOADS]}
    args = SimpleNamespace(
        python=Path("/python"),
        metadata=write(tmp_path / "metadata.json", {}),
        fixtures=write(tmp_path / "fixtures.json", fixtures),
        matrix=write(tmp_path / "matrix.json", matrix),
        output=tmp_path / "run",
        qualification_only=qualification_only,
    )
    args.matrix_sha256 = runner.sha_file(args.matrix)
    monkeypatch.setattr(runner, "bind_inputs", lambda *unused: ({}, fixtures))
    calls = []

    def sample(candidate, item, directory, args, timeout):
        calls.append(candidate)
        return valid_row(candidate, item, directory)

    monkeypatch.setattr(runner, "sample", sample)
    return args, calls


def test_five_balanced_positions():
    orders = runner.balanced_order(5, 4127)
    assert orders == runner.balanced_order(5, 4127)
    for candidate in runner.CANDIDATES:
        assert [
            sum(order[pos] == candidate for order in orders) for pos in range(5)
        ] == [1] * 5


def test_complete_40_qualification_200_timing_with_separate_runtime_identities(
    tmp_path, monkeypatch
):
    args, calls = setup(tmp_path, monkeypatch)
    assert runner.execute(args) == 0
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == 240
    assert report["qualification_valid"] == report["qualification_attempted"] == 40
    assert report["timing_valid"] == report["timing_attempted"] == 200
    assert report["complete"] and report["retries"] == report["exclusions"] == 0
    assert (
        report["python_identities"]["rust-ffi-before"]
        != report["python_identities"]["rust-ffi"]
    )


def test_qualification_only(tmp_path, monkeypatch):
    args, calls = setup(tmp_path, monkeypatch, True)
    assert runner.execute(args) == 0
    report = json.loads((args.output / "summary.json").read_text())
    assert (
        len(calls) == 40 and report["qualification_complete"] and not report["complete"]
    )


def test_reserve_five_timeouts_before_admitting_round(tmp_path, monkeypatch):
    args, calls = setup(tmp_path, monkeypatch)
    clock = iter((0, 0, 5701))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_valid"] == 5
    assert report["stop_reason"] == "insufficient_remaining_declared_budget"


@pytest.mark.parametrize("candidate", runner.CANDIDATES)
def test_dispatch_binds_binary_or_distinct_runtime(candidate, tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    metadata = {
        "standalone_before": {"path": str(before), "sha256": runner.sha_file(before)},
        "standalone": {"path": str(after), "sha256": runner.sha_file(after)},
        "measurement_supervisor": {"path": "/supervisor", "sha256": "fixed"},
        "runtimes": {
            name: {
                "python": "/" + name + "/python",
                "metadata": "/" + name + "/metadata",
            }
            for name in ("reference", "rust-ffi-before", "rust-ffi")
        },
    }
    args = SimpleNamespace(bound_metadata=metadata)
    calls = []
    monkeypatch.setattr(runner, "run_process", lambda *args: calls.append(args) or {})
    runner.sample(candidate, fixture("native-pack"), tmp_path / "attempt", args, 300)
    command, directory, timeout, supervisor = calls[0]
    assert timeout == 300 and supervisor == metadata["measurement_supervisor"]
    if candidate.startswith("rust-standalone"):
        assert command[0] == str(before if candidate.endswith("-before") else after)
    else:
        assert command[0] == metadata["runtimes"][candidate]["python"]
        assert "bounded_worker" in command[3]
        assert (
            command[command.index("--metadata") + 1]
            == metadata["runtimes"][candidate]["metadata"]
        )
        assert command[command.index("--candidate") + 1] == (
            "rust-ffi" if candidate == "rust-ffi-before" else candidate
        )


@pytest.mark.parametrize("change", ["domain", "file", "identity", "count", "nonfinite"])
def test_rejects_invalid_output_even_if_all_candidates_agree(change, tmp_path):
    item = fixture("captured-regression")
    row = valid_row("rust-ffi-before", item, tmp_path / "attempt")
    expected = {
        key: value
        for key, value in row["payload"]["identity"].items()
        if key != "check_wall_seconds"
    }
    if change == "domain":
        row["payload"]["result"]["outputs"][0]["status"] = 0
    if change == "file":
        (Path(row["output_directory"]) / "receipt.json").write_text("altered")
    if change == "identity":
        row["payload"]["identity"]["executable"] = "other"
    if change == "count":
        row["payload"]["native_transactions"] = 0
    if change == "nonfinite":
        row["payload"]["result"]["outputs"][0]["payload"]["value"] = float("nan")
    runner.validate_sample(row, item, expected)
    assert row["status"] == "invalid_output"


def test_matching_output_still_requires_reference(tmp_path):
    item = fixture("native-pack")
    directory = tmp_path / "attempt"
    row = valid_row("rust-standalone", item, directory)
    runner.validate_sample(row, item, None)
    assert runner.compare_pending([row], [directory])
    assert (
        row["status"] == "pending_comparison"
        and row["output_comparison"] == "reference_unavailable"
    )


def test_failed_qualification_preserves_attempt_and_stops(tmp_path, monkeypatch):
    args, calls = setup(tmp_path, monkeypatch)
    original = runner.sample

    def failed(*args):
        row = original(*args)
        row["status"] = "measurement_error"
        return row

    monkeypatch.setattr(runner, "sample", failed)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_attempted"] == 1
    assert report["qualification_valid"] == report["timing_attempted"] == 0


def frozen_metadata(tmp_path, monkeypatch):
    metadata = {
        "source_commit": "baseline",
        "source_tree_sha256": "tree",
        "harness_sources": {},
    }
    for key in (
        "root_wheel",
        "native_wheel",
        "dependency_lock",
        "native_cargo_lock",
        "standalone",
        "standalone_before",
        "offline_cli",
        "control_cli",
        "measurement_supervisor",
        "measurement_source",
        "harness_revision",
    ):
        path = tmp_path / key
        path.write_bytes(key.encode())
        metadata[key] = {"path": str(path), "sha256": runner.sha_file(path)}
    source = tmp_path / "harness/comparison_v2.py"
    source.parent.mkdir()
    monkeypatch.setattr(runner, "__file__", str(source))
    for name in runner.HARNESS_FILES:
        path = source.with_name(name)
        path.write_bytes(name.encode())
        metadata["harness_sources"][str(path)] = runner.sha_file(path)
    item = fixture("native-pack")
    for step, operation in zip(
        item["request"]["steps"],
        ("verify-native", "verify-native-receipt", "report"),
        strict=True,
    ):
        step["operation"] = operation
    fixtures = write(tmp_path / "fixtures.json", {"workloads": [item]})
    matrix = write(tmp_path / "matrix.json", {})
    for key, path in (("fixtures_manifest", fixtures), ("matrix", matrix)):
        metadata[key] = {"path": str(path), "sha256": runner.sha_file(path)}
    before = dict(metadata)
    wheel = tmp_path / "before-wheel"
    wheel.write_bytes(b"previous-native")
    before["native_wheel"] = {"path": str(wheel), "sha256": runner.sha_file(wheel)}
    before_path = write(tmp_path / "before-metadata.json", before)
    metadata["before_runtime_metadata"] = {
        "path": str(before_path),
        "sha256": runner.sha_file(before_path),
    }
    python = tmp_path / "python"
    python.write_bytes(b"runtime")
    meta = tmp_path / "metadata.json"
    metadata["runtimes"] = {
        name: {
            "python": str(python),
            "metadata": str(before_path if name == "rust-ffi-before" else meta),
        }
        for name in ("reference", "rust-ffi-before", "rust-ffi")
    }
    write(meta, metadata)
    return meta, fixtures, matrix, metadata, before


@pytest.mark.parametrize(
    "changed",
    [
        "measurement_supervisor",
        "measurement_source",
        "harness_revision",
        "before_runtime_metadata",
        "before_wheel",
        "harness",
        "fixtures_manifest",
    ],
)
def test_binding_rejects_changed_measurement_or_runtime_artifact(
    changed, tmp_path, monkeypatch
):
    meta, fixtures, matrix, metadata, before = frozen_metadata(tmp_path, monkeypatch)
    runner.bind_inputs(meta, fixtures, matrix)
    if changed == "before_wheel":
        path = Path(before["native_wheel"]["path"])
    elif changed == "harness":
        path = Path(next(iter(metadata["harness_sources"])))
    else:
        path = Path(metadata[changed]["path"])
    path.write_bytes(path.read_bytes() + b" changed")
    with pytest.raises(ValueError):
        runner.bind_inputs(meta, fixtures, matrix)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_exit",
        "missing_timeout",
        "missing_errno",
        "string_timeout",
        "boolean_exit",
        "bad_wait_status",
        "partial_json",
        "stdout_read",
        "sample_write",
        "timeout_zero_exit",
    ],
)
def test_post_dispatch_failures_are_attempted_measurements(
    failure, tmp_path, monkeypatch
):
    supervisor = tmp_path / "supervisor"
    supervisor.write_bytes(b"bound")
    identity = {"path": str(supervisor), "sha256": runner.sha_file(supervisor)}
    monkeypatch.setattr(runner.sys, "platform", "linux")
    metrics = {
        "format": "invarlock/linux-process-measurement-v2",
        "exit_code": 0,
        "wait_status": 0,
        "exec_errno": 0,
        "timed_out": False,
        "wall_seconds": 0.1,
        "user_cpu_seconds": 0.01,
        "system_cpu_seconds": 0.01,
        "max_rss_raw": 1024,
        "max_rss_raw_unit": "KiB",
        "max_rss_bytes": 1048576,
    }
    if failure.startswith("missing_"):
        del metrics[
            {
                "missing_exit": "exit_code",
                "missing_timeout": "timed_out",
                "missing_errno": "exec_errno",
            }[failure]
        ]
    if failure == "string_timeout":
        metrics["timed_out"] = "false"
    if failure == "boolean_exit":
        metrics["exit_code"] = False
    if failure == "bad_wait_status":
        metrics["wait_status"] = 256
    if failure == "timeout_zero_exit":
        metrics["timed_out"] = True

    def popen(command, **kwargs):
        Path(command[1]).write_text(
            "{" if failure == "partial_json" else json.dumps(metrics)
        )
        kwargs["stdout"].write(b'{"ok":true}')
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    if failure == "stdout_read":
        original = Path.read_bytes

        def read(path):
            if path.name == "stdout.txt":
                raise OSError("injected post-dispatch read failure")
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", read)
    if failure == "sample_write":

        def fail_write(*args):
            raise OSError("injected storage failure")

        monkeypatch.setattr(runner, "write_json", fail_write)
    row = runner.run_process(["/bin/true"], tmp_path / "attempt", 1, identity)
    assert row["attempt_dispatched"] is True
    assert row["status"] == (
        "timeout" if failure == "timeout_zero_exit" else "measurement_error"
    )
    if failure == "sample_write":
        assert "sample_record_write_error" in row


def test_post_dispatch_sample_record_failure_remains_in_final_denominator(
    tmp_path, monkeypatch
):
    args, calls = setup(tmp_path, monkeypatch)

    def fail_capture(*args):
        raise OSError("injected sample record write failure")

    monkeypatch.setattr(runner, "compare_pending", fail_capture)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_attempted"] == 1
    assert report["qualification_valid"] == report["timing_attempted"] == 0
    assert report["stop_reason"] == "sample_record_write_failed"
