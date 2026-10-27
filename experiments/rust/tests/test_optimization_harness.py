"""Guard the fresh comparison's artifact, output and dispatch boundaries."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from invarlock_rust_experiment import optimization_runner as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def matrix():
    return {
        "candidates": list(runner.CANDIDATES),
        "rounds": 4,
        "seed": 4127,
        "timeout_seconds": 300,
        "total_budget_seconds": 7200,
        "workloads": list(runner.WORKLOADS),
    }


def fixture(name="captured-nll-1000"):
    return {
        "name": name,
        "files": {},
        "request": {
            "steps": [{"operation": "report", "config": {}}],
            "files": {"receipt": "$OUTPUT/receipt.json"},
        },
    }


def valid_row(candidate, item, directory):
    directory.mkdir(parents=True)
    output = directory.parent / (directory.name + "-inputs") / "outputs"
    output.mkdir(parents=True)
    receipt = output / "receipt.json"
    receipt.write_text("unchanged")
    return {
        "candidate": candidate,
        "workload": item["name"],
        "status": "valid",
        "exit_code": 0,
        "timed_out": False,
        "output_directory": str(output),
        "wall_seconds": 1.0,
        "user_cpu_seconds": 0.8,
        "system_cpu_seconds": 0.1,
        "max_rss_bytes": 1024,
        "payload": {
            "ok": True,
            "operation_wall_seconds": 0.5,
            "native_transactions": 0 if candidate == "reference" else 1,
            "identity": {
                "executable": "/python",
                "wheel_bindings": {"root_wheel": "same", "native_wheel": "same"},
                "check_wall_seconds": 0.1,
            },
            "result": {
                "outputs": [{"status": 0, "payload": {"value": 1}}],
                "files": {"receipt": "sha256:" + runner.sha_file(receipt)},
            },
        },
    }


def test_balance_and_deterministic_order():
    orders = runner.balanced_order(4, 4127)
    assert orders == runner.balanced_order(4, 4127)
    for candidate in runner.CANDIDATES:
        assert [
            sum(order[pos] == candidate for order in orders) for pos in range(4)
        ] == [1] * 4


@pytest.mark.parametrize("candidate", runner.CANDIDATES)
def test_dispatch_uses_distinct_frozen_binaries_and_existing_worker(
    tmp_path, monkeypatch, candidate
):
    before, after = tmp_path / "before", tmp_path / "after"
    before.write_bytes(b"old")
    after.write_bytes(b"new")
    args = SimpleNamespace(
        python=Path("/installed/python"),
        metadata=Path("/metadata"),
        bound_metadata={
            "standalone_before": {
                "path": str(before),
                "sha256": runner.sha_file(before),
            },
            "standalone": {"path": str(after), "sha256": runner.sha_file(after)},
        },
    )
    calls = []
    monkeypatch.setattr(
        runner,
        "run_process",
        lambda command, directory, timeout: calls.append((command, timeout)) or {},
    )
    runner.sample(candidate, fixture(), tmp_path / "attempt", args, 300)
    command, timeout = calls[0]
    assert timeout == 300
    if candidate.startswith("rust-standalone"):
        assert command[0] == str(before if candidate.endswith("-before") else after)
    else:
        assert command[:4] == [
            "/installed/python",
            "-I",
            "-c",
            runner.full_runner.BOOTSTRAP,
        ]
        assert command[command.index("--candidate") + 1] == candidate


def artifacts(tmp_path, monkeypatch):
    metadata = {
        "source_commit": "fixed",
        "source_tree_sha256": "fixed",
        "harness_sources": {},
    }
    for name in (
        "root_wheel",
        "native_wheel",
        "dependency_lock",
        "native_cargo_lock",
        "standalone",
        "standalone_before",
        "offline_cli",
        "control_cli",
    ):
        path = tmp_path / name
        path.write_text(name)
        metadata[name] = {"path": str(path), "sha256": runner.sha_file(path)}
    source = tmp_path / "optimization_runner.py"
    source.write_text("frozen")
    monkeypatch.setattr(runner, "__file__", str(source))
    for name in runner.HARNESS_FILES:
        path = source.with_name(name)
        path.write_text(name)
        metadata["harness_sources"][str(path)] = runner.sha_file(path)
    fixtures = write(tmp_path / "fixtures.json", {"workloads": [fixture()]})
    frozen_matrix = write(tmp_path / "matrix.json", matrix())
    for name, path in (("fixtures_manifest", fixtures), ("matrix", frozen_matrix)):
        metadata[name] = {"path": str(path), "sha256": runner.sha_file(path)}
    meta = write(tmp_path / "metadata.json", metadata)
    return meta, fixtures, frozen_matrix, metadata


@pytest.mark.parametrize(
    "changed",
    [
        "standalone_before",
        "standalone",
        "native_wheel",
        "control_cli",
        "fixtures_manifest",
        "matrix",
        "harness",
    ],
)
def test_changed_artifact_prevents_binding(tmp_path, monkeypatch, changed):
    meta, fixtures, frozen_matrix, metadata = artifacts(tmp_path, monkeypatch)
    runner.bind_inputs(meta, fixtures, frozen_matrix)
    if changed == "harness":
        target = Path(next(iter(metadata["harness_sources"])))
    else:
        target = Path(metadata[changed]["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        runner.bind_inputs(meta, fixtures, frozen_matrix)


def test_output_files_are_rehashed_not_trusted(tmp_path):
    row = valid_row("rust-ffi", fixture(), tmp_path / "attempt")
    (Path(row["output_directory"]) / "receipt.json").write_text("changed on disk")
    runner.validate_sample(row, fixture(), None)
    assert row["status"] == "invalid_output"
    assert "file hashes" in row["validation_error"]


@pytest.mark.parametrize("change", ["native_count", "runtime", "output_count"])
def test_count_and_runtime_drift_fail(tmp_path, change):
    row = valid_row("rust-ffi", fixture(), tmp_path / "attempt")
    expected = {
        key: value
        for key, value in row["payload"]["identity"].items()
        if key != "check_wall_seconds"
    }
    if change == "native_count":
        row["payload"]["native_transactions"] = 0
    elif change == "runtime":
        row["payload"]["identity"]["executable"] = "/other/python"
    else:
        row["payload"]["result"]["outputs"] = []
    runner.validate_sample(row, fixture(), expected)
    assert row["status"] == "invalid_output"


def test_exact_output_mismatch_and_reference_unavailable_are_retained(tmp_path):
    item = fixture()
    directories = [tmp_path / "candidate", tmp_path / "reference"]
    rows = [
        valid_row(candidate, item, directory)
        for candidate, directory in zip(
            ("rust-standalone", "reference"), directories, strict=True
        )
    ]
    for row in rows:
        runner.validate_sample(row, item, None)
    assert runner.compare_pending(rows[:1], directories[:1])
    assert rows[0]["status"] == "pending_comparison"
    assert rows[0]["output_comparison"] == "reference_unavailable"
    rows[0]["payload"]["result"]["outputs"][0]["payload"]["value"] = 2
    assert not runner.compare_pending(rows, directories)
    assert rows[0]["status"] == "wrong_output"
    assert (
        json.loads((directories[0] / "sample.json").read_text())["status"]
        == "wrong_output"
    )


def setup_execution(tmp_path, monkeypatch, *, qualification_only=False):
    fixtures = {"workloads": [fixture(name) for name in runner.WORKLOADS]}
    args = SimpleNamespace(
        python=Path("/installed/python"),
        metadata=write(tmp_path / "metadata.json", {}),
        fixtures=write(tmp_path / "fixtures.json", fixtures),
        matrix=write(tmp_path / "matrix.json", matrix()),
        output=tmp_path / "run",
        qualification_only=qualification_only,
    )
    args.matrix_sha256 = runner.sha_file(args.matrix)
    monkeypatch.setattr(runner, "bind_inputs", lambda *unused: ({}, fixtures))
    calls = []

    def sample(candidate, item, directory, args, timeout):
        calls.append((candidate, item["name"]))
        return valid_row(candidate, item, directory)

    monkeypatch.setattr(runner, "sample", sample)
    return args, calls


def test_complete_comparison_keeps_32_qualification_and_128_measurement_attempts(
    tmp_path, monkeypatch
):
    args, calls = setup_execution(tmp_path, monkeypatch)
    assert runner.execute(args) == 0
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == 160
    assert report["complete"] and report["comparison_complete"]
    assert report["qualification_attempted"] == report["qualification_valid"] == 32
    assert report["timing_attempted"] == report["timing_valid"] == 128
    assert report["retries"] == report["exclusions"] == 0


def test_qualification_only_is_not_a_complete_comparison(tmp_path, monkeypatch):
    args, calls = setup_execution(tmp_path, monkeypatch, qualification_only=True)
    assert runner.execute(args) == 0
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == 32
    assert report["qualification_complete"]
    assert not report["complete"] and not report["comparison_complete"]
    assert report["timing_attempted"] == 0 and report["timing_planned"] == 128


def test_failed_qualification_stops_dispatch_and_retains_denominator(
    tmp_path, monkeypatch
):
    args, calls = setup_execution(tmp_path, monkeypatch)
    original = runner.sample

    def failing(*args):
        row = original(*args)
        row["payload"]["native_transactions"] = 999
        return row

    monkeypatch.setattr(runner, "sample", failing)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_attempted"] == 1
    assert report["qualification_valid"] == report["timing_attempted"] == 0
    assert report["stop_reason"] == "qualification_failed"
    paths = list(args.output.glob("qualification-*/sample.json"))
    assert (
        len(paths) == 1
        and json.loads(paths[0].read_text())["status"] == "invalid_output"
    )


def test_budget_guard_preserves_prior_attempts_and_never_dispatches_a_partial_round(
    tmp_path, monkeypatch
):
    args, calls = setup_execution(tmp_path, monkeypatch)
    clock = iter((0, 0, 6001))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert (
        len(calls)
        == report["qualification_attempted"]
        == report["qualification_valid"]
        == 4
    )
    assert report["timing_attempted"] == 0
    assert report["stop_reason"] == "insufficient_remaining_declared_budget"


def test_pre_dispatch_failure_is_separate_from_process_denominator(
    tmp_path, monkeypatch
):
    args, _ = setup_execution(tmp_path, monkeypatch)

    def failing(*unused):
        raise ValueError("binary changed before dispatch")

    monkeypatch.setattr(runner, "sample", failing)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert report["qualification_attempted"] == report["timing_attempted"] == 0
    assert report["stop_reason"] == "pre_dispatch_failed"
    assert (args.output / "pre-dispatch-failure.json").is_file()


@pytest.mark.parametrize(
    "status", ["timeout", "crash", "nonzero_exit", "invalid_output", "launch_error"]
)
def test_process_failures_stop_without_becoming_valid(tmp_path, monkeypatch, status):
    args, calls = setup_execution(tmp_path, monkeypatch)
    original = runner.sample

    def failing(*args):
        row = original(*args)
        row["status"] = status
        row["payload"] = None
        return row

    monkeypatch.setattr(runner, "sample", failing)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_attempted"] == 1
    assert report["qualification_valid"] == report["timing_attempted"] == 0
    retained = next(args.output.glob("qualification-*/sample.json"))
    assert json.loads(retained.read_text())["status"] == status


def test_budget_stop_after_qualification_keeps_qualification_complete(
    tmp_path, monkeypatch
):
    args, calls = setup_execution(tmp_path, monkeypatch)
    clock = iter([0] * 9 + [6001])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_valid"] == 32
    assert report["qualification_complete"] and not report["comparison_complete"]
    assert report["timing_attempted"] == 0


def test_actual_process_record_is_updated_after_independent_validation(tmp_path):
    item = fixture()
    row = valid_row("reference", item, tmp_path / "source")
    measured = tmp_path / "measured"
    command = [
        sys.executable,
        "-I",
        "-c",
        "import sys; print(sys.argv[1])",
        json.dumps(row["payload"]),
    ]
    captured = runner.run_process(command, measured, 5)
    assert captured["status"] == "valid"
    captured.update(
        candidate="reference",
        workload=item["name"],
        output_directory=row["output_directory"],
    )
    runner.validate_sample(captured, item, None)
    assert runner.compare_pending([captured], [measured])
    retained = json.loads((measured / "sample.json").read_text())
    assert retained["status"] == "valid"
    assert retained["observed_output_files"] == row["payload"]["result"]["files"]
    assert retained["output_comparison"] == "exact_except_declared_report_destinations"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_is_rejected_and_raw_payload_retained(tmp_path, value):
    item = fixture()
    directory = tmp_path / "attempt"
    row = valid_row("reference", item, directory)
    row["payload"]["result"]["outputs"][0]["payload"]["value"] = value
    raw = json.dumps(row["payload"])
    (directory / "stdout.txt").write_text(raw)
    runner.validate_sample(row, item, None)
    assert row["status"] == "invalid_output"
    assert row["payload"] is None
    assert row["payload_retained_in"] == "stdout.txt"
    assert not runner.compare_pending([row], [directory])
    retained = json.loads((directory / "sample.json").read_text())
    runner.full_runner.canonical(retained)
    assert retained["status"] == "invalid_output"
    assert (directory / "stdout.txt").read_text() == raw


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_actual_nonfinite_process_output_keeps_final_denominator(
    tmp_path, monkeypatch, value
):
    args, calls = setup_execution(tmp_path, monkeypatch)

    def malformed_sample(candidate, item, directory, unused_args, timeout):
        calls.append((candidate, item["name"]))
        row = valid_row(candidate, item, tmp_path / "source")
        row["payload"]["result"]["outputs"][0]["payload"]["value"] = value
        raw = json.dumps(row["payload"])
        command = [sys.executable, "-I", "-c", "import sys; print(sys.argv[1])", raw]
        captured = runner.run_process(command, directory, timeout)
        captured.update(
            candidate=candidate,
            workload=item["name"],
            output_directory=row["output_directory"],
        )
        return captured

    monkeypatch.setattr(runner, "sample", malformed_sample)
    assert runner.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert len(calls) == report["qualification_attempted"] == 1
    assert report["qualification_valid"] == report["timing_attempted"] == 0
    assert report["stop_reason"] == "qualification_failed"
    retained_path = next(args.output.glob("qualification-*/sample.json"))
    retained = json.loads(retained_path.read_text())
    runner.full_runner.canonical(retained)
    assert retained["status"] == "invalid_output" and retained["payload"] is None
    assert retained["payload_retained_in"] == "stdout.txt"
    assert (retained_path.parent / "stdout.txt").read_text().strip()
