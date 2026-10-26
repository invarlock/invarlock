"""Small conformance checks; these do not establish speed or installed-path validity."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from invarlock_rust_experiment.adapters import candidate_adapter, optimized_interval
from invarlock_rust_experiment.runner import balanced_order, classify


@pytest.mark.parametrize("metric", ["exact_match", "normalized_nll_per_utf8_byte"])
@pytest.mark.parametrize("count", [1, 2, 7])
def test_optimized_matches_reference_exactly(metric, count):
    from invarlock.evidence_pack_contract import _paired_resampling_interval

    args = {
        "metric": metric,
        "baseline_scores": [1 + i / 7 for i in range(count)],
        "subject_scores": [0.75 + (i % 3) / 9 for i in range(count)],
        "schedule_sha256": hashlib.sha256(b"conformance").hexdigest(),
    }
    assert tuple(v.hex() for v in optimized_interval(**args)) == tuple(
        v.hex() for v in _paired_resampling_interval(**args)
    )


def test_adapter_restores_both_aliases_even_on_error():
    from invarlock import evidence_pack_contract as contract
    from invarlock.evaluation_comparison import likelihood

    before = (
        contract._paired_resampling_interval,
        likelihood._paired_resampling_interval,
    )
    with pytest.raises(RuntimeError):
        with candidate_adapter("optimized") as counter:
            assert (
                contract._paired_resampling_interval
                is likelihood._paired_resampling_interval
            )
            contract._paired_resampling_interval(
                metric="exact_match",
                baseline_scores=[1.0],
                subject_scores=[0.0],
                schedule_sha256="0" * 64,
            )
            assert counter.calls == 1
            raise RuntimeError("exercise cleanup")
    assert (
        contract._paired_resampling_interval,
        likelihood._paired_resampling_interval,
    ) == before


def test_failure_accounting_and_balanced_order():
    assert classify(0, False, {"ok": True}, True) == "valid"
    assert classify(0, False, {"ok": True}, False) == "wrong_output"
    assert classify(-9, True, None, False) == "timeout"
    assert classify(-11, False, None, False) == "crash"
    assert classify(2, False, {"ok": False}, False) == "nonzero_exit"
    assert classify(0, False, None, False) == "invalid_output"
    orders = balanced_order(5, 1729)
    assert orders == balanced_order(5, 1729)
    assert all(set(order) == {"reference", "optimized", "rust"} for order in orders)
    for candidate in orders[0]:
        counts = [
            sum(order[position] == candidate for order in orders)
            for position in range(3)
        ]
        assert max(counts) - min(counts) <= 1


def test_real_process_crash_timeout_and_valid_output_are_retained(tmp_path):
    import json

    from invarlock_rust_experiment.runner import run_process

    success = run_process(
        [sys.executable, "-I", "-c", "print('{\"ok\":true}')"], tmp_path / "success", 10
    )
    assert success["status"] == "valid"
    assert success["max_rss_bytes"] > 0
    assert success["user_cpu_seconds"] >= 0
    crash = run_process(
        [
            sys.executable,
            "-I",
            "-c",
            "import os,signal;os.kill(os.getpid(),signal.SIGKILL)",
        ],
        tmp_path / "crash",
        10,
    )
    assert crash["status"] == "crash"
    timeout = run_process(
        [sys.executable, "-I", "-c", "import time;time.sleep(20)"],
        tmp_path / "timeout",
        0.05,
    )
    assert timeout["status"] == "timeout"
    for name, expected in (
        ("success", success),
        ("crash", crash),
        ("timeout", timeout),
    ):
        assert json.loads((tmp_path / name / "sample.json").read_bytes()) == expected
        assert (tmp_path / name / "stdout.txt").exists()
        assert (tmp_path / name / "stderr.txt").exists()
    with pytest.raises(FileExistsError):
        run_process([sys.executable, "-I", "-c", "pass"], tmp_path / "success", 10)


def test_launch_error_is_retained(tmp_path):
    from invarlock_rust_experiment.runner import run_process

    record = run_process([str(tmp_path / "absent-python")], tmp_path / "launch", 1)
    assert record["status"] == "launch_error"
    assert record["exit_code"] is None
    assert (tmp_path / "launch/sample.json").exists()


def test_summary_keeps_failure_denominator_and_uses_only_valid_pairs():
    from invarlock_rust_experiment.runner import summarize

    samples = [
        {
            "count": 1000,
            "stage": "kernel",
            "candidate": "reference",
            "round": 0,
            "status": "valid",
            "wall_seconds": 2.0,
        },
        {
            "count": 1000,
            "stage": "kernel",
            "candidate": "rust",
            "round": 0,
            "status": "wrong_output",
            "wall_seconds": 0.1,
        },
        {
            "count": 1000,
            "stage": "kernel",
            "candidate": "optimized",
            "round": 0,
            "status": "valid",
            "wall_seconds": 1.5,
        },
    ]
    result = summarize(samples)
    assert (result["attempted"], result["valid"], result["failures"]) == (3, 2, 1)
    rust = next(
        group
        for group in result["paired_ratios"]
        if group["candidate"] == "rust" and group["stage"] == "kernel"
    )
    assert rust["valid_pairs"] == 0
    assert rust["median_ratio"] is None
    assert "p95" not in str(result)


def test_command_preserves_virtual_environment_interpreter_symlink(tmp_path):
    from invarlock_rust_experiment.runner import command

    target = tmp_path / "base-python"
    target.write_text("base")
    virtual = tmp_path / "venv-python"
    virtual.symlink_to(target)
    argv = command(virtual.absolute(), tmp_path, "kernel", tmp_path, tmp_path)
    assert argv[0] == str(virtual)
    assert argv[0] != str(target)


def sample(candidate, *, digest="same", status="valid", wall=10.0, operation=1.0):
    return {
        "candidate": candidate,
        "count": 1000,
        "stage": "kernel",
        "round": 0,
        "status": status,
        "execution_status": status,
        "wall_seconds": wall,
        "payload": {
            "ok": status == "valid",
            "result_sha256": digest,
            "operation_wall_seconds": operation,
        },
    }


def test_candidate_crash_preserves_consistent_observed_reference():
    from invarlock_rust_experiment.runner import assess_group

    records = [sample("reference"), sample("optimized"), sample("rust", status="crash")]
    assessment = assess_group(records, 5)
    assert assessment["reference_ok"] is True
    assert assessment["reference_complete"] is False
    assert [r["status"] for r in records] == ["valid", "valid", "crash"]
    assert all(r["execution_status"] == "valid" for r in records[:2])


def test_missing_reference_after_candidate_stop_is_not_reference_failure():
    from invarlock_rust_experiment.runner import assess_group

    records = [sample("optimized"), sample("rust", status="crash")]
    assessment = assess_group(records, 5)
    assert assessment["reference_state"] == "not_observed"
    assert records[0]["status"] == "unevaluable_missing_reference"
    assert records[0]["execution_status"] == "valid"


def test_kernel_primary_summaries_and_pairs_use_operation_time():
    from invarlock_rust_experiment.runner import summarize

    result = summarize(
        [
            sample("reference", wall=10, operation=2),
            sample("rust", wall=8, operation=0.5),
        ]
    )
    rust = next(
        g
        for g in result["groups"]
        if g["candidate"] == "rust" and g["stage"] == "kernel"
    )
    assert rust["primary_time_scope"] == "kernel_operation"
    assert rust["median_operation_wall_seconds"] == 0.5
    pair = next(
        g
        for g in result["paired_ratios"]
        if g["candidate"] == "rust" and g["stage"] == "kernel"
    )
    assert pair["reference_over_candidate_operation_wall_ratios"] == [4]
    assert pair["reference_over_candidate_process_wall_ratios"] == [1.25]
    assert pair["median_ratio"] == 4


def test_wheel_binding_excludes_mutable_distribution_records(tmp_path, monkeypatch):
    import json
    import zipfile

    from invarlock_rust_experiment import worker

    installed = tmp_path / "installed"
    metadata = {}
    for key, prefix in (
        ("root_wheel", "invarlock"),
        ("native_wheel", "invarlock_native"),
    ):
        package = installed / prefix
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"immutable-package")
        wheel_path = tmp_path / f"{prefix}.whl"
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr(f"{prefix}/__init__.py", b"immutable-package")
            wheel.writestr(
                f"{prefix}_experiment-0.1.dist-info/RECORD",
                b"wheel-record-changes-on-install",
            )
        metadata[key] = {
            "path": str(wheel_path),
            "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))

    class Distribution:
        def locate_file(self, name):
            return installed / name

    monkeypatch.setattr(
        worker.importlib.metadata, "distribution", lambda name: Distribution()
    )
    assert (
        worker.check_wheel_binding(path)["native_wheel"]["installed_files_checked"] == 1
    )
    (installed / "invarlock_native/__init__.py").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="installed bytes differ"):
        worker.check_wheel_binding(path)


def test_regression_fixture_records_exact_canonical_decision_enum(tmp_path):
    import json

    from invarlock_rust_experiment.fixtures import make_fixture

    make_fixture(tmp_path / "regression", 2, "regression")
    expected = json.loads((tmp_path / "regression/expected.json").read_bytes())
    report = json.loads(
        (tmp_path / "regression/pack/reports/evaluation.report.json").read_bytes()
    )
    assert expected["decision"] == report["decision"] == "regression"


def test_matrix_stops_on_candidate_crash_without_erasing_valid_pairs(
    tmp_path, monkeypatch
):
    import argparse
    import json

    from invarlock_rust_experiment import runner

    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {"counts": [1000, 5000], "rounds": 5, "seed": 1729, "timeout_seconds": 1}
        )
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    monkeypatch.setattr(runner, "verify_metadata", lambda path: {})
    monkeypatch.setattr(
        runner,
        "balanced_order",
        lambda rounds, seed: [["reference", "optimized", "rust"]] * rounds,
    )
    dispatched = []

    def fake_process(argv, directory, timeout):
        directory.mkdir()
        stage = argv[argv.index("--stage") + 1]
        candidate = argv[argv.index("--candidate") + 1]
        dispatched.append((stage, candidate))
        if stage == "fixture":
            return {"status": "valid"}
        return sample(candidate, status="crash" if candidate == "rust" else "valid")

    monkeypatch.setattr(runner, "run_process", fake_process)
    args = argparse.Namespace(
        matrix=matrix,
        matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        metadata=metadata,
        output=tmp_path / "output",
        python=Path(sys.executable),
    )
    assert runner.execute_matrix(args) == 1
    assert dispatched == [
        ("fixture", "reference"),
        ("kernel", "reference"),
        ("kernel", "optimized"),
        ("kernel", "rust"),
    ]
    assessment = json.loads((args.output / "assessment-n1000-kernel.json").read_bytes())
    assert assessment["reference_ok"] is True
    assert assessment["matrix_complete"] is False
    assert [s["status"] for s in assessment["samples"]] == ["valid", "valid", "crash"]
    result = json.loads((args.output / "summary.json").read_bytes())
    assert (
        result["valid"] == 2 and result["failures"] == 1 and result["unevaluable"] == 0
    )
