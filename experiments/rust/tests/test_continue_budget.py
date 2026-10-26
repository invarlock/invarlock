"""A continuation must never retry, skip or reorder a measured attempt."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "continue_budget", Path(__file__).parents[1] / "recovery" / "continue_budget.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CANDIDATES = ["reference", "optimized", "rust-ffi", "rust-standalone"]
MATRIX = {"workloads": ["small", "large"], "seed": 4127}


def order(rounds, seed):
    offset = seed % 4
    base = CANDIDATES[offset:] + CANDIDATES[:offset]
    return [base[i:] + base[:i] for i in range(rounds)]


def records():
    return [
        {
            "workload": name,
            "round": round_index,
            "position": position,
            "candidate": candidate,
            "status": "valid",
        }
        for index, name in enumerate(MATRIX["workloads"])
        for round_index, candidates in enumerate(order(4, MATRIX["seed"] + index))
        for position, candidate in enumerate(candidates)
    ]


@pytest.mark.parametrize("completed", range(9))
def test_all_complete_prefixes_resume_only_remaining_rounds(completed):
    expected = [(name, i) for name in MATRIX["workloads"] for i in range(4)]
    assert (
        MODULE.remaining_rounds(MATRIX, records()[: 4 * completed], order)
        == expected[completed:]
    )


@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 7, 31])
def test_partial_round_is_not_retried(count):
    with pytest.raises(ValueError, match="partial, duplicate, or reordered"):
        MODULE.remaining_rounds(MATRIX, records()[:count], order)


@pytest.mark.parametrize(
    "failure", ["crashed", "timed_out", "output_mismatch", "pending"]
)
def test_failed_round_cannot_be_continued(failure):
    rows = records()[:4]
    rows[0]["status"] = failure
    with pytest.raises(ValueError, match="unsuccessful"):
        MODULE.remaining_rounds(MATRIX, rows, order)


def test_duplicate_attempt_rejected():
    rows = records()[:4]
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.remaining_rounds(MATRIX, rows, order)


def test_changed_candidate_order_rejected():
    rows = records()[:4]
    rows[0]["candidate"], rows[1]["candidate"] = (
        rows[1]["candidate"],
        rows[0]["candidate"],
    )
    with pytest.raises(ValueError, match="reordered"):
        MODULE.remaining_rounds(MATRIX, rows, order)


def test_gap_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        MODULE.remaining_rounds(MATRIX, records()[4:8], order)


def test_unordered_file_discovery_does_not_change_original_order():
    rows = records()[:12]
    rows.reverse()
    assert MODULE.remaining_rounds(MATRIX, rows, order)[0] == ("small", 3)


def runtime_row(executable="/qualified/python", version="3.12"):
    return {
        "candidate": "reference",
        "command": [executable],
        "payload": {"identity": {"python": version, "check_wall_seconds": 0.1}},
    }


def test_different_python_executable_rejected():
    with pytest.raises(ValueError, match="qualified executable"):
        MODULE.validate_python([runtime_row()], Path("/other/python"))


def test_changed_retained_runtime_rejected():
    with pytest.raises(ValueError, match="runtime identities"):
        MODULE.validate_python(
            [runtime_row(), runtime_row(version="3.13")], Path("/qualified/python")
        )


def test_identity_timing_is_not_identity():
    row = runtime_row()
    later = copy.deepcopy(row)
    later["payload"]["identity"]["check_wall_seconds"] = 0.2
    assert MODULE.validate_python([row, later], Path("/qualified/python")) == {
        "python": "3.12"
    }


@pytest.mark.parametrize("missing", ["sample", "request", "neither"])
def test_incomplete_attempt_directories_rejected(tmp_path, missing):
    attempt = tmp_path / "measurement-small-0-0-reference"
    inputs = tmp_path / (attempt.name + "-inputs")
    attempt.mkdir()
    inputs.mkdir()
    if missing != "sample":
        (attempt / "sample.json").write_text("{}")
    if missing != "request":
        (inputs / "request.json").write_text("{}")
    if missing == "neither":
        assert MODULE.recorded_attempts(tmp_path) == [attempt / "sample.json"]
    else:
        with pytest.raises(ValueError, match="interrupted"):
            MODULE.recorded_attempts(tmp_path)


def test_alternate_harness_tree_rejected(tmp_path):
    with pytest.raises(ValueError, match="frozen harness"):
        MODULE.validate_runner(
            tmp_path / "runner.py", {"/qualified/runner.py": "same"}, lambda _: "same"
        )


def test_changed_harness_bytes_rejected(tmp_path):
    path = tmp_path / "runner.py"
    with pytest.raises(ValueError, match="frozen harness"):
        MODULE.validate_runner(path, {str(path): "original"}, lambda _: "changed")
