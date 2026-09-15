"""Actual fixed subprocess outcomes, with no model quality claim."""

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from invarlock.engine import compare_runs, digest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/hosted-service/agent_outcomes.py"
)


def module():
    spec = importlib.util.spec_from_file_location("agent_outcomes", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def fixture():
    script = module()
    return script, script.capture_fixture()


def project(script, capture):
    return script.project_capture(
        capture,
        expected_capture_digest=digest(capture),
        expected_harness_digest=script.harness_digest(),
    )


def test_actual_files_and_tests_override_identical_success_text(fixture):
    script, capture = fixture
    baseline, subject = capture["baseline"], capture["subject"]
    assert baseline["final_message"] == subject["final_message"]
    assert baseline["candidate_digest"] != subject["candidate_digest"]
    assert baseline["returncode"] == 0
    assert subject["returncode"] == 1
    assert "OK" in baseline["stderr"]
    assert "AssertionError: 2 != 1" in subject["stderr"]
    runs = project(script, capture)
    assert runs["baseline"]["records"][0]["output"] == "pass"
    assert runs["subject"]["records"][0]["output"] == "fail"
    assert runs["subject"]["records"][0]["expected"] == "pass"
    assert (
        runs["subject"]["records"][0]["context"]["final_message"]
        == script.FINAL_MESSAGE
    )
    assert runs["subject"]["artifact_digest"] == subject["candidate_digest"]
    assert runs["subject"]["source_digest"] == digest(capture)


def test_exact_match_scores_check_outcomes_not_completion_message(fixture):
    script, capture = fixture
    runs = project(script, capture)
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "task_completion",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 0.1,
                "maximum_interval_width": 1,
                "subject_minimum": 0.5,
            }
        ],
    }
    comparison = compare_runs(
        baseline=runs["baseline"], subject=runs["subject"], policy=policy
    )
    result = comparison["metrics"][0]
    assert result["baseline_mean"] == 1
    assert result["subject_mean"] == 0
    assert comparison["decision"] == "regression"


def test_each_fixed_check_uses_its_own_removed_directory(monkeypatch):
    script = module()
    actual_run = script.subprocess.run
    seen = []

    def run(command, **kwargs):
        seen.append(kwargs["cwd"])
        assert command == [script.sys.executable, "-I", "check.py"]
        assert kwargs["timeout"] == script.TIMEOUT_SECONDS
        assert kwargs["env"] == {}
        return actual_run(command, **kwargs)

    monkeypatch.setattr(script.subprocess, "run", run)
    script.capture_fixture()
    assert len(set(seen)) == 2
    assert all(not path.exists() for path in seen)


def test_capture_pin_rejects_changed_subject_even_with_updated_inner_digest(fixture):
    script, capture = fixture
    expected = digest(capture)
    capture["subject"]["candidate_files"]["solution.py"] += "# changed\n"
    capture["subject"]["candidate_digest"] = digest(
        capture["subject"]["candidate_files"]
    )
    with pytest.raises(ValueError, match="capture digest"):
        script.project_capture(
            capture,
            expected_capture_digest=expected,
            expected_harness_digest=script.harness_digest(),
        )


@pytest.mark.parametrize(
    "binding", ["starting", "test", "candidate", "outcome", "harness"]
)
def test_inconsistent_bindings_rejected_with_recomputed_outer_pin(fixture, binding):
    script, original = fixture
    capture = copy.deepcopy(original)
    if binding == "starting":
        capture["starting_files"]["solution.py"] += "# changed\n"
    elif binding == "test":
        capture["test_files"]["check.py"] = "print('OK')\n"
    elif binding == "candidate":
        capture["subject"]["candidate_files"] = capture["baseline"]["candidate_files"]
    elif binding == "outcome":
        capture["subject"]["outcome"] = "pass"
    else:
        capture["harness_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="binding|outcome|harness"):
        project(script, capture)


def test_projection_never_executes_imported_files(fixture, monkeypatch):
    script, capture = fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("offline projection executed a process")

    monkeypatch.setattr(script.subprocess, "run", forbidden)
    assert project(script, capture)["subject"]["records"][0]["output"] == "fail"


def test_timeout_is_a_retained_failure(monkeypatch):
    script = module()

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(script.subprocess, "run", timeout)
    capture = script.capture_fixture()
    assert capture["subject"]["timed_out"] is True
    assert capture["subject"]["returncode"] is None
    assert project(script, capture)["subject"]["records"][0]["output"] == "fail"


def test_output_limit_refuses_completed_capture(monkeypatch):
    script = module()

    def excessive(command, **kwargs):
        kwargs["stdout"].write(b"x" * (script.MAX_OUTPUT_BYTES + 1))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", excessive)
    with pytest.raises(ValueError, match="output exceeds"):
        script.capture_fixture()


def test_fixed_check_cannot_mutate_retained_candidate(monkeypatch):
    script = module()

    def mutate(command, **kwargs):
        (kwargs["cwd"] / "solution.py").write_text("changed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="changed retained files"):
        script.capture_fixture()


def test_cli_emits_canonical_runs_and_refuses_repeated_destination(tmp_path, capsys):
    script = module()
    output = tmp_path / "outcomes"
    assert script.main(["--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    capture = json.loads((output / "capture.json").read_text())
    assert result["capture_digest"] == digest(capture)
    assert (
        result["capture_digest"]
        == "sha256:"
        + hashlib.sha256((output / "capture.json").read_bytes()).hexdigest()
    )
    assert result["baseline"] == "pass" and result["subject"] == "fail"
    assert (
        json.loads((output / "subject.json").read_text())["records"][0]["output"]
        == "fail"
    )
    with pytest.raises(FileExistsError):
        script.main(["--output", str(output)])


def test_unknown_candidate_is_not_an_execution_interface(monkeypatch):
    script = module()
    with pytest.raises(ValueError, match="unknown fixed candidate"):
        script._capture_variant("arbitrary.py")
