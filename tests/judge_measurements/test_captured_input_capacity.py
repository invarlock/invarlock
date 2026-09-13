"""Captured judge inputs share bounded immutable reads with measurement imports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from unittest.mock import Mock

import pytest

from invarlock import captured_evaluation
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.judge_measurements import captured_workflow as workflow
from invarlock.judge_measurements import native_workflow
from tests.core.test_native_judge_transaction import _completed_measurements
from tests.judge_measurements.test_captured_scorer_workflow import _request


@pytest.fixture
def material(tmp_path, monkeypatch):
    path, _, recipe, runs = _request(tmp_path)
    request = load_evaluation_request(path)
    plan, _ = workflow.prepare_evaluator_judge(recipe, *runs)
    measurements = _completed_measurements(
        plan=plan, baseline_run=runs[0], subject_run=runs[1]
    )
    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_text(json.dumps(measurements))
    request = replace(
        request, judge=replace(request.judge, measurements=measurement_path)
    )
    environment = Mock(return_value={"network_calls": 0})
    monkeypatch.setattr(native_workflow, "collection_preflight", environment)
    monkeypatch.setattr(
        native_workflow,
        "collect_frozen",
        lambda **kwargs: pytest.fail("capacity checks must not collect"),
    )
    return request, environment


def _paths(request):
    paths = [request.baseline.path, request.subject.path, request.policy]
    if request.judge.measurements is not None:
        paths.append(request.judge.measurements)
    return paths


def _preflight(request):
    return workflow.preflight_captured_judge(
        request, signing_key_path=None, unsigned=True
    ).payload


def test_measurements_use_their_larger_allowance_not_the_run_byte_limit(
    material, monkeypatch
):
    request, environment = material
    run_limit = max(
        len(source.path.read_bytes()) for source in (request.baseline, request.subject)
    )
    measurement_size = len(request.judge.measurements.read_bytes())
    assert measurement_size > run_limit
    monkeypatch.setattr(workflow, "MAX_INPUT_BYTES", run_limit)
    monkeypatch.setattr(workflow, "MEASUREMENTS_MAX_BYTES", measurement_size)
    result = _preflight(request)
    assert result["measurements"]["status"] == "complete"
    environment.assert_not_called()


@pytest.mark.parametrize("target", ["baseline", "subject", "recipe", "measurements"])
def test_each_input_is_bounded_before_parsing_or_environment(
    material, monkeypatch, target
):
    request, environment = material
    if target in ("baseline", "subject"):
        path = getattr(request, target).path
        path.write_bytes(path.read_bytes() + b" " * 32)
        limit_name = "MAX_INPUT_BYTES"
    elif target == "recipe":
        path, limit_name = request.policy, "NATIVE_POLICY_MAX_BYTES"
    else:
        path, limit_name = request.judge.measurements, "MEASUREMENTS_MAX_BYTES"
    monkeypatch.setattr(workflow, limit_name, len(path.read_bytes()) - 1)
    with pytest.raises(workflow.JudgeWorkflowError, match="byte|limit"):
        _preflight(request)
    environment.assert_not_called()
    assert not request.evidence.exists()
    assert not request.judge.workspace.exists()


@pytest.mark.parametrize("collect", [False, True])
def test_combined_budget_accepts_exact_bytes_and_refuses_one_byte_less(
    material, monkeypatch, collect
):
    request, environment = material
    if collect:
        request = replace(request, judge=replace(request.judge, measurements=None))
    total = sum(len(path.read_bytes()) for path in _paths(request))
    monkeypatch.setattr(workflow, "MAX_WORKFLOW_BYTES", total)
    result = _preflight(request)
    assert result["planned_trials"] == 4
    assert result["collection_available"] is collect
    environment.reset_mock()
    monkeypatch.setattr(workflow, "MAX_WORKFLOW_BYTES", total - 1)
    with pytest.raises(workflow.JudgeWorkflowError, match="byte|limit"):
        _preflight(request)
    environment.assert_not_called()
    assert not request.evidence.exists()
    assert not request.judge.workspace.exists()


def test_exhausted_combined_budget_rejects_before_next_file_read(material, monkeypatch):
    request, environment = material
    monkeypatch.setattr(
        workflow, "MAX_WORKFLOW_BYTES", len(request.baseline.path.read_bytes())
    )
    reads = []
    original = captured_evaluation.read_file

    def observed(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(captured_evaluation, "read_file", observed)
    with pytest.raises(
        workflow.JudgeWorkflowError, match="combined.*byte|byte.*allowance"
    ):
        _preflight(request)
    assert reads == [request.baseline.path]
    environment.assert_not_called()


@pytest.mark.parametrize("collect", [False, True])
def test_preflight_reads_each_input_once_and_parses_those_exact_bytes(
    material, monkeypatch, collect
):
    request, environment = material
    if collect:
        request = replace(request, judge=replace(request.judge, measurements=None))
    paths = _paths(request)
    original = captured_evaluation.read_file
    reads = Counter()

    def read_then_replace(path, limit):
        raw = original(path, limit)
        reads[path] += 1
        path.write_text("file changed after its secured read")
        return raw

    monkeypatch.setattr(captured_evaluation, "read_file", read_then_replace)
    result = _preflight(request)
    assert result["planned_trials"] == 4
    assert result["collection_available"] is collect
    assert reads == Counter(dict.fromkeys(paths, 1))
    if collect:
        assert result["maximum_admitted_calls"] >= result["planned_trials"]
        environment.assert_called_once()
    else:
        environment.assert_not_called()


def test_default_captured_read_limit_and_run_loading_stay_unchanged(
    material, monkeypatch
):
    request, _ = material
    original = captured_evaluation.read_file
    observed = []

    def read(path, limit):
        observed.append(limit)
        return original(path, limit)

    monkeypatch.setattr(captured_evaluation, "read_file", read)
    assert captured_evaluation._run(request, "baseline")["run_id"] == "baseline"
    assert observed == [captured_evaluation.contracts.MAX_INPUT_BYTES]
