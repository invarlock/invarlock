"""Bounded regression audit of the selected native source adapters."""

import json
from dataclasses import replace

import pytest

from invarlock.captured_evaluation import (
    CapturedEvaluationError,
    evaluate_captured_request,
    preflight_captured_request,
)
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.evaluation_records.templates import example_project
from tests.core.test_captured_evaluation import _request


@pytest.mark.parametrize("preflight", [False, True])
def test_judge_configuration_guard_precedes_native_dispatch(
    tmp_path, monkeypatch, preflight
):
    from invarlock import native_backend

    root = tmp_path.resolve()
    request = _request(root, *example_project("classification"))
    request = replace(request, judge=object())

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid judge configuration reached native dispatch")

    monkeypatch.setattr(native_backend, "captured_request", forbidden)
    call = preflight_captured_request if preflight else evaluate_captured_request
    with pytest.raises(
        CapturedEvaluationError, match="judge configuration requires metric: judge"
    ):
        call(request, unsigned=True, backend="rust")


def test_scoring_failure_keeps_public_error_context(tmp_path):
    root = tmp_path.resolve()
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = [policy["metrics"][0]]
    policy["metrics"][0].update(kind="numeric_tolerance", configuration={})
    for run in (baseline, subject):
        run["records"][0]["expected"] = "not a numeric target"
    request = _request(root, baseline, subject, policy)
    assert (
        preflight_captured_request(request, unsigned=True, backend="rust").record_count
        == 40
    )
    with pytest.raises(CapturedEvaluationError) as expected:
        evaluate_captured_request(request, unsigned=True)
    with pytest.raises(CapturedEvaluationError) as actual:
        evaluate_captured_request(request, unsigned=True, backend="rust")
    assert str(actual.value) == str(expected.value)
    assert not request.evidence.exists()


def test_explicit_jsonl_import_adapter_preserves_publication(tmp_path):
    root = tmp_path.resolve()
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = [policy["metrics"][0]]
    authored = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {"policy": "policy.json"},
        "output": {"evidence": "reference"},
    }
    for side, run in [("baseline", baseline), ("subject", subject)]:
        (root / f"{side}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in run["records"])
        )
        authored["comparison"][side] = {
            "path": f"{side}.jsonl",
            "adapter": "jsonl",
            "source": run["source"],
            "run_id": run["run_id"],
            "artifact_digest": run["artifact_digest"],
        }
    (root / "policy.json").write_text(json.dumps(policy))
    (root / "request.json").write_text(json.dumps(authored))
    request = load_evaluation_request(root / "request.json")
    expected = evaluate_captured_request(request, unsigned=True)
    actual = evaluate_captured_request(
        replace(request, evidence=root / "native"), unsigned=True, backend="rust"
    )

    def files(path):
        return {
            p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob("*")
            if p.is_file()
        }

    assert files(expected.evidence_path) == files(actual.evidence_path)


def test_large_nonnegative_bootstrap_budget_is_valid(tmp_path):
    root = tmp_path.resolve()
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = [policy["metrics"][0]]
    request = _request(root, baseline, subject, policy)
    budget = 10**80
    expected = evaluate_captured_request(
        request, unsigned=True, max_bootstrap_draws=budget
    )
    actual = evaluate_captured_request(
        replace(request, evidence=root / "native"),
        unsigned=True,
        max_bootstrap_draws=budget,
        backend="rust",
    )
    assert replace(actual, evidence_path=expected.evidence_path) == expected
