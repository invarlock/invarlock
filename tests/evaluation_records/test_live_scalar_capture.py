"""Actual local SDK metrics with an explicitly synthetic task; no model calls."""

from __future__ import annotations

import builtins
import copy
import importlib.metadata
import importlib.util
import os
import socket
from functools import wraps
from pathlib import Path

import pytest

from invarlock.evaluation_records.scalar_integrations import export_records

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/integrations/evaluator-live/scalar.py"
)
SPEC = importlib.util.spec_from_file_location("live_scalar_example", SOURCE)
assert SPEC and SPEC.loader
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)


def _require_sdk(evaluator, monkeypatch):
    package = "evaluate" if evaluator == "hugging-face-evaluate" else evaluator
    required = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", evaluator)
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != LIVE.VERSIONS[evaluator]:
        if required:
            pytest.fail(
                f"requires {package}=={LIVE.VERSIONS[evaluator]}, found {installed}"
            )
        pytest.skip(f"optional pinned {package} SDK not installed")
    for key, value in {
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
        "RAGAS_DO_NOT_TRACK": "true",
        "OPIK_TRACK_DISABLE": "true",
        "OTEL_SDK_DISABLED": "true",
        "LANGSMITH_TRACING": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(key, value)

    def forbid_network(*args, **kwargs):
        raise AssertionError(
            "live scalar test uses a synthetic task and local SDK metrics"
        )

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)


def _observe_real_metric(evaluator, monkeypatch):
    if evaluator == "deepeval":
        from deepeval.metrics import ExactMatchMetric

        target, name = ExactMatchMetric, "measure"
    elif evaluator == "ragas":
        from ragas.metrics.collections import ExactMatch

        target, name = ExactMatch, "ascore"
    elif evaluator == "hugging-face-evaluate":
        from evaluate import Metric

        target, name = Metric, "compute"
    elif evaluator == "autoevals":
        from autoevals import ExactMatch

        target, name = ExactMatch, "__call__"
    elif evaluator == "openevals":
        from openevals import exact

        target, name = exact, "exact_match"
    elif evaluator == "arize-phoenix-evals":
        from phoenix.evals import metrics

        target, name = metrics, "exact_match"
    else:
        from opik.evaluation.metrics import Equals

        target, name = Equals, "score"
    original = getattr(target, name)
    calls = []

    @wraps(original)
    def observed(*args, **kwargs):
        """Observe the call while executing the actual installed SDK method."""
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, observed)
    return calls


@pytest.mark.parametrize("evaluator", LIVE.VERSIONS)
def test_live_scalar_fresh_task_and_real_sdk_metric(evaluator, monkeypatch, tmp_path):
    _require_sdk(evaluator, monkeypatch)
    metric_calls = _observe_real_metric(evaluator, monkeypatch)
    kinds = ("match", "mismatch", "likelihood", "error", "reference-free", "exception")
    cases = [
        {
            "id": kind,
            "input": f"Synthetic test instruction: {kind}",
            "expected": None if kind == "reference-free" else "Answer",
            "metadata": {"slice": "synthetic-test"},
        }
        for kind in kinds
    ]
    original = copy.deepcopy(cases)
    task_calls = []

    def synthetic_task(case):
        task_calls.append(case["id"])
        kind = case["id"]
        case["metadata"]["task_mutation"] = "must not alter supplied case"
        if kind == "exception":
            raise RuntimeError("synthetic task failure")
        result = {
            "output": "Wrong" if kind == "mismatch" else "Answer",
            "metadata": {"task_provenance": {"kind": "synthetic-test", "call": kind}},
        }
        if kind == "likelihood":
            result["output"] = None
            result["metadata"]["invarlock_likelihood"] = {
                "logprob_sum": -0.125,
                "token_count": 7,
                "test_fixture": "serialization only; not model evidence",
            }
        elif kind == "error":
            result.update(output=None, error="synthetic returned failure")
        return result

    original_import = builtins.__import__

    def no_core_import(name, *args, **kwargs):
        if name == "invarlock" or name.startswith("invarlock."):
            raise AssertionError("SDK capture cannot import the recipient package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_core_import)
    payload = LIVE.run(evaluator, cases, synthetic_task, tmp_path)
    monkeypatch.setattr(builtins, "__import__", original_import)
    assert cases == original
    assert task_calls == list(kinds)
    assert len(metric_calls) == 2
    rows = export_records(evaluator, payload)
    assert [row["id"] for row in rows] == list(kinds)
    assert [row["output"] for row in rows] == [
        "Answer",
        "Wrong",
        None,
        None,
        "Answer",
        None,
    ]
    assert all(row["scores"] == {} for row in rows)
    assert all(row["metadata"] == {"slice": "synthetic-test"} for row in rows)
    assert rows[2]["error"] is None
    assert rows[2]["likelihood"]["logprob_sum"] == -0.125
    assert rows[3]["error"] == "synthetic returned failure"
    assert rows[5]["error"] == "RuntimeError: synthetic task failure"
    for index, entry in enumerate(payload):
        execution = entry["capture_execution"]
        assert execution["task_invocations"] == 1
        assert execution["metric_api"] == LIVE.METRIC_APIS[evaluator]
        assert execution["metric_status"] == ("completed" if index < 2 else "skipped")
        assert entry["source_case"] == original[index]
        if index >= 2:
            assert "metric_result" not in entry
    score_key = {
        "ragas": "value",
        "opik": "value",
        "hugging-face-evaluate": "exact_match",
    }.get(evaluator, "score")
    assert payload[0]["metric_result"][score_key] == 1
    assert payload[1]["metric_result"][score_key] == 0


def _mock_sdk_setup(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.3.0")
    monkeypatch.setattr(LIVE, "_metric", lambda *args: lambda entry: {"score": 1})


def test_live_scalar_rejects_invalid_schedule_before_task(monkeypatch, tmp_path):
    _mock_sdk_setup(monkeypatch)
    calls = []
    case = {"id": "a", "input": "Question", "expected": "Answer", "metadata": {}}
    for cases in ([], [case, case], [{**case, "input": None}], [{**case, "id": 1}]):
        with pytest.raises(ValueError):
            LIVE.run("autoevals", cases, lambda row: calls.append(row), tmp_path)
    assert calls == []


@pytest.mark.parametrize(
    "result", [{}, {"output": {}}, {"output": "A", "metadata": []}]
)
def test_live_scalar_rejects_invalid_task_result(result, monkeypatch, tmp_path):
    _mock_sdk_setup(monkeypatch)
    with pytest.raises(ValueError):
        LIVE.run(
            "autoevals", [{"id": "a", "input": "Q"}], lambda case: result, tmp_path
        )


def test_live_scalar_conflicting_provenance_rejected(monkeypatch, tmp_path):
    _mock_sdk_setup(monkeypatch)
    with pytest.raises(ValueError, match="metadata conflicts"):
        LIVE.run(
            "autoevals",
            [{"id": "a", "input": "Q", "metadata": {"slice": "a"}}],
            lambda case: {"output": "A", "metadata": {"slice": "b"}},
            tmp_path,
        )


def test_live_scalar_metric_failure_does_not_become_task_failure(monkeypatch, tmp_path):
    _mock_sdk_setup(monkeypatch)

    def fail_metric(entry):
        raise RuntimeError("synthetic metric failure")

    monkeypatch.setattr(LIVE, "_metric", lambda *args: fail_metric)
    payload = LIVE.run(
        "autoevals",
        [{"id": "a", "input": "Q", "expected": "A"}],
        lambda case: {"output": "A"},
        tmp_path,
    )
    row = export_records("autoevals", payload)[0]
    assert row["output"] == "A" and row["error"] is None
    assert payload[0]["capture_execution"]["metric_status"] == "failed"
    assert "metric_result" not in payload[0]
