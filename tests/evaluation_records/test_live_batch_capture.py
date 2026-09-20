"""Real SDK execution using explicitly synthetic tasks; no model or network calls."""

from __future__ import annotations

import builtins
import copy
import importlib.metadata
import importlib.util
import json
import os
import socket
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest

from invarlock.evaluation_records.batch_integrations import export_records

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/integrations/evaluator-live/batch.py"
)
SPEC = importlib.util.spec_from_file_location("live_batch_example", SOURCE)
assert SPEC and SPEC.loader
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)


def _cases():
    return [
        {
            "id": kind,
            "input": "Same synthetic instruction for independent case identities",
            "expected": "Answer",
            "metadata": {"slice": "synthetic-test"},
        }
        for kind in ("match", "mismatch", "likelihood", "error", "exception")
    ]


def _require_sdk(evaluator, monkeypatch, tmp_path):
    required = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", evaluator)
    selected = os.environ.get("INVARLOCK_LIVE_EVALUATOR")
    if selected and selected != evaluator:
        pytest.skip("another isolated SDK environment selected")
    try:
        installed = importlib.metadata.version(evaluator)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != LIVE.VERSIONS[evaluator]:
        if required or selected == evaluator:
            pytest.fail(
                f"requires {evaluator}=={LIVE.VERSIONS[evaluator]}, found {installed}"
            )
        pytest.skip(f"optional pinned {evaluator} SDK not installed")
    for key, value in {
        "AZURE_TELEMETRY_DISABLED": "1",
        "OTEL_SDK_DISABLED": "true",
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "OPENAI_API_KEY": "synthetic-test-only-not-a-credential",
    }.items():
        monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "synthetic local SDK capture must never contact the network"
        )

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _observe_execution(evaluator, monkeypatch):
    if evaluator == "pydantic-evals":
        from pydantic_evals import Dataset

        target, name = Dataset, "evaluate_sync"
    elif evaluator == "azure-ai-evaluation":
        import azure.ai.evaluation

        target, name = azure.ai.evaluation, "evaluate"
    elif evaluator == "evidently":
        from evidently import Dataset

        target, name = Dataset, "from_pandas"
    elif evaluator == "mlflow":
        import mlflow

        target, name = mlflow.models, "evaluate"
    else:
        from trulens.apps.basic import TruBasicApp

        target, name = TruBasicApp, "__enter__"
    real = getattr(target, name)
    calls = []

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, observe)
    return calls


@pytest.mark.parametrize("evaluator", LIVE.VERSIONS)
def test_real_sdk_execution_preserves_complete_synthetic_schedule(
    evaluator, monkeypatch, tmp_path
):
    _require_sdk(evaluator, monkeypatch, tmp_path)
    sdk_calls = _observe_execution(evaluator, monkeypatch)
    cases = _cases()
    original = copy.deepcopy(cases)
    seen = []
    measured = {
        "synthetic_test_fact": True,
        "source": "synthetic-runtime",
        "logprob_sum": -2.0,
    }

    def synthetic_task(case):
        seen.append(case["id"])
        case["metadata"]["local_mutation"] = True
        if case["id"] == "exception":
            raise RuntimeError("synthetic execution exception")
        result = {"output": "Wrong" if case["id"] == "mismatch" else "Answer"}
        if case["id"] == "likelihood":
            result = {"output": None, "metadata": {"invarlock_likelihood": measured}}
        if case["id"] == "error":
            result = {
                "output": "partial answer",
                "error": "synthetic execution failure",
            }
        return result

    original_import = builtins.__import__

    def sdk_only(name, *args, **kwargs):
        if name == "invarlock" or name.startswith("invarlock."):
            raise AssertionError("SDK capture must not import the recipient package")
        return original_import(name, *args, **kwargs)

    workdir = tmp_path / "capture"
    workdir.mkdir()
    with monkeypatch.context() as sdk_capture:
        sdk_capture.setattr(builtins, "__import__", sdk_only)
        native = LIVE.run(evaluator, cases, synthetic_task, workdir)
    assert sdk_calls
    assert sorted(seen) == sorted(case["id"] for case in original)
    assert cases == original
    assert json.loads((workdir / "native.json").read_text()) == native
    assert (workdir / "application-captures.json").is_file()
    rows = {row["id"]: row for row in export_records(evaluator, native)}
    assert set(rows) == set(seen)
    for case in original:
        row = rows[case["id"]]
        assert (row["input"], row["expected"]) == (case["input"], case["expected"])
        assert row["metadata"]["slice"] == "synthetic-test"
        assert "local_mutation" not in row["metadata"]
    assert rows["match"]["output"] == "Answer"
    assert rows["mismatch"]["output"] == "Wrong"
    assert rows["likelihood"]["output"] is None
    assert rows["likelihood"]["error"] is None
    assert rows["likelihood"]["likelihood"] == measured
    assert rows["error"]["output"] == "partial answer"
    assert rows["error"]["error"] == "synthetic execution failure"
    assert rows["exception"]["error"] == "RuntimeError: synthetic execution exception"
    if evaluator == "trulens":
        assert (workdir / "trulens.sqlite").is_file()
        assert all(
            row["calls"] and row["source_record_id"] for row in native["records"]
        )
        with pytest.raises(ValueError, match="fresh SDK worker"):
            LIVE.run(
                evaluator,
                cases,
                lambda _: pytest.fail("must preflight"),
                tmp_path / "second",
            )
    else:
        assert rows["match"]["scores"] and 1.0 in rows["match"]["scores"].values()
        assert 0.0 in rows["mismatch"]["scores"].values()
    if evaluator == "mlflow":
        assert (workdir / "sdk-evaluation/artifacts").is_dir()
        assert (workdir / "artifacts").is_dir()


@pytest.mark.parametrize(
    "cases",
    [
        None,
        [],
        [{}],
        [{"id": " ", "input": "x", "expected": "y", "metadata": {}}],
        _cases() + [_cases()[0]],
    ],
)
def test_invalid_planned_schedule_rejected_before_task(cases):
    with pytest.raises(ValueError, match="cases"):
        LIVE.Capture(cases, lambda _: pytest.fail("invalid case must not execute"))


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"output": 2},
        {"output": "x", "error": False},
        {"output": "x", "error": ""},
        {"output": "x", "metadata": None},
    ],
)
def test_malformed_task_result_rejected(result):
    capture = LIVE.Capture(_cases(), lambda _: result)
    with pytest.raises(ValueError, match="task result"):
        capture.invoke("match", _cases()[0]["input"])


def test_capture_membership_conflicts_and_failure():
    case = _cases()[0]
    capture = LIVE.Capture([case], lambda _: {"output": "Answer"})
    with pytest.raises(ValueError, match="identity or input"):
        capture.invoke("match", "changed input")
    with pytest.raises(ValueError, match="complete planned"):
        capture.rows()
    capture.invoke("match", case["input"])
    with pytest.raises(ValueError, match="more than once"):
        capture.invoke("match", case["input"])
    assert capture.rows()[0]["output"] == "Answer"
    for metadata, match in [
        ({"slice": "changed"}, "conflicts"),
        ({"invarlock_task_capture": {}}, "reserved"),
    ]:
        capture = LIVE.Capture(
            [case], lambda _, metadata=metadata: {"output": None, "metadata": metadata}
        )
        with pytest.raises(ValueError, match=match):
            capture.invoke("match", case["input"])


def test_json_serialization_preserves_types_and_rejects_loss():
    @dataclass
    class Value:
        label: str

    class Kind(Enum):
        OK = "ok"

    assert LIVE._plain(
        [
            Value("x"),
            Kind.OK,
            date(2026, 1, 2),
            timedelta(seconds=2),
            UUID(int=0),
            (None, True),
        ]
    ) == [
        {"label": "x"},
        "ok",
        "2026-01-02",
        2.0,
        "00000000-0000-0000-0000-000000000000",
        [None, True],
    ]
    for value in (float("nan"), float("inf"), {1: "bad"}, object()):
        with pytest.raises(ValueError):
            LIVE._plain(value)


def test_preflight_does_not_overwrite_or_run_tasks(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        LIVE.run("other", _cases(), None, tmp_path)
    (tmp_path / "existing").write_text("preserve")
    with pytest.raises(ValueError, match="must be empty"):
        LIVE.run(
            "pydantic-evals",
            _cases(),
            lambda _: pytest.fail("must preflight"),
            tmp_path,
        )
    assert (tmp_path / "existing").read_text() == "preserve"


@pytest.mark.parametrize(
    "changed",
    ["duplicate", "input", "expected", "output", "metadata", "error", "missing"],
)
def test_native_export_drift_rejected(changed, tmp_path, monkeypatch):
    case = _cases()[0]

    def fake_sdk(capture, _):
        capture.invoke(case["id"], case["input"])
        rows = capture.rows()
        if changed == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif changed == "missing":
            rows.clear()
        elif changed == "expected":
            rows[0]["reference"] = "changed"
        else:
            rows[0][changed] = "changed"
        return {"rows": rows}

    monkeypatch.setattr(LIVE, "_evidently", fake_sdk)
    with pytest.raises(ValueError, match="SDK export"):
        LIVE.run("evidently", [case], lambda _: {"output": "Answer"}, tmp_path)
    assert not (tmp_path / "native.json").exists()


def test_reserved_capture_metadata_rejected_before_task():
    cases = _cases()
    cases[0]["metadata"]["invarlock_task_capture"] = {"output": "invented"}
    with pytest.raises(ValueError, match="reserved"):
        LIVE.Capture(cases, lambda _: pytest.fail("must preflight reserved metadata"))
