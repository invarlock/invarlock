"""Actual SDK objects use the same capture entry point as offline exports."""

import importlib.metadata
import json
import os
from pathlib import Path

import pytest

from invarlock.engine import EvaluationRecordsError, export_evaluator_result, load_run
from tests.examples.test_langfuse_export import experiment
from tests.examples.test_langfuse_export import sdk as sdk


def capture(evaluator, value, path, version, run_id, ids):
    return export_evaluator_result(
        evaluator,
        value,
        path,
        expected_ids=ids,
        source_version=version,
        run_id=run_id,
        artifact_digest="sha256:" + "a" * 64,
    )


def test_actual_langfuse_experiment_uses_core_entrypoint(tmp_path, sdk):
    value = experiment(sdk)
    path = tmp_path / "export.json"
    run = capture("langfuse", value, path, "4.14.1", value.run_name, ["one", "two"])
    assert [row["output"] for row in run["records"]] == ["Paris", "Paris"]
    assert run == load_run(
        path,
        adapter="evaluator-json",
        source={"name": "langfuse", "version": "4.14.1"},
        run_id=value.run_name,
        artifact_digest="sha256:" + "a" * 64,
    )


@pytest.mark.parametrize(
    "fault", ["missing", "version", "type", "item", "empty", "nan"]
)
def test_actual_langfuse_bad_capture_does_not_publish(tmp_path, sdk, fault):
    value = experiment(
        sdk, fail=fault == "missing", output=float("nan") if fault == "nan" else "Paris"
    )
    version = "wrong" if fault == "version" else "4.14.1"
    if fault == "type":
        value = object()
    elif fault == "item":
        value.item_results[0].item = object()
    elif fault == "empty":
        value.item_results.clear()
    with pytest.raises(EvaluationRecordsError):
        capture(
            "langfuse",
            value,
            tmp_path / "export.json",
            version,
            getattr(value, "run_name", "run"),
            ["one", "two"],
        )
    assert not (tmp_path / "export.json").exists()


def test_actual_inspect_log_uses_core_entrypoint(tmp_path):
    qualification = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in (
        "1",
        "inspect-ai",
    )
    required = qualification or os.environ.get("INVARLOCK_REQUIRE_INSPECT_SDK") == "1"
    # The retained evaluator profile and runtime SDK gates have separate pins.
    supported = ("0.3.254",) if qualification else ("0.3.254", "0.3.263")
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError:
        if required:
            pytest.fail("required Inspect AI SDK is missing")
        pytest.skip("optional Inspect AI SDK is not installed")
    if version not in supported:
        if required:
            pytest.fail(f"required Inspect AI version in {supported}, found {version}")
        pytest.skip(f"requires pinned Inspect AI version in {supported}")
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec

    data = json.loads(
        (Path(__file__).parent / "fixtures/inspect-0.3.254.json").read_bytes()
    )
    # Retained fixture contains source samples, not the EvalSpec header. This
    # smoke exercises the actual SDK serializer, without claiming a fresh run.
    data["eval"] = EvalSpec(
        created="2026-01-01T00:00:00Z",
        task="retained-samples",
        dataset=EvalDataset(samples=40),
        model="retained-fixture",
        config=EvalConfig(),
    )
    value = EvalLog.model_validate(data)
    version = importlib.metadata.version("inspect-ai")
    path = tmp_path / "export.json"
    ids = [str(sample.id) for sample in value.samples]
    run = capture("inspect-ai", value, path, version, "run", ids)
    assert len(run["records"]) == 40
    assert json.loads(path.read_bytes())["payload"] == value.model_dump(mode="json")
    for wrong_value, wrong_version in [(object(), version), (value, "wrong")]:
        with pytest.raises(EvaluationRecordsError):
            capture(
                "inspect-ai", wrong_value, tmp_path / "bad", wrong_version, "run", ids
            )
    assert not (tmp_path / "bad").exists()
