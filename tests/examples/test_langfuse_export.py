"""Real Langfuse SDK capture, measured in the examples coverage shard.

Set INVARLOCK_REQUIRE_LANGFUSE_SDK=1 to fail if the pinned SDK is unavailable.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sdk(monkeypatch):
    required = os.environ.get("INVARLOCK_REQUIRE_LANGFUSE_SDK") == "1"
    try:
        version = importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "4.14.1":
        if required:
            pytest.fail("required Langfuse 4.14.1 SDK is not installed")
        pytest.skip("optional Langfuse 4.14.1 SDK is not installed")
    import httpx
    from langfuse import Evaluation, Langfuse

    def forbidden(*args, **kwargs):
        pytest.fail("offline Langfuse capture attempted HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    client = Langfuse(
        public_key="offline-public", secret_key="offline-secret", tracing_enabled=False
    )
    return client, Evaluation


@pytest.fixture
def exporter():
    spec = importlib.util.spec_from_file_location(
        "langfuse_export_test", ROOT / "examples/integrations/langfuse_export.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment(sdk, *, fail=False, output="Paris"):
    client, Evaluation = sdk

    def task(*, item, **_):
        if fail and item["metadata"]["invarlock_id"] == "two":
            raise RuntimeError("deliberate synthetic task failure")
        return output

    return client.run_experiment(
        name="synthetic-offline-sdk-journey",
        data=[
            {
                "input": "Capital of France?",
                "expected_output": "Paris",
                "metadata": {"invarlock_id": identity},
            }
            for identity in ("one", "two")
        ],
        task=task,
        evaluators=[
            lambda *, output, expected_output, **_: Evaluation(
                name="exact_match", value=output == expected_output, data_type="BOOLEAN"
            )
        ],
        max_concurrency=1,
    )


def test_real_sdk_result_exports_and_imports_without_summary_loss(
    tmp_path, sdk, exporter
):
    from invarlock.evaluation_records.adapters import load_run

    result = experiment(sdk)
    path = tmp_path / "experiment.json"
    exported = exporter.write_experiment_export(
        result, path, expected_ids=["one", "two"]
    )
    assert set(exported["result"]) == set(vars(result))
    assert len(exported["result"]["item_results"]) == 2
    run = load_run(
        path,
        adapter="langfuse-json",
        source={"name": "langfuse", "version": "4.14.1"},
        run_id=result.run_name,
        artifact_digest="sha256:" + "a" * 64,
    )
    assert [row["output"] for row in run["records"]] == ["Paris", "Paris"]
    assert all(row["scores"] == {"exact_match": 1.0} for row in run["records"])
    with pytest.raises(FileExistsError):
        exporter.write_experiment_export(result, path, expected_ids=["one", "two"])


def test_real_sdk_task_failure_cannot_silently_shrink_expected_cases(sdk, exporter):
    result = experiment(sdk, fail=True)
    assert len(result.item_results) == 1
    with pytest.raises(ValueError, match="every expected case"):
        exporter.serialize_experiment(result, expected_ids=["one", "two"])


@pytest.mark.parametrize(
    "output", [float("nan"), float("inf"), object(), {1: "non-text key"}, ("tuple",)]
)
def test_real_sdk_non_json_outputs_fail_before_export(tmp_path, sdk, exporter, output):
    result = experiment(sdk, output=output)
    with pytest.raises((TypeError, ValueError)):
        exporter.write_experiment_export(
            result, tmp_path / "experiment.json", expected_ids=["one", "two"]
        )
    assert not (tmp_path / "experiment.json").exists()


@pytest.mark.parametrize("expected", [[], ["one", "one"], ["one", "missing"]])
def test_real_sdk_expected_ids_are_independent_and_exact(sdk, exporter, expected):
    with pytest.raises(ValueError):
        exporter.serialize_experiment(experiment(sdk), expected_ids=expected)


def test_real_hosted_dataset_item_retains_public_sdk_fields(tmp_path, sdk, exporter):
    from datetime import UTC, datetime

    from langfuse.api import DatasetItem, DatasetStatus
    from langfuse.experiment import ExperimentItemResult, ExperimentResult

    from invarlock.evaluation_records.adapters import load_run

    item = DatasetItem(
        id="hosted-item",
        status=DatasetStatus.ACTIVE,
        input="Question",
        expected_output="Answer",
        metadata={"category": "offline-fixture"},
        dataset_id="dataset",
        dataset_name="offline-dataset",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        media_references=[],
    )
    result = ExperimentResult(
        name="offline-hosted-shape",
        run_name="offline-hosted-run",
        description=None,
        item_results=[
            ExperimentItemResult(
                item=item,
                output="Answer",
                evaluations=[],
                trace_id=None,
                dataset_run_id="run",
            )
        ],
        run_evaluations=[],
        experiment_id="run",
        dataset_run_id="run",
    )
    path = tmp_path / "hosted.json"
    value = exporter.write_experiment_export(result, path, expected_ids=["hosted-item"])
    captured_item = value["result"]["item_results"][0]["item"]
    assert set(captured_item) == set(type(item).model_fields)
    assert captured_item["expected_output"] == item.expected_output
    assert captured_item["status"] == "ACTIVE"
    assert captured_item["created_at"] == item.created_at.isoformat()
    run = load_run(
        path,
        adapter="langfuse-json",
        source={"name": "langfuse", "version": "4.14.1"},
        run_id=result.run_name,
        artifact_digest="sha256:" + "a" * 64,
    )
    assert run["records"][0]["id"] == "hosted-item"
    assert run["records"][0]["output"] == "Answer"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "conflict", "object"])
def test_invalid_result_identity_never_creates_export(
    tmp_path, sdk, exporter, mutation
):
    result = experiment(sdk)
    item = result.item_results[0].item
    if mutation == "duplicate":
        item["metadata"]["invarlock_id"] = "two"
    elif mutation == "missing":
        item["metadata"].pop("invarlock_id")
    elif mutation == "conflict":
        item["id"] = "conflicting-hosted-id"
    else:
        result.item_results[0].item = object()
    with pytest.raises(ValueError):
        exporter.write_experiment_export(
            result, tmp_path / "invalid.json", expected_ids=["one", "two"]
        )
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("limit", ["MAX_INPUT_BYTES", "MAX_RECORDS"])
def test_capture_limits_fail_before_creating_export(
    tmp_path, monkeypatch, sdk, exporter, limit
):
    result = experiment(sdk)
    monkeypatch.setattr(exporter, limit, 1)
    with pytest.raises(ValueError):
        exporter.write_experiment_export(
            result, tmp_path / "oversized.json", expected_ids=["one", "two"]
        )
    assert not (tmp_path / "oversized.json").exists()


def test_capture_requires_the_qualified_sdk_and_its_result(monkeypatch, sdk, exporter):
    with pytest.raises(ValueError, match="actual Langfuse"):
        exporter.serialize_experiment({}, expected_ids=["one"])
    result = experiment(sdk)
    monkeypatch.setattr(exporter.importlib.metadata, "version", lambda _: "unqualified")
    with pytest.raises(ValueError, match="4.14.1 is required"):
        exporter.serialize_experiment(result, expected_ids=["one", "two"])


def test_reference_capture_reproduces_complete_retained_model_facts(
    tmp_path, monkeypatch, sdk
):
    spec = importlib.util.spec_from_file_location(
        "langfuse_reference_capture",
        ROOT / "examples/captured-results/references/langfuse/capture.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "replay"
    monkeypatch.setattr("sys.argv", ["capture.py", "--output", str(destination)])
    module.main()
    reference = json.loads((destination / "reference.json").read_bytes())
    assert len(reference["exports"]) == 4
    for name, binding in reference["exports"].items():
        assert binding["record_count"] == 400
        exported = json.loads((destination / name).read_bytes())
        original = json.loads((ROOT / reference["sources"][name]["path"]).read_bytes())
        assert [row["output"] for row in exported["result"]["item_results"]] == [
            row["output"] for row in original["records"]
        ]
    with pytest.raises(FileExistsError):
        module.main()
