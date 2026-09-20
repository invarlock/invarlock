"""Optional pinned SDK checks for the retained-measurement handoff helper."""

from __future__ import annotations

import copy
import importlib.metadata
import os
import socket

import pytest

from invarlock.evaluation_records.scalar_integrations import export_records
from tests.evaluation_records.sdk_scalar_roundtrip import VERSIONS, roundtrip
from tests.evaluation_records.test_scalar_integrations import native_entry


def _set_output(evaluator, entry, output):
    container, key = {
        "deepeval": ("test_case", "actual_output"),
        "ragas": ("sample", "response"),
        "lighteval": ("model_response", "text"),
        "hugging-face-evaluate": (None, "predictions"),
        "autoevals": (None, "output"),
        "openevals": (None, "outputs"),
        "arize-phoenix-evals": ("record", "output"),
        "opik": ("dataset_item", "output"),
    }[evaluator]
    (entry[container] if container else entry)[key] = (
        [output] if key in ("text", "predictions") else output
    )


def _require_sdk_offline(evaluator, monkeypatch):
    package = "evaluate" if evaluator == "hugging-face-evaluate" else evaluator
    required = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", evaluator)
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != VERSIONS[evaluator]:
        if required:
            pytest.fail(f"requires {package}=={VERSIONS[evaluator]}, found {installed}")
        pytest.skip(f"optional {package}=={VERSIONS[evaluator]} SDK not installed")
    for key, value in {
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
        "RAGAS_DO_NOT_TRACK": "true",
        "OPIK_TRACK_DISABLE": "true",
        "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(key, value)

    def forbid_network(*args, **kwargs):
        raise AssertionError("SDK serialization must not access a network service")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)


@pytest.mark.parametrize("evaluator", VERSIONS)
def test_actual_scalar_sdk_retains_answers_likelihood_and_failures(
    evaluator, tmp_path, monkeypatch
):
    _require_sdk_offline(evaluator, monkeypatch)
    entries = []
    for kind in ("answer", "likelihood", "failure", "ungraded"):
        entry = native_entry(evaluator)
        entry["id"] = kind
        entry["metadata"]["invarlock_scores"] = {"quality": 0.875}
        if kind == "likelihood":
            _set_output(evaluator, entry, None)
            # Deliberately test serialization only: central capture validates
            # the complete typed facts in the installed-recipient journey.
            entry["metadata"]["invarlock_likelihood"] = {
                "logprob_sum": -0.125,
                "token_count": 7,
                "retained_nested_fact": {"value": [None, 3, "exact"]},
            }
        elif kind == "failure":
            _set_output(evaluator, entry, None)
            entry["error"] = "Original generation failed"
        elif kind == "ungraded":
            del entry["metric_result"]
        entries.append(entry)
    untouched = copy.deepcopy(entries)
    result = roundtrip(evaluator, entries, tmp_path=tmp_path)
    assert entries == untouched
    rows = export_records(evaluator, result)
    assert [row["id"] for row in rows] == [
        "answer",
        "likelihood",
        "failure",
        "ungraded",
    ]
    assert [row["output"] for row in rows] == ["Answer", None, None, "Answer"]
    assert rows[1]["likelihood"] == entries[1]["metadata"]["invarlock_likelihood"]
    assert rows[1]["error"] is None
    assert rows[2]["error"] == "Original generation failed"
    assert "metric_result" not in result[3]
    assert export_records(evaluator, result) == rows


@pytest.mark.parametrize(
    ("predictions", "references"),
    [
        ([None, None], ["Answer", "Answer"]),
        ([None, "Answer"], ["Answer", "Answer"]),
        ([None, None], [None, None]),
        ([None, "Answer", None, "Answer"], [None, "Answer", "Answer", None]),
    ],
    ids=(
        "all-null-answers",
        "null-first-answer",
        "all-null-columns",
        "mixed-null-first",
    ),
)
def test_evaluate_null_column_batches(predictions, references, tmp_path, monkeypatch):
    evaluator = "hugging-face-evaluate"
    _require_sdk_offline(evaluator, monkeypatch)
    entries = []
    for index, (prediction, reference) in enumerate(
        zip(predictions, references, strict=True)
    ):
        entry = native_entry(evaluator)
        entry["id"] = f"case-{index}"
        entry["predictions"] = [prediction]
        entry["references"] = [reference]
        if prediction is None:
            entry["metadata"]["invarlock_likelihood"] = {"logprob_sum": -0.125}
        entries.append(entry)
    original = copy.deepcopy(entries)
    serialized = roundtrip(evaluator, entries, tmp_path=tmp_path)
    assert entries == original
    assert serialized == original
    restored = export_records(evaluator, serialized)
    assert [row["output"] for row in restored] == predictions
    assert [row["expected"] for row in restored] == references
    assert [row["id"] for row in restored] == [row["id"] for row in entries]


def test_scalar_roundtrip_requires_known_pinned_sdk(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unsupported scalar SDK"):
        roundtrip("unknown", [], tmp_path=tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(ValueError, match="requires deepeval==4.1.3"):
        roundtrip("deepeval", [], tmp_path=tmp_path)
