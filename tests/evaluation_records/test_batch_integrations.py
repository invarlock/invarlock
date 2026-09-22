"""Contract tests for native report/table/attempt integrations without SDK calls."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.batch_integrations import (
    export_records,
    serialize_results,
)


@dataclasses.dataclass
class Metric:
    value: float | bool
    reason: str = "native rationale"


def test_garak_single_text_sdk_messages_preserve_original_native_context():
    message = {
        "text": "question",
        "lang": None,
        "data_path": None,
        "data_type": None,
        "data_checksum": None,
        "notes": {},
    }
    attempt = {
        "uuid": "native",
        "status": 2,
        "prompt": {"turns": [{"role": "user", "content": message}], "notes": {}},
        "outputs": [{**message, "text": "answer"}],
    }
    source = {
        "native_id": "native:0",
        "id": "original",
        "input": "question",
        "output": "answer",
        "expected": "answer",
        "metadata": {"slice": "sdk"},
    }
    row = export_records("garak", {"attempts": [attempt], "source_cases": [source]})[0]
    assert (row["id"], row["input"], row["output"], row["expected"]) == (
        "original",
        "question",
        "answer",
        "answer",
    )
    assert row["context"]["upstream_record"]["prompt"] == attempt["prompt"]
    assert row["context"]["upstream_record"]["outputs"] == attempt["outputs"]
    assert row["context"]["source_case"] == source


@pytest.mark.parametrize(
    "fault", ["multiple", "system", "attachment", "extra", "nontext"]
)
def test_garak_ambiguous_sdk_message_cannot_be_joined_to_plain_text(fault):
    message = {"text": "question"}
    prompt = {"turns": [{"role": "user", "content": message}]}
    if fault == "multiple":
        prompt["turns"].append({"role": "assistant", "content": {"text": "earlier"}})
    elif fault == "system":
        prompt["turns"][0]["role"] = "system"
    elif fault == "attachment":
        message["data_path"] = "retained-image.png"
    elif fault == "extra":
        message["unknown"] = "retained-field"
    else:
        message["text"] = None
    attempt = {"uuid": "native", "status": 2, "prompt": prompt, "outputs": ["answer"]}
    # Structured prompts remain available for an explicit projection; no implicit
    # selection drops other messages, attachments, or unknown model inputs.
    row = export_records("garak", {"attempts": [attempt]})[0]
    assert row["input"] == prompt
    with pytest.raises(EvaluationRecordsError, match="conflicts with actual attempt"):
        export_records(
            "garak",
            {
                "attempts": [attempt],
                "source_cases": [
                    {
                        "native_id": "native:0",
                        "id": "original",
                        "input": "question",
                        "expected": "answer",
                    }
                ],
            },
        )


@dataclasses.dataclass
class ReportCase:
    name: str
    inputs: object
    output: object
    expected_output: object
    scores: object = dataclasses.field(default_factory=dict)
    assertions: object = dataclasses.field(default_factory=dict)
    metadata: object = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Report:
    cases: list
    failures: list = dataclasses.field(default_factory=list)
    report_evaluator_failures: list = dataclasses.field(default_factory=list)


class Table:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, *, orient):
        assert orient == "records"
        return self.rows


class Dataset:
    def as_dataframe(self):
        return Table(
            [
                {
                    "record_id": "a",
                    "input": "question",
                    "output": "answer",
                    "reference": "answer",
                    "exact_match": True,
                }
            ]
        )


class Attempt:
    def as_dict(self):
        return {
            "entry_type": "attempt",
            "uuid": "attempt-a",
            "status": 2,
            "prompt": "question",
            "outputs": ["answer"],
            "detector_results": {"attack": [0.0]},
            "targets": ["attack target"],
        }


class EvaluationResult:
    metrics = {"accuracy": 0.5}
    tables = {
        "eval_results_table": Table(
            [
                {
                    "record_id": "a",
                    "input": "question",
                    "predictions": "answer",
                    "targets": "answer",
                    "accuracy/score": 1.0,
                }
            ]
        )
    }


def event(kind, data, *, ident="a", event_id=None):
    value = {"sample_id": ident, "type": kind, "data": data, "run_id": "run"}
    if event_id is not None:
        value["event_id"] = event_id
    return value


def test_pydantic_report_keeps_task_failures_evaluator_failures_and_native_values():
    report = Report(
        [
            ReportCase(
                "a",
                {"question": "hi"},
                "yes",
                "yes",
                {"quality": Metric(0.75)},
                {"equals": Metric(True)},
                {"slice": "A", "structured": {"count": 2}},
            )
        ],
        [
            {
                "name": "b",
                "inputs": "failed input",
                "expected_output": "expected",
                "error_message": "timeout",
                "error_stacktrace": "stack",
            }
        ],
        [{"evaluator_name": "aggregate", "error_message": "report failed"}],
    )
    rows = export_records("pydantic-evals", report)
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["scores"] == {"quality": 0.75, "equals": 1.0}
    assert rows[0]["metadata"] == {"slice": "A"}
    assert (
        rows[0]["context"]["upstream_record"]["scores"]["quality"]["reason"]
        == "native rationale"
    )
    assert rows[0]["context"]["upstream_summary"]["report_evaluator_failures"]
    assert rows[1]["output"] is None and rows[1]["error"] == "timeout"


def test_pydantic_evaluator_failure_preserves_model_output():
    rows = export_records(
        "pydantic-evals",
        {
            "cases": [
                {
                    "name": "a",
                    "inputs": "q",
                    "output": "actual",
                    "evaluator_failures": [{"error_message": "judge timeout"}],
                }
            ]
        },
    )
    assert rows[0]["output"] == "actual"
    assert rows[0]["error"] is None


def test_azure_maps_namespaced_rows_and_keeps_aggregate_separate():
    result = {
        "metrics": {"exact_match.exact_match": 0.5},
        "rows": [
            {
                "inputs.record_id": "a",
                "inputs.query": "q",
                "inputs.response": "A",
                "inputs.ground_truth": "A",
                "outputs.exact_match.exact_match": 1.0,
                "outputs.judge.reason": "matched",
            },
            {
                "inputs.record_id": "b",
                "inputs.query": "q2",
                "inputs.response": None,
                "outputs.judge.error": {"message": "timeout"},
            },
        ],
    }
    rows = export_records("azure-ai-evaluation", result)
    assert rows[0]["scores"] == {"exact_match.exact_match": 1.0}
    assert rows[0]["input"] == "q" and rows[0]["expected"] == "A"
    assert rows[0]["context"]["upstream_summary"]["metrics"] == result["metrics"]
    assert rows[1]["error"] is None


def test_evidently_sdk_dataset_preserves_descriptor_column():
    rows = export_records("evidently", Dataset())
    assert rows[0]["scores"] == {"exact_match": 1.0}
    assert rows[0]["output"] == "answer"


def test_evidently_explicit_descriptors_do_not_guess_numeric_inputs_as_scores():
    rows = export_records(
        "evidently",
        {
            "rows": [{"record_id": "a", "output": "text", "length": 4, "age": 12}],
            "score_columns": ["length"],
            "report": {"metrics": [1]},
        },
    )
    assert rows[0]["scores"] == {"length": 4.0}
    assert rows[0]["context"]["upstream_summary"]["report"] == {"metrics": [1]}


def test_mlflow_native_tables_and_source_prediction_table():
    rows = export_records("mlflow", EvaluationResult())
    assert rows[0]["scores"] == {"accuracy/score": 1.0}
    assert rows[0]["expected"] == rows[0]["output"] == "answer"
    source = {
        "metrics": {"accuracy": 0.5},
        "prediction_table": {
            "columns": ["case", "prompt", "label", "prediction"],
            "data": [["a", "q", "x", "y"]],
        },
        "columns": {"id": "case", "input": "prompt", "expected": "label"},
    }
    rows = export_records("mlflow", source)
    assert rows[0]["scores"] == {}
    assert rows[0]["output"] == "y" and rows[0]["expected"] == "x"


def test_garak_tracks_generations_detectors_and_failed_slots():
    rows = export_records(
        "garak",
        [
            {
                "uuid": "attempt",
                "status": 2,
                "prompt": "attack",
                "outputs": ["response", None],
                "detector_results": {"detector": [0.7, None]},
                "targets": ["bad content"],
            }
        ],
    )
    assert [row["id"] for row in rows] == ["attempt:0", "attempt:1"]
    assert rows[0]["scores"] == {"detector": 0.7}
    assert rows[0]["expected"] is None
    assert rows[1]["error"] == "Garak generation returned no output"


def test_garak_lifecycle_keeps_history_without_duplicating_case():
    rows = export_records(
        "garak",
        {
            "entries": [
                {"entry_type": "init", "version": "1"},
                {"uuid": "a", "status": 1, "prompt": "q", "outputs": []},
                {
                    "uuid": "a",
                    "status": 2,
                    "prompt": "q",
                    "outputs": ["actual"],
                    "detector_results": {"unsafe": [False]},
                },
                {"entry_type": "eval", "passed": 1},
            ]
        },
    )
    assert len(rows) == 1
    assert len(rows[0]["context"]["upstream_record"]["attempt_history"]) == 2
    assert len(rows[0]["context"]["upstream_summary"]["report_entries"]) == 2


def test_garak_unfinished_attempt_is_captured_without_output():
    row = export_records(
        "garak", [{"uuid": "a", "status": 1, "prompt": "q", "outputs": []}]
    )[0]
    assert row["output"] is None and row["error"] == "Garak attempt incomplete"


def test_garak_source_cases_join_stable_ids_and_explicit_references():
    row = export_records(
        "garak",
        {
            "attempts": [Attempt()],
            "source_cases": [
                {
                    "native_id": "attempt-a:0",
                    "id": "planned-a",
                    "input": "question",
                    "expected": "answer",
                    "metadata": {
                        "slice": "short",
                        "invarlock_likelihood": {"explicit": True},
                    },
                }
            ],
        },
    )[0]
    assert row["id"] == "planned-a" and row["expected"] == "answer"
    assert row["metadata"] == {"slice": "short"}
    assert row["likelihood"] == {"explicit": True}
    assert row["context"]["upstream_record"]["targets"] == ["attack target"]


def test_openai_evals_joins_events_by_sample_and_keeps_cond_logp_unpromoted():
    entries = [
        {"spec": {"eval_name": "native"}},
        event("metrics", {"accuracy": 1}, ident="b"),
        event("sampling", {"prompt": "q", "sampled": "a"}),
        event("match", {"correct": True, "expected": "a", "sampled": "a"}),
        event("cond_logp", {"prompt": "q", "completion": "a", "logp": -2}),
        event("error", {"message": "timeout"}, ident="b"),
        {"final_report": {"accuracy": 0.5}},
    ]
    rows = {row["id"]: row for row in export_records("openai-evals", entries)}
    assert rows["a"]["scores"] == {"match": 1.0}
    assert rows["a"]["output"] == rows["a"]["expected"] == "a"
    assert "likelihood" not in rows["a"]
    assert rows["b"]["error"] == "timeout" and rows["b"]["output"] is None
    assert len(rows["a"]["context"]["upstream_record"]["events"]) == 3


@pytest.mark.parametrize(
    "summary_kind", ["spec", "final_report", "nested_final_report"]
)
def test_openai_evals_rejects_conflicting_summary_run_identity(summary_kind):
    summary = (
        {"spec": {"run_id": "other-run"}}
        if summary_kind == "spec"
        else (
            {"final_report": {"accuracy": 1.0}, "run_id": "other-run"}
            if summary_kind == "final_report"
            else {"final_report": {"run_id": "other-run", "accuracy": 1.0}}
        )
    )
    with pytest.raises(EvaluationRecordsError, match="multiple runs"):
        export_records(
            "openai-evals",
            {
                "events": [
                    summary,
                    event("sampling", {"prompt": "q", "sampled": "a"}),
                ]
            },
        )


def test_trulens_feedback_table_uses_declared_feedback_names():
    rows = export_records(
        "trulens",
        (
            Table(
                [
                    {
                        "record_id": "a",
                        "input": "q",
                        "output": "a",
                        "Relevance": 0.9,
                        "latency": 2.0,
                    }
                ]
            ),
            ["Relevance"],
        ),
    )
    assert rows[0]["scores"] == {"Relevance": 0.9}


def test_trulens_record_feedback_result_and_failure():
    rows = export_records(
        "trulens",
        {
            "records": [
                {
                    "record_id": "a",
                    "main_input": "q",
                    "main_output": "a",
                    "ground_truth": "a",
                    "feedback_results": [
                        {
                            "name": "quality",
                            "record_id": "a",
                            "result": 0.8,
                            "calls": [{"args": {"response": "a"}, "ret": 0.8}],
                        },
                        {"name": "broken", "error": "provider failure", "result": None},
                    ],
                }
            ]
        },
    )
    assert rows[0]["scores"] == {"quality": 0.8}
    assert rows[0]["error"] is None
    assert rows[0]["output"] == "a"


@pytest.mark.parametrize(
    ("provider", "native"),
    [
        (
            "pydantic-evals",
            Report([ReportCase("a", "q", "a", "a", {"score": Metric(1)})]),
        ),
        (
            "azure-ai-evaluation",
            {"rows": [{"inputs.record_id": "a", "inputs.response": "a"}]},
        ),
        ("evidently", Dataset()),
        ("mlflow", EvaluationResult()),
        ("garak", [Attempt()]),
        ("openai-evals", [event("sampling", {"prompt": "q", "sampled": "a"})]),
        (
            "trulens",
            (
                Table([{"record_id": "a", "input": "q", "output": "a", "score": 1}]),
                ["score"],
            ),
        ),
    ],
)
def test_sdk_serialization_is_detached_reparse_equivalent(provider, native):
    serialized = serialize_results(provider, native)
    parsed = json.loads(json.dumps(serialized, allow_nan=False))
    assert export_records(provider, native) == export_records(provider, parsed)


@pytest.mark.parametrize(
    ("provider", "native"),
    [
        ("pydantic-evals", {"cases": [{"name": "a", "inputs": "q"}]}),
        (
            "pydantic-evals",
            {"cases": [], "failures": ["failure string lost case identity"]},
        ),
        (
            "pydantic-evals",
            {
                "cases": [
                    {
                        "name": "a",
                        "inputs": "q",
                        "output": "a",
                        "scores": {"x": {"value": "high"}},
                    }
                ]
            },
        ),
        (
            "pydantic-evals",
            {
                "cases": [
                    {
                        "name": "a",
                        "inputs": "q",
                        "output": "a",
                        "scores": {"x": 1},
                        "assertions": {"x": True},
                    }
                ]
            },
        ),
        ("azure-ai-evaluation", {"metrics": {"score": 0.9}}),
        ("azure-ai-evaluation", {"rows": [{"inputs.response": "a"}]}),
        ("evidently", {"rows": [{"output": "a", "exact_match": True}]}),
        (
            "evidently",
            {"rows": [{"record_id": "a", "output": "a"}], "score_columns": ["absent"]},
        ),
        ("mlflow", {"metrics": {"accuracy": 1}}),
        (
            "mlflow",
            {
                "prediction_table": [
                    {"record_id": "a", "target": "a", "accuracy/score": 1}
                ]
            },
        ),
        (
            "mlflow",
            {
                "prediction_table": {
                    "columns": ["record_id", "record_id"],
                    "data": [["a", "b"]],
                }
            },
        ),
        (
            "garak",
            [
                {
                    "uuid": "a",
                    "status": 2,
                    "prompt": "q",
                    "outputs": ["a"],
                    "detector_results": {"x": [1, 0]},
                }
            ],
        ),
        ("garak", [{"entry_type": "eval", "passed": 100}]),
        ("garak", [{"uuid": "a", "status": 2, "prompt": "q", "outputs": ["a"]}] * 2),
        (
            "openai-evals",
            [event("match", {"correct": True, "picked": "a", "expected": "a"})],
        ),
        (
            "openai-evals",
            [event("sampling", {"sampled": "a"}), event("sampling", {"sampled": "b"})],
        ),
        (
            "openai-evals",
            [
                event("sampling", {"sampled": "a"}),
                event("match", {"sampled": "b", "correct": False}),
            ],
        ),
        (
            "openai-evals",
            [
                event("sampling", {"sampled": "a"}, event_id=1),
                event("match", {"correct": True}, event_id=1),
            ],
        ),
        (
            "trulens",
            {
                "records": [{"record_id": "a", "input": "q", "output": "a"}],
                "feedback_columns": ["missing"],
            },
        ),
        (
            "trulens",
            {
                "records": [
                    {
                        "record_id": "a",
                        "main_input": "q",
                        "main_output": "a",
                        "feedback_results": [
                            {"name": "x", "record_id": "b", "result": 1}
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_malformed_or_aggregate_only_evidence_is_rejected(provider, native):
    with pytest.raises(EvaluationRecordsError):
        export_records(provider, native)


@pytest.mark.parametrize("provider", ["evidently", "mlflow"])
def test_tables_never_stringify_unknown_values_or_nonfinite_scores(provider):
    for bad in (object(), float("nan"), float("inf")):
        with pytest.raises(EvaluationRecordsError):
            export_records(
                provider,
                {
                    "rows": [
                        {
                            "record_id": "a",
                            "output": "a",
                            "prediction": "a",
                            "native_detail": bad,
                        }
                    ]
                },
            )


def test_duplicate_independent_ids_rejected():
    case = {"name": "same", "inputs": "q", "output": "a"}
    with pytest.raises(EvaluationRecordsError, match="duplicate"):
        export_records("pydantic-evals", {"cases": [case, case]})


def test_explicit_likelihood_is_carried_only_not_inferred_or_stringified():
    evidence = {"token_count": 2, "sum_logprob": -3.0}
    row = export_records(
        "openai-evals",
        [
            event(
                "sampling",
                {
                    "prompt": "q",
                    "sampled": "a",
                    "metadata": {"slice": "A", "invarlock_likelihood": evidence},
                    "timestamp": datetime(2026, 9, 20, tzinfo=UTC),
                },
            )
        ],
    )[0]
    assert row["likelihood"] == evidence and row["metadata"] == {"slice": "A"}
    assert (
        row["context"]["upstream_record"]["events"][0]["data"]["timestamp"]
        == "2026-09-20T00:00:00+00:00"
    )


@pytest.mark.parametrize("change", ["membership", "input", "output"])
def test_garak_source_cases_cannot_relabel_unrelated_execution(change):
    case = {
        "native_id": "attempt-a:0",
        "id": "planned",
        "input": "question",
        "expected": "answer",
    }
    case[{"membership": "native_id", "input": "input", "output": "output"}[change]] = (
        "different"
    )
    with pytest.raises(EvaluationRecordsError):
        export_records("garak", {"attempts": [Attempt()], "source_cases": [case]})


def test_trulens_supplied_feedback_joins_records_and_native_meta():
    rows = export_records(
        "trulens",
        {
            "records": [
                {
                    "record_id": "a",
                    "main_input": "q",
                    "main_output": "a",
                    "meta": {"slice": "A", "invarlock_likelihood": {"explicit": True}},
                }
            ],
            "feedback_results": [{"record_id": "a", "name": "quality", "result": 1}],
        },
    )
    assert rows[0]["metadata"] == {"slice": "A"}
    assert rows[0]["scores"] == {"quality": 1.0}
    assert rows[0]["likelihood"] == {"explicit": True}
    with pytest.raises(EvaluationRecordsError, match="unknown record IDs"):
        export_records(
            "trulens",
            {
                "records": [{"record_id": "a", "main_input": "q", "main_output": "a"}],
                "feedback_results": [
                    {"record_id": "other", "name": "quality", "result": 1}
                ],
            },
        )


@pytest.mark.parametrize(
    "provider",
    [
        "pydantic-evals",
        "evidently",
        "mlflow",
        "garak",
        "trulens",
        "openai-evals",
        "azure-ai-evaluation",
    ],
)
def test_installed_sdk_native_objects_offline(provider, monkeypatch, tmp_path):
    """Optional pinned-SDK probes; minimal installations retain contract tests."""
    import importlib.metadata
    import importlib.util
    import os
    import socket

    versions = {
        "pydantic-evals": "2.18.0",
        "evidently": "0.7.21",
        "mlflow": "3.14.0",
        "garak": "0.15.1",
        "trulens": "2.9.0",
        "azure-ai-evaluation": "1.18.1",
    }
    required = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", provider)
    package = "evals" if provider == "openai-evals" else provider
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        if required:
            pytest.fail(f"required SDK {package} is missing")
        pytest.skip(f"optional {package} SDK is not installed")
    if provider in versions and installed != versions[provider]:
        if required:
            pytest.fail(
                f"required SDK {package}=={versions[provider]}, found {installed}"
            )
        pytest.skip(f"requires pinned {package}=={versions[provider]}")
    if provider == "openai-evals":
        lock = (
            Path(__file__).parents[2]
            / "examples/evaluator-qualification/locks/openai-evals.txt"
        )
        expected_revision = lock.read_text().strip().rsplit("@", 1)[-1]
        direct_text = importlib.metadata.distribution(package).read_text(
            "direct_url.json"
        )
        direct = json.loads(direct_text) if direct_text is not None else {}
        if direct.get("vcs_info", {}).get("commit_id") != expected_revision:
            if required:
                pytest.fail("required OpenAI Evals source revision differs from lock")
            pytest.skip("requires pinned OpenAI Evals source revision")
    modules = {
        "pydantic-evals": "pydantic_evals",
        "evidently": "evidently",
        "mlflow": "mlflow",
        "garak": "garak",
        "trulens": "trulens",
        "openai-evals": "evals",
        "azure-ai-evaluation": "azure.ai.evaluation",
    }
    try:
        available = importlib.util.find_spec(modules[provider])
    except ModuleNotFoundError:
        available = None
    if available is None:
        if required:
            pytest.fail(f"required SDK {provider} cannot be imported")
        pytest.skip(f"optional {provider} SDK is not installed")

    def forbid_network(*args, **kwargs):
        raise AssertionError("SDK capture tests cannot access network services")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "unused-offline")
    monkeypatch.setenv("AZURE_TELEMETRY_DISABLED", "1")
    if provider == "pydantic-evals":
        import asyncio

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            value = Dataset(
                name="offline",
                cases=[Case(name="a", inputs="q", expected_output="yes")],
                evaluators=[EqualsExpected()],
            ).evaluate_sync(lambda value: "yes")
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    elif provider == "evidently":
        import pandas as pd
        from evidently import DataDefinition, Dataset
        from evidently.descriptors import ExactMatch

        value = Dataset.from_pandas(
            pd.DataFrame(
                [{"record_id": "a", "input": "q", "output": "yes", "reference": "yes"}]
            ),
            data_definition=DataDefinition(),
            descriptors=[
                ExactMatch(columns=["output", "reference"], alias="exact_match")
            ],
        )
    elif provider == "mlflow":
        import pandas as pd
        from mlflow.models import EvaluationResult
        from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact

        from tests.evaluation_records.sdk_batch_roundtrip import roundtrip

        result = EvaluationResult(metrics={"accuracy": 1.0}, artifacts={})
        value = {
            "metrics": result.metrics,
            "prediction_table": pd.DataFrame(
                [{"record_id": "a", "input": "q", "prediction": "yes", "target": "yes"}]
            ),
        }
        capture = serialize_results(provider, value)
        roundtrip(provider, capture, tmp_path=tmp_path / "artifact-roundtrip")
        load_content = JsonEvaluationArtifact._load_content_from_file

        def altered_artifact(self, path):
            content = load_content(self, path)
            content[0]["prediction"] = "changed after SDK save"
            return content

        # This must fail only after the actual SDK reload, proving the handoff
        # uses artifact bytes rather than the original in-memory prediction table.
        with monkeypatch.context() as patch:
            patch.setattr(
                JsonEvaluationArtifact, "_load_content_from_file", altered_artifact
            )
            with pytest.raises(AssertionError, match="SDK changed a field output"):
                roundtrip(provider, capture, tmp_path=tmp_path / "altered-artifact")
    elif provider == "garak":
        from garak.attempt import Attempt

        try:
            from garak.attempt import Message
        except ImportError:
            attempt = Attempt(prompt="q", status=2)
        else:
            attempt = Attempt(prompt=Message(text="q"), status=2)
        attempt.outputs = ["yes"]
        attempt.detector_results = {"native": [1.0]}
        value = [attempt]
    elif provider == "trulens":
        from trulens.core.schema.feedback import FeedbackResult
        from trulens.core.schema.record import Record

        record = Record(app_id="app", main_input="q", main_output="yes", calls=[])
        feedback = FeedbackResult(
            record_id=record.record_id, name="quality", result=1.0
        )
        value = {"records": [record], "feedback_results": [feedback]}
    elif provider == "azure-ai-evaluation":
        from azure.ai.evaluation import evaluate

        def exact_match(*, response: str, ground_truth: str) -> dict[str, float]:
            return {"exact_match": float(response == ground_truth)}

        data = tmp_path / "data.jsonl"
        data.write_text(
            json.dumps(
                {
                    "record_id": "a",
                    "input": "q",
                    "response": "yes",
                    "ground_truth": "yes",
                }
            )
            + "\n"
        )
        value = evaluate(
            data=data,
            evaluators={"exact_match": exact_match},
            output_path=tmp_path / "results.json",
            fail_on_evaluator_errors=True,
        )
    else:
        from evals.record import Event

        value = [
            Event(
                run_id="run",
                event_id=1,
                sample_id="a",
                type="sampling",
                data={"prompt": "q", "sampled": "yes"},
                created_by="offline",
                created_at="2026-09-20",
            )
        ]
    payload = serialize_results(provider, value)
    rows = export_records(provider, value)
    assert rows == export_records(
        provider, json.loads(json.dumps(payload, allow_nan=False))
    )
    assert len(rows) == 1
    assert rows[0]["error"] is None


@pytest.mark.parametrize(
    "provider",
    [
        "pydantic-evals",
        "azure-ai-evaluation",
        "evidently",
        "mlflow",
        "garak",
        "openai-evals",
        "trulens",
    ],
)
def test_upstream_grading_is_optional_and_explicit_observations_are_portable(provider):
    metadata = {
        "slice": "short",
        "invarlock_scores": {"custom": 0.8},
        "invarlock_likelihood": {"explicit": True},
    }
    native = {
        "pydantic-evals": {
            "cases": [{"name": "a", "inputs": "q", "output": "a", "metadata": metadata}]
        },
        "azure-ai-evaluation": {
            "rows": [
                {
                    "inputs.record_id": "a",
                    "inputs.input": "q",
                    "inputs.response": "a",
                    "inputs.metadata": metadata,
                }
            ]
        },
        "evidently": {
            "rows": [
                {"record_id": "a", "input": "q", "output": "a", "metadata": metadata}
            ]
        },
        "mlflow": {
            "prediction_table": [
                {
                    "record_id": "a",
                    "input": "q",
                    "prediction": "a",
                    "metadata": metadata,
                }
            ]
        },
        "garak": {
            "attempts": [
                {
                    "uuid": "a",
                    "status": 2,
                    "prompt": "q",
                    "outputs": ["a"],
                    "metadata": metadata,
                }
            ]
        },
        "openai-evals": {
            "events": [
                event("sampling", {"prompt": "q", "sampled": "a", "metadata": metadata})
            ]
        },
        "trulens": {
            "records": [
                {
                    "record_id": "a",
                    "main_input": "q",
                    "main_output": "a",
                    "meta": metadata,
                }
            ]
        },
    }[provider]
    row = export_records(provider, native)[0]
    assert row["scores"] == {"custom": 0.8}
    assert row["metadata"] == {"slice": "short"}
    assert row["likelihood"] == {"explicit": True}
    assert row["output"] == "a" and row["error"] is None
    assert metadata["invarlock_scores"] == {"custom": 0.8}


def test_explicit_observation_cannot_override_native_score():
    with pytest.raises(EvaluationRecordsError, match="conflicts"):
        export_records(
            "evidently",
            {
                "rows": [
                    {
                        "record_id": "a",
                        "output": "a",
                        "exact_match": True,
                        "metadata": {"invarlock_scores": {"exact_match": 0}},
                    }
                ]
            },
        )
    row = export_records(
        "evidently",
        {
            "rows": [
                {
                    "record_id": "a",
                    "output": "a",
                    "exact_match": True,
                    "metadata": {"invarlock_scores": {"exact_match": 1}},
                }
            ]
        },
    )[0]
    assert row["scores"] == {"exact_match": 1}


def test_azure_failure_without_response_is_retained():
    row = export_records(
        "azure-ai-evaluation",
        {
            "rows": [
                {
                    "inputs.record_id": "a",
                    "inputs.query": "q",
                    "error": {"message": "generation failed"},
                }
            ]
        },
    )[0]
    assert row["error"] == "generation failed" and row["output"] is None
