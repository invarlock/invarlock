"""Adversarial review of public capture boundaries and native failure semantics."""

from __future__ import annotations

import copy
import json

import pytest

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.adapters import _harness, _inspect, _promptfoo, _rows
from invarlock.evaluation_records.batch_integrations import export_records
from invarlock.evaluation_records.integrations import export_evaluator_result


def inspect_log():
    return {
        "version": 2,
        "status": "success",
        "samples": [
            {
                "id": "a",
                "input": "q",
                "target": "a",
                "output": {"choices": [{"message": {"content": "a"}}]},
                "scores": {},
                "epoch": 1,
            }
        ],
    }


def harness_row():
    return {
        "doc_id": "a",
        "doc": "q",
        "target": "a",
        "arguments": [["q", {}]],
        "filtered_resps": ["a"],
    }


def promptfoo_row():
    return {
        "testIdx": 0,
        "promptIdx": 0,
        "testCase": {"vars": {"input": "q"}},
        "prompt": "q",
        "response": {"output": "a"},
    }


@pytest.mark.parametrize(
    "evaluator", ["inspect-ai", "lm-evaluation-harness", "promptfoo"]
)
@pytest.mark.parametrize("metadata", [False, True, [], "", 0])
def test_public_export_rejects_malformed_metadata_without_publishing(
    evaluator, metadata, tmp_path
):
    if evaluator == "inspect-ai":
        payload = inspect_log()
        payload["samples"][0]["metadata"] = metadata
    elif evaluator == "lm-evaluation-harness":
        payload = [harness_row()]
        payload[0]["metadata"] = metadata
    else:
        payload = [promptfoo_row()]
        payload[0]["testCase"]["metadata"] = metadata
    destination = tmp_path / "rejected.json"
    with pytest.raises(EvaluationRecordsError, match="metadata"):
        export_evaluator_result(
            evaluator,
            payload,
            destination,
            expected_ids=["a"],
            source_version="1",
            run_id="run",
            artifact_digest="sha256:" + "a" * 64,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "evaluator", ["inspect-ai", "lm-evaluation-harness", "promptfoo"]
)
def test_null_metadata_is_absent_metadata_without_erasing_native_context(evaluator):
    if evaluator == "inspect-ai":
        payload = inspect_log()
        payload["samples"][0]["metadata"] = None
        row = _inspect(payload)[0]
    elif evaluator == "lm-evaluation-harness":
        payload = harness_row()
        payload["metadata"] = None
        row = _harness([payload])[0]
    else:
        payload = promptfoo_row()
        payload["testCase"]["metadata"] = None
        row = _promptfoo([payload])[0]
    assert row["metadata"] == {}
    assert row["output"] == "a"
    native = row["context"]["upstream_record"]
    assert (native["testCase"] if evaluator == "promptfoo" else native)[
        "metadata"
    ] is None


def test_inspect_failed_sample_does_not_authorize_selecting_one_of_multiple_answers():
    log = inspect_log()
    row = log["samples"][0]
    row["error"] = {"message": "sample failed"}
    row["output"]["choices"].append({"message": {"content": "different"}})
    with pytest.raises(EvaluationRecordsError, match="one completion"):
        _inspect(log)
    row["output"]["choices"] = []
    captured = _inspect(log)[0]
    assert captured["output"] is None and captured["error"] == "upstream_error"
    assert captured["context"]["upstream_record"]["error"] == row["error"]


@pytest.mark.parametrize("epoch", [True, False, 1.0, "1"])
def test_inspect_epoch_identity_requires_an_integer(epoch):
    log = inspect_log()
    log["samples"][0]["epoch"] = epoch
    with pytest.raises(EvaluationRecordsError, match="epochs"):
        _inspect(log)


@pytest.mark.parametrize("field", ["scores", "output"])
@pytest.mark.parametrize("value", [False, [], 0, ""])
def test_inspect_false_like_wrong_types_are_not_treated_as_absent(field, value):
    log = inspect_log()
    log["samples"][0][field] = value
    with pytest.raises(EvaluationRecordsError, match=field):
        _inspect(log)


def test_harness_task_failure_retains_failed_case_and_actual_output_if_available():
    row = harness_row()
    row["error"] = {"message": "generation interrupted"}
    captured = _harness([row])[0]
    assert captured["output"] == "a" and captured["error"] == "upstream_error"
    row["filtered_resps"] = [None]
    captured = _harness([row])[0]
    assert captured["output"] is None and captured["error"] == "upstream_error"
    assert captured["context"]["upstream_record"]["error"] == row["error"]


def test_rows_reject_count_before_reading_native_record_shapes(monkeypatch):
    monkeypatch.setattr("invarlock.evaluation_records.adapters.MAX_RECORDS", 1)
    with pytest.raises(EvaluationRecordsError):
        _rows([object(), object()], "native")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 10**400])
def test_native_scores_cannot_escape_finite_observation_boundary(score):
    log = inspect_log()
    log["samples"][0]["scores"] = {"quality": {"value": score}}
    with pytest.raises(EvaluationRecordsError, match="finite"):
        _inspect(log)


@pytest.mark.parametrize(
    "evaluator", ["pydantic-evals", "trulens", "azure-ai-evaluation"]
)
def test_failed_upstream_grading_does_not_fail_a_successful_model_output(evaluator):
    payloads = {
        "pydantic-evals": {
            "cases": [
                {
                    "name": "a",
                    "inputs": "q",
                    "output": "a",
                    "expected_output": "a",
                    "evaluator_failures": [{"error_message": "upstream judge timeout"}],
                }
            ]
        },
        "trulens": {
            "records": [
                {
                    "record_id": "a",
                    "main_input": "q",
                    "main_output": "a",
                    "ground_truth": "a",
                    "feedback_results": [
                        {
                            "name": "judge",
                            "result": None,
                            "status": "failed",
                            "error": "upstream judge timeout",
                        }
                    ],
                }
            ]
        },
        "azure-ai-evaluation": {
            "rows": [
                {
                    "inputs.record_id": "a",
                    "inputs.input": "q",
                    "inputs.response": "a",
                    "inputs.ground_truth": "a",
                    "outputs.judge.error": {"message": "upstream judge timeout"},
                }
            ]
        },
    }
    untouched = copy.deepcopy(payloads[evaluator])
    row = export_records(evaluator, payloads[evaluator])[0]
    assert row["error"] is None
    assert row["output"] == row["expected"] == "a"
    assert row["scores"] == {}
    assert "upstream judge timeout" in json.dumps(row["context"])
    assert payloads[evaluator] == untouched


def test_unsupported_source_and_bad_version_cannot_publish(tmp_path):
    payload = [{"id": "a", "input": "q", "output": "a"}]
    for evaluator, version in [
        ("not-an-evaluator", "1"),
        ("autoevals", True),
        ("autoevals", ""),
    ]:
        destination = tmp_path / "rejected.json"
        with pytest.raises(EvaluationRecordsError):
            export_evaluator_result(
                evaluator,
                payload,
                destination,
                expected_ids=["a"],
                source_version=version,
                run_id="run",
                artifact_digest="sha256:" + "a" * 64,
            )
        assert not destination.exists()


@pytest.mark.parametrize("source_case", [False, True])
@pytest.mark.parametrize("status", [1, 2])
def test_garak_likelihood_only_completion_does_not_invent_generation_failure(
    source_case, status
):
    metadata = {
        "invarlock_likelihood": {"explicit": "central validation binds these fields"}
    }
    attempt = {"uuid": "a", "status": status, "prompt": "q", "outputs": [None]}
    payload = {"attempts": [attempt]}
    if source_case:
        payload["source_cases"] = [
            {
                "native_id": "a:0",
                "id": "planned",
                "input": "q",
                "expected": "answer",
                "metadata": metadata,
            }
        ]
    else:
        attempt["metadata"] = metadata
    row = export_records("garak", payload)[0]
    assert row["output"] is None and "likelihood" in row
    assert row["error"] == ("Garak attempt incomplete" if status == 1 else None)
    attempt["error"] = "actual task failure"
    row = export_records("garak", payload)[0]
    assert row["error"] == "actual task failure"


@pytest.mark.parametrize(
    ("evaluator", "payload"),
    [
        ("pydantic-evals", {"cases": [], "failures": [{"name": "a", "inputs": "q"}]}),
        (
            "pydantic-evals",
            {
                "cases": [
                    {"name": "a", "inputs": "q", "output": "a", "error": {"code": 500}}
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
                        "likelihood": {"x": 1},
                        "metadata": {"invarlock_likelihood": {"x": 2}},
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
                        "scores": {"overflow": 10**400},
                    }
                ]
            },
        ),
        (
            "azure-ai-evaluation",
            {
                "rows": [
                    {
                        "inputs.record_id": "a",
                        "inputs.response": "a",
                        "metadata": {"slice": "a"},
                        "inputs.metadata": {"slice": "b"},
                    }
                ]
            },
        ),
        ("azure-ai-evaluation", {"rows": [{"inputs.record_id": "a"}]}),
        ("mlflow", {"rows": {"columns": ["record_id", "prediction"], "data": [["a"]]}}),
        (
            "evidently",
            {
                "rows": [{"record_id": "a", "output": "a"}],
                "columns": {"unknown": "field"},
            },
        ),
        (
            "evidently",
            {"rows": [{"record_id": "a", "output": "a"}], "columns": {"output": False}},
        ),
        (
            "evidently",
            {"rows": [{"record_id": "a", "output": "a"}], "score_columns": ["x", "x"]},
        ),
        (
            "evidently",
            {"rows": [{"record_id": "a", "output": "a"}], "score_columns": [False]},
        ),
        ("garak", {"entries": [{"entry_type": "unrecognized"}]}),
        (
            "garak",
            {
                "attempts": [
                    {"uuid": "a", "status": True, "prompt": "q", "outputs": ["a"]}
                ]
            },
        ),
        (
            "garak",
            {"attempts": [{"uuid": "a", "status": 9, "prompt": "q", "outputs": ["a"]}]},
        ),
        ("garak", {"attempts": [{"uuid": "a", "status": 2, "outputs": ["a"]}]}),
        (
            "garak",
            {
                "attempts": [
                    {"uuid": "a", "status": 1, "prompt": "q", "outputs": []},
                    {"uuid": "a", "status": 2, "prompt": "changed", "outputs": ["a"]},
                ]
            },
        ),
        (
            "trulens",
            {
                "records": [{"record_id": "a", "input": "q", "output": "a"}],
                "feedback_columns": ["x", "x"],
            },
        ),
        (
            "trulens",
            {
                "records": [{"record_id": "a", "input": "q", "output": "a"}],
                "feedback_columns": [False],
            },
        ),
        (
            "trulens",
            {
                "records": [
                    {
                        "record_id": "a",
                        "input": "q",
                        "output": "a",
                        "meta": {},
                        "metadata": {"slice": "a"},
                    }
                ]
            },
        ),
        ("trulens", {"records": [{"record_id": "a", "input": "q"}]}),
        ("trulens", {"records": [{"record_id": "a", "output": "a"}]}),
        (
            "trulens",
            {
                "records": [
                    {
                        "record_id": "a",
                        "input": "q",
                        "output": "a",
                        "feedback_results": [
                            {"name": "x", "result": 1},
                            {"name": "x", "result": 1},
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_batch_ambiguous_sources_and_malformed_native_details_are_rejected(
    evaluator, payload
):
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, payload)


@pytest.mark.parametrize(
    "conflict", ["run", "metric", "correct", "metadata", "likelihood"]
)
def test_openai_evals_rejects_ambiguous_sample_claims(conflict):
    events = [
        {
            "sample_id": "a",
            "run_id": "run",
            "type": "sampling",
            "data": {
                "prompt": "q",
                "sampled": "a",
                "metadata": {"slice": "a"},
                "likelihood": {"x": 1},
            },
        },
        {
            "sample_id": "a",
            "run_id": "run",
            "type": "match",
            "data": {"correct": True, "expected": "a"},
        },
    ]
    if conflict == "run":
        events[1]["run_id"] = "other"
    elif conflict == "metric":
        events.append({"sample_id": "a", "type": "metrics", "data": {"match": 1}})
    elif conflict == "correct":
        events[1]["data"]["correct"] = 1
    elif conflict == "metadata":
        events[1]["data"]["metadata"] = {"slice": "b"}
    else:
        events[1]["data"]["likelihood"] = {"x": 2}
    with pytest.raises(EvaluationRecordsError):
        export_records("openai-evals", {"events": events})


def test_openai_evals_http_final_report_and_explicit_evidence_are_retained():
    likelihood = {"explicit": True}
    row = export_records(
        "openai-evals",
        [
            {"type": "final_report", "sample_id": None, "data": {"metric": 0.1}},
            {
                "type": "match",
                "sample_id": "a",
                "event_id": 0,
                "data": {
                    "prompt": "q",
                    "sampled": "a",
                    "expected": "a",
                    "correct": True,
                    "likelihood": likelihood,
                },
            },
            {"type": "extra", "sample_id": "a", "data": {"likelihood": likelihood}},
        ],
    )[0]
    assert row["likelihood"] == likelihood
    assert row["context"]["upstream_summary"]["log_entries"][0]["data"] == {
        "metric": 0.1
    }


@pytest.mark.parametrize("conflict", ["duplicate", "missing_expected", "likelihood"])
def test_garak_independent_source_evidence_cannot_conflict(conflict):
    attempt = {
        "uuid": "a",
        "status": 2,
        "prompt": "q",
        "outputs": ["a"],
        "metadata": {"invarlock_likelihood": {"x": 1}},
    }
    case = {
        "native_id": "a:0",
        "id": "planned",
        "input": "q",
        "expected": "a",
        "output": "a",
        "metadata": {"invarlock_likelihood": {"x": 1}},
    }
    sources = [case]
    if conflict == "duplicate":
        sources.append(copy.deepcopy(case))
    elif conflict == "missing_expected":
        del case["expected"]
    else:
        case["metadata"]["invarlock_likelihood"] = {"x": 2}
    with pytest.raises(EvaluationRecordsError):
        export_records("garak", {"attempts": [attempt], "source_cases": sources})


def test_native_null_scores_preserve_missingness_and_task_error_messages():
    row = export_records(
        "evidently",
        {
            "rows": [
                {
                    "record_id": "a",
                    "exact_match": None,
                    "error": {"error_message": "task failed"},
                }
            ]
        },
    )[0]
    assert (
        row["scores"] == {} and row["output"] is None and row["error"] == "task failed"
    )
    row = export_records(
        "trulens",
        {
            "records": [
                {
                    "record_id": "a",
                    "main_input": "q",
                    "main_error": {"msg": "task failed"},
                    "quality": None,
                }
            ],
            "feedback_columns": ["quality"],
        },
    )[0]
    assert row["scores"] == {} and row["error"] == "task failed"


def test_native_json_protocols_preserve_supported_values_without_general_stringification():
    from datetime import date, timedelta
    from enum import Enum
    from types import SimpleNamespace
    from uuid import UUID

    from invarlock.evaluation_records.batch_integrations import serialize_results

    class Status(Enum):
        READY = "ready"

    class Model:
        def model_dump(self, *, mode):
            assert mode == "python"
            return {
                "cases": [
                    {
                        "name": "a",
                        "inputs": "q",
                        "output": "a",
                        "metadata": {
                            "day": date(2026, 9, 20),
                            "duration": timedelta(seconds=2),
                            "status": Status.READY,
                            "uuid": UUID(int=0),
                        },
                    }
                ]
            }

    payload = serialize_results("pydantic-evals", Model())
    assert payload["cases"][0]["metadata"] == {
        "day": "2026-09-20",
        "duration": 2.0,
        "status": "ready",
        "uuid": "00000000-0000-0000-0000-000000000000",
    }

    class Scalar:
        __module__ = "numpy"

        def item(self):
            return 0.75

    class Record:
        __module__ = "trulens.core.schema.record"
        model_fields = {
            key: SimpleNamespace(exclude=key == "runtime_future")
            for key in ["record_id", "main_input", "main_output", "runtime_future"]
        }
        record_id = "a"
        main_input = "q"
        main_output = "a"
        runtime_future = object()

    rows = export_records(
        "trulens",
        {
            "records": [Record()],
            "feedback_results": [
                {"record_id": "a", "name": "quality", "result": Scalar()}
            ],
        },
    )
    assert rows[0]["scores"] == {"quality": 0.75}
    assert "runtime_future" not in rows[0]["context"]["upstream_record"]


def test_native_json_rejects_cycles_nonstrings_and_nested_capacity_overflow(
    monkeypatch,
):
    from invarlock.evaluation_records.batch_integrations import serialize_results

    case = {"name": "a", "inputs": "q", "output": "a"}
    case["metadata"] = {"loop": case}
    with pytest.raises(EvaluationRecordsError, match="nesting"):
        serialize_results("pydantic-evals", {"cases": [case]})
    with pytest.raises(EvaluationRecordsError, match="string keys"):
        serialize_results(
            "pydantic-evals",
            {
                "cases": [
                    {
                        "name": "a",
                        "inputs": "q",
                        "output": "a",
                        "metadata": {1: "wrong"},
                    }
                ]
            },
        )
    monkeypatch.setattr(
        "invarlock.evaluation_records.batch_integrations.MAX_RECORDS", 1
    )
    with pytest.raises(EvaluationRecordsError, match="record limit"):
        serialize_results(
            "pydantic-evals",
            {"cases": [{"name": "a", "inputs": "q", "output": "a", "details": [1, 2]}]},
        )
    with pytest.raises(EvaluationRecordsError, match="unsupported"):
        serialize_results("unsupported", {})


def test_harness_logger_named_arguments_bind_the_recorded_continuation():
    row = harness_row()
    row["arguments"] = {"gen_args_0": {"arg_0": "q", "arg_1": "a"}}
    row["filtered_resps"] = [[-0.5, True]]
    row["metadata"] = {"invarlock_likelihood": {"logprob_sum": -0.5}}
    captured = _harness([row])[0]
    assert captured["output"] is None
    assert captured["context"]["arguments"] == row["arguments"]
    for malformed in (
        {"gen_args_0": {"arg_0": "q", "arg_1": "wrong"}},
        {"gen_args_0": {"arg_0": "q", "arg_1": "a", "arg_2": "extra"}},
        {
            "gen_args_0": {"arg_0": "q", "arg_1": "a"},
            "gen_args_1": {"arg_0": "q", "arg_1": "a"},
        },
    ):
        row["arguments"] = malformed
        with pytest.raises(EvaluationRecordsError, match="likelihood facts differ"):
            _harness([row])


@pytest.mark.parametrize("measured,greedy", [("-0.5", "True"), ("-0.5", "False")])
def test_harness_logger_stringified_likelihood_must_match_explicit_facts_exactly(
    measured, greedy
):
    row = harness_row()
    row["arguments"] = {"gen_args_0": {"arg_0": "q", "arg_1": "a"}}
    row["filtered_resps"] = [[measured, greedy]]
    row["metadata"] = {"invarlock_likelihood": {"logprob_sum": -0.5}}
    captured = _harness([row])[0]
    assert captured["output"] is None
    assert captured["context"]["upstream_record"]["filtered_resps"] == [
        [measured, greedy]
    ]
    for invalid in [
        ["-0.50", "True"],
        [" -0.5", "True"],
        ["-0.5", "true"],
        ["-0.5", "1"],
    ]:
        row["filtered_resps"] = [invalid]
        with pytest.raises(EvaluationRecordsError, match="likelihood facts differ"):
            _harness([row])


@pytest.mark.parametrize(
    "grade",
    [
        "excellent",
        {"quality": "good", "safety": True},
        ["correct", "clear"],
        True,
        False,
        None,
    ],
)
def test_new_inspect_profile_preserves_native_grades_without_blocking_rescoring(
    grade, tmp_path
):
    from invarlock.evaluation_records.adapters import load_run
    from invarlock.evaluator_capture import evaluator_input_capabilities

    payload = inspect_log()
    sample = payload["samples"][0]
    sample["scores"] = {
        "native_grade": {"value": grade, "explanation": "native rationale"},
        "numeric": {"value": 0.75},
        "match": {"value": "C"},
    }
    sample["metadata"] = {"slice": "small", "invarlock_scores": {"custom": 0.5}}
    destination = tmp_path / "new-profile.json"
    run = export_evaluator_result(
        "inspect-ai",
        payload,
        destination,
        expected_ids=["a"],
        source_version="1",
        run_id="run",
        artifact_digest="sha256:" + "a" * 64,
    )
    row = run["records"][0]
    assert row["scores"] == {"numeric": 0.75, "match": 1.0, "custom": 0.5}
    assert (
        row["context"]["upstream_record"]["scores"]["native_grade"]
        == sample["scores"]["native_grade"]
    )
    assert row["metadata"] == {"slice": "small"}
    capabilities = evaluator_input_capabilities(run)
    assert capabilities["exact_match"]["usable_count"] == 1
    assert capabilities["judge"]["usable_count"] == 1
    loaded = load_run(
        destination,
        adapter="evaluator-json",
        source={"name": "inspect-ai", "version": "1"},
        run_id="run",
        artifact_digest="sha256:" + "a" * 64,
    )
    assert loaded == run
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload))
    with pytest.raises(EvaluationRecordsError, match="not numeric"):
        load_run(
            legacy,
            adapter="inspect-json",
            source={"name": "inspect-ai", "version": "1"},
            run_id="run",
            artifact_digest="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize(
    "grade", [float("nan"), float("inf"), {"nested": float("nan")}]
)
def test_new_inspect_profile_still_rejects_nonfinite_grade_data(grade, tmp_path):
    payload = inspect_log()
    payload["samples"][0]["scores"] = {"grade": {"value": grade}}
    destination = tmp_path / "invalid.json"
    with pytest.raises(EvaluationRecordsError, match="finite"):
        export_evaluator_result(
            "inspect-ai",
            payload,
            destination,
            expected_ids=["a"],
            source_version="1",
            run_id="run",
            artifact_digest="sha256:" + "a" * 64,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "evaluator",
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
@pytest.mark.parametrize("metadata", [None, False, [], "", 0])
def test_batch_native_metadata_requires_object_or_null(evaluator, metadata):
    payload = {
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
                {
                    "sample_id": "a",
                    "type": "sampling",
                    "data": {"prompt": "q", "sampled": "a", "metadata": metadata},
                }
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
    }[evaluator]
    if metadata is not None:
        with pytest.raises(EvaluationRecordsError, match="metadata"):
            export_records(evaluator, payload)
    else:
        row = export_records(evaluator, payload)[0]
        assert row["metadata"] == {} and row["output"] == "a"
