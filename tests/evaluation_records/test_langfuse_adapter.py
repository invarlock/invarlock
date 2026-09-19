"""Langfuse imports preserve identity and do not upgrade scalar observations."""

import copy
import hashlib
import json

import pytest

from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.adapters import _parse_run_bytes
from invarlock.evaluator_capture import evaluator_input_capabilities

SOURCE = {"name": "langfuse", "version": "4.14.1"}
OPTIONS = {
    "adapter": "langfuse-json",
    "source": SOURCE,
    "run_id": "run",
    "artifact_digest": "sha256:" + "a" * 64,
}


def export():
    return {
        "format": "invarlock/langfuse-export-v1",
        "sdk_version": "4.14.1",
        "result": {
            "name": "experiment",
            "run_name": "run",
            "experiment_id": "experiment-id",
            "run_evaluations": [],
            "item_results": [
                {
                    "item": {
                        "input": {"question": "question", "context": "original"},
                        "expected_output": "yes",
                        "metadata": {"invarlock_id": "case-1", "category": "basic"},
                    },
                    "output": "yes",
                    "evaluations": [
                        {"name": "match", "value": True, "data_type": "BOOLEAN"}
                    ],
                    "trace_id": "trace-id",
                    "dataset_run_id": None,
                }
            ],
        },
    }


def parse(value, **options):
    return _parse_run_bytes(json.dumps(value).encode(), **{**OPTIONS, **options})


def test_identity_raw_binding_and_projection():
    native = export()
    run = parse(
        native, input_projection={"kind": "json-pointer", "pointer": "/input/question"}
    )
    row = run["records"][0]
    assert row["id"] == "case-1" and row["input"] == "question"
    assert (
        row["context"]["langfuse"]["item_result"] == native["result"]["item_results"][0]
    )
    assert row["context"]["langfuse"]["experiment_id"] == "experiment-id"
    assert (
        row["context"]["input_projection"]["source"]["input"]
        == native["result"]["item_results"][0]["item"]["input"]
    )
    assert row["scores"] == {"match": 1.0}
    assert (
        run["source_digest"]
        == "sha256:" + hashlib.sha256(json.dumps(native).encode()).hexdigest()
    )
    assert evaluator_input_capabilities(run)["judge"]["usable_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(format="invarlock/langfuse-export-v2"),
        lambda x: x.update(unknown=True),
        lambda x: x.update(sdk_version="other"),
        lambda x: x["result"].update(run_name="other"),
        lambda x: x["result"].update(item_results=[]),
        lambda x: x["result"].update(item_results=[{}]),
        lambda x: x["result"]["item_results"][0]["item"]["metadata"].pop(
            "invarlock_id"
        ),
        lambda x: x["result"]["item_results"][0]["item"].update(id="different"),
        lambda x: x["result"]["item_results"][0]["item"]["metadata"].update(
            invarlock_id=True
        ),
        lambda x: x["result"]["item_results"][0].pop("output"),
        lambda x: x["result"]["item_results"][0].update(dataset_run_id="other"),
        lambda x: x["result"]["item_results"][0]["evaluations"].append(
            {"name": "match", "value": 0}
        ),
        lambda x: x["result"]["item_results"][0]["evaluations"][0].update(
            data_type="NUMERIC"
        ),
        lambda x: x["result"]["item_results"][0]["item"]["metadata"].update(
            invarlock_likelihood={"token_count": 4}
        ),
        lambda x: x["result"]["item_results"][0]["item"]["metadata"].update(
            invarlock_error=False
        ),
        lambda x: x["result"]["item_results"].append(
            copy.deepcopy(x["result"]["item_results"][0])
        ),
    ],
)
def test_malformed_exports_fail_closed(change):
    native = export()
    change(native)
    with pytest.raises(EvaluationRecordsError):
        parse(native)


@pytest.mark.parametrize(
    "output,error",
    [(None, None), (None, "timeout"), ({"answer": "yes"}, None), ("yes", "timeout")],
)
def test_null_failed_and_structured_outputs_are_not_fabricated(output, error):
    native = export()
    item = native["result"]["item_results"][0]
    item["output"] = output
    item["item"]["metadata"]["invarlock_error"] = error
    run = parse(native)
    row = run["records"][0]
    assert row["output"] == output and row["error"] == error
    assert evaluator_input_capabilities(run)["exact_match"]["usable_count"] == 0


def test_scores_and_tokens_cannot_create_likelihood_or_judge_evidence():
    native = export()
    item = native["result"]["item_results"][0]
    item["evaluations"] += [
        {"name": "judge", "value": 0.95},
        {"name": "label", "value": "pass", "data_type": "CATEGORICAL"},
    ]
    item["item"]["metadata"]["usage"] = {"input_tokens": 5, "output_tokens": 3}
    run = parse(native)
    assert run["records"][0]["scores"] == {"match": 1.0, "judge": 0.95}
    assert "likelihood" not in run["records"][0]
    assert (
        evaluator_input_capabilities(run)["normalized_nll_per_utf8_byte"][
            "usable_count"
        ]
        == 0
    )
    assert run["score_provenance"] == {}


def test_explicit_likelihood_uses_shared_binding_validation():
    native = export()
    item = native["result"]["item_results"][0]["item"]
    facts = {
        "basis": "reference_continuation",
        "logprob_sum": -3.0,
        "token_count": 1,
        "utf8_byte_count": 3,
        "input_digest": digest(item["input"]),
        "reference_digest": digest("yes"),
        "artifact_digest": OPTIONS["artifact_digest"],
        "configuration_digest": digest("configuration"),
        "tokenizer_digest": digest("tokenizer"),
        "source": SOURCE,
    }
    item["metadata"]["invarlock_likelihood"] = facts
    run = parse(
        native, input_projection={"kind": "json-pointer", "pointer": "/input/question"}
    )
    assert run["records"][0]["likelihood"] == facts
    assert (
        evaluator_input_capabilities(run)["normalized_nll_per_utf8_byte"][
            "usable_count"
        ]
        == 1
    )
    facts["input_digest"] = digest("wrong")
    with pytest.raises(EvaluationRecordsError, match="binding differs"):
        parse(native)


def test_dataset_identity_bindings():
    native = export()
    native["result"]["dataset_run_id"] = "experiment-id"
    item = native["result"]["item_results"][0]
    item["dataset_run_id"] = "experiment-id"
    item["item"].update(id="case-1")
    item["item"]["metadata"].pop("invarlock_id")
    assert parse(native)["records"][0]["id"] == "case-1"
    native["result"]["dataset_run_id"] = "other"
    with pytest.raises(EvaluationRecordsError):
        parse(native)


@pytest.mark.parametrize(
    "path,value",
    [
        (("result", "run_evaluations"), {}),
        (("result", "description"), False),
        (("result", "item_results", 0, "item", "metadata"), []),
        (("result", "item_results", 0, "trace_id"), 123),
        (("result", "item_results", 0, "evaluations", 0, "metadata"), []),
        (("result", "item_results", 0, "evaluations", 0, "comment"), 4),
        (("result", "item_results", 0, "evaluations", 0, "config_id"), 4),
        (("result", "item_results", 0, "evaluations", 0, "value"), "true"),
        (("result", "item_results", 0, "evaluations", 0, "name"), ""),
        (("result", "item_results", 0, "item", "metadata", "invarlock_id"), "x" * 4097),
    ],
)
def test_malformed_optional_fields_rejected(path, value):
    native = export()
    target = native
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(EvaluationRecordsError):
        parse(native)


def test_hosted_missing_item_id_is_rejected():
    native = export()
    native["result"]["dataset_run_id"] = "experiment-id"
    native["result"]["item_results"][0]["dataset_run_id"] = "experiment-id"
    with pytest.raises(EvaluationRecordsError, match="native ID"):
        parse(native)


def test_hosted_absent_metadata_and_null_reference_remain_usable_for_judge():
    native = export()
    native["result"].update(
        dataset_run_id="experiment-id",
        dataset_run_url="https://example.org/run",
        description="",
    )
    row = native["result"]["item_results"][0]
    row["dataset_run_id"] = "experiment-id"
    row["item"] = {"id": "dataset-item", "input": "question", "metadata": None}
    row["evaluations"] = [
        {
            "name": "numeric",
            "value": 0.5,
            "data_type": "NUMERIC",
            "comment": "",
            "metadata": {},
            "config_id": None,
        },
        {"name": "binary", "value": 0, "data_type": "BOOLEAN"},
    ]
    run = parse(native)
    assert run["records"][0]["expected"] is None
    assert run["records"][0]["scores"] == {"numeric": 0.5, "binary": 0.0}
    assert evaluator_input_capabilities(run)["judge"]["usable_count"] == 1


def test_record_and_evaluation_limits(monkeypatch):
    from invarlock.evaluation_records import langfuse

    monkeypatch.setattr(langfuse, "MAX_RECORDS", 1)
    native = export()
    native["result"]["item_results"] *= 2
    with pytest.raises(EvaluationRecordsError, match="bounded array"):
        parse(native)
    native = export()
    native["result"]["run_evaluations"] = [
        {"name": "one", "value": 1},
        {"name": "two", "value": 2},
    ]
    with pytest.raises(EvaluationRecordsError, match="bounded array"):
        parse(native)


def test_missing_source_and_duplicate_json_keys_fail():
    with pytest.raises(EvaluationRecordsError, match="source name/version"):
        parse(export(), source=None)
    raw = json.dumps(export()).replace(
        '"sdk_version": "4.14.1"', '"sdk_version": "4.14.1", "sdk_version": "other"'
    )
    with pytest.raises(EvaluationRecordsError, match="duplicate"):
        _parse_run_bytes(raw.encode(), **OPTIONS)


def hosted_export():
    native = export()
    native["result"]["dataset_run_id"] = "experiment-id"
    row = native["result"]["item_results"][0]
    row["dataset_run_id"] = "experiment-id"
    row["item"].update(
        id="case-1",
        status="ACTIVE",
        dataset_id="dataset",
        dataset_name="Questions",
        source_trace_id=None,
        source_observation_id="observation",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
        media_references=[
            {
                "field": "input",
                "reference_string": "opaque media reference",
                "json_path": "$.image",
                "media": {"id": "media-id"},
            }
        ],
    )
    return native


def test_hosted_public_fields_are_retained_without_fetching_media():
    native = hosted_export()
    row = parse(native)["records"][0]
    assert (
        row["context"]["langfuse"]["item_result"] == native["result"]["item_results"][0]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "UNKNOWN"),
        ("dataset_id", 1),
        ("media_references", {}),
        ("media_references", [False]),
    ],
)
def test_invalid_hosted_public_fields_fail(field, value):
    native = hosted_export()
    native["result"]["item_results"][0]["item"][field] = value
    with pytest.raises(EvaluationRecordsError):
        parse(native)


def test_hosted_fields_cannot_hide_a_missing_native_id():
    native = hosted_export()
    native["result"]["item_results"][0]["item"].pop("id")
    with pytest.raises(EvaluationRecordsError, match="native ID"):
        parse(native)
