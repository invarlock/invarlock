"""Public native imports retain source identity and reject malformed observations."""

from __future__ import annotations

import hashlib
import json

import pytest

from invarlock.engine import (
    EvaluationRecordsError,
    digest,
    load_run,
    write_evaluator_export,
)

ARTIFACT = "sha256:" + "a" * 64


def inspect_log():
    return {
        "version": 2,
        "status": "success",
        "samples": [
            {
                "id": "one",
                "input": "Q",
                "target": "A",
                "output": {"choices": [{"message": {"content": "A"}}]},
                "metadata": {"slice": "test"},
                "scores": {},
            }
        ],
    }


def options(evaluator="inspect-ai", version="0.3.254"):
    return {
        "adapter": "evaluator-native-json",
        "source": {"name": evaluator, "version": version},
        "run_id": "run",
        "artifact_digest": ARTIFACT,
    }


def test_bare_inspect_export_is_bound_to_original_bytes(tmp_path):
    native = inspect_log()
    raw = json.dumps(native, indent=2).encode()
    path = tmp_path / "bare.json"
    captured = write_evaluator_export(raw, path, expected_ids=["one"], **options())
    assert path.read_bytes() == raw
    assert captured == load_run(path, **options())
    assert captured["source_digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert captured["records"][0]["context"]["upstream_record"] == native["samples"][0]
    assert captured["records"][0]["metadata"] == {"slice": "test"}


def test_bare_langfuse_result_uses_explicit_sdk_identity(tmp_path):
    result = {
        "name": "experiment",
        "run_name": "run",
        "experiment_id": "experiment-id",
        "run_evaluations": [],
        "item_results": [
            {
                "item": {
                    "input": "Q",
                    "expected_output": "A",
                    "metadata": {"invarlock_id": "one", "slice": "test"},
                },
                "output": "A",
                "evaluations": [],
                "trace_id": "trace-id",
                "dataset_run_id": None,
            }
        ],
    }
    raw = json.dumps(result).encode()
    path = tmp_path / "langfuse-result.json"
    run = write_evaluator_export(
        raw, path, expected_ids=["one"], **options("langfuse", "4.14.1")
    )
    assert run == load_run(path, **options("langfuse", "4.14.1"))
    assert run["source"] == {"name": "langfuse", "version": "4.14.1"}
    assert run["source_digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    row = run["records"][0]
    assert (row["id"], row["input"], row["expected"], row["output"]) == (
        "one",
        "Q",
        "A",
        "A",
    )
    assert row["context"]["langfuse"]["item_result"] == result["item_results"][0]
    with pytest.raises(EvaluationRecordsError, match="run_id"):
        load_run(path, **{**options("langfuse", "4.14.1"), "run_id": "other"})


@pytest.mark.parametrize("missing", ["source", "run_id"])
def test_bare_native_import_requires_explicit_identity_before_publication(
    tmp_path, missing
):
    supplied = options()
    supplied.pop(missing)
    destination = tmp_path / "absent.json"
    with pytest.raises(
        EvaluationRecordsError, match="requires source name/version and run_id"
    ):
        write_evaluator_export(
            json.dumps(inspect_log()).encode(),
            destination,
            expected_ids=["one"],
            **supplied,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("metadata", False, "metadata"),
        ("metadata", [], "metadata"),
        ("scores", [1], "scores must be an object"),
        ("scores", {"": 1}, "nonempty names"),
        ("error", False, "task error"),
        ("error", {}, "task error"),
        ("error", "", "task error"),
        ("output", {"choices": {}}, "choices must be an array"),
    ],
)
def test_native_inspect_malformed_fields_cannot_publish(
    tmp_path, field, value, message
):
    native = inspect_log()
    native["samples"][0][field] = value
    path = tmp_path / "rejected.json"
    with pytest.raises(EvaluationRecordsError, match=message):
        write_evaluator_export(
            json.dumps(native).encode(), path, expected_ids=["one"], **options()
        )
    assert not path.exists()


def test_jsonl_primitive_line_is_not_silently_skipped(tmp_path):
    path = tmp_path / "primitive.jsonl"
    path.write_text("\n42\n")
    with pytest.raises(EvaluationRecordsError, match="array of record objects"):
        load_run(path, **{**options(), "adapter": "jsonl"})


def test_native_harness_null_output_can_carry_explicit_measured_likelihood(tmp_path):
    source = {"name": "lm-evaluation-harness", "version": "0.4.12"}
    facts = {
        "basis": "reference_continuation",
        "logprob_sum": -2.0,
        "token_count": 1,
        "utf8_byte_count": 1,
        "input_digest": digest("Q"),
        "reference_digest": digest("A"),
        "artifact_digest": ARTIFACT,
        "source": source,
        "configuration_digest": "sha256:" + "b" * 64,
        "tokenizer_digest": "sha256:" + "c" * 64,
    }
    native = [
        {
            "doc_id": "one",
            "doc": "Q",
            "target": "A",
            "arguments": [["Q", "A"]],
            "filtered_resps": [None],
            "metadata": {"invarlock_likelihood": facts},
        }
    ]
    path = tmp_path / "likelihood.json"
    path.write_text(json.dumps(native))
    run = load_run(path, **options(source["name"], source["version"]))
    row = run["records"][0]
    assert row["output"] is None and row["error"] is None
    assert row["likelihood"] == facts
    assert row["context"]["upstream_record"] == native[0]


@pytest.mark.parametrize("identifier", [False, "", " "])
def test_native_promptfoo_rejects_invalid_explicit_case_id(tmp_path, identifier):
    native = [
        {
            "testIdx": 0,
            "promptIdx": 0,
            "testCase": {
                "vars": {"question": "Q"},
                "metadata": {"invarlock_id": identifier},
            },
            "prompt": "Q",
            "response": {"output": "A"},
        }
    ]
    path = tmp_path / "promptfoo.json"
    path.write_text(json.dumps(native))
    with pytest.raises(
        EvaluationRecordsError, match="invarlock_id must be a nonempty string"
    ):
        load_run(path, **options("promptfoo", "0.121.19"))
