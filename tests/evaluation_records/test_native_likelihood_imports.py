"""Reference likelihoods must be explicit and agree with native observations."""

import copy
import json

import pytest

from invarlock.engine import (
    EvaluationRecordsError,
    digest,
    evaluator_input_capabilities,
    load_run,
)


def native_case(adapter):
    source = {"name": adapter, "version": "test-profile"}
    facts = {
        "basis": "reference_continuation",
        "logprob_sum": -2.0,
        "token_count": 1,
        "utf8_byte_count": 1,
        "input_digest": digest("Q"),
        "reference_digest": digest("A"),
        "artifact_digest": "sha256:" + "a" * 64,
        "source": source,
        "configuration_digest": "sha256:" + "b" * 64,
        "tokenizer_digest": "sha256:" + "c" * 64,
    }
    metadata = {"invarlock_likelihood": facts}
    if adapter == "inspect-json":
        row = {
            "id": "one",
            "input": "Q",
            "target": "A",
            "output": {"choices": []},
            "metadata": metadata,
        }
        payload = {"version": 2, "status": "success", "samples": [row]}
    elif adapter == "lm-eval-samples":
        row = {
            "doc_id": "one",
            "doc": "Q",
            "target": "A",
            "arguments": [["Q", "A"]],
            "filtered_resps": [[-2.0, False]],
            "metadata": metadata,
        }
        payload = [row]
    else:
        metadata["invarlock_expected"] = "A"
        row = {
            "testIdx": "one",
            "promptIdx": 0,
            "testCase": {"vars": "Q", "metadata": metadata},
            "prompt": {"raw": "Q"},
            "response": {"output": None},
        }
        payload = [row]
    return payload, row, metadata, source


def imported(tmp_path, adapter, payload, source):
    raw = (
        json.dumps(payload)
        if adapter == "inspect-json"
        else "\n".join(map(json.dumps, payload))
    )
    path = tmp_path / "capture.json"
    path.write_text(raw)
    return load_run(
        path,
        adapter=adapter,
        source=source,
        run_id="run",
        artifact_digest="sha256:" + "a" * 64,
    )


@pytest.mark.parametrize(
    "adapter", ["inspect-json", "lm-eval-samples", "promptfoo-jsonl"]
)
def test_native_likelihood_facts_and_original_fields_survive(tmp_path, adapter):
    payload, row, metadata, source = native_case(adapter)
    original = copy.deepcopy(row)
    run = imported(tmp_path, adapter, payload, source)
    captured = run["records"][0]
    assert captured["likelihood"] == metadata["invarlock_likelihood"]
    assert captured["context"]["upstream_record"] == original
    assert captured["output"] is None
    assert (
        evaluator_input_capabilities(run)["normalized_nll_per_utf8_byte"][
            "usable_count"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "adapter", ["inspect-json", "lm-eval-samples", "promptfoo-jsonl"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("source", {"name": "other", "version": "1"}),
        ("input_digest", "sha256:" + "0" * 64),
        ("utf8_byte_count", 2),
        ("logprob_sum", float("nan")),
        ("token_count", True),
    ],
)
def test_mismatched_likelihood_facts_are_rejected(tmp_path, adapter, field, value):
    payload, _, metadata, source = native_case(adapter)
    metadata["invarlock_likelihood"][field] = value
    with pytest.raises(EvaluationRecordsError):
        imported(tmp_path, adapter, payload, source)


@pytest.mark.parametrize(
    "response,arguments",
    [
        ([[-3.0, False]], [["Q", "A"]]),
        ([[-2.0, 1]], [["Q", "A"]]),
        ([[-2.0, False]], [["Q", "B"]]),
        ([[-2.0, False]], []),
        ([[-2.0, False], [-2.0, False]], [["Q", "A"]]),
    ],
)
def test_harness_native_result_must_match_declared_likelihood(
    tmp_path, response, arguments
):
    payload, row, _, source = native_case("lm-eval-samples")
    row.update(filtered_resps=response, arguments=arguments)
    with pytest.raises(EvaluationRecordsError):
        imported(tmp_path, "lm-eval-samples", payload, source)


def test_inspect_likelihood_does_not_allow_multiple_answers(tmp_path):
    payload, row, _, source = native_case("inspect-json")
    row["output"]["choices"] = [{"message": {"content": "A"}}] * 2
    with pytest.raises(EvaluationRecordsError, match="one completion"):
        imported(tmp_path, "inspect-json", payload, source)
