"""Explicit capture and text projection preserve and replay original case inputs."""

import copy
import json
from pathlib import Path

import pytest

from invarlock.evaluation_comparison.comparison import _check_run
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.adapters import _parse_run_bytes, load_run
from invarlock.evaluator_capture import (
    capture_evaluator_run,
    evaluator_input_capabilities,
    verify_input_pair,
    verify_input_projection,
)

OPTIONS = {
    "source": {"name": "evaluator", "version": "1.2.3"},
    "run_id": "baseline",
    "artifact_digest": "sha256:" + "a" * 64,
}


@pytest.mark.parametrize("pointer", ["/input", "/input/missing", "/input/0", "/output"])
def test_projection_refuses_ambiguous_missing_and_outcome_inputs(pointer):
    raw = b'{"id":"one","input":{"messages":["question"]},"expected":"yes","output":"yes"}'
    with pytest.raises(EvaluationRecordsError, match="projection"):
        _parse_run_bytes(
            raw,
            adapter="jsonl",
            input_projection={"kind": "json-pointer", "pointer": pointer},
            **OPTIONS,
        )


def test_projection_replay_rejects_forged_text():
    from invarlock.evaluator_capture import (
        capture_evaluator_run,
        verify_input_projection,
    )

    run = capture_evaluator_run(
        [
            {
                "id": "one",
                "input": {"text": "question"},
                "expected": "yes",
                "output": "yes",
            }
        ],
        input_projection={"kind": "json-pointer", "pointer": "/input/text"},
        **OPTIONS,
    )
    row = copy.deepcopy(run["records"][0])
    row["input"] = "forged"
    with pytest.raises(EvaluationRecordsError, match="projection"):
        verify_input_projection(row)


def case(**updates):
    return {
        "id": "one",
        "input": "question",
        "expected": "yes",
        "output": "yes",
        **updates,
    }


def projected(row=None, pointer="/input"):
    return capture_evaluator_run(
        [case() if row is None else row],
        input_projection={"kind": "json-pointer", "pointer": pointer},
        **OPTIONS,
    )


@pytest.mark.parametrize(
    "filename,adapter,pointer",
    [
        ("inspect-0.3.254.json", "inspect-json", "/input"),
        ("lm-eval-0.4.12.jsonl", "lm-eval-samples", "/context/arguments/0/0"),
        ("promptfoo-0.121.19.jsonl", "promptfoo-jsonl", "/context/prompt"),
    ],
)
def test_real_export_projection_preserves_sources_and_default_outputs(
    filename, adapter, pointer
):
    path = Path(__file__).parent / "fixtures" / filename
    original = load_run(path, adapter=adapter, **OPTIONS)
    run = load_run(
        path,
        adapter=adapter,
        input_projection={"kind": "json-pointer", "pointer": pointer},
        **OPTIONS,
    )
    assert len(run["records"]) == 40
    assert run["source_digest"] == original["source_digest"]
    for old, row in zip(original["records"], run["records"], strict=True):
        assert row["output"] == old["output"]
        assert row["context"]["input_projection"]["source"] == {
            "input": old["input"],
            "context": old["context"],
        }
        assert all(
            row["context"][key] == value for key, value in old["context"].items()
        )
        assert isinstance(row["input"], str)
        verify_input_projection(row)
    assert evaluator_input_capabilities(run)["judge"]["usable_count"] == 40
    assert _parse_run_bytes(json.dumps(run).encode()) == run


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"kind": "template", "pointer": "/input"},
        {"kind": "json-pointer", "pointer": "/input", "template": "{input}"},
        {"kind": "json-pointer", "pointer": None},
        {"kind": "json-pointer", "pointer": ""},
        {"kind": "json-pointer", "pointer": "#/input"},
        {"kind": "json-pointer", "pointer": "/input/~"},
        {"kind": "json-pointer", "pointer": "/input/~2"},
        {"kind": "json-pointer", "pointer": "/" + "a" * 4096},
        {"kind": "json-pointer", "pointer": "/expected"},
        "lambda row: str(row['input'])",
    ],
)
def test_projection_configuration_is_closed_and_declarative(configuration):
    with pytest.raises(EvaluationRecordsError, match="projection"):
        capture_evaluator_run([case()], input_projection=configuration, **OPTIONS)


@pytest.mark.parametrize(
    "pointer",
    [
        "/input/00",
        "/input/-1",
        "/input/-",
        "/input/2",
        "/input/12",
        "/input/١",
        "/input/0/nope",
    ],
)
def test_array_projection_rejects_noncanonical_or_missing_indices(pointer):
    with pytest.raises(EvaluationRecordsError, match="projection"):
        projected(case(input=["first", "second"]), pointer)


@pytest.mark.parametrize("context", [None, ["facts"], "context", {}])
def test_projection_preserves_nonobject_context(context):
    run = projected(case(context=context))
    row = run["records"][0]
    verify_input_projection(row)
    assert row["context"]["input_projection"]["source"]["context"] == context


def test_structured_messages_and_escaped_object_keys_select_only_explicit_text():
    original = case(
        input={
            "a/b": {
                "~": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ]
            }
        }
    )
    run = projected(original, "/input/a~1b/~0/1/content")
    assert run["records"][0]["input"] == "user"
    verify_input_projection(run["records"][0])
    original["input"]["a/b"]["~"][1]["content"] = "mutated"
    assert (
        run["records"][0]["context"]["input_projection"]["source"]["input"]["a/b"]["~"][
            1
        ]["content"]
        == "user"
    )
    assert (
        projected(case(input={"": "empty-key"}), "/input/")["records"][0]["input"]
        == "empty-key"
    )
    assert projected(case(input=""))["records"][0]["input"] == ""


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row["context"].update(input_projection=None),
        lambda row: row["context"]["input_projection"].update(extra=True),
        lambda row: row["context"]["input_projection"].update(source=None),
        lambda row: row["context"]["input_projection"]["source"].update(extra=True),
        lambda row: row["context"]["input_projection"].update(
            configuration_digest="sha256:" + "b" * 64
        ),
        lambda row: row["context"]["input_projection"].update(
            source_digest="sha256:" + "b" * 64
        ),
        lambda row: row["context"].update(prompt="forged"),
    ],
)
def test_forged_retained_projection_rejected_by_independent_run_validation(change):
    run = projected(case(context={"prompt": "question"}))
    change(run["records"][0])
    with pytest.raises(EvaluationRecordsError, match="projection"):
        _check_run(run)


def test_reserved_projection_cannot_be_nested_or_projected_twice():
    run = projected()
    with pytest.raises(EvaluationRecordsError, match="reserved"):
        projected(run["records"][0])
    row = copy.deepcopy(run["records"][0])
    binding = row["context"]["input_projection"]
    binding["source"]["context"] = {"input_projection": {}}
    binding["source_digest"] = digest(binding["source"])
    with pytest.raises(EvaluationRecordsError, match="context"):
        verify_input_projection(row)


@pytest.mark.parametrize("value", [None, True, 7, {}, ["text"]])
def test_no_implicit_nontext_conversion(value):
    with pytest.raises(EvaluationRecordsError, match="text string"):
        projected(case(input=value))


def test_projection_pairing_retains_source_identity_and_mapping():
    row = case(input={"question": "question", "id": 1}, context={"prompt": "question"})
    left = projected(row, "/context/prompt")["records"][0]
    right = projected({**row, "output": "no"}, "/context/prompt")["records"][0]
    verify_input_pair(left, right)
    assert left["context"]["input_projection"] == right["context"]["input_projection"]
    for changed in [
        projected(
            {**row, "input": {"question": "question", "id": 2}}, "/context/prompt"
        )["records"][0],
        projected(row, "/input/question")["records"][0],
        capture_evaluator_run([case()], **OPTIONS)["records"][0],
    ]:
        with pytest.raises(EvaluationRecordsError, match="original input changed"):
            verify_input_pair(left, changed)
        with pytest.raises(EvaluationRecordsError, match="original input changed"):
            verify_input_pair(changed, left)
    verify_input_pair(case(context=[]), case(context="anything"))


@pytest.mark.parametrize("records", [[], {}, [1], [{"score": 0.5}], [case(), case()]])
def test_capture_rejects_aggregate_invented_or_duplicate_cases(records):
    with pytest.raises(EvaluationRecordsError):
        capture_evaluator_run(records, **OPTIONS)


@pytest.mark.parametrize(
    "updates",
    [
        {"source": {}},
        {"source": None},
        {"run_id": ""},
        {"artifact_digest": "latest"},
        {"source": {"name": "evaluator", "version": ""}},
    ],
)
def test_capture_requires_valid_explicit_identities(updates):
    with pytest.raises(EvaluationRecordsError):
        capture_evaluator_run([case()], **{**OPTIONS, **updates})


def test_capture_bound_and_detached_records(monkeypatch):
    import invarlock.evaluator_capture as capture

    monkeypatch.setattr(capture, "MAX_RECORDS", 1)
    with pytest.raises(EvaluationRecordsError, match="bounded"):
        capture_evaluator_run([case(), case(id="two")], **OPTIONS)
    row = case(context={"temperature": 0})
    run = capture_evaluator_run([row], **OPTIONS)
    row["context"]["temperature"] = 1
    assert run["records"][0]["context"] == {"temperature": 0}
    assert run["score_provenance"] == {}
    assert "runtime" not in run


def test_capabilities_report_facts_and_error_rows_without_runtime_claims():
    run = capture_evaluator_run(
        [
            case(),
            case(id="structured", input={"messages": ["q"]}),
            case(id="no-reference", expected=None),
            case(id="bad-reference", expected=1),
            case(id="error", error="upstream_error"),
            case(id="no-output", output=None),
        ],
        **OPTIONS,
    )
    result = evaluator_input_capabilities(run)
    assert result["exact_match"]["usable_count"] == 2
    assert result["judge"]["usable_count"] == 2
    assert result["judge"]["unavailable_ids"] == [
        "structured",
        "bad-reference",
        "error",
        "no-output",
    ]
    assert result["normalized_nll_per_utf8_byte"]["usable_count"] == 0
    assert (
        "bound_reference_likelihood"
        in result["normalized_nll_per_utf8_byte"]["required_facts"]
    )


def test_canonical_import_rejects_projection_override():
    import json

    raw = json.dumps(capture_evaluator_run([case()], **OPTIONS)).encode()
    with pytest.raises(EvaluationRecordsError, match="cannot be overridden"):
        _parse_run_bytes(
            raw, input_projection={"kind": "json-pointer", "pointer": "/input"}
        )


def test_capture_preserves_bound_likelihood_and_exposes_nll_availability():
    row = case(output=None)
    facts = {
        "basis": "reference_continuation",
        "logprob_sum": -3.0,
        "token_count": 1,
        "utf8_byte_count": 3,
        "input_digest": digest(row["input"]),
        "reference_digest": digest(row["expected"]),
        "artifact_digest": OPTIONS["artifact_digest"],
        "configuration_digest": digest("configuration"),
        "tokenizer_digest": digest("tokenizer"),
        "source": OPTIONS["source"],
    }
    row["likelihood"] = facts
    run = capture_evaluator_run([row], **OPTIONS)
    assert run["records"][0]["likelihood"] == facts
    capabilities = evaluator_input_capabilities(run)
    assert capabilities["normalized_nll_per_utf8_byte"]["usable_count"] == 1
    assert capabilities["exact_match"]["usable_count"] == 0
    assert capabilities["judge"]["usable_count"] == 0
    row["likelihood"]["logprob_sum"] = -7
    assert run["records"][0]["likelihood"]["logprob_sum"] == -3
    row["input"] = {"question": "question"}
    row["output"] = "yes"
    row["likelihood"]["input_digest"] = digest(row["input"])
    projected_run = projected(row, "/input/question")
    assert projected_run["records"][0]["likelihood"] == row["likelihood"]
    assert all(
        item["usable_count"] == 1
        for item in evaluator_input_capabilities(projected_run).values()
    )
    assert load_run_bytes(projected_run) == projected_run


def load_run_bytes(run):
    return _parse_run_bytes(json.dumps(run).encode())


@pytest.mark.parametrize("change", ["original", "projected", "input-digest"])
def test_projected_likelihood_rejects_tampered_input_bindings(change):
    row = case(input={"question": "question", "fact": "original"})
    row["likelihood"] = {
        "basis": "reference_continuation",
        "logprob_sum": -3,
        "token_count": 1,
        "utf8_byte_count": 3,
        "input_digest": digest(row["input"]),
        "reference_digest": digest(row["expected"]),
        "artifact_digest": OPTIONS["artifact_digest"],
        "configuration_digest": digest("configuration"),
        "tokenizer_digest": digest("tokenizer"),
        "source": OPTIONS["source"],
    }
    run = projected(row, "/input/question")
    captured = run["records"][0]
    if change == "original":
        binding = captured["context"]["input_projection"]
        binding["source"]["input"]["fact"] = "changed"
        binding["source_digest"] = digest(binding["source"])
    elif change == "projected":
        captured["input"] = "changed"
    else:
        captured["likelihood"]["input_digest"] = digest(captured["input"])
    with pytest.raises(EvaluationRecordsError, match="projection|input_digest"):
        load_run_bytes(run)


def test_capture_score_authority_requires_explicit_provenance_object():
    with pytest.raises(EvaluationRecordsError, match="score provenance"):
        capture_evaluator_run([case()], score_provenance=7, **OPTIONS)
    provenance = {
        "quality": {
            "kind": "external_metric",
            "source": "upstream",
            "version": "1",
            "unit": "score",
            "rubric_digest": None,
        }
    }
    run = capture_evaluator_run(
        [case(scores={"quality": 0.7})], score_provenance=provenance, **OPTIONS
    )
    assert run["score_provenance"] == provenance
    provenance["quality"]["version"] = "2"
    assert run["score_provenance"]["quality"]["version"] == "1"
