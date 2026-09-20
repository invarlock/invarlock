"""Native SDK shape and adversarial export coverage, with no inference calls."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.scalar_integrations import EVALUATORS, export_records


def native_entry(evaluator: str) -> dict:
    entry = {"id": "case-1", "metadata": {"slice": "baseline", "nested": {"fact": 1}}}
    payloads = {
        "deepeval": {
            "test_case": {
                "input": "Question?",
                "actual_output": "Answer",
                "expected_output": "Answer",
                "context": ["source"],
            },
            "metric_result": {"score": 1.0, "reason": "match"},
        },
        "ragas": {
            "sample": {
                "user_input": "Question?",
                "response": "Answer",
                "reference": "Answer",
                "retrieved_contexts": ["source"],
            },
            "metric_result": {"value": 1.0, "reason": None, "traces": None},
        },
        "lighteval": {
            "doc": {
                "query": "Question?",
                "choices": ["Wrong", "Answer"],
                "gold_index": 1,
            },
            "model_response": {"text": ["Answer"], "logprobs": [-0.125]},
            "metric_result": 1.0,
        },
        "hugging-face-evaluate": {
            "input": "Question?",
            "predictions": ["Answer"],
            "references": ["Answer"],
            "metric_result": {"exact_match": 1.0},
        },
        "autoevals": {
            "input": "Question?",
            "output": "Answer",
            "expected": "Answer",
            "metric_result": {"name": "ExactMatch", "score": 1.0, "metadata": {}},
        },
        "openevals": {
            "inputs": "Question?",
            "outputs": "Answer",
            "reference_outputs": "Answer",
            "metric_result": {"key": "exact_match", "score": True},
        },
        "arize-phoenix-evals": {
            "record": {"input": "Question?", "output": "Answer", "expected": "Answer"},
            "metric_result": {"score": 1.0, "name": "exact_match", "kind": "code"},
        },
        "opik": {
            "dataset_item": {
                "input": "Question?",
                "output": "Answer",
                "reference": "Answer",
            },
            "metric_result": {
                "name": "equals_metric",
                "value": 1.0,
                "scoring_failed": False,
            },
        },
    }
    return {**entry, **payloads[evaluator]}


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_native_case_mapping_and_offline_roundtrip(evaluator):
    entry = native_entry(evaluator)
    untouched = copy.deepcopy(entry)
    row = export_records(evaluator, [entry])[0]
    assert (row["id"], row["input"], row["expected"], row["output"]) == (
        "case-1",
        "Question?",
        "Answer",
        "Answer",
    )
    assert row["scores"] == {}
    assert row["metadata"] == {"slice": "baseline"}
    assert "likelihood" not in row
    native = json.loads(json.dumps(row["context"]["upstream_record"], allow_nan=False))
    assert export_records(evaluator, [native]) == [row]
    assert entry == untouched
    native["metadata"]["nested"]["fact"] = 99
    assert entry == untouched


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_native_likelihood_is_explicit_and_retained_for_central_validation(evaluator):
    entry = native_entry(evaluator)
    entry["metadata"]["invarlock_likelihood"] = {"deliberately": "not validated here"}
    row = export_records(evaluator, [entry])[0]
    assert row["likelihood"] == entry["metadata"]["invarlock_likelihood"]
    assert row["scores"] == {}


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize(
    "mutation", ["no_id", "bad_id", "nan", "object", "metadata", "error"]
)
def test_invalid_wrappers_fail_closed(evaluator, mutation):
    entry = native_entry(evaluator)
    if mutation == "no_id":
        del entry["id"]
    elif mutation == "bad_id":
        entry["id"] = 5
    elif mutation == "nan":
        entry["metadata"]["invalid"] = float("nan")
    elif mutation == "object":
        entry["metadata"]["invalid"] = object()
    elif mutation == "metadata":
        entry["metadata"] = []
    else:
        entry["error"] = {}
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, [entry])


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_missing_outputs_and_aggregate_results_rejected(evaluator):
    entry = native_entry(evaluator)
    paths = {
        "deepeval": ("test_case", "actual_output"),
        "ragas": ("sample", "response"),
        "lighteval": ("model_response", "text"),
        "hugging-face-evaluate": (None, "predictions"),
        "autoevals": (None, "output"),
        "openevals": (None, "outputs"),
        "arize-phoenix-evals": ("record", "output"),
        "opik": ("dataset_item", "output"),
    }
    parent, key = paths[evaluator]
    del (entry[parent] if parent else entry)[key]
    with pytest.raises(EvaluationRecordsError, match="missing required"):
        export_records(evaluator, [entry])
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, {"aggregate": 1.0})
    with pytest.raises(EvaluationRecordsError, match="unique"):
        export_records(evaluator, [native_entry(evaluator), native_entry(evaluator)])


@pytest.mark.parametrize("indices", [-1, True, 2, [], [0, 0], [0, "1"]])
def test_lighteval_invalid_gold_indices(indices):
    entry = native_entry("lighteval")
    entry["doc"]["gold_index"] = indices
    with pytest.raises(EvaluationRecordsError, match="gold_index"):
        export_records("lighteval", [entry])


@pytest.mark.parametrize("indices", [None, False, True, -1, 1, 0.0, "0", [], [0]])
def test_lighteval_reference_free_profile_rejects_malformed_gold_indices(indices):
    entry = native_entry("lighteval")
    entry["doc"].update(choices=[], gold_index=indices)
    del entry["metric_result"]
    with pytest.raises(EvaluationRecordsError, match="gold_index"):
        export_records("lighteval", [entry])


def test_lighteval_multiple_golds_and_raw_text_preserved():
    entry = native_entry("lighteval")
    entry["doc"]["gold_index"] = [0, 1]
    entry["model_response"]["text_post_processed"] = ["Altered"]
    row = export_records("lighteval", [entry])[0]
    assert row["expected"] == ["Wrong", "Answer"]
    assert row["output"] == "Answer"
    assert "likelihood" not in row


@pytest.mark.parametrize("evaluator", ["lighteval", "hugging-face-evaluate", "ragas"])
def test_multiple_generations_rejected(evaluator):
    entry = native_entry(evaluator)
    if evaluator == "lighteval":
        entry["model_response"]["text"].append("extra")
    elif evaluator == "hugging-face-evaluate":
        entry["predictions"].append("extra")
    else:
        entry["sample"]["multi_responses"] = ["Answer", "extra"]
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, [entry])


def test_no_implicit_sdk_serialization():
    class Unsupported:
        def __str__(self):
            raise AssertionError("must not stringify")

        def model_dump(self):
            raise AssertionError("must not dump")

    entry = native_entry("deepeval")
    entry["test_case"] = SimpleNamespace(**entry["test_case"])
    entry["metric_result"] = SimpleNamespace(score=1.0, reason="match")
    assert export_records("deepeval", [entry])[0]["output"] == "Answer"
    entry["test_case"].context = [Unsupported()]
    with pytest.raises(EvaluationRecordsError, match="explicitly map"):
        export_records("deepeval", [entry])


def test_bounds_cycles_and_unknown_evaluator(monkeypatch):
    with pytest.raises(EvaluationRecordsError, match="unsupported"):
        export_records("missing", [])
    with pytest.raises(EvaluationRecordsError):
        export_records("ragas", [])
    monkeypatch.setattr(
        "invarlock.evaluation_records.scalar_integrations.MAX_RECORDS", 1
    )
    with pytest.raises(EvaluationRecordsError, match="bounded"):
        export_records("ragas", [native_entry("ragas")] * 2)
    entry = native_entry("ragas")
    entry["metadata"]["loop"] = entry
    with pytest.raises(EvaluationRecordsError, match="nesting"):
        export_records("ragas", [entry])


# Optional SDK tests run in the matrix's isolated pinned environments. Imports
# and deterministic local metrics only; no model, network, service or API calls.
_VERSIONS = {
    "deepeval": "4.1.3",
    "ragas": "0.4.3",
    "lighteval": "0.13.0",
    "evaluate": "0.4.6",
    "autoevals": "0.3.0",
    "openevals": "0.2.0",
    "arize-phoenix-evals": "3.3.0",
    "opik": "2.2.7",
}


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_pinned_sdk_objects(evaluator, monkeypatch, tmp_path):
    package = "evaluate" if evaluator == "hugging-face-evaluate" else evaluator
    required = os.environ.get("INVARLOCK_REQUIRE_EVALUATOR_SDK") in ("1", evaluator)
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        if required:
            pytest.fail(f"required SDK {package}=={_VERSIONS[package]} is missing")
        pytest.skip(f"optional {package} SDK not installed")
    if installed != _VERSIONS[package]:
        if required:
            pytest.fail(
                f"required SDK {package}=={_VERSIONS[package]}, found {installed}"
            )
        pytest.skip(f"requires pinned {package}=={_VERSIONS[package]}")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    monkeypatch.setenv("OPIK_TRACK_DISABLE", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    entry = native_entry(evaluator)
    if evaluator == "deepeval":
        from deepeval.metrics import ExactMatchMetric
        from deepeval.test_case import LLMTestCase

        entry["test_case"] = LLMTestCase(**entry["test_case"])
        entry["metric_result"] = ExactMatchMetric()
        entry["metric_result"].measure(entry["test_case"])
        _documented_deepeval_raw_capture(entry["test_case"], tmp_path, monkeypatch)
    elif evaluator == "ragas":
        import asyncio

        from ragas import SingleTurnSample
        from ragas.metrics.collections import ExactMatch

        entry["sample"] = SingleTurnSample(**entry["sample"])
        entry["metric_result"] = asyncio.run(
            ExactMatch().ascore(reference="Answer", response="Answer")
        )
    elif evaluator == "lighteval":
        from lighteval.metrics.metrics_sample import ExactMatches
        from lighteval.models.model_output import ModelResponse
        from lighteval.tasks.requests import Doc

        entry["doc"] = Doc(**entry["doc"])
        entry["model_response"] = ModelResponse(**entry["model_response"])
        entry["metric_result"] = ExactMatches().compute(
            doc=entry["doc"], model_response=entry["model_response"]
        )
        # The real SDK documents empty choices for unreferenced generation;
        # capturing it requires neither an invented gold answer nor a metric.
        reference_free = {
            "id": "generation-only",
            "doc": Doc(query="Write a greeting", choices=[], gold_index=0),
            "model_response": ModelResponse(text=["Hello"]),
        }
        generated = export_records(evaluator, [reference_free])[0]
        assert generated["input"] == "Write a greeting"
        assert generated["output"] == "Hello"
        assert generated["expected"] is None and generated["scores"] == {}
        retained = generated["context"]["upstream_record"]
        assert retained["doc"]["choices"] == []
        assert type(retained["doc"]["gold_index"]) is int
        assert retained["doc"]["gold_index"] == 0
        assert export_records(evaluator, [retained]) == [generated]
    elif evaluator == "autoevals":
        from autoevals import ExactMatch

        entry["metric_result"] = ExactMatch()(output="Answer", expected="Answer")
    elif evaluator == "openevals":
        from openevals.exact import exact_match

        entry["metric_result"] = exact_match(
            outputs="Answer", reference_outputs="Answer"
        )
    elif evaluator == "arize-phoenix-evals":
        from phoenix.evals.metrics import exact_match

        entry["metric_result"] = exact_match("Answer", "Answer")
    elif evaluator == "opik":
        from opik.evaluation.metrics import Equals

        entry["metric_result"] = Equals().score(output="Answer", reference="Answer")
    else:
        # No evaluate.load(): that may fetch remote metric code. The installed
        # library accepts compute() dictionaries; construct an offline metric.
        import datasets
        import evaluate

        class LocalExact(evaluate.Metric):
            def _info(self):
                return evaluate.MetricInfo(
                    description="local exact",
                    citation="",
                    inputs_description="",
                    features=datasets.Features(
                        {
                            "predictions": datasets.Value("string"),
                            "references": datasets.Value("string"),
                        }
                    ),
                )

            def _compute(self, predictions, references):
                return {
                    "exact_match": sum(
                        p == r for p, r in zip(predictions, references, strict=True)
                    )
                    / len(predictions)
                }

        entry["metric_result"] = LocalExact(
            cache_dir=str(tmp_path / "metric-cache")
        ).compute(predictions=entry["predictions"], references=entry["references"])
    row = export_records(evaluator, [entry])[0]
    assert row["output"] == row["expected"] == "Answer"
    assert row["scores"] == {}
    assert export_records(evaluator, [row["context"]["upstream_record"]]) == [row]


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_explicit_numeric_observations(evaluator):
    entry = native_entry(evaluator)
    entry["metadata"]["invarlock_scores"] = {"quality": 0.75}
    row = export_records(evaluator, [entry])[0]
    assert row["scores"] == {"quality": 0.75}
    assert export_records(evaluator, [row["context"]["upstream_record"]]) == [row]


@pytest.mark.parametrize(
    "scores", [[], {"": 1}, {"quality": True}, {"quality": "1"}, {"quality": None}]
)
def test_explicit_scores_fail_closed(scores):
    entry = native_entry("autoevals")
    entry["metadata"]["invarlock_scores"] = scores
    with pytest.raises(EvaluationRecordsError, match="invarlock_scores"):
        export_records("autoevals", [entry])


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_likelihood_only_case_does_not_invent_generated_text(evaluator):
    entry = native_entry(evaluator)
    containers = {
        "deepeval": ("test_case", "actual_output"),
        "ragas": ("sample", "response"),
        "lighteval": ("model_response", "text"),
        "hugging-face-evaluate": (None, "predictions"),
        "autoevals": (None, "output"),
        "openevals": (None, "outputs"),
        "arize-phoenix-evals": ("record", "output"),
        "opik": ("dataset_item", "output"),
    }
    parent, key = containers[evaluator]
    (entry[parent] if parent else entry)[key] = (
        [None] if key in ("text", "predictions") else None
    )
    with pytest.raises(EvaluationRecordsError, match="output"):
        export_records(evaluator, [entry])
    entry["metadata"]["invarlock_likelihood"] = {
        "typed": "central capture validates this"
    }
    row = export_records(evaluator, [entry])[0]
    assert row["output"] is None and row["error"] is None


@pytest.mark.parametrize("evaluator", ["openevals", "arize-phoenix-evals", "opik"])
def test_multiple_native_metrics_are_observations(evaluator):
    entry = native_entry(evaluator)
    entry["metric_result"] = [entry["metric_result"], entry["metric_result"]]
    assert export_records(evaluator, [entry])[0]["scores"] == {}
    entry["metric_result"] = [entry["metric_result"]]
    with pytest.raises(EvaluationRecordsError, match="flat"):
        export_records(evaluator, [entry])


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_capture_without_upstream_grading(evaluator):
    from invarlock.evaluation_records.scalar_integrations import serialize_results

    entry = native_entry(evaluator)
    del entry["metric_result"]
    row = export_records(evaluator, [entry])[0]
    assert row["output"] == row["expected"] == "Answer"
    assert row["scores"] == {}
    assert "metric_result" not in row["context"]["upstream_record"]
    assert serialize_results(evaluator, [entry]) == [entry]
    assert export_records(evaluator, serialize_results(evaluator, [entry])) == [row]


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_reference_free_judge_capture(evaluator, tmp_path):
    from invarlock.engine import (
        evaluator_input_capabilities,
        export_evaluator_result,
        load_run,
    )

    entry = native_entry(evaluator)
    paths = {
        "deepeval": ("test_case", "expected_output"),
        "ragas": ("sample", "reference"),
        "hugging-face-evaluate": (None, "references"),
        "autoevals": (None, "expected"),
        "openevals": (None, "reference_outputs"),
        "arize-phoenix-evals": ("record", "expected"),
        "opik": ("dataset_item", "reference"),
    }
    if evaluator == "lighteval":
        entry["doc"].update(choices=[], gold_index=0)
    else:
        container, field = paths[evaluator]
        del (entry[container] if container else entry)[field]
    del entry["metric_result"]
    records = export_records(evaluator, [entry])
    assert records[0]["expected"] is None
    native_path, envelope_path = tmp_path / "native.json", tmp_path / "export.json"
    native_path.write_text(json.dumps([entry], allow_nan=False))
    run = export_evaluator_result(
        evaluator,
        [entry],
        envelope_path,
        expected_ids=[entry["id"]],
        source_version="test",
        run_id="reference-free",
        artifact_digest="sha256:" + "a" * 64,
    )
    capabilities = evaluator_input_capabilities(run)
    assert capabilities["judge"]["usable_count"] == 1
    assert capabilities["exact_match"]["usable_count"] == 0
    assert capabilities["normalized_nll_per_utf8_byte"]["usable_count"] == 0
    for adapter, path in (
        ("evaluator-native-json", native_path),
        ("evaluator-json", envelope_path),
    ):
        imported = load_run(
            path,
            adapter=adapter,
            source={"name": evaluator, "version": "test"},
            run_id="reference-free",
            artifact_digest="sha256:" + "a" * 64,
        )
        assert imported["records"] == run["records"]
        assert evaluator_input_capabilities(imported) == capabilities
        assert imported["records"][0]["metadata"] == {"slice": "baseline"}


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("output", [{"answer": "structured"}, ["structured", 7], 42])
def test_structured_outputs_are_preserved_without_text_coercion(evaluator, output):
    from invarlock.evaluator_capture import (
        capture_evaluator_run,
        evaluator_input_capabilities,
    )

    entry = native_entry(evaluator)
    paths = {
        "deepeval": ("test_case", "actual_output"),
        "ragas": ("sample", "response"),
        "lighteval": ("model_response", "text"),
        "hugging-face-evaluate": (None, "predictions"),
        "autoevals": (None, "output"),
        "openevals": (None, "outputs"),
        "arize-phoenix-evals": ("record", "output"),
        "opik": ("dataset_item", "output"),
    }
    container, field = paths[evaluator]
    (entry[container] if container else entry)[field] = (
        [output] if field in ("text", "predictions") else output
    )
    del entry["metric_result"]
    rows = export_records(evaluator, [entry])
    assert rows[0]["output"] == output
    assert export_records(evaluator, [rows[0]["context"]["upstream_record"]]) == rows
    run = capture_evaluator_run(
        rows,
        source={"name": evaluator, "version": "test"},
        run_id="structured",
        artifact_digest="sha256:" + "a" * 64,
    )
    capabilities = evaluator_input_capabilities(run)
    assert capabilities["judge"]["usable_count"] == 0
    assert capabilities["exact_match"]["usable_count"] == 0


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_mixed_failures_keep_projection_bindings_through_both_import_routes(
    evaluator, tmp_path
):
    from invarlock.engine import (
        evaluator_input_capabilities,
        export_evaluator_result,
        load_run,
    )
    from invarlock.evaluator_capture import verify_input_projection

    input_container, input_field, output_container, output_field = {
        "deepeval": ("test_case", "input", "test_case", "actual_output"),
        "ragas": ("sample", "user_input", "sample", "response"),
        "lighteval": ("doc", "query", "model_response", "text"),
        "hugging-face-evaluate": (None, "input", None, "predictions"),
        "autoevals": (None, "input", None, "output"),
        "openevals": (None, "inputs", None, "outputs"),
        "arize-phoenix-evals": ("record", "input", "record", "output"),
        "opik": ("dataset_item", "input", "dataset_item", "output"),
    }[evaluator]
    entries = []
    for ident, output, error in (
        ("complete", "Answer", None),
        ("failed", None, "Generation failed"),
        ("partial", "Partial answer", "Generation interrupted"),
    ):
        entry = native_entry(evaluator)
        entry.update(id=ident, error=error)
        del entry["metric_result"]
        (entry[input_container] if input_container else entry)[input_field] = {
            "text": "Question?",
            "original_flag": True,
        }
        (entry[output_container] if output_container else entry)[output_field] = (
            [output] if output_field in ("text", "predictions") else output
        )
        entries.append(entry)
    original = copy.deepcopy(entries)
    options = {
        "run_id": "mixed",
        "artifact_digest": "sha256:" + "a" * 64,
        "input_projection": {"kind": "json-pointer", "pointer": "/input/text"},
    }
    envelope = tmp_path / "export.json"
    exported = export_evaluator_result(
        evaluator,
        entries,
        envelope,
        expected_ids=["complete", "failed", "partial"],
        source_version="test",
        **options,
    )
    raw = tmp_path / "native.json"
    raw.write_text(json.dumps(entries))
    assert entries == original
    for adapter, path in (
        ("evaluator-json", envelope),
        ("evaluator-native-json", raw),
    ):
        imported = load_run(
            path,
            adapter=adapter,
            source={"name": evaluator, "version": "test"},
            **options,
        )
        assert imported["records"] == exported["records"]
        records = imported["records"]
        assert [row["id"] for row in records] == ["complete", "failed", "partial"]
        assert [row["output"] for row in records] == ["Answer", None, "Partial answer"]
        assert [row["error"] for row in records] == [
            None,
            "Generation failed",
            "Generation interrupted",
        ]
        for row, native in zip(records, entries, strict=True):
            assert row["input"] == "Question?"
            assert row["context"]["upstream_record"] == native
            verify_input_projection(row)
        capabilities = evaluator_input_capabilities(imported)
        for scorer in ("exact_match", "judge"):
            assert capabilities[scorer]["usable_count"] == 1
            assert capabilities[scorer]["unavailable_ids"] == ["failed", "partial"]
        forged = copy.deepcopy(records[1])
        # Python considers True == 1, but the retained JSON binding must not.
        forged["context"]["input_projection"]["source"]["input"]["original_flag"] = 1
        with pytest.raises(EvaluationRecordsError, match="projection"):
            verify_input_projection(forged)


@pytest.mark.parametrize(
    "fault",
    [
        "nonstring_key",
        "empty_object",
        "empty_metric",
        "bad_metric",
        "nameless_metric",
        "nonrecord",
        "null_input",
    ],
)
def test_native_scalar_shape_corruptions(fault):
    evaluator = "deepeval"
    entry = native_entry(evaluator)
    if fault == "nonstring_key":
        entry["test_case"][1] = "not a JSON key"
    elif fault == "empty_object":
        entry["test_case"] = object()
    elif fault in ("empty_metric", "bad_metric"):
        evaluator = "hugging-face-evaluate"
        entry = native_entry(evaluator)
        entry["metric_result"] = {} if fault == "empty_metric" else 1.0
    elif fault == "nameless_metric":
        evaluator = "autoevals"
        entry = native_entry(evaluator)
        entry["metric_result"]["name"] = ""
    elif fault == "nonrecord":
        evaluator = "opik"
        entry = native_entry(evaluator)
        entry["dataset_item"] = SimpleNamespace(input="Q", output="A")
    else:
        entry["test_case"]["input"] = None
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, [entry])


@pytest.mark.parametrize("evaluator", ["ragas", "openevals"])
def test_empty_or_unsupported_metric_lists(evaluator):
    entry = native_entry(evaluator)
    entry["metric_result"] = []
    with pytest.raises(EvaluationRecordsError):
        export_records(evaluator, [entry])


def _documented_deepeval_raw_capture(test_case, tmp_path, monkeypatch):
    from invarlock.evaluation_record_contracts.contracts import digest
    from invarlock.evaluation_records.adapters import load_run

    document = (
        Path(__file__).resolve().parents[2]
        / "examples/evaluator-qualification/maintained/CAPTURE.md"
    )
    blocks = [
        part.split("```", 1)[0]
        for part in document.read_text().split("```python\n")[1:]
    ]
    test_case.metadata = {
        "category": "documented-slice",
        "invarlock_scores": {"quality": 0.75},
        "invarlock_likelihood": {
            "basis": "reference_continuation",
            "logprob_sum": -0.5,
            "token_count": 1,
            "utf8_byte_count": len(test_case.expected_output.encode("utf-8")),
            "input_digest": digest(test_case.input),
            "reference_digest": digest(test_case.expected_output),
            "artifact_digest": "sha256:" + "a" * 64,
            "configuration_digest": digest("documented-configuration"),
            "tokenizer_digest": digest("documented-tokenizer"),
            "source": {"name": "deepeval", "version": "4.1.3"},
        },
    }
    monkeypatch.chdir(tmp_path)
    namespace = {
        "captured_test_cases": [("case-1", test_case)],
        "planned_ids": ["case-1"],
    }
    exec(compile(blocks[0], str(document), "exec"), namespace)
    path = tmp_path / "subject-native.json"
    captured = json.loads(path.read_text())
    assert captured[0]["metadata"] == test_case.metadata
    run = load_run(
        path,
        adapter="evaluator-native-json",
        source={"name": "deepeval", "version": "4.1.3"},
        run_id="documented",
        artifact_digest="sha256:" + "a" * 64,
    )
    assert run["records"][0]["metadata"] == {"category": "documented-slice"}
    assert run["records"][0]["scores"] == {"quality": 0.75}
    assert run["records"][0]["likelihood"] == test_case.metadata["invarlock_likelihood"]
    # Independent scheduled membership is checked by the actual documented code.
    path.unlink()
    namespace["planned_ids"] = ["case-1", "missing"]
    with pytest.raises(ValueError, match="complete planned schedule"):
        exec(compile(blocks[0], str(document), "exec"), namespace)
    assert not path.exists()
    return blocks[1], document, test_case


def test_documented_deepeval_python_capture(tmp_path, monkeypatch):
    pytest.importorskip("deepeval")
    if importlib.metadata.version("deepeval") != "4.1.3":
        pytest.skip("requires pinned deepeval==4.1.3")
    from deepeval.test_case import LLMTestCase

    from invarlock.evaluation_records.adapters import load_run

    test_case = LLMTestCase(
        input="Question?", actual_output="Answer", expected_output="Answer"
    )
    code, document, test_case = _documented_deepeval_raw_capture(
        test_case, tmp_path, monkeypatch
    )
    namespace = {
        "captured_test_cases": [("case-1", test_case)],
        "planned_ids": ["case-1"],
        "model_artifact_digest": "sha256:" + "a" * 64,
    }
    exec(compile(code, str(document), "exec"), namespace)
    run = load_run(
        tmp_path / "subject-export.json",
        adapter="evaluator-json",
        source={"name": "deepeval", "version": "4.1.3"},
        run_id="subject-campaign",
        artifact_digest="sha256:" + "a" * 64,
    )
    assert run["records"][0]["metadata"] == {"category": "documented-slice"}
    assert run["records"][0]["scores"] == {"quality": 0.75}
    assert run["records"][0]["likelihood"] == test_case.metadata["invarlock_likelihood"]
