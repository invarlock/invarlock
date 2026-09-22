"""Dedicated exports use the same installed boundary and preserve native facts."""

import json
from pathlib import Path

import pytest

from invarlock.engine import (
    EVALUATORS,
    EvaluationRecordsError,
    export_evaluator_result,
    load_run,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "native_ids,expected_error",
    [
        (("declared-run", "declared-run"), None),
        ((None, None), None),
        (("other-run", "other-run"), None),
        (("declared-run", None), "mixed run IDs"),
        (("declared-run", "other-run"), "multiple runs"),
    ],
)
def test_openai_evals_native_and_capture_run_identities_remain_distinct(
    tmp_path, native_ids, expected_error
):
    events = [
        {
            "sample_id": "case-a",
            "type": "sampling",
            "data": {"prompt": "Question", "sampled": "Answer"},
        },
        {
            "sample_id": "case-a",
            "type": "match",
            "data": {"correct": True, "expected": "Answer"},
        },
    ]
    for event, native_id in zip(events, native_ids, strict=True):
        if native_id is not None:
            event["run_id"] = native_id
    result = {"events": events}
    options = {
        "source_version": "1",
        "run_id": "declared-run",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    raw_path = tmp_path / "native.json"
    raw_path.write_text(json.dumps(result), encoding="utf-8")
    for capture in (
        lambda: export_evaluator_result(
            "openai-evals",
            result,
            tmp_path / "export.json",
            expected_ids=["case-a"],
            **options,
        ),
        lambda: load_run(
            raw_path,
            adapter="evaluator-native-json",
            source={"name": "openai-evals", "version": "1"},
            run_id=options["run_id"],
            artifact_digest=options["artifact_digest"],
        ),
    ):
        if expected_error is None:
            run = capture()
            assert run["run_id"] == "declared-run"
            assert (
                run["records"][0]["context"]["upstream_record"]["events"][0].get(
                    "run_id"
                )
                == native_ids[0]
            )
        else:
            with pytest.raises(EvaluationRecordsError, match=expected_error):
                capture()
    if expected_error is not None:
        assert not (tmp_path / "export.json").exists()


def test_export_capacity_and_nesting_fail_before_publication(tmp_path, monkeypatch):
    from invarlock.evaluation_records import integrations

    nested = {}
    for _ in range(70):
        nested = {"child": nested}
    path = tmp_path / "export"
    options = {"expected_ids": ["one"], "source_version": "1", "run_id": "run"}
    with pytest.raises(EvaluationRecordsError, match="nesting"):
        export_evaluator_result("inspect-ai", nested, path, **options)
    monkeypatch.setattr(integrations, "MAX_INPUT_BYTES", 1)
    with pytest.raises(EvaluationRecordsError, match="byte limit"):
        export_evaluator_result("inspect-ai", {}, path, **options)
    assert not path.exists()


@pytest.mark.parametrize("field", ["source_version", "run_id"])
@pytest.mark.parametrize("invalid", ["x" * 129, "value\nwith-control"])
def test_export_identity_bounds_match_the_canonical_run(tmp_path, field, invalid):
    raw = (FIXTURES / "inspect-0.3.254.json").read_bytes()
    result = json.loads(raw)
    options = {
        "expected_ids": [str(row["id"]) for row in result["samples"]],
        "source_version": "1",
        "run_id": "run",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    options[field] = invalid
    destination = tmp_path / "export.json"
    with pytest.raises(EvaluationRecordsError):
        export_evaluator_result("inspect-ai", result, destination, **options)
    assert not destination.exists()


def test_optional_sdk_failure_does_not_publish(tmp_path, monkeypatch):
    from invarlock.evaluation_records import integrations

    def unavailable(_):
        raise ImportError("optional SDK unavailable")

    monkeypatch.setattr(integrations, "import_module", unavailable)
    path = tmp_path / "export"
    with pytest.raises(EvaluationRecordsError, match="optional SDK unavailable"):
        export_evaluator_result(
            "inspect-ai",
            object(),
            path,
            expected_ids=["one"],
            source_version="1",
            run_id="run",
        )
    assert not path.exists()


@pytest.mark.parametrize(
    "evaluator,filename,version",
    [
        ("inspect-ai", "inspect-0.3.254.json", "0.3.254"),
        ("lm-evaluation-harness", "lm-eval-0.4.12.jsonl", "0.4.12"),
        ("promptfoo", "promptfoo-0.121.19.jsonl", "0.121.19"),
    ],
)
def test_original_sdk_exports_use_shared_capture_and_import(
    tmp_path, evaluator, filename, version
):
    raw = (FIXTURES / filename).read_text()
    result = (
        json.loads(raw)
        if filename.endswith(".json")
        else [json.loads(line) for line in raw.splitlines()]
    )
    rows = result["samples"] if evaluator == "inspect-ai" else result
    ids = [
        str(row["id"])
        if evaluator == "inspect-ai"
        else str(row["doc_id"])
        if evaluator == "lm-evaluation-harness"
        else f"{row['testIdx']}:{row['promptIdx']}"
        for row in rows
    ]
    path = tmp_path / "export.json"
    before = json.dumps(result)
    run = export_evaluator_result(
        evaluator,
        result,
        path,
        expected_ids=ids,
        source_version=version,
        run_id="retained",
        artifact_digest="sha256:" + "a" * 64,
    )
    assert json.dumps(result) == before
    envelope = json.loads(path.read_bytes())
    assert envelope["payload"] == result
    assert len(run["records"]) == 40
    assert run == load_run(
        path,
        adapter="evaluator-json",
        source={"name": evaluator, "version": version},
        run_id="retained",
        artifact_digest="sha256:" + "a" * 64,
    )
    for record, native in zip(run["records"], rows, strict=True):
        assert record["context"]["upstream_record"] == native
    with pytest.raises(FileExistsError):
        export_evaluator_result(
            evaluator,
            result,
            path,
            expected_ids=ids,
            source_version=version,
            run_id="retained",
            artifact_digest="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize("evaluator", [None, [], {}, "unknown"])
def test_unknown_profile_cannot_publish(tmp_path, evaluator):
    with pytest.raises(EvaluationRecordsError):
        export_evaluator_result(
            evaluator,
            {},
            tmp_path / "export",
            expected_ids=["one"],
            source_version="1",
            run_id="run",
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    "mutation", ["source", "version", "run", "format", "unknown", "missing", "payload"]
)
def test_envelope_identity_and_closed_contract(tmp_path, mutation):
    value = {
        "format": "invarlock/evaluator-export-v1",
        "evaluator": "inspect-ai",
        "source_version": "1",
        "run_id": "run",
        "payload": {
            "version": 2,
            "status": "success",
            "samples": [
                {
                    "id": "one",
                    "input": "Q",
                    "target": "A",
                    "output": {"choices": [{"message": {"content": "A"}}]},
                }
            ],
        },
    }
    if mutation == "source":
        value["evaluator"] = "promptfoo"
    elif mutation == "version":
        value["source_version"] = "2"
    elif mutation == "run":
        value["run_id"] = "other"
    elif mutation == "format":
        value["format"] = "unknown"
    elif mutation == "unknown":
        value["extra"] = True
    elif mutation == "missing":
        value.pop("payload")
    else:
        value["payload"] = 7
    path = tmp_path / "export.json"
    path.write_text(json.dumps(value))
    with pytest.raises(EvaluationRecordsError):
        load_run(
            path,
            adapter="evaluator-json",
            source={"name": "inspect-ai", "version": "1"},
            run_id="run",
            artifact_digest="sha256:" + "a" * 64,
        )


def test_all_nineteen_declared_profiles_are_present():
    matrix = json.loads(
        (
            Path(__file__).parents[2] / "examples/evaluator-qualification/matrix.json"
        ).read_bytes()
    )
    assert set(EVALUATORS) == {profile["profile_id"] for profile in matrix["profiles"]}


@pytest.mark.parametrize(
    "bad", [{1: "changed key"}, {"nested": object()}, {"nested": float("inf")}]
)
def test_export_never_stringifies_or_coerces_native_fields(tmp_path, bad):
    with pytest.raises(EvaluationRecordsError):
        export_evaluator_result(
            "inspect-ai",
            bad,
            tmp_path / "export",
            expected_ids=["one"],
            source_version="1",
            run_id="run",
        )
    assert not (tmp_path / "export").exists()
