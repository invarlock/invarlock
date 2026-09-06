"""Native assertion failures are evaluable outputs, not provider failures."""

import copy
import json
from pathlib import Path

import pytest

from invarlock.pipeline import PipelineError, compare_runs, load_run
from invarlock.pipeline.templates import example_project

FIXTURE = Path(__file__).parent / "fixtures/promptfoo-0.121.19-assertion-failure.json"


def _load(tmp_path, row):
    path = tmp_path / "native.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return load_run(
        path,
        adapter="promptfoo-jsonl",
        source={"name": "promptfoo", "version": "0.121.19"},
        run_id="native-failure",
        artifact_digest="sha256:" + "a" * 64,
    )


def test_actual_native_failed_assertion_keeps_the_model_response_evaluable(tmp_path):
    row = json.loads(FIXTURE.read_bytes())
    run = _load(tmp_path, row)
    assert run["records"][0]["error"] is None
    assert run["records"][0]["output"] == row["response"]["output"]
    assert run["records"][0]["scores"]["score"] == 0


def test_failed_quality_assertions_produce_regression_not_insufficient_evidence(
    tmp_path,
):
    row = json.loads(FIXTURE.read_bytes())
    run = _load(tmp_path, row)
    # Repeat the native shape for a deterministic statistics unit test. These
    # copies are not presented as separate model generations.
    run["records"] = [dict(run["records"][0], id=str(i)) for i in range(40)]
    baseline = copy.deepcopy(run)
    for record in baseline["records"]:
        record["output"] = record["expected"]
    _, _, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    result = compare_runs(baseline, run, policy)
    assert result["decision"] == "regression"
    assert result["metrics"][0]["candidate_mean"] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(success=True),
        lambda r: r["gradingResult"].update(pass_=True),
        lambda r: r.update(gradingResult=None),
        lambda r: r["gradingResult"].update(score=0.5),
        lambda r: r["response"].pop("output"),
        lambda r: r["response"].update(error="provider failed"),
        lambda r: r.update(failureReason=True),
        lambda r: r.update(failureReason=3),
        lambda r: r.update(score=True),
        lambda r: r["gradingResult"].update(score="0"),
        lambda r: r["gradingResult"].update(reason=None),
        lambda r: r.pop("error"),
        lambda r: r.update(error="unrelated execution failure"),
    ],
)
def test_ambiguous_native_failure_flags_fail_closed(tmp_path, change):
    row = json.loads(FIXTURE.read_bytes())
    # Python cannot use the native JSON field 'pass' as a keyword argument.
    change(row)
    if "pass_" in (row.get("gradingResult") or {}):
        row["gradingResult"]["pass"] = row["gradingResult"].pop("pass_")
    with pytest.raises(PipelineError):
        _load(tmp_path, row)


@pytest.mark.parametrize("native_reason", [None, 2])
def test_provider_errors_remain_incomplete_even_if_partial_output_exists(
    tmp_path, native_reason
):
    row = json.loads(FIXTURE.read_bytes())
    row["error"] = "provider timed out"
    if native_reason is None:
        del row["failureReason"]
    else:
        row["failureReason"] = native_reason
    row["response"]["error"] = "provider timed out"
    run = _load(tmp_path, row)
    assert run["records"][0]["error"] == "upstream_error"


def _successful_row():
    row = json.loads(FIXTURE.read_bytes())
    row.update(failureReason=0, success=True, score=1)
    row.pop("error")
    row["gradingResult"] = {"pass": True, "score": 1, "reason": "Assertion passed"}
    row["response"]["output"] = row["testCase"]["metadata"]["invarlock_expected"]
    return row


def test_native_success_remains_evaluable(tmp_path):
    assert _load(tmp_path, _successful_row())["records"][0]["error"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(error="execution failed"),
        lambda r: r["response"].update(error="provider failed"),
        lambda r: r.update(success=False),
        lambda r: r["gradingResult"].update({"pass": False}),
        lambda r: r.update(failureReason=2),
    ],
)
def test_native_success_cannot_contradict_failure_fields(tmp_path, change):
    row = _successful_row()
    change(row)
    with pytest.raises(PipelineError):
        _load(tmp_path, row)
