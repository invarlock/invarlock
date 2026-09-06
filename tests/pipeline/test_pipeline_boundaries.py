"""Release decisions must reject ambiguous data and never upgrade judgments."""

from copy import deepcopy

import pytest

from invarlock.pipeline import PipelineError, compare_runs, make_run


def policy(kind="normalized_match"):
    metric = {
        "name": "quality",
        "kind": kind,
        "configuration": {},
        "direction": "higher",
        "unit": "score",
        "aggregation": "mean",
        "minimum_count": 2,
        "maximum_regression": 1.0,
        "maximum_interval_width": 2.0,
        "candidate_minimum": 0.8,
    }
    return {"format": "invarlock/pipeline-policy-v1", "metrics": [metric], "slices": []}


def run(outputs=("yes", "no")):
    return make_run(
        [
            {
                "id": str(i),
                "input": f"question {i}",
                "expected": target,
                "output": output,
                "metadata": {"group": "a"},
            }
            for i, (target, output) in enumerate(
                zip(("yes", "no"), outputs, strict=True)
            )
        ],
        source={"name": "evaluation", "version": "1"},
        run_id="test",
        artifact_digest="sha256:" + "a" * 64,
    )


def test_normalized_import_preserves_the_customer_metric():
    result = compare_runs(run(), run((" YES ", "No")), policy())
    assert result["decision"] == "pass"
    assert result["metrics"][0]["scoring_assurance"] == "recomputed"


@pytest.mark.parametrize(
    "mutation", ["duplicate", "input", "target", "missing", "slice"]
)
def test_no_silent_pairing_or_slice_drift(mutation):
    baseline, candidate = run(), run()
    if mutation == "duplicate":
        candidate["records"][1]["id"] = "0"
    elif mutation == "input":
        candidate["records"][0]["input"] = "changed"
    elif mutation == "target":
        candidate["records"][0]["expected"] = "changed"
    elif mutation == "missing":
        candidate["records"].pop()
    else:
        candidate["records"][0]["metadata"]["group"] = "b"
    with pytest.raises(PipelineError):
        compare_runs(baseline, candidate, policy())


def test_recorded_scores_require_explicit_provenance_policy():
    baseline, candidate = run(), run()
    for value in (baseline, candidate):
        for row in value["records"]:
            row["scores"] = {"judge": 0.9}
    selected = policy("recorded")
    selected["metrics"][0]["score_key"] = "judge"
    with pytest.raises(PipelineError, match="provenance"):
        compare_runs(baseline, candidate, selected)
    provenance = {
        "kind": "judge",
        "unit": "score",
        "source": "judge-model",
        "version": "pinned-1",
        "rubric_digest": "sha256:" + "b" * 64,
    }
    selected["metrics"][0]["accepted_provenance"] = provenance
    for value in (baseline, candidate):
        value["score_provenance"] = {"judge": provenance}
    result = compare_runs(baseline, candidate, selected)
    assert result["decision"] == "pass"
    assert result["metrics"][0]["scoring_assurance"] == "recorded"
    candidate = deepcopy(candidate)
    candidate["score_provenance"]["judge"]["version"] = "different"
    with pytest.raises(PipelineError, match="provenance"):
        compare_runs(baseline, candidate, selected)


def test_missing_scores_are_insufficient_not_a_smaller_passing_sample():
    baseline, candidate = run(), run()
    candidate["records"][0]["error"] = "timeout"
    result = compare_runs(baseline, candidate, policy())
    assert result["decision"] == "insufficient_evidence"


def test_unknown_policy_fields_cannot_be_ignored():
    selected = policy()
    selected["metrics"][0]["minimun_count"] = 999
    with pytest.raises(PipelineError):
        compare_runs(run(), run(), selected)


def test_absolute_floor_and_regression_are_both_required():
    result = compare_runs(run(("bad", "bad")), run(("bad", "bad")), policy())
    assert result["decision"] == "regression"


def test_a_slice_can_block_an_acceptable_overall_mean():
    selected = policy()
    selected["metrics"][0]["minimum_count"] = 1
    selected["slices"] = [{"name": "group-a", "where": {"group": "a"}}]
    result = compare_runs(run(), run(), selected)
    assert len(result["metrics"]) == 2
    assert result["metrics"][1]["slice"] == "group-a"
