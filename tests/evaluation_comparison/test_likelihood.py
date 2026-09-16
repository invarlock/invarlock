"""Typed captured reference likelihood validation and native arithmetic parity."""

import pytest

from invarlock.evaluation_comparison.comparison import compare_runs, make_run
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evidence_pack_contract import _paired_resampling_interval

KIND = "normalized_nll_per_utf8_byte"
CONFIG = digest("configuration")
TOKENIZER = digest("tokenizer")
SOURCE = {"name": "external", "version": "1"}


def row(identifier="a", logprob=-4.0):
    value = {"id": identifier, "input": "prompt", "expected": "é", "output": None}
    value["likelihood"] = {
        "basis": "reference_continuation",
        "logprob_sum": logprob,
        "token_count": 1,
        "utf8_byte_count": 2,
        "input_digest": digest(value["input"]),
        "reference_digest": digest(value["expected"]),
        "artifact_digest": digest("model"),
        "configuration_digest": CONFIG,
        "tokenizer_digest": TOKENIZER,
        "source": SOURCE.copy(),
    }
    return value


def run(rows):
    return make_run(rows, source=SOURCE, run_id="test", artifact_digest=digest("model"))


def policy():
    return {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "nll",
                "kind": KIND,
                "configuration": {
                    "configuration_digest": CONFIG,
                    "baseline_tokenizer_digest": TOKENIZER,
                    "subject_tokenizer_digest": TOKENIZER,
                },
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": 1,
                "ratio_max": 1.1,
                "maximum_interval_width": 10,
            }
        ],
        "slices": [],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("basis", "generated_answer"),
        ("logprob_sum", 0.1),
        ("logprob_sum", True),
        ("logprob_sum", float("nan")),
        ("logprob_sum", float("inf")),
        ("token_count", True),
        ("token_count", 1.0),
        ("token_count", 0),
        ("utf8_byte_count", 2.0),
        ("utf8_byte_count", True),
        ("utf8_byte_count", 1),
        ("input_digest", digest("other")),
        ("reference_digest", digest("other")),
        ("artifact_digest", digest("other")),
        ("source", {"name": "other", "version": "1"}),
        ("unexpected", 1),
    ],
)
def test_rejects_malformed_or_unbound_likelihood(key, value):
    record = row()
    record["likelihood"][key] = value
    with pytest.raises(EvaluationRecordsError):
        run([record])


def test_native_ratio_interval_and_captured_assurance():
    baseline = run([row("a", -4), row("b", -8)])
    subject = run([row("a", -3), row("b", -6)])
    result = compare_runs(baseline, subject, policy())["metrics"][0]
    seed = digest(
        [
            {key: item[key] for key in ("id", "input", "expected", "metadata")}
            for item in baseline["records"]
        ]
    )
    lower, upper = _paired_resampling_interval(
        metric=KIND,
        baseline_scores=[2, 4],
        subject_scores=[1.5, 3],
        schedule_sha256=seed.removeprefix("sha256:"),
    )
    assert result["ratio"] == 0.75
    assert result["delta"] == -0.75
    assert result["interval"]["lower"] == lower
    assert result["interval"]["upper"] == upper
    assert result["interval_unit"] == "ratio"
    assert result["scoring_assurance"] == "recomputed"
    assert result["decision"] == "pass"


def test_missing_likelihood_is_insufficient():
    missing = row()
    del missing["likelihood"]
    result = compare_runs(run([row()]), run([missing]), policy())["metrics"][0]
    assert result["decision"] == "insufficient_evidence"
    assert result["missing_ids"] == ["a"]
    assert result["ratio"] is None


@pytest.mark.parametrize("expected", [None, {}, [], 4, ""])
def test_reference_text_required(expected):
    record = row()
    record["expected"] = expected
    record["likelihood"]["reference_digest"] = digest(expected)
    with pytest.raises(EvaluationRecordsError, match="reference text"):
        run([record])


@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("field", ["configuration_digest", "tokenizer_digest"])
def test_policy_identity_mismatch_even_on_errored_rows(side, field):
    left, right = row(), row()
    selected = left if side == "baseline" else right
    selected["likelihood"][field] = digest("other")
    selected["error"] = "failed"
    with pytest.raises(EvaluationRecordsError, match="differs from policy"):
        compare_runs(run([left]), run([right]), policy())


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", "higher"),
        ("unit", "score"),
        ("ratio_max", 0),
        ("ratio_max", True),
        ("ratio_max", float("inf")),
        ("maximum_regression", 0),
        ("score_key", "nll"),
        ("configuration", {}),
        ("configuration", {"extra": 1}),
        ("aggregation", "sum"),
        ("maximum_interval_width", 0),
        ("subject_minimum", True),
    ],
)
def test_closed_likelihood_policy(field, value):
    configured = policy()
    configured["metrics"][0][field] = value
    with pytest.raises(EvaluationRecordsError):
        compare_runs(run([row()]), run([row()]), configured)


def test_missing_ratio_threshold():
    configured = policy()
    del configured["metrics"][0]["ratio_max"]
    with pytest.raises(EvaluationRecordsError):
        compare_runs(run([row()]), run([row()]), configured)


def test_distinct_approved_tokenizers_and_zero_subject():
    right = row(logprob=0)
    right["likelihood"]["tokenizer_digest"] = digest("subject tokenizer")
    right["likelihood"]["token_count"] = 7
    configured = policy()
    configured["metrics"][0]["configuration"]["subject_tokenizer_digest"] = digest(
        "subject tokenizer"
    )
    result = compare_runs(run([row()]), run([right]), configured)["metrics"][0]
    assert result["decision"] == "pass"
    assert result["ratio"] == 0


def test_zero_baseline_cannot_define_ratio():
    with pytest.raises(
        EvaluationRecordsError, match="baseline mean must be greater than zero"
    ):
        compare_runs(run([row(logprob=0)]), run([row()]), policy())


def test_native_bootstrap_zero_denominator_fails_closed():
    with pytest.raises(
        EvaluationRecordsError, match="baseline mean must be greater than zero"
    ):
        compare_runs(
            run([row("a", 0), row("b", -4)]), run([row("a"), row("b")]), policy()
        )


def test_overflow_means_fail_closed():
    with pytest.raises(EvaluationRecordsError, match="finite numeric range"):
        compare_runs(
            run([row(str(i), -1.7e308) for i in range(3)]),
            run([row(str(i)) for i in range(3)]),
            policy(),
        )


def test_ratio_overflow_fails_closed():
    with pytest.raises(EvaluationRecordsError, match="finite numeric range"):
        compare_runs(
            run([row(logprob=-1e-320)]), run([row(logprob=-1.7e308)]), policy()
        )


def test_missing_or_errored_likelihood_does_not_select_complete_subset():
    bad = row("b")
    bad["error"] = "source failed"
    result = compare_runs(run([row("a"), row("b")]), run([row("a"), bad]), policy())[
        "metrics"
    ][0]
    assert result["missing_ids"] == ["b"]
    assert result["ratio"] is None
    assert result["baseline_mean"] is None
    assert "reference-continuation" in result["reasons"][0]


def test_likelihood_minimum_count_and_empty_slice():
    configured = policy()
    configured["metrics"][0]["minimum_count"] = 2
    configured["slices"] = [{"name": "empty", "where": {"group": "x"}}]
    metrics = compare_runs(run([row()]), run([row()]), configured)["metrics"]
    assert [m["decision"] for m in metrics] == ["insufficient_evidence"] * 2
    assert [m["count"] for m in metrics] == [1, 0]


def test_ratio_bound_is_inclusive_and_rejects_regression():
    configured = policy()
    configured["metrics"][0]["ratio_max"] = 1.25
    assert (
        compare_runs(run([row()]), run([row(logprob=-5)]), configured)["decision"]
        == "pass"
    )
    result = compare_runs(run([row()]), run([row(logprob=-6)]), configured)["metrics"][
        0
    ]
    assert result["decision"] == "regression"
    assert result["reasons"] == ["upper ratio interval bound exceeds ratio_max"]


def test_ratio_interval_width_and_absolute_bounds():
    configured = policy()
    configured["metrics"][0].update(ratio_max=3, maximum_interval_width=0.01)
    left = run([row("a"), row("b")])
    right = run([row("a", -2), row("b", -8)])
    result = compare_runs(left, right, configured)["metrics"][0]
    assert result["decision"] == "insufficient_evidence"
    assert result["reasons"] == ["interval is too wide"]
    configured["metrics"][0]["subject_maximum"] = 2
    result = compare_runs(left, right, configured)["metrics"][0]
    assert result["decision"] == "regression"
    assert result["reasons"] == ["subject mean fails absolute bounds"]


def test_likelihood_budget_charges_native_replicates_before_missingness():
    configured = policy()
    with pytest.raises(EvaluationRecordsError, match="bootstrap draws"):
        compare_runs(run([row()]), run([row()]), configured, max_bootstrap_draws=2047)
    assert (
        compare_runs(run([row()]), run([row()]), configured, max_bootstrap_draws=2048)[
            "decision"
        ]
        == "pass"
    )


@pytest.mark.parametrize("with_policy", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_report_shows_ratio_units_and_captured_basis(with_policy, missing):
    from invarlock.record_reporting import _metric_views

    other = row(logprob=-3)
    if missing:
        del other["likelihood"]
    configured = policy()
    result = compare_runs(run([row()]), run([other]), configured)
    metric = _metric_views(
        result, {"nll": configured["metrics"][0]} if with_policy else {}
    )[0]
    assert "captured reference-continuation likelihood" in " ".join(metric.notes)
    if missing:
        assert metric.change == "Unavailable"
        assert metric.interval is None
    else:
        assert metric.change == "0.75 ratio"
        assert metric.baseline == "2 nats / byte"
        assert metric.interval.neutral == 1
        assert metric.interval.threshold_direction == (
            "maximum" if with_policy else None
        )
        assert metric.interval.unit == "ratio"
        assert metric.interval.estimate == 0.75
        if with_policy:
            check = next(c for c in metric.checks if c.name == "Maximum NLL ratio")
            assert check.required == "<= 1.1 ratio"


@pytest.mark.parametrize(
    "field,value",
    [("direction", "higher"), ("unit", "score"), ("maximum_regression", 0)],
)
def test_shared_policy_semantic_validation(field, value):
    from invarlock.evaluation_comparison.likelihood import validate_likelihood_policy

    metric = policy()["metrics"][0]
    metric[field] = value
    with pytest.raises(EvaluationRecordsError, match="normalized NLL"):
        validate_likelihood_policy(metric)


def test_generated_output_does_not_change_reference_likelihood():
    left, right = row(), row(logprob=-3)
    left["output"], right["output"] = "generated baseline text", "different output"
    result = compare_runs(run([left]), run([right]), policy())
    assert result["metrics"][0]["ratio"] == 0.75
    assert "runtime facts" in result["limitations"][-1]


def test_likelihood_payload_closed_and_complete():
    for field in row()["likelihood"]:
        record = row()
        del record["likelihood"][field]
        with pytest.raises(EvaluationRecordsError):
            run([record])
    record = row()
    record["likelihood"] = None
    with pytest.raises(EvaluationRecordsError):
        run([record])


def test_missing_case_and_changed_reference_fail_pairing():
    with pytest.raises(EvaluationRecordsError, match="record IDs differ"):
        compare_runs(run([row("a")]), run([row("b")]), policy())
    changed = row()
    changed["expected"] = "ab"
    changed["likelihood"]["reference_digest"] = digest("ab")
    with pytest.raises(EvaluationRecordsError, match="expected changed"):
        compare_runs(run([row()]), run([changed]), policy())


def test_output_contract_requires_ratio_and_rejects_recorded_assurance():
    from copy import deepcopy

    from invarlock.evaluation_record_contracts.contracts import validate

    comparison = compare_runs(run([row()]), run([row()]), policy())
    for field in ("ratio", "interval_unit"):
        changed = deepcopy(comparison)
        del changed["metrics"][0][field]
        with pytest.raises(EvaluationRecordsError):
            validate(changed, "comparison")
    comparison["metrics"][0]["scoring_assurance"] = "recorded"
    with pytest.raises(EvaluationRecordsError):
        validate(comparison, "comparison")


def test_native_parity_with_nonconstant_paired_ratios():
    baseline = run([row("a", -4), row("b", -8)])
    subject = run([row("a", -3), row("b", -7)])
    result = compare_runs(baseline, subject, policy())["metrics"][0]
    seed = digest(
        [
            {key: item[key] for key in ("id", "input", "expected", "metadata")}
            for item in baseline["records"]
        ]
    )
    expected = _paired_resampling_interval(
        metric=KIND,
        baseline_scores=[2, 4],
        subject_scores=[1.5, 3.5],
        schedule_sha256=seed.removeprefix("sha256:"),
    )
    assert result["ratio"] == 2.5 / 3
    assert result["delta"] == -0.5
    assert (result["interval"]["lower"], result["interval"]["upper"]) == expected
    assert expected == (0.75, 0.875)


def test_mean_of_byte_normalized_record_scores_not_pooled_tokens_or_bytes():
    left = [row("a", -4), row("b", -4)]
    right = [row("a", -2), row("b", -8)]
    for records in (left, right):
        records[1]["expected"] = "four"
        records[1]["likelihood"].update(
            reference_digest=digest("four"), utf8_byte_count=4, token_count=3
        )
    result = compare_runs(run(left), run(right), policy())["metrics"][0]
    assert result["baseline_mean"] == 1.5
    assert result["subject_mean"] == 1.5
    assert result["ratio"] == 1.0


@pytest.mark.parametrize(
    "logprob,expected_decision", [(-3, "pass"), (-6, "regression")]
)
def test_signed_captured_likelihood_replay_receipt_and_reports(
    tmp_path, logprob, expected_decision
):
    import json

    from invarlock.captured_evaluation import evaluate_captured_request
    from invarlock.captured_reporting import _load, _view
    from invarlock.captured_verification import verify_captured_evidence
    from invarlock.report_presentation import render_html, render_markdown
    from tests.core.test_captured_evaluation import _key, _request
    from tests.core.test_captured_verification import _anchors, _private_key

    request = _request(tmp_path, run([row()]), run([row(logprob=logprob)]), policy())
    evaluate_captured_request(request, signing_key_path=_key(tmp_path))
    anchors = _anchors(request)
    verifier, _ = _private_key(tmp_path)
    receipt = tmp_path / "receipt.json"
    result = verify_captured_evidence(
        request.evidence,
        policy_path=tmp_path / "policy.json",
        expected_baseline_run=anchors["baseline"],
        expected_subject_run=anchors["subject"],
        expected_request_digest=anchors["request"],
        expected_signer=anchors["signer"],
        receipt_path=receipt,
        verifier_signing_key_path=verifier,
        verifier_identity="recipient",
    )
    assert result["kind"] == "captured"
    assert result["ok"] is (expected_decision == "pass")
    statement = json.loads(receipt.read_text())["statement"]
    assert statement["scoring_assurance"] == [
        {
            "name": "nll",
            "slice": "overall",
            "kind": KIND,
            "scoring_assurance": "recomputed",
        }
    ]
    manifest, payloads, signer, _ = _load(request.evidence)
    view = _view(manifest, payloads, signer)
    assert view.decision == expected_decision
    assert view.technical["metrics"][0]["unit"] == "nats_per_utf8_byte"
    for rendered in (render_html(view), render_markdown(view)):
        assert "nats / byte" in rendered
        assert "No absolute minimum score" not in rendered
        assert "ratio" in rendered
        assert "captured reference-continuation likelihood facts" in rendered
