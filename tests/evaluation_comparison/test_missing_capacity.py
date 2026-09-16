"""Missing-result capacity checks preserve every ID and the existing wire form."""

import copy

import pytest

from invarlock.evaluation_comparison.capacity import (
    check_missing_id_capacity,
    missing_pair,
)
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evidence_pack_contract import canonical_json_bytes


def pair(identifier="case", *, left_error=None, right_error=None, scores=True):
    left = {
        "id": identifier,
        "error": left_error,
        "scores": {"quality": 1} if scores else {},
    }
    right = {
        "id": identifier,
        "error": right_error,
        "scores": {"quality": 1} if scores else {},
    }
    return left, right


BINARY = {"kind": "exact_match"}
RECORDED = {"kind": "recorded", "score_key": "quality"}


@pytest.mark.parametrize("side", [0, 1])
def test_errors_and_recorded_score_absence_on_either_side_are_missing(side):
    rows = pair()
    rows[side]["error"] = "execution error"
    assert missing_pair(*rows, BINARY)
    rows = pair()
    rows[side]["scores"] = {}
    assert missing_pair(*rows, RECORDED)
    assert not missing_pair(*rows, BINARY)


def test_complete_rows_are_not_charged_as_missing_even_with_long_ids():
    selected = [pair("😀" * 120 + str(i)) for i in range(30)]
    assert (
        check_missing_id_capacity([("overall", selected)], [BINARY], byte_limit=2) == 2
    )


def test_exact_utf8_and_escaped_id_size_accepts_boundary_without_mutation():
    selected = [
        pair('é😀\\"', left_error="failed"),
        pair("second", scores=False),
        pair("complete"),
    ]
    scopes = [("overall", selected), ("overlap", selected), ("subset", selected[:1])]
    metrics = [BINARY, RECORDED]
    before = copy.deepcopy(scopes)
    expected = sum(
        len(
            canonical_json_bytes(
                [a["id"] for a, b in rows if missing_pair(a, b, metric)], newline=False
            )
        )
        for _, rows in scopes
        for metric in metrics
    )
    assert check_missing_id_capacity(scopes, metrics, byte_limit=expected) == expected
    with pytest.raises(EvaluationRecordsError, match="missing-ID arrays"):
        check_missing_id_capacity(scopes, metrics, byte_limit=expected - 1)
    assert scopes == before


def test_maximum_overlap_is_charged_for_every_metric_without_large_fixture():
    selected = [pair("😀", left_error="failed")]
    scopes = [(f"scope-{i}", selected) for i in range(17)]
    metrics = [BINARY] * 16
    expected = 17 * 16 * len(canonical_json_bytes(["😀"], newline=False))
    with pytest.raises(
        EvaluationRecordsError, match="complete policy has not been evaluated"
    ):
        check_missing_id_capacity(scopes, metrics, byte_limit=expected - 1)
    assert check_missing_id_capacity(scopes, metrics, byte_limit=expected) == expected


def test_missing_capacity_does_not_consult_output_values_or_constant_scores():
    selected = [pair("a", left_error="failed"), pair("b", scores=False)]
    for left, right in selected:
        left["output"] = right["output"] = object()
    expected = len(canonical_json_bytes(["a", "b"], newline=False))
    assert (
        check_missing_id_capacity(
            [("overall", selected)], [RECORDED], byte_limit=expected
        )
        == expected
    )


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_invalid_local_byte_limit_rejected(limit):
    with pytest.raises(EvaluationRecordsError, match="non-negative integer"):
        check_missing_id_capacity([], [], byte_limit=limit)


def test_comparison_rejects_amplification_before_first_metric(monkeypatch):
    from invarlock.evaluation_comparison import comparison
    from invarlock.evaluation_record_contracts import contracts
    from invarlock.evaluation_records.templates import example_project

    baseline, candidate, policy = example_project("classification")
    for run in (baseline, candidate):
        for i, row in enumerate(run["records"]):
            row["id"] = "😀" * 120 + str(i)
            row["error"] = "capture incomplete"
            row["metadata"]["included"] = "yes"
    policy["metrics"] = [
        {**policy["metrics"][0], "name": f"metric-{i}"} for i in range(16)
    ]
    policy["slices"] = [
        {"name": f"slice-{i}", "where": {"included": "yes"}} for i in range(16)
    ]
    limit = max(len(canonical_json_bytes(v)) for v in (baseline, candidate, policy))
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", limit)

    def forbidden(*args, **kwargs):
        pytest.fail("amplification must be rejected before the first metric")

    monkeypatch.setattr(comparison, "_metric_result", forbidden)
    with pytest.raises(EvaluationRecordsError, match="missing-ID arrays"):
        comparison.compare_runs(baseline, candidate, policy)
