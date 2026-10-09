from __future__ import annotations

import json
import math
import runpy
from fractions import Fraction as F
from pathlib import Path

import pytest

from examples.score_contributions import PairedCase, mean_contributions


def test_small_changes_and_rounding_reconcile_without_rewriting_a_report():
    cases = [
        PairedCase("large", "large", F(2**54), F(2**54)),
        PairedCase("small", "small", F(0), F(1)),
    ]
    result = mean_contributions(cases, reported_delta=F(0), limit=1)
    assert result["exact_delta"] == "1/2"
    assert result["rounding_residual"] == "-1/2"
    assert result["reported_delta"] == "0"
    assert result["authority"] == "none"
    assert result["cases"][0]["case_id"] == "small"
    assert result["omitted_case_count"] == 1
    assert result["omitted_contribution"] == "0"
    assert F(result["visible_contribution"]) + F(result["rounding_residual"]) == 0


def test_paired_float_subtraction_is_not_an_exact_oracle():
    high = float(2**54)
    assert math.fsum([1.0 - high, 1.0 + high]) / 2 == 0.0
    result = mean_contributions(
        [PairedCase("a", "a", F(high), F(1)), PairedCase("b", "b", F(-high), F(1))],
        reported_delta=F(1),
    )
    assert result["exact_delta"] == "1"
    assert result["rounding_residual"] == "0"


def test_unequal_units_and_omitted_contributions_keep_the_actual_weights():
    cases = [
        PairedCase("a", "unit-a", F(0), F(1)),
        PairedCase("b", "unit-b", F(1), F(0)),
        PairedCase("c", "unit-b", F(0), F(0)),
        PairedCase("d", "unit-b", F(0), F(0)),
    ]
    reported = F("0.333333333333333")
    result = mean_contributions(cases, reported_delta=reported, limit=1)
    assert result["unit_count"] == 2
    assert result["exact_delta"] == "1/3"
    assert result["cases"][0]["weight"] == "1/2"
    assert result["omitted_contribution"] == "-1/6"
    assert (
        F(result["visible_contribution"])
        + F(result["omitted_contribution"])
        + F(result["rounding_residual"])
        == reported
    )
    assert (
        mean_contributions(list(reversed(cases)), reported_delta=reported, limit=1)
        == result
    )


def test_decimal_case_means_keep_repetition_averaging_exact():
    # Repetition means are supplied after averaging within each complete case.
    result = mean_contributions(
        [PairedCase("a", "u", F(1, 3), F(2, 3)), PairedCase("b", "v", F(1), F(0))],
        reported_delta=F(-1, 3),
    )
    assert result["exact_delta"] == "-1/3"
    assert sum(F(case["weight"]) for case in result["cases"]) == 1


def test_subnormal_residual_and_empty_visible_view_survive_json():
    tiny = F.from_float(math.ulp(0.0))
    result = mean_contributions(
        [PairedCase("a", "a", F(0), tiny), PairedCase("b", "b", F(0), F(0))],
        reported_delta=F(0),
        limit=0,
    )
    wire = json.loads(json.dumps(result))
    assert F(wire["exact_delta"]) == tiny / 2
    assert F(wire["rounding_residual"]) == -tiny / 2
    assert wire["cases"] == []
    assert F(wire["omitted_contribution"]) == tiny / 2


@pytest.mark.parametrize(
    "cases,reported,limit,message",
    [
        ([], F(0), 1, "complete paired"),
        ([PairedCase("a", "u", F(0), F(0))], F(0), True, "limit"),
        ([PairedCase("a", "u", F(0), F(0))], F(0), -1, "limit"),
        ([PairedCase("a", "u", F(0), F(0))], F(0), 201, "limit"),
        ([PairedCase("a", "u", F(0), F(0))], 0.0, 1, "reported_delta"),
        (["not a paired case"], F(0), 1, "PairedCase"),
        ([PairedCase("", "u", F(0), F(0))], F(0), 1, "nonempty"),
        ([PairedCase("a", " ", F(0), F(0))], F(0), 1, "nonempty"),
        ([PairedCase("a", "u", F(0), F(0))] * 2, F(0), 1, "unique"),
        ([PairedCase("a", "u", 0.0, F(0))], F(0), 1, "Fractions"),
    ],
)
def test_incomplete_or_ambiguous_inputs_are_not_silently_reinterpreted(
    cases, reported, limit, message
):
    with pytest.raises(ValueError, match=message):
        mean_contributions(cases, reported_delta=reported, limit=limit)


def test_example_runs_without_model_or_evidence_mutation(capsys):
    script = Path(__file__).resolve().parents[2] / "examples/score_contributions.py"
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["exact_delta"] == "1/2"
    assert result["rounding_residual"] == "-1/2"
