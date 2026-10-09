from __future__ import annotations

import hashlib
import json
from decimal import Decimal as D

import pytest

from invarlock.judge_measurements.analysis import JudgeAnalysisPolicy
from tests._policy_boundary_support import check_boundaries
from tests.judge_measurements.test_analysis import _analyze, _bundle, _retain


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_complete_judge_analysis_boundaries_and_pairwise_conjunction(direction):
    plan, data = _bundle(groups=tuple(f"u{i}" for i in range(64)))
    for trial in data["trials"]:
        value = int(trial["case_id"].split("-")[1]) % 2
        rating = "correct" if value else "incorrect"
        trial["parse"].update(rating=rating, value=str(value))
        response = json.dumps({"rating": rating}, separators=(",", ":"))
        trial["attempts"][0]["response"].update(
            text=response, sha256=hashlib.sha256(response.encode()).hexdigest()
        )
    _retain(data)
    defaults = {
        "direction": direction,
        "allowed_degradation": D(1),
        "maximum_interval_width": D(2),
        "minimum_units": 1,
        "subject_bound": D(0) if direction == "higher" else D(1),
    }

    def compare(changes):
        return _analyze(plan, data, JudgeAnalysisPolicy(**(defaults | changes)))

    result = compare({})
    effect, subject = result.effect_interval, result.subject_interval
    check_boundaries(
        {
            "allowed_degradation": -effect.lower
            if direction == "higher"
            else effect.upper,
            "maximum_interval_width": max(
                effect.upper - effect.lower, subject.upper - subject.lower
            ),
            "minimum_units": 64,
            "subject_bound": subject.lower if direction == "higher" else subject.upper,
        },
        lambda changes: compare(changes).decision == "pass",
        relaxing_down={"minimum_units", "subject_bound"}
        if direction == "higher"
        else {"minimum_units"},
    )
