import math

from invarlock.guards.variance_batching import safe_mean
from invarlock.guards.variance_policy import predictive_gate_outcome


def test_safe_mean_branches():
    assert safe_mean(None, default=1.0) == 1.0
    assert safe_mean([], default=2.0) == 2.0
    assert math.isclose(safe_mean([1.0, 3.0], default=0.0) or 0.0, 2.0)


def test_predictive_gate_outcome_one_sided_paths():
    # CI unavailable
    assert predictive_gate_outcome(0.0, None, 0.1, True) == (False, "ci_unavailable")

    # One-sided failure paths
    assert predictive_gate_outcome(-0.5, (0.1, 0.2), 0.1, True) == (
        False,
        "ci_contains_zero",
    )
    assert predictive_gate_outcome(0.1, (-0.3, -0.1), 0.1, True) == (
        False,
        "mean_not_negative",
    )
    assert predictive_gate_outcome(-0.05, (-0.3, -0.1), 0.2, True) == (
        False,
        "gain_below_threshold",
    )

    # One-sided pass
    assert predictive_gate_outcome(-0.3, (-0.3, -0.1), 0.05, True) == (
        True,
        "ci_gain_met",
    )


def test_predictive_gate_outcome_two_sided_paths():
    assert predictive_gate_outcome(-0.2, (-0.5, 0.1), 0.05, False) == (
        False,
        "ci_contains_zero",
    )
    assert predictive_gate_outcome(-0.2, (-0.1, -0.05), 0.2, False) == (
        False,
        "gain_below_threshold",
    )
    assert predictive_gate_outcome(-0.3, (-0.5, -0.2), 0.05, False) == (
        True,
        "ci_gain_met",
    )
