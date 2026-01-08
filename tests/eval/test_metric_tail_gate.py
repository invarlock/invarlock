import math


def test_tail_summary_quantiles_linear_interpolation():
    from invarlock.eval.tail_stats import compute_tail_summary

    deltas = [3.0, 0.0, 2.0, 1.0]
    summary = compute_tail_summary(deltas, quantiles=(0.5, 0.9, 0.95, 0.99), epsilon=1.5)

    assert summary["n"] == 4
    assert summary["q50"] == 1.5
    assert math.isclose(summary["q90"], 2.7, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["q95"], 2.85, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["q99"], 2.97, rel_tol=0, abs_tol=1e-12)
    assert summary["max"] == 3.0
    assert summary["tail_mass"] == 0.5


def test_tail_summary_tail_mass_weighted():
    from invarlock.eval.tail_stats import compute_tail_summary

    deltas = [0.0, 2.0, 3.0]
    weights = [1.0, 10.0, 10.0]
    summary = compute_tail_summary(deltas, quantiles=(0.5,), epsilon=1.5, weights=weights)

    assert summary["tail_mass"] == 2 / 3
    assert math.isclose(summary["tail_mass_weighted"], 20.0 / 21.0, rel_tol=0, abs_tol=1e-12)
    assert summary["tail_mass_weighted_by"] == "weights"


def test_metric_tail_gate_underpowered_is_not_evaluated():
    from invarlock.eval.tail_stats import evaluate_metric_tail

    result = evaluate_metric_tail(
        deltas=[0.1, 0.2],
        policy={
            "mode": "fail",
            "min_windows": 3,
            "quantile": 0.95,
            "quantile_max": 0.0,
            "epsilon": 0.0,
            "mass_max": 0.0,
        },
    )

    assert result["evaluated"] is False
    assert result["passed"] is True
