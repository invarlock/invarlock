import math

import pytest


def test_tail_summary_quantiles_linear_interpolation():
    from invarlock.eval.tail_stats import compute_tail_summary

    deltas = [3.0, 0.0, 2.0, 1.0]
    summary = compute_tail_summary(
        deltas, quantiles=(0.5, 0.9, 0.95, 0.99), epsilon=1.5
    )

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
    summary = compute_tail_summary(
        deltas, quantiles=(0.5,), epsilon=1.5, weights=weights
    )

    assert summary["tail_mass"] == 2 / 3
    assert math.isclose(
        summary["tail_mass_weighted"], 20.0 / 21.0, rel_tol=0, abs_tol=1e-12
    )
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


def test_linear_quantile_handles_edge_inputs() -> None:
    from invarlock.eval.tail_stats import _linear_quantile

    assert math.isnan(_linear_quantile([], 0.5))
    assert _linear_quantile([7.0], 0.5) == 7.0
    assert _linear_quantile([1.0, 2.0, 3.0], -1.0) == 1.0
    assert _linear_quantile([1.0, 2.0, 3.0], 2.0) == 3.0
    assert _linear_quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_tail_summary_handles_invalid_inputs_and_empty_result() -> None:
    from invarlock.eval.tail_stats import compute_tail_summary

    summary = compute_tail_summary(
        [object(), "nan", float("inf")],
        quantiles=(0.25, "bad", 1.5, -0.1),
        epsilon=-5,
        weights=[1.0, 2.0, 3.0],
    )

    assert summary["n"] == 0
    assert summary["epsilon"] == 0.0
    assert summary["max"] is None
    assert summary["tail_mass"] == 0.0
    assert summary["q25"] is None
    assert summary["q100"] is None
    assert summary["q0"] is None
    assert "tail_mass_weighted" not in summary


def test_tail_summary_clamps_invalid_weights_and_quantiles() -> None:
    from invarlock.eval.tail_stats import compute_tail_summary

    summary = compute_tail_summary(
        [0.0, 2.0, "skip", 5.0],
        quantiles=("bad", 0.0, 1.0),
        epsilon="oops",
        weights=[float("nan"), -2.0, 99.0, 3.0],
    )

    assert summary["epsilon"] == 0.0
    assert summary["n"] == 3
    assert summary["q0"] == 0.0
    assert summary["q100"] == 5.0
    assert summary["tail_mass"] == pytest.approx(2 / 3)
    assert summary["tail_mass_weighted"] == pytest.approx(1.0)
    assert summary["tail_mass_weighted_by"] == "weights"


def test_tail_summary_skips_invalid_unweighted_deltas() -> None:
    from invarlock.eval.tail_stats import compute_tail_summary

    summary = compute_tail_summary([0.0, "bad", None, 2.0], quantiles=(0.5,))

    assert summary["n"] == 2
    assert summary["q50"] == 1.0
    assert summary["tail_mass"] == pytest.approx(0.5)


def test_tail_summary_rejects_bool_numeric_inputs() -> None:
    from invarlock.eval.tail_stats import compute_tail_summary

    summary = compute_tail_summary(
        [True, 2.0],
        quantiles=(0.5,),
        epsilon=True,
        weights=[True, 2.0],
    )

    assert summary["n"] == 1
    assert summary["epsilon"] == 0.0
    assert summary["q50"] == 2.0
    assert summary["tail_mass"] == 1.0
    assert summary["tail_mass_weighted"] == 1.0


def test_metric_tail_gate_warns_and_normalizes_invalid_policy_values() -> None:
    from invarlock.eval.tail_stats import evaluate_metric_tail

    result = evaluate_metric_tail(
        deltas=[0.0, 1.0, 3.0],
        policy={
            "mode": "LOUD",
            "min_windows": object(),
            "quantile": "bad",
            "quantile_max": 0.5,
            "epsilon": -1,
            "mass_max": 2.0,
        },
    )

    assert result["mode"] == "warn"
    assert result["evaluated"] is True
    assert result["passed"] is False
    assert result["warned"] is True
    assert result["policy"] == {
        "mode": "warn",
        "min_windows": 1,
        "quantile": 0.95,
        "quantile_max": 0.5,
        "epsilon": 0.0,
        "mass_max": 1.0,
    }
    assert [v["type"] for v in result["violations"]] == ["quantile_max_exceeded"]


def test_metric_tail_gate_off_and_missing_thresholds_skip_evaluation() -> None:
    from invarlock.eval.tail_stats import evaluate_metric_tail

    off_result = evaluate_metric_tail(
        deltas=[1.0, 2.0, 3.0],
        policy={"mode": "off", "quantile_max": 0.1, "mass_max": 0.1},
    )
    no_threshold_result = evaluate_metric_tail(
        deltas=[1.0, 2.0, 3.0],
        policy={"mode": "fail"},
    )

    assert off_result["evaluated"] is False
    assert off_result["passed"] is True
    assert off_result["violations"] == []
    assert no_threshold_result["evaluated"] is False
    assert no_threshold_result["passed"] is True


def test_metric_tail_gate_fail_mode_reports_fail_without_warning() -> None:
    from invarlock.eval.tail_stats import evaluate_metric_tail

    result = evaluate_metric_tail(
        deltas=[0.1, 0.2, 0.3, 0.4],
        policy={"mode": "fail", "quantile": 0.5, "quantile_max": 0.15},
    )

    assert result["evaluated"] is True
    assert result["passed"] is False
    assert result["warned"] is False
    assert result["violations"] == [
        {
            "type": "quantile_max_exceeded",
            "quantile": 0.5,
            "observed": 0.25,
            "threshold": 0.15,
        }
    ]


def test_metric_tail_gate_tail_mass_violation_is_reported() -> None:
    from invarlock.eval.tail_stats import evaluate_metric_tail

    result = evaluate_metric_tail(
        deltas=[1.0, 1.0, 1.0],
        policy={"mode": "fail", "epsilon": 0.0, "mass_max": 0.5},
    )

    assert result["evaluated"] is True
    assert result["passed"] is False
    assert result["warned"] is False
    assert result["violations"] == [
        {
            "type": "tail_mass_exceeded",
            "epsilon": 0.0,
            "observed": 1.0,
            "threshold": 0.5,
        }
    ]


def test_metric_tail_gate_handles_non_finite_quantile_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.eval.tail_stats as tail_stats

    def fake_compute_tail_summary(*args: object, **kwargs: object) -> dict[str, object]:
        return {"n": 3, "q50": "not-finite", "tail_mass": 0.0}

    monkeypatch.setattr(tail_stats, "compute_tail_summary", fake_compute_tail_summary)
    result = tail_stats.evaluate_metric_tail(
        deltas=[1.0, 2.0, 3.0],
        policy={"mode": "fail", "quantile": 0.5, "quantile_max": 0.1},
    )

    assert result["evaluated"] is True
    assert result["passed"] is True
    assert result["violations"] == []


def test_metric_tail_gate_ignores_bool_policy_thresholds() -> None:
    from invarlock.eval.tail_stats import evaluate_metric_tail

    result = evaluate_metric_tail(
        deltas=[0.0, 1.0, 3.0],
        policy={
            "mode": "fail",
            "min_windows": True,
            "quantile": True,
            "quantile_max": True,
            "mass_max": False,
        },
    )

    assert result["evaluated"] is False
    assert result["passed"] is True
    assert result["policy"] == {
        "mode": "fail",
        "min_windows": 1,
        "quantile": 0.95,
        "quantile_max": None,
        "epsilon": 0.0001,
        "mass_max": None,
    }
