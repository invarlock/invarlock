from __future__ import annotations

from invarlock.reporting import guards_spectral


def test_spectral_quantile_and_summary_skip_malformed_observations() -> None:
    assert guards_spectral._compute_quantile([], 0.5) == 0.0
    assert guards_spectral._summarize_from_z_scores({}, {}) == ({}, {})
    quantiles, top = guards_spectral._summarize_from_z_scores(
        {"missing": 9, "bad": "no", "m1": -2, "m2": 4},
        {"bad": "attention", "m1": "attention", "m2": "attention"},
    )
    assert quantiles["attention"]["max"] == 4.0
    assert [entry["module"] for entry in top["attention"]] == ["m2", "m1"]


def test_guard_summary_normalizes_malformed_family_statistics() -> None:
    summary, quantiles, families, caps, top, deadband = (
        guards_spectral._build_guard_summary(
            guard_metrics={
                "deadband": "bad",
                "stability_score": "bad",
                "family_stats": {
                    "attention": {
                        "max": "bad",
                        "mean": 2,
                        "count": "bad",
                        "violations": 1,
                        "kappa": "bad",
                    },
                    "noise": "bad",
                },
                "top_z_scores": {
                    "attention": [
                        "noise",
                        {"module": "bad", "z": "bad"},
                        {"module": "m1", "z": -3},
                        {"module": "m2", "z": 2},
                    ],
                    "noise": "bad",
                },
            },
            guard_policy={"sigma_quantile": "bad"},
            default_deadband="bad",
            default_sigma_quantile=0.95,
            default_caps={"attention": {"kappa": 2.0}},
            max_sigma_ratio=1.0,
            median_sigma_ratio=1.0,
            max_spectral_norm=2.0,
            mean_spectral_norm=1.0,
            baseline_max=None,
            baseline_mean=None,
        )
    )
    assert deadband is None
    assert "sigma_quantile" not in summary
    assert families == {"attention": {"mean": 2.0, "violations": 1}}
    assert caps == {"attention": {"kappa": 2.0}}
    assert quantiles == {}
    assert [entry["module"] for entry in top["attention"]] == ["m1", "m2"]


def test_z_score_derivation_only_fills_missing_families() -> None:
    quantiles, top = guards_spectral._derive_z_score_tables(
        {
            "final_z_scores": {"m1": 2.0, "m2": 3.0},
            "module_family_map": {"m1": "attention", "m2": "mlp"},
        },
        {},
        {},
        {"attention": [{"module": "retained", "z": 9.0}]},
    )
    assert set(quantiles) == {"attention", "mlp"}
    assert top["attention"][0]["module"] == "retained"
    assert top["mlp"][0]["module"] == "m2"


def test_metrics_only_ratio_fallback_is_fail_safe() -> None:
    summary: dict = {}
    guards_spectral._apply_metrics_only_ratio_fallback(
        {"metrics": {"spectral": {"sigma_ratios": [1, 3, 2]}}},  # type: ignore[arg-type]
        {},
        summary,
    )
    assert summary == {"max_sigma_ratio": 3.0, "median_sigma_ratio": 2.0}

    malformed: dict = {}
    guards_spectral._apply_metrics_only_ratio_fallback(
        {"metrics": {"spectral": {"sigma_ratios": ["bad"]}}},  # type: ignore[arg-type]
        {},
        malformed,
    )
    assert malformed == {}


def test_spectral_policy_and_violation_outputs_ignore_invalid_records() -> None:
    assert guards_spectral._build_policy_output({}, 0.95, None, "balanced") is None
    policy = guards_spectral._build_policy_output(
        {"sigma_quantile": "bad"},
        0.95,
        {"method": "bh"},
        "balanced",
    )
    assert policy == {
        "sigma_quantile": "bad",
        "multiple_testing": {"method": "bh"},
    }
    assert guards_spectral._build_top_violations(None) is None
    assert guards_spectral._build_top_violations(
        {
            "violations": [
                "noise",
                {"module": "m", "z_score": "bad"},
                {"module": "m2", "z_score": 3},
            ]
        }
    ) == [
        {"module": "m", "family": None, "kappa": None, "severity": "warn"},
        {
            "module": "m2",
            "family": None,
            "kappa": None,
            "severity": "warn",
            "z_score": 3.0,
        },
    ]


def test_spectral_result_handles_invalid_multiple_testing_counts() -> None:
    summary: dict = {}
    result = guards_spectral._build_spectral_result(
        "balanced",
        {"violations": []},
        {},
        {},
        {},
        1,
        summary,
        {"attention": {"violations": 1}},
        {},
        {},
        {},
        {"m": "bad"},
        None,
        0.95,
        None,
        5,
        False,
        "bad",
    )
    assert result["bh_family_count"] == 1
    assert result["caps_applied_by_family"] == {"attention": 1}
    assert summary["status"] == "capped"
    assert "modules_checked" not in summary
