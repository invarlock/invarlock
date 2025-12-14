from __future__ import annotations

import math

from invarlock.calibration import spectral_null, variance_ve


def test_spectral_null_helpers_cover_edge_cases() -> None:
    assert spectral_null._finite01(0.0)
    assert spectral_null._finite01(1.0)
    assert spectral_null._finite01(0.5)
    assert spectral_null._finite01("0.5")
    assert not spectral_null._finite01(-0.1)
    assert not spectral_null._finite01(1.1)
    assert not spectral_null._finite01(float("nan"))
    assert not spectral_null._finite01("nope")

    assert spectral_null._bh_reject_families({}, alpha=0.05, m=4) == set()
    assert spectral_null._bh_reject_families({"ffn": 0.01}, alpha="bad", m=4) == set()
    assert spectral_null._bh_reject_families({"ffn": 0.01}, alpha=-0.1, m=4) == set()
    assert (
        spectral_null._bh_reject_families({"ffn": float("nan")}, alpha=0.05, m=4)
        == set()
    )

    selected = spectral_null._bh_reject_families(
        {"a": 0.01, "b": 0.5, "c": float("nan")}, alpha=0.05, m=4
    )
    assert selected == {"a"}

    assert spectral_null._bonferroni_reject_families({}, alpha=0.05, m=4) == set()
    assert (
        spectral_null._bonferroni_reject_families({"a": 0.01}, alpha="bad", m=4)
        == set()
    )
    assert spectral_null._bonferroni_reject_families({"a": 0.01}, alpha=0.05, m=2) == {
        "a"
    }
    assert (
        spectral_null._bonferroni_reject_families({"a": 0.01}, alpha=2.0, m=2) == set()
    )

    metrics = {
        "family_z_summary": {"ffn": {"max": 2.0}, "bad": "x"},
        "family_z_quantiles": {"ffn": {"max": 3.0}, "attn": {"max": float("nan")}},
    }
    assert spectral_null._extract_family_max_z(metrics) == {"ffn": 3.0}

    assert spectral_null._extract_multiple_testing({"multiple_testing": "nope"}) == {}
    mt = spectral_null._extract_multiple_testing(
        {"multiple_testing": {"method": "Bonferroni", "alpha": "0.1", "m": "4"}}
    )
    assert mt["method"] == "bonferroni"
    assert math.isclose(mt["alpha"], 0.1)
    assert mt["m"] == 4

    assert spectral_null._selected_families_for_alpha(
        {"a": 0.01}, method="bonferroni", alpha=0.05, m=4
    ) == {"a"}


def test_spectral_null_summary_covers_additional_parse_branches() -> None:
    reports = [
        "not a report",
        {"guards": [{"name": "other", "metrics": {}}]},
        {
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": 0,
                        "caps_exceeded": False,
                        "multiple_testing": {"method": "", "alpha": "bad", "m": "bad"},
                        "family_z_summary": "nope",
                        "family_z_quantiles": {"ffn": "nope", "attn": {"max": "bad"}},
                        "multiple_testing_selection": {
                            "family_pvalues": {"ffn": "bad", "attn": 2.0},
                            "family_violation_counts": "nope",
                            "families_selected": ["ffn"],
                        },
                    },
                    "violations": [],
                }
            ]
        },
    ]

    summary = spectral_null.summarize_null_sweep_reports(
        reports,
        tier="balanced",
        safety_margin=0.0,
        target_any_warning_rate=1.0,
    )
    assert summary["recommendations"]["multiple_testing"]["alpha"] == 0.05


def test_spectral_null_summary_handles_no_solution_for_alpha_target() -> None:
    reports = [
        {
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": 1,
                        "caps_exceeded": False,
                        "family_z_summary": {"ffn": {"max": 1.0}},
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                        "multiple_testing_selection": {
                            "family_pvalues": {"ffn": 0.0},
                            "families_selected": ["ffn"],
                        },
                    },
                    "violations": [{"family": "ffn"}],
                }
            ],
        }
    ]
    summary = spectral_null.summarize_null_sweep_reports(
        reports,
        tier="balanced",
        safety_margin=0.0,
        target_any_warning_rate=0.0,
    )
    assert summary["recommendations"]["multiple_testing"]["alpha"] == 0.05


def test_spectral_null_summary_recommends_kappa_and_alpha() -> None:
    reports = [
        {
            "meta": {"seeds": {"python": 1}, "auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": 1,
                        "caps_exceeded": False,
                        "family_z_summary": {"ffn": {"max": 4.0}},
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                        "multiple_testing_selection": {
                            "family_pvalues": {"ffn": 0.001},
                            "families_selected": ["ffn"],
                            "family_violation_counts": {"ffn": 1},
                        },
                    },
                    "violations": [{"family": "ffn", "type": "family_z_cap"}],
                }
            ],
        },
        {
            "meta": {"seeds": {"python": 2}, "auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": 0,
                        "caps_exceeded": False,
                        "family_z_summary": {"ffn": {"max": 3.0}},
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                        "multiple_testing_selection": {
                            "family_pvalues": {},
                            "families_selected": [],
                            "family_violation_counts": {},
                        },
                    },
                    "violations": [],
                }
            ],
        },
    ]

    summary = spectral_null.summarize_null_sweep_reports(
        reports,
        tier="balanced",
        safety_margin=0.05,
        target_any_warning_rate=0.49,
    )

    rec = summary["recommendations"]
    assert rec["family_caps"]["ffn"] == 4.2
    assert 0.0 < rec["multiple_testing"]["alpha"] <= 0.05


def test_spectral_null_summary_halves_alpha_to_meet_target_rate() -> None:
    reports = []
    for seed in range(5):
        reports.append(
            {
                "meta": {"seeds": {"python": seed}, "auto": {"tier": "balanced"}},
                "guards": [
                    {
                        "name": "spectral",
                        "metrics": {
                            "caps_applied": 1,
                            "caps_exceeded": False,
                            "family_z_summary": {"ffn": {"max": 1.0}},
                            "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                            "multiple_testing_selection": {
                                # Selected at alpha=0.05 (0.01 <= 0.0125), rejected at alpha=0.025.
                                "family_pvalues": {"ffn": 0.01},
                                "families_selected": ["ffn"],
                                "family_violation_counts": {"ffn": 1},
                            },
                        },
                        "violations": [{"family": "ffn"}],
                    }
                ],
            }
        )

    summary = spectral_null.summarize_null_sweep_reports(
        reports,
        tier="balanced",
        safety_margin=2.0,  # invalid → fallback to default
        target_any_warning_rate=-1.0,  # invalid → fallback to default
    )
    rec = summary["recommendations"]
    assert rec["multiple_testing"]["alpha"] == 0.025


def test_variance_ve_helpers_cover_edge_cases() -> None:
    assert variance_ve._coerce_delta_ci((-0.1, 0.2)) == (-0.1, 0.2)
    assert variance_ve._coerce_delta_ci([0.1]) is None
    assert variance_ve._coerce_delta_ci(["a", "b"]) is None
    assert variance_ve._coerce_delta_ci([float("nan"), 0.0]) is None

    assert (
        variance_ve._gain_lower_bound(mean_delta=None, delta_ci=None, one_sided=True)
        == 0.0
    )
    assert (
        variance_ve._gain_lower_bound(
            mean_delta=-0.01, delta_ci=(-0.02, 0.0), one_sided=True
        )
        == 0.0
    )
    assert (
        variance_ve._gain_lower_bound(
            mean_delta=0.0, delta_ci=(-0.02, -0.01), one_sided=True
        )
        == 0.0
    )
    assert (
        variance_ve._gain_lower_bound(
            mean_delta=None, delta_ci=(-0.02, -0.01), one_sided=False
        )
        == 0.01
    )

    thr, rate = variance_ve._recommend_threshold_for_target_rate(
        [], target_rate=0.1, safety_margin=0.0
    )
    assert thr == 0.0 and rate == 0.0

    thr0, rate0 = variance_ve._recommend_threshold_for_target_rate(
        [0.1, 0.1, 0.1], target_rate=0.0, safety_margin=0.1
    )
    assert thr0 == 0.11 and rate0 == 0.0

    thr1, rate1 = variance_ve._recommend_threshold_for_target_rate(
        [0.1, 0.1, 0.1], target_rate=0.4, safety_margin=0.0
    )
    assert thr1 == 0.1 and rate1 == 0.0

    thr2, rate2 = variance_ve._recommend_threshold_for_target_rate(
        [0.2, 0.1, 0.0], target_rate=1.0 / 3.0, safety_margin=0.0
    )
    assert thr2 == 0.2 and rate2 <= (1.0 / 3.0) + 1e-9


def test_variance_ve_recommend_min_effect_hits_target_enable_rate() -> None:
    reports = [
        {
            "meta": {"seeds": {"python": 1}, "auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "mean_delta": -0.015,
                            "delta_ci": (-0.02, -0.01),
                        }
                    },
                }
            ],
        },
        {
            "meta": {"seeds": {"python": 2}, "auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "mean_delta": -0.005,
                            "delta_ci": (-0.02, 0.005),
                        }
                    },
                }
            ],
        },
        {
            "meta": {"seeds": {"python": 3}, "auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "mean_delta": -0.035,
                            "delta_ci": (-0.04, -0.03),
                        }
                    },
                }
            ],
        },
    ]

    summary = variance_ve.summarize_ve_sweep_reports(
        reports,
        tier="balanced",
        target_enable_rate=1.0 / 3.0,
        safety_margin=0.0,
        predictive_one_sided=True,
    )
    assert summary["recommendations"]["min_effect_lognll"] == 0.03
    assert summary["recommendations"]["expected_enable_rate"] <= (1.0 / 3.0) + 1e-9


def test_variance_ve_summary_skips_missing_predictive_gate() -> None:
    reports = [
        {"guards": [{"name": "variance", "metrics": {"predictive_gate": "nope"}}]},
        {"guards": [{"name": "other", "metrics": {}}]},
        {
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {"evaluated": True, "delta_ci": (-1, -1)}
                    },
                }
            ]
        },
    ]
    summary = variance_ve.summarize_ve_sweep_reports(reports, tier="balanced")
    assert summary["n_runs"] == 1
