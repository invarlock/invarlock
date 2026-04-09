from __future__ import annotations

import math

import pytest

from invarlock.reporting.primary_metric_utils import (
    _attach_classification_primary_metric_fallback,
    _attach_ppl_analysis_fields,
    _attach_primary_metric_from_report,
    _ensure_primary_metric_display_ci,
    _finalize_primary_metric_snapshot,
    _resolve_logspace_ci,
    attach_primary_metric,
)


def test_attach_primary_metric_from_report_with_ppl_analysis():
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {"kind": "ppl_mlm", "final": 4.0},
            "logloss_delta_ci": (0.1, 0.2),
        },
        "evaluation_windows": {
            "preview": {"logloss": [1.0, 2.0], "token_counts": [10, 10]},
            "final": {"logloss": [2.0], "token_counts": [20]},
        },
    }
    baseline_ref = {"primary_metric": {"final": 2.0}}
    ppl_analysis = {"unstable": True}

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=ppl_analysis,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["analysis_basis"] == "mean_logloss"
    assert pm["analysis_point_preview"] == pytest.approx(1.5)
    assert pm["analysis_point_final"] == pytest.approx(2.0)
    assert pm["ratio_vs_baseline"] == pytest.approx(2.0)
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.1)),
        pytest.approx(math.exp(0.2)),
    ]
    assert pm["ci"] == [0.1, 0.2]
    assert pm["unstable"] is True


def test_attach_primary_metric_classification_fallback(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {"final": {"correct_total": 80, "total": 100}},
        },
        "meta": {"model_id": "awesome-vqa"},
    }
    baseline_raw = {
        "metrics": {"classification": {"final": {"correct_total": 70, "total": 100}}}
    }
    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=baseline_raw,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "vqa_accuracy"
    assert pm["final"] == pytest.approx(0.8)
    assert pm["display_ci"] == [pm["final"], pm["final"]]
    assert pm["ratio_vs_baseline"] == pytest.approx(10.0)


def test_attach_primary_metric_uses_report_windows(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {"metrics": {"loss_type": "mlm"}}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda report, kind, baseline: {"kind": kind, "final": 1.23},
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"] == {
        "kind": "ppl_mlm",
        "final": 1.23,
        "display_ci": [1.23, 1.23],
    }


def test_attach_primary_metric_display_ci_fallback(monkeypatch):
    evaluation_report = {"primary_metric": {"ratio_vs_baseline": 1.2}}
    report = {}
    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"]["display_ci"] == [1.2, 1.2]


def test_attach_primary_metric_marks_nonfinite_as_degraded():
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.2,
                "final": float("nan"),
                "ratio_vs_baseline": float("inf"),
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["degraded"] is True
    assert pm["degraded_reason"] == "non_finite_pm"


def test_attach_primary_metric_skips_ratio_nan_without_baseline():
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.2,
                "final": 1.2,
                "ratio_vs_baseline": float("nan"),
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["degraded"] is False
    assert "degraded_reason" not in pm


def test_attach_primary_metric_recomputes_ratio_without_marking_degraded():
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.2,
                "final": 1.8,
                "ratio_vs_baseline": float("nan"),
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref={"primary_metric": {"final": 1.5}},
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["ratio_vs_baseline"] == pytest.approx(1.2)
    assert pm["degraded"] is False
    assert "degraded_reason" not in pm


def test_attach_primary_metric_marks_non_finite_delta_when_baseline_is_zero() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.2,
                "final": 1.8,
                "ratio_vs_baseline": float("nan"),
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref={"primary_metric": {"final": 0.0}},
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["degraded"] is True
    assert pm["degraded_reason"] == "non_finite_delta"
    assert pm["display_ci"] == [1.8, 1.8]


def test_attach_primary_metric_retries_window_computation(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {"metrics": {"loss_type": "s2s"}}

    import invarlock.eval.primary_metric as pm_mod

    calls: list[str] = []

    def _fake_compute(report, *, kind, baseline):
        calls.append(kind)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"kind": kind, "final": 2.5}

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda report, kind, baseline: _fake_compute(
            report, kind=kind, baseline=baseline
        ),
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "ppl_seq2seq"
    assert pm["final"] == pytest.approx(2.5)
    assert pm["display_ci"] == [2.5, 2.5]
    assert calls == ["ppl_seq2seq", "ppl_seq2seq"]


def test_attach_primary_metric_classification_numeric_baseline_ref(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {
                "final": 0.65,
            },
        },
        "meta": {"model_id": "invarlock-base"},
    }
    baseline_ref = {
        "metrics": {
            "classification": {
                "final": 0.55,
            },
        }
    }

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert pm["final"] == pytest.approx(0.65)
    assert pm["ratio_vs_baseline"] == pytest.approx(10.0)


def test_attach_primary_metric_ignores_bool_baseline_reference() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 3.0,
                "final": 4.0,
            }
        }
    }
    baseline_ref = {"primary_metric": {"final": True}}

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert "ratio_vs_baseline" not in pm


def test_attach_primary_metric_replaces_bool_display_ci_with_numeric_fallback() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.5,
                "final": 2.0,
                "display_ci": [True, False],
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["display_ci"] == [2.0, 2.0]


def test_attach_primary_metric_classification_fallback_ignores_bool_baseline(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {"final": {"correct_total": 8, "total": 10}},
        },
        "meta": {"model_id": "awesome-vqa"},
    }
    baseline_raw = {"metrics": {"classification": {"final": True}}}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=baseline_raw,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["final"] == pytest.approx(0.8)
    assert "ratio_vs_baseline" not in pm
    assert pm["display_ci"] == [pytest.approx(0.8), pytest.approx(0.8)]


def test_attach_primary_metric_display_ci_default_when_no_numeric(monkeypatch):
    evaluation_report = {"primary_metric": {"kind": "mystery"}}
    report: dict[str, object] = {}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"]["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_handles_bad_ppl_analysis(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        }
    }
    baseline_ref = {"primary_metric": {"final": 1.0}}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    class Boom:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("bad")

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=Boom(),
    )

    pm = evaluation_report["primary_metric"]
    assert pm["ratio_vs_baseline"] == pytest.approx(2.0)


def test_attach_primary_metric_uses_metrics_ci_when_stats_is_non_mapping() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 2.0},
            "logloss_delta_ci": (0.1, 0.2),
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref={"primary_metric": {"final": 1.0}},
        ppl_analysis={"stats": "bad"},
    )

    pm = evaluation_report["primary_metric"]
    assert pm["ci"] == [0.1, 0.2]
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.1)),
        pytest.approx(math.exp(0.2)),
    ]


def test_attach_primary_metric_handles_bad_ppl_analysis_stats_without_ci() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 2.0,
            },
            "logloss_delta_ci": (0.1, 0.2),
        }
    }

    class _BoomStats(dict):
        def get(self, *_args, **_kwargs):  # type: ignore[override]
            raise RuntimeError("bad stats")

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref={"primary_metric": {"final": 1.0}},
        ppl_analysis={"stats": _BoomStats()},
    )

    pm = evaluation_report["primary_metric"]
    assert pm["ratio_vs_baseline"] == pytest.approx(2.0)
    assert pm["ci"] == [0.1, 0.2]
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.1)),
        pytest.approx(math.exp(0.2)),
    ]


def test_attach_primary_metric_classification_handles_non_numeric_final(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {
                # final payload lacks usable totals to force pm_point None
                "final": {"correct_total": "many", "total": "few"},
            },
        },
        "meta": {"model_id": "invarlock"},
    }

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert "final" not in pm
    assert pm["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_handles_bad_kind_and_ci_fallbacks() -> None:
    evaluation_report: dict[str, object] = {}

    class _BadKind:
        def __str__(self) -> str:
            raise RuntimeError("bad kind")

    report = {
        "metrics": {
            "primary_metric": {
                "kind": _BadKind(),
                "preview": 1.5,
                "final": 2.0,
                "ci": ["bad", "worse"],
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["final"] == 2.0
    assert pm["display_ci"] == [2.0, 2.0]


def test_attach_primary_metric_falls_back_when_ppl_ci_is_non_finite() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 2.0,
                "ci": [float("nan"), float("nan")],
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["display_ci"] == [2.0, 2.0]


def test_attach_primary_metric_skips_exploding_primary_metric_get() -> None:
    evaluation_report: dict[str, object] = {}

    class _ExplodingMetrics(dict):
        def get(self, key, default=None):  # type: ignore[override]
            if key == "primary_metric":
                raise RuntimeError("boom")
            return super().get(key, default)

    attach_primary_metric(
        evaluation_report,
        {"metrics": _ExplodingMetrics()},
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "ppl_causal"
    assert pm["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_recovers_when_display_ci_lookup_raises() -> None:
    evaluation_report: dict[str, object] = {}

    class _ExplodingDisplayCIDict(dict):
        def get(self, key, default=None):  # type: ignore[override]
            if key == "display_ci":
                raise RuntimeError("boom")
            return super().get(key, default)

    report = {
        "metrics": {
            "primary_metric": _ExplodingDisplayCIDict(
                {"kind": "ppl_causal", "final": 2.0}
            )
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["final"] == 2.0
    assert "display_ci" not in pm


def test_attach_primary_metric_classification_without_baseline(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {"final": {"correct_total": 55, "total": 100}},
        },
        "meta": {"model_id": "invarlock"},
    }

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw={},
        baseline_ref={},
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert pm["final"] == pytest.approx(0.55)
    assert "ratio_vs_baseline" not in pm


def test_attach_primary_metric_classification_bad_final_and_bad_baseline(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {"classification": {"final": "bad"}},
        "meta": {"model_id": "invarlock"},
    }
    baseline_raw = {"metrics": {"classification": {"final": "worse"}}}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=baseline_raw,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert "final" not in pm
    assert "ratio_vs_baseline" not in pm
    assert pm["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_leaves_non_dict_metric_when_classification_missing(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, object] = {"primary_metric": "bad"}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        {"metrics": {"classification": []}},
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"] == "bad"


def test_attach_primary_metric_tolerates_exploding_classification_lookup(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, object] = {"primary_metric": "bad"}

    class _ExplodingClassificationMetrics(dict):
        def get(self, key, default=None):  # type: ignore[override]
            if key == "classification":
                raise RuntimeError("boom")
            return super().get(key, default)

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        {"metrics": _ExplodingClassificationMetrics()},
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"] == "bad"


def test_attach_primary_metric_replaces_non_dict_existing_metric_via_classification(
    monkeypatch,
) -> None:
    evaluation_report: dict[str, object] = {"primary_metric": "bad"}
    report = {
        "metrics": {"classification": {"final": {"correct_total": 6, "total": 10}}},
        "meta": {"model_id": "invarlock"},
    }

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert isinstance(pm, dict)
    assert pm["kind"] == "accuracy"
    assert pm["final"] == pytest.approx(0.6)
    assert pm["display_ci"] == [pytest.approx(0.6), pytest.approx(0.6)]


def test_attach_primary_metric_invalid_reason_when_only_invalid_flag_present():
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 2.0,
                "final": 2.0,
                "ratio_vs_baseline": 1.0,
                "invalid": True,
            }
        }
    }
    baseline_ref = {"primary_metric": {"final": 2.0}}

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["degraded"] is True
    assert pm["degraded_reason"] == "primary_metric_invalid"


def test_resolve_logspace_ci_falls_back_when_stats_are_not_mapping() -> None:
    assert _resolve_logspace_ci(
        {"logloss_delta_ci": (0.1, 0.2)},
        {"stats": "bad", "logloss_delta_ci": (0.3, 0.4)},
    ) == (0.1, 0.2)


def test_attach_primary_metric_handles_non_finite_ci_values() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 2.0,
                "ci": [float("nan"), 0.2],
            }
        }
    }

    attach_primary_metric(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["display_ci"] == [2.0, 2.0]


def test_ensure_primary_metric_display_ci_recovers_from_exploding_display_lookup() -> (
    None
):
    class _ExplodingDisplayPM(dict):
        def get(self, key, default=None):  # type: ignore[override]
            if key == "display_ci":
                raise RuntimeError("boom")
            return super().get(key, default)

    evaluation_report = {
        "primary_metric": _ExplodingDisplayPM(
            {"kind": "ppl_causal", "final": 2.0, "preview": 1.5}
        )
    }

    _ensure_primary_metric_display_ci(evaluation_report)

    assert "display_ci" not in evaluation_report["primary_metric"]


def test_attach_classification_primary_metric_fallback_skips_non_dict_input() -> None:
    evaluation_report: dict[str, object] = {}
    _attach_classification_primary_metric_fallback(
        evaluation_report,
        {"metrics": {"classification": []}},
        baseline_raw=None,
        baseline_ref=None,
    )

    assert evaluation_report == {}


def test_attach_classification_primary_metric_fallback_swallows_report_get_errors() -> (
    None
):
    evaluation_report: dict[str, object] = {}

    class _ExplodingReport(dict):
        def get(self, key, default=None):  # type: ignore[override]
            if key == "metrics":
                raise RuntimeError("boom")
            return super().get(key, default)

    _attach_classification_primary_metric_fallback(
        evaluation_report,
        _ExplodingReport(),
        baseline_raw=None,
        baseline_ref=None,
    )

    assert evaluation_report == {}


def test_primary_metric_private_helpers_cover_error_and_fallback_paths() -> None:
    class _BoomDict(dict):
        def get(self, key, default=None):  # type: ignore[override]
            raise RuntimeError(f"boom:{key}")

    assert _resolve_logspace_ci(_BoomDict(), None) is None

    pm_copy = {"kind": "ppl_causal"}
    _attach_ppl_analysis_fields(
        pm_copy,
        report=_BoomDict(),
        metrics_map={},
        ppl_analysis=None,
    )
    assert "ci" not in pm_copy

    finalized = _finalize_primary_metric_snapshot(
        {
            "kind": "ppl_causal",
            "preview": 1.0,
            "final": 2.0,
            "ci": [object(), object()],
        },
        report={},
        metrics_map={},
        baseline_ref={"primary_metric": {"final": 1.0}},
        ppl_analysis=_BoomDict(),
    )
    assert finalized["ratio_vs_baseline"] == pytest.approx(2.0)
    assert finalized["display_ci"] == [2.0, 2.0]

    evaluation_report: dict[str, object] = {}
    _attach_primary_metric_from_report(
        evaluation_report,
        _BoomDict(),
        baseline_ref=None,
        ppl_analysis=None,
    )
    assert evaluation_report == {}

    _ensure_primary_metric_display_ci({})

    class _ExplodingReport(dict):
        def get(self, key, default=None):  # type: ignore[override]
            raise RuntimeError("explode")

    _ensure_primary_metric_display_ci(_ExplodingReport())


def test_attach_classification_primary_metric_fallback_handles_bad_payload_shapes() -> (
    None
):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {"classification": {"final": "bad-shape"}},
        "meta": {"model_id": "invarlock"},
    }

    _attach_classification_primary_metric_fallback(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert "final" not in pm


def test_attach_classification_primary_metric_fallback_skips_bad_baseline_ratio() -> (
    None
):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {"classification": {"final": 0.8}},
        "meta": {"model_id": "invarlock"},
    }
    baseline_ref = {
        "metrics": {"classification": {"final": {"correct_total": 1, "total": 0}}}
    }

    _attach_classification_primary_metric_fallback(
        evaluation_report,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["final"] == pytest.approx(0.8)
    assert "ratio_vs_baseline" not in pm
