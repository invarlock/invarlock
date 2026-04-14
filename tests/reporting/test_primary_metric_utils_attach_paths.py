from __future__ import annotations

import math

import pytest

from invarlock.reporting.primary_metric_utils import (
    _attach_ppl_analysis_fields,
    _attach_primary_metric_from_windows,
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


def test_resolve_logspace_ci_prefers_ppl_analysis_pairing_and_falls_back():
    assert _resolve_logspace_ci(
        {"logloss_delta_ci": (0.3, 0.4)},
        {"stats": {"pairing": "paired_baseline"}, "logloss_delta_ci": (0.1, 0.2)},
    ) == (0.1, 0.2)
    assert _resolve_logspace_ci(
        {"logloss_delta_ci": (0.3, 0.4)},
        {"stats": {"pairing": "independent"}, "logloss_delta_ci": (0.1, 0.2)},
    ) == (0.3, 0.4)


def test_attach_ppl_analysis_fields_populates_mean_logloss_and_ci():
    pm_copy = {"kind": "ppl_causal"}
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [1.0, 2.0], "token_counts": [1, 1]},
            "final": {"logloss": [3.0], "token_counts": [2]},
        }
    }
    _attach_ppl_analysis_fields(
        pm_copy,
        report=report,
        metrics_map={"logloss_delta_ci": (0.1, 0.2)},
        ppl_analysis={
            "stats": {"pairing": "paired_baseline"},
            "logloss_delta_ci": (0.1, 0.2),
        },
    )

    assert pm_copy["analysis_basis"] == "mean_logloss"
    assert pm_copy["analysis_point_preview"] == pytest.approx(1.5)
    assert pm_copy["analysis_point_final"] == pytest.approx(3.0)
    assert pm_copy["ci"] == [0.1, 0.2]


def test_attach_ppl_analysis_fields_skips_non_finite_window_means():
    pm_preview = {"kind": "ppl_causal"}
    pm_final = {"kind": "ppl_causal"}
    base_report = {
        "evaluation_windows": {
            "preview": {"logloss": [], "token_counts": []},
            "final": {"logloss": [3.0], "token_counts": [2]},
        }
    }
    _attach_ppl_analysis_fields(
        pm_preview,
        report=base_report,
        metrics_map={},
        ppl_analysis=None,
    )
    assert "analysis_point_preview" not in pm_preview
    assert pm_preview["analysis_point_final"] == pytest.approx(3.0)

    _attach_ppl_analysis_fields(
        pm_final,
        report={
            "evaluation_windows": {
                "preview": {"logloss": [1.0, 2.0], "token_counts": [1, 1]},
                "final": {"logloss": [], "token_counts": []},
            }
        },
        metrics_map={},
        ppl_analysis=None,
    )
    assert pm_final["analysis_point_preview"] == pytest.approx(1.5)
    assert "analysis_point_final" not in pm_final


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
    assert pm["kind"] == "accuracy"
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


def test_attach_primary_metric_from_windows_uses_seq2seq_kind(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {"metrics": {"loss_type": "seq2seq"}}
    seen: dict[str, object] = {}

    import invarlock.eval.primary_metric as pm_mod

    def _stub(report, kind, baseline):  # noqa: ANN001
        seen["kind"] = kind
        seen["baseline"] = baseline
        return {"kind": kind, "final": 1.23}

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        _stub,
        raising=False,
    )

    _attach_primary_metric_from_windows(evaluation_report, report, {"x": 1})

    assert seen["kind"] == "ppl_seq2seq"
    assert seen["baseline"] == {"x": 1}
    assert evaluation_report["primary_metric"] == {
        "kind": "ppl_seq2seq",
        "final": 1.23,
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


def test_attach_primary_metric_display_ci_defaults_and_marks_estimated():
    evaluation_report = {"primary_metric": {"kind": "ppl_causal"}}

    _ensure_primary_metric_display_ci(evaluation_report)

    assert evaluation_report["primary_metric"]["display_ci"] == [1.0, 1.0]
    assert evaluation_report["primary_metric"]["estimated"] is True


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


def test_finalize_primary_metric_snapshot_marks_non_finite_delta():
    pm_copy = {
        "kind": "ppl_causal",
        "preview": 1.2,
        "final": 1.8,
        "ratio_vs_baseline": float("nan"),
    }

    out = _finalize_primary_metric_snapshot(
        pm_copy,
        report={},
        metrics_map={},
        baseline_ref={"primary_metric": {"final": 0.0}},
        ppl_analysis=None,
    )

    assert out["degraded"] is True
    assert out["degraded_reason"] == "non_finite_delta"


def test_finalize_primary_metric_snapshot_preserves_existing_degraded_reason():
    pm_copy = {
        "kind": "ppl_causal",
        "preview": 1.2,
        "final": 1.8,
        "ratio_vs_baseline": 1.5,
        "degraded_reason": "preset",
    }

    out = _finalize_primary_metric_snapshot(
        pm_copy,
        report={},
        metrics_map={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert out["degraded"] is True
    assert out["degraded_reason"] == "preset"


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


def test_finalize_primary_metric_snapshot_marks_primary_metric_invalid():
    pm_copy = {
        "kind": "ppl_causal",
        "preview": 1.2,
        "final": 1.8,
        "ratio_vs_baseline": 1.5,
        "invalid": True,
    }

    out = _finalize_primary_metric_snapshot(
        pm_copy,
        report={},
        metrics_map={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert out["degraded"] is True
    assert out["degraded_reason"] == "primary_metric_invalid"


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
