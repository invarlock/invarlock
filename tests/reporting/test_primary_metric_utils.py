from __future__ import annotations

import math

import pytest

from invarlock.reporting.primary_metric_utils import attach_primary_metric


def test_attach_primary_metric_from_report_with_ppl_analysis():
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=ppl_analysis,
    )

    pm = certificate["primary_metric"]
    assert pm["analysis_basis"] == "mean_logloss"
    assert pm["analysis_point_preview"] == pytest.approx(1.5)
    assert pm["analysis_point_final"] == pytest.approx(2.0)
    assert pm["ratio_vs_baseline"] == pytest.approx(2.0)
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.1)),
        pytest.approx(math.exp(0.2)),
    ]
    assert pm["ci"] == (0.1, 0.2)
    assert pm["unstable"] is True


def test_attach_primary_metric_classification_fallback(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=baseline_raw,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["kind"] == "vqa_accuracy"
    assert pm["final"] == pytest.approx(0.8)
    assert pm["display_ci"] == [pm["final"], pm["final"]]
    assert pm["ratio_vs_baseline"] == pytest.approx(10.0)


def test_attach_primary_metric_uses_report_windows(monkeypatch):
    certificate: dict[str, object] = {}
    report = {"metrics": {"loss_type": "mlm"}}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda report, kind, baseline: {"kind": kind, "final": 1.23},
        raising=False,
    )

    attach_primary_metric(
        certificate,
        report,
        baseline_raw={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert certificate["primary_metric"] == {
        "kind": "ppl_mlm",
        "final": 1.23,
        "display_ci": [1.23, 1.23],
    }


def test_attach_primary_metric_display_ci_fallback(monkeypatch):
    certificate = {"primary_metric": {"ratio_vs_baseline": 1.2}}
    report = {}
    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert certificate["primary_metric"]["display_ci"] == [1.2, 1.2]


def test_attach_primary_metric_marks_nonfinite_as_degraded():
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["degraded"] is True
    assert pm["degraded_reason"] == "non_finite_pm"


def test_attach_primary_metric_skips_ratio_nan_without_baseline():
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["degraded"] is False
    assert "degraded_reason" not in pm


def test_attach_primary_metric_retries_window_computation(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["kind"] == "ppl_seq2seq"
    assert pm["final"] == pytest.approx(2.5)
    assert pm["display_ci"] == [2.5, 2.5]
    assert calls == ["ppl_seq2seq", "ppl_seq2seq"]


def test_attach_primary_metric_classification_numeric_baseline_ref(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert pm["final"] == pytest.approx(0.65)
    assert pm["ratio_vs_baseline"] == pytest.approx(10.0)
    assert pm["display_ci"] == [pytest.approx(0.65), pytest.approx(0.65)]


def test_attach_primary_metric_display_ci_default_when_no_numeric(monkeypatch):
    certificate = {"primary_metric": {"kind": "mystery"}}
    report: dict[str, object] = {}

    import invarlock.eval.primary_metric as pm_mod

    monkeypatch.setattr(
        pm_mod,
        "compute_primary_metric_from_report",
        lambda *_, **__: None,
        raising=False,
    )

    attach_primary_metric(
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert certificate["primary_metric"]["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_handles_bad_ppl_analysis(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        ppl_analysis=Boom(),
    )

    pm = certificate["primary_metric"]
    assert pm["ratio_vs_baseline"] == pytest.approx(2.0)


def test_attach_primary_metric_classification_handles_non_numeric_final(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw=None,
        baseline_ref=None,
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert "final" not in pm
    assert pm["display_ci"] == [1.0, 1.0]


def test_attach_primary_metric_classification_without_baseline(monkeypatch):
    certificate: dict[str, object] = {}
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
        certificate,
        report,
        baseline_raw={},
        baseline_ref={},
        ppl_analysis=None,
    )

    pm = certificate["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert pm["final"] == pytest.approx(0.55)
    assert "ratio_vs_baseline" not in pm
