from __future__ import annotations

import pytest

import invarlock.reporting.primary_metric_utils as primary_metric_utils
from invarlock.reporting.primary_metric_utils import (
    _attach_classification_primary_metric_fallback,
    _attach_ppl_analysis_fields,
    _attach_primary_metric_from_report,
    _classification_final_counts,
    _ensure_primary_metric_display_ci,
    _finalize_primary_metric_snapshot,
    _resolve_logspace_ci,
    _wilson_accuracy_ci,
    attach_primary_metric,
)


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


def test_classification_final_counts_handles_example_and_invalid_paths() -> None:
    assert _classification_final_counts({"classification": "bad"}) is None
    assert _classification_final_counts({"classification": {"final": "bad"}}) is None
    assert _classification_final_counts(
        {"classification": {"final": {"example_correct": [True, False, 1]}}}
    ) == (2, 3)
    assert _classification_final_counts(
        {
            "classification": {
                "final": {"total": 5, "example_correct": [True, False, True]}
            }
        }
    ) == (2, 5)
    assert (
        _classification_final_counts(
            {"classification": {"final": {"example_correct": "bad"}}}
        )
        is None
    )
    assert (
        _classification_final_counts({"classification": {"final": {"total": 3}}})
        is None
    )
    assert (
        _classification_final_counts(
            {"classification": {"final": {"correct_total": 4, "total": 3}}}
        )
        is None
    )

    class _BrokenMetrics(dict):
        def get(self, key, default=None):  # type: ignore[override]
            raise RuntimeError(f"broken:{key}")

    assert _classification_final_counts(_BrokenMetrics()) is None


def test_wilson_accuracy_ci_invalid_and_exception_paths(monkeypatch) -> None:
    assert _wilson_accuracy_ci(0, 0) is None
    assert _wilson_accuracy_ci(-1, 10) is None
    assert _wilson_accuracy_ci(11, 10) is None

    monkeypatch.setattr(primary_metric_utils.math, "sqrt", lambda _value: "bad")
    assert _wilson_accuracy_ci(1, 2) is None


def test_finalize_accuracy_primary_metric_falls_back_without_usable_counts() -> None:
    pm = _finalize_primary_metric_snapshot(
        {"kind": "accuracy", "final": 0.7},
        report={},
        metrics_map={"classification": {"final": {"correct_total": 1, "total": 0}}},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert pm["display_ci"] == [0.7, 0.7]


def test_finalize_accuracy_primary_metric_falls_back_when_wilson_ci_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        primary_metric_utils,
        "_wilson_accuracy_ci",
        lambda _correct, _total: None,
    )

    pm = _finalize_primary_metric_snapshot(
        {"kind": "accuracy", "final": 0.6},
        report={},
        metrics_map={"classification": {"final": {"correct_total": 6, "total": 10}}},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert pm["display_ci"] == [0.6, 0.6]


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
    assert pm["display_ci"] == [2.0, 2.0]
    assert evaluation_report["report_build"]["synthesized_fields"] == [
        {
            "field": "primary_metric.display_ci",
            "reason": "computed_from_primary_metric_point",
            "source": "primary_metric_utils._attach_primary_metric_from_report",
        }
    ]


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


def test_attach_accuracy_primary_metric_uses_classification_count_ci(monkeypatch):
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.55,
                "ratio_vs_baseline": 0.0,
            },
            "classification": {
                "final": {"correct_total": 55, "total": 100},
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
        baseline_raw={},
        baseline_ref={},
        ppl_analysis=None,
    )

    pm = evaluation_report["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert pm["ci"][0] < 0.55 < pm["ci"][1]
    assert pm["display_ci"] == pm["ci"]
    assert evaluation_report["report_build"]["synthesized_fields"] == [
        {
            "field": "primary_metric.display_ci",
            "reason": "computed_from_primary_metric_ci",
            "source": "primary_metric_utils._attach_primary_metric_from_report",
        }
    ]


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


def test_finalize_primary_metric_snapshot_ignores_display_ci_write_errors() -> None:
    class _DisplayWriteFails(dict):
        def __setitem__(self, key, value):  # type: ignore[override]
            if key == "display_ci":
                raise RuntimeError("display write failed")
            return super().__setitem__(key, value)

    out = _finalize_primary_metric_snapshot(
        _DisplayWriteFails({"kind": "accuracy", "final": 2.0}),
        report={},
        metrics_map={},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert out["final"] == 2.0
    assert "display_ci" not in out


def test_attach_primary_metric_from_report_recovers_from_final_display_lookup_error(
    monkeypatch,
) -> None:
    class _FinalMetric(dict):
        def __contains__(self, key):  # type: ignore[override]
            return key == "display_ci" or super().__contains__(key)

        def __getitem__(self, key):  # type: ignore[override]
            if key == "display_ci":
                raise RuntimeError("display lookup failed")
            return super().__getitem__(key)

    def _finalize_stub(*_args, **_kwargs):
        return _FinalMetric({"kind": "accuracy", "final": 0.8})

    monkeypatch.setattr(
        primary_metric_utils,
        "_finalize_primary_metric_snapshot",
        _finalize_stub,
    )

    evaluation_report: dict[str, object] = {}
    _attach_primary_metric_from_report(
        evaluation_report,
        {"metrics": {"primary_metric": {"kind": "accuracy", "final": 0.8}}},
        baseline_ref=None,
        ppl_analysis=None,
    )

    assert evaluation_report["primary_metric"]["final"] == 0.8


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
