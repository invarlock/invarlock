from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.reporting.run_report_metrics_contract import (
    enrich_run_report_metrics,
)


def test_enrich_run_report_metrics_adds_classification_primary_metric_and_stats(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "invarlock.eval.primary_metric.compute_primary_metric_from_report",
        lambda report, *, kind, baseline=None: {
            "kind": kind,
            "preview": 9.0,
            "final": 10.0,
            "ratio_vs_baseline": 2.0,
        },
    )

    report = {
        "metrics": {
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "primary_metric": {"final": 8.0},
        },
        "data": {"preview_n": 2, "final_n": 2},
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2], [3, 4]]},
            "final": {"input_ids": [[5, 6], [7, 8]]},
        },
    }
    core_report = SimpleNamespace(
        evaluation_windows=report["evaluation_windows"],
        metrics={"primary_metric": {"degraded": False}},
    )
    run_config = SimpleNamespace(
        context={"eval": {"loss": {"resolved_type": "classification"}}}
    )
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=2, final_n=2))

    result = enrich_run_report_metrics(
        report=report,
        core_report=core_report,
        run_config=run_config,
        cfg=cfg,
        model_profile=SimpleNamespace(),
        baseline_requested=False,
        baseline_report_data={"metrics": {"primary_metric": {"final": 5.0}}},
        metric_kind="ppl_causal",
        resolved_loss_type="classification",
        effective_preview=2,
        effective_final=2,
        profile_normalized="dev",
        window_plan={
            "coverage_ok": True,
            "preview_total_tokens": 4,
            "final_total_tokens": 4,
            "min_tokens_target": 4,
            "tokens_floor_met": True,
        },
        debug_metric_diffs_enabled=True,
        resolve_metric_and_provider_fn=lambda *args, **kwargs: (
            "ppl_causal",
            None,
            {"reps": 10, "ci_level": 0.95},
        ),
    )

    assert result.pairing_violations == ()
    assert result.debug_diffs_line
    assert result.report["metrics"]["classification"]["counts_source"] == "measured"
    assert result.report["metrics"]["accuracy"] == 1.0
    assert result.report["metrics"]["primary_metric"]["reps"] == 10
    assert result.report["metrics"]["primary_metric"]["ci_level"] == 0.95
    assert result.report["dataset"]["windows"]["stats"]["coverage"] is True


def test_enrich_run_report_metrics_uses_pseudo_counts_and_returns_pairing_violation() -> (
    None
):
    report = {
        "metrics": {
            "window_match_fraction": 0.5,
            "window_overlap_fraction": 0.0,
        },
        "data": {"preview_n": 3, "final_n": 4},
    }
    core_report = SimpleNamespace(evaluation_windows=None, metrics={})
    run_config = SimpleNamespace(
        context={"eval": {"loss": {"resolved_type": "classification"}}}
    )
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=3, final_n=4))

    result = enrich_run_report_metrics(
        report=report,
        core_report=core_report,
        run_config=run_config,
        cfg=cfg,
        model_profile=SimpleNamespace(),
        baseline_requested=True,
        baseline_report_data=None,
        metric_kind=None,
        resolved_loss_type="classification",
        effective_preview=3,
        effective_final=4,
        profile_normalized="ci",
        window_plan=None,
        debug_metric_diffs_enabled=False,
        resolve_metric_and_provider_fn=lambda *args, **kwargs: (None, None, {}),
    )

    assert (
        result.report["metrics"]["classification"]["counts_source"] == "pseudo_config"
    )
    assert (
        "accuracy: pseudo counts from preview_n/final_n"
        in result.report["provenance"]["metric_notes"]
    )
    assert len(result.pairing_violations) == 2
    assert "window_match_fraction" in result.pairing_violations[0].message


def test_enrich_run_report_metrics_preserves_measured_multimodal_classification() -> (
    None
):
    report = {
        "metrics": {
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        },
        "data": {"preview_n": 1, "final_n": 1},
        "evaluation_windows": {
            "preview": {
                "records": [{"id": "ex-1", "correct": True}],
                "example_ids": ["ex-1"],
            },
            "final": {
                "records": [{"id": "ex-2", "correct": False}],
                "example_ids": ["ex-2"],
            },
        },
    }
    core_report = SimpleNamespace(
        evaluation_windows=report["evaluation_windows"],
        metrics={
            "classification": {
                "preview": {"correct_total": 1, "total": 1},
                "final": {"correct_total": 0, "total": 1},
                "n_correct": 0,
                "n_total": 1,
                "counts_source": "measured",
                "estimated": False,
            },
            "primary_metric": {"degraded": False},
        },
    )
    run_config = SimpleNamespace(
        context={"eval": {"loss": {"resolved_type": "classification"}}}
    )
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=1, final_n=1))

    result = enrich_run_report_metrics(
        report=report,
        core_report=core_report,
        run_config=run_config,
        cfg=cfg,
        model_profile=SimpleNamespace(),
        baseline_requested=False,
        baseline_report_data=None,
        metric_kind="accuracy",
        resolved_loss_type="classification",
        effective_preview=1,
        effective_final=1,
        profile_normalized="dev",
        window_plan=None,
        debug_metric_diffs_enabled=False,
        resolve_metric_and_provider_fn=lambda *args, **kwargs: (
            "accuracy",
            None,
            {},
        ),
    )

    assert result.report["metrics"]["classification"]["counts_source"] == "measured"
    assert result.report["metrics"]["classification"]["n_total"] == 1
    assert result.report["metrics"]["classification"]["n_correct"] == 0


def test_enrich_run_report_metrics_derives_measured_counts_from_accuracy_metric() -> (
    None
):
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "preview": 1.0,
                "final": 0.0,
                "n_preview": 1,
                "n_final": 1,
            }
        },
        "data": {"preview_n": 1, "final_n": 1},
        "evaluation_windows": {
            "preview": {"records": [{"id": "ex-1"}], "example_ids": ["ex-1"]},
            "final": {"records": [{"id": "ex-2"}], "example_ids": ["ex-2"]},
        },
    }
    core_report = SimpleNamespace(
        evaluation_windows=report["evaluation_windows"], metrics={}
    )
    run_config = SimpleNamespace(
        context={"eval": {"loss": {"resolved_type": "classification"}}}
    )
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=1, final_n=1))

    result = enrich_run_report_metrics(
        report=report,
        core_report=core_report,
        run_config=run_config,
        cfg=cfg,
        model_profile=SimpleNamespace(),
        baseline_requested=False,
        baseline_report_data=None,
        metric_kind="accuracy",
        resolved_loss_type="classification",
        effective_preview=1,
        effective_final=1,
        profile_normalized="dev",
        window_plan=None,
        debug_metric_diffs_enabled=False,
        resolve_metric_and_provider_fn=lambda *args, **kwargs: (
            "accuracy",
            None,
            {},
        ),
    )

    assert result.report["metrics"]["classification"]["counts_source"] == "measured"
    assert result.report["metrics"]["classification"]["preview"]["correct_total"] == 1
    assert result.report["metrics"]["classification"]["final"]["correct_total"] == 0


def test_enrich_run_report_metrics_rejects_unknown_metric_kind() -> None:
    report = {"metrics": {}, "data": {"preview_n": 1, "final_n": 1}}
    core_report = SimpleNamespace(evaluation_windows=None, metrics={})
    run_config = SimpleNamespace(context={"eval": {"loss": {"resolved_type": "ce"}}})
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=1, final_n=1))

    with pytest.raises(ValueError, match="Unsupported metric kind"):
        enrich_run_report_metrics(
            report=report,
            core_report=core_report,
            run_config=run_config,
            cfg=cfg,
            model_profile=SimpleNamespace(),
            baseline_requested=False,
            baseline_report_data=None,
            metric_kind="perplexity",
            resolved_loss_type="ce",
            effective_preview=1,
            effective_final=1,
            profile_normalized="dev",
            window_plan=None,
            debug_metric_diffs_enabled=False,
            resolve_metric_and_provider_fn=lambda *args, **kwargs: (
                "perplexity",
                None,
                {},
            ),
        )
