from __future__ import annotations

import json
from copy import deepcopy

from invarlock.reporting.certificate import make_certificate, validate_certificate
from invarlock.reporting.report_types import create_empty_report


def _mk_baseline() -> dict:
    b = create_empty_report()
    b["meta"].update(
        {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        }
    )
    b["data"].update(
        {
            "dataset": "ds",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        }
    )
    b["edit"]["name"] = "baseline"
    b["edit"]["plan_digest"] = "baseline_noop"
    b["edit"]["deltas"]["params_changed"] = 0
    b["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
    }
    b["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2],
            "logloss": [1.0, 1.0],
            "token_counts": [10, 10],
        }
    }
    b["artifacts"]["checkpoint_path"] = None
    b["flags"] = {"guard_recovered": False, "rollback_reason": None}
    return b


def test_make_certificate_covers_baseline_ratio_identity_branches(monkeypatch) -> None:
    # Avoid heavy bootstrap compute while still exercising the paired-window CI path.
    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )

    baseline = _mk_baseline()

    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        }
    )
    report["data"].update(
        {
            "dataset": "ds",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        }
    )
    report["edit"]["name"] = "noop"
    report["edit"]["plan_digest"] = "noop"
    report["edit"]["deltas"]["params_changed"] = 0
    report["evaluation_windows"] = deepcopy(baseline["evaluation_windows"])
    report["metrics"]["bootstrap"] = {
        "method": "percentile",
        "replicates": 10,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
    }
    report["metrics"]["paired_delta_summary"] = {"mean": 0.0, "degenerate": False}
    report["metrics"]["window_plan"] = {"profile": "dev"}
    report["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)

    # Exercise both sides of the baseline-ratio identity check:
    #  - ratio_vs_baseline == exp(baseline_delta_mean) (within tolerance)
    #  - ratio_vs_baseline mismatches exp(baseline_delta_mean)
    for ratio in (1.0, 1.2):
        run = deepcopy(report)
        run["metrics"]["primary_metric"] = {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": ratio,
        }
        cert = make_certificate(run, baseline)
        assert validate_certificate(cert)


def test_make_certificate_synthesizes_display_ci_from_ratio_or_defaults() -> None:
    baseline = _mk_baseline()

    cases = [
        # (primary_metric payload, report.config.guards, expected display_ci, expect ratio token)
        (
            {"kind": "accuracy", "ratio_vs_baseline": 1.25},
            {"variance": {"enabled": True}},
            [1.25, 1.25],
            True,
        ),
        (
            {"kind": "accuracy"},
            "not-a-dict",
            [1.0, 1.0],
            False,
        ),
    ]

    for pm, guards, expected_ci, expect_ratio in cases:
        report = create_empty_report()
        report["meta"].update(
            {
                "model_id": "m",
                "adapter": "hf",
                "device": "cpu",
                "seed": 1,
                "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
            }
        )
        report["data"].update(
            {
                "dataset": "ds",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 2,
                "final_n": 2,
            }
        )
        report["edit"]["name"] = "noop"
        report["edit"]["plan_digest"] = "noop"
        report["edit"]["deltas"]["params_changed"] = 0
        report["metrics"]["primary_metric"] = pm
        report["metrics"]["bootstrap"] = {"replicates": 0}
        report["config"] = {"guards": guards}
        report["provenance"] = {"dataset_split": "validation", "split_fallback": False}

        cert = make_certificate(report, baseline)
        assert validate_certificate(cert)

        pm_out = cert.get("primary_metric", {})
        assert isinstance(pm_out, dict)
        assert pm_out.get("display_ci") == expected_ci

        summary = (cert.get("telemetry", {}) or {}).get("summary_line", "")
        assert isinstance(summary, str) and summary.startswith("INVARLOCK_TELEMETRY ")
        assert "ci=" in summary and "width=" in summary
        assert ("ratio=" in summary) is expect_ratio

        # Ensure JSON serialization remains stable for downstream reporters.
        json.dumps(cert, sort_keys=True, default=str)

