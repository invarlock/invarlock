from __future__ import annotations

import importlib
import json
from copy import deepcopy

import pytest

from invarlock.reporting import certificate as cert_mod
from invarlock.reporting import certificate_schema as cert_schema_mod
from invarlock.reporting import primary_metric_utils
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


def test_make_certificate_synthesizes_display_ci_from_ratio_or_defaults(monkeypatch) -> None:
    baseline = _mk_baseline()

    def attach_stub(certificate, report, baseline_raw, baseline_ref, ppl_analysis):  # noqa: ANN001,ARG001
        pm = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        pm = dict(pm) if isinstance(pm, dict) else {"kind": "accuracy"}
        pm.pop("display_ci", None)
        pm.pop("ci", None)
        certificate["primary_metric"] = pm

    # Ensure our test payload survives normalization and triggers the local fallback block.
    monkeypatch.setattr(cert_mod, "_normalize_and_validate_report", lambda r: r, raising=False)
    monkeypatch.setattr(primary_metric_utils, "attach_primary_metric", attach_stub)

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


def test_enforce_pairing_and_coverage_branch_matrix() -> None:
    base = {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "paired_windows": 180,
        "actual_preview": 180,
        "actual_final": 180,
        "coverage": {
            "preview": {"used": 180},
            "final": {"used": 180},
            "replicates": {"used": 1200},
        },
    }

    # Pass-through: everything present.
    cert_mod._enforce_pairing_and_coverage(dict(base), window_plan_profile="ci", tier="balanced")

    # Fill missing actual_* from coverage dict.
    stats = dict(base)
    stats.pop("actual_preview")
    stats.pop("actual_final")
    cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Mixed: actual_preview present, actual_final derived from coverage.
    stats = dict(base)
    stats["actual_final"] = None
    cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Mixed: actual_final present, actual_preview derived from coverage.
    stats = dict(base)
    stats["actual_preview"] = None
    cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Coverage fallback disabled -> missing preview/final counts hard-fails.
    stats = {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "paired_windows": 180,
        "actual_preview": None,
        "actual_final": None,
        "coverage": None,
    }
    with pytest.raises(ValueError, match="preview/final window counts"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Preview/final mismatch hard-fails.
    stats = dict(base)
    stats["actual_final"] = 179
    with pytest.raises(ValueError, match="matching preview/final counts"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Missing bootstrap coverage stats hard-fails.
    stats = dict(base)
    stats["coverage"] = None
    with pytest.raises(ValueError, match="bootstrap coverage stats"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Replicates derived from stats.bootstrap when coverage.replicates.used missing.
    stats = dict(base)
    stats["coverage"] = {
        "preview": {"used": 180},
        "final": {"used": 180},
        "replicates": {"used": None},
    }
    stats["bootstrap"] = {"replicates": 1200}
    cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Replicates remain missing when stats.bootstrap not a dict -> hard-fail.
    stats = dict(base)
    stats["coverage"] = {
        "preview": {"used": 180},
        "final": {"used": 180},
        "replicates": {"used": None},
    }
    stats["bootstrap"] = "nope"
    with pytest.raises(ValueError, match="preview/final/replicates coverage stats"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Preview below tier floor hard-fails.
    stats = dict(base)
    stats["coverage"] = {
        "preview": {"used": 179},
        "final": {"used": 180},
        "replicates": {"used": 1200},
    }
    with pytest.raises(ValueError, match="tier floors"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")

    # Replicates below tier floor hard-fails.
    stats = dict(base)
    stats["coverage"] = {
        "preview": {"used": 180},
        "final": {"used": 180},
        "replicates": {"used": 1199},
    }
    with pytest.raises(ValueError, match="bootstrap replicates"):
        cert_mod._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")


def test_propagate_pairing_stats_early_returns() -> None:
    cert_mod._propagate_pairing_stats(certificate=None, ppl_analysis=None)  # type: ignore[arg-type]
    cert_mod._propagate_pairing_stats(certificate={}, ppl_analysis=None)
    cert_mod._propagate_pairing_stats(certificate={"dataset": None}, ppl_analysis={})
    cert_mod._propagate_pairing_stats(certificate={"dataset": {"windows": None}}, ppl_analysis={})
    cert_mod._propagate_pairing_stats(
        certificate={"dataset": {"windows": {"stats": None}}}, ppl_analysis={}
    )


def test_normalize_override_entry_variants() -> None:
    assert cert_mod._normalize_override_entry(None) == []
    assert cert_mod._normalize_override_entry("a.yaml") == ["a.yaml"]
    assert cert_mod._normalize_override_entry(["a.yaml", None, 5]) == ["a.yaml", "5"]


def test_normalize_baseline_derives_ppl_from_primary_metric() -> None:
    baseline = {
        "meta": {"model_id": "m"},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0, "preview": 9.0}
        },
        "edit": {"name": "baseline", "plan_digest": "baseline_noop", "deltas": {"params_changed": 0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
    }
    out = cert_mod._normalize_baseline(baseline)
    assert out.get("ppl_final") == 10.0
    assert out.get("ppl_preview") == 9.0


def test_prepare_guard_overhead_section_keeps_skip_reason() -> None:
    payload = {"skipped": True, "skip_reason": " because ", "overhead_threshold": 0.01}
    out, ok = cert_mod._prepare_guard_overhead_section(payload)
    assert ok is True
    assert out.get("skip_reason") == "because"


def test_compute_validation_flags_acceptance_bounds_and_accuracy_tiny_relax(monkeypatch) -> None:
    # Acceptance bounds (min/max) parsing and PM ratio fallback.
    flags = cert_mod._compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": float("nan")},
        spectral={},
        rmt={},
        invariants={"status": "ok"},
        tier="balanced",
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        pm_acceptance_range={"min": 0.9, "max": 1.1},
    )
    assert flags.get("primary_metric_acceptable") in {True, False}

    # Tiny-relax accuracy branch: accept missing delta and small n.
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    flags2 = cert_mod._compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 0.0},
        spectral={},
        rmt={},
        invariants={"status": "ok"},
        tier="balanced",
        primary_metric={"kind": "accuracy", "ratio_vs_baseline": None, "n_final": 1},
        dataset_capacity={"examples_available": 1},
    )
    assert flags2.get("primary_metric_acceptable") is True


def test_certificate_module_schema_tightening_branches(monkeypatch) -> None:
    original_schema = cert_schema_mod.CERTIFICATE_JSON_SCHEMA
    try:
        monkeypatch.setattr(
            cert_schema_mod, "CERTIFICATE_JSON_SCHEMA", {"properties": "nope"}
        )
        importlib.reload(cert_mod)

        monkeypatch.setattr(
            cert_schema_mod,
            "CERTIFICATE_JSON_SCHEMA",
            {"properties": {"validation": "nope"}},
        )
        importlib.reload(cert_mod)
    finally:
        monkeypatch.setattr(cert_schema_mod, "CERTIFICATE_JSON_SCHEMA", original_schema)
        importlib.reload(cert_mod)


def test_make_certificate_ratio_ci_fallback_skips_non_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )
    monkeypatch.setattr(cert_mod, "_coerce_interval", lambda _v: (0.0,))  # type: ignore[assignment]

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
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
    }

    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
