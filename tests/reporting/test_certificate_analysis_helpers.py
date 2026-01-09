from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from invarlock.reporting import certificate as cert


def test_enforce_drift_ratio_identity_raises_for_ci_profile():
    with pytest.raises(
        ValueError,
        match="Paired ΔlogNLL mean is inconsistent with reported drift ratio",
    ):
        cert._enforce_drift_ratio_identity(
            paired_windows=1,
            delta_mean=math.log(1.6),
            drift_ratio=1.1,
            window_plan_profile="ci",
        )


def test_enforce_drift_ratio_identity_accepts_matching_ratio():
    ratio = math.log(1.01)
    assert cert._enforce_drift_ratio_identity(
        paired_windows=2,
        delta_mean=ratio,
        drift_ratio=1.01,
        window_plan_profile="ci",
    ) == pytest.approx(1.01)


def test_enforce_drift_ratio_identity_tolerates_dev_profile():
    ratio = math.log(1.2)
    assert cert._enforce_drift_ratio_identity(
        paired_windows=2,
        delta_mean=ratio,
        drift_ratio=1.1,
        window_plan_profile="dev",
    ) == pytest.approx(math.exp(ratio))


def test_enforce_ratio_ci_alignment_raises_on_mismatch():
    with pytest.raises(ValueError, match="CI mismatch"):
        cert._enforce_ratio_ci_alignment(
            "paired_baseline",
            (1.2, 1.3),
            (0.0, 0.0),
        )


def test_enforce_ratio_ci_alignment_ignores_non_paired_sources():
    # Should no-op when ratio_ci comes from non-paired sources
    cert._enforce_ratio_ci_alignment("manual", (1.0, 1.1), (0.0, 0.1))


def test_enforce_ratio_ci_alignment_returns_on_bad_intervals():
    # Paired source but malformed intervals should early-return without raising
    cert._enforce_ratio_ci_alignment("paired_baseline", (1.0,), (0.0, 0.1))


def test_enforce_ratio_ci_alignment_skips_non_finite_bounds():
    # Non-finite observed bounds should be ignored (continue branch)
    cert._enforce_ratio_ci_alignment("paired_baseline", (math.nan, 1.0), (0.0, 0.0))


def test_enforce_display_ci_alignment_backfills_ci_and_display_ci_in_dev():
    pm = {"kind": "ppl_causal"}
    cert._enforce_display_ci_alignment(
        "paired_baseline", pm, (0.0, 0.1), window_plan_profile="dev"
    )
    assert pm["ci"] == (0.0, 0.1)
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.0)),
        pytest.approx(math.exp(0.1)),
    ]


def test_enforce_display_ci_alignment_raises_on_missing_ci_in_ci_profile():
    pm = {"kind": "ppl_causal", "display_ci": (1.0, 1.1)}
    with pytest.raises(ValueError, match="primary_metric.ci missing"):
        cert._enforce_display_ci_alignment(
            "paired_baseline", pm, (math.nan, math.nan), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_raises_on_mismatch_in_ci_profile():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1), "display_ci": (1.5, 1.6)}
    with pytest.raises(ValueError, match="display_ci mismatch"):
        cert._enforce_display_ci_alignment(
            "paired_baseline", pm, (0.0, 0.1), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_noop_for_non_paired():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1)}
    cert._enforce_display_ci_alignment("manual", pm, (0.0, 0.1), "dev")
    assert pm["ci"] == (0.0, 0.1)


def test_enforce_display_ci_alignment_noop_for_non_ppl_metric():
    pm = {"kind": "accuracy"}
    cert._enforce_display_ci_alignment("paired_baseline", pm, (0.0, 0.1), "dev")
    assert pm["kind"] == "accuracy"


def test_enforce_display_ci_alignment_returns_on_empty_metric():
    cert._enforce_display_ci_alignment("paired_baseline", {}, (0.0, 0.1), "dev")


def test_enforce_display_ci_alignment_dev_missing_ci_no_logloss_ci():
    pm = {"kind": "ppl_causal"}
    cert._enforce_display_ci_alignment(
        "paired_baseline", pm, (math.nan, math.nan), window_plan_profile="dev"
    )
    assert "ci" not in pm


def test_enforce_display_ci_alignment_raises_on_missing_display_ci_in_ci_profile():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1)}
    with pytest.raises(ValueError, match="primary_metric.display_ci missing"):
        cert._enforce_display_ci_alignment(
            "paired_baseline", pm, (0.0, 0.1), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_dev_overwrites_mismatch():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1), "display_ci": [1.5, 1.6]}
    cert._enforce_display_ci_alignment(
        "paired_baseline", pm, (0.0, 0.1), window_plan_profile="dev"
    )
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.0)),
        pytest.approx(math.exp(0.1)),
    ]


def test_enforce_pairing_and_coverage_uses_fallback_counts():
    stats = {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "paired_windows": 200,
        "actual_preview": None,
        "actual_final": None,
        "coverage": {
            "preview": {"used": 200},
            "final": {"used": 200},
            "replicates": {"used": None},
        },
        "bootstrap": {"replicates": 1200},
    }
    cert._enforce_pairing_and_coverage(stats, window_plan_profile="ci", tier="balanced")


def test_enforce_pairing_and_coverage_returns_on_dev_profile():
    cert._enforce_pairing_and_coverage({}, window_plan_profile="dev", tier="balanced")


def test_enforce_pairing_and_coverage_raises_on_missing_stats():
    with pytest.raises(ValueError, match="Missing dataset window stats"):
        cert._enforce_pairing_and_coverage(
            None, window_plan_profile="ci", tier="balanced"
        )


def test_fallback_paired_windows_uses_coverage_preview():
    coverage = {"preview": {"used": 7}}
    assert cert._fallback_paired_windows(0, coverage) == 7
    assert cert._fallback_paired_windows(2, coverage) == 2


def test_prepare_guard_overhead_section_ratio_threshold():
    payload = {
        "bare_ppl": 10.0,
        "guarded_ppl": 10.5,
        "warnings": ["slow"],
        "messages": ["note"],
        "checks": {"ratio": True},
        "overhead_threshold": 0.01,
    }
    sanitized, passed = cert._prepare_guard_overhead_section(payload)
    assert sanitized["evaluated"] is True
    assert sanitized["overhead_ratio"] == pytest.approx(1.05)
    assert passed is False  # ratio above 1% threshold should fail
    assert sanitized["warnings"] == ["slow"]
    assert sanitized["checks"] == {"ratio": True}


def test_prepare_guard_overhead_section_soft_pass_when_ratio_missing():
    sanitized, passed = cert._prepare_guard_overhead_section({"messages": ["x"]})
    assert sanitized["evaluated"] is False
    assert sanitized["passed"] is True
    assert "Guard overhead ratio unavailable" in sanitized["errors"][0]


def test_compute_quality_overhead_ratio_basis(monkeypatch):
    bare = {"metrics": {"primary_metric": {"final": 10.0}}}
    guarded = {"metrics": {"primary_metric": {"final": 11.0}}}
    raw = {"bare_report": bare, "guarded_report": guarded}

    monkeypatch.setattr(
        cert,
        "compute_primary_metric_from_report",
        lambda report, kind=None: report["metrics"]["primary_metric"],
    )
    monkeypatch.setattr(
        cert,
        "get_metric",
        lambda kind: SimpleNamespace(direction="lower"),
        raising=False,
    )

    result = cert._compute_quality_overhead_from_guard(raw, "ppl_causal")
    assert result == {
        "basis": "ratio",
        "value": pytest.approx(1.1),
        "kind": "ppl_causal",
    }


def test_compute_quality_overhead_accuracy_delta(monkeypatch):
    bare = {"metrics": {"primary_metric": {"final": 0.7}}}
    guarded = {"metrics": {"primary_metric": {"final": 0.8}}}
    raw = {"bare_report": bare, "guarded_report": guarded}

    monkeypatch.setattr(
        cert,
        "compute_primary_metric_from_report",
        lambda report, kind=None: report["metrics"]["primary_metric"],
    )
    monkeypatch.setattr(
        cert,
        "get_metric",
        lambda kind: SimpleNamespace(direction="higher"),
        raising=False,
    )

    result = cert._compute_quality_overhead_from_guard(raw, "accuracy")
    assert result["basis"] == "delta_pp"
    assert result["kind"] == "accuracy"
    assert result["value"] == pytest.approx(10.0)  # (0.8-0.7)*100


def test_collect_backend_versions_with_fake_torch(monkeypatch):
    class _FakeProps:
        name = "FakeGPU"
        major = 9
        minor = 0

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda idx: _FakeProps(),
    )
    fake_torch = SimpleNamespace(
        __version__="1.0.0",
        version=SimpleNamespace(cuda="12.0", cudnn="8.0", git_version="abc123"),
        cuda=fake_cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = cert._collect_backend_versions()
    assert info["torch"] == "1.0.0"
    assert info["device_name"] == "FakeGPU"
    assert info["sm_capability"] == "9.0"


def test_compute_validation_flags_accuracy_hysteresis(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.05,
                    "min_tokens": 100,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.3,
                    "min_examples": 20,
                    "min_examples_fraction": 0.4,
                    "hysteresis_delta_pp": 0.15,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    ppl = {
        "preview_final_ratio": 1.0,
        "ratio_vs_baseline": 1.12,
        "ratio_ci": (1.05, 1.13),
    }
    spectral = {"caps_applied": 4, "caps_exceeded": True, "summary": {}}
    rmt = {"stable": True}
    invariants = {"status": "ok"}
    guard_overhead = {"overhead_ratio": 1.12, "overhead_threshold": 0.05}
    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": 0.2, "n_final": 10}
    dataset_capacity = {"examples_available": 40}
    ppl_metrics = {"preview_total_tokens": 60, "final_total_tokens": 60}

    flags = cert._compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics=ppl_metrics,
        guard_overhead=guard_overhead,
        primary_metric=primary_metric,
        dataset_capacity=dataset_capacity,
    )

    assert flags["preview_final_drift_acceptable"] is True
    assert flags["invariants_pass"] is True
    assert flags["rmt_stable"] is True
    assert flags["spectral_stable"] is False
    assert flags["guard_overhead_acceptable"] is False
    assert flags["primary_metric_acceptable"] is False
    assert flags["hysteresis_applied"] is True
    assert flags["primary_metric_tail_acceptable"] is True


def test_compute_validation_flags_core_gates_ppl_and_tail_fail(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                    "ratio_limit_base": 1.10,
                },
            },
            "spectral": {"max_caps": 1},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    flags = cert._compute_validation_flags(
        {
            "preview_final_ratio": 1.20,  # out of [0.95, 1.05]
            "ratio_vs_baseline": 1.25,  # > ratio limit
            "ratio_ci": (1.20, 1.30),
        },
        {"caps_applied": 2, "max_caps": 1, "caps_exceeded": False, "summary": {}},
        {"stable": False},
        {"status": "fail"},
        tier="balanced",
        guard_overhead={"overhead_ratio": 1.20, "overhead_threshold": 0.0},
        pm_tail={"mode": "fail", "evaluated": True, "passed": False},
    )

    assert flags["preview_final_drift_acceptable"] is False
    assert flags["primary_metric_acceptable"] is False
    assert flags["invariants_pass"] is False
    assert flags["spectral_stable"] is False
    assert flags["rmt_stable"] is False
    assert flags["guard_overhead_acceptable"] is False
    assert flags["primary_metric_tail_acceptable"] is False


def test_compute_validation_flags_tiny_relax_allows_unevaluated_overhead(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 5},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")

    guard_overhead = {"passed": False, "evaluated": False, "errors": ["missing"]}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        dataset_capacity={"tokens_available": 0},
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_edit_digest_detects_quantization():
    report = {
        "edit": {
            "name": "quant_rtn",
            "config": {"scope": "ffn", "plan": {"target_sparsity": 0.5}},
        }
    }
    digest = cert._compute_edit_digest(report)
    assert digest["family"] == "quantization"
    assert digest["version"] == 1


def test_compute_edit_digest_defaults_when_missing():
    digest = cert._compute_edit_digest({})
    assert digest["family"] == "cert_only"


def test_compute_confidence_label_accuracy_medium():
    certificate = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": (0.6, 0.9),
            "unstable": True,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.5}},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "accuracy"
    assert label["label"] == "Medium"


def test_compute_confidence_label_low_when_gate_fails():
    certificate = {
        "validation": {"primary_metric_acceptable": False},
        "primary_metric": {"kind": "ppl_causal", "display_ci": (1.0, 1.02)},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "ppl_ratio"
    assert label["label"] == "Low"


def test_normalize_baseline_handles_schema_v1():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"commit_sha": "abcdef1234567890", "model_id": "demo"},
        "metrics": {"ppl_final": 42.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "ok"},
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["ppl_final"] == 42.0
    assert normalized["spectral"] == {"caps": 1}


def test_normalize_baseline_infers_primary_metric():
    baseline = {
        "meta": {"model_id": "demo", "adapter": "hf"},
        "edit": {"name": "baseline", "deltas": {"params_changed": 0}, "plan": {}},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.0,
                "final": 10.0,
            },
            "bootstrap": {"coverage": {"used": 1}},
            "spectral": {},
            "rmt": {},
            "invariants": {},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.2]}},
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["ppl_final"] == 10.0
    assert normalized["ppl_preview"] == 9.0
    assert normalized["evaluation_windows"]["final"]["logloss"] == [0.2]


def test_normalize_baseline_warns_on_invalid(capsys):
    baseline = {
        "meta": {"model_id": "demo"},
        "edit": {"name": "quantize", "plan": {}, "deltas": {"params_changed": 5}},
        "metrics": {
            "ppl_final": 0.5,
            "ppl_preview": 0.5,
            "spectral": {},
            "rmt": {},
            "invariants": {},
        },
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["ppl_final"] == pytest.approx(50.797)
    out = capsys.readouterr().out
    assert "Invalid baseline detected" in out


def test_compute_validation_flags_guard_overhead_ratio_failure(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )

    guard_overhead = {"overhead_ratio": 1.05, "overhead_threshold": 0.01}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )

    assert flags["guard_overhead_acceptable"] is False


def test_compute_validation_flags_guard_overhead_ratio_passes_with_tiny_relax(
    monkeypatch,
):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")

    guard_overhead = {"overhead_ratio": 1.05, "overhead_threshold": 0.01}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.2},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.2},
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        dataset_capacity={"tokens_available": 0},
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_validation_flags_guard_overhead_passes_when_ratio_missing(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )

    guard_overhead = {"overhead_threshold": 0.01}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 0.9},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 0.9},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_validation_flags_accuracy_fails_low_examples(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -0.2,
                    "min_examples": 200,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -0.1, "n_final": 50}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )

    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_accuracy_respects_dataset_fraction_floor(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -0.5,
                    "min_examples": 0,
                    "min_examples_fraction": 0.25,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -0.3, "n_final": 15}
    dataset_capacity = {"examples_available": 80}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity=dataset_capacity,
    )

    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_accuracy_passes_with_hysteresis(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -1.0,
                    "min_examples": 50,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.5,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -1.2, "n_final": 80}
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"examples_available": 200},
    )

    assert flags["primary_metric_acceptable"] is True
    assert flags["hysteresis_applied"] is True


def test_validate_certificate_uses_jsonschema(monkeypatch):
    class DummySchema:
        def __init__(self):
            self.calls = 0

        def validate(self, instance, schema):
            self.calls += 1

    dummy = DummySchema()
    monkeypatch.setattr(cert, "jsonschema", dummy, raising=False)
    certificate = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-1",
        "primary_metric": {"kind": "ppl_causal"},
        "validation": {"primary_metric_acceptable": True},
    }
    assert cert.validate_certificate(certificate) is True
    assert dummy.calls == 1


def test_validate_certificate_falls_back_when_jsonschema_fails(monkeypatch):
    class FailingSchema:
        def validate(self, instance, schema):
            raise ValueError("boom")

    monkeypatch.setattr(cert, "jsonschema", FailingSchema(), raising=False)
    certificate = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-2",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }
    assert cert.validate_certificate(certificate) is True


def test_validate_certificate_rejects_invalid_flags(monkeypatch):
    monkeypatch.setattr(cert, "jsonschema", None, raising=False)
    certificate = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-3",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": "yes"},
    }
    assert cert.validate_certificate(certificate) is False


def test_load_validation_allowlist_prefers_contracts_file(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    root = Path(cert.__file__).resolve().parents[3]
    contracts_dir = root / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    key_file = contracts_dir / "validation_keys.json"
    original = key_file.read_text(encoding="utf-8") if key_file.exists() else None

    keys = ["primary_metric_acceptable", "guard_overhead_acceptable", "custom_flag"]
    try:
        key_file.write_text(json.dumps(keys), encoding="utf-8")
        loaded = cert._load_validation_allowlist()
        assert loaded == {str(k) for k in keys}
    finally:
        if original is None:
            key_file.unlink(missing_ok=True)
        else:
            key_file.write_text(original, encoding="utf-8")


def test_validate_certificate_handles_mapping_errors() -> None:
    class ExplodingMapping(dict):
        def get(self, *_args, **_kwargs):
            raise ValueError("boom")

    certificate = ExplodingMapping()
    # ValueError raised inside validate_certificate should be caught and return False.
    assert cert.validate_certificate(certificate) is False


def test_propagate_pairing_stats_adds_missing_fields():
    certificate = {"dataset": {"windows": {}}}
    ppl_analysis = {
        "stats": {
            "pairing": "paired_baseline",
            "paired_windows": 4,
            "coverage": {"preview": {"used": 3}},
            "window_match_fraction": 0.9,
            "window_overlap_fraction": 0.4,
            "window_pairing_reason": "id_match",
            "requested_preview": 2,
            "requested_final": 2,
            "actual_preview": 2,
            "actual_final": 2,
            "coverage_ok": True,
        }
    }
    cert._propagate_pairing_stats(certificate, ppl_analysis)
    stats = certificate["dataset"]["windows"]["stats"]
    assert stats["pairing"] == "paired_baseline"
    assert stats["paired_windows"] == 4
    assert stats["coverage"]["preview"]["used"] == 3
    assert stats["window_match_fraction"] == pytest.approx(0.9)
    assert stats["window_pairing_reason"] == "id_match"


def test_propagate_pairing_stats_ignores_missing_dataset():
    cert._propagate_pairing_stats({}, {"stats": {}})


def test_normalize_baseline_handles_v1_schema_structure():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"commit_sha": "abcdef1234567890", "model_id": "gpt2"},
        "metrics": {"ppl_final": 25.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "ok"},
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["model_id"] == "gpt2"
    assert normalized["ppl_final"] == 25.0
    assert normalized["spectral"] == {"caps": 1}


def test_normalize_baseline_falls_back_for_invalid_ppl():
    baseline = {
        "meta": {"model_id": "demo"},
        "edit": {"name": "baseline", "plan": {}, "deltas": {"params_changed": 0}},
        "metrics": {"ppl_final": 0.9, "spectral": {}, "rmt": {}, "invariants": {}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["ppl_final"] == pytest.approx(50.797)


def test_normalize_baseline_extracts_runreport_payload():
    baseline = {
        "meta": {"model_id": "demo", "auto": {"tier": "balanced"}},
        "edit": {
            "name": "baseline",
            "plan": {"target_sparsity": 0.0},
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 12.0, "preview": 11.0},
            "spectral": {"caps_applied": 0},
            "rmt": {"stable": True},
            "invariants": {"status": "ok"},
            "bootstrap": {},
            "window_overlap_fraction": 0.4,
            "window_match_fraction": 1.0,
        },
        "evaluation_windows": {
            "final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]},
        },
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["run_id"] is not None
    assert normalized["spectral"]["caps_applied"] == 0
    assert normalized["evaluation_windows"]["final"]["logloss"] == [0.1, 0.2]


def test_extract_edit_metadata_infers_scope_from_budgets():
    report = {
        "edit": {
            "plan": {
                "head_budget": {"ratio": 0.5},
                "mlp_budget": {"ratio": 0.5},
            },
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
        }
    }
    metadata = cert._extract_edit_metadata(report, {})
    assert metadata["scope"] == "heads+ffn"


def test_extract_edit_metadata_uses_config_plan_fallback():
    report = {
        "edit": {
            "config": {"plan": {"scope": "ffn", "ranking": "l2"}},
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
        }
    }
    metadata = cert._extract_edit_metadata(report, {})
    assert metadata["scope"] == "ffn"
    assert metadata["ranking"] == "l2"


def test_compute_report_digest_returns_none_for_non_dict():
    assert cert._compute_report_digest(None) is None


def test_compute_edit_digest_quantization_family():
    report = {"edit": {"name": "quant_rtn", "config": {"scope": "ffn"}}}
    digest = cert._compute_edit_digest(report)
    assert digest["family"] == "quantization"


def test_compute_edit_digest_handles_faulty_mapping():
    class Faulty:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    digest = cert._compute_edit_digest(Faulty())
    assert digest["family"] == "cert_only"


def test_compute_confidence_label_accuracy_high():
    certificate = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "accuracy", "display_ci": (0.7, 0.72)},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.05}},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["label"] == "High"
    assert label["basis"] == "accuracy"


def test_extract_structural_deltas_captures_bitwidth_and_ranks():
    report = {
        "edit": {
            "name": "quant_rtn",
            "plan": {"scope": "heads"},
            "config": {"plan": {"seed": 19}},
            "deltas": {
                "params_changed": 10,
                "bitwidth_map": {
                    "layer1": {"bitwidth": 4, "group_size": 32, "params": 512}
                },
                "rank_map": {
                    "layer1": {
                        "rank": 8,
                        "params_saved": 128,
                        "energy_retained": 0.95,
                        "deploy_mode": "recompose",
                        "savings_mode": "realized",
                        "realized_params_saved": 64,
                        "theoretical_params_saved": 80,
                        "realized_params": 900,
                        "theoretical_params": 920,
                        "skipped": False,
                    }
                },
                "savings": {"deploy_mode": "recompose"},
            },
        },
        "meta": {"seed": 7},
    }
    structure = cert._extract_structural_deltas(report)
    assert "bitwidths" in structure
    assert "ranks" in structure
    diag = structure["compression_diagnostics"]
    assert diag["algorithm_details"]["seed"] == 7


def test_build_provenance_block_uses_schedule_digest(monkeypatch):
    monkeypatch.setattr(
        cert, "_collect_backend_versions", lambda: {"python": "x.y"}, raising=False
    )
    report = {"provenance": {}, "meta": {"model_id": "model"}}
    baseline_ref = {"run_id": "baseline-1"}
    artifacts = {"generated_at": "now", "report_path": "/tmp/report"}
    policy = {"source": "auto"}
    ppl = {"window_plan": {"profile": "dev"}}

    provenance = cert._build_provenance_block(
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        artifacts_payload=artifacts,
        policy_provenance=policy,
        schedule_digest="abc123",
        ppl_analysis=ppl,
        current_run_id="edited-1",
    )

    assert provenance["provider_digest"] == {"ids_sha256": "abc123"}
    assert provenance["window_plan_digest"] == "abc123"
    assert provenance["window_plan"]["profile"] == "dev"


def test_compute_validation_flags_marks_moe_observed():
    flags = cert._compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        moe={"top_k": 1},
        guard_overhead={"passed": True},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )
    assert flags["moe_observed"] is True
    assert flags["moe_identity_ok"] is True


def test_compute_validation_flags_reconciles_ppl_primary_metric_ratio(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.setattr(
        cert, "get_tier_policies", lambda *_a, **_k: dict(fake_policies)
    )
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.2}
    primary_metric = {"kind": "ppl_causal", "ratio_vs_baseline": 1.05}
    flags = cert._compute_validation_flags(
        ppl,
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )
    assert flags["primary_metric_acceptable"] is True


def test_extract_compression_diagnostics_no_modifications():
    inference_record = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    diagnostics = cert._extract_compression_diagnostics(
        "quant_rtn",
        {"scope": "ffn", "clamp_ratio": 0.0},
        {"params_changed": 0},
        {},
        inference_record,
    )
    assert diagnostics["execution_status"] == "no_modifications"
    assert diagnostics["target_analysis"] == {
        "modules_found": 0,
        "modules_eligible": 0,
        "modules_modified": 0,
        "scope": "ffn",
    }
    assert diagnostics["parameter_analysis"] == {
        "bitwidth": {"value": "unknown", "effectiveness": "ineffective"},
        "clamp_ratio": {"value": 0.0, "effectiveness": "disabled"},
    }
    assert diagnostics["algorithm_details"] == {
        "scope_targeting": "ffn",
        "seed": "unknown",
    }
    assert diagnostics["warnings"] == [
        "No parameters were modified - algorithm may be too conservative",
        "Check scope configuration and parameter thresholds",
        "FFN scope may not match model architecture - try 'all' scope",
    ]
    assert diagnostics["inferred"] == dict.fromkeys(
        ("scope", "seed", "rank_policy", "frac"), False
    )
    assert "inference_source" not in diagnostics
    assert "inference_log" not in diagnostics


def test_extract_compression_diagnostics_quant_success():
    inference_record = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    deltas = {
        "params_changed": 5,
        "bitwidth_map": {"layer1": {"bitwidth": 8, "group_size": 32, "params": 256}},
    }
    diagnostics = cert._extract_compression_diagnostics(
        "quant_rtn",
        {"scope": "attn", "clamp_ratio": 0.5},
        deltas,
        {},
        inference_record,
    )
    assert diagnostics["execution_status"] == "successful"
    assert diagnostics["target_analysis"] == {
        "modules_found": 1,
        "modules_eligible": 1,
        "modules_modified": 1,
        "scope": "attn",
    }
    assert diagnostics["parameter_analysis"] == {
        "bitwidth": {"value": 8, "effectiveness": "applied"},
        "group_size": {"value": 32, "effectiveness": "used"},
        "clamp_ratio": {"value": 0.5, "effectiveness": "applied"},
    }
    assert diagnostics["algorithm_details"] == {
        "scope_targeting": "attn",
        "seed": "unknown",
        "modules_quantized": 1,
        "quantization_type": "grouped",
        "total_params_quantized": 256,
        "estimated_memory_saved_mb": 0.0,
    }
    # Successful quantization diagnostics should remain edit-agnostic and avoid
    # prescriptive parameter hints (e.g., suggesting 4-bit alternatives).
    assert diagnostics["warnings"] == []
    assert diagnostics["inferred"] == dict.fromkeys(
        ("scope", "seed", "rank_policy", "frac"), False
    )
    assert "inference_source" not in diagnostics
    assert "inference_log" not in diagnostics


def test_extract_rank_information_tracks_skipped_modules():
    deltas = {
        "rank_map": {
            "layer.0": {"rank": 8, "params_saved": 10},
            "layer.1": {"rank": 0, "params_saved": 0, "skipped": True},
        },
        "savings": {"deploy_mode": "recompose"},
    }
    info = cert._extract_rank_information({"frac": 0.2}, deltas)
    assert "per_module" in info
    assert info["skipped_modules"] == ["layer.1"]


def test_build_provenance_block_respects_existing_provider_digest():
    report = {
        "provenance": {"provider_digest": {"source": "pre"}},
        "artifacts": {},
    }
    provenance = cert._build_provenance_block(
        report,
        {"artifacts": {"logs_path": "/logs/base.log"}},
        {"run_id": "base-1"},
        {"generated_at": "now", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "abc123",
        {},
        "run-1",
    )
    assert provenance["provider_digest"] == {"source": "pre"}
    assert provenance["baseline"]["report_path"] == "/logs/base.log"


def test_build_provenance_block_fallbacks_to_schedule_digest():
    provenance = cert._build_provenance_block(
        {},
        {},
        {"run_id": "base-1"},
        {"generated_at": "now", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "deadbeef",
        {},
        "run-2",
    )
    assert provenance["provider_digest"] == {"ids_sha256": "deadbeef"}


def test_build_provenance_block_transfers_dataset_split_and_window_plan():
    report = {
        "provenance": {"dataset_split": "eval", "split_fallback": True},
        "artifacts": {},
    }
    ppl_analysis = {"window_plan": {"profile": "ci"}}
    provenance = cert._build_provenance_block(
        report,
        {},
        {"run_id": "base-2"},
        {"generated_at": "ts", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "cafebabe",
        ppl_analysis,
        "run-3",
    )
    assert provenance["dataset_split"] == "eval"
    assert provenance["split_fallback"] is True
    assert provenance["window_plan"]["profile"] == "ci"
    assert provenance["window_ids_digest"] == "cafebabe"


def test_compute_confidence_label_handles_unknown_metric_kind():
    certificate = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "custom_metric", "display_ci": (2.0, 2.5)},
        "resolved_policy": {},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "primary_metric"
    assert label["label"] == "Low"
