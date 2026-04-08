from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from invarlock.reporting import report_overhead as overhead_mod
from invarlock.reporting import report_primary_metric_policy as pm_policy
from invarlock.reporting import report_provenance as provenance_mod


def test_enforce_drift_ratio_identity_raises_for_ci_profile():
    with pytest.raises(
        ValueError,
        match="Paired ΔlogNLL mean is inconsistent with reported drift ratio",
    ):
        pm_policy.enforce_drift_ratio_identity(
            paired_windows=1,
            delta_mean=math.log(1.6),
            drift_ratio=1.1,
            window_plan_profile="ci",
        )


def test_enforce_drift_ratio_identity_accepts_matching_ratio():
    ratio = math.log(1.01)
    assert pm_policy.enforce_drift_ratio_identity(
        paired_windows=2,
        delta_mean=ratio,
        drift_ratio=1.01,
        window_plan_profile="ci",
    ) == pytest.approx(1.01)


def test_enforce_drift_ratio_identity_tolerates_dev_profile():
    ratio = math.log(1.2)
    assert pm_policy.enforce_drift_ratio_identity(
        paired_windows=2,
        delta_mean=ratio,
        drift_ratio=1.1,
        window_plan_profile="dev",
    ) == pytest.approx(math.exp(ratio))


def test_enforce_ratio_ci_alignment_raises_on_mismatch():
    with pytest.raises(ValueError, match="CI mismatch"):
        pm_policy.enforce_ratio_ci_alignment(
            "paired_baseline",
            (1.2, 1.3),
            (0.0, 0.0),
        )


def test_enforce_ratio_ci_alignment_ignores_non_paired_sources():
    pm_policy.enforce_ratio_ci_alignment("manual", (1.0, 1.1), (0.0, 0.1))


def test_enforce_ratio_ci_alignment_returns_on_bad_intervals():
    pm_policy.enforce_ratio_ci_alignment("paired_baseline", (1.0,), (0.0, 0.1))


def test_enforce_ratio_ci_alignment_skips_non_finite_bounds():
    pm_policy.enforce_ratio_ci_alignment("paired_baseline", (math.nan, 1.0), (0.0, 0.0))


def test_enforce_display_ci_alignment_backfills_ci_and_display_ci_in_dev():
    pm = {"kind": "ppl_causal"}
    pm_policy.enforce_display_ci_alignment(
        "paired_baseline", pm, (0.0, 0.1), window_plan_profile="dev"
    )
    assert pm["ci"] == [0.0, 0.1]
    assert pm["display_ci"] == [
        pytest.approx(math.exp(0.0)),
        pytest.approx(math.exp(0.1)),
    ]


def test_enforce_display_ci_alignment_raises_on_missing_ci_in_ci_profile():
    pm = {"kind": "ppl_causal", "display_ci": (1.0, 1.1)}
    with pytest.raises(ValueError, match="primary_metric.ci missing"):
        pm_policy.enforce_display_ci_alignment(
            "paired_baseline", pm, (math.nan, math.nan), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_raises_on_mismatch_in_ci_profile():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1), "display_ci": (1.5, 1.6)}
    with pytest.raises(ValueError, match="display_ci mismatch"):
        pm_policy.enforce_display_ci_alignment(
            "paired_baseline", pm, (0.0, 0.1), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_noop_for_non_paired():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1)}
    pm_policy.enforce_display_ci_alignment("manual", pm, (0.0, 0.1), "dev")
    assert pm["ci"] == (0.0, 0.1)


def test_enforce_display_ci_alignment_noop_for_non_ppl_metric():
    pm = {"kind": "accuracy"}
    pm_policy.enforce_display_ci_alignment("paired_baseline", pm, (0.0, 0.1), "dev")
    assert pm["kind"] == "accuracy"


def test_enforce_display_ci_alignment_returns_on_empty_metric():
    pm_policy.enforce_display_ci_alignment("paired_baseline", {}, (0.0, 0.1), "dev")


def test_enforce_display_ci_alignment_returns_on_kind_coercion_error() -> None:
    class _BadGet(dict):
        def get(self, *_a, **_k):  # noqa: ANN001
            raise RuntimeError("boom")

    pm_policy.enforce_display_ci_alignment(
        "paired_baseline", _BadGet({"kind": "ppl_causal"}), (0.0, 0.1), "dev"
    )


def test_enforce_display_ci_alignment_dev_missing_ci_no_logloss_ci():
    pm = {"kind": "ppl_causal"}
    pm_policy.enforce_display_ci_alignment(
        "paired_baseline", pm, (math.nan, math.nan), window_plan_profile="dev"
    )
    assert "ci" not in pm


def test_enforce_display_ci_alignment_raises_on_missing_display_ci_in_ci_profile():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1)}
    with pytest.raises(ValueError, match="primary_metric.display_ci missing"):
        pm_policy.enforce_display_ci_alignment(
            "paired_baseline", pm, (0.0, 0.1), window_plan_profile="ci"
        )


def test_enforce_display_ci_alignment_dev_overwrites_mismatch():
    pm = {"kind": "ppl_causal", "ci": (0.0, 0.1), "display_ci": [1.5, 1.6]}
    pm_policy.enforce_display_ci_alignment(
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
    pm_policy.enforce_pairing_and_coverage(
        stats, window_plan_profile="ci", tier="balanced"
    )


def test_enforce_pairing_and_coverage_returns_on_dev_profile():
    pm_policy.enforce_pairing_and_coverage(
        {}, window_plan_profile="dev", tier="balanced"
    )


def test_enforce_pairing_and_coverage_raises_on_missing_pairing_fractions() -> None:
    stats = {"paired_windows": 1}
    with pytest.raises(ValueError, match="window_match_fraction"):
        pm_policy.enforce_pairing_and_coverage(
            stats, window_plan_profile="ci", tier="balanced"
        )

    stats2 = {"window_match_fraction": 1.0, "paired_windows": 1}
    with pytest.raises(ValueError, match="window_overlap_fraction"):
        pm_policy.enforce_pairing_and_coverage(
            stats2, window_plan_profile="ci", tier="balanced"
        )


def test_enforce_pairing_and_coverage_raises_on_imperfect_pairing_and_overlap() -> None:
    with pytest.raises(ValueError, match="perfect pairing"):
        pm_policy.enforce_pairing_and_coverage(
            {
                "window_match_fraction": 0.9,
                "window_overlap_fraction": 0.0,
                "paired_windows": 1,
            },
            window_plan_profile="ci",
            tier="balanced",
        )

    with pytest.raises(ValueError, match="non-overlapping windows"):
        pm_policy.enforce_pairing_and_coverage(
            {
                "window_match_fraction": 1.0,
                "window_overlap_fraction": 1e-6,
                "paired_windows": 1,
            },
            window_plan_profile="ci",
            tier="balanced",
        )


@pytest.mark.parametrize("paired_windows", ["bad", -1, 1.2])
def test_enforce_pairing_and_coverage_raises_on_invalid_paired_windows(
    paired_windows,
) -> None:
    with pytest.raises(ValueError, match="paired_windows"):
        pm_policy.enforce_pairing_and_coverage(
            {
                "window_match_fraction": 1.0,
                "window_overlap_fraction": 0.0,
                "paired_windows": paired_windows,
            },
            window_plan_profile="ci",
            tier="balanced",
        )


def test_enforce_pairing_and_coverage_raises_on_missing_stats():
    with pytest.raises(ValueError, match="Missing dataset window stats"):
        pm_policy.enforce_pairing_and_coverage(
            None, window_plan_profile="ci", tier="balanced"
        )


def test_fallback_paired_windows_uses_coverage_preview():
    coverage = {"preview": {"used": 7}}
    assert pm_policy.fallback_paired_windows(0, coverage) == 7
    assert pm_policy.fallback_paired_windows(2, coverage) == 2


def test_prepare_guard_overhead_section_ratio_threshold():
    payload = {
        "bare_ppl": 10.0,
        "guarded_ppl": 10.5,
        "warnings": ["slow"],
        "messages": ["note"],
        "checks": {"ratio": True},
        "overhead_threshold": 0.01,
    }
    sanitized, passed = overhead_mod.prepare_guard_overhead_section(payload)
    assert sanitized["evaluated"] is True
    assert sanitized["overhead_ratio"] == pytest.approx(1.05)
    assert passed is False
    assert sanitized["diagnostics"] == [
        {
            "kind": "guard_overhead_message",
            "severity": "info",
            "message": "note",
            "details": {},
        },
        {
            "kind": "guard_overhead_warning",
            "severity": "warning",
            "message": "slow",
            "details": {},
        },
    ]
    assert sanitized["checks"] == {"ratio": True}


def test_prepare_guard_overhead_section_soft_pass_when_ratio_missing():
    sanitized, passed = overhead_mod.prepare_guard_overhead_section({"messages": ["x"]})
    assert sanitized["evaluated"] is False
    assert sanitized["passed"] is True
    assert sanitized["diagnostics"] == [
        {
            "kind": "guard_overhead_message",
            "severity": "info",
            "message": "x",
            "details": {},
        }
    ]


def test_compute_quality_overhead_ratio_basis():
    bare = {"metrics": {"primary_metric": {"final": 10.0}}}
    guarded = {"metrics": {"primary_metric": {"final": 11.0}}}
    raw = {"bare_report": bare, "guarded_report": guarded}

    result = overhead_mod.compute_quality_overhead_from_guard(
        raw,
        "ppl_causal",
        compute_primary_metric_from_report_fn=lambda report, kind=None: report[
            "metrics"
        ]["primary_metric"],
        get_metric_fn=lambda kind: SimpleNamespace(direction="lower"),
    )
    assert result == {
        "basis": "ratio",
        "value": pytest.approx(1.1),
        "kind": "ppl_causal",
    }


def test_compute_quality_overhead_accuracy_delta():
    bare = {"metrics": {"primary_metric": {"final": 0.7}}}
    guarded = {"metrics": {"primary_metric": {"final": 0.8}}}
    raw = {"bare_report": bare, "guarded_report": guarded}

    result = overhead_mod.compute_quality_overhead_from_guard(
        raw,
        "accuracy",
        compute_primary_metric_from_report_fn=lambda report, kind=None: report[
            "metrics"
        ]["primary_metric"],
        get_metric_fn=lambda kind: SimpleNamespace(direction="higher"),
    )
    assert result["basis"] == "delta_pp"
    assert result["kind"] == "accuracy"
    assert result["value"] == pytest.approx(10.0)


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

    info = provenance_mod.collect_backend_versions()
    assert info["torch"] == "1.0.0"
    assert info["device_name"] == "FakeGPU"
    assert info["sm_capability"] == "9.0"
