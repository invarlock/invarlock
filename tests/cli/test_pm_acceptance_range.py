from invarlock.core.config_runtime import InvarLockConfig
from invarlock.core.run_policy import resolve_pm_acceptance_range
from invarlock.reporting.report_validation import compute_validation_flags


def test_pm_acceptance_range_ignores_env_override(monkeypatch):
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "0.9")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "1.2")
    cfg = InvarLockConfig(
        {"primary_metric": {"acceptance_range": {"min": 0.95, "max": 1.05}}}
    )
    resolved = resolve_pm_acceptance_range(cfg)

    assert resolved["min"] == 0.95
    assert resolved["max"] == 1.05


def test_evaluation_report_acceptance_range_applied():
    base_kwargs = {
        "ppl": {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.12},
        "spectral": {},
        "rmt": {"stable": True},
        "invariants": {"status": "pass"},
        "_ppl_metrics": {"preview_total_tokens": 60000, "final_total_tokens": 60000},
        "guard_overhead": {},
        "primary_metric": {"ratio_vs_baseline": 1.12},
        "moe": {},
        "dataset_capacity": {"tokens_available": 120000},
    }

    relaxed_flags = compute_validation_flags(
        **base_kwargs,
        tier="balanced",
        target_ratio=None,
        pm_acceptance_range={"min": 0.95, "max": 1.15},
    )
    assert relaxed_flags["primary_metric_acceptable"] is True

    strict_flags = compute_validation_flags(
        **base_kwargs,
        tier="balanced",
        target_ratio=None,
        pm_acceptance_range={"min": 0.95, "max": 1.10},
    )
    assert strict_flags["primary_metric_acceptable"] is False
