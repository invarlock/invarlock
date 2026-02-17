import pytest

from invarlock.reporting import report_builder as cert


def test_resolve_pm_acceptance_range_branches(monkeypatch):
    # No explicit bounds -> empty payload
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)
    assert cert._resolve_pm_acceptance_range_from_report({}) == {}

    # Context-provided bounds with missing min should use default minimum.
    report_ctx = {
        "context": {
            "primary_metric": {"acceptance_range": {"min": None, "max": "1.3"}},
        }
    }
    adjusted = cert._resolve_pm_acceptance_range_from_report(report_ctx)
    assert adjusted["min"] == pytest.approx(0.95)
    assert adjusted["max"] == pytest.approx(1.3)

    # Explicit context bounds without env overrides should pass through
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)
    passthrough = cert._resolve_pm_acceptance_range_from_report(
        {"context": {"pm_acceptance_range": {"min": 0.97, "max": 1.02}}}
    )
    assert passthrough == {"min": 0.97, "max": 1.02}

    # Invalid ordering from explicit report metadata is clamped for monotonicity.
    clamped = cert._resolve_pm_acceptance_range_from_report(
        {"context": {"pm_acceptance_range": {"min": 1.2, "max": 1.0}}}
    )
    assert clamped == {"min": 1.2, "max": 1.2}
