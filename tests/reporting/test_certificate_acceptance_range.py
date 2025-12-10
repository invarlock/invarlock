import pytest

from invarlock.reporting import certificate as cert


def test_resolve_pm_acceptance_range_branches(monkeypatch):
    # No explicit bounds -> empty payload
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)
    assert cert._resolve_pm_acceptance_range_from_report({}) == {}

    # Context-provided bounds with missing min and env overrides -> sanitize negative env values
    report_ctx = {
        "context": {
            "primary_metric": {"acceptance_range": {"min": None, "max": "1.3"}},
        }
    }
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "0")
    env_adjusted = cert._resolve_pm_acceptance_range_from_report(report_ctx)
    assert env_adjusted["min"] == pytest.approx(0.95)
    assert env_adjusted["max"] == pytest.approx(1.10)

    # Explicit context bounds without env overrides should pass through
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)
    passthrough = cert._resolve_pm_acceptance_range_from_report(
        {"context": {"pm_acceptance_range": {"min": 0.97, "max": 1.02}}}
    )
    assert passthrough == {"min": 0.97, "max": 1.02}

    # Env min > env max forces max to align with min for monotonicity
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "1.2")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "1.0")
    clamped = cert._resolve_pm_acceptance_range_from_report({})
    assert clamped == {"min": 1.2, "max": 1.2}
