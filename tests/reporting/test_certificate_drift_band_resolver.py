import pytest

from invarlock.reporting import certificate as cert


def test_resolve_pm_drift_band_from_report_branches(monkeypatch):
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    assert cert._resolve_pm_drift_band_from_report({}) == {}

    report_ctx = {
        "context": {"primary_metric": {"drift_band": {"min": "bad", "max": "1.20"}}}
    }
    out = cert._resolve_pm_drift_band_from_report(report_ctx)
    assert out == {"min": pytest.approx(0.95), "max": pytest.approx(1.2)}

    report_list = {"context": {"primary_metric": {"drift_band": [0.9, 1.2]}}}
    out2 = cert._resolve_pm_drift_band_from_report(report_list)
    assert out2 == {"min": 0.9, "max": 1.2}

    report_alt = {"context": {"pm_drift_band": {"min": 0.9, "max": 1.1}}}
    out3 = cert._resolve_pm_drift_band_from_report(report_alt)
    assert out3 == {"min": 0.9, "max": 1.1}

    report_meta = {"meta": {"pm_drift_band": {"min": 0.9, "max": 1.1}}}
    out4 = cert._resolve_pm_drift_band_from_report(report_meta)
    assert out4 == {"min": 0.9, "max": 1.1}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "0")
    env_adjusted = cert._resolve_pm_drift_band_from_report(report_ctx)
    assert env_adjusted == {"min": 0.95, "max": 1.05}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "1.2")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "1.1")
    clamped = cert._resolve_pm_drift_band_from_report(report_ctx)
    assert clamped == {"min": 0.95, "max": 1.05}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "bad")
    out5 = cert._resolve_pm_drift_band_from_report(report_alt)
    assert out5 == {"min": 0.9, "max": 1.1}

    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)
    nonfinite = {
        "context": {"primary_metric": {"drift_band": {"min": float("nan"), "max": 1.2}}}
    }
    out6 = cert._resolve_pm_drift_band_from_report(nonfinite)
    assert out6 == {"min": 0.95, "max": 1.2}
