from __future__ import annotations

import pytest

from invarlock.cli.commands import run as run_mod


def test_resolve_pm_drift_band_returns_empty_without_explicit_config(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    assert run_mod._resolve_pm_drift_band(None) == {}
    assert run_mod._resolve_pm_drift_band({}) == {}
    assert (
        run_mod._resolve_pm_drift_band({"primary_metric": {"drift_band": None}}) == {}
    )


def test_resolve_pm_drift_band_parses_cfg_dict_and_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    cfg = {"primary_metric": {"drift_band": {"min": "bad", "max": "1.20"}}}
    out = run_mod._resolve_pm_drift_band(cfg)
    assert out == {"min": 0.95, "max": 1.2}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "0")
    out2 = run_mod._resolve_pm_drift_band(cfg)
    assert out2 == {"min": 0.95, "max": 1.05}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "1.2")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "1.1")
    out3 = run_mod._resolve_pm_drift_band(cfg)
    assert out3 == {"min": 0.95, "max": 1.05}

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "bad")
    out4 = run_mod._resolve_pm_drift_band(cfg)
    assert out4["max"] == pytest.approx(1.2)


def test_resolve_pm_drift_band_parses_list_variant_and_invalid_values(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    out = run_mod._resolve_pm_drift_band({"primary_metric": {"drift_band": [0.9, 1.2]}})
    assert out == {"min": 0.9, "max": 1.2}

    cfg = {"primary_metric": {"drift_band": ["bad", "1.2"]}}
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "0.9")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "1.1")
    out2 = run_mod._resolve_pm_drift_band(cfg)
    assert out2 == {"min": 0.9, "max": 1.1}


def test_resolve_pm_drift_band_accepts_partial_cfg_dict(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    out = run_mod._resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"max": 1.2}}}
    )
    assert out == {"min": 0.95, "max": 1.2}

    out2 = run_mod._resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"min": 0.9}}}
    )
    assert out2 == {"min": 0.9, "max": 1.05}
