from __future__ import annotations

import pytest

from invarlock.core.exceptions import ConfigError
from invarlock.core.run_policy import resolve_pm_drift_band


def test_resolve_pm_drift_band_returns_empty_without_explicit_config(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    assert resolve_pm_drift_band(None) == {}
    assert resolve_pm_drift_band({}) == {}
    assert resolve_pm_drift_band({"primary_metric": {"drift_band": None}}) == {}


def test_resolve_pm_drift_band_parses_cfg_dict_and_ignores_env(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    cfg = {"primary_metric": {"drift_band": {"min": "bad", "max": "1.20"}}}
    with pytest.raises(ConfigError, match="drift_band.min"):
        resolve_pm_drift_band(cfg)

    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "0")
    with pytest.raises(ConfigError, match="drift_band.min"):
        resolve_pm_drift_band(cfg)


def test_resolve_pm_drift_band_parses_list_variant_and_invalid_values(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    out = resolve_pm_drift_band({"primary_metric": {"drift_band": [0.9, 1.2]}})
    assert out == {"min": 0.9, "max": 1.2}

    cfg = {"primary_metric": {"drift_band": ["bad", "1.2"]}}
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MIN", "0.9")
    monkeypatch.setenv("INVARLOCK_PM_DRIFT_MAX", "1.1")
    with pytest.raises(ConfigError, match="drift_band"):
        resolve_pm_drift_band(cfg)


def test_resolve_pm_drift_band_accepts_partial_cfg_dict(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    out = resolve_pm_drift_band({"primary_metric": {"drift_band": {"max": 1.2}}})
    assert out == {"min": 0.95, "max": 1.2}

    out2 = resolve_pm_drift_band({"primary_metric": {"drift_band": {"min": 0.9}}})
    assert out2 == {"min": 0.9, "max": 1.05}


def test_resolve_pm_drift_band_clamps_invalid_bounds(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_DRIFT_MAX", raising=False)

    with pytest.raises(ConfigError, match="drift_band.min"):
        resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": -0.1, "max": 1.2}}}
        )

    with pytest.raises(ConfigError, match="drift_band.max"):
        resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": 0.9, "max": 0.0}}}
        )

    with pytest.raises(ConfigError, match="must be less than max"):
        resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": 1.2, "max": 1.1}}}
        )
