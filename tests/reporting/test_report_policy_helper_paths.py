from __future__ import annotations

import builtins
from typing import Any

from invarlock.reporting import report_policy as policy


class _BadFloat:
    def __float__(self) -> float:
        raise TypeError("bad-float")


class _BombLE(float):
    def __le__(self, _other: object) -> bool:
        raise RuntimeError("<= bomb")

    def __lt__(self, _other: object) -> bool:
        return False

    def __ge__(self, _other: object) -> bool:
        return False


class _BombLT(float):
    def __le__(self, _other: object) -> bool:
        return False

    def __lt__(self, _other: object) -> bool:
        raise RuntimeError("< bomb")

    def __ge__(self, _other: object) -> bool:
        return False


class _BombGE(float):
    def __le__(self, _other: object) -> bool:
        return False

    def __lt__(self, _other: object) -> bool:
        return False

    def __ge__(self, _other: object) -> bool:
        raise RuntimeError(">= bomb")


def _patch_float_with_bombs(monkeypatch) -> None:
    base_float = builtins.float

    def _float_with_bombs(value: Any) -> Any:
        if value == "bomb_le":
            return _BombLE(1.0)
        if value == "bomb_lt":
            return _BombLT(1.0)
        if value == "bomb_ge":
            return _BombGE(1.0)
        if isinstance(value, (_BombLE, _BombLT, _BombGE)):
            return base_float(value)
        return base_float(value)

    monkeypatch.setattr(policy, "float", _float_with_bombs, raising=False)


def test_resolve_pm_acceptance_range_from_report_exception_paths(monkeypatch) -> None:
    out_fallback = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": {"acceptance_range": {"min": _BadFloat(), "max": 1.2}}
            },
            "meta": {"pm_acceptance_range": {"min": 0.97, "max": 1.03}},
        }
    )
    assert out_fallback == {"min": 0.97, "max": 1.2}

    _patch_float_with_bombs(monkeypatch)

    out_le = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "pm_acceptance_range": {"min": "bomb_le", "max": "bomb_le"},
            }
        }
    )
    assert out_le == {"min": 0.95, "max": 1.1}

    out_lt = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "pm_acceptance_range": {"min": "bomb_lt", "max": "bomb_lt"},
            }
        }
    )
    assert out_lt == {"min": 1.0, "max": 1.1}

    # Explicit context bounds on both sides should skip fallback range lookup.
    out_ctx_complete = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": {"acceptance_range": {"min": 0.98, "max": 1.02}}
            }
        }
    )
    assert out_ctx_complete == {"min": 0.98, "max": 1.02}


def test_resolve_pm_acceptance_range_from_alt_context_non_dict_and_nonpositive() -> (
    None
):
    out = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": "not-a-dict",
                "pm_acceptance_range": {"min": 0, "max": -2},
            }
        }
    )

    assert out == {"min": 0.95, "max": 1.1}


def test_resolve_pm_drift_band_from_report_exception_paths(monkeypatch) -> None:
    out_alt = policy.resolve_pm_drift_band_from_report(
        {
            "context": {
                "primary_metric": {"drift_band": {"min": None, "max": _BadFloat()}},
                "pm_drift_band": {"min": 0.9, "max": 1.1},
            }
        }
    )
    assert out_alt == {"min": 0.9, "max": 1.1}

    out_meta = policy.resolve_pm_drift_band_from_report(
        {
            "context": {
                "primary_metric": {"drift_band": {"min": _BadFloat(), "max": None}},
            },
            "meta": {"pm_drift_band": {"min": 0.92, "max": 1.02}},
        }
    )
    assert out_meta == {"min": 0.92, "max": 1.02}

    _patch_float_with_bombs(monkeypatch)

    out_le = policy.resolve_pm_drift_band_from_report(
        {
            "context": {
                "primary_metric": {"drift_band": {"min": "bomb_le", "max": "bomb_le"}},
            }
        }
    )
    assert out_le == {"min": 0.95, "max": 1.05}

    out_ge = policy.resolve_pm_drift_band_from_report(
        {
            "context": {
                "primary_metric": {"drift_band": {"min": "bomb_ge", "max": "bomb_ge"}},
            }
        }
    )
    assert out_ge == {"min": 0.95, "max": 1.05}

    # Non-dict/non-list drift payload should take the explicit false branch.
    out_unstructured = policy.resolve_pm_drift_band_from_report(
        {"context": {"primary_metric": {"drift_band": "unstructured"}}}
    )
    assert out_unstructured == {}


def test_resolve_pm_drift_band_nonpositive_bounds_reset_to_defaults() -> None:
    out = policy.resolve_pm_drift_band_from_report(
        {"context": {"primary_metric": {"drift_band": {"min": 0, "max": -1}}}}
    )

    assert out == {"min": 0.95, "max": 1.05}


def test_resolve_pm_drift_band_inverted_bounds_reset_to_defaults() -> None:
    out = policy.resolve_pm_drift_band_from_report(
        {"context": {"primary_metric": {"drift_band": {"min": 1.2, "max": 1.1}}}}
    )

    assert out == {"min": 0.95, "max": 1.05}


def test_resolve_tiny_relax_from_report_edges() -> None:
    assert policy.resolve_tiny_relax_from_report(None) is False
    assert (
        policy.resolve_tiny_relax_from_report(
            {"context": {"eval": {"tiny_relax": "1"}}}
        )
        is True
    )
    assert (
        policy.resolve_tiny_relax_from_report({"auto": {"tiny_relax": "off"}}) is False
    )

    assert (
        policy.resolve_tiny_relax_from_report(
            {
                "auto": {"tiny_relax": "maybe"},
                "meta": {"auto": {"tiny_relax": "yes"}},
            }
        )
        is True
    )

    assert policy.resolve_tiny_relax_from_report({"meta": {"auto": {}}}) is False
    assert (
        policy.resolve_tiny_relax_from_report(
            {"auto": {"tiny_relax": "maybe"}, "meta": "not-a-dict"}
        )
        is False
    )
