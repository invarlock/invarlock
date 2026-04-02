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
            }
        }
    )
    assert out_fallback == {"min": 0.95, "max": 1.2}

    _patch_float_with_bombs(monkeypatch)

    out_le = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": {
                    "acceptance_range": {"min": "bomb_le", "max": "bomb_le"}
                },
            }
        }
    )
    assert out_le == {"min": 0.95, "max": 1.1}

    out_lt = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": {
                    "acceptance_range": {"min": "bomb_lt", "max": "bomb_lt"}
                },
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
                "primary_metric": {"acceptance_range": {"min": 0, "max": -2}},
            }
        }
    )

    assert out == {"min": 0.95, "max": 1.1}


def test_resolve_pm_acceptance_range_ignores_non_mapping_primary_metric() -> None:
    out = policy.resolve_pm_acceptance_range_from_report(
        {"context": {"primary_metric": "not-a-mapping"}}
    )
    assert out == {}


def test_resolve_pm_acceptance_range_ignores_missing_primary_metric_context() -> None:
    out = policy.resolve_pm_acceptance_range_from_report({"context": {"other": {}}})
    assert out == {}


def test_resolve_pm_acceptance_range_ignores_unstructured_acceptance_range() -> None:
    out = policy.resolve_pm_acceptance_range_from_report(
        {"context": {"primary_metric": {"acceptance_range": "unstructured"}}}
    )
    assert out == {}


def test_resolve_pm_acceptance_range_clamps_and_normalizes_inverted_bounds() -> None:
    out = policy.resolve_pm_acceptance_range_from_report(
        {
            "context": {
                "primary_metric": {"acceptance_range": {"min": 0, "max": 0.8}},
            }
        }
    )

    assert out == {"min": 0.95, "max": 0.95}


def test_resolve_pm_drift_band_from_report_exception_paths(monkeypatch) -> None:
    out_alt = policy.resolve_pm_drift_band_from_report(
        {
            "context": {
                "primary_metric": {"drift_band": {"min": None, "max": _BadFloat()}},
            }
        }
    )
    assert out_alt == {}

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
        policy.resolve_tiny_relax_from_report(
            {"context": {"run": {"tiny_relax": "off"}}}
        )
        is False
    )
    assert policy.resolve_tiny_relax_from_report({"meta": {"auto": {}}}) is False


def test_resolve_tiny_relax_handles_falsey_string_branch() -> None:
    assert (
        policy.resolve_tiny_relax_from_report(
            {"context": {"run": {"tiny_relax": "off"}}}
        )
        is False
    )
    assert (
        policy.resolve_tiny_relax_from_report(
            {"context": {"eval": {"tiny_relax": "false"}}}
        )
        is False
    )
    assert (
        policy.resolve_tiny_relax_from_report(
            {"context": {"eval": {"tiny_relax": "false"}}}
        )
        is False
    )


def test_resolve_tiny_relax_prefers_run_context_and_falls_through_invalid_run() -> None:
    assert (
        policy.resolve_tiny_relax_from_report(
            {"context": {"run": {"tiny_relax": "off"}, "eval": {"tiny_relax": "on"}}}
        )
        is False
    )
    assert (
        policy.resolve_tiny_relax_from_report(
            {
                "context": {
                    "run": {"tiny_relax": "maybe"},
                    "eval": {"tiny_relax": "yes"},
                }
            }
        )
        is True
    )
