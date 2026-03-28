from __future__ import annotations

import builtins
from typing import Any

import pytest

from invarlock.core import run_policy as policy


class _BadFloat:
    def __float__(self) -> float:
        raise TypeError("bad-float")


class _BombLE:
    def __init__(self, value: float) -> None:
        self.value = value

    def __le__(self, _other: object) -> bool:
        raise RuntimeError("<= bomb")

    def __lt__(self, _other: object) -> bool:
        return False

    def __ge__(self, _other: object) -> bool:
        return False


class _BombLT:
    def __init__(self, value: float) -> None:
        self.value = value

    def __le__(self, _other: object) -> bool:
        return False

    def __lt__(self, _other: object) -> bool:
        raise RuntimeError("< bomb")

    def __ge__(self, _other: object) -> bool:
        return False


class _BombGE:
    def __init__(self, value: float) -> None:
        self.value = value

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
            return base_float(value.value)
        return base_float(value)

    monkeypatch.setattr(policy, "float", _float_with_bombs, raising=False)


def test_resolve_pm_acceptance_range_comparison_exception_paths(monkeypatch) -> None:
    _patch_float_with_bombs(monkeypatch)

    out_le = policy.resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"min": "bomb_le", "max": "bomb_le"}}}
    )
    assert out_le == {"min": 0.95, "max": 1.1}

    out_lt = policy.resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"min": "bomb_lt", "max": "bomb_lt"}}}
    )
    assert out_lt == {"min": 1.0, "max": 1.1}


def test_resolve_pm_drift_band_exception_paths(monkeypatch) -> None:
    out_parse = policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"min": 0.9, "max": _BadFloat()}}}
    )
    assert out_parse == {"min": 0.9, "max": 1.05}

    def _boom(_obj: object) -> dict[str, object]:
        raise RuntimeError("boom")

    assert (
        policy.resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": 0.9, "max": 1.1}}},
            coerce_mapping_fn=_boom,
        )
        == {}
    )

    _patch_float_with_bombs(monkeypatch)

    out_le = policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"min": "bomb_le", "max": "bomb_le"}}}
    )
    assert out_le == {"min": 0.95, "max": 1.05}

    out_ge = policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"min": "bomb_ge", "max": "bomb_ge"}}}
    )
    assert out_ge == {"min": 0.95, "max": 1.05}


def test_resolve_pm_acceptance_range_and_drift_band_with_explicit_mapper() -> None:
    def _mapper(obj: object) -> dict[str, Any]:
        return obj if isinstance(obj, dict) else {}

    acceptance_min = policy.resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"min": 1.0}}},
        coerce_mapping_fn=_mapper,
    )
    assert acceptance_min == {"min": 1.0, "max": 1.1}

    acceptance_max = policy.resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"max": 1.2}}},
        coerce_mapping_fn=_mapper,
    )
    assert acceptance_max == {"min": 0.95, "max": 1.2}

    drift_min = policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"min": 0.9}}},
        coerce_mapping_fn=_mapper,
    )
    assert drift_min == {"min": 0.9, "max": 1.05}

    drift_max = policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": {"max": 1.2}}},
        coerce_mapping_fn=_mapper,
    )
    assert drift_max == {"min": 0.95, "max": 1.2}


def test_resolve_guard_overhead_threshold_and_tier_target_exceptions(
    monkeypatch,
) -> None:
    def _boom(_obj: object) -> dict[str, object]:
        raise RuntimeError("boom")

    assert policy.resolve_guard_overhead_threshold(
        {"primary_metric": {"overhead_threshold": 0.2}},
        coerce_mapping_fn=_boom,
    ) == pytest.approx(policy.GUARD_OVERHEAD_THRESHOLD)
    assert policy.resolve_guard_overhead_threshold(
        {"primary_metric": {"overhead_threshold": 0.2}},
        default_threshold=0.02,
        coerce_mapping_fn=_boom,
    ) == pytest.approx(0.02)

    monkeypatch.setattr(
        policy,
        "resolve_tier_policies",
        lambda *args, **kwargs: {"metrics": {"pm_ratio": {"min_tokens": "7"}}},
    )
    assert policy.resolve_pm_min_tokens_target(tier=None, profile=None) == 7

    monkeypatch.setattr(
        policy,
        "resolve_tier_policies",
        lambda *args, **kwargs: {"metrics": {"pm_ratio": {"min_tokens": "bad"}}},
    )
    assert policy.resolve_pm_min_tokens_target(tier=None, profile=None) == 0


def test_skip_policy_and_bool_coercion_edges() -> None:
    assert policy.coerce_bool_like(1) is True
    assert policy.coerce_bool_like(0) is False
    assert policy.coerce_bool_like("yes") is True
    assert policy.coerce_bool_like("off") is False
    assert policy.coerce_bool_like("unknown") is None

    assert policy.resolve_skip_overhead_policy(
        {}, coerce_mapping_fn=lambda _obj: []
    ) == (
        False,
        None,
    )

    cfg = {"context": {"eval": {"skip_overhead_check": "on"}}}
    assert policy.resolve_skip_overhead_policy(cfg) == (
        True,
        "config:context.eval.skip_overhead_check",
    )

    measure, skip, source = policy.should_measure_overhead("ci", cfg)
    assert (measure, skip, source) == (
        False,
        True,
        "config:context.eval.skip_overhead_check",
    )

    run_cfg = {"context": {"run": {"skip_overhead_check": "1"}}}
    assert policy.resolve_skip_overhead_policy(run_cfg) == (
        True,
        "config:context.run.skip_overhead_check",
    )
    assert policy.should_measure_overhead("dev", run_cfg) == (False, False, None)
