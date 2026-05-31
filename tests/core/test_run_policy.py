from __future__ import annotations

from typing import Any

import pytest

from invarlock.core import run_policy as policy
from invarlock.core.exceptions import ConfigError


class _BadFloat:
    def __float__(self) -> float:
        raise TypeError("bad-float")


def test_resolve_pm_drift_band_exception_paths(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="drift_band.max"):
        policy.resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": 0.9, "max": _BadFloat()}}}
        )

    def _boom(_obj: object) -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        policy.resolve_pm_drift_band(
            {"primary_metric": {"drift_band": {"min": 0.9, "max": 1.1}}},
            coerce_mapping_fn=_boom,
        )


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


def test_coerce_mapping_propagates_model_dump_errors() -> None:
    class _BadDump:
        def model_dump(self) -> dict[str, object]:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        policy.coerce_mapping(_BadDump())


def test_resolve_guard_overhead_threshold_and_tier_target_exceptions(
    monkeypatch,
) -> None:
    def _boom(_obj: object) -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        policy.resolve_guard_overhead_threshold(
            {"primary_metric": {"overhead_threshold": 0.2}},
            coerce_mapping_fn=_boom,
        )
    with pytest.raises(RuntimeError, match="boom"):
        policy.resolve_guard_overhead_threshold(
            {"primary_metric": {"overhead_threshold": 0.2}},
            default_threshold=0.02,
            coerce_mapping_fn=_boom,
        )

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
    with pytest.raises(ConfigError, match="min_tokens"):
        policy.resolve_pm_min_tokens_target(tier=None, profile=None)


def test_skip_policy_and_bool_coercion_edges() -> None:
    assert policy.coerce_bool_like(True) is True
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


def test_mapping_coercion_helpers_cover_data_dump_vars_and_fallbacks() -> None:
    class _WithData:
        def __init__(self) -> None:
            self._data = {"source": "data"}

    class _WithDump:
        def model_dump(self) -> dict[str, str]:
            return {"source": "dump"}

    class _WithVars:
        def __init__(self) -> None:
            self.source = "vars"

    class _DataRaisesAttributeError:
        def __init__(self) -> None:
            self.source = "vars-after-data"

        def __getattribute__(self, name: str) -> object:
            if name == "_data":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    class _WeirdDict:
        @property
        def __dict__(self):  # noqa: D401
            return []

    assert policy._coerce_optional_float(None) is None
    assert policy._ensure_mapping({"ok": True}) == {"ok": True}
    assert policy._ensure_mapping(["bad"]) == {}
    assert policy.coerce_mapping(_WithData()) == {"source": "data"}
    assert policy.coerce_mapping(_WithDump()) == {"source": "dump"}
    assert policy.coerce_mapping(_WithVars()) == {"source": "vars"}
    assert policy.coerce_mapping(_DataRaisesAttributeError()) == {
        "source": "vars-after-data"
    }
    assert policy.coerce_mapping(_WeirdDict()) == {}
    assert policy.coerce_mapping(object()) == {}


def test_coerce_mapping_falls_through_when_data_attr_is_unreadable() -> None:
    class _DataRaisesAttributeError:
        def __init__(self) -> None:
            self.source = "vars-after-data-error"

        def __getattribute__(self, name: str) -> object:
            if name == "_data":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    assert policy.coerce_mapping(_DataRaisesAttributeError()) == {
        "source": "vars-after-data-error"
    }


@pytest.mark.parametrize(
    ("cfg", "path_fragment"),
    [
        ({"primary_metric": {"acceptance_range": []}}, "acceptance_range"),
        (
            {"primary_metric": {"acceptance_range": {"min": _BadFloat()}}},
            "acceptance_range.min",
        ),
        (
            {"primary_metric": {"acceptance_range": {"max": _BadFloat()}}},
            "acceptance_range.max",
        ),
        (
            {"primary_metric": {"acceptance_range": {"min": 0.0}}},
            "acceptance_range.min",
        ),
        (
            {"primary_metric": {"acceptance_range": {"max": 0.0}}},
            "acceptance_range.max",
        ),
        (
            {"primary_metric": {"acceptance_range": {"min": 1.2, "max": 1.1}}},
            "acceptance_range",
        ),
    ],
)
def test_resolve_pm_acceptance_range_rejects_invalid_inputs(
    cfg: dict[str, Any],
    path_fragment: str,
) -> None:
    with pytest.raises(ConfigError, match=path_fragment):
        policy.resolve_pm_acceptance_range(cfg)


def test_resolve_pm_acceptance_range_handles_empty_and_absent_bounds() -> None:
    assert policy.resolve_pm_acceptance_range(None) == {}
    assert policy.resolve_pm_acceptance_range({"primary_metric": {}}) == {}
    assert (
        policy.resolve_pm_acceptance_range({"primary_metric": {"acceptance_range": {}}})
        == {}
    )


@pytest.mark.parametrize(
    ("cfg", "path_fragment"),
    [
        ({"primary_metric": {"drift_band": "bad"}}, "drift_band"),
        (
            {"primary_metric": {"drift_band": [0.9, _BadFloat()]}},
            "drift_band",
        ),
        (
            {"primary_metric": {"drift_band": {"min": 0.0, "max": 1.1}}},
            "drift_band.min",
        ),
        (
            {"primary_metric": {"drift_band": {"min": 0.9, "max": 0.0}}},
            "drift_band.max",
        ),
        (
            {"primary_metric": {"drift_band": {"min": 1.1, "max": 1.1}}},
            "drift_band",
        ),
    ],
)
def test_resolve_pm_drift_band_rejects_invalid_inputs(
    cfg: dict[str, Any],
    path_fragment: str,
) -> None:
    with pytest.raises(ConfigError, match=path_fragment):
        policy.resolve_pm_drift_band(cfg)


def test_resolve_pm_drift_band_covers_empty_and_sequence_forms() -> None:
    assert policy.resolve_pm_drift_band({"primary_metric": {}}) == {}
    assert policy.resolve_pm_drift_band({"primary_metric": {"drift_band": {}}}) == {}
    assert policy.resolve_pm_drift_band(
        {"primary_metric": {"drift_band": [0.9, 1.2]}}
    ) == {"min": 0.9, "max": 1.2}


def test_guard_overhead_threshold_and_dataset_split_helpers() -> None:
    assert policy.resolve_guard_overhead_threshold(None, default_threshold=0.02) == 0.02
    assert (
        policy.resolve_guard_overhead_threshold(
            {"primary_metric": {"overhead_threshold": 0.5}}
        )
        == 0.5
    )
    with pytest.raises(ConfigError, match="overhead_threshold"):
        policy.resolve_guard_overhead_threshold(
            {"primary_metric": {"overhead_threshold": -1}}
        )

    assert policy.choose_dataset_split(requested="test", available=["validation"]) == (
        "test",
        False,
    )
    assert policy.choose_dataset_split(
        requested=None,
        available=["train", "eval"],
    ) == ("eval", True)
    assert policy.choose_dataset_split(
        requested="",
        available=["zzz", "aaa"],
    ) == ("aaa", True)
    assert policy.choose_dataset_split(requested=None, available=None) == (
        "validation",
        True,
    )


def test_resolve_pm_min_tokens_target_rejects_negative_values(monkeypatch) -> None:
    monkeypatch.setattr(
        policy,
        "resolve_tier_policies",
        lambda *args, **kwargs: {"metrics": {"pm_ratio": {"min_tokens": -1}}},
    )

    with pytest.raises(ConfigError, match="min_tokens"):
        policy.resolve_pm_min_tokens_target(tier="balanced", profile="ci")
