"""Fail-closed tests for the packaged tier policy authority."""

from __future__ import annotations

from copy import deepcopy
from unittest import mock

import pytest

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.guards.policies import (
    get_rmt_policy,
    get_spectral_policy,
    get_variance_policy,
)
from invarlock.guards.rmt_policy import get_rmt_policy as get_direct_rmt_policy
from invarlock.guards.tier_config import (
    TierConfigError,
    _load_yaml,
    _validate_tier_config,
    clear_tier_config_cache,
    get_rmt_epsilon,
    get_spectral_caps,
    get_tier_guard_config,
    get_variance_min_effect,
    load_tier_config,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_tier_config_cache()


def test_packaged_tier_policy_loads_all_current_tiers_and_guards() -> None:
    config = load_tier_config()

    assert set(config) == {"balanced", "conservative", "aggressive"}
    for tier in config.values():
        assert set(tier) == {"spectral_guard", "rmt_guard", "variance_guard"}


def test_packaged_accessors_return_current_values_and_isolated_copies() -> None:
    caps = get_spectral_caps("balanced")
    epsilon = get_rmt_epsilon("conservative")

    assert caps == {"ffn": 3.849, "attn": 3.018, "embed": 1.05, "other": 0.0}
    assert epsilon == {"ffn": 0.01, "attn": 0.01, "embed": 0.01, "other": 0.01}
    assert get_variance_min_effect("conservative") == 0.016

    caps["ffn"] = 99.0
    assert get_spectral_caps("balanced")["ffn"] == 3.849


def test_load_tier_config_is_cached_and_cache_can_be_cleared() -> None:
    first = load_tier_config()
    assert load_tier_config() is first

    clear_tier_config_cache()
    second = load_tier_config()

    assert second == first
    assert second is not first


@pytest.mark.parametrize("tier", ["none", "unknown", "", None])
def test_unknown_and_none_tiers_fail_instead_of_selecting_balanced(
    tier: object,
) -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        get_tier_guard_config(tier, "spectral_guard")  # type: ignore[arg-type]


def test_unknown_guard_fails_instead_of_returning_empty_policy() -> None:
    with pytest.raises(ValueError, match="Unknown guard"):
        get_tier_guard_config("balanced", "unknown")  # type: ignore[arg-type]


def test_missing_packaged_policy_file_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.guards.tier_config as tier_config

    monkeypatch.setattr(
        tier_config, "_TIERS_YAML_PATH", tier_config.Path("/missing/tiers.yaml")
    )

    with pytest.raises(TierConfigError, match="missing"):
        _load_yaml()


def test_missing_yaml_dependency_fails_explicitly() -> None:
    with mock.patch.dict("sys.modules", {"yaml": None}):
        with pytest.raises(TierConfigError, match="PyYAML is required"):
            _load_yaml()


def test_non_mapping_yaml_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import invarlock.guards.tier_config as tier_config

    policy = tmp_path / "tiers.yaml"
    policy.write_text("just a string\n", encoding="utf-8")
    monkeypatch.setattr(tier_config, "_TIERS_YAML_PATH", policy)

    with pytest.raises(TierConfigError, match="must be a mapping"):
        _load_yaml()


def test_yaml_read_or_parse_failure_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.guards.tier_config as tier_config

    monkeypatch.setattr(tier_config.Path, "is_file", lambda _self: True)
    monkeypatch.setattr(
        tier_config.Path,
        "read_text",
        lambda _self, **_kwargs: "balanced: [",
    )

    with pytest.raises(TierConfigError, match="Failed to load"):
        _load_yaml()


def test_policy_inventory_rejects_missing_or_unknown_tiers() -> None:
    valid = _load_yaml()

    missing = deepcopy(valid)
    missing.pop("balanced")
    with pytest.raises(TierConfigError, match="missing=.*balanced"):
        _validate_tier_config(missing)

    unknown = deepcopy(valid)
    unknown["none"] = deepcopy(valid["balanced"])
    with pytest.raises(TierConfigError, match="unknown=.*none"):
        _validate_tier_config(unknown)


def test_policy_inventory_rejects_missing_guard_or_required_key() -> None:
    valid = _load_yaml()

    missing_guard = deepcopy(valid)
    missing_guard["balanced"].pop("spectral_guard")
    with pytest.raises(TierConfigError, match="invalid section inventory"):
        _validate_tier_config(missing_guard)

    missing_key = deepcopy(valid)
    missing_key["balanced"]["rmt_guard"].pop("margin")
    with pytest.raises(TierConfigError, match="invalid key inventory"):
        _validate_tier_config(missing_key)

    unknown_key = deepcopy(valid)
    unknown_key["balanced"]["rmt_guard"]["legacy_default"] = 0.1
    with pytest.raises(TierConfigError, match="invalid key inventory"):
        _validate_tier_config(unknown_key)


def test_policy_inventory_rejects_non_mapping_tier_and_guard() -> None:
    valid = _load_yaml()

    invalid_tier = deepcopy(valid)
    invalid_tier["balanced"] = []
    with pytest.raises(TierConfigError, match="Tier 'balanced' must be a mapping"):
        _validate_tier_config(invalid_tier)

    invalid_guard = deepcopy(valid)
    invalid_guard["balanced"]["spectral_guard"] = []
    with pytest.raises(TierConfigError, match="must be a mapping"):
        _validate_tier_config(invalid_guard)


def _set_nested(data: dict, path: tuple[str, ...], value: object) -> None:
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("spectral_guard", "sigma_quantile"), True, "must be numeric"),
        (("spectral_guard", "scope"), "weights", "invalid scope"),
        (("spectral_guard", "correction_enabled"), 1, "must be boolean"),
        (("spectral_guard", "ignore_preview_inflation"), 0, "must be boolean"),
        (("spectral_guard", "max_spectral_norm"), "unbounded", "numeric or null"),
        (("spectral_guard", "max_caps"), True, "must be an integer"),
        (("spectral_guard", "family_caps"), {"ffn": 1.0}, "invalid inventory"),
        (
            ("spectral_guard", "family_caps", "ffn"),
            "wide",
            "values must be numeric",
        ),
        (("spectral_guard", "multiple_testing"), [], "must be a mapping"),
        (
            ("spectral_guard", "multiple_testing", "method"),
            "none",
            "multiple_testing is invalid",
        ),
        (("rmt_guard", "q"), object(), "q must be 'auto' or numeric"),
        (("rmt_guard", "margin"), False, "must be numeric"),
        (("rmt_guard", "correct"), 1, "must be boolean"),
        (("rmt_guard", "epsilon_by_family"), {}, "invalid inventory"),
        (("rmt_guard", "epsilon_by_family", "ffn"), True, "values must be numeric"),
        (("variance_guard", "alpha"), False, "must be numeric"),
        (("variance_guard", "max_calib"), True, "must be an integer"),
        (("variance_guard", "scope"), "all", "invalid scope"),
        (("variance_guard", "mode"), "estimate", "invalid mode"),
        (("variance_guard", "predictive_gate"), 1, "must be boolean"),
        (("variance_guard", "clamp"), [0.5], "two numeric bounds"),
        (("variance_guard", "tap"), [], "string or string list"),
        (("variance_guard", "calibration"), {}, "invalid inventory"),
        (
            ("variance_guard", "calibration", "windows"),
            True,
            "values must be integers",
        ),
    ],
)
def test_packaged_policy_rejects_semantically_invalid_guard_values(
    path: tuple[str, ...], value: object, message: str
) -> None:
    """Invalid policy values must fail before they can authorize a run."""

    invalid = deepcopy(_load_yaml())
    _set_nested(invalid["balanced"], path, value)

    with pytest.raises(TierConfigError, match=message):
        _validate_tier_config(invalid)


@pytest.mark.parametrize("metrics", [{}, [], None])
def test_packaged_policy_requires_nonempty_metric_authority(metrics: object) -> None:
    invalid = deepcopy(_load_yaml())
    invalid["balanced"]["metrics"] = metrics

    with pytest.raises(TierConfigError, match="metrics must be a non-empty mapping"):
        _validate_tier_config(invalid)


def test_guard_policy_resolvers_propagate_packaged_policy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.guards.policies as policies

    def _fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise TierConfigError("packaged policy unavailable")

    monkeypatch.setattr(policies, "get_tier_guard_config", _fail)

    for resolver in (get_spectral_policy, get_rmt_policy, get_variance_policy):
        with pytest.raises(TierConfigError, match="Failed to resolve packaged"):
            resolver("balanced")


def test_all_runtime_policy_resolution_paths_agree() -> None:
    for tier in ("balanced", "conservative", "aggressive"):
        resolved = resolve_tier_policies(tier)

        assert resolved["spectral"] == get_spectral_policy(tier)
        assert resolved["rmt"] == get_rmt_policy(tier)
        assert resolved["rmt"] == get_direct_rmt_policy(tier)
        assert resolved["variance"] == get_variance_policy(tier)
