import pytest

from invarlock.core.auto_tuning import (
    EDIT_ADJUSTMENTS,
    TIER_POLICIES,
    _get_tier_description,
    get_tier_summary,
    resolve_tier_policies,
    validate_tier_config,
)


@pytest.mark.parametrize("tier", TIER_POLICIES.keys())
def test_resolve_tier_policies_returns_deep_copy(tier: str) -> None:
    """Mutating the resolved policy must not alter the tier defaults."""
    policies = resolve_tier_policies(tier)
    policies["spectral"]["deadband"] = 999

    # Base tier definitions remain unchanged.
    assert TIER_POLICIES[tier]["spectral"]["deadband"] != 999


def test_resolve_tier_policies_applies_edit_adjustments() -> None:
    tier = "balanced"
    edit_name = next(iter(EDIT_ADJUSTMENTS.keys()))
    adjustments = EDIT_ADJUSTMENTS[edit_name]

    policies = resolve_tier_policies(tier, edit_name)

    for guard_name, guard_overrides in adjustments.items():
        for key, expected in guard_overrides.items():
            assert policies[guard_name][key] == expected


def test_resolve_tier_policies_applies_explicit_overrides() -> None:
    overrides = {
        "spectral": {"deadband": 0.42},
        "new_guard": {"alpha": 0.1},
    }

    resolved = resolve_tier_policies("balanced", explicit_overrides=overrides)

    assert resolved["spectral"]["deadband"] == 0.42
    # Overrides for guards that do not exist should be added verbatim.
    assert resolved["new_guard"] == overrides["new_guard"]


def test_resolve_tier_policies_unknown_tier_raises() -> None:
    with pytest.raises(ValueError):
        resolve_tier_policies("not-a-tier")


def test_get_tier_summary_success() -> None:
    summary = get_tier_summary("balanced", "quant_rtn")

    assert summary["tier"] == "balanced"
    assert summary["edit_name"] == "quant_rtn"
    assert "policies" in summary
    # Balanced tier description is stable and human readable.
    assert summary["description"] == _get_tier_description("balanced")


def test_get_tier_summary_invalid_tier() -> None:
    summary = get_tier_summary("invalid", "quant_rtn")

    assert summary["error"].startswith("Unknown tier")
    assert "valid_tiers" in summary


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"tier": "balanced"}, (True, None)),
        ({"tier": "balanced", "enabled": True}, (True, None)),
        ({"tier": "balanced", "probes": 3}, (True, None)),
        (
            {"tier": "balanced", "enabled": "yes"},
            (False, "'enabled' must be a boolean"),
        ),
        (
            {"tier": "balanced", "probes": "more"},
            (False, "'probes' must be an integer"),
        ),
        ({"tier": "unknown"}, (False, "Invalid tier 'unknown'.")),
    ],
)
def test_validate_tier_config(config, expected) -> None:
    result = validate_tier_config(config)
    # Partial match for invalid tier message (includes list of options).
    if not expected[0] and expected[1].startswith("Invalid tier"):
        assert result[0] is False
        assert result[1].startswith("Invalid tier")
    else:
        assert result == expected


def test_validate_tier_config_requires_dict() -> None:
    assert validate_tier_config(None) == (False, "Config must be a dictionary")


def test_validate_tier_config_requires_tier_key() -> None:
    assert validate_tier_config({}) == (False, "Missing 'tier' in auto configuration")
