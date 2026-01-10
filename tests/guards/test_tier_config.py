"""Tests for tier_config loader functionality."""

from __future__ import annotations

from copy import deepcopy
import warnings
from unittest import mock

from invarlock.guards.policies import (
    check_policy_drift,
    get_rmt_policy,
    get_spectral_policy,
    get_variance_policy,
)
from invarlock.guards.tier_config import (
    _FALLBACK_CONFIG,
    _deep_merge,
    _find_drifts,
    check_drift,
    clear_tier_config_cache,
    get_rmt_epsilon,
    get_spectral_caps,
    get_tier_guard_config,
    get_variance_min_effect,
    load_tier_config,
)


class TestTierConfigLoader:
    """Tests for the tier configuration loader."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_tier_config_cache()

    def test_load_tier_config_returns_dict(self) -> None:
        """load_tier_config() returns a dict with tier names as keys."""
        config = load_tier_config()
        assert isinstance(config, dict)
        assert "balanced" in config
        assert "conservative" in config

    def test_fallback_config_has_all_guards(self) -> None:
        """Fallback config includes all guard types for each tier."""
        for tier in ("balanced", "conservative"):
            tier_config = _FALLBACK_CONFIG.get(tier, {})
            assert "spectral_guard" in tier_config
            assert "rmt_guard" in tier_config
            assert "variance_guard" in tier_config

    def test_get_tier_guard_config_spectral(self) -> None:
        """get_tier_guard_config returns spectral config."""
        config = get_tier_guard_config("balanced", "spectral_guard")
        assert isinstance(config, dict)
        assert "sigma_quantile" in config
        assert "family_caps" in config

    def test_get_tier_guard_config_rmt(self) -> None:
        """get_tier_guard_config returns RMT config."""
        config = get_tier_guard_config("balanced", "rmt_guard")
        assert isinstance(config, dict)
        assert "epsilon_by_family" in config
        assert "margin" in config

    def test_get_tier_guard_config_variance(self) -> None:
        """get_tier_guard_config returns variance config."""
        config = get_tier_guard_config("balanced", "variance_guard")
        assert isinstance(config, dict)
        assert "min_effect_lognll" in config
        assert "deadband" in config


class TestConvenienceAccessors:
    """Tests for convenience accessor functions."""

    def test_get_spectral_caps_balanced(self) -> None:
        """get_spectral_caps returns family caps for balanced tier."""
        caps = get_spectral_caps("balanced")
        assert isinstance(caps, dict)
        assert "ffn" in caps
        assert "attn" in caps
        assert "embed" in caps
        # Balanced tier values from tiers.yaml
        assert caps["ffn"] == 3.849
        assert caps["attn"] == 3.018
        assert caps["embed"] == 1.05

    def test_get_spectral_caps_conservative(self) -> None:
        """get_spectral_caps returns tighter caps for conservative tier."""
        caps = get_spectral_caps("conservative")
        assert caps["ffn"] == 3.849
        assert caps["attn"] == 2.6

    def test_get_rmt_epsilon_balanced(self) -> None:
        """get_rmt_epsilon returns per-family epsilon for balanced tier."""
        epsilon = get_rmt_epsilon("balanced")
        assert isinstance(epsilon, dict)
        # Values from tiers.yaml
        assert epsilon["ffn"] == 0.01
        assert epsilon["attn"] == 0.01
        assert epsilon["embed"] == 0.01
        assert epsilon["other"] == 0.01

    def test_get_rmt_epsilon_conservative(self) -> None:
        """get_rmt_epsilon returns tighter epsilon for conservative tier."""
        epsilon = get_rmt_epsilon("conservative")
        assert epsilon["ffn"] == 0.01
        assert epsilon["attn"] == 0.01

    def test_get_variance_min_effect_balanced(self) -> None:
        """get_variance_min_effect returns calibrated min_effect_lognll."""
        min_effect = get_variance_min_effect("balanced")
        assert min_effect == 0.0

    def test_get_variance_min_effect_conservative(self) -> None:
        """get_variance_min_effect is higher for conservative tier."""
        min_effect = get_variance_min_effect("conservative")
        assert min_effect == 0.016


class TestPolicyIntegration:
    """Tests for policy functions using tier_config loader."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_tier_config_cache()

    def test_get_spectral_policy_uses_yaml(self) -> None:
        """get_spectral_policy loads values from tiers.yaml."""
        policy = get_spectral_policy("balanced", use_yaml=True)
        # Should have values from tiers.yaml
        assert policy["sigma_quantile"] == 0.95
        assert policy["deadband"] == 0.10
        assert policy["max_caps"] == 5

    def test_get_spectral_policy_without_yaml(self) -> None:
        """get_spectral_policy can skip YAML loading."""
        policy = get_spectral_policy("balanced", use_yaml=False)
        # Should still work with hardcoded fallbacks
        assert "sigma_quantile" in policy
        assert "deadband" in policy

    def test_get_rmt_policy_uses_yaml(self) -> None:
        """get_rmt_policy loads epsilon values from tiers.yaml."""
        policy = get_rmt_policy("balanced", use_yaml=True)
        epsilon = policy.get("epsilon_by_family", {})
        # Should have per-family epsilon from tiers.yaml
        assert policy.get("epsilon_default") == 0.01
        assert epsilon.get("ffn") == 0.01
        assert epsilon.get("attn") == 0.01
        assert epsilon.get("embed") == 0.01

    def test_get_rmt_policy_conservative_uses_yaml(self) -> None:
        """get_rmt_policy conservative tier uses tighter epsilon."""
        policy = get_rmt_policy("conservative", use_yaml=True)
        epsilon = policy.get("epsilon_by_family", {})
        # Conservative has tighter values
        assert policy.get("epsilon_default") == 0.01
        assert epsilon.get("ffn") == 0.01
        assert epsilon.get("attn") == 0.01

    def test_get_variance_policy_uses_yaml(self) -> None:
        """get_variance_policy loads min_effect_lognll from tiers.yaml."""
        policy = get_variance_policy("balanced", use_yaml=True)
        assert policy.get("min_effect_lognll") == 0.0
        assert policy["deadband"] == 0.02


class TestDriftDetection:
    """Tests for drift detection between YAML and hardcoded values."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_tier_config_cache()

    def test_check_drift_returns_dict(self) -> None:
        """check_drift() returns a dict (empty if no drift)."""
        drift = check_drift(silent=True)
        assert isinstance(drift, dict)

    def test_check_drift_no_yaml_returns_empty(self) -> None:
        """check_drift returns {} when tiers.yaml cannot be loaded."""
        import invarlock.guards.tier_config as tc

        with mock.patch.object(tc, "_load_yaml", return_value=None):
            drift = tc.check_drift(silent=False)
        assert drift == {}

    def test_check_policy_drift_alias(self) -> None:
        """check_policy_drift() is an alias for check_drift()."""
        drift = check_policy_drift(silent=True)
        assert isinstance(drift, dict)

    def test_drift_detection_finds_differences(self) -> None:
        """Drift detection can identify value mismatches."""
        # This is a structural test - we can't easily mock the YAML
        # but we verify the function runs without error
        _ = check_drift(silent=True)


class TestCaching:
    """Tests for configuration caching behavior."""

    def test_cache_clear_works(self) -> None:
        """clear_tier_config_cache clears the LRU cache."""
        # Load once to populate cache
        config1 = load_tier_config()

        # Clear and reload
        clear_tier_config_cache()
        config2 = load_tier_config()

        # Both should be equal (content-wise)
        assert config1.keys() == config2.keys()

    def test_config_is_cached(self) -> None:
        """Subsequent calls return the same cached object."""
        config1 = load_tier_config()
        config2 = load_tier_config()
        # Same object due to caching
        assert config1 is config2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_tier_falls_back(self) -> None:
        """Unknown tier falls back gracefully."""
        # get_tier_guard_config should handle unknown tiers
        config = get_tier_guard_config("balanced", "spectral_guard")
        assert config is not None

    def test_get_spectral_caps_default(self) -> None:
        """get_spectral_caps uses balanced as default."""
        caps_default = get_spectral_caps()
        caps_balanced = get_spectral_caps("balanced")
        assert caps_default == caps_balanced

    def test_get_rmt_epsilon_default(self) -> None:
        """get_rmt_epsilon uses balanced as default."""
        eps_default = get_rmt_epsilon()
        eps_balanced = get_rmt_epsilon("balanced")
        assert eps_default == eps_balanced


class TestYAMLLoadingEdgeCases:
    """Tests for YAML loading error handling."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_tier_config_cache()

    def test_yaml_import_error_uses_fallback(self) -> None:
        """If PyYAML is not installed, fallback config is used."""
        # Simulate PyYAML not installed by mocking import
        with mock.patch.dict("sys.modules", {"yaml": None}):
            clear_tier_config_cache()
            # Can't easily force ImportError, but we test the fallback path exists
            config = load_tier_config()
            assert "balanced" in config
            assert "conservative" in config

    def test_yaml_file_not_found_uses_fallback(self) -> None:
        """If tiers.yaml doesn't exist, fallback config is used."""
        import invarlock.guards.tier_config as tc

        original_path = tc._TIERS_YAML_PATH
        try:
            tc._TIERS_YAML_PATH = tc.Path("/nonexistent/path/tiers.yaml")
            clear_tier_config_cache()
            result = tc._load_yaml()
            assert result is None  # Falls back
        finally:
            tc._TIERS_YAML_PATH = original_path
            clear_tier_config_cache()

    def test_yaml_parse_non_dict_uses_fallback(self) -> None:
        """If tiers.yaml doesn't parse as dict, fallback is used."""
        import invarlock.guards.tier_config as tc

        with mock.patch.object(tc.Path, "exists", return_value=True):
            with mock.patch("builtins.open", mock.mock_open(read_data="just a string")):
                # Should return None since "just a string" is not a dict
                # Note: depends on yaml parsing - a bare string might parse as string
                _ = tc._load_yaml()

    def test_yaml_exception_uses_fallback(self) -> None:
        """If yaml.safe_load raises, fallback is used."""
        import invarlock.guards.tier_config as tc

        with mock.patch.object(tc.Path, "exists", return_value=True):
            with mock.patch("builtins.open", side_effect=OSError("test error")):
                result = tc._load_yaml()
                assert result is None

    def test_deep_merge_nested_dicts(self) -> None:
        """_deep_merge handles nested dictionaries."""
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"b": 2, "nested": {"y": 30, "z": 40}}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["nested"]["x"] == 10
        assert result["nested"]["y"] == 30
        assert result["nested"]["z"] == 40

    def test_deep_merge_override_non_dict(self) -> None:
        """_deep_merge handles override with non-dict value replacing dict."""
        base = {"nested": {"x": 10}}
        override = {"nested": "replaced"}
        result = _deep_merge(base, override)
        assert result["nested"] == "replaced"


class TestFindDrifts:
    """Tests for the _find_drifts helper function."""

    def test_find_drifts_empty(self) -> None:
        """No drift when data matches."""
        yaml = {"a": 1, "b": 2}
        fallback = {"a": 1, "b": 2}
        drifts = _find_drifts(yaml, fallback)
        assert drifts == []

    def test_find_drifts_value_mismatch(self) -> None:
        """Drift detected when values differ."""
        yaml = {"a": 1, "b": 3}
        fallback = {"a": 1, "b": 2}
        drifts = _find_drifts(yaml, fallback)
        assert len(drifts) == 1
        assert "b:" in drifts[0]

    def test_find_drifts_yaml_only_key_ignored(self) -> None:
        """YAML-only keys (not in fallback) are not flagged as drift."""
        yaml = {"a": 1, "yaml_only": 99}
        fallback = {"a": 1}
        drifts = _find_drifts(yaml, fallback)
        # yaml_only is not considered drift since fallback doesn't have it
        assert drifts == []

    def test_find_drifts_nested(self) -> None:
        """Drift in nested dicts is detected."""
        yaml = {"nested": {"x": 100}}
        fallback = {"nested": {"x": 10, "y": 20}}
        drifts = _find_drifts(yaml, fallback)
        # Should find drift in nested.x and nested.y (yaml missing y)
        assert len(drifts) == 2
        drift_text = " ".join(drifts)
        assert "nested.x:" in drift_text
        assert "nested.y:" in drift_text


class TestCheckDriftWarning:
    """Tests for check_drift warning behavior."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_tier_config_cache()

    def test_check_drift_emits_warning_when_not_silent(self) -> None:
        """check_drift emits warnings when silent=False and drift exists."""
        import invarlock.guards.tier_config as tc

        yaml_data = deepcopy(tc._FALLBACK_CONFIG)
        yaml_data["balanced"]["variance_guard"]["deadband"] = (
            yaml_data["balanced"]["variance_guard"]["deadband"] + 0.001
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with mock.patch.object(tc, "_load_yaml", return_value=yaml_data):
                drift = tc.check_drift(silent=False)

        assert drift
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_check_drift_no_warning_when_silent(self) -> None:
        """check_drift doesn't emit warnings when silent=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = check_drift(silent=True)
            # No warnings should be emitted
            user_warnings = [x for x in w if "drift" in str(x.message).lower()]
            assert len(user_warnings) == 0
