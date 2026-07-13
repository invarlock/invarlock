import pytest

import invarlock.guards as guards_pkg
import invarlock.guards.policies as guard_policies
from invarlock.core.exceptions import GuardError, PolicyViolationError, ValidationError
from invarlock.guards.policies import (
    create_custom_rmt_policy,
    create_custom_spectral_policy,
    create_custom_variance_policy,
    enforce_validation_gate,
    get_policy_for_model_size,
    get_rmt_policy,
    get_rmt_policy_for_model_size,
    get_spectral_policy,
    get_validation_gate,
    get_variance_policy,
    get_variance_policy_for_model_size,
)
from invarlock.guards.rmt_policy import apply_rmt_policy_overrides
from invarlock.guards.tier_config import (
    TierConfigError,
)
from invarlock.guards.tier_config import (
    get_tier_guard_config as get_packaged_tier_guard_config,
)

guards_pkg.policies = guard_policies


def _patch_tier_guard_config(monkeypatch: pytest.MonkeyPatch, replacement) -> None:
    monkeypatch.setattr(
        guard_policies,
        "get_tier_guard_config",
        replacement,
        raising=True,
    )


def test_get_variance_policy_overlay_and_load_failure(monkeypatch: pytest.MonkeyPatch):
    """Ensure variance policy overlays and packaged-policy failures are explicit."""

    def _fake_variance_tier(name: str, guard: str):
        assert guard == "variance_guard"
        assert name == "balanced"
        config = get_packaged_tier_guard_config("balanced", "variance_guard")
        config.update(
            {
                "deadband": 0.07,
                "min_effect_lognll": 0.002,
                "min_abs_adjust": 0.03,
                "max_scale_step": 0.02,
                "topk_backstop": 3,
                "predictive_one_sided": True,
            }
        )
        return config

    # Overlay branch
    _patch_tier_guard_config(monkeypatch, _fake_variance_tier)
    overlaid = get_variance_policy("balanced")
    assert overlaid["deadband"] == 0.07
    assert overlaid["min_effect_lognll"] == 0.002
    assert overlaid["min_abs_adjust"] == 0.03
    assert overlaid["max_scale_step"] == 0.02
    assert overlaid["topk_backstop"] == 3

    # Partial policy objects are rejected rather than filled from constants.
    def _partial_variance_tier(name: str, guard: str):
        assert guard == "variance_guard"
        assert name == "balanced"
        return {"deadband": 0.09}

    _patch_tier_guard_config(monkeypatch, _partial_variance_tier)
    with pytest.raises(TierConfigError, match="packaged variance policy"):
        get_variance_policy("balanced")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("tiers lookup failed")

    _patch_tier_guard_config(monkeypatch, _boom)
    with pytest.raises(TierConfigError, match="packaged variance policy"):
        get_variance_policy("balanced")


def test_get_variance_policy_rejects_incomplete_packaged_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _variance_without_deadband(name: str, guard: str):
        assert guard == "variance_guard"
        assert name == "balanced"
        return {"min_effect_lognll": 0.004}

    _patch_tier_guard_config(monkeypatch, _variance_without_deadband)

    with pytest.raises(TierConfigError, match="packaged variance policy"):
        get_variance_policy("balanced")


def test_get_variance_policy_ignores_bool_numeric_overlay_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _invalid_variance_tier(name: str, guard: str):
        assert guard == "variance_guard"
        assert name == "balanced"
        return {
            "deadband": True,
            "min_effect_lognll": False,
            "min_abs_adjust": True,
            "max_scale_step": False,
            "topk_backstop": True,
            "predictive_one_sided": True,
        }

    _patch_tier_guard_config(monkeypatch, _invalid_variance_tier)

    with pytest.raises(TierConfigError, match="packaged variance policy"):
        get_variance_policy("balanced")


def test_get_spectral_policy_and_errors():
    assert get_spectral_policy("balanced")["deadband"] == 0.10
    from invarlock.core.exceptions import GuardError, ValidationError

    with pytest.raises(GuardError):
        get_spectral_policy("unknown")
    with pytest.raises(ValidationError):
        create_custom_spectral_policy(sigma_quantile=1.5)
    with pytest.raises(ValidationError):
        create_custom_spectral_policy(deadband=0.75)
    with pytest.raises(ValidationError):
        create_custom_spectral_policy(sigma_quantile=True)
    with pytest.raises(ValidationError):
        create_custom_spectral_policy(deadband=False)
    with pytest.raises(ValidationError):
        create_custom_spectral_policy(scope="bad")


def test_get_spectral_policy_overlays_and_load_failure(monkeypatch: pytest.MonkeyPatch):
    """Exercise spectral overlays and explicit packaged-policy failure."""

    def _fake_tier_config(name: str, guard: str):
        assert guard == "spectral_guard"
        assert name == "balanced"
        config = get_packaged_tier_guard_config("balanced", "spectral_guard")
        config.update(
            {
                "sigma_quantile": 0.77,
                "deadband": 0.09,
                "scope": "all",
                "max_caps": 9,
                "family_caps": {
                    "ffn": 4.0,
                    "attn": 3.0,
                    "embed": 2.0,
                    "other": 1.0,
                },
                "multiple_testing": {"method": "bh", "alpha": 0.01, "m": 3},
            }
        )
        return config

    _patch_tier_guard_config(monkeypatch, _fake_tier_config)
    overlay = get_spectral_policy("balanced")
    assert overlay["sigma_quantile"] == 0.77
    assert overlay["deadband"] == 0.09
    assert overlay["scope"] == "all"
    assert overlay["max_caps"] == 9
    assert overlay["family_caps"]["attn"] == {"kappa": 3.0}
    assert overlay["multiple_testing"]["alpha"] == 0.01

    # Partial policy objects are rejected rather than filled from constants.
    def _partial_tier_config(name: str, guard: str):
        assert guard == "spectral_guard"
        assert name == "balanced"
        return {"deadband": 0.11}

    _patch_tier_guard_config(monkeypatch, _partial_tier_config)
    with pytest.raises(TierConfigError, match="packaged spectral policy"):
        get_spectral_policy("balanced")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("tiers lookup failed")

    _patch_tier_guard_config(monkeypatch, _boom)
    with pytest.raises(TierConfigError, match="packaged spectral policy"):
        get_spectral_policy("balanced")


def test_get_spectral_policy_ignores_invalid_overlay_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _invalid_tier_config(name: str, guard: str):
        assert guard == "spectral_guard"
        assert name == "balanced"
        return {
            "sigma_quantile": True,
            "scope": "attn",
            "max_caps": False,
            "max_spectral_norm": 7.5,
        }

    _patch_tier_guard_config(monkeypatch, _invalid_tier_config)

    with pytest.raises(TierConfigError, match="packaged spectral policy"):
        get_spectral_policy("balanced")


def test_model_size_policy_selection():
    assert get_policy_for_model_size(50_000_000)["deadband"] == 0.15  # aggressive
    assert get_policy_for_model_size(500_000_000)["deadband"] == 0.10  # balanced
    assert get_policy_for_model_size(2_000_000_000)["deadband"] == 0.05  # conservative


def test_rmt_model_size_policy_selection():
    """Exercise RMT model-size helper across thresholds."""

    small = get_rmt_policy_for_model_size(50_000_000)
    medium = get_rmt_policy_for_model_size(500_000_000)
    large = get_rmt_policy_for_model_size(2_000_000_000)

    assert small["margin"] == 1.8  # aggressive
    assert medium["margin"] == 1.5  # balanced
    assert large["margin"] == 1.3  # conservative


def test_rmt_policy_and_errors():
    assert get_rmt_policy("balanced")["margin"] == 1.5
    with pytest.raises(GuardError):
        get_rmt_policy("nope")
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(q=-1.0)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(deadband=0.75)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(margin=0.5)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(q=True)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(deadband=True)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(margin=False)

    # Happy path: valid parameters produce a policy without raising.
    policy = create_custom_rmt_policy(q=0.5, deadband=0.25, margin=1.6, correct=False)
    assert isinstance(policy, dict)
    assert policy["q"] == 0.5
    assert policy["deadband"] == 0.25
    assert policy["margin"] == 1.6


def test_apply_rmt_policy_overrides_handles_invalid_and_partial_activation_config() -> (
    None
):
    class _Guard:
        def __init__(self) -> None:
            self.q = 0.5
            self.deadband = 0.1
            self.margin = 1.5
            self.correct = True
            self.epsilon_default = 0.08
            self.epsilon_by_family = {"attn": 0.08}
            self._require_activation = False
            self.estimator = {"type": "power_iter", "iters": 3, "init": "ones"}
            self.activation_sampling = {
                "windows": {"count": 4, "indices_policy": "evenly_spaced"}
            }

        def _set_epsilon_by_family(self, value) -> None:  # noqa: ANN001
            if isinstance(value, dict):
                self.epsilon_by_family.update(value)

        def _set_epsilon_default(self, value) -> None:  # noqa: ANN001
            self.epsilon_default = float(value)

    guard = _Guard()

    apply_rmt_policy_overrides(
        guard,
        {
            "q": object(),
            "estimator": {"iters": -2, "init": "bad"},
            "activation": {
                "sampling": {"windows": {"count": "bad", "indices_policy": "last"}}
            },
            "activation_required": True,
        },
    )

    assert guard.q == "auto"
    assert guard.estimator == {"type": "power_iter", "iters": 1, "init": "ones"}
    assert guard.activation_sampling["windows"] == {
        "count": 4,
        "indices_policy": "last",
    }
    assert guard._require_activation is True

    apply_rmt_policy_overrides(guard, {"activation": {"sampling": []}})
    apply_rmt_policy_overrides(guard, {"activation": {"sampling": {"windows": []}}})

    assert guard.activation_sampling["windows"] == {
        "count": 4,
        "indices_policy": "last",
    }


def test_apply_rmt_policy_overrides_normalizes_module_patterns() -> None:
    class _Guard:
        def __init__(self) -> None:
            self.q = 0.5
            self.deadband = 0.1
            self.margin = 1.5
            self.correct = True
            self.epsilon_default = 0.08
            self.epsilon_by_family = {}
            self.module_include_patterns = ()
            self.module_exclude_patterns = ()

        def _set_epsilon_by_family(self, value) -> None:  # noqa: ANN001
            if isinstance(value, dict):
                self.epsilon_by_family.update(value)

        def _set_epsilon_default(self, value) -> None:  # noqa: ANN001
            self.epsilon_default = float(value)

    guard = _Guard()

    apply_rmt_policy_overrides(
        guard,
        {
            "module_include_patterns": [" model.* ", "", "model.*", 7],
            "module_exclude_patterns": " *.audio_tower.* ",
        },
    )

    assert guard.module_include_patterns == ("model.*",)
    assert guard.module_exclude_patterns == ("*.audio_tower.*",)
    assert guard.epsilon_by_family == {
        "attn": 0.08,
        "ffn": 0.08,
        "embed": 0.08,
        "other": 0.08,
    }

    apply_rmt_policy_overrides(guard, {"module_include_patterns": None})
    assert guard.module_include_patterns == ()


def test_get_rmt_policy_overlay_and_load_failure(monkeypatch: pytest.MonkeyPatch):
    """Exercise RMT overlays and explicit packaged-policy failure."""

    def _fake_rmt_tier(name: str, guard: str):
        assert guard == "rmt_guard"
        assert name == "balanced"
        config = get_packaged_tier_guard_config("balanced", "rmt_guard")
        config.update(
            {
                "deadband": 0.21,
                "margin": 1.7,
                "epsilon_by_family": {
                    "ffn": 0.2,
                    "attn": 0.3,
                    "embed": 0.4,
                    "other": 0.5,
                },
            }
        )
        return config

    _patch_tier_guard_config(monkeypatch, _fake_rmt_tier)
    overlaid = get_rmt_policy("balanced")
    assert overlaid["deadband"] == 0.21
    assert overlaid["margin"] == 1.7
    assert overlaid["epsilon_by_family"]["attn"] == 0.3

    # Partial policy objects are rejected rather than filled from constants.
    def _partial_rmt_tier(name: str, guard: str):
        assert guard == "rmt_guard"
        assert name == "balanced"
        return {"deadband": 0.19}

    _patch_tier_guard_config(monkeypatch, _partial_rmt_tier)
    with pytest.raises(TierConfigError, match="packaged RMT policy"):
        get_rmt_policy("balanced")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("tiers lookup failed")

    _patch_tier_guard_config(monkeypatch, _boom)
    with pytest.raises(TierConfigError, match="packaged RMT policy"):
        get_rmt_policy("balanced")


def test_get_rmt_policy_ignores_invalid_overlay_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _invalid_rmt_tier(name: str, guard: str):
        assert guard == "rmt_guard"
        assert name == "balanced"
        return {
            "margin": True,
            "epsilon_default": False,
            "epsilon_by_family": {"ffn": True},
        }

    _patch_tier_guard_config(monkeypatch, _invalid_rmt_tier)

    with pytest.raises(TierConfigError, match="packaged RMT policy"):
        get_rmt_policy("balanced")


def test_get_rmt_policy_ignores_bool_deadband_and_margin_overlay_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _invalid_rmt_tier(name: str, guard: str):
        assert guard == "rmt_guard"
        assert name == "balanced"
        return {
            "deadband": True,
            "margin": False,
            "epsilon_default": 0.22,
        }

    _patch_tier_guard_config(monkeypatch, _invalid_rmt_tier)

    with pytest.raises(TierConfigError, match="packaged RMT policy"):
        get_rmt_policy("balanced")


def test_empty_packaged_guard_objects_fail_instead_of_using_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tier_guard_config(monkeypatch, lambda *_a, **_k: {})

    for resolver in (get_spectral_policy, get_rmt_policy, get_variance_policy):
        with pytest.raises(TierConfigError, match="Failed to resolve packaged"):
            resolver("balanced")


def test_variance_policy_and_errors():
    # Model-size helper should route to different named policies
    small = get_variance_policy_for_model_size(50_000_000)
    medium = get_variance_policy_for_model_size(500_000_000)
    large = get_variance_policy_for_model_size(2_000_000_000)

    assert small["deadband"] == 0.12  # aggressive
    assert medium["deadband"] == 0.02  # balanced
    assert large["deadband"] == 0.03  # conservative

    with pytest.raises(ValidationError):
        create_custom_variance_policy(min_gain=1.5)
    with pytest.raises(ValidationError):
        create_custom_variance_policy(max_calib=20)
    with pytest.raises(ValidationError):
        create_custom_variance_policy(scope="x")
    with pytest.raises(ValidationError):
        create_custom_variance_policy(clamp=(2.0, 1.0))
    with pytest.raises(ValidationError):
        create_custom_variance_policy(deadband=0.75)
    with pytest.raises(ValidationError):
        create_custom_variance_policy(mode="x")
    with pytest.raises(ValidationError):
        create_custom_variance_policy(min_rel_gain=2.0)
    with pytest.raises(ValidationError):
        create_custom_variance_policy(alpha=2.0)

    # Happy path: valid parameters yield a policy without raising.
    policy = create_custom_variance_policy(
        min_gain=0.3,
        max_calib=200,
        scope="both",
        clamp=(0.5, 2.0),
        deadband=0.1,
        seed=123,
        mode="ci",
        min_rel_gain=0.01,
        alpha=0.2,
    )
    assert isinstance(policy, dict)
    assert policy["scope"] == "both"
    assert policy["clamp"] == (0.5, 2.0)


def test_validation_gate_unknown():
    assert get_validation_gate("strict")["require_branch_balance"] is True
    with pytest.raises(GuardError):
        get_validation_gate("nope")


def test_enforce_validation_gate_no_violations():
    """All metrics within limits should pass without raising."""
    gate = get_validation_gate("standard")
    metrics = {
        "caps_applied": 1,
        "total_layers": 10,
        "primary_metric_ratio": 1.01,
        "branch_balance_ok": True,
    }
    result = enforce_validation_gate(metrics, gate)
    assert result is None


def test_enforce_validation_gate_ignores_malformed_primary_metric_ratio() -> None:
    class Metrics:
        def get(self, key, default=None):
            if key == "caps_applied":
                return 0
            if key == "total_layers":
                return 1
            if key == "primary_metric_ratio":
                raise RuntimeError("bad ratio")
            return default

    gate = get_validation_gate("standard")

    assert enforce_validation_gate(Metrics(), gate) is None

    # total_layers == 0 exercises the false branch of the capping-rate check.
    assert (
        enforce_validation_gate(
            {
                "caps_applied": 5,
                "total_layers": 0,
                "primary_metric_ratio": 1.0,
                "branch_balance_ok": True,
            },
            gate,
        )
        is None
    )


def test_enforce_validation_gate_capping_and_metric_violations():
    gate = get_validation_gate("strict")

    # Capping rate violation
    metrics_caps = {
        "caps_applied": 6,
        "total_layers": 10,
        "primary_metric_ratio": 1.0,
        "branch_balance_ok": True,
    }
    with pytest.raises(PolicyViolationError) as exc_caps:
        enforce_validation_gate(metrics_caps, gate)
    details_caps = getattr(exc_caps.value, "details", {})
    violations_caps = details_caps.get("violations") or []
    assert any(v.get("type") == "capping_rate" for v in violations_caps)

    # Primary metric degradation violation
    metrics_pm = {
        "caps_applied": 0,
        "total_layers": 10,
        "primary_metric_ratio": 1.10,
        "branch_balance_ok": True,
    }
    with pytest.raises(PolicyViolationError) as exc_pm:
        enforce_validation_gate(metrics_pm, gate)
    details_pm = getattr(exc_pm.value, "details", {})
    violations_pm = details_pm.get("violations") or []
    assert any(v.get("type") == "primary_metric_degradation" for v in violations_pm)


def test_enforce_validation_gate_branch_balance_and_malformed_metrics():
    gate = get_validation_gate("standard")

    # Branch balance violation
    metrics_branch = {
        "caps_applied": 0,
        "total_layers": 10,
        "primary_metric_ratio": 1.0,
        "branch_balance_ok": False,
    }
    with pytest.raises(PolicyViolationError) as exc_branch:
        enforce_validation_gate(metrics_branch, gate)
    details_branch = getattr(exc_branch.value, "details", {})
    violations_branch = details_branch.get("violations") or []
    assert any(v.get("type") == "branch_balance" for v in violations_branch)

    # Malformed metrics for capping ratio should be tolerated
    malformed = {
        "caps_applied": "not-a-number",
        "total_layers": 10,
        "primary_metric_ratio": 1.0,
        "branch_balance_ok": True,
    }
    enforce_validation_gate(malformed, gate)

    # Malformed primary_metric_ratio should also be tolerated.
    malformed_ratio = {
        "caps_applied": 0,
        "total_layers": 10,
        "primary_metric_ratio": "not-a-number",
        "branch_balance_ok": True,
    }
    enforce_validation_gate(malformed_ratio, gate)

    # For a permissive gate, branch-balance violations may still fail on metric
    # degradation, but the branch-balance toggle should not add an extra violation.
    permissive = get_validation_gate("permissive")
    with pytest.raises(PolicyViolationError) as exc_perm:
        enforce_validation_gate(
            {
                "caps_applied": 0,
                "total_layers": 10,
                "primary_metric_ratio": 1.5,
                "branch_balance_ok": False,
            },
            permissive,
        )
    perm_details = getattr(exc_perm.value, "details", {})
    perm_violations = perm_details.get("violations") or []
    assert all(v.get("type") != "branch_balance" for v in perm_violations)


def test_enforce_validation_gate_ignores_bool_numeric_metrics() -> None:
    gate = get_validation_gate("standard")
    metrics = {
        "caps_applied": True,
        "total_layers": True,
        "primary_metric_ratio": True,
        "branch_balance_ok": True,
    }

    assert enforce_validation_gate(metrics, gate) is None
