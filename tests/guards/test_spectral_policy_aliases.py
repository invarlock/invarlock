import pytest
import torch.nn as nn

import invarlock.guards.spectral_policy as spectral_policy
from invarlock.core.exceptions import ValidationError
from invarlock.guards.spectral import SpectralGuard


def test_prepare_rejects_contraction_alias():
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard()
    with pytest.raises(ValueError, match=r"sigma_quantile"):
        guard.prepare(
            model,
            adapter=None,
            calib=None,
            policy={"contraction": 0.9, "family_caps": {"ffn": 2.0}},
        )


def test_normalize_family_caps_filters_invalid_entries_and_defaults_when_empty():
    mixed = spectral_policy.normalize_family_caps(
        {
            "ffn": {"kappa": 2.2, "bad": "ignore"},
            "attn": 3.4,
            "embed": True,
            "other": {"bad": object()},
        }
    )

    assert mixed["ffn"] == {"kappa": 2.2}
    assert mixed["attn"] == {"kappa": 3.4}
    assert "embed" not in mixed
    assert "other" not in mixed

    defaulted = spectral_policy.normalize_family_caps(
        {"ffn": {"kappa": float("inf")}, "attn": object()}
    )
    assert defaulted == spectral_policy.default_family_caps()


def test_policy_helpers_reject_non_mapping_and_can_omit_value_details():
    err = spectral_policy._policy_invalid("sigma_quantile", "bad")
    assert err.details == {"param": "sigma_quantile", "reason": "bad"}

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy._require_policy_mapping("family_caps", [])


@pytest.mark.parametrize(
    ("value", "kwargs"),
    [
        ("oops", {}),
        (float("inf"), {}),
        (-1, {"minimum": 0.0}),
        (2.0, {"maximum": 1.0}),
    ],
)
def test_require_policy_float_rejects_invalid_values(value, kwargs):
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy._require_policy_float("sigma_quantile", value, **kwargs)


@pytest.mark.parametrize(
    ("value", "checker", "param_name"),
    [
        (True, spectral_policy._require_policy_float, "sigma_quantile"),
        (False, spectral_policy._require_policy_float, "sigma_quantile"),
        (True, spectral_policy._require_policy_int, "max_caps"),
        (False, spectral_policy._require_policy_int, "max_caps"),
    ],
)
def test_require_policy_numeric_helpers_reject_bools(value, checker, param_name):
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        checker(param_name, value)


def test_normalize_multiple_testing_config_rejects_bad_method_and_zero_alpha():
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_multiple_testing_config({"method": "holm"})

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_multiple_testing_config({"alpha": 0.0})

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_multiple_testing_config({"alpha": True})

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_multiple_testing_config({"m": False})


def test_normalize_estimator_config_rejects_unknown_init():
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_estimator_config({"iters": 2, "init": "random"})

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_estimator_config({"iters": True})


def test_normalize_degeneracy_config_rejects_bool_ratios():
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_degeneracy_config(
            {"stable_rank": {"warn_ratio": True}}
        )

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.normalize_degeneracy_config(
            {"norm_collapse": {"fatal_ratio": False}}
        )


@pytest.mark.parametrize(
    ("policy", "param"),
    [
        ({"sigma_quantile": True}, "sigma_quantile"),
        ({"deadband": False}, "deadband"),
        ({"max_spectral_norm": True}, "max_spectral_norm"),
        ({"max_caps": True}, "max_caps"),
    ],
)
def test_apply_policy_overrides_rejects_bool_numeric_values(policy, param):
    guard = SpectralGuard()
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_policy.apply_policy_overrides(guard, policy)


def test_apply_policy_overrides_rejects_multipletesting_alias():
    guard = SpectralGuard()

    with pytest.raises(ValidationError) as excinfo:
        spectral_policy.apply_policy_overrides(
            guard,
            {"multipletesting": {"method": "bh", "alpha": 0.05, "m": 4}},
        )

    assert excinfo.value.details == {
        "param": "multipletesting",
        "hint": "Use spectral.multiple_testing instead.",
    }
