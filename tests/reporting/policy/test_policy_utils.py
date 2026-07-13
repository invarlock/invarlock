from __future__ import annotations

import hashlib
import json

import pytest

import invarlock.reporting.policy_utils as policy_mod


class BadFloat(float):
    def __new__(cls, value: float = 1.0):
        return float.__new__(cls, value)

    def __float__(self):
        raise TypeError("bad float")


def test_compute_variance_policy_digest_handles_known_keys():
    policy = {"deadband": 0.1, "min_abs_adjust": 0.01, "unrelated": 5}
    digest = policy_mod._compute_variance_policy_digest(policy)
    assert digest and len(digest) == 16
    assert policy_mod._compute_variance_policy_digest({}) == ""


def test_compute_thresholds_payload_uses_tier_defaults(monkeypatch):
    fake_resolved = {"variance": {"min_effect_lognll": 0.2}}
    payload = policy_mod._compute_thresholds_payload("balanced", fake_resolved)
    assert payload["tier"] == "balanced"
    assert "pm_ratio" in payload
    assert payload["variance"]["min_effect_lognll"] == 0.2
    assert "accuracy" in payload
    assert set(payload["accuracy"]).issuperset(
        {"delta_min_pp", "min_examples", "min_examples_fraction", "hysteresis_delta_pp"}
    )
    assert isinstance(payload["accuracy"]["delta_min_pp"], float)
    assert isinstance(payload["accuracy"]["min_examples"], int)


def test_resolve_policy_tier_uses_only_canonical_metadata():
    report = {
        "meta": {"auto": {"tier": "Aggressive"}},
        "context": {"policy_tier": "conservative"},
    }
    assert policy_mod._resolve_policy_tier(report) == "aggressive"
    with pytest.raises(ValueError, match="meta.auto.tier"):
        policy_mod._resolve_policy_tier({})


def test_format_helpers():
    caps = {"famA": {"kappa": 0.5}, "famB": 0.3}
    formatted_caps = policy_mod._format_family_caps(caps)
    assert formatted_caps["famA"]["kappa"] == 0.5
    eps_map = policy_mod._format_epsilon_map({"famA": 0.1, "famB": "skip"})
    assert eps_map == {"famA": 0.1}


def test_format_helpers_handle_bad_values():
    caps = {"bad": {"kappa": BadFloat()}, "direct": BadFloat()}
    formatted = policy_mod._format_family_caps(caps)
    assert "bad" not in formatted and "direct" not in formatted
    eps = policy_mod._format_epsilon_map({"fam": BadFloat()})
    assert eps == {}


def test_build_resolved_policies_merges_inputs(monkeypatch):
    fake_tier = {
        "spectral": {
            "deadband": 0.2,
            "family_caps": {"A": {"kappa": 0.4}},
            "max_caps": 4,
        },
        "rmt": {"margin": 1.4, "epsilon_default": 0.2},
        "variance": {"min_effect_lognll": 0.05, "max_adjusted_modules": 2},
        "metrics": {"confidence": {"ppl_ratio_width_max": 0.1}},
    }
    monkeypatch.setattr(
        policy_mod, "resolve_tier_policies", lambda *_a, **_k: fake_tier
    )
    spectral = {
        "policy": {"sigma_quantile": 0.9, "scope": "heads", "correction_enabled": True},
        "family_caps": {"B": {"kappa": 0.6}},
        "multiple_testing": {"method": "bonferroni", "alpha": 0.05, "m": 3},
    }
    rmt = {"epsilon_by_family": {"fam": 0.2}, "correct": 1}
    variance = {"predictive_gate": {"sided": "one_sided"}}
    resolved = policy_mod._build_resolved_policies("balanced", spectral, rmt, variance)
    assert resolved["spectral"]["sigma_quantile"] == 0.9
    assert resolved["spectral"]["multiple_testing"]["method"] == "bonferroni"
    assert resolved["rmt"]["epsilon_by_family"] == {"fam": 0.2}
    assert resolved["variance"]["predictive_one_sided"] is True
    assert resolved["confidence"]["ppl_ratio_width_max"] == 0.1


def test_build_resolved_policies_invalid_numbers(monkeypatch):
    fake_tier = {"spectral": {"deadband": 0.25, "max_caps": 7}}
    monkeypatch.setattr(
        policy_mod, "resolve_tier_policies", lambda *_a, **_k: fake_tier
    )
    spectral = {
        "deadband": "bad",
        "max_caps": "oops",
        "policy": {"sigma_quantile": 0.8},
    }
    rmt = {"margin": "bad"}
    variance = {}
    resolved = policy_mod._build_resolved_policies("balanced", spectral, rmt, variance)
    assert resolved["spectral"]["deadband"] == 0.1
    assert resolved["spectral"]["max_caps"] == 7
    assert resolved["rmt"]["margin"] == 1.5


def test_build_resolved_policies_bad_policy_object(monkeypatch):
    fake_tier = {"spectral": {"sigma_quantile": 0.97, "deadband": 0.15}}
    monkeypatch.setattr(
        policy_mod, "resolve_tier_policies", lambda *_a, **_k: fake_tier
    )

    class FlakyPolicy:
        def get(self, key, default=None):
            if key == "sigma_quantile":
                raise RuntimeError("boom")
            return default

    spectral = {"policy": FlakyPolicy()}
    resolved = policy_mod._build_resolved_policies("balanced", spectral, {}, {})
    assert resolved["spectral"]["sigma_quantile"] == 0.97


def test_build_resolved_policies_confidence_invalid_numbers(monkeypatch):
    fake_tier = {
        "spectral": {},
        "metrics": {
            "confidence": {
                "ppl_ratio_width_max": "bad",
                "accuracy_delta_pp_width_max": "bad",
            }
        },
    }
    monkeypatch.setattr(
        policy_mod, "resolve_tier_policies", lambda *_a, **_k: fake_tier
    )
    resolved = policy_mod._build_resolved_policies("balanced", {}, {}, {})
    assert "confidence" in resolved and resolved["confidence"] == {}


def test_extract_effective_policies_populates_missing(monkeypatch):
    fake_tier = {"spectral": {"deadband": 0.2}, "rmt": {"margin": 1.5}}
    monkeypatch.setattr(
        policy_mod, "get_tier_policies", lambda *_a, **_k: {"balanced": fake_tier}
    )
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "spectral",
                "metrics": {"sigma_quantile": 0.95, "max_caps": 3, "deadband": 0.2},
            },
            {
                "name": "rmt",
                "metrics": {"deadband_used": 0.2, "margin_used": 1.6},
            },
        ],
    }
    policies = policy_mod._extract_effective_policies(report)
    assert "spectral" in policies and "rmt" in policies
    assert policies["spectral"]["deadband"] == 0.2


def test_extract_effective_policies_rmt_injects_epsilon_default(monkeypatch):
    fake_tier = {"spectral": {}, "rmt": {}}
    monkeypatch.setattr(
        policy_mod, "get_tier_policies", lambda *_a, **_k: {"balanced": fake_tier}
    )
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "policy": {"margin": 1.4},
                "metrics": {"epsilon_default": 0.4},
            }
        ],
    }
    policies = policy_mod._extract_effective_policies(report)
    assert policies["rmt"]["epsilon_default"] == 0.4


def test_extract_effective_policies_adds_default_status(monkeypatch):
    monkeypatch.setattr(
        policy_mod, "get_tier_policies", lambda *_a, **_k: {"balanced": {"misc": None}}
    )
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"spectral": {}, "rmt": {}},
    }
    policies = policy_mod._extract_effective_policies(report)
    assert policies["spectral"]["status"] == "default_config"
    assert policies["rmt"]["status"] == "default_config"


def test_normalize_override_entry_and_extract_overrides():
    assert policy_mod._normalize_override_entry(["a", None, "b"]) == ["a", "b"]
    assert policy_mod._normalize_override_entry(42) == []
    report = {
        "meta": {
            "policy_overrides": "tierA",
            "auto": {"overrides": ["tierB", "tierA"]},
        },
        "config": {"overrides": ["tierC"], "extra": 1},
    }
    overrides = policy_mod._extract_policy_overrides(report)
    assert overrides == ["tierA", "tierB", "tierC"]


def test_compute_policy_digest_stable():
    digest1 = policy_mod._compute_policy_digest({"a": 1, "b": 2})
    digest2 = policy_mod._compute_policy_digest({"b": 2, "a": 1})
    assert digest1 == digest2


def test_compute_policy_digest_matches_assurance_spec():
    policy = {"b": 2, "a": 1, "nested": {"c": 3}}
    canonical = json.dumps(policy, sort_keys=True, default=str)
    expected = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert policy_mod._compute_policy_digest(policy) == expected


@pytest.mark.parametrize(
    "metrics",
    [
        "not-a-policy",
        {"pm_ratio": "bad"},
        {"pm_tail": "bad"},
        {"accuracy": "bad"},
    ],
)
def test_threshold_payload_rejects_non_mapping_policy_sections(metrics):
    payload = policy_mod._compute_thresholds_payload("balanced", {"metrics": metrics})
    assert payload["tier"] == "balanced"
    assert payload["pm_ratio"]["ratio_limit_base"] > 0


def test_threshold_payload_falls_back_from_uncoercible_optional_numbers():
    payload = policy_mod._compute_thresholds_payload(
        "balanced",
        {
            "metrics": {
                "pm_ratio": {"ratio_limit_base": BadFloat()},
                "pm_tail": {
                    "quantile": BadFloat(),
                    "epsilon": BadFloat(),
                    "quantile_max": object(),
                    "mass_max": object(),
                },
            }
        },
    )
    assert payload["pm_tail"]["quantile"] == 0.95
    assert payload["pm_tail"]["epsilon"] == 0.0
    assert payload["pm_tail"]["quantile_max"] is None
    assert payload["pm_tail"]["mass_max"] is None


def test_build_resolved_policies_retains_observed_contracts(monkeypatch):
    monkeypatch.setattr(
        policy_mod,
        "resolve_tier_policies",
        lambda *_a, **_k: {"spectral": {}, "rmt": {}, "variance": {}},
    )
    resolved = policy_mod._build_resolved_policies(
        "none",
        {"measurement_contract": {"algorithm": "power"}},
        {"measurement_contract": {"algorithm": "rmt"}},
        {
            "policy": {
                "deadband": 0.2,
                "min_abs_adjust": 0.01,
                "max_scale_step": 0.3,
                "min_effect_lognll": 0.04,
                "predictive_one_sided": False,
                "topk_backstop": 2,
                "max_adjusted_modules": 4,
            }
        },
    )
    assert resolved["spectral"]["measurement_contract"]["algorithm"] == "power"
    assert resolved["rmt"]["measurement_contract"]["algorithm"] == "rmt"
    assert resolved["variance"]["deadband"] == 0.2
    assert resolved["variance"]["max_adjusted_modules"] == 4


def test_extract_effective_policies_sanitizes_all_guard_fallbacks(monkeypatch):
    monkeypatch.setattr(
        policy_mod,
        "get_tier_policies",
        lambda: {"balanced": {"spectral": {"deadband": 0.1}}},
    )
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "spectral",
                "policy": {"sigma_quantile": "invalid", "max_spectral_norm": 0},
            },
            {
                "name": "variance",
                "metrics": {
                    "scope": "mlp",
                    "min_gain_threshold": 0.5,
                    "target_modules": 3,
                    "ve_enabled": True,
                },
            },
            {
                "name": "invariants",
                "metrics": {"checks_performed": 4, "violations_found": 1},
            },
        ],
    }
    policies = policy_mod._extract_effective_policies(report)
    assert policies["spectral"]["sigma_quantile"] == "invalid"
    assert policies["spectral"]["max_spectral_norm"] is None
    assert policies["variance"]["target_modules"] == 3
    assert policies["invariants"]["violations_found"] == 1


def test_policy_override_extraction_accepts_all_metadata_locations():
    report = {
        "meta": {
            "policy_overrides": ("one",),
            "overrides": {"two", None},
            "auto": {"overrides": "three"},
        },
        "config": {"overrides": "four"},
    }
    assert policy_mod._extract_policy_overrides(report) == [
        "one",
        "two",
        "three",
        "four",
    ]
