"""Unit tests for tiered guard policies and predictive gate semantics."""

from pathlib import Path

import pytest
import yaml

from invarlock.core.auto_tuning import TIER_POLICIES
from invarlock.guards.variance import _predictive_gate_outcome


def test_predictive_gate_one_sided_behaviour():
    """Balanced tier uses one-sided improvement semantics."""

    min_effect = 9e-4

    # CI strictly below zero, mean negative, sufficient magnitude → pass.
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.0015,
        delta_ci=(-0.0026, -0.0010),
        min_effect=min_effect,
        one_sided=True,
    )
    assert passed is True
    assert reason == "ci_gain_met"

    # Mean not negative despite CI lower bound < 0 → fail.
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.0001,
        delta_ci=(-0.0020, -0.0003),
        min_effect=min_effect,
        one_sided=True,
    )
    assert passed is False
    assert reason == "mean_not_negative"

    # Mean negative but magnitude below min_effect → fail.
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.0004,
        delta_ci=(-0.0012, -0.0001),
        min_effect=min_effect,
        one_sided=True,
    )
    assert passed is False
    assert reason == "gain_below_threshold"


def test_predictive_gate_two_sided_behaviour():
    """Conservative tier requires two-sided improvement with larger min effect."""

    min_effect = 0.0018

    # CI strictly below zero with sufficient gain → pass.
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.0025,
        delta_ci=(-0.0040, -0.0019),
        min_effect=min_effect,
        one_sided=False,
    )
    assert passed is True
    assert reason == "ci_gain_met"

    # CI crosses zero → fail.
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.0024,
        delta_ci=(-0.0035, 0.0002),
        min_effect=min_effect,
        one_sided=False,
    )
    assert passed is False
    assert reason == "ci_contains_zero"

    # CI negative but gain lower bound below min effect → fail.
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.0015,
        delta_ci=(-0.0025, -0.0003),
        min_effect=min_effect,
        one_sided=False,
    )
    assert passed is False
    assert reason == "gain_below_threshold"


def test_tier_policies_align_with_documented_knobs():
    """Ensure tier documentation and auto-tuning defaults agree on key knobs."""

    tier_doc_path = Path("src/invarlock/_data/runtime/tiers.yaml")
    with tier_doc_path.open(encoding="utf-8") as handle:
        tier_doc = yaml.safe_load(handle)

    balanced_doc = tier_doc["balanced"]["variance_guard"]
    conservative_doc = tier_doc["conservative"]["variance_guard"]
    balanced_spectral_doc = tier_doc["balanced"]["spectral_guard"]
    conservative_spectral_doc = tier_doc["conservative"]["spectral_guard"]

    assert balanced_doc["predictive_one_sided"] is True
    assert conservative_doc["predictive_one_sided"] is False

    assert balanced_doc["max_adjusted_modules"] == 1
    assert conservative_doc["max_adjusted_modules"] == 0

    assert balanced_spectral_doc["scope"] == "all"
    assert balanced_spectral_doc["max_spectral_norm"] is None
    assert balanced_spectral_doc["multiple_testing"]["method"] == "bh"
    assert conservative_spectral_doc["scope"] == "ffn"

    balanced_policy = TIER_POLICIES["balanced"]["variance"]
    conservative_policy = TIER_POLICIES["conservative"]["variance"]
    balanced_spectral_policy = TIER_POLICIES["balanced"]["spectral"]
    conservative_spectral_policy = TIER_POLICIES["conservative"]["spectral"]

    assert balanced_policy["predictive_one_sided"] is True
    assert conservative_policy["predictive_one_sided"] is False

    assert balanced_policy["max_adjusted_modules"] == 1
    assert conservative_policy["max_adjusted_modules"] == 0

    assert balanced_spectral_policy["scope"] == "all"
    assert balanced_spectral_policy.get("max_spectral_norm") is None
    assert balanced_spectral_policy["multiple_testing"]["method"] == "bh"
    assert conservative_spectral_policy["scope"] == "ffn"

    # Keep min_effect knobs in sync with pilot-derived documentation.
    assert (
        pytest.approx(balanced_policy["min_effect_lognll"])
        == balanced_doc["min_effect_lognll"]
    )
    assert (
        pytest.approx(conservative_policy["min_effect_lognll"])
        == conservative_doc["min_effect_lognll"]
    )
