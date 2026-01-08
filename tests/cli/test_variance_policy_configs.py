"""Tier policy knobs validation (balanced vs conservative)."""

from pathlib import Path

import pytest
import yaml


def _load_tier_section(tier: str, section: str) -> dict:
    tier_path = Path("src/invarlock/_data/runtime/tiers.yaml")
    with tier_path.open() as handle:
        tiers = yaml.safe_load(handle)
    assert tier in tiers, f"Tier '{tier}' not documented in runtime tiers.yaml"
    doc = tiers[tier].get(section, {})
    assert doc, f"Tier '{tier}' missing {section} section"
    return doc


def test_balanced_tier_doc_matches_config():
    doc = _load_tier_section("balanced", "variance_guard")
    assert doc["deadband"] == pytest.approx(0.02)
    assert doc["min_abs_adjust"] == pytest.approx(0.012)
    assert doc["max_scale_step"] == pytest.approx(0.03)
    assert doc["min_effect_lognll"] == pytest.approx(0.0)
    assert doc["predictive_one_sided"] is True
    assert doc["topk_backstop"] == 1
    assert doc["max_adjusted_modules"] == 1


def test_conservative_tier_doc_matches_config():
    doc = _load_tier_section("conservative", "variance_guard")
    assert doc["deadband"] == pytest.approx(0.03)
    assert doc["min_abs_adjust"] == pytest.approx(0.02)
    assert doc["max_scale_step"] == pytest.approx(0.015)
    assert doc["min_effect_lognll"] == pytest.approx(0.016)
    assert doc["predictive_one_sided"] is False
    assert doc["topk_backstop"] == 0
    assert doc["max_adjusted_modules"] == 0


def test_balanced_spectral_doc_knobs():
    doc = _load_tier_section("balanced", "spectral_guard")
    assert doc["sigma_quantile"] == pytest.approx(0.95)
    assert doc["deadband"] == pytest.approx(0.10)
    # Calibrated values from spectral sweep
    assert doc["family_caps"]["ffn"] == pytest.approx(3.849)
    assert doc["family_caps"]["attn"] == pytest.approx(3.423)
    assert doc["family_caps"]["embed"] == pytest.approx(3.1)
    assert doc["family_caps"]["other"] == pytest.approx(3.1)


def test_conservative_spectral_doc_knobs():
    doc = _load_tier_section("conservative", "spectral_guard")
    assert doc["sigma_quantile"] == pytest.approx(0.90)
    assert doc["deadband"] == pytest.approx(0.05)
    assert doc["family_caps"]["ffn"] == pytest.approx(3.849)
    assert doc["family_caps"]["attn"] == pytest.approx(2.6)
    assert doc["family_caps"]["embed"] == pytest.approx(2.8)
    assert doc["family_caps"]["other"] == pytest.approx(2.8)


def test_balanced_rmt_doc_knobs():
    doc = _load_tier_section("balanced", "rmt_guard")
    assert doc["deadband"] == pytest.approx(0.10)
    assert doc["margin"] == pytest.approx(1.5)
    assert doc["epsilon_default"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["ffn"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["attn"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["embed"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["other"] == pytest.approx(0.01)


def test_conservative_rmt_doc_knobs():
    doc = _load_tier_section("conservative", "rmt_guard")
    assert doc["deadband"] == pytest.approx(0.05)
    assert doc["margin"] == pytest.approx(1.3)
    assert doc["epsilon_default"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["ffn"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["attn"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["embed"] == pytest.approx(0.01)
    assert doc["epsilon_by_family"]["other"] == pytest.approx(0.01)


def test_variance_knobs_documented_in_tiers():
    # The variance guard knobs are now documented in tiers.yaml
    # Configs no longer embed per-edit variance policies.
    bal = _load_tier_section("balanced", "variance_guard")
    cons = _load_tier_section("conservative", "variance_guard")
    # Simple sanity checks on expected keys
    for doc in (bal, cons):
        for key in (
            "deadband",
            "min_abs_adjust",
            "max_scale_step",
            "min_effect_lognll",
            "predictive_one_sided",
            "topk_backstop",
            "max_adjusted_modules",
        ):
            assert key in doc
