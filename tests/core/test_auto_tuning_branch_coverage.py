import pytest

from invarlock.core import auto_tuning as at


def test_load_runtime_yaml_falls_back_to_package_resources(tmp_path) -> None:
    # Config root exists but does not contain runtime/tiers.yaml -> fall back to packaged tiers.yaml.
    data = at._load_runtime_yaml(str(tmp_path), "tiers.yaml")
    assert isinstance(data, dict) and "balanced" in data


def test_load_runtime_yaml_config_root_non_mapping_raises(tmp_path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "tiers.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        at._load_runtime_yaml(str(tmp_path), "tiers.yaml")


def test_load_runtime_yaml_resource_non_mapping_returns_none(monkeypatch) -> None:
    # Exercise the packaged-resource branch raising ValueError and being caught.
    monkeypatch.setattr(at.yaml, "safe_load", lambda *_a, **_k: ["not-a-mapping"])
    assert at._load_runtime_yaml(None, "tiers.yaml") is None


def test_load_runtime_yaml_resource_files_error_returns_none(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(at._ires, "files", _boom)
    assert at._load_runtime_yaml(None, "tiers.yaml") is None


def test_normalize_family_caps_and_multiple_testing_variants() -> None:
    assert at._normalize_family_caps(None) == {}
    caps = {
        "ffn": {"kappa": 2.0},
        "attn": 3.0,
        "embed": {"kappa": "not-a-number"},
        "other": {"kappa": None},
    }
    assert at._normalize_family_caps(caps) == {
        "ffn": {"kappa": 2.0},
        "attn": {"kappa": 3.0},
    }

    assert at._normalize_multiple_testing(None) == {}
    assert at._normalize_multiple_testing(
        {"method": "BH", "alpha": "0.1", "m": "4"}
    ) == {"method": "bh", "alpha": 0.1, "m": 4}
    # Bad numeric fields are ignored (exception paths).
    assert (
        at._normalize_multiple_testing({"method": None, "alpha": "nope", "m": "nope"})
        == {}
    )


def test_tier_entry_to_policy_maps_sections_and_skips_bad_sections() -> None:
    out = at._tier_entry_to_policy(
        {
            "metrics": {"pm_ratio": {"ratio_limit_base": 1.2}},
            "spectral_guard": {
                "family_caps": {"ffn": 3.0, "attn": {"kappa": 2.5}},
                "multiple_testing": {"method": "BH", "alpha": "0.1", "m": "4"},
            },
            "rmt_guard": {"epsilon_by_family": {"ffn": 0.1, "bad": "x"}},
            "variance_guard": {"deadband": 0.1},
        }
    )
    assert out["metrics"]["pm_ratio"]["ratio_limit_base"] == pytest.approx(1.2)
    assert out["spectral"]["family_caps"] == {
        "ffn": {"kappa": 3.0},
        "attn": {"kappa": 2.5},
    }
    assert out["spectral"]["multiple_testing"]["method"] == "bh"
    assert out["rmt"]["epsilon_by_family"] == {"ffn": 0.1}
    assert out["variance"]["deadband"] == pytest.approx(0.1)

    assert (
        at._tier_entry_to_policy(
            {"metrics": "bad", "spectral": ["bad"], "rmt": "bad", "variance": None}
        )
        == {}
    )


def test_load_tier_policies_cached_skips_non_dict_and_adds_new_tier(tmp_path) -> None:
    at.clear_tier_policies_cache()
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "tiers.yaml").write_text(
        """
balanced:
  metrics:
    pm_ratio:
      ratio_limit_base: 1.11
custom:
  spectral_guard:
    deadband: 0.123
bad_entry: 7
""".lstrip(),
        encoding="utf-8",
    )
    pol = at.get_tier_policies(config_root=str(tmp_path))
    assert pol["balanced"]["metrics"]["pm_ratio"]["ratio_limit_base"] == pytest.approx(
        1.11
    )
    assert "custom" in pol and pol["custom"]["spectral"]["deadband"] == pytest.approx(
        0.123
    )
    assert "bad_entry" not in pol


def test_profile_overrides_and_edit_adjustments_branching(
    tmp_path, monkeypatch
) -> None:
    at.clear_tier_policies_cache()
    runtime = tmp_path / "runtime"
    profiles = runtime / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (runtime / "tiers.yaml").write_text("balanced: {}\n", encoding="utf-8")
    (profiles / "ci.yaml").write_text(
        """
guards:
  spectral:
    deadband: 0.123
  new_guard:
    alpha: 0.9
  rmt: 7
""".lstrip(),
        encoding="utf-8",
    )

    policies = at.resolve_tier_policies(
        "balanced", profile="ci", config_root=str(tmp_path)
    )
    assert policies["spectral"]["deadband"] == pytest.approx(0.123)
    assert policies["new_guard"]["alpha"] == pytest.approx(0.9)

    # Cover edit-adjustment branch where guard is missing (skip).
    monkeypatch.setitem(at.EDIT_ADJUSTMENTS, "dummy_edit", {"missing_guard": {"x": 1}})
    policies2 = at.resolve_tier_policies(
        "balanced", edit_name="dummy_edit", config_root=str(tmp_path)
    )
    assert "missing_guard" not in policies2
