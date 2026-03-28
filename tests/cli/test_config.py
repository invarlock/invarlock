from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.core.config_runtime as config_mod
from invarlock.core.config_runtime import (
    AutoConfig,
    InvarLockConfig,
    OutputConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    apply_profile,
    resolve_edit_kind,
)


def test_invarlock_config_is_explicit_mutable_mapping() -> None:
    base = {"model": {"id": "gpt2"}, "extra": 1}
    cfg = InvarLockConfig.from_sections(**base, edit={"name": "noop"})
    assert cfg["model"]["id"] == "gpt2"
    assert cfg.require_section("edit")["name"] == "noop"
    assert cfg.get("missing") is None
    cfg.setdefault("context", {})
    cfg["context"]["mode"] = "local"
    assert cfg.section("context") == {"mode": "local"}


def test_invarlock_config_section_accessors_fail_closed() -> None:
    cfg = InvarLockConfig.from_sections(model={"id": "gpt2"}, extra=7)
    assert cfg.section("missing") is None
    assert cfg.section("model") == {"id": "gpt2"}
    with pytest.raises(KeyError, match="required"):
        cfg.require_section("dataset")
    with pytest.raises(TypeError, match="must be a mapping"):
        cfg.section("extra")


def test_eval_section_preserves_loss_and_runtime_fields() -> None:
    cfg = InvarLockConfig.from_sections(
        eval={
            "loss": {
                "type": "mlm",
                "mask_prob": 0.2,
                "random_token_prob": 0.1,
                "original_token_prob": 0.05,
            },
            "capacity_fast": True,
            "max_pm_ratio": 1.25,
            "spike_threshold": 2.5,
        }
    )

    assert cfg.eval.loss is not None
    assert cfg.eval.loss.type == "mlm"
    assert cfg.eval.loss.mask_prob == pytest.approx(0.2)
    assert cfg.eval.capacity_fast is True
    assert cfg.eval.max_pm_ratio == pytest.approx(1.25)
    assert cfg.eval.spike_threshold == pytest.approx(2.5)
    assert cfg.section("eval") == {
        "bootstrap": {"replicates": 1000, "alpha": 0.05, "ci_band": 0.1},
        "loss": {
            "type": "mlm",
            "mask_prob": 0.2,
            "random_token_prob": 0.1,
            "original_token_prob": 0.05,
        },
        "spike_threshold": 2.5,
        "max_pm_ratio": 1.25,
        "capacity_fast": True,
    }


def test_guard_configs_family_caps_and_sigma_quantile():
    sg = SpectralGuardConfig(sigma_quantile=0.2)
    assert sg.sigma_quantile == 0.2
    # SpectralGuardConfig normalizes family_caps
    sg2 = SpectralGuardConfig(family_caps={"fam": 3.0, "x": {"kappa": 1.5}})
    assert sg2.family_caps == {"fam": {"kappa": 3.0}, "x": {"kappa": 1.5}}


@pytest.mark.parametrize("clamp", [[], [0.1], [0.5, 0.1]])
def test_variance_guard_config_clamp_validation(clamp):
    with pytest.raises(ValueError):
        VarianceGuardConfig(clamp=clamp)  # type: ignore[arg-type]


def test_variance_guard_config_happy_path_sets_floor() -> None:
    vg = VarianceGuardConfig(clamp=[0.1, 0.9])
    assert vg.clamp == [0.1, 0.9]
    assert vg.absolute_floor_ppl == 0.05


def test_output_config_accepts_path_and_str(tmp_path: Path) -> None:
    # Path is preserved
    p = tmp_path / "runs"
    oc_path = OutputConfig(dir=p)
    assert oc_path.dir == p
    # str is coerced to Path
    oc_str = OutputConfig(dir=str(p))
    assert isinstance(oc_str.dir, Path)
    assert oc_str.dir == p


def test_auto_config_valid_values_pass() -> None:
    cfg = AutoConfig(probes=3, target_pm_ratio=1.1)
    assert cfg.probes == 3 and cfg.target_pm_ratio == 1.1


def test_resolve_edit_kind_unknown_raises():
    with pytest.raises(ValueError):
        resolve_edit_kind("not-a-kind")


def test_apply_profile_ci_raises_when_runtime_profile_missing(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_runtime_yaml", lambda *_a, **_k: None)
    cfg = InvarLockConfig.from_sections(dataset={"provider": "wikitext2"})
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "ci")


def test_apply_profile_ci_missing_runtime_profile_ignores_env(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_runtime_yaml", lambda *_a, **_k: None)
    monkeypatch.setenv("INVARLOCK_CI_PREVIEW", "not-an-int")
    monkeypatch.setenv("INVARLOCK_CI_FINAL", "also-bad")
    cfg = InvarLockConfig.from_sections(dataset={"provider": "wikitext2"})
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "ci")
