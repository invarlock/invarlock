from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.cli.config as config_mod
from invarlock.cli.config import (
    AutoConfig,
    InvarLockConfig,
    OutputConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    _Obj,
    apply_profile,
    resolve_edit_kind,
)


def test__obj_attribute_access_and_get():
    o = _Obj({"a": {"b": 1}, "c": 2})
    assert o.c == 2
    assert isinstance(o.a, _Obj)
    with pytest.raises(AttributeError):
        _ = o.missing  # noqa: F841
    assert o.get("missing", 7) == 7


def test__obj_non_mapping_paths() -> None:
    o = _Obj(123)
    with pytest.raises(TypeError):
        _ = o["x"]  # noqa: F841
    assert o.get("anything", 9) == 9


def test_invarlock_config_merges_data_and_sections() -> None:
    base = {"model": {"id": "gpt2"}, "extra": 1}
    cfg = InvarLockConfig(data=base, edit={"name": "noop"})
    assert cfg.model.id == "gpt2"
    assert cfg.edit.name == "noop"
    with pytest.raises(AttributeError):
        _ = cfg.missing  # noqa: F841


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
    cfg = InvarLockConfig(dataset={"provider": "wikitext2"})
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "ci")


def test_apply_profile_ci_missing_runtime_profile_ignores_env(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_runtime_yaml", lambda *_a, **_k: None)
    monkeypatch.setenv("INVARLOCK_CI_PREVIEW", "not-an-int")
    monkeypatch.setenv("INVARLOCK_CI_FINAL", "also-bad")
    cfg = InvarLockConfig(dataset={"provider": "wikitext2"})
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "ci")
