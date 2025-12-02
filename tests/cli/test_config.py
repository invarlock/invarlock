from __future__ import annotations

import pytest

from invarlock.cli.config import (
    InvarLockConfig,
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


def test_guard_configs_family_caps_and_contraction_alias():
    # SpectralGuardConfig handles contraction alias → sigma_quantile
    sg = SpectralGuardConfig(contraction=0.2)
    assert sg.sigma_quantile == 0.2 and sg.contraction is None
    # SpectralGuardConfig normalizes family_caps
    sg2 = SpectralGuardConfig(family_caps={"fam": 3.0, "x": {"kappa": 1.5}})
    assert sg2.family_caps == {"fam": {"kappa": 3.0}, "x": {"kappa": 1.5}}


@pytest.mark.parametrize("clamp", [[], [0.1], [0.5, 0.1]])
def test_variance_guard_config_clamp_validation(clamp):
    with pytest.raises(ValueError):
        VarianceGuardConfig(clamp=clamp)  # type: ignore[arg-type]


def test_resolve_edit_kind_unknown_raises():
    with pytest.raises(ValueError):
        resolve_edit_kind("not-a-kind")


def test_apply_profile_ci_env_overrides(monkeypatch):
    # Defaults overwritten by env when profile==ci
    monkeypatch.setenv("INVARLOCK_CI_PREVIEW", "42")
    monkeypatch.setenv("INVARLOCK_CI_FINAL", "84")
    cfg = InvarLockConfig(dataset={"provider": "wikitext2"})
    out = apply_profile(cfg, "ci")
    d = out.data.get("dataset", {})
    assert d.get("preview_n") == 42 and d.get("final_n") == 84
