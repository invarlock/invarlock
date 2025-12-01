from __future__ import annotations

import pytest

from invarlock.cli.config import (
    DatasetConfig,
    EvalBootstrapConfig,
    InvarLockConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    apply_edit_override,
    apply_profile,
    load_config,
    resolve_edit_kind,
)
from invarlock.cli.device import (
    get_device_info,
    is_device_available,
    resolve_device,
    validate_device_for_config,
)


def test_dataset_eval_spectral_variance_configs():
    # DatasetConfig validation
    with pytest.raises(ValueError):
        DatasetConfig(seq_len=8, stride=16)
    # EvalBootstrapConfig invalid
    with pytest.raises(ValueError):
        EvalBootstrapConfig(replicates=0)
    with pytest.raises(ValueError):
        EvalBootstrapConfig(alpha=0.0)
    # Spectral alias + caps normalization
    s = SpectralGuardConfig(
        contraction=0.9, family_caps={"attn": 1.2, "mlp": {"kappa": 0.8}}
    )
    assert s.sigma_quantile == 0.9 and s.contraction is None
    assert s.family_caps == {"attn": {"kappa": 1.2}, "mlp": {"kappa": 0.8}}
    # Variance clamp validation and default floor
    with pytest.raises(ValueError):
        VarianceGuardConfig(clamp=[1, 1])
    v = VarianceGuardConfig()
    assert v.absolute_floor_ppl is not None


def test_config_load_and_profile(tmp_path, monkeypatch):
    # Create included YAML
    inc = tmp_path / "inc.yaml"
    inc.write_text("dataset: {seq_len: 8, stride: 8}", encoding="utf-8")
    # Main YAML with defaults + include
    main = tmp_path / "cfg.yaml"
    main.write_text(
        """
defaults:
  edit:
    name: quant_rtn
guards:
  variance:
    mode: ci
    min_effect_lognll: 0.1
    clamp: [0.0, 1.0]
dataset: !include inc.yaml
        """,
        encoding="utf-8",
    )
    cfg = load_config(main)
    assert isinstance(cfg, InvarLockConfig)
    assert cfg.edit.name == "quant_rtn"
    # apply_profile(ci) falls back to env defaults when runtime profiles missing
    monkeypatch.delenv("INVARLOCK_CONFIG_ROOT", raising=False)
    monkeypatch.setenv("INVARLOCK_CI_PREVIEW", "10")
    monkeypatch.setenv("INVARLOCK_CI_FINAL", "20")
    cfg2 = apply_profile(cfg, "ci")
    assert cfg2.dataset.preview_n == 10 and cfg2.dataset.final_n == 20
    # resolve_edit_kind and override
    assert resolve_edit_kind("quant") == "quant_rtn"
    with pytest.raises(ValueError):
        resolve_edit_kind("unknown")
    cfg3 = apply_edit_override(cfg2, "quant")
    assert cfg3.edit.name == "quant_rtn" and cfg3.edit.kind == "quant"


def test_device_helpers(monkeypatch):
    # is_device_available without torch present should return False for cuda/mps
    assert is_device_available("cpu") is True
    assert is_device_available("cuda") in {False, True}  # tolerant across envs
    # resolve_device respects explicit invalid device
    with pytest.raises(RuntimeError):
        resolve_device("invalid")
    # auto resolves to something valid; in no-torch env it's cpu
    auto = resolve_device("auto")
    assert auto in {"cpu", "mps", "cuda:0"}
    ok, msg = validate_device_for_config("cpu", {"required_device": "cpu"})
    assert ok and msg == ""
    ok2, msg2 = validate_device_for_config("cpu", {"required_device": "cuda"})
    assert not ok2 and "requires device" in msg2
    info = get_device_info()
    assert set(info.keys()) >= {"cpu", "cuda", "mps", "auto_selected"}
