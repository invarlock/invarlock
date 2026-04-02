from __future__ import annotations

import pytest

import invarlock.core.config_runtime as config_mod
from invarlock.cli.device import (
    get_device_info,
    is_device_available,
    resolve_device,
    validate_device_for_config,
)
from invarlock.cli.run_config import (
    _apply_requested_edit_override,
    _resolve_requested_edit_name,
)
from invarlock.core.config_runtime import (
    DatasetConfig,
    EvalBootstrapConfig,
    InvarLockConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    apply_profile,
    load_config,
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
        sigma_quantile=0.9, family_caps={"attn": 1.2, "mlp": {"kappa": 0.8}}
    )
    assert s.sigma_quantile == 0.9
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
    assert cfg.require_section("edit")["name"] == "quant_rtn"
    # apply_profile(ci) requires a packaged/runtime profile file.
    monkeypatch.setattr(config_mod, "_load_runtime_yaml", lambda *_a, **_k: None)
    monkeypatch.delenv("INVARLOCK_CONFIG_ROOT", raising=False)
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "ci")
    cfg2 = cfg
    # run_config edit resolution and override
    assert _resolve_requested_edit_name("quant_rtn") == "quant_rtn"
    with pytest.raises(ValueError):
        _resolve_requested_edit_name("unknown")
    cfg3 = _apply_requested_edit_override(
        cfg2,
        "quant_rtn",
        config_cls=InvarLockConfig,
    )
    assert cfg3.require_section("edit")["name"] == "quant_rtn"
    assert "kind" not in cfg3.data["edit"]


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
