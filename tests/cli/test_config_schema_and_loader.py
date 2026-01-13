import textwrap
from pathlib import Path

import pytest

from invarlock.cli.config import (
    AutoConfig,
    DatasetConfig,
    EvalBootstrapConfig,
    InvarLockConfig,
    OutputConfig,
    RMTGuardConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    _deep_merge_dicts,
    apply_edit_override,
    apply_profile,
    load_config,
    resolve_edit_kind,
)


def test_resolve_edit_kind_and_apply_override_roundtrip():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_gpt2"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    name = resolve_edit_kind("quant")
    assert name == "quant_rtn"
    updated = apply_edit_override(cfg, "quant")
    assert updated.edit.name == "quant_rtn" and updated.edit.kind == "quant"
    with pytest.raises(ValueError):
        resolve_edit_kind("unknown")


def test_dataset_and_variance_validators_raise():
    with pytest.raises(ValueError):
        DatasetConfig(seq_len=128, stride=256)  # stride > seq_len
    with pytest.raises(ValueError):
        VarianceGuardConfig(clamp=[1.0, 0.1])  # invalid clamp order


def test_spectral_guard_alias_hydration_and_caps_normalization():
    sg = SpectralGuardConfig(
        sigma_quantile=0.9, family_caps={"ffn": 2.5, "attn": {"kappa": 3}}
    )
    assert sg.sigma_quantile == 0.9 and sg.family_caps["ffn"]["kappa"] == 2.5


def test_auto_config_bounds_and_output_dir_coercion(tmp_path: Path):
    with pytest.raises(ValueError):
        AutoConfig(probes=11)
    with pytest.raises(ValueError):
        AutoConfig(target_pm_ratio=0.9)
    out = OutputConfig(dir=str(tmp_path / "runs"))
    assert isinstance(out.dir, Path)


def test_load_config_with_include_and_defaults_merge(tmp_path: Path):
    base = tmp_path / "base_defaults.yaml"
    base.write_text(
        textwrap.dedent(
            """
            model:
              id: gpt2
              adapter: hf_gpt2
            edit:
              name: quant_rtn
              plan: {}
            """
        )
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            defaults: !include {base.name}
            dataset:
              preview_n: 10
              final_n: 10
            """
        )
    )
    cfg = load_config(cfg_path)
    assert (
        isinstance(cfg, InvarLockConfig)
        and cfg.dataset.preview_n == 10
        and cfg.model.id == "gpt2"
    )


def test_load_config_variance_guard_default_mode_and_floor(tmp_path: Path) -> None:
    cfg_path = tmp_path / "guard_cfg.yaml"
    cfg_path.write_text(
        "guards: {variance: {clamp: [0.0, 1.0], absolute_floor_ppl: 0.1}}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    var_cfg = cfg.data["guards"]["variance"]
    assert isinstance(var_cfg, VarianceGuardConfig)
    assert var_cfg.mode == "ci"
    assert var_cfg.absolute_floor_ppl == 0.1


def test_load_config_guard_mode_overrides_normalize_and_validate(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "guard_modes.yaml"
    cfg_path.write_text(
        "guards:\n  spectral: {mode: FAST}\n  rmt: {mode: strict}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"guards\.spectral\.mode is deprecated"):
        load_config(cfg_path)


def test_load_config_guard_mode_overrides_reject_invalid(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_guard_mode.yaml"
    cfg_path.write_text("guards: {spectral: {mode: turbo}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"guards\.spectral\.mode is deprecated"):
        load_config(cfg_path)


def test_load_config_raises_on_bad_defaults_type(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "defaults: 123\nmodel: {id: gpt2, adapter: hf_gpt2}\nedit: {name: quant_rtn, plan: {}}\n"
    )
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_apply_profile_ci_cpu_and_unknown_profile():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_gpt2"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    ci_cpu = apply_profile(cfg, "ci_cpu")
    # Expect device forced to CPU and stride set
    assert ci_cpu.model.device == "cpu" and ci_cpu.dataset.stride > 0
    with pytest.raises(ValueError):
        apply_profile(cfg, "unknown")


def test_apply_profile_ci_and_release():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_gpt2"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    ci = apply_profile(cfg, "ci")
    assert ci.dataset.preview_n >= 200 and ci.eval.bootstrap.replicates >= 1200
    rel = apply_profile(cfg, "release")
    assert rel.dataset.preview_n >= 240 and rel.eval.bootstrap.replicates >= 3200


def test_load_config_include_missing_file(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "defaults: !include does_not_exist.yaml\nmodel: {id: gpt2, adapter: hf_gpt2}\nedit: {name: quant_rtn, plan: {}}\n"
    )
    with pytest.raises(FileNotFoundError):
        load_config(cfg_path)


def test_load_config_none_and_nondict(tmp_path: Path):
    # None
    cfg_null = tmp_path / "null.yaml"
    cfg_null.write_text("null\n")
    with pytest.raises(ValueError):
        load_config(cfg_null)
    # Non-dict
    cfg_list = tmp_path / "list.yaml"
    cfg_list.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(cfg_list)


def test_rmt_guard_config_epsilon_paths_and_bootstrap_bounds():
    # Dict epsilon ok
    rg = RMTGuardConfig(epsilon={"ffn": 0.1})
    assert isinstance(rg.epsilon, dict)
    # Scalar epsilon ok; replicate/alpha validators
    eb = EvalBootstrapConfig(replicates=5, alpha=0.1, ci_band=0.0)
    assert eb.replicates == 5
    with pytest.raises(ValueError):
        EvalBootstrapConfig(replicates=0, alpha=0.1)
    with pytest.raises(ValueError):
        EvalBootstrapConfig(replicates=1, alpha=1.0)


def test_deep_merge_dicts_merges_and_overwrites():
    base = {"a": {"b": 1}, "x": 1}
    override = {"a": {"c": 2}, "x": {"y": 3}, "z": 4}
    out = _deep_merge_dicts(base, override)
    assert (
        out["a"]["b"] == 1
        and out["a"]["c"] == 2
        and isinstance(out["x"], dict)
        and out["z"] == 4
    )
