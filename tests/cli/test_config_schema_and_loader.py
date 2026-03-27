import textwrap
from pathlib import Path

import pytest

from invarlock.core import config_runtime as cfg_mod
from invarlock.core.config_runtime import (
    AutoConfig,
    DatasetConfig,
    EvalBootstrapConfig,
    InvarLockConfig,
    OutputConfig,
    RMTGuardConfig,
    SpectralGuardConfig,
    VarianceGuardConfig,
    _deep_merge,
    apply_edit_override,
    apply_profile,
    load_config,
    resolve_edit_kind,
)
from invarlock.core.config_dependencies import (
    absolute_path_no_resolve,
    inspect_config_dependencies,
    iter_absolute_path_strings,
)


def test_resolve_edit_kind_and_apply_override_roundtrip():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_causal"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    name = resolve_edit_kind("quant_rtn")
    assert name == "quant_rtn"
    updated = apply_edit_override(cfg, "quant_rtn")
    assert updated.edit.name == "quant_rtn"
    assert "kind" not in updated.data["edit"]
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
              adapter: hf_causal
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
    with pytest.raises(ValueError, match=r"guards\.spectral\.mode is not supported"):
        load_config(cfg_path)


def test_load_config_guard_mode_overrides_reject_invalid(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_guard_mode.yaml"
    cfg_path.write_text("guards: {spectral: {mode: turbo}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"guards\.spectral\.mode is not supported"):
        load_config(cfg_path)


def test_load_config_rejects_legacy_edit_parameters(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_edit_parameters.yaml"
    cfg_path.write_text(
        "edit: {name: quant_rtn, parameters: {bitwidth: 8}}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"edit\.parameters is not supported"):
        load_config(cfg_path)


def test_load_config_rejects_legacy_edit_kind(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_edit_kind.yaml"
    cfg_path.write_text("edit: {name: quant_rtn, kind: quant}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"edit\.kind is not supported"):
        load_config(cfg_path)


def test_load_config_rejects_legacy_assurance_block(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy_assurance.yaml"
    cfg_path.write_text("assurance: {mode: strict}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"assurance\.\* is not supported"):
        load_config(cfg_path)


def test_load_config_raises_on_bad_defaults_type(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "defaults: 123\nmodel: {id: gpt2, adapter: hf_causal}\nedit: {name: quant_rtn, plan: {}}\n"
    )
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_apply_profile_ci_cpu_and_unknown_profile():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_causal"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    ci_cpu = apply_profile(cfg, "ci_cpu")
    # Expect device forced to CPU and stride set
    assert ci_cpu.model.device == "cpu" and ci_cpu.dataset.stride > 0
    with pytest.raises(ValueError):
        apply_profile(cfg, "unknown")


def test_apply_profile_ci_and_release():
    cfg = InvarLockConfig(
        model={"id": "gpt2", "adapter": "hf_causal"},
        edit={"name": "quant_rtn", "plan": {}},
    )
    ci = apply_profile(cfg, "ci")
    assert ci.dataset.preview_n == 240 and ci.dataset.final_n == 240
    assert ci.eval.bootstrap.replicates >= 1200
    assert ci.primary_metric.overhead_threshold == pytest.approx(0.01)
    rel = apply_profile(cfg, "release")
    assert rel.dataset.preview_n >= 240 and rel.eval.bootstrap.replicates >= 3200


def test_load_config_include_missing_file(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "defaults: !include does_not_exist.yaml\nmodel: {id: gpt2, adapter: hf_causal}\nedit: {name: quant_rtn, plan: {}}\n"
    )
    with pytest.raises(FileNotFoundError):
        load_config(cfg_path)


def test_load_config_include_cycle_detected(tmp_path: Path) -> None:
    main = tmp_path / "main.yaml"
    a_path = tmp_path / "a.yaml"
    b_path = tmp_path / "b.yaml"

    main.write_text(f"defaults: !include {a_path.name}\n", encoding="utf-8")
    a_path.write_text(f"defaults: !include {b_path.name}\n", encoding="utf-8")
    b_path.write_text(f"defaults: !include {a_path.name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Config !include cycle detected"):
        load_config(main)


def test_load_config_include_depth_guard(tmp_path: Path) -> None:
    main = tmp_path / "main.yaml"
    depth = 18
    chain_files = [tmp_path / f"inc_{idx}.yaml" for idx in range(depth)]

    main.write_text(f"defaults: !include {chain_files[0].name}\n", encoding="utf-8")
    for idx, chain_file in enumerate(chain_files):
        if idx < depth - 1:
            chain_file.write_text(
                f"defaults: !include {chain_files[idx + 1].name}\n",
                encoding="utf-8",
            )
        else:
            chain_file.write_text(
                "model: {id: gpt2, adapter: hf_causal}\n"
                "edit: {name: quant_rtn, plan: {}}\n",
                encoding="utf-8",
            )

    with pytest.raises(ValueError, match=r"Config !include depth exceeds"):
        load_config(main)


def test_obj_mapping_scalar_and_non_mapping_paths() -> None:
    obj = cfg_mod._Obj({"nested": {"value": 1}, "scalar": 7})

    assert obj["scalar"] == 7
    assert obj.scalar == 7
    assert obj.nested.value == 1
    assert cfg_mod._Obj("plain").get("missing", "fallback") == "fallback"


def test_dataset_and_path_iteration_helper_edges(tmp_path: Path) -> None:
    dataset = DatasetConfig(seq_len=8, stride=8)
    absolute_a = tmp_path / "a.jsonl"
    absolute_b = tmp_path / "b.jsonl"

    found = iter_absolute_path_strings([str(absolute_a), ("   ", {str(absolute_b)})])

    assert dataset.seq_len == 8
    assert found == {
        absolute_path_no_resolve(absolute_a),
        absolute_path_no_resolve(absolute_b),
    }
    assert absolute_path_no_resolve("relative/config.yaml").is_absolute()


def test_load_runtime_yaml_env_root_missing_file_falls_back_to_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(tmp_path))

    data = cfg_mod._load_runtime_yaml("profiles", "ci.yaml")

    assert isinstance(data, dict)
    assert data


def test_inspect_config_dependencies_tracks_nested_includes_and_absolute_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    data_file = external_root / "dataset.jsonl"
    data_file.write_text('{"text":"hello"}\n', encoding="utf-8")
    nested = external_root / "nested.yaml"
    nested.write_text(
        textwrap.dedent(
            f"""
            dataset:
              file: {data_file}
            model:
              id: gpt2
              adapter: hf_causal
            edit:
              name: noop
              plan: {{}}
            """
        ),
        encoding="utf-8",
    )
    main = repo_root / "config.yaml"
    main.write_text(
        f"defaults: !include ../external/{nested.name}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "1")

    scan = inspect_config_dependencies(main)

    assert scan.config_paths == tuple(
        sorted((main.resolve(), nested.resolve()), key=str)
    )
    assert scan.referenced_paths == (data_file.resolve(),)


def test_inspect_config_dependencies_rejects_outside_include_without_override(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "nested.yaml").write_text("model: {}\n", encoding="utf-8")
    main = repo_root / "config.yaml"
    main.write_text("defaults: !include ../external/nested.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1"):
        inspect_config_dependencies(main)


def test_inspect_config_dependencies_allows_outside_include_with_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    nested = external_root / "nested.yaml"
    nested.write_text(
        "model: {id: gpt2, adapter: hf_causal}\nedit: {name: noop, plan: {}}\n",
        encoding="utf-8",
    )
    main = repo_root / "config.yaml"
    main.write_text("defaults: !include ../external/nested.yaml\n", encoding="utf-8")
    monkeypatch.setenv("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "1")

    scan = inspect_config_dependencies(main)

    assert scan.config_paths == tuple(
        sorted((main.resolve(), nested.resolve()), key=str)
    )


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


def test_deep_merge_merges_and_overwrites():
    base = {"a": {"b": 1}, "x": 1}
    override = {"a": {"c": 2}, "x": {"y": 3}, "z": 4}
    out = _deep_merge(base, override)
    assert (
        out["a"]["b"] == 1
        and out["a"]["c"] == 2
        and isinstance(out["x"], dict)
        and out["z"] == 4
    )
