from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.core.config_runtime as config_mod
from invarlock.core.config_runtime import (
    AutoConfig,
    EditConfig,
    EvalBootstrapConfig,
    EvalConfig,
    GuardsConfig,
    InvarLockConfig,
    ModelConfig,
    OutputConfig,
    VarianceGuardConfig,
)


def test_section_normalization_drops_none_and_rejects_non_dataclasses() -> None:
    payload = config_mod._normalize_section_value(
        {
            "a": 1,
            "b": None,
            "nested": [1, None, {"keep": 2, "drop": None}],
            "tupled": (1, None, 3),
        }
    )

    assert payload == {"a": 1, "nested": [1, {"keep": 2}], "tupled": (1, 3)}

    with pytest.raises(TypeError, match="Expected dataclass instance"):
        config_mod._section_dataclass_payload(object())


def test_section_mixin_exposes_known_and_extra_fields() -> None:
    edit = EditConfig(name="quant_rtn", _extra={"note": "x"})

    assert edit["name"] == "quant_rtn"
    assert edit["note"] == "x"
    assert "name" in edit
    assert "note" in edit
    assert 3 not in edit
    assert edit.get("missing", "fallback") == "fallback"

    edit["name"] = "noop"
    edit["other"] = 7
    assert edit.name == "noop"
    assert edit._extra["other"] == 7


def test_output_eval_and_guard_sections_normalize_runtime_values(tmp_path: Path) -> None:
    output = OutputConfig(dir=str(tmp_path / "runs"), model_dir=str(tmp_path / "models"))
    assert output.dir == tmp_path / "runs"
    assert output.model_dir == tmp_path / "models"

    eval_cfg = EvalConfig(
        bootstrap={
            "replicates": 8,
            "alpha": 0.2,
            "ci_band": 0.15,
            "method": "percentile",
            "seed": 1,
        },
        loss={"type": "mlm", "mask_prob": 0.2},
    )
    assert isinstance(eval_cfg.bootstrap, EvalBootstrapConfig)
    assert eval_cfg.bootstrap.replicates == 8
    assert eval_cfg.bootstrap._extra == {"seed": 1}
    assert eval_cfg.loss is not None and eval_cfg.loss.type == "mlm"

    guards = GuardsConfig(
        order=["variance"],
        variance={"clamp": [0.1, 0.9], "scope": "ffn"},
    )
    assert isinstance(guards.variance, dict)


def test_invarlock_config_accessors_mutation_and_model_dump() -> None:
    cfg = InvarLockConfig.from_sections(
        model={"id": "gpt2"},
        edit={"name": "quant_rtn", "plan": {"bitwidth": 8}},
        dataset={"provider": "wikitext2"},
        output={"dir": "."},
        auto={"enabled": False, "probes": 2, "target_pm_ratio": 1.1},
        guards={"order": ["variance"], "variance": {"clamp": [0.1, 0.9]}},
        eval={"loss": {"type": "causal"}},
        context={"run_id": "abc"},
    )

    assert isinstance(cfg.model, ModelConfig)
    assert cfg.edit.name == "quant_rtn"
    assert cfg.dataset.provider == "wikitext2"
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.auto, AutoConfig)
    assert isinstance(cfg.guards.variance, VarianceGuardConfig)
    assert isinstance(cfg.eval, EvalConfig)
    assert cfg.context == {"run_id": "abc"}
    assert sorted(iter(cfg)) == sorted(cfg.data)
    assert len(cfg) == len(cfg.data)

    dumped = cfg.model_dump()
    dumped["context"]["run_id"] = "mutated"
    assert cfg.context["run_id"] == "abc"

    del cfg["context"]
    with pytest.raises(KeyError, match="required"):
        _ = cfg.context


def test_invarlock_config_missing_sections_fail_closed() -> None:
    cfg = InvarLockConfig({})
    for attr in ("model", "edit", "dataset", "output", "auto", "guards", "eval"):
        with pytest.raises(KeyError, match="required"):
            getattr(cfg, attr)

    cfg["context"] = "not-a-mapping"
    with pytest.raises(TypeError, match="must be a mapping"):
        _ = cfg.context


def test_load_runtime_yaml_tolerates_inner_package_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingLeaf:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:  # noqa: ARG002
            raise FileNotFoundError("missing")

    class _PkgRoot:
        def joinpath(self, _part: str) -> _MissingLeaf:
            return _MissingLeaf()

    class _Pkg:
        def files(self, _name: str) -> _PkgRoot:
            return _PkgRoot()

    monkeypatch.delenv("INVARLOCK_CONFIG_ROOT", raising=False)
    monkeypatch.setattr(config_mod, "_ires", _Pkg())

    assert config_mod._load_runtime_yaml("profiles", "ci.yaml") is None
