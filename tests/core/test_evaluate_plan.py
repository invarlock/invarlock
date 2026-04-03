from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.core.evaluate_plan import (
    DEFAULT_EVALUATE_GUARDS_ORDER,
    build_baseline_run_config,
    build_evaluate_command_plan,
    build_subject_edit_run_config,
    default_preset_data_for_adapter,
    determine_subject_label,
    normalize_model_id,
    resolve_evaluate_assurance_policy,
    resolve_evaluate_tmp_dir,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
)


def test_default_preset_data_for_adapter_uses_expected_sequence_lengths() -> None:
    assert default_preset_data_for_adapter("hf_causal")["dataset"]["seq_len"] == 512
    assert default_preset_data_for_adapter("hf_mlm")["dataset"]["seq_len"] == 128


def test_normalize_model_id_strips_hf_prefix_for_hf_adapters() -> None:
    assert normalize_model_id("hf:org/model", "hf_causal") == "org/model"
    assert normalize_model_id("hf:org/model", "custom") == "hf:org/model"


def test_normalize_model_id_reraises_unexpected_adapter_text_errors() -> None:
    class _BadAdapterName:
        def __str__(self) -> str:
            raise AssertionError("explode")

    with pytest.raises(AssertionError, match="explode"):
        normalize_model_id("hf:org/model", _BadAdapterName())  # type: ignore[arg-type]


def test_sanitize_preset_data_for_evaluate_removes_pinned_device() -> None:
    payload = {"model": {"device": "cuda", "id": "demo"}}

    sanitized = sanitize_preset_data_for_evaluate(payload)

    assert sanitized == {"model": {"id": "demo"}}
    assert payload == {"model": {"device": "cuda", "id": "demo"}}


def test_resolve_guards_order_prefers_preset_then_default() -> None:
    assert resolve_guards_order({}) == DEFAULT_EVALUATE_GUARDS_ORDER
    assert resolve_guards_order({"guards": {"order": ["g1", "g2"]}}) == ["g1", "g2"]


def test_determine_subject_label_matches_compare_and_edit_modes() -> None:
    assert (
        determine_subject_label(
            edit_label="quant_rtn",
            edit_config=None,
            source_model_id="a",
            subject_model_id="b",
        )
        == "quant_rtn"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config=None,
            source_model_id="a",
            subject_model_id="b",
        )
        == "custom"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config=None,
            source_model_id="a",
            subject_model_id="a",
        )
        == "noop"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config="edit.yaml",
            source_model_id="a",
            subject_model_id="b",
        )
        is None
    )


def test_build_baseline_run_config_injects_evaluate_context() -> None:
    cfg = build_baseline_run_config(
        {"dataset": {"provider": "wikitext2"}},
        model_id="org/model",
        adapter_name="hf_causal",
        output_dir="runs/source",
        profile="dev",
        tier="balanced",
        guards_order=["g1", "g2"],
    )

    assert cfg["model"] == {"id": "org/model", "adapter": "hf_causal"}
    assert cfg["edit"] == {"name": "noop", "plan": {}}
    assert cfg["output"] == {"dir": "runs/source"}
    assert cfg["guards"] == {"order": ["g1", "g2"]}
    assert cfg["context"] == {"profile": "dev", "tier": "balanced"}


def test_build_subject_edit_run_config_normalizes_placeholders_and_guards() -> None:
    cfg = build_subject_edit_run_config(
        {"dataset": {"provider": "wikitext2"}},
        {"model": {"id": "<MODEL_ID>"}, "edit": {"name": "quant_rtn"}},
        subject_model_id="hf:org/model",
        adapter_name="hf_causal",
        output_dir="runs/edited",
        profile="ci",
        tier="balanced",
        guards_order=["invariants", "variance"],
    )

    assert cfg["model"] == {"id": "hf:org/model", "adapter": "hf_causal"}
    assert cfg["output"] == {"dir": "runs/edited"}
    assert cfg["context"] == {"profile": "ci", "tier": "balanced"}
    assert cfg["guards"] == {"order": ["invariants", "variance"]}


def test_resolve_evaluate_assurance_policy_rejects_unknown_assurance() -> None:
    with pytest.raises(ValueError):
        resolve_evaluate_assurance_policy(
            assurance="remote",
            allow_host_execution=False,
        )


def test_resolve_evaluate_assurance_policy_marks_trusted_local_host_enabled() -> None:
    policy = resolve_evaluate_assurance_policy(
        assurance="trusted-local",
        allow_host_execution=False,
    )

    assert policy.assurance == "trusted-local"
    assert policy.allow_host_execution is True
    assert policy.prefer_local_files_only is True
    assert policy.allow_unattested_artifacts is True


def test_resolve_evaluate_tmp_dir_uses_explicit_candidate(tmp_path: Path) -> None:
    resolved = resolve_evaluate_tmp_dir(str(tmp_path / "chosen"))

    assert resolved == (tmp_path / "chosen").resolve()
    assert resolved.exists()


def test_build_evaluate_command_plan_collects_core_execution_inputs(
    tmp_path: Path,
) -> None:
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("guards:\n  order: [shape_ok]\n", encoding="utf-8")

    plan = build_evaluate_command_plan(
        baseline_model_id="hf:org/source",
        subject_model_id="hf:org/subject",
        adapter="auto",
        profile="ci",
        tier="balanced",
        preset=str(preset_path),
        out="runs",
        edit_config=None,
        edit_label=None,
        resolve_auto_adapter_fn=lambda _model_id: "hf_causal",
        load_yaml_fn=lambda _path: {"guards": {"order": ["shape_ok"]}},
        tmp_dir_candidate=str(tmp_path / "scratch"),
    )

    assert plan.profile_name == "ci"
    assert plan.adapter_name == "hf_causal"
    assert plan.adapter_auto is True
    assert plan.source_model_id == "org/source"
    assert plan.subject_model_id == "org/subject"
    assert plan.baseline_config["output"] == {"dir": "runs/source"}
    assert plan.guards_order == ["shape_ok"]
    assert plan.subject_label == "custom"
    assert plan.tmp_dir == (tmp_path / "scratch").resolve()
