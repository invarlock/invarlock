from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.core.evaluate_plan import (
    DEFAULT_EVALUATE_GUARDS_ORDER,
    EvaluateExecutionPolicy,
    build_baseline_run_config,
    build_evaluate_command_plan,
    build_subject_edit_run_config,
    build_subject_noop_run_config,
    default_preset_data_for_adapter,
    determine_subject_label,
    normalize_model_id,
    resolve_evaluate_execution_policy,
    resolve_evaluate_tmp_dir,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
    stable_text,
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
        normalize_model_id("hf:org/model", _BadAdapterName())


def test_text_normalization_helpers_use_fallback_for_expected_text_errors() -> None:
    class _BadText:
        def __str__(self) -> str:
            raise TypeError("no text")

    assert normalize_model_id("hf:org/model", _BadText()) == "hf:org/model"
    assert stable_text(_BadText(), fallback="fallback") == "fallback"


def test_sanitize_preset_data_for_evaluate_removes_pinned_device() -> None:
    payload = {"model": {"device": "cuda", "id": "demo"}}

    sanitized = sanitize_preset_data_for_evaluate(payload)

    assert sanitized == {"model": {"id": "demo"}}
    assert payload == {"model": {"device": "cuda", "id": "demo"}}


def test_sanitize_preset_data_for_evaluate_fills_missing_dataset_windows() -> None:
    payload = {
        "dataset": {
            "provider": "local_jsonl",
            "file": "samples.jsonl",
            "text_field": "text",
            "max_samples": 16,
        }
    }

    sanitized = sanitize_preset_data_for_evaluate(payload, adapter_name="hf_causal")

    assert sanitized["dataset"]["provider"] == "local_jsonl"
    assert sanitized["dataset"]["file"] == "samples.jsonl"
    assert sanitized["dataset"]["max_samples"] == 16
    assert sanitized["dataset"]["preview_n"] == 64
    assert sanitized["dataset"]["final_n"] == 64
    assert sanitized["dataset"]["seq_len"] == 512
    assert sanitized["dataset"]["stride"] == 512
    assert "preview_n" not in payload["dataset"]


def test_sanitize_preset_data_for_evaluate_keeps_non_mapping_dataset() -> None:
    payload = {"dataset": "custom-provider"}

    sanitized = sanitize_preset_data_for_evaluate(payload, adapter_name="hf_causal")

    assert sanitized == {"dataset": "custom-provider"}


def test_resolve_guards_order_prefers_preset_then_default() -> None:
    assert resolve_guards_order({}) == DEFAULT_EVALUATE_GUARDS_ORDER
    assert resolve_guards_order({"guards": {"order": ["g1", "g2"]}}) == ["g1", "g2"]
    with pytest.raises(ValueError, match="canonical guard chain"):
        resolve_guards_order(
            {"guards": {"order": ["g1", "g2"]}},
            require_canonical=True,
        )


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
        assurance_mode="off",
    )

    assert cfg["model"] == {"id": "org/model", "adapter": "hf_causal"}
    assert cfg["edit"] == {"name": "noop", "plan": {}}
    assert cfg["output"] == {"dir": "runs/source"}
    assert cfg["guards"] == {"order": ["g1", "g2"]}
    assert cfg["context"] == {
        "profile": "dev",
        "tier": "balanced",
        "assurance": {"mode": "off"},
        "runtime": {"execution_mode": "unknown"},
    }
    assert cfg["assurance"] == {"mode": "off"}


def test_build_baseline_run_config_uses_only_typed_model_identity() -> None:
    identity = {"kind": "remote_revision", "revision": "0" * 40}
    cfg = build_baseline_run_config(
        {},
        model_id="org/model",
        adapter_name="hf_causal",
        model_identity=identity,
        output_dir="runs/source",
        profile="ci",
        tier="balanced",
        guards_order=["invariants"],
        assurance_mode="off",
    )

    assert cfg["model"] == {
        "id": "org/model",
        "adapter": "hf_causal",
        "model_identity": identity,
    }


@pytest.mark.parametrize(
    "legacy_field",
    ["revision", "model_revision", "model_checkpoint_tree_sha256"],
)
def test_build_baseline_run_config_rejects_legacy_identity_fields(
    legacy_field: str,
) -> None:
    with pytest.raises(ValueError, match="Preset config uses legacy model identity"):
        build_baseline_run_config(
            {"model": {legacy_field: "stale"}},
            model_id="org/model",
            adapter_name="hf_causal",
            model_identity={"kind": "remote_revision", "revision": "a" * 40},
            output_dir="runs/source",
            profile="ci",
            tier="balanced",
            guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
            assurance_mode="strict",
        )


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
        assurance_mode="off",
    )

    assert cfg["model"] == {"id": "hf:org/model", "adapter": "hf_causal"}
    assert cfg["output"] == {"dir": "runs/edited"}
    assert cfg["context"] == {
        "profile": "ci",
        "tier": "balanced",
        "assurance": {"mode": "off"},
        "runtime": {"execution_mode": "unknown"},
    }
    assert cfg["guards"] == {"order": ["invariants", "variance"]}


@pytest.mark.parametrize(
    "legacy_field",
    ["revision", "model_revision", "model_checkpoint_tree_sha256"],
)
def test_build_subject_edit_run_config_rejects_legacy_identity_fields(
    legacy_field: str,
) -> None:
    with pytest.raises(ValueError, match="legacy model identity field"):
        build_subject_edit_run_config(
            {},
            {
                "model": {"id": "<MODEL_ID>", legacy_field: "stale"},
                "edit": {"name": "noop"},
            },
            subject_model_id="org/model",
            adapter_name="hf_causal",
            model_identity={"kind": "remote_revision", "revision": "a" * 40},
            output_dir="runs/edited",
            profile="ci",
            tier="balanced",
            guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
            assurance_mode="strict",
        )


@pytest.mark.parametrize(
    "legacy_field",
    ["revision", "model_revision", "model_checkpoint_tree_sha256"],
)
def test_build_subject_edit_run_config_rejects_legacy_preset_identity_fields(
    legacy_field: str,
) -> None:
    with pytest.raises(ValueError, match="Preset config uses legacy model identity"):
        build_subject_edit_run_config(
            {"model": {legacy_field: "stale"}},
            {"model": {"id": "<MODEL_ID>"}, "edit": {"name": "noop"}},
            subject_model_id="org/model",
            adapter_name="hf_causal",
            model_identity={"kind": "remote_revision", "revision": "a" * 40},
            output_dir="runs/edited",
            profile="ci",
            tier="balanced",
            guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
            assurance_mode="strict",
        )


def test_subject_configs_propagate_only_typed_model_identity() -> None:
    revision = "a" * 40
    digest = "sha256:" + "b" * 64
    noop = build_subject_noop_run_config(
        {},
        model_id="org/model",
        adapter_name="hf_causal",
        model_identity={"kind": "remote_revision", "revision": revision},
        output_dir="runs/edited",
        profile="ci",
        tier="balanced",
        guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
        assurance_mode="strict",
    )
    edited = build_subject_edit_run_config(
        {},
        {"model": {"id": "<MODEL_ID>"}, "edit": {"name": "quant_rtn"}},
        subject_model_id="/models/local",
        adapter_name="hf_causal",
        model_identity={"kind": "local_checkpoint_tree", "sha256": digest},
        output_dir="runs/edited",
        profile="ci",
        tier="balanced",
        guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
        assurance_mode="strict",
    )
    edited_remote = build_subject_edit_run_config(
        {},
        {"model": {"id": "<MODEL_ID>"}, "edit": {"name": "quant_rtn"}},
        subject_model_id="org/model",
        adapter_name="hf_causal",
        model_identity={"kind": "remote_revision", "revision": revision},
        output_dir="runs/edited-remote",
        profile="ci",
        tier="balanced",
        guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
        assurance_mode="strict",
    )

    assert noop["model"]["model_identity"]["revision"] == revision
    assert "revision" not in noop["model"]
    assert "model_checkpoint_tree_sha256" not in noop["model"]
    assert edited["model"]["model_identity"]["sha256"] == digest
    assert "model_checkpoint_tree_sha256" not in edited["model"]
    assert "revision" not in edited["model"]
    assert edited_remote["model"]["model_identity"]["revision"] == revision
    assert "revision" not in edited_remote["model"]
    assert "model_checkpoint_tree_sha256" not in edited_remote["model"]


def test_strict_remote_evaluate_requires_canonical_revisions(tmp_path: Path) -> None:
    common = {
        "baseline_model_id": "org/source",
        "subject_model_id": "org/subject",
        "baseline_adapter": "hf_causal",
        "subject_adapter": "hf_causal",
        "profile": "ci",
        "tier": "balanced",
        "preset": None,
        "out": "runs",
        "edit_config": None,
        "edit_label": None,
        "resolve_auto_adapter_fn": lambda _model_id: "hf_causal",
        "load_yaml_fn": lambda _path: {},
        "tmp_dir_candidate": str(tmp_path / "scratch"),
        "assurance_mode": "strict",
    }
    with pytest.raises(ValueError, match="baseline.*40-64 lowercase hexadecimal"):
        build_evaluate_command_plan(**common)
    with pytest.raises(ValueError, match="subject.*40-64 lowercase hexadecimal"):
        build_evaluate_command_plan(**common, baseline_revision="a" * 40)

    plan = build_evaluate_command_plan(
        **common,
        baseline_revision="a" * 40,
        subject_revision="b" * 64,
    )
    assert plan.baseline_config["model"]["model_identity"] == {
        "kind": "remote_revision",
        "revision": "a" * 40,
    }
    assert "revision" not in plan.baseline_config["model"]
    assert plan.subject_identity == {
        "kind": "remote_revision",
        "revision": "b" * 64,
    }


def test_strict_local_evaluate_hashes_checkpoint_trees(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    subject = tmp_path / "subject"
    for root, weight in ((baseline, b"base"), (subject, b"subject")):
        root.mkdir()
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "model.safetensors").write_bytes(weight)

    plan = build_evaluate_command_plan(
        baseline_model_id=str(baseline),
        subject_model_id=str(subject),
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        profile="ci",
        tier="balanced",
        preset=None,
        out="runs",
        edit_config=None,
        edit_label=None,
        resolve_auto_adapter_fn=lambda _model_id: "hf_causal",
        load_yaml_fn=lambda _path: {},
        tmp_dir_candidate=str(tmp_path / "scratch"),
        assurance_mode="strict",
    )

    assert plan.baseline_config["model"]["model_identity"]["sha256"].startswith(
        "sha256:"
    )
    assert plan.subject_identity is not None
    assert plan.subject_identity["sha256"].startswith("sha256:")
    assert "model_checkpoint_tree_sha256" not in plan.baseline_config["model"]
    assert "revision" not in plan.baseline_config["model"]


def test_build_subject_edit_run_config_strict_rejects_custom_preset_guard_order() -> (
    None
):
    with pytest.raises(ValueError, match="canonical guard chain"):
        build_subject_edit_run_config(
            {"dataset": {"provider": "wikitext2"}},
            {"guards": {"order": ["variance"]}},
            subject_model_id="org/model",
            adapter_name="hf_causal",
            output_dir="runs/edited",
            profile="ci",
            tier="balanced",
            guards_order=DEFAULT_EVALUATE_GUARDS_ORDER,
            assurance_mode="strict",
        )


def test_resolve_evaluate_execution_policy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        resolve_evaluate_execution_policy(
            execution_mode="remote",
            allow_host_execution=False,
        )


def test_resolve_evaluate_execution_policy_marks_host_mode_host_enabled() -> None:
    policy = resolve_evaluate_execution_policy(
        execution_mode="host",
        allow_host_execution=False,
    )

    assert isinstance(policy, EvaluateExecutionPolicy)
    assert policy.execution_mode == "host"
    assert policy.allow_host_execution is True
    assert policy.prefer_local_files_only is True
    assert policy.allow_unverified_provenance is True


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
        baseline_revision="0" * 40,
        profile="ci",
        tier="balanced",
        preset=str(preset_path),
        out="runs",
        edit_config=None,
        edit_label=None,
        resolve_auto_adapter_fn=lambda _model_id: "hf_causal",
        load_yaml_fn=lambda _path: {"guards": {"order": ["shape_ok"]}},
        tmp_dir_candidate=str(tmp_path / "scratch"),
        assurance_mode="off",
    )

    assert plan.profile_name == "ci"
    assert plan.baseline_adapter_name == "hf_causal"
    assert plan.subject_adapter_name == "hf_causal"
    assert plan.adapter_auto is True
    assert plan.baseline_adapter_auto is True
    assert plan.subject_adapter_auto is True
    assert plan.source_model_id == "org/source"
    assert plan.subject_model_id == "org/subject"
    assert plan.baseline_config["output"] == {"dir": "runs/source"}
    assert plan.baseline_config["model"]["model_identity"] == {
        "kind": "remote_revision",
        "revision": "0" * 40,
    }
    assert "revision" not in plan.baseline_config["model"]
    assert "revision" not in plan.preset_data.get("model", {})
    assert plan.guards_order == ["shape_ok"]
    assert plan.assurance_mode == "off"
    assert plan.subject_label == "custom"
    assert plan.tmp_dir == (tmp_path / "scratch").resolve()


def test_build_evaluate_command_plan_supports_split_side_adapters(
    tmp_path: Path,
) -> None:
    resolved: list[str] = []

    def resolve_auto(model_id: str) -> str:
        resolved.append(model_id)
        return "hf_bnb"

    plan = build_evaluate_command_plan(
        baseline_model_id="hf:org/source",
        subject_model_id="hf:org/subject-4bit",
        baseline_adapter="hf_causal",
        subject_adapter="auto",
        profile="ci",
        tier="balanced",
        preset=None,
        out="runs",
        edit_config=None,
        edit_label=None,
        resolve_auto_adapter_fn=resolve_auto,
        load_yaml_fn=lambda _path: {},
        tmp_dir_candidate=str(tmp_path / "scratch"),
        assurance_mode="off",
    )

    assert resolved == ["hf:org/subject-4bit"]
    assert plan.baseline_adapter_name == "hf_causal"
    assert plan.subject_adapter_name == "hf_bnb"
    assert plan.adapter_auto is True
    assert plan.baseline_adapter_auto is False
    assert plan.subject_adapter_auto is True
    assert plan.source_model_id == "org/source"
    assert plan.subject_model_id == "org/subject-4bit"
    assert plan.baseline_config["model"]["adapter"] == "hf_causal"


def test_build_evaluate_command_plan_strict_rejects_custom_guard_order(
    tmp_path: Path,
) -> None:
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("guards:\n  order: [shape_ok]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical guard chain"):
        build_evaluate_command_plan(
            baseline_model_id="hf:org/source",
            subject_model_id="hf:org/subject",
            profile="ci",
            tier="balanced",
            preset=str(preset_path),
            out="runs",
            edit_config=None,
            edit_label=None,
            resolve_auto_adapter_fn=lambda _model_id: "hf_causal",
            load_yaml_fn=lambda _path: {"guards": {"order": ["shape_ok"]}},
            tmp_dir_candidate=str(tmp_path / "scratch"),
            assurance_mode="strict",
        )


def test_build_evaluate_command_plan_strict_rejects_dev_and_aggressive(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="profile ci or release"):
        build_evaluate_command_plan(
            baseline_model_id="gpt2",
            subject_model_id="gpt2",
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            profile="dev",
            tier="aggressive",
            preset=None,
            out="runs",
            edit_config=None,
            edit_label=None,
            resolve_auto_adapter_fn=lambda _model_id: "hf_causal",
            load_yaml_fn=lambda _path: {},
            tmp_dir_candidate=str(tmp_path / "scratch"),
            assurance_mode="strict",
        )
