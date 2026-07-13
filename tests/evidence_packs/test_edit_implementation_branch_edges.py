from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evidence_packs.python.editing import implementations as edits


@pytest.mark.parametrize(
    ("edit_type", "param1", "param2", "expected"),
    [
        ("quant_rtn", "bad", "bad", {"bits": 0, "group_size": 0}),
        ("magnitude_prune", "bad", "", {"ratio": 0.0}),
        (
            edits.SYNTHETIC_LOWRANK_DELTA,
            "bad",
            "bad",
            {"rank": 0, "scale": 0.0},
        ),
        (
            edits.SYNTHETIC_DENSE_UPDATE,
            "bad",
            "bad",
            {"step_size": 0.0, "iterations": 0},
        ),
    ],
)
def test_resolved_spec_payloads_preserve_invalid_values_without_fabricating_numbers(
    edit_type: str,
    param1: str,
    param2: str,
    expected: dict[str, object],
) -> None:
    spec = edits.ResolvedEditSpec(
        status="invalid",
        edit_type=edit_type,
        param1=param1,
        param2=param2,
    )

    assert spec.skip is False
    assert spec.selected is False
    payload = spec.to_batch_payload()
    assert {key: payload[key] for key in expected} == expected

    skipped = edits.ResolvedEditSpec(status="skipped", edit_type=edit_type)
    assert skipped.skip is True


def test_legacy_tuned_entry_helpers_are_not_exposed() -> None:
    for name in ("_load_json_object", "_load_tuned_entry", "_model_id_for"):
        assert not hasattr(edits, name), f"{name} survived the no-legacy cutover"


@pytest.mark.parametrize(
    ("spec", "version", "expected_dir"),
    [
        ("bnb_8bit:8:all", "deployable", "quant_8bit_deployable"),
        ("bnb_4bit:4:all", "stress", "quant_4bit_stress"),
    ],
)
def test_resolve_deployable_bitsandbytes_specs(
    tmp_path: Path, spec: str, version: str, expected_dir: str
) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path,
        edit_spec=spec,
        version_hint=version,
    )

    assert resolved.status == "selected"
    assert resolved.scope == "all"
    assert resolved.edit_dir_name == expected_dir


@pytest.mark.parametrize("spec", ["bnb_4bit:8:all", "bnb_8bit:8:ffn"])
def test_resolve_deployable_bitsandbytes_specs_fail_closed(
    tmp_path: Path, spec: str
) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path,
        edit_spec=spec,
        version_hint="clean",
    )

    assert resolved.status == "invalid"


@pytest.mark.parametrize(
    "edit_spec",
    [
        "quant_rtn:clean",
        "magnitude_prune:clean:ffn",
        f"{edits.SYNTHETIC_LOWRANK_DELTA}:clean",
        f"{edits.SYNTHETIC_DENSE_UPDATE}:clean",
    ],
)
def test_clean_specs_require_a_selection_receipt(
    edit_spec: str, tmp_path: Path
) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=edit_spec,
    )

    assert not resolved.selected
    assert resolved.reason == "clean_selection_requires_receipt"
    assert resolved.version == "clean"


@pytest.mark.parametrize("edit_type", ["fp8_quant", "lowrank_svd"])
def test_clean_specs_reject_generated_families_without_replay_contract(
    edit_type: str,
    tmp_path: Path,
) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=f"{edit_type}:clean",
    )
    assert not resolved.selected
    assert resolved.reason == "generated_edit_requires_dedicated_replay_contract"


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ("quant_rtn:bad:32:ffn", "invalid_quant_params"),
        ("magnitude_prune:bad:ffn", "invalid_prune_sparsity"),
        (
            f"{edits.SYNTHETIC_LOWRANK_DELTA}:0:0.1:ffn",
            "invalid_synthetic_lowrank_rank",
        ),
        (
            f"{edits.SYNTHETIC_LOWRANK_DELTA}:2:0:ffn",
            "invalid_synthetic_lowrank_scale",
        ),
        (
            f"{edits.SYNTHETIC_DENSE_UPDATE}:0:2:ffn",
            "invalid_synthetic_dense_step_size",
        ),
        (
            f"{edits.SYNTHETIC_DENSE_UPDATE}:0.1:0:ffn",
            "invalid_synthetic_dense_iterations",
        ),
    ],
)
def test_invalid_specs_are_rejected_with_specific_reason(
    spec: str, reason: str, tmp_path: Path
) -> None:
    resolved = edits.resolve_edit_spec(model_output_dir=tmp_path, edit_spec=spec)
    assert resolved.status == "invalid"
    assert resolved.reason == reason
    assert resolved.edit_dir_name == ""


@pytest.mark.parametrize(
    "spec",
    (
        "quant_rtn:4:32:ffn:ignored",
        "synthetic_lowrank_delta:4:2:ffn:ignored",
        "synthetic_dense_update:0.01:2:ffn:ignored",
        "quant_rtn:4:32",
        "synthetic_lowrank_delta:4:2",
        "synthetic_dense_update:0.01:2",
        "quant_rtn:clean:ffn",
        "synthetic_lowrank_delta:clean:ffn",
        "synthetic_dense_update:clean:ffn",
    ),
)
def test_verifier_grade_specs_reject_unbound_or_ignored_fields(
    spec: str, tmp_path: Path
) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=spec,
        version_hint="stress",
    )

    assert resolved.status == "invalid"
    assert resolved.reason == "invalid_verifier_grade_transformation_arity"
    assert resolved.edit_dir_name == ""


@pytest.mark.parametrize(
    "spec",
    [
        "lowrank_svd:bad:ffn",
        "fp8_quant::ffn",
        "LOWRANK-SVD:bad:ffn",
        "FP8-QUANT::ffn",
    ],
)
def test_unverifiable_generated_specs_fail_closed_before_parameter_parsing(
    spec: str,
    tmp_path: Path,
) -> None:
    resolved = edits.resolve_edit_spec(model_output_dir=tmp_path, edit_spec=spec)
    assert resolved.status == "invalid"
    assert resolved.reason == "generated_edit_requires_dedicated_replay_contract"
    assert resolved.edit_dir_name == ""


def test_directory_names_and_batch_entries_are_deterministic(tmp_path: Path) -> None:
    assert (
        edits._default_edit_dir_name(
            edit_type="magnitude_prune", param1="bad", param2="", version="v1"
        )
        == "prune_0pct_v1"
    )
    assert (
        edits._default_edit_dir_name(
            edit_type="custom", param1="", param2="", version="v1"
        )
        == "custom_v1"
    )
    assert (
        edits._default_edit_dir_name(
            edit_type="custom", param1="", param2="", version=""
        )
        == ""
    )
    assert edits.resolve_batch_entry(spec_entry=[], model_output_dir=tmp_path) is None
    resolved = edits.resolve_batch_entry(
        spec_entry={"spec": "fp8_quant:e4m3:ffn", "version": "stress"},
        model_output_dir=tmp_path,
    )
    assert resolved is not None
    assert resolved.status == "invalid"
    assert resolved.edit_dir_name == ""
    assert resolved.reason == "generated_edit_requires_dedicated_replay_contract"


def test_real_training_aliases_are_rejected_by_synthetic_entrypoint() -> None:
    assert edits.real_training_edit_migration_message("fine_tune")
    with pytest.raises(ValueError, match="real PEFT/LoRA integration"):
        edits.reject_real_training_edit("lora_merge")
