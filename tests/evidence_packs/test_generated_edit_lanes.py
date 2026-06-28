from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evidence_packs.python import create_edit_model as single_edit_mod
from scripts.evidence_packs.python import create_edits_batch as batch_edit_mod
from scripts.evidence_packs.python.editing.implementations import (
    apply_dense_lora_merge_delta,
    apply_tiny_fine_tune_update,
    build_fine_tune_validation_metadata,
    build_lora_merge_validation_metadata,
    build_validation_edit_metadata,
    resolve_edit_spec,
    storage_format_for_edit,
    validate_edit_metadata,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Linear(4, 4, bias=False)


def test_generated_lora_and_fine_tune_tensor_helpers_are_exposed() -> None:
    assert callable(apply_dense_lora_merge_delta)
    assert callable(apply_tiny_fine_tune_update)


def test_validation_edit_metadata_knows_generated_lora_and_fine_tune_formats() -> None:
    assert storage_format_for_edit("lora_merge") == "merged_dense_checkpoint"
    assert storage_format_for_edit("fine_tune") == "fine_tuned_dense_checkpoint"


def test_resolve_edit_spec_supports_generated_lora_and_fine_tune_defaults(
    tmp_path: Path,
) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()
    (model_output_dir / ".model_id").write_text("org/model", encoding="utf-8")
    tuned_file = tmp_path / "tuned_edit_params.json"
    tuned_file.write_text(
        json.dumps(
            {
                "models": {
                    "org/model": {
                        "lora_merge": {
                            "status": "selected",
                            "rank": 4,
                            "alpha": 8,
                            "scope": "attn",
                            "edit_dir_name": "lora_rank4_clean",
                        },
                        "fine_tune": {
                            "status": "selected",
                            "learning_rate": 0.0001,
                            "steps": 1,
                            "scope": "ffn",
                            "edit_dir_name": "fine_tune_step1_clean",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    lora = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="lora_merge:clean:attn",
        version_hint="clean",
        tuned_path=str(tuned_file),
    )
    fine_tune = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="fine_tune:clean:ffn",
        version_hint="clean",
        tuned_path=str(tuned_file),
    )

    assert lora.to_shell_payload() == {
        "status": "selected",
        "reason": "",
        "edit_type": "lora_merge",
        "param1": "4",
        "param2": "8",
        "scope": "attn",
        "version": "clean",
        "edit_dir_name": "lora_rank4_clean",
    }
    assert lora.to_batch_payload() | {"edit_dir_name": "lora_rank4_clean"} == {
        "type": "lora_merge",
        "status": "selected",
        "reason": "",
        "scope": "attn",
        "edit_dir_name": "lora_rank4_clean",
        "version": "clean",
        "rank": 4,
        "alpha": 8.0,
    }
    assert fine_tune.to_shell_payload() == {
        "status": "selected",
        "reason": "",
        "edit_type": "fine_tune",
        "param1": "0.0001",
        "param2": "1",
        "scope": "ffn",
        "version": "clean",
        "edit_dir_name": "fine_tune_step1_clean",
    }
    assert fine_tune.to_batch_payload() | {
        "edit_dir_name": "fine_tune_step1_clean"
    } == {
        "type": "fine_tune",
        "status": "selected",
        "reason": "",
        "scope": "ffn",
        "edit_dir_name": "fine_tune_step1_clean",
        "version": "clean",
        "learning_rate": 0.0001,
        "steps": 1,
    }


def test_generated_lora_and_fine_tune_direct_specs_validate(tmp_path: Path) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()

    lora = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="lora_merge:4:8:attn",
        version_hint="clean",
    )
    fine_tune = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="fine_tune:0.0001:1:ffn",
        version_hint="clean",
    )
    bad_lora = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="lora_merge:nope:8:attn",
        version_hint="clean",
    )
    bad_fine_tune = resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec="fine_tune:0.0001:0:ffn",
        version_hint="clean",
    )

    assert lora.selected
    assert lora.edit_dir_name == "lora_rank4_clean"
    assert lora.to_batch_payload()["rank"] == 4
    assert lora.to_batch_payload()["alpha"] == 8.0
    assert fine_tune.selected
    assert fine_tune.edit_dir_name == "fine_tune_step1_clean"
    assert fine_tune.to_batch_payload()["learning_rate"] == 0.0001
    assert fine_tune.to_batch_payload()["steps"] == 1
    assert bad_lora.status == "invalid"
    assert bad_lora.reason == "invalid_lora_rank"
    assert bad_fine_tune.status == "invalid"
    assert bad_fine_tune.reason == "invalid_fine_tune_steps"


def test_batch_edit_dir_name_covers_known_edit_families() -> None:
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "quant_rtn", "bits": 4},
            "clean",
        )
        == "quant_4bit_clean"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "fp8_quant", "format": "e4m3"},
            "clean",
        )
        == "fp8_e4m3_clean"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "magnitude_prune", "ratio": 0.25},
            "stress",
        )
        == "prune_25pct_stress"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "lowrank_svd", "rank": 16},
            "stress",
        )
        == "svd_rank16_stress"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "lora_merge", "rank": 4},
            "clean",
        )
        == "lora_rank4_clean"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "fine_tune", "steps": 3},
            "stress",
        )
        == "fine_tune_step3_stress"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "custom_family"},
            "clean",
        )
        == "custom_family_clean"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "quant_rtn", "edit_dir_name": "from_preset"},
            "clean",
        )
        == "from_preset"
    )


def test_batch_edit_strategy_and_determinism_modes(monkeypatch) -> None:
    monkeypatch.setenv("PACK_BATCH_EDIT_STRATEGY", "deepcopy")
    assert batch_edit_mod._batch_edit_strategy() == "deepcopy"

    monkeypatch.setenv("PACK_BATCH_EDIT_STRATEGY", "reload")
    assert batch_edit_mod._batch_edit_strategy() == "reload"

    monkeypatch.setenv("PACK_BATCH_EDIT_STRATEGY", "invalid")
    with pytest.raises(ValueError, match="PACK_BATCH_EDIT_STRATEGY"):
        batch_edit_mod._batch_edit_strategy()

    previous_grad = torch.is_grad_enabled()
    previous_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        monkeypatch.setenv("PACK_DETERMINISM", "strict")
        batch_edit_mod._configure_determinism()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.is_grad_enabled() is False

        monkeypatch.setenv("PACK_DETERMINISM", "throughput")
        batch_edit_mod._configure_determinism()
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_grad_enabled(previous_grad)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn


def test_batch_resolve_pending_spec_entry_paths(monkeypatch, tmp_path: Path) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()

    class Resolved:
        def __init__(
            self,
            *,
            skip: bool = False,
            selected: bool = True,
            status: str = "selected",
        ) -> None:
            self.skip = skip
            self.selected = selected
            self.status = status

        def to_batch_payload(self) -> dict[str, object]:
            return {
                "type": "lora_merge",
                "rank": 4,
                "alpha": 8,
                "scope": "attn",
            }

    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry="not-a-dict",
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(batch_edit_mod, "resolve_batch_entry", lambda **_kwargs: None)
    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn"},
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(skip=True),
    )
    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn"},
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(selected=False, status="missing"),
    )
    with pytest.raises(ValueError, match="Tuned edit preset missing"):
        batch_edit_mod._resolve_pending_spec_entry(
            spec_entry={"spec": "lora_merge:clean:attn"},
            model_output_dir=model_output_dir,
        )

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(),
    )
    monkeypatch.setattr(batch_edit_mod, "_edit_artifact_complete", lambda _path: True)
    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn"},
        model_output_dir=model_output_dir,
    ) == (None, 1, 0)

    monkeypatch.setattr(batch_edit_mod, "_edit_artifact_complete", lambda _path: False)
    pending, created, failed = batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn", "version": "clean"},
        model_output_dir=model_output_dir,
    )
    assert created == 0
    assert failed == 0
    assert pending is not None
    parsed, edit_path = pending
    assert parsed["type"] == "lora_merge"
    assert edit_path == model_output_dir / "models" / "lora_rank4_clean"


def test_batch_process_spec_counts_successes_and_failures(
    monkeypatch, tmp_path: Path
) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()
    calls: list[Path] = []

    pending = (
        {"type": "lora_merge", "rank": 4, "alpha": 8, "scope": "attn"},
        model_output_dir / "models" / "lora_rank4_clean",
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: (pending, 0, 0),
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_create_edit_artifact",
        lambda **kwargs: calls.append(kwargs["edit_path"]),
    )

    assert batch_edit_mod._process_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn"},
        model_output_dir=model_output_dir,
        model=object(),
        tokenizer=object(),
    ) == (1, 0)
    assert calls == [model_output_dir / "models" / "lora_rank4_clean"]

    monkeypatch.setattr(
        batch_edit_mod,
        "_create_edit_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert batch_edit_mod._process_spec_entry(
        spec_entry={"spec": "lora_merge:clean:attn"},
        model_output_dir=model_output_dir,
        model=object(),
        tokenizer=object(),
    ) == (0, 1)

    sequence = iter(
        [
            (None, 1, 0),
            (pending, 0, 0),
            (pending, 0, 0),
        ]
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: next(sequence),
    )
    monkeypatch.setattr(batch_edit_mod, "_load_baseline_model", lambda _path: object())
    outcomes = iter([None, RuntimeError("failed")])

    def _create_or_fail(**_kwargs: object) -> None:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome

    monkeypatch.setattr(batch_edit_mod, "_create_edit_artifact", _create_or_fail)
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    assert batch_edit_mod._process_edit_specs_reloading_model(
        edit_specs=[{"spec": "a"}, {"spec": "b"}, {"spec": "c"}],
        baseline_path=tmp_path / "baseline",
        model_output_dir=model_output_dir,
        tokenizer=object(),
    ) == (2, 1)


def test_generated_lora_and_fine_tune_tensor_helpers_mutate_target_scopes() -> None:
    class InitializedTinyModel(TinyModel):
        def __init__(self) -> None:
            super().__init__()
            with torch.no_grad():
                self.attn.weight.copy_(torch.arange(16, dtype=torch.float32).view(4, 4))
                self.mlp.weight.copy_(
                    torch.arange(16, 32, dtype=torch.float32).view(4, 4)
                )

    lora_model = InitializedTinyModel()
    before_attn = lora_model.attn.weight.detach().clone()
    before_mlp = lora_model.mlp.weight.detach().clone()
    lora_stats = apply_dense_lora_merge_delta(
        lora_model,
        rank=2,
        alpha=8,
        scope="attn",
    )

    assert lora_stats.edited_tensors == 1
    assert lora_stats.edited_params == lora_model.attn.weight.numel()
    assert lora_stats.details["rank"] == 2
    assert lora_stats.details["alpha"] == 8.0
    assert lora_stats.details["total_delta_norm"] > 0
    assert not torch.equal(lora_model.attn.weight, before_attn)
    assert torch.equal(lora_model.mlp.weight, before_mlp)

    fine_tune_model = InitializedTinyModel()
    before_attn = fine_tune_model.attn.weight.detach().clone()
    before_mlp = fine_tune_model.mlp.weight.detach().clone()
    fine_tune_stats = apply_tiny_fine_tune_update(
        fine_tune_model,
        learning_rate=0.0001,
        steps=3,
        scope="ffn",
    )

    assert fine_tune_stats.edited_tensors == 1
    assert fine_tune_stats.edited_params == fine_tune_model.mlp.weight.numel()
    assert fine_tune_stats.details["steps"] == 3
    assert fine_tune_stats.details["learning_rate"] == 0.0001
    assert fine_tune_stats.details["total_update_norm"] > 0
    assert torch.equal(fine_tune_model.attn.weight, before_attn)
    assert not torch.equal(fine_tune_model.mlp.weight, before_mlp)


def test_generated_lora_and_fine_tune_updates_respect_scale_parameters() -> None:
    class TinyAttnModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.attn.weight.fill_(1.0)

    low_alpha_model = TinyAttnModel()
    high_alpha_model = TinyAttnModel()
    apply_dense_lora_merge_delta(low_alpha_model, rank=2, alpha=2, scope="attn")
    apply_dense_lora_merge_delta(high_alpha_model, rank=2, alpha=8, scope="attn")

    assert (high_alpha_model.attn.weight - 1.0).norm() > (
        low_alpha_model.attn.weight - 1.0
    ).norm()

    one_step_model = TinyAttnModel()
    three_step_model = TinyAttnModel()
    apply_tiny_fine_tune_update(
        one_step_model,
        learning_rate=0.0001,
        steps=1,
        scope="attn",
    )
    apply_tiny_fine_tune_update(
        three_step_model,
        learning_rate=0.0001,
        steps=3,
        scope="attn",
    )

    assert (three_step_model.attn.weight - 1.0).norm() > (
        one_step_model.attn.weight - 1.0
    ).norm()


def test_generated_lora_and_fine_tune_metadata_builders_record_contract_fields() -> (
    None
):
    lora_stats = apply_dense_lora_merge_delta(
        TinyModel(),
        rank=4,
        alpha=8,
        scope="attn",
    )
    lora_metadata = build_lora_merge_validation_metadata(
        scope="attn",
        rank=4,
        alpha=8,
        stats=lora_stats,
    )

    assert lora_metadata["edit_type"] == "lora_merge"
    assert lora_metadata["actual_storage_format"] == "merged_dense_checkpoint"
    assert lora_metadata["edit_provenance"]["edit_method"] == (
        "deterministic_dense_lowrank_merge"
    )
    assert lora_metadata["edit_provenance"]["edit_count"] == 1
    assert lora_metadata["edit_topology"]["artifact_kind"] == "merged_adapter"
    assert lora_metadata["delta_privacy"]["delta_available"] == "hash_only"
    assert lora_metadata["modified_matrices"] == lora_stats.edited_tensors
    assert validate_edit_metadata(lora_metadata) == []

    fine_tune_stats = apply_tiny_fine_tune_update(
        TinyModel(),
        learning_rate=0.0001,
        steps=3,
        scope="ffn",
    )
    fine_tune_metadata = build_fine_tune_validation_metadata(
        scope="ffn",
        learning_rate=0.0001,
        steps=3,
        stats=fine_tune_stats,
    )

    assert fine_tune_metadata["edit_type"] == "fine_tune"
    assert fine_tune_metadata["actual_storage_format"] == "fine_tuned_dense_checkpoint"
    assert fine_tune_metadata["edit_provenance"]["edit_method"] == (
        "deterministic_tiny_fine_tune_update"
    )
    assert fine_tune_metadata["edit_provenance"]["edit_count"] == 3
    assert fine_tune_metadata["edit_topology"]["artifact_kind"] == "checkpoint"
    assert fine_tune_metadata["delta_privacy"]["privacy_sensitivity"] == "public"
    assert fine_tune_metadata["modified_matrices"] == fine_tune_stats.edited_tensors
    assert validate_edit_metadata(fine_tune_metadata) == []


def test_generated_lora_and_fine_tune_metadata_helpers_validate() -> None:
    stats = type(
        "Stats",
        (),
        {
            "edited_tensors": 1,
            "edited_params": 4,
            "total_params": 8,
            "details": {},
            "coverage_payload": lambda self: {
                "edited_tensors": 1,
                "edited_params": 4,
                "total_params": 8,
            },
        },
    )()

    lora_metadata = build_validation_edit_metadata(
        edit_type="lora_merge",
        scope="attn",
        parameters={"rank": 4, "alpha": 8.0},
        coverage=stats.coverage_payload(),
        edit_provenance={
            "edit_family": "lora_merge",
            "edit_method": "deterministic_dense_lowrank_merge",
            "edit_count": 1,
            "dynamic_runtime_required": False,
        },
        edit_impact={
            "scenario_types": [
                "target_success",
                "near_neighbor",
                "unrelated_locality",
                "general_ability_sentinel",
            ]
        },
        extra={
            "edit_topology": {
                "artifact_kind": "merged_adapter",
                "runtime_activation_policy": "static_merged_checkpoint",
                "training_or_edit_data_ref": "deterministic-generator-no-private-data",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "public",
                "public_raw_delta_approved": False,
            },
        },
    )
    fine_tune_metadata = build_validation_edit_metadata(
        edit_type="fine_tune",
        scope="ffn",
        parameters={"learning_rate": 0.0001, "steps": 1},
        coverage=stats.coverage_payload(),
        edit_provenance={
            "edit_family": "fine_tune",
            "edit_method": "deterministic_tiny_fine_tune_update",
            "edit_count": 1,
            "dynamic_runtime_required": False,
        },
        edit_impact={
            "scenario_types": [
                "target_success",
                "unrelated_locality",
                "general_ability_sentinel",
            ]
        },
        extra={
            "edit_topology": {
                "artifact_kind": "checkpoint",
                "runtime_activation_policy": "static_checkpoint",
                "training_or_edit_data_ref": "deterministic-generator-no-private-data",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "public",
                "public_raw_delta_approved": False,
            },
        },
    )

    assert lora_metadata["actual_storage_format"] == "merged_dense_checkpoint"
    assert fine_tune_metadata["actual_storage_format"] == "fine_tuned_dense_checkpoint"
    assert validate_edit_metadata(lora_metadata) == []
    assert validate_edit_metadata(fine_tune_metadata) == []


def test_batch_edit_artifact_executes_generated_lora_and_fine_tune_paths() -> None:
    for parsed_spec, expected_type in (
        (
            {"type": "lora_merge", "rank": 2, "alpha": 4, "scope": "attn"},
            "lora_merge",
        ),
        (
            {"type": "fine_tune", "learning_rate": 0.0001, "steps": 1, "scope": "ffn"},
            "fine_tune",
        ),
    ):
        model = TinyModel()
        _edited, metadata = batch_edit_mod._build_edited_model_and_metadata(
            model,
            parsed_spec,
            clone_model=True,
        )

        assert metadata["edit_type"] == expected_type
        assert metadata["artifact_class"] == "validation_subject_checkpoint"
        assert metadata["edit_provenance"]["dynamic_runtime_required"] is False
        assert metadata["edit_topology"]["training_or_edit_data_ref"] == (
            "deterministic-generator-no-private-data"
        )
        assert metadata["delta_privacy"]["privacy_sensitivity"] == "public"
        assert validate_edit_metadata(metadata) == []


def test_single_create_lora_and_fine_tune_validate_parameters_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(single_edit_mod, "_configure_determinism", lambda: None)

    def _fail_load(*args: object, **kwargs: object) -> tuple[object, object]:
        raise AssertionError("model load should not run for invalid parameters")

    monkeypatch.setattr(single_edit_mod, "_load_model_and_tokenizer", _fail_load)

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        single_edit_mod._create_lora_merge(
            SimpleNamespace(
                baseline_path="baseline",
                output_path="out",
                rank="0",
                alpha="8",
                scope="attn",
            )
        )
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        single_edit_mod._create_fine_tune(
            SimpleNamespace(
                baseline_path="baseline",
                output_path="out",
                learning_rate="0",
                steps="1",
                scope="ffn",
            )
        )


def test_single_create_lora_and_fine_tune_emit_generated_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(single_edit_mod, "_configure_determinism", lambda: None)
    monkeypatch.setattr(
        single_edit_mod,
        "_load_model_and_tokenizer",
        lambda _path: (TinyModel(), object()),
    )
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(
        single_edit_mod,
        "_save_model",
        lambda **kwargs: saved.append(kwargs),
    )

    assert (
        single_edit_mod._create_lora_merge(
            SimpleNamespace(
                baseline_path="baseline",
                output_path=str(tmp_path / "lora"),
                rank="2",
                alpha="8",
                scope="attn",
            )
        )
        == 0
    )
    assert (
        single_edit_mod._create_fine_tune(
            SimpleNamespace(
                baseline_path="baseline",
                output_path=str(tmp_path / "fine"),
                learning_rate="0.0001",
                steps="2",
                scope="ffn",
            )
        )
        == 0
    )

    assert [entry["metadata"]["edit_type"] for entry in saved] == [
        "lora_merge",
        "fine_tune",
    ]
    for entry in saved:
        metadata = entry["metadata"]
        assert metadata["edit_provenance"]["dynamic_runtime_required"] is False
        assert metadata["edit_topology"]["training_or_edit_data_ref"] == (
            "deterministic-generator-no-private-data"
        )
        assert metadata["delta_privacy"]["delta_available"] == "hash_only"
        assert validate_edit_metadata(metadata) == []
