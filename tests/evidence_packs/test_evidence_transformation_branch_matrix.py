from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.peft_runtime import PeftRuntimeError
from invarlock.training_state_evidence import TrainingStateEvidenceError
from scripts.evidence_packs.python import create_edits_batch as batch
from scripts.evidence_packs.python import preset_generator, task_tools_reports
from scripts.evidence_packs.python.editing import training_runtime as training
from scripts.evidence_packs.python.editing import (
    training_runtime_evidence,
    transformation_contract,
    validate_artifact,
)
from scripts.evidence_packs.python.editing.training_contract import (
    load_training_profile,
)


def test_batch_preflight_and_path_helpers_reject_ambiguous_edit_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    batch._preflight_reject_real_training_specs([None])
    assert batch._raw_edit_type(None) == ""
    with pytest.raises(ValueError, match="real training edit"):
        batch._get_edit_dir_name({"type": "fine_tune"}, "clean")
    assert (
        batch._get_edit_dir_name(
            {"type": "custom", "edit_dir_name": "operator-owned"}, "clean"
        )
        == "operator-owned"
    )
    with pytest.raises(ValueError, match="non-pruning edit"):
        batch._create_streaming_magnitude_prune_artifact(
            baseline_path=tmp_path,
            parsed_spec={"type": "quant_rtn"},
            edit_path=tmp_path / "out",
        )
    with pytest.raises(ValueError, match="Unsupported batch edit type"):
        batch._materialize_pending_edit_artifact(
            baseline_path=tmp_path,
            parsed_spec={"type": "custom"},
            edit_path=tmp_path / "out",
        )

    monkeypatch.setattr(
        batch,
        "canonical_transformation_spec",
        lambda *_args, **_kwargs: {"parameters": None},
    )
    with pytest.raises(
        transformation_contract.TransformationContractError,
        match="canonical transformation parameters missing",
    ):
        batch._canonical_transformation_inputs(
            {"type": "quant_rtn", "bits": 4, "group_size": 32, "scope": "ffn"}
        )


def test_batch_resolved_selection_uses_exact_canonical_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parsed = {
        "type": "synthetic_dense_update",
        "step_size": 0.001,
        "iterations": 2,
        "scope": "ffn",
    }
    expected = batch._get_edit_dir_name(parsed, "clean")
    resolved = SimpleNamespace(
        skip=False,
        selected=True,
        status="selected",
        to_batch_payload=lambda: dict(parsed),
    )
    monkeypatch.setattr(batch, "resolve_batch_entry", lambda **_kwargs: resolved)

    pending, created, failed = batch._resolve_pending_spec_entry(
        spec_entry={
            "spec": "synthetic_dense_update:clean:ffn",
            "selection_edit_dir_name": expected,
        },
        model_output_dir=tmp_path,
    )

    assert created == failed == 0
    assert pending is not None
    payload, path = pending
    assert payload["edit_dir_name"] == expected
    assert path == tmp_path / "models" / expected


def test_batch_resolution_surfaces_selection_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = SimpleNamespace(
        skip=False,
        selected=False,
        status="rejected",
        reason="candidate receipt mismatch",
        to_batch_payload=lambda: {"type": "magnitude_prune", "ratio": 0.5},
    )
    monkeypatch.setattr(batch, "resolve_batch_entry", lambda **_kwargs: resolved)

    with pytest.raises(ValueError, match="candidate receipt mismatch"):
        batch._resolve_pending_spec_entry(
            spec_entry={"spec": "magnitude_prune:0.5:ffn"},
            model_output_dir=tmp_path,
        )


def test_batch_clear_memory_releases_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(batch.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(batch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(batch.torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    batch._clear_memory()

    assert calls == ["gc", "cuda"]


def test_training_runtime_translates_package_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def state_failure(*_args: object, **_kwargs: object) -> object:
        raise TrainingStateEvidenceError("state evidence rejected")

    for package_name, invoke in (
        (
            "_package_tensor_state_sha256",
            lambda: training_runtime_evidence.tensor_state_sha256({}, torch=object()),
        ),
        (
            "_package_state_manifest",
            lambda: training_runtime_evidence._state_manifest({}, torch=object()),
        ),
        (
            "_package_streaming_lora_delta_evidence",
            lambda: training_runtime_evidence._streaming_lora_delta_evidence(
                baseline_manifest={},
                baseline_targets={},
                after={},
                torch=object(),
            ),
        ),
    ):
        monkeypatch.setattr(training_runtime_evidence, package_name, state_failure)
        with pytest.raises(
            training.TrainingRuntimeError, match="state evidence rejected"
        ):
            invoke()

    def peft_failure(*_args: object, **_kwargs: object) -> object:
        raise PeftRuntimeError("adapter state rejected")

    monkeypatch.setattr(
        training_runtime_evidence, "_package_peft_base_state", peft_failure
    )
    with pytest.raises(training.TrainingRuntimeError, match="adapter state rejected"):
        training_runtime_evidence._peft_base_state(object())
    monkeypatch.setattr(
        training_runtime_evidence, "_package_peft_merge_target_names", peft_failure
    )
    with pytest.raises(training.TrainingRuntimeError, match="adapter state rejected"):
        training_runtime_evidence._peft_merge_target_names(object(), {})


def test_training_runtime_translates_model_load_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_failure(*_args: object, **_kwargs: object) -> object:
        raise training.TrainingModelLoadError("model load rejected")

    deps = training.RuntimeDependencies(
        torch=object(),
        auto_model=object(),
        auto_tokenizer=object(),
        optimizer_cls=object(),
        transformers_version="test",
    )
    monkeypatch.setattr(training, "_package_load_model_with_diagnostics", load_failure)
    with pytest.raises(training.TrainingRuntimeError, match="model load rejected"):
        training._load_model_with_diagnostics(
            deps,
            "source",
            load_options={},
            expected_unexpected_keys=(),
            label="subject",
        )

    monkeypatch.setattr(
        training,
        "_load_model_with_diagnostics",
        lambda *_args, **_kwargs: (object(), {}),
    )
    monkeypatch.setattr(training, "_package_configure_causal_lm_loss", load_failure)
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    with pytest.raises(training.TrainingRuntimeError, match="model load rejected"):
        training._load_profile_baseline(deps, profile, load_options={})


def test_edit_artifact_cli_reports_fail_closed_result_in_both_output_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert validate_artifact._valid_digest("sha256:" + "a" * 64)
    assert not validate_artifact._valid_digest(None)
    missing = tmp_path / "missing"
    assert validate_artifact.main(["validate_artifact.py", str(missing)]) == 1
    assert "edit artifact directory not found" in capsys.readouterr().err

    assert (
        validate_artifact.main(
            ["validate_artifact.py", str(missing), "--require-metadata", "--json"]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert '"ok": false' in captured.out
    result = validate_artifact.validate_edit_artifact(missing)
    assert not result
    assert result.to_json_payload()["ok"] is False

    empty = tmp_path / "empty"
    empty.mkdir()
    incomplete = validate_artifact.validate_edit_artifact(empty, require_metadata=True)
    assert not incomplete
    assert {
        "config.json missing",
        "tokenizer files missing",
        "model weights missing or invalid",
        "edit_metadata.json missing",
    }.issubset(set(incomplete.issues or []))

    monkeypatch.setattr(
        validate_artifact,
        "validate_edit_artifact",
        lambda *_args, **_kwargs: validate_artifact.EditArtifactValidationResult(
            ok=True
        ),
    )
    assert validate_artifact.main(["validate_artifact.py", str(missing)]) == 0
    assert capsys.readouterr() == ("", "")


def test_deployable_and_pruning_cli_persist_machine_readable_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        validate_artifact,
        "validate_deployable_artifact",
        lambda *_args, **_kwargs: {"ok": True, "kind": "deployable"},
    )
    deployable_out = tmp_path / "reports" / "deployable.json"
    assert (
        validate_artifact._validate_deployable_cli(
            [str(tmp_path), "--out", str(deployable_out)]
        )
        == 0
    )
    assert '"kind": "deployable"' in deployable_out.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(
        validate_artifact,
        "validate_pruning_artifact",
        lambda *_args, **_kwargs: {"ok": False, "kind": "pruning"},
    )
    pruning_out = tmp_path / "reports" / "pruning.json"
    assert (
        validate_artifact._validate_pruning_cli(
            [
                str(tmp_path),
                "--baseline",
                str(tmp_path),
                "--scope",
                "ffn",
                "--target-sparsity",
                "0.5",
                "--out",
                str(pruning_out),
                "--json",
            ]
        )
        == 1
    )
    assert '"kind": "pruning"' in pruning_out.read_text(encoding="utf-8")
    assert '"kind": "pruning"' in capsys.readouterr().out


def test_transformation_cli_prints_result_without_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        validate_artifact,
        "validate_transformation_artifact",
        lambda *_args, **_kwargs: {"ok": True, "kind": "transformation"},
    )
    assert (
        validate_artifact._validate_transformation_cli(
            [
                str(tmp_path),
                "--baseline",
                str(tmp_path),
                "--edit-type",
                "quant_rtn",
                "--parameters-json",
                '{"bits":4,"group_size":32}',
                "--scope",
                "ffn",
            ]
        )
        == 0
    )
    assert '"kind": "transformation"' in capsys.readouterr().out


def test_preset_generator_main_resolves_defaults_and_reports_every_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def generate(**kwargs: object) -> tuple[Path, Path, list[Path]]:
        observed.update(kwargs)
        return (
            tmp_path / "preset.json",
            tmp_path / "stats.json",
            [tmp_path / "derived-a.json", tmp_path / "derived-b.json"],
        )

    monkeypatch.setattr(preset_generator, "generate_preset", generate)
    assert (
        preset_generator.main(
            [
                "--cal-dir",
                str(tmp_path / "cal"),
                "--preset-file",
                str(tmp_path / "preset.json"),
                "--model-name",
                "fixture",
                "--model-path",
                "model",
            ]
        )
        == 0
    )
    assert observed["edit_types"] == list(preset_generator.DEFAULT_PRESET_EDIT_TYPES)
    assert observed["dataset_provider"] == "wikitext2"
    output = capsys.readouterr().out
    assert "Saved preset" in output
    assert output.count("Saved derived preset") == 2

    observed.clear()
    assert (
        preset_generator.main(
            [
                "--cal-dir",
                str(tmp_path / "cal"),
                "--preset-file",
                str(tmp_path / "preset.json"),
                "--model-name",
                "fixture",
                "--model-path",
                "model",
                "--edit-types",
                "quant_rtn, magnitude_prune",
            ]
        )
        == 0
    )
    assert observed["edit_types"] == ["quant_rtn", "magnitude_prune"]


def test_preset_yaml_helpers_fail_explicitly_without_optional_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preset_generator, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML is unavailable"):
        preset_generator._yaml_safe_load("{}")
    with pytest.raises(RuntimeError, match="PyYAML is unavailable"):
        preset_generator._yaml_safe_dump({}, sort_keys=True)

    monkeypatch.setattr(preset_generator, "_YAML_AVAILABLE", False)
    monkeypatch.setenv("INVARLOCK_DATASET_PROVIDER_YAML", "kind: local_jsonl")
    with pytest.raises(SystemExit, match="PyYAML is unavailable"):
        preset_generator._resolve_dataset_provider_spec("local_jsonl")


def test_preset_hf_provider_ignores_unknown_boolean_and_binds_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("INVARLOCK_HF_TRUST_REMOTE_CODE", "not-a-boolean")
    monkeypatch.setenv("INVARLOCK_HF_CACHE_DIR", str(tmp_path))

    provider = preset_generator._resolve_dataset_provider_spec("hf_text")

    assert provider["cache_dir"] == str(tmp_path)
    assert "trust_remote_code" not in provider


def test_generate_preset_emits_all_calibrated_guard_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        preset_generator,
        "_load_guard_order_and_assurance",
        lambda _path: (["spectral", "rmt", "variance"], {"strict": True}),
    )
    monkeypatch.setattr(preset_generator, "load_records", lambda **_kwargs: [{}])
    monkeypatch.setattr(
        preset_generator,
        "calibrate_drift",
        lambda _records: {
            "mean": 1.0,
            "std": 0.0,
            "band_compatible": True,
            "suggested_band": [0.9, 1.1],
        },
    )
    monkeypatch.setattr(
        preset_generator,
        "calibrate_spectral",
        lambda _records, **_kwargs: (
            {"sigma_quantile": 0.9, "deadband": 0.01, "max_caps": 9},
            {"attn": 4},
        ),
    )
    monkeypatch.setattr(
        preset_generator,
        "calibrate_rmt",
        lambda _records, **_kwargs: (
            {"margin": 0.1, "deadband": 0.02},
            {"attn": 0.03},
        ),
    )
    monkeypatch.setattr(
        preset_generator,
        "calibrate_variance",
        lambda _records: {"relative": 0.05},
    )
    monkeypatch.setattr(preset_generator, "_YAML_AVAILABLE", True)
    monkeypatch.setattr(
        preset_generator,
        "_yaml_safe_dump",
        lambda payload, **_kwargs: "preset: " + str(bool(payload)) + "\n",
    )

    preset, stats, derived = preset_generator.generate_preset(
        cal_dir=tmp_path,
        preset_file=tmp_path / "preset.yaml",
        model_name="fixture",
        model_path="model",
        tier="balanced",
        dataset_provider="wikitext2",
        seq_len=16,
        stride=8,
        preview_n=2,
        final_n=2,
        edit_types=["quant_rtn"],
    )

    assert preset.suffix == ".yaml"
    assert stats.is_file()
    assert derived[0].suffix == ".yaml"
    stats_payload = __import__("json").loads(stats.read_text(encoding="utf-8"))
    assert stats_payload["spectral"]["family_caps"] == {"attn": 4}
    assert stats_payload["rmt"]["epsilon_by_family"] == {"attn": 0.03}


def test_generate_preset_rejects_missing_calibration_and_ignores_invalid_band(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        preset_generator,
        "_load_guard_order_and_assurance",
        lambda _path: ([], {}),
    )
    monkeypatch.setattr(preset_generator, "load_records", lambda **_kwargs: [])
    with pytest.raises(SystemExit, match="No calibration records"):
        preset_generator.generate_preset(
            cal_dir=tmp_path,
            preset_file=tmp_path / "preset.json",
            model_name="fixture",
            model_path="model",
            tier="balanced",
            dataset_provider="wikitext2",
            seq_len=16,
            stride=8,
            preview_n=2,
            final_n=2,
            edit_types=[],
        )


def test_structural_failure_report_persists_bound_runtime_manifest(
    tmp_path: Path,
) -> None:
    source_report = tmp_path / "source.json"
    source_report.write_text(
        json.dumps(
            {
                "run_id": "original",
                "meta": {"seed": 7},
                "data": {"dataset": "fixture", "preview_n": 2, "final_n": 3},
                "edit": {"name": "corrupt"},
                "metrics": {
                    "primary_metric": {
                        "kind": "accuracy",
                        "delta_vs_baseline_pp": -3.0,
                        "preview": 0.8,
                        "final": 0.7,
                        "drift_band": {"min": -1.0, "max": 1.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source-runtime.json"
    source_manifest.write_text(json.dumps({"context": None}), encoding="utf-8")
    out = tmp_path / "nested" / "failure.json"
    args = argparse.Namespace(
        out=str(out),
        source_report=str(source_report),
        source_runtime_manifest=str(source_manifest),
        error_type="truncated_checkpoint",
        message="checkpoint ended early",
        edited_report="edited.json",
        edited_events="events.jsonl",
    )

    assert task_tools_reports._structural_failure_report(args) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    manifest = json.loads(
        (out.parent / "runtime.manifest.json").read_text(encoding="utf-8")
    )
    assert payload["run_id"].startswith("original-structural-failure")
    assert payload["primary_metric"]["kind"] == "accuracy"
    assert "ratio_vs_baseline" not in payload["primary_metric"]
    assert manifest["report"]["filename"] == out.name
    assert manifest["context"]["evidence_pack_structural_failure"]["error_type"] == (
        "truncated_checkpoint"
    )


def test_structural_report_helpers_fail_closed_on_malformed_inputs(
    tmp_path: Path,
) -> None:
    assert task_tools_reports._load_json_optional(None) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert task_tools_reports._load_json_optional(malformed) is None
    malformed.write_text("[]", encoding="utf-8")
    assert task_tools_reports._load_json_optional(malformed) is None
    assert task_tools_reports._load_json_object(malformed) == {}

    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps({"scenarios": [None, {"id": ""}, {"id": "ok"}]}))
    assert set(task_tools_reports._scenario_index(scenarios)) == {"ok"}
    malformed.write_text(json.dumps({"scenarios": {}}), encoding="utf-8")
    assert task_tools_reports._scenario_index(malformed) == {}

    outside = tmp_path.parent / "outside" / "edit_metadata.json"
    assert task_tools_reports._scenario_from_report_metadata(tmp_path, outside) is None
    shallow = tmp_path / "reports" / "x"
    assert task_tools_reports._scenario_from_report_metadata(tmp_path, shallow) is None
    error_path = (
        tmp_path / "reports" / "run" / "errors" / "scenario" / "edit_metadata.json"
    )
    assert task_tools_reports._scenario_from_report_metadata(tmp_path, error_path) == (
        "scenario"
    )
    normal_path = tmp_path / "reports" / "run" / "scenario" / "x" / "edit_metadata.json"
    assert task_tools_reports._scenario_from_report_metadata(tmp_path, normal_path) == (
        "scenario"
    )

    for generation_kind, expected in (
        ("error", task_tools_reports.FAULT_INJECTION_FIXTURE),
        ("deployable_edit", task_tools_reports.DEPLOYABLE_OPTIMIZED_SUBJECT),
        ("edit", task_tools_reports.VALIDATION_SUBJECT_CHECKPOINT),
        ("unknown", "unknown"),
    ):
        assert (
            task_tools_reports._scenario_artifact_class(
                {"generation": {"kind": generation_kind}}
            )
            == expected
        )
    assert (
        task_tools_reports._scenario_artifact_class({"artifact_class": "explicit"})
        == "explicit"
    )

    with pytest.raises(ValueError, match="source report is required"):
        task_tools_reports._build_structural_base_report(None)
    fallback = task_tools_reports._build_structural_base_report(
        {
            "meta": [],
            "data": {"preview_n": "bad", "final_n": -1, "seq_len": 4},
            "edit": [],
            "artifacts": [],
            "evaluation_windows": [],
            "flags": {"fixture": True},
        }
    )
    assert fallback["dataset"]["windows"] == {
        "preview": 0,
        "final": 0,
        "seed": None,
        "stats": {},
    }
    assert fallback["dataset"]["seq_len"] == 4

    task_tools_reports._write_structural_runtime_manifest(
        out_path=tmp_path / "unused.json",
        source_runtime_manifest=None,
        error_type="unused",
        message="unused",
    )


def test_edit_summary_command_writes_output_and_structural_builder_repairs_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        task_tools_reports,
        "build_edit_artifact_summary",
        lambda *_args: {"schema": "summary", "ok": True},
    )
    output = tmp_path / "nested" / "summary.json"
    assert (
        task_tools_reports._edit_artifact_summary(
            argparse.Namespace(
                out=str(output),
                pack_dir=str(tmp_path),
                scenarios=str(tmp_path / "scenarios.json"),
            )
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True

    base = {
        "run_id": "run",
        "meta": [],
        "validation": {"existing": True},
        "guard_metric_impact": {"evaluated": False},
        "primary_metric": [],
        "spectral": {"old": True},
        "rmt": {"old": True},
    }
    payload = task_tools_reports.build_structural_failure_report(
        error_type="invalid_shape",
        message="shape mismatch",
        base_report=base,
        source_report={"data": []},
        source_report_path=None,
        edited_report_path=None,
        edited_events_path=None,
    )
    assert payload["meta"]["structural_failure"]["error_type"] == "invalid_shape"
    assert payload["validation"]["existing"] is True
    assert payload["guard_metric_impact"]["evaluated"] is False
    assert payload["spectral"]["old"] is True
    assert payload["rmt"]["old"] is True
