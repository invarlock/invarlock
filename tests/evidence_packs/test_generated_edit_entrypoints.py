from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evidence_packs.python import create_edit_model as single_edit_mod
from scripts.evidence_packs.python import create_edits_batch as batch_edit_mod


def test_single_edit_script_path_entrypoint_loads_all_editing_modules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/evidence_packs/python/create_edit_model.py"),
            "--help",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "verify-training-profile" in result.stdout


@pytest.mark.parametrize(
    ("creator", "arguments", "edit_type", "parameters", "scope"),
    [
        (
            "_create_quant_rtn",
            {"bits": "4", "group_size": "32", "scope": "FFN"},
            "quant_rtn",
            {"bits": 4, "group_size": 32},
            "ffn",
        ),
        (
            "_create_synthetic_lowrank_delta",
            {"rank": "2", "scale": "8", "scope": "attn"},
            "synthetic_lowrank_delta",
            {"rank": 2, "scale": 8.0},
            "attn",
        ),
        (
            "_create_synthetic_dense_update",
            {"step_size": "0.0001", "iterations": "2", "scope": "ffn"},
            "synthetic_dense_update",
            {"step_size": 0.0001, "iterations": 2},
            "ffn",
        ),
    ],
)
def test_single_supported_transforms_materialize_replayable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    creator: str,
    arguments: dict[str, str],
    edit_type: str,
    parameters: dict[str, object],
    scope: str,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(single_edit_mod, "_configure_determinism", lambda: None)
    monkeypatch.setattr(
        single_edit_mod,
        "materialize_transformation_artifact",
        lambda **kwargs: (
            observed.update(kwargs)
            or {
                "selected_tensors": 2,
                "selected_params": 8,
                "actual_changes": {"value_changed_params": 8},
            }
        ),
    )

    args = SimpleNamespace(
        baseline_path=str(tmp_path / "baseline"),
        output_path=str(tmp_path / "subject"),
        max_output_shard_mib=16,
        restart=True,
        **arguments,
    )
    assert getattr(single_edit_mod, creator)(args) == 0
    assert observed == {
        "baseline_path": tmp_path / "baseline",
        "output_path": tmp_path / "subject",
        "edit_type": edit_type,
        "parameters": parameters,
        "scope": scope,
        "max_output_shard_bytes": 16 * 1024 * 1024,
        "restart": True,
    }


def test_single_transform_contract_rejects_invalid_values_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(single_edit_mod, "_configure_determinism", lambda: None)
    monkeypatch.setattr(
        single_edit_mod,
        "materialize_transformation_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid values must not reach materialization")
        ),
    )

    with pytest.raises(ValueError, match="quant_rtn.bits"):
        single_edit_mod._create_quant_rtn(
            SimpleNamespace(
                baseline_path="baseline",
                output_path="out",
                bits="1",
                group_size="32",
                scope="ffn",
            )
        )


def test_single_transform_contract_rejects_unknown_family_and_malformed_canonical_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(scope="ffn")
    with pytest.raises(
        single_edit_mod.TransformationContractError,
        match="no verifier-grade transformation contract",
    ):
        single_edit_mod._canonical_transformation_inputs(args, edit_type="unknown")

    monkeypatch.setattr(
        single_edit_mod,
        "canonical_transformation_spec",
        lambda *_args: {"parameters": None},
    )
    with pytest.raises(
        single_edit_mod.TransformationContractError,
        match="canonical transformation parameters missing",
    ):
        single_edit_mod._canonical_transformation_inputs(
            SimpleNamespace(bits=4, group_size=32, scope="ffn"),
            edit_type="quant_rtn",
        )


def test_single_entrypoint_reports_contract_error_and_clears_cuda_memory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        single_edit_mod.torch,
        "set_grad_enabled",
        lambda _flag: calls.append("grad"),
    )
    monkeypatch.setattr(single_edit_mod.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(single_edit_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        single_edit_mod.torch.cuda, "empty_cache", lambda: calls.append("cuda")
    )

    single_edit_mod._configure_determinism()
    assert single_edit_mod.main(["quant-rtn", "missing", "out", "1", "32", "ffn"]) == 2

    assert calls == ["grad", "grad", "gc", "cuda"]
    assert "ERROR:" in capsys.readouterr().err
    with pytest.raises(ValueError, match="step_size"):
        single_edit_mod._create_synthetic_dense_update(
            SimpleNamespace(
                baseline_path="baseline",
                output_path="out",
                step_size="0",
                iterations="1",
                scope="ffn",
            )
        )


@pytest.mark.parametrize("subcommand", ["fp8-quant", "lowrank-svd"])
def test_single_cli_removes_unverifiable_generated_subcommands(subcommand: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        single_edit_mod.build_parser().parse_args([subcommand])
    assert exc_info.value.code == 2


def test_profile_driven_training_subcommand_is_a_real_runtime_frontdoor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    profile = SimpleNamespace(
        edit_type="lora_merge",
        profile_id="tiny_lora",
        profile_sha256="sha256:" + "1" * 64,
    )
    receipt = {
        "receipt_sha256": "sha256:" + "2" * 64,
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        single_edit_mod,
        "load_training_profile",
        lambda profile_id, **options: (
            observed.update(profile_id=profile_id, profile_options=options) or profile
        ),
    )
    monkeypatch.setattr(
        single_edit_mod,
        "run_training_profile",
        lambda loaded_profile, output_path, **options: (
            observed.update(
                loaded_profile=loaded_profile,
                output_path=output_path,
                runtime_options=options,
            )
            or SimpleNamespace(
                subject_dir=Path(output_path),
                receipt_path=Path(output_path) / "training_receipt.json",
                receipt=receipt,
            )
        ),
    )
    monkeypatch.setattr(single_edit_mod, "_clear_memory", lambda: None)

    output = tmp_path / "subject"
    assert (
        single_edit_mod.main(
            [
                "train-profile",
                "tiny_lora",
                str(output),
                "--profiles-path",
                str(tmp_path / "profiles.json"),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_id"] == "tiny_lora"
    assert payload["subject_dir"] == str(output)
    assert observed["loaded_profile"] is profile
    assert observed["runtime_options"] == {
        "repo_root": tmp_path.resolve(),
        "local_files_only": True,
    }


def test_training_profile_verifier_subcommand_recomputes_artifact_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    profile = SimpleNamespace(profile_id="tiny_full_ft")
    receipt = {"receipt_sha256": "sha256:" + "3" * 64}
    subject = tmp_path / "subject"
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        single_edit_mod,
        "load_training_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        single_edit_mod,
        "verify_training_artifact",
        lambda loaded_profile, subject_path, **options: (
            observed.update(
                profile=loaded_profile,
                subject_path=subject_path,
                options=options,
            )
            or receipt
        ),
    )
    monkeypatch.setattr(single_edit_mod, "_clear_memory", lambda: None)

    assert (
        single_edit_mod.main(
            [
                "verify-training-profile",
                "tiny_full_ft",
                str(subject),
                "--repo-root",
                str(tmp_path),
                "--allow-network",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert payload["receipt_sha256"] == receipt["receipt_sha256"]
    assert observed["profile"] is profile
    assert observed["subject_path"] == subject
    assert observed["options"]["local_files_only"] is False


@pytest.mark.parametrize(
    ("spec", "synthetic_name", "training_path"),
    [
        (
            "lora_merge:4:8:attn",
            "synthetic_lowrank_delta",
            "real PEFT/LoRA integration or training campaign",
        ),
        (
            "fine_tune:0.0001:2:ffn",
            "synthetic_dense_update",
            "real fine-tune integration or training campaign",
        ),
    ],
)
def test_batch_main_rejects_real_training_specs_before_materialization_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    spec: str,
    synthetic_name: str,
    training_path: str,
) -> None:
    def _fail_setup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real training specs must fail during batch preflight")

    monkeypatch.setattr(batch_edit_mod, "_configure_determinism", _fail_setup)

    assert (
        batch_edit_mod.main(
            [
                "--baseline",
                str(tmp_path / "baseline"),
                "--model-output-dir",
                str(tmp_path / "model"),
                "--edit-specs-json",
                json.dumps([{"spec": spec, "version": "clean"}]),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert synthetic_name in error
    assert training_path in error


def test_batch_magnitude_prune_uses_streaming_materializer_without_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()
    edit_path = model_output_dir / "models" / "prune"
    parsed = {"type": "magnitude_prune", "ratio": 0.25, "scope": "ffn"}
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: ((parsed, edit_path), 0, 0),
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "materialize_magnitude_pruned_artifact",
        lambda **kwargs: (
            observed.update(kwargs)
            or {"selected_tensors": 2, "effective_changed_params": 8}
        ),
    )
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    assert batch_edit_mod._process_edit_specs(
        edit_specs=[{"spec": "magnitude_prune:0.25:ffn"}],
        baseline_path=tmp_path / "baseline",
        model_output_dir=model_output_dir,
    ) == (1, 0)
    assert observed == {
        "baseline_path": tmp_path / "baseline",
        "output_path": edit_path,
        "sparsity": 0.25,
        "scope": "ffn",
    }


@pytest.mark.parametrize(
    ("parsed", "expected_type", "expected_parameters"),
    [
        (
            {"type": "quant_rtn", "bits": 4, "group_size": 32, "scope": "ffn"},
            "quant_rtn",
            {"bits": 4, "group_size": 32},
        ),
        (
            {
                "type": "synthetic_lowrank_delta",
                "rank": 2,
                "scale": 8.0,
                "scope": "attn",
            },
            "synthetic_lowrank_delta",
            {"rank": 2, "scale": 8.0},
        ),
        (
            {
                "type": "synthetic_dense_update",
                "step_size": 0.0001,
                "iterations": 2,
                "scope": "ffn",
            },
            "synthetic_dense_update",
            {"step_size": 0.0001, "iterations": 2},
        ),
    ],
)
def test_batch_supported_transforms_use_streaming_materializer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parsed: dict[str, object],
    expected_type: str,
    expected_parameters: dict[str, object],
) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()
    edit_path = model_output_dir / "models" / "subject"
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: ((parsed, edit_path), 0, 0),
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "materialize_transformation_artifact",
        lambda **kwargs: (
            observed.update(kwargs)
            or {
                "selected_tensors": 2,
                "actual_changes": {"value_changed_params": 8},
            }
        ),
    )
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    assert batch_edit_mod._process_edit_specs(
        edit_specs=[{"spec": "ignored"}],
        baseline_path=tmp_path / "baseline",
        model_output_dir=model_output_dir,
    ) == (1, 0)
    assert observed == {
        "baseline_path": tmp_path / "baseline",
        "output_path": edit_path,
        "edit_type": expected_type,
        "parameters": expected_parameters,
        "scope": parsed["scope"],
    }


@pytest.mark.parametrize("legacy_strategy", ["reload", "deepcopy", "invalid"])
def test_batch_main_rejects_removed_strategy_selector_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    legacy_strategy: str,
) -> None:
    monkeypatch.setenv("PACK_BATCH_EDIT_STRATEGY", legacy_strategy)
    monkeypatch.setattr(
        batch_edit_mod,
        "_configure_determinism",
        lambda: (_ for _ in ()).throw(
            AssertionError("removed strategy selector reached runtime setup")
        ),
    )

    assert (
        batch_edit_mod.main(
            [
                "--baseline",
                str(tmp_path / "baseline"),
                "--model-output-dir",
                str(tmp_path / "model"),
                "--edit-specs-json",
                json.dumps([{"spec": "magnitude_prune:0.25:ffn", "version": "stress"}]),
            ]
        )
        == 1
    )
    assert "no longer supported" in capsys.readouterr().err


def test_batch_main_refuses_forged_final_artifact_without_reuse(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    model_output_dir = tmp_path / "model"
    occupied_dir = batch_edit_mod._get_edit_dir_name(
        {
            "type": "quant_rtn",
            "bits": 4,
            "group_size": 32,
            "scope": "ffn",
        },
        "stress",
    )
    occupied_path = model_output_dir / "models" / occupied_dir
    occupied_path.mkdir(parents=True)
    # Metadata/receipt-shaped files cannot make an untrusted final tree
    # reusable; the path must be rejected before any generic completeness probe.
    for name in (
        "config.json",
        "edit_metadata.json",
        "transformation_materialization.json",
    ):
        (occupied_path / name).write_text("{}\n", encoding="utf-8")
    (occupied_path / "model.safetensors").write_bytes(b"forged final artifact")
    monkeypatch.setattr(
        batch_edit_mod,
        "materialize_transformation_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("occupied final path must not be materialized or reused")
        ),
    )
    monkeypatch.setattr(batch_edit_mod, "_configure_determinism", lambda: None)
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    assert (
        batch_edit_mod.main(
            [
                "--baseline",
                str(tmp_path / "baseline"),
                "--model-output-dir",
                str(model_output_dir),
                "--edit-specs-json",
                json.dumps(
                    [
                        {
                            "spec": "quant_rtn:4:32:ffn",
                            "version": "stress",
                        }
                    ]
                ),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "refusing final artifact reuse" in captured.err
    assert "Batch complete: 0 created, 1 failed" in captured.out


def test_batch_module_exposes_no_mutable_model_generation_apis() -> None:
    assert not hasattr(batch_edit_mod, "_build_edited_model_and_metadata")
    assert not hasattr(batch_edit_mod, "_process_edit_specs_reloading_model")


@pytest.mark.parametrize("spec", ["fp8_quant:e4m3:ffn", "lowrank_svd:8:ffn"])
def test_batch_main_rejects_unverifiable_generated_specs_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    spec: str,
) -> None:
    monkeypatch.setattr(
        batch_edit_mod,
        "_configure_determinism",
        lambda: (_ for _ in ()).throw(
            AssertionError("unsupported generated edit reached setup")
        ),
    )
    assert (
        batch_edit_mod.main(
            [
                "--baseline",
                str(tmp_path / "baseline"),
                "--model-output-dir",
                str(tmp_path / "model"),
                "--edit-specs-json",
                json.dumps([{"spec": spec, "version": "clean"}]),
            ]
        )
        == 1
    )
    assert "dedicated storage and replay contract" in capsys.readouterr().err
