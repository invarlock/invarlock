from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from invarlock.clean_selection.artifacts import build_selection_execution_receipt
from invarlock.clean_selection.common import (
    EVALUATION_SCHEDULE_SCHEMA,
    SELECTION_CONFIG_SCHEMA,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_PARAMETERS_SCHEMA,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
)
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.transformation_target_manifest import (
    TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _context_paths(tmp_path: Path, *, corrupt_runtime: bool) -> dict[str, Path]:
    baseline_path = tmp_path / "baseline"
    subject_path = tmp_path / "subject"
    baseline_path.mkdir()
    subject_path.mkdir()
    _write(baseline_path / "config.json", {"model_type": "qwen2"})
    _write(subject_path / "config.json", {"model_type": "qwen2"})
    baseline = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline_path),
    }
    artifact = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(subject_path),
    }
    transformation = {
        "edit_type": "quant_rtn",
        "parameters": {"bits": 4, "group_size": 2},
        "scope": "ffn",
    }
    config = {
        "schema": SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "org/frozen-eval",
            "revision": "c" * 40,
            "split": "validation",
            "content_sha256": "sha256:" + "d" * 64,
        },
        "seed": 17,
        "schedule": {
            "schema": EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 1,
            "max_examples": 2,
            "batch_size": 1,
            "shuffle": False,
        },
    }
    spec = {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": "quant_rtn",
        "algorithm": "groupwise_rtn_dequantized_per_row_v1",
        "parameters": transformation["parameters"],
    }
    receipt = build_selection_execution_receipt(
        original_model_key="org/model",
        candidate_id="candidate",
        transformation=transformation,
        baseline_identity=baseline,
        selection_config=config,
    )
    target_manifest = {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": "quant_rtn",
        "algorithm": spec["algorithm"],
        "parameters": transformation["parameters"],
        "scope": "ffn",
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "a" * 64,
        "layer_count": 2,
        "targets": [
            {
                "name": "model.layers.0.mlp.up_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
                "role": "ffn",
                "layer": 0,
            }
        ],
    }
    target_manifest_sha256 = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                target_manifest, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
    )
    replay = {
        "schema": TRANSFORMATION_REPLAY_SCHEMA,
        "ok": True,
        "issues": [],
        "edit_type": "quant_rtn",
        "transformation": spec,
        "algorithm": spec["algorithm"],
        "parameters": transformation["parameters"],
        "scope": "ffn",
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "a" * 64,
        "layer_count": 2,
        "target_manifest": target_manifest,
        "target_manifest_sha256": target_manifest_sha256,
        "baseline_identity": baseline,
        "artifact_identity": artifact,
    }
    runtime = {
        "schema": "invarlock/transformation-runtime-reload-proof-v1",
        "ok": True,
        "replay_schema": replay["schema"],
        "edit_type": "quant_rtn",
        "artifact_identity": artifact,
        "replay_artifact_identity": artifact,
        "prompt_sha256": "sha256:"
        + hashlib.sha256(
            b"InvarLock verifier-grade transformation runtime proof."
        ).hexdigest(),
        "device": "cpu",
        "input_device": "cpu",
        "reload_runs": 2,
        "token_ids_sha256": "sha256:" + "e" * 64,
        "token_ids_shape": [1, 2],
        "logits_sha256": "sha256:" + "f" * 64,
        "logits_shape": [1, 2, 4],
        "all_logits_finite": True,
        "repeat_deterministic": True,
        "load_diagnostics": {
            "schema": "invarlock/pretrained-load-diagnostics-v1",
            "reloads": [
                {
                    "unexpected_keys": [],
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
                {
                    "unexpected_keys": [],
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
            ],
        },
        "storage_key_audit": {
            "schema": "invarlock/safetensors-storage-key-audit-v1",
            "reloads": [
                {
                    "artifact_storage_key_count": 1,
                    "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                    "model_state_key_count": 2,
                    "model_state_keys_sha256": "sha256:" + "5" * 64,
                    "unexpected_storage_keys": [],
                },
                {
                    "artifact_storage_key_count": 1,
                    "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                    "model_state_key_count": 2,
                    "model_state_keys_sha256": "sha256:" + "5" * 64,
                    "unexpected_storage_keys": [],
                },
            ],
        },
    }
    if corrupt_runtime:
        runtime.pop("logits_shape")
    config_path = tmp_path / "config.json"
    receipt_path = tmp_path / "execution.json"
    replay_path = tmp_path / "replay.json"
    runtime_path = tmp_path / "runtime.json"
    _write(config_path, config)
    _write(receipt_path, receipt)
    _write(replay_path, replay)
    _write(runtime_path, runtime)
    return {
        "config": config_path,
        "receipt": receipt_path,
        "replay": replay_path,
        "runtime": runtime_path,
        "baseline": baseline_path,
        "subject": subject_path,
    }


def _run_real_evaluate(
    paths: dict[str, Path], *, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("invarlock")
    assert executable.is_file()
    environment = os.environ.copy()
    for name in (
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
    ):
        environment.pop(name, None)
    environment["INVARLOCK_CONTAINER_EXECUTION"] = "1"
    return subprocess.run(
        [
            str(executable),
            "evaluate",
            "--baseline",
            str(paths["baseline"]),
            "--subject",
            str(paths["subject"]),
            "--execution-mode",
            "container",
            "--assurance",
            "strict",
            "--clean-selection-config",
            str(paths["config"]),
            "--clean-selection-execution-receipt",
            str(paths["receipt"]),
            "--clean-selection-replay",
            str(paths["replay"]),
            "--clean-selection-runtime-proof",
            str(paths["runtime"]),
            "--clean-selection-repeat-index",
            "0",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_cli_accepts_complete_clean_selection_flags_before_model_work(
    tmp_path: Path,
) -> None:
    result = _run_real_evaluate(
        _context_paths(tmp_path, corrupt_runtime=False), tmp_path=tmp_path
    )

    assert result.returncode != 0  # Deliberately missing model inputs.
    assert "Invalid clean-selection evaluator context" not in result.stderr
    assert "candidate baseline checkpoint" not in result.stdout + result.stderr


def test_installed_cli_fails_closed_on_truncated_pre_evaluation_runtime_proof(
    tmp_path: Path,
) -> None:
    result = _run_real_evaluate(
        _context_paths(tmp_path, corrupt_runtime=True), tmp_path=tmp_path
    )

    assert result.returncode == 2
    assert "Invalid clean-selection evaluator context" in result.stderr
    assert "unbound or missing fields" in result.stderr
