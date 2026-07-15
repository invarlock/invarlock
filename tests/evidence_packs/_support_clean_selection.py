from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invarlock.clean_selection.artifacts import build_selection_execution_receipt
from invarlock.clean_selection.bundle import select_clean_transformation
from invarlock.clean_selection.candidate import canonical_candidate_set_sha256
from invarlock.clean_selection.common import (
    CANDIDATE_EVALUATION_SCHEMA,
    CANDIDATE_RECORD_SCHEMA,
    CLEAN_SELECTION_BUNDLE_SCHEMA,
    CLEAN_SELECTION_CONTRACT_VERSION,
    DECISION_RULE_SCHEMA,
    EVALUATION_SCHEDULE_SCHEMA,
    EVALUATOR_PROVENANCE_SCHEMA,
    REPORT_SELECTION_BINDING_SCHEMA,
    SELECTION_CONFIG_SCHEMA,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_PARAMETERS_SCHEMA,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
    canonical_json_sha256,
    raw_file_sha256,
)
from invarlock.transformation_target_manifest import (
    TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
)

_RUNTIME_RELOAD_PROMPT_SHA256 = (
    "sha256:"
    + hashlib.sha256(
        b"InvarLock verifier-grade transformation runtime proof."
    ).hexdigest()
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _identity(character: str) -> dict[str, str]:
    return {"kind": "local_checkpoint_tree", "sha256": "sha256:" + character * 64}


def _selection_config() -> dict[str, object]:
    return {
        "schema": SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "org/frozen-eval",
            "revision": "a" * 40,
            "split": "validation",
            "content_sha256": "sha256:" + "b" * 64,
        },
        "seed": 17,
        "schedule": {
            "schema": EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 2,
            "max_examples": 2,
            "batch_size": 1,
            "shuffle": False,
        },
    }


def _transformation_spec(
    *, edit_type: str, parameters: dict[str, object]
) -> dict[str, object]:
    algorithms = {
        "quant_rtn": "groupwise_rtn_dequantized_per_row_v1",
        "synthetic_lowrank_delta": "deterministic_synthetic_lowrank_delta_v1",
        "synthetic_dense_update": "deterministic_synthetic_dense_update_v1",
    }
    return {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": edit_type,
        "algorithm": algorithms[edit_type],
        "parameters": parameters,
    }


def _target_manifest(
    *, transformation: dict[str, object], scope: str
) -> dict[str, object]:
    role, name = {
        "ffn": ("ffn", "model.layers.0.mlp.up_proj.weight"),
        "attn": ("attn", "model.layers.0.self_attn.q_proj.weight"),
    }[scope]
    return {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": transformation["edit_type"],
        "algorithm": transformation["algorithm"],
        "parameters": transformation["parameters"],
        "scope": scope,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "a" * 64,
        "layer_count": 2,
        "targets": [
            {
                "name": name,
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
                "role": role,
                "layer": 0,
            }
        ],
    }


def _native_execution_provenance(
    *,
    config: dict[str, object],
    execution_sha256: str,
    candidate_id: str,
    transformation: dict[str, object],
    baseline: dict[str, str],
    run_id: str,
    repeat_index: int,
    windows: dict[str, list[int]],
) -> dict[str, object]:
    return {
        "schema": EVALUATOR_PROVENANCE_SCHEMA,
        "execution_receipt_sha256": execution_sha256,
        "selection_config_sha256": canonical_json_sha256(config),
        "original_model_key": "org/model",
        "candidate_id": candidate_id,
        "repeat_index": repeat_index,
        "report_run_id": run_id,
        "transformation": transformation,
        "baseline_identity": baseline,
        "dataset": config["dataset"],
        "seed": config["seed"],
        "effective_schedule": config["schedule"],
        "ordered_two_arm_schedule_sha256": canonical_json_sha256(windows),
    }


def _candidate_report(
    *,
    root: Path,
    candidate_id: str,
    repeat_index: int,
    artifact: dict[str, str],
    baseline: dict[str, str],
    transformation: dict[str, object],
    config: dict[str, object],
    execution_sha256: str,
    quality_ratio: float,
    bind_report: bool,
) -> tuple[Path, Path]:
    report_path = (
        root / "candidates" / candidate_id / f"repeat-{repeat_index}.report.json"
    )
    manifest_path = (
        root
        / "candidates"
        / candidate_id
        / f"repeat-{repeat_index}.runtime.manifest.json"
    )
    start = repeat_index * 10
    windows = {
        "preview": [start, start + 1],
        "final": [start + 2, start + 3],
    }
    run_id = f"{candidate_id}-repeat-{repeat_index}"
    report: dict[str, object] = {
        "schema_version": "v1",
        "run_id": run_id,
        "artifacts": {},
        "plugins": {},
        "meta": {
            "model_id": "org/model",
            "model_identity": artifact,
            "seed": config["seed"],
        },
        "dataset": {
            "provider": "frozen_local",
            "dataset_name": config["dataset"]["name"],  # type: ignore[index]
            "revision": config["dataset"]["revision"],  # type: ignore[index]
            "split": config["dataset"]["split"],  # type: ignore[index]
            "seq_len": 8,
            "hash": {
                "source": "explicit_token_ids",
                "preview": "sha256:" + "1" * 64,
                "final": "sha256:" + "2" * 64,
            },
            "windows": {
                "preview": 2,
                "final": 2,
                "seed": config["seed"],
                "stats": {},
            },
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 2.0,
            "final": 2.0 * quality_ratio,
            "ratio_vs_baseline": quality_ratio,
        },
        "baseline_ref": {"model_identity": baseline},
        "assurance": {
            "mode": "strict",
            "report_local_verdict": "pass",
            "canonical_guard_chain_enforced": True,
            "fallback_fields_used": False,
            "blocking_reasons": [],
        },
        "validation": {
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "primary_metric_tail_acceptable": True,
            "guard_metric_impact_acceptable": True,
            "guard_warning_policy_acceptable": True,
        },
        "invariants": {"passed": True, "supported": True},
        "evaluation_windows": {
            "preview": {"window_ids": windows["preview"]},
            "final": {"window_ids": windows["final"]},
        },
        "provenance": {
            "clean_selection_execution": _native_execution_provenance(
                config=config,
                execution_sha256=execution_sha256,
                candidate_id=candidate_id,
                transformation=transformation,
                baseline=baseline,
                run_id=run_id,
                repeat_index=repeat_index,
                windows=windows,
            )
        },
    }
    if bind_report:
        report["clean_selection"] = {
            "schema": REPORT_SELECTION_BINDING_SCHEMA,
            "selection_config_sha256": canonical_json_sha256(config),
            "execution_receipt_sha256": execution_sha256,
            "candidate_id": candidate_id,
            "original_model_key": "org/model",
            "repeat_index": repeat_index,
            "transformation": transformation,
            "baseline_identity": baseline,
            "artifact_identity": artifact,
            "quality_loss": quality_ratio - 1.0,
        }
    _write(report_path, report)
    _write(
        manifest_path,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-07-10T12:00:00+00:00",
            "verifier_contract_version": "runtime-manifest-v1",
            "report": {
                "path": report_path.name,
                "filename": report_path.name,
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
            "config": {
                "path": "selection_config.json",
                "sha256": "f" * 64,
                "source": "file",
            },
            "execution_mode": "container",
            "runtime": {
                "image_ref": "invarlock-test@sha256:" + "e" * 64,
                "image_digest": "sha256:" + "e" * 64,
                "container_execution": True,
                "allow_network": False,
                "allow_remote_code": False,
                "allow_third_party_plugins": False,
            },
            "context": {
                "clean_selection_execution": {
                    "execution_receipt_sha256": execution_sha256,
                    "selection_config_sha256": canonical_json_sha256(config),
                    "original_model_key": "org/model",
                    "candidate_id": candidate_id,
                    "repeat_index": repeat_index,
                    "report_run_id": run_id,
                    "transformation": transformation,
                    "baseline_identity": baseline,
                }
            },
        },
    )
    return report_path, manifest_path


def _candidate(
    root: Path,
    *,
    candidate_id: str,
    parameters: dict[str, object],
    scope: str,
    quality_ratio: float,
    artifact_character: str,
    config: dict[str, object],
    bind_reports: bool = True,
) -> dict[str, object]:
    baseline = _identity("c")
    artifact = _identity(artifact_character)
    transformation = {
        "edit_type": "quant_rtn",
        "parameters": parameters,
        "scope": scope,
    }
    candidate_dir = root / "candidates" / candidate_id
    execution_path = candidate_dir / "execution.json"
    execution = build_selection_execution_receipt(
        original_model_key="org/model",
        candidate_id=candidate_id,
        transformation=transformation,
        baseline_identity=baseline,
        selection_config=config,
    )
    _write(execution_path, execution)
    execution_sha256 = raw_file_sha256(execution_path)
    report_runs: list[dict[str, object]] = []
    for repeat_index in range(2):
        report_path, manifest_path = _candidate_report(
            root=root,
            candidate_id=candidate_id,
            repeat_index=repeat_index,
            artifact=artifact,
            baseline=baseline,
            transformation=transformation,
            config=config,
            execution_sha256=execution_sha256,
            quality_ratio=quality_ratio,
            bind_report=bind_reports,
        )
        report_runs.append(
            {
                "report": {
                    "path": str(report_path.relative_to(root)),
                    "sha256": raw_file_sha256(report_path),
                    "artifact_identity": artifact,
                    "baseline_identity": baseline,
                },
                "runtime_manifest": {
                    "path": str(manifest_path.relative_to(root)),
                    "sha256": raw_file_sha256(manifest_path),
                },
            }
        )
    spec = _transformation_spec(edit_type="quant_rtn", parameters=parameters)
    target_manifest = _target_manifest(transformation=spec, scope=scope)
    replay_path = candidate_dir / "transformation_replay.json"
    runtime_path = candidate_dir / "runtime_reload_proof.json"
    _write(
        replay_path,
        {
            "schema": TRANSFORMATION_REPLAY_SCHEMA,
            "ok": True,
            "issues": [],
            "edit_type": "quant_rtn",
            "transformation": spec,
            "algorithm": spec["algorithm"],
            "parameters": parameters,
            "scope": scope,
            "scope_policy": TRANSFORMATION_SCOPE_POLICY,
            "model_type": "qwen2",
            "architecture": "decoder",
            "config_sha256": "sha256:" + "a" * 64,
            "layer_count": 2,
            "target_manifest": target_manifest,
            "target_manifest_sha256": canonical_json_sha256(target_manifest),
            "baseline_identity": baseline,
            "artifact_identity": artifact,
        },
    )
    _write(
        runtime_path,
        {
            "schema": "invarlock/transformation-runtime-reload-proof-v1",
            "ok": True,
            "replay_schema": TRANSFORMATION_REPLAY_SCHEMA,
            "edit_type": "quant_rtn",
            "artifact_identity": artifact,
            "replay_artifact_identity": artifact,
            "prompt_sha256": _RUNTIME_RELOAD_PROMPT_SHA256,
            "device": "cpu",
            "input_device": "cpu",
            "token_ids_sha256": "sha256:" + "3" * 64,
            "token_ids_shape": [1, 2],
            "logits_sha256": "sha256:" + "4" * 64,
            "logits_shape": [1, 2, 4],
            "all_logits_finite": True,
            "repeat_deterministic": True,
            "reload_runs": 2,
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
                        "artifact_storage_keys_sha256": "sha256:" + "5" * 64,
                        "model_state_key_count": 2,
                        "model_state_keys_sha256": "sha256:" + "6" * 64,
                        "unexpected_storage_keys": [],
                    },
                    {
                        "artifact_storage_key_count": 1,
                        "artifact_storage_keys_sha256": "sha256:" + "5" * 64,
                        "model_state_key_count": 2,
                        "model_state_keys_sha256": "sha256:" + "6" * 64,
                        "unexpected_storage_keys": [],
                    },
                ],
            },
        },
    )
    return {
        "candidate_id": candidate_id,
        "transformation": transformation,
        "evaluation": {
            "schema": CANDIDATE_EVALUATION_SCHEMA,
            "selection_config_sha256": canonical_json_sha256(config),
            "execution": {
                "path": str(execution_path.relative_to(root)),
                "sha256": execution_sha256,
            },
            "reports": report_runs,
            "replay": {
                "path": str(replay_path.relative_to(root)),
                "sha256": raw_file_sha256(replay_path),
                "artifact_identity": artifact,
                "baseline_identity": baseline,
            },
            "runtime": {
                "path": str(runtime_path.relative_to(root)),
                "sha256": raw_file_sha256(runtime_path),
                "artifact_identity": artifact,
                "replay_artifact_identity": artifact,
                "baseline_identity": baseline,
            },
            "metrics": {"quality_loss": quality_ratio - 1.0},
        },
    }


def _record(root: Path, *, bind_reports: bool = True) -> dict[str, object]:
    config = _selection_config()
    record: dict[str, object] = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": "org/model",
        "baseline_identity": _identity("c"),
        "selection_domain": {
            "edit_type": "quant_rtn",
            "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        },
        "selection_config": config,
        "decision_rule": {
            "schema": DECISION_RULE_SCHEMA,
            "kind": "lexicographic_metrics_v1",
            "metric_order": ["quality_loss"],
            "tie_breaker": "candidate_id_ascending",
        },
        "candidates": [
            _candidate(
                root,
                candidate_id="attn8",
                parameters={"bits": 8, "group_size": 32},
                scope="attn",
                quality_ratio=1.01,
                artifact_character="d",
                config=config,
                bind_reports=bind_reports,
            ),
            _candidate(
                root,
                candidate_id="ffn4",
                parameters={"bits": 4, "group_size": 32},
                scope="ffn",
                quality_ratio=1.04,
                artifact_character="e",
                config=config,
                bind_reports=bind_reports,
            ),
        ],
    }
    record["candidate_set_sha256"] = canonical_candidate_set_sha256(record)
    return record


def _bundle(root: Path, record: dict[str, object]) -> tuple[Path, dict[str, object]]:
    selected = select_clean_transformation(record)
    bundle = {
        "schema": CLEAN_SELECTION_BUNDLE_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "entries": [selected],
    }
    bundle_path = root / "selection_bundle.json"
    _write(bundle_path, bundle)
    return bundle_path, bundle


def _refresh_bundle(
    root: Path, record: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    record["candidate_set_sha256"] = canonical_candidate_set_sha256(
        {key: value for key, value in record.items() if key != "candidate_set_sha256"}
    )
    return _bundle(root, record)


def _candidate_mapping(record: dict[str, object], index: int = 0) -> dict[str, object]:
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[index]
    assert isinstance(candidate, dict)
    return candidate


def _report_reference(
    candidate: dict[str, object], repeat_index: int
) -> dict[str, object]:
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    report_run = reports[repeat_index]
    assert isinstance(report_run, dict)
    reference = report_run["report"]
    assert isinstance(reference, dict)
    return reference


def _manifest_reference(
    candidate: dict[str, object], repeat_index: int
) -> dict[str, object]:
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    report_run = reports[repeat_index]
    assert isinstance(report_run, dict)
    reference = report_run["runtime_manifest"]
    assert isinstance(reference, dict)
    return reference


def _refresh_report_manifest(
    root: Path, candidate: dict[str, object], repeat_index: int
) -> None:
    report_ref = _report_reference(candidate, repeat_index)
    manifest_ref = _manifest_reference(candidate, repeat_index)
    report_path = root / str(report_ref["path"])
    manifest_path = root / str(manifest_ref["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _write(manifest_path, manifest)
    report_ref["sha256"] = raw_file_sha256(report_path)
    manifest_ref["sha256"] = raw_file_sha256(manifest_path)
