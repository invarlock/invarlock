from __future__ import annotations

import hashlib
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
    canonical_json_sha256,
    raw_file_sha256,
)
from invarlock.evidence_pack_edit_common import (
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
)
from scripts.evidence_packs.python.editing.transformation_contract import (
    canonical_transformation_spec,
)
from tests.evidence_packs._support_transformation_pack import (
    _CONFIG_DIGEST,
    _IDENTITY_A,
    _IDENTITY_B,
    _IDENTITY_C,
    _LAYER_COUNT,
    _alternative_parameters,
    _candidate_target_manifest,
    _digest,
    _runtime_reload_proof,
    _write_json,
)


def _write_clean_selection_candidate(
    *,
    root: Path,
    config: dict[str, object],
    candidate_id: str,
    edit_type: str,
    parameters: dict[str, object],
    transformation: dict[str, object],
    scope: str,
    artifact_identity: dict[str, str],
    quality_loss: float,
) -> dict[str, object]:
    """Build retained sidecars for one distinct clean-selection candidate."""

    candidate_dir = root / "candidates" / candidate_id
    selected_transform = {
        "edit_type": edit_type,
        "parameters": parameters,
        "scope": scope,
    }
    execution = build_selection_execution_receipt(
        original_model_key="org/model",
        candidate_id=candidate_id,
        transformation=selected_transform,
        baseline_identity=_IDENTITY_B,
        selection_config=config,
    )
    execution_path = candidate_dir / "execution.json"
    _write_json(execution_path, execution)
    execution_sha256 = raw_file_sha256(execution_path)
    report_runs: list[dict[str, object]] = []
    for repeat_index in range(2):
        windows = {
            "preview": [repeat_index * 10, repeat_index * 10 + 1],
            "final": [repeat_index * 10 + 2, repeat_index * 10 + 3],
        }
        run_id = f"{candidate_id}-repeat-{repeat_index}"
        report = {
            "schema_version": "v1",
            "run_id": run_id,
            "artifacts": {},
            "plugins": {},
            "meta": {
                "model_id": "org/model",
                "model_identity": artifact_identity,
                "seed": 17,
            },
            "dataset": {
                "provider": "frozen_local",
                "dataset_name": "org/frozen-eval",
                "revision": "a" * 40,
                "split": "validation",
                "seq_len": 8,
                "hash": {
                    "source": "explicit_token_ids",
                    "preview": "sha256:" + "1" * 64,
                    "final": "sha256:" + "2" * 64,
                },
                "windows": {
                    "preview": 2,
                    "final": 2,
                    "seed": 17,
                    "stats": {},
                },
            },
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 2.0,
                "final": 2.02,
                "ratio_vs_baseline": 1.0 + quality_loss,
            },
            "baseline_ref": {"model_identity": _IDENTITY_B},
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
                "clean_selection_execution": {
                    "schema": EVALUATOR_PROVENANCE_SCHEMA,
                    "execution_receipt_sha256": execution_sha256,
                    "selection_config_sha256": canonical_json_sha256(config),
                    "original_model_key": "org/model",
                    "candidate_id": candidate_id,
                    "repeat_index": repeat_index,
                    "report_run_id": run_id,
                    "transformation": selected_transform,
                    "baseline_identity": _IDENTITY_B,
                    "dataset": config["dataset"],
                    "seed": 17,
                    "effective_schedule": config["schedule"],
                    "ordered_two_arm_schedule_sha256": canonical_json_sha256(windows),
                }
            },
            "clean_selection": {
                "schema": REPORT_SELECTION_BINDING_SCHEMA,
                "selection_config_sha256": canonical_json_sha256(config),
                "execution_receipt_sha256": execution_sha256,
                "candidate_id": candidate_id,
                "original_model_key": "org/model",
                "repeat_index": repeat_index,
                "transformation": selected_transform,
                "baseline_identity": _IDENTITY_B,
                "artifact_identity": artifact_identity,
                "quality_loss": quality_loss,
            },
        }
        report_path = candidate_dir / f"repeat-{repeat_index}.report.json"
        manifest_path = candidate_dir / f"repeat-{repeat_index}.runtime.manifest.json"
        _write_json(report_path, report)
        _write_json(
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
                        "transformation": selected_transform,
                        "baseline_identity": _IDENTITY_B,
                    }
                },
            },
        )
        report_runs.append(
            {
                "report": {
                    "path": str(report_path.relative_to(root)),
                    "sha256": raw_file_sha256(report_path),
                    "artifact_identity": artifact_identity,
                    "baseline_identity": _IDENTITY_B,
                },
                "runtime_manifest": {
                    "path": str(manifest_path.relative_to(root)),
                    "sha256": raw_file_sha256(manifest_path),
                },
            }
        )
    candidate_replay_path = candidate_dir / "transformation_replay.json"
    target_manifest = _candidate_target_manifest(
        transformation=transformation, scope=scope
    )
    candidate_replay = {
        "schema": TRANSFORMATION_REPLAY_SCHEMA,
        "ok": True,
        "issues": [],
        "edit_type": edit_type,
        "transformation": transformation,
        "algorithm": transformation["algorithm"],
        "parameters": parameters,
        "scope": scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": _CONFIG_DIGEST,
        "layer_count": _LAYER_COUNT,
        "target_manifest": target_manifest,
        "target_manifest_sha256": _digest(target_manifest),
        "baseline_identity": _IDENTITY_B,
        "artifact_identity": artifact_identity,
    }
    _write_json(candidate_replay_path, candidate_replay)
    candidate_runtime_path = candidate_dir / "runtime_reload_proof.json"
    _write_json(candidate_runtime_path, _runtime_reload_proof(candidate_replay))
    candidate = {
        "candidate_id": candidate_id,
        "transformation": selected_transform,
        "evaluation": {
            "schema": CANDIDATE_EVALUATION_SCHEMA,
            "selection_config_sha256": canonical_json_sha256(config),
            "execution": {
                "path": str(execution_path.relative_to(root)),
                "sha256": execution_sha256,
            },
            "reports": report_runs,
            "replay": {
                "path": str(candidate_replay_path.relative_to(root)),
                "sha256": raw_file_sha256(candidate_replay_path),
                "artifact_identity": artifact_identity,
                "baseline_identity": _IDENTITY_B,
            },
            "runtime": {
                "path": str(candidate_runtime_path.relative_to(root)),
                "sha256": raw_file_sha256(candidate_runtime_path),
                "artifact_identity": artifact_identity,
                "replay_artifact_identity": artifact_identity,
                "baseline_identity": _IDENTITY_B,
            },
            "metrics": {"quality_loss": quality_loss},
        },
    }

    return candidate


def _write_clean_selection_bundle(
    *,
    pack: Path,
    edit_type: str,
    parameters: dict[str, object],
    transformation: dict[str, object],
    scope: str,
) -> Path:
    """Build a retained, non-vacuous two-candidate selection fixture."""

    root = pack / "metadata" / "clean_selection"
    config: dict[str, object] = {
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
    candidate = _write_clean_selection_candidate(
        root=root,
        config=config,
        candidate_id="candidate",
        edit_type=edit_type,
        parameters=parameters,
        transformation=transformation,
        scope=scope,
        artifact_identity=_IDENTITY_A,
        quality_loss=0.01,
    )
    alternative_transformation = canonical_transformation_spec(
        edit_type, _alternative_parameters(edit_type)
    )
    alternative_parameters = alternative_transformation["parameters"]
    assert isinstance(alternative_parameters, dict)
    alternative = _write_clean_selection_candidate(
        root=root,
        config=config,
        candidate_id="candidate_alt",
        edit_type=edit_type,
        parameters=alternative_parameters,
        transformation=alternative_transformation,
        scope=scope,
        artifact_identity=_IDENTITY_C,
        quality_loss=0.02,
    )
    record: dict[str, object] = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": "org/model",
        "baseline_identity": _IDENTITY_B,
        "selection_domain": {
            "edit_type": edit_type,
            "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        },
        "selection_config": config,
        "decision_rule": {
            "schema": DECISION_RULE_SCHEMA,
            "kind": "lexicographic_metrics_v1",
            "metric_order": ["quality_loss"],
            "tie_breaker": "candidate_id_ascending",
        },
        "candidates": [candidate, alternative],
    }
    record["candidate_set_sha256"] = canonical_candidate_set_sha256(record)
    bundle = {
        "schema": CLEAN_SELECTION_BUNDLE_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "entries": [select_clean_transformation(record)],
    }
    bundle_path = root / "selection_bundle.json"
    _write_json(bundle_path, bundle)
    return bundle_path
