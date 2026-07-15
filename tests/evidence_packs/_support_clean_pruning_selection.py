from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from invarlock.clean_pruning_selection_artifacts import (
    build_clean_pruning_candidate_report_binding,
    build_clean_pruning_evaluator_execution_provenance,
)
from invarlock.clean_pruning_selection_common import (
    CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA,
    CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA,
    CLEAN_PRUNING_DECISION_RULE_SCHEMA,
    CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA,
    CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA,
    CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA,
    CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
    CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
    PRUNING_ALGORITHM,
    PRUNING_REPLAY_SCHEMA,
    PRUNING_SCOPE_POLICY,
    PRUNING_STORAGE_POLICY,
    PRUNING_TARGET_MANIFEST_SCHEMA,
    canonical_json_sha256,
    raw_file_sha256,
)
from invarlock.clean_pruning_selection_contract import (
    build_clean_pruning_execution_receipt,
    canonical_clean_pruning_bundle_sha256,
    canonical_clean_pruning_candidate_set_sha256,
    select_clean_pruning,
)
from invarlock.clean_pruning_selection_contracts.snapshot import (
    referenced_clean_pruning_candidate_paths,
)
from scripts.evidence_packs.python.editing.clean_pruning_selection_bundle import (
    stage_clean_pruning_selection_bundle,
)
from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
)

_RUNTIME_PROMPT_SHA256 = (
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
        "schema": CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA,
        "dataset": {
            "name": "org/frozen-pruning-eval",
            "revision": "a" * 40,
            "split": "validation",
            "content_sha256": "sha256:" + "b" * 64,
        },
        "seed": 17,
        "schedule": {
            "schema": CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA,
            "candidate_order": "candidate_id_ascending",
            "evaluation_repeats": 2,
            "max_examples": 2,
            "batch_size": 1,
            "shuffle": False,
        },
    }


def _pruning(scope: str, sparsity: float) -> dict[str, object]:
    return {
        "edit_type": "magnitude_prune",
        "scope": scope,
        "target_sparsity": sparsity,
    }


def _target_manifest(scope: str) -> dict[str, object]:
    target_name = (
        "model.layers.0.self_attn.q_proj.weight"
        if scope == "attn"
        else "model.layers.0.mlp.up_proj.weight"
    )
    return {
        "schema": PRUNING_TARGET_MANIFEST_SCHEMA,
        "scope": scope,
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "c" * 64,
        "targets": [
            {
                "name": target_name,
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
            }
        ],
    }


def _replay(
    *,
    pruning: dict[str, object],
    baseline: dict[str, str],
    artifact: dict[str, str],
) -> dict[str, object]:
    manifest = _target_manifest(cast(str, pruning["scope"]))
    return {
        "schema": PRUNING_REPLAY_SCHEMA,
        "ok": True,
        "edit_type": "magnitude_prune",
        "scope": pruning["scope"],
        "target_sparsity": pruning["target_sparsity"],
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "c" * 64,
        "target_manifest": manifest,
        "target_manifest_sha256": canonical_json_sha256(manifest),
        "baseline_identity": baseline,
        "artifact_identity": artifact,
        "checked_tensors": 2,
        "selected_tensors": 1,
        "selected_params": 4,
        "total_params": 8,
        "expected_pruned_params": 2,
        "expected_changed_params": 2,
        "observed_changed_params": 2,
        "original_zero_params": 0,
        "observed_zero_params": 2,
        "out_of_scope_tensors_checked": 1,
        "out_of_scope_bytes_checked": 16,
        "support_files_checked": 2,
        "issues": [],
    }


def _runtime(artifact: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "invarlock/transformation-runtime-reload-proof-v1",
        "ok": True,
        "replay_schema": PRUNING_REPLAY_SCHEMA,
        "edit_type": "magnitude_prune",
        "artifact_identity": artifact,
        "replay_artifact_identity": artifact,
        "prompt_sha256": _RUNTIME_PROMPT_SHA256,
        "device": "cpu",
        "input_device": "cpu",
        "reload_runs": 2,
        "token_ids_sha256": "sha256:" + "3" * 64,
        "token_ids_shape": [1, 2],
        "logits_sha256": "sha256:" + "4" * 64,
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
    }


def _bind_clean_replay_to_manifest(
    replay: dict[str, object],
    manifest: dict[str, object],
) -> None:
    """Keep every ordinary replay field self-consistent with a forged manifest."""

    for field in ("scope", "model_type", "architecture", "config_sha256"):
        replay[field] = manifest[field]
    replay["target_manifest"] = manifest
    replay["target_manifest_sha256"] = canonical_json_sha256(manifest)
    targets = manifest["targets"]
    assert isinstance(targets, list)
    selected_params = sum(
        int(target["numel"])
        for target in targets
        if isinstance(target, dict) and isinstance(target.get("numel"), int)
    )
    selected_tensors = len(targets)
    expected_pruned = selected_params // 2
    replay["selected_tensors"] = selected_tensors
    replay["selected_params"] = selected_params
    replay["checked_tensors"] = selected_tensors + 1
    replay["expected_pruned_params"] = expected_pruned
    replay["expected_changed_params"] = expected_pruned
    replay["observed_changed_params"] = expected_pruned
    replay["observed_zero_params"] = expected_pruned


def _make_targets_noncanonical(manifest: dict[str, object]) -> None:
    manifest["targets"] = [
        {
            "name": "model.layers.1.mlp.down_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
        },
        {
            "name": "model.layers.0.mlp.up_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
        },
    ]


def _write_report(
    *,
    root: Path,
    candidate_id: str,
    repeat_index: int,
    baseline: dict[str, str],
    artifact: dict[str, str],
    pruning: dict[str, object],
    config: dict[str, object],
    execution_receipt: dict[str, object],
    execution_sha256: str,
    replay: dict[str, object],
    runtime: dict[str, object],
    quality_ratio: float,
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
    windows = {"preview": [start, start + 1], "final": [start + 2, start + 3]}
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
    }
    report["provenance"] = {
        "clean_pruning_selection_execution": (
            build_clean_pruning_evaluator_execution_provenance(
                report=report,
                execution_receipt=execution_receipt,
                execution_receipt_sha256=execution_sha256,
                repeat_index=repeat_index,
            )
        )
    }
    report["clean_pruning_selection"] = build_clean_pruning_candidate_report_binding(
        report=report,
        replay=replay,
        runtime=runtime,
        original_model_key="org/model",
        candidate_id=candidate_id,
        pruning=pruning,
        selection_config=config,
        execution_receipt=execution_receipt,
        execution_receipt_sha256=execution_sha256,
        repeat_index=repeat_index,
    )
    _write(report_path, report)
    _write(
        manifest_path,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-07-11T12:00:00+00:00",
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
                "clean_pruning_selection_execution": {
                    "execution_receipt_sha256": execution_sha256,
                    "selection_config_sha256": canonical_json_sha256(config),
                    "original_model_key": "org/model",
                    "candidate_id": candidate_id,
                    "repeat_index": repeat_index,
                    "report_run_id": run_id,
                    "pruning": pruning,
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
    pruning: dict[str, object],
    artifact_char: str,
    quality_ratios: tuple[float, float],
    config: dict[str, object],
) -> dict[str, object]:
    baseline = _identity("c")
    artifact = _identity(artifact_char)
    candidate_dir = root / "candidates" / candidate_id
    execution_path = candidate_dir / "execution.json"
    execution = build_clean_pruning_execution_receipt(
        original_model_key="org/model",
        candidate_id=candidate_id,
        pruning=pruning,
        baseline_identity=baseline,
        selection_config=config,
    )
    _write(execution_path, execution)
    execution_sha256 = raw_file_sha256(execution_path)
    replay = _replay(pruning=pruning, baseline=baseline, artifact=artifact)
    runtime = _runtime(artifact)
    reports: list[dict[str, object]] = []
    for repeat_index, quality_ratio in enumerate(quality_ratios):
        report_path, manifest_path = _write_report(
            root=root,
            candidate_id=candidate_id,
            repeat_index=repeat_index,
            baseline=baseline,
            artifact=artifact,
            pruning=pruning,
            config=config,
            execution_receipt=execution,
            execution_sha256=execution_sha256,
            replay=replay,
            runtime=runtime,
            quality_ratio=quality_ratio,
        )
        reports.append(
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
    replay_path = candidate_dir / "pruning_replay.json"
    runtime_path = candidate_dir / "runtime_reload_proof.json"
    _write(replay_path, replay)
    _write(runtime_path, runtime)
    return {
        "candidate_id": candidate_id,
        "pruning": pruning,
        "evaluation": {
            "schema": CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA,
            "selection_config_sha256": canonical_json_sha256(config),
            "execution": {
                "path": str(execution_path.relative_to(root)),
                "sha256": execution_sha256,
            },
            "reports": reports,
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
            "metrics": {
                "quality_loss": (sum(quality_ratios) / len(quality_ratios)) - 1.0
            },
        },
    }


def _record(root: Path) -> dict[str, object]:
    config = _selection_config()
    record: dict[str, object] = {
        "schema": CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "original_model_key": "org/model",
        "baseline_identity": _identity("c"),
        "selection_domain": {
            "edit_type": "magnitude_prune",
            "scope_policy": PRUNING_SCOPE_POLICY,
            "pruning_algorithm": PRUNING_ALGORITHM,
            "storage_policy": PRUNING_STORAGE_POLICY,
            "target_manifest_schema": PRUNING_TARGET_MANIFEST_SCHEMA,
        },
        "selection_config": config,
        "decision_rule": {
            "schema": CLEAN_PRUNING_DECISION_RULE_SCHEMA,
            "metric": "mean_quality_loss_from_strict_reports_v1",
            "direction": "minimize",
            "tie_breaker": "candidate_id_ascending",
        },
        "candidates": [
            _candidate(
                root,
                candidate_id="attn-20",
                pruning=_pruning("attn", 0.5),
                artifact_char="d",
                quality_ratios=(1.04, 1.02),
                config=config,
            ),
            _candidate(
                root,
                candidate_id="ffn-20",
                pruning=_pruning("ffn", 0.5),
                artifact_char="e",
                quality_ratios=(1.01, 1.03),
                config=config,
            ),
        ],
    }
    record["candidate_set_sha256"] = canonical_clean_pruning_candidate_set_sha256(
        record
    )
    return record


def _bundle(root: Path, record: dict[str, object]) -> tuple[Path, dict[str, object]]:
    selected = select_clean_pruning(record)
    entries = [selected]
    bundle = {
        "schema": CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "entries": entries,
        "bundle_sha256": canonical_clean_pruning_bundle_sha256(entries),
    }
    path = root / "clean_pruning_selection_bundle.json"
    _write(path, bundle)
    return path, bundle


def _candidate_mapping(record: dict[str, object], index: int) -> dict[str, object]:
    candidates = cast(list[dict[str, object]], record["candidates"])
    return candidates[index]


def _refresh_record_and_bundle(
    root: Path, record: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    record["candidate_set_sha256"] = canonical_clean_pruning_candidate_set_sha256(
        {key: value for key, value in record.items() if key != "candidate_set_sha256"}
    )
    return _bundle(root, record)


def _refresh_report_and_manifest(
    root: Path, candidate: dict[str, object], repeat_index: int
) -> None:
    evaluation = cast(dict[str, object], candidate["evaluation"])
    reports = cast(list[dict[str, object]], evaluation["reports"])
    report_run = reports[repeat_index]
    report_ref = cast(dict[str, object], report_run["report"])
    manifest_ref = cast(dict[str, str], report_run["runtime_manifest"])
    report_path = root / cast(str, report_ref["path"])
    manifest_path = root / manifest_ref["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _write(manifest_path, manifest)
    report_ref["sha256"] = raw_file_sha256(report_path)
    manifest_ref["sha256"] = raw_file_sha256(manifest_path)


def _stage_snapshot(root: Path, bundle_path: Path, bundle: dict[str, object]) -> Path:
    stage = root / "metadata" / "clean_pruning_selection"
    stage.mkdir(parents=True)
    (stage / CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME).write_bytes(
        bundle_path.read_bytes()
    )
    for relative in referenced_clean_pruning_candidate_paths(bundle):
        source = root / relative
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return stage


def _final_clean_pruning_pack(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Build one final pack by importing the winner's exact campaign bytes."""

    campaign = tmp_path / "campaign"
    record = _record(campaign)
    bundle_path, bundle = _bundle(campaign, record)
    pack = tmp_path / "pack"
    selection_root = pack / "metadata" / "clean_pruning_selection"
    assert (
        stage_clean_pruning_selection_bundle(
            bundle_path=bundle_path,
            destination=selection_root,
            evidence_root=campaign,
        )
        == selection_root / CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME
    )
    _write(
        pack / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {
                    "id": "prune_clean",
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": "magnitude_prune:clean",
                        "version": "clean",
                    },
                }
            ]
        },
    )

    selected_wrapper = cast(dict[str, object], bundle["entries"][0])
    selected = cast(dict[str, object], selected_wrapper["selected_entry"])
    receipt = cast(dict[str, object], selected["selection_receipt"])
    evaluation = cast(dict[str, object], receipt["selected_evaluation"])
    report_run = cast(list[dict[str, object]], evaluation["reports"])[0]
    report_reference = cast(dict[str, object], report_run["report"])
    manifest_reference = cast(dict[str, object], report_run["runtime_manifest"])
    replay_reference = cast(dict[str, object], evaluation["replay"])
    runtime_reference = cast(dict[str, object], evaluation["runtime"])
    report_dir = pack / "reports" / "org__model" / "prune_clean" / "run_1"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "evaluation.report.json"
    manifest_path = report_dir / "runtime.manifest.json"
    replay_path = report_dir / "pruning_replay.json"
    runtime_path = report_dir / "runtime_reload_proof.json"
    report_path.write_bytes(
        campaign.joinpath(cast(str, report_reference["path"])).read_bytes()
    )
    manifest_path.write_bytes(
        campaign.joinpath(cast(str, manifest_reference["path"])).read_bytes()
    )
    replay_path.write_bytes(
        campaign.joinpath(cast(str, replay_reference["path"])).read_bytes()
    )
    runtime_path.write_bytes(
        campaign.joinpath(cast(str, runtime_reference["path"])).read_bytes()
    )

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    pruning = cast(dict[str, object], receipt["selected_pruning"])
    _write(
        report_dir / "edit_metadata.json",
        build_validation_edit_metadata(
            edit_type="magnitude_prune",
            scope=cast(str, pruning["scope"]),
            parameters={"target_sparsity": pruning["target_sparsity"]},
            coverage={
                "edited_tensors": replay["selected_tensors"],
                "edited_params": replay["selected_params"],
                "total_params": 8,
                "coverage_ratio": 0.5,
            },
            extra={
                "target_sparsity": pruning["target_sparsity"],
                "actual_sparsity": pruning["target_sparsity"],
                "effective_changed_params": replay["observed_changed_params"],
                "scope_policy": replay["scope_policy"],
                "pruning_algorithm": replay["pruning_algorithm"],
                "storage_policy": replay["storage_policy"],
                "model_type": replay["model_type"],
                "pruning_architecture": replay["architecture"],
                "config_sha256": replay["config_sha256"],
                "target_manifest": replay["target_manifest"],
                "target_manifest_sha256": replay["target_manifest_sha256"],
            },
        ),
    )
    return pack, report_path, manifest_path, runtime_path
