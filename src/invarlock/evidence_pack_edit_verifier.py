"""Top-level evidence-pack edit proof orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_deployable_validation import (
    _deployable_binding_errors,
    _deployable_sidecar_consistency_errors,
)
from invarlock.evidence_pack_edit_common import (
    _SHA256_RE,
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    DEPLOYABLE_SIDECARS,
    PRUNING_REPLAY_SIDECAR,
    TRANSFORMATION_REPLAY_SIDECAR,
    VALIDATION_SUBJECT_CHECKPOINT,
    _load_json_sidecar,
    _report_model_name,
    _report_scenario_id,
    _typed_scenario_index_from_pack,
)
from invarlock.evidence_pack_edit_validation import (
    _metadata_consistency_errors,
)
from invarlock.evidence_pack_pruning_validation import (
    _clean_pruning_selection_errors,
    _is_clean_pruning_scenario,
    _pruning_replay_errors,
)
from invarlock.evidence_pack_scenario_contract import (
    ProofHandler,
    ScenarioContract,
)
from invarlock.evidence_pack_training_validation import (
    _require_training_evidence_proof,
)
from invarlock.evidence_pack_transformation_contract import (
    _is_exact_json_value,
    _require_runtime_reload_proof,
)
from invarlock.evidence_pack_transformation_replay import (
    _transformation_replay_errors,
)

_PruningRunBindings = dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], str]]
_TransformationRunBindings = dict[
    tuple[str, str],
    tuple[dict[str, Any], dict[str, Any], str, dict[str, object], str],
]


def _verify_transformation_report(
    *,
    pack_dir: Path,
    scenario_id: str,
    spec: dict[str, Any],
    contract: ScenarioContract,
    report: dict[str, Any],
    metadata: dict[str, Any],
    report_dir: Path,
    report_model_name: str | None,
    seen_replays: set[str],
    run_bindings: _TransformationRunBindings,
) -> list[str]:
    errors: list[str] = []
    sidecar_path = report_dir / TRANSFORMATION_REPLAY_SIDECAR
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        return [
            f"{scenario_id}: transformation replay sidecar missing: "
            f"{TRANSFORMATION_REPLAY_SIDECAR}"
        ]

    payload, sidecar_error = _load_json_sidecar(sidecar_path)
    if sidecar_error is not None or payload is None:
        return [
            f"{scenario_id}: transformation replay sidecar invalid: {sidecar_error}"
        ]
    seen_replays.add(scenario_id)
    errors.extend(
        _transformation_replay_errors(
            scenario_id=scenario_id,
            report=report,
            metadata=metadata,
            payload=payload,
            spec=spec,
            pack_dir=pack_dir,
            report_dir=report_dir,
            report_model_name=report_model_name,
        )
    )
    errors.extend(
        _require_runtime_reload_proof(
            scenario_id=scenario_id,
            report_dir=report_dir,
            report=report,
            replay=payload,
            expected_edit_type=(
                contract.edit.edit_type.value if contract.edit is not None else ""
            ),
        )
    )
    artifact_identity = payload.get("artifact_identity")
    baseline_identity = payload.get("baseline_identity")
    target_manifest_sha256 = payload.get("target_manifest_sha256")
    transformation = payload.get("transformation")
    scope = payload.get("scope")
    if not (
        isinstance(artifact_identity, dict)
        and isinstance(baseline_identity, dict)
        and isinstance(target_manifest_sha256, str)
        and _SHA256_RE.fullmatch(target_manifest_sha256) is not None
        and isinstance(transformation, dict)
        and isinstance(scope, str)
    ):
        return errors
    if report_model_name is None:
        errors.append(f"{scenario_id}: transformation report has no model path")
        return errors

    key = (report_model_name, scenario_id)
    current = (
        artifact_identity,
        baseline_identity,
        target_manifest_sha256,
        transformation,
        scope,
    )
    previous = run_bindings.get(key)
    if previous is None:
        run_bindings[key] = current
        return errors
    label = f"{report_model_name}/{scenario_id}"
    if previous[0] != current[0]:
        errors.append(
            label + ": repeated transformation runs disagree on artifact identity"
        )
    if previous[1] != current[1]:
        errors.append(
            label + ": repeated transformation runs disagree on baseline identity"
        )
    if previous[2] != current[2]:
        errors.append(
            label + ": repeated transformation runs disagree on target manifest digest"
        )
    if not _is_exact_json_value(previous[3], current[3]):
        errors.append(
            label + ": repeated transformation runs disagree on transformation contract"
        )
    if previous[4] != current[4]:
        errors.append(label + ": repeated transformation runs disagree on scope")
    return errors


def _verify_pruning_report(
    *,
    pack_dir: Path,
    scenario_id: str,
    spec: dict[str, Any],
    report_path: Path,
    report: dict[str, Any],
    metadata: dict[str, Any],
    report_dir: Path,
    report_model_name: str | None,
    seen_replays: set[str],
    run_bindings: _PruningRunBindings,
) -> list[str]:
    errors: list[str] = []
    sidecar_path = report_dir / PRUNING_REPLAY_SIDECAR
    if not sidecar_path.is_file():
        return [
            f"{scenario_id}: pruning replay sidecar missing: {PRUNING_REPLAY_SIDECAR}"
        ]

    payload, sidecar_error = _load_json_sidecar(sidecar_path)
    if sidecar_error is not None or payload is None:
        return [f"{scenario_id}: pruning replay sidecar invalid: {sidecar_error}"]
    seen_replays.add(scenario_id)
    errors.extend(
        _pruning_replay_errors(
            scenario_id=scenario_id,
            report=report,
            metadata=metadata,
            payload=payload,
            spec=spec,
        )
    )
    if _is_clean_pruning_scenario(spec):
        errors.extend(
            _clean_pruning_selection_errors(
                pack_dir=pack_dir,
                scenario_id=scenario_id,
                report_path=report_path,
                report_dir=report_dir,
                report_model_name=report_model_name,
                report=report,
                metadata=metadata,
                payload=payload,
            )
        )
    errors.extend(
        _require_runtime_reload_proof(
            scenario_id=scenario_id,
            report_dir=report_dir,
            report=report,
            replay=payload,
            expected_edit_type="magnitude_prune",
        )
    )
    artifact_identity = payload.get("artifact_identity")
    baseline_identity = payload.get("baseline_identity")
    target_manifest_sha256 = payload.get("target_manifest_sha256")
    if not (
        isinstance(artifact_identity, dict)
        and isinstance(baseline_identity, dict)
        and isinstance(target_manifest_sha256, str)
        and _SHA256_RE.fullmatch(target_manifest_sha256) is not None
    ):
        return errors
    if report_model_name is None:
        errors.append(f"{scenario_id}: pruning report has no model path")
        return errors

    key = (report_model_name, scenario_id)
    current = (artifact_identity, baseline_identity, target_manifest_sha256)
    previous = run_bindings.get(key)
    if previous is None:
        run_bindings[key] = current
        return errors
    label = f"{report_model_name}/{scenario_id}"
    if previous[0] != current[0]:
        errors.append(label + ": repeated pruning runs disagree on artifact identity")
    if previous[1] != current[1]:
        errors.append(label + ": repeated pruning runs disagree on baseline identity")
    if previous[2] != current[2]:
        errors.append(
            label + ": repeated pruning runs disagree on target manifest digest"
        )
    return errors


def _verify_edit_metadata_consistency(pack_dir: Path) -> list[str]:
    scenarios, contracts, errors = _typed_scenario_index_from_pack(pack_dir)
    if not scenarios and not errors:
        return []
    if errors:
        return errors

    deployable_scenarios = {
        scenario_id
        for scenario_id, contract in contracts.items()
        if contract.proof_handler is ProofHandler.DEPLOYABLE_BITSANDBYTES
    }
    seen_deployable_reports: set[str] = set()
    pruning_scenarios = {
        scenario_id
        for scenario_id, contract in contracts.items()
        if contract.proof_handler is ProofHandler.MAGNITUDE_PRUNING_REPLAY
    }
    transformation_scenarios = {
        scenario_id
        for scenario_id, contract in contracts.items()
        if contract.proof_handler is ProofHandler.TRANSFORMATION_REPLAY
    }
    training_scenarios = {
        scenario_id
        for scenario_id, contract in contracts.items()
        if contract.proof_handler is ProofHandler.EXTERNAL_TRAINING
    }
    seen_pruning_reports: set[str] = set()
    seen_pruning_replays: set[str] = set()
    seen_transformation_reports: set[str] = set()
    seen_transformation_replays: set[str] = set()
    seen_training_reports: set[str] = set()
    seen_training_proofs: set[str] = set()
    pruning_run_bindings: _PruningRunBindings = {}
    transformation_run_bindings: _TransformationRunBindings = {}

    for report_path in sorted(pack_dir.glob("reports/**/evaluation.report.json")):
        scenario_id = _report_scenario_id(pack_dir, report_path)
        if scenario_id is None:
            continue
        contract = contracts.get(scenario_id)
        spec = scenarios.get(scenario_id)
        if contract is None or not isinstance(spec, dict):
            errors.append(f"{scenario_id}: report has no accepted typed scenario")
            continue
        artifact_class = contract.artifact_class.value
        if artifact_class not in {
            VALIDATION_SUBJECT_CHECKPOINT,
            DEPLOYABLE_OPTIMIZED_SUBJECT,
        }:
            continue
        declared_pruning = (
            contract.proof_handler is ProofHandler.MAGNITUDE_PRUNING_REPLAY
        )
        if declared_pruning:
            seen_pruning_reports.add(scenario_id)
        declared_transformation = (
            contract.proof_handler is ProofHandler.TRANSFORMATION_REPLAY
        )
        if declared_transformation:
            seen_transformation_reports.add(scenario_id)
        declared_training = contract.proof_handler is ProofHandler.EXTERNAL_TRAINING
        if declared_training:
            seen_training_reports.add(scenario_id)

        report_dir = report_path.parent
        report_model_name = _report_model_name(pack_dir, report_path)
        report, report_error = _load_json_sidecar(report_path)
        if report_error is not None or report is None:
            errors.append(f"{scenario_id}: evaluation report invalid: {report_error}")
            continue
        metadata_path = report_dir / "edit_metadata.json"
        if not metadata_path.is_file():
            errors.append(f"{scenario_id}: edit_metadata.json missing next to report")
            continue
        metadata, metadata_error = _load_json_sidecar(metadata_path)
        if metadata_error is not None or metadata is None:
            errors.append(
                f"{scenario_id}: edit_metadata.json invalid: {metadata_error}"
            )
            continue
        errors.extend(
            _metadata_consistency_errors(
                scenario_id=scenario_id,
                spec=spec,
                metadata=metadata,
            )
        )

        if contract.proof_handler is ProofHandler.DEPLOYABLE_BITSANDBYTES:
            seen_deployable_reports.add(scenario_id)
            deployable_payloads: dict[str, dict[str, Any]] = {}
            for sidecar in DEPLOYABLE_SIDECARS:
                sidecar_path = report_dir / sidecar
                if not sidecar_path.is_file():
                    errors.append(
                        f"{scenario_id}: deployable sidecar missing: {sidecar}"
                    )
                else:
                    payload, sidecar_error = _load_json_sidecar(sidecar_path)
                    if sidecar_error is not None:
                        errors.append(
                            f"{scenario_id}: deployable sidecar invalid "
                            f"({sidecar}): {sidecar_error}"
                        )
                    else:
                        deployable_payloads[sidecar] = cast(dict[str, Any], payload)
                        errors.extend(
                            _deployable_sidecar_consistency_errors(
                                scenario_id=scenario_id,
                                sidecar=sidecar,
                                payload=cast(dict[str, Any], payload),
                            )
                        )
            if all(sidecar in deployable_payloads for sidecar in DEPLOYABLE_SIDECARS):
                errors.extend(
                    _deployable_binding_errors(
                        scenario_id=scenario_id,
                        spec=spec,
                        report=report,
                        metadata=metadata,
                        report_dir=report_dir,
                        sidecars=deployable_payloads,
                    )
                )
        if declared_transformation:
            errors.extend(
                _verify_transformation_report(
                    pack_dir=pack_dir,
                    scenario_id=scenario_id,
                    spec=spec,
                    contract=contract,
                    report=report,
                    metadata=metadata,
                    report_dir=report_dir,
                    report_model_name=report_model_name,
                    seen_replays=seen_transformation_replays,
                    run_bindings=transformation_run_bindings,
                )
            )
        if declared_pruning:
            errors.extend(
                _verify_pruning_report(
                    pack_dir=pack_dir,
                    scenario_id=scenario_id,
                    spec=spec,
                    report_path=report_path,
                    report=report,
                    metadata=metadata,
                    report_dir=report_dir,
                    report_model_name=report_model_name,
                    seen_replays=seen_pruning_replays,
                    run_bindings=pruning_run_bindings,
                )
            )
        if declared_training:
            training_errors = _require_training_evidence_proof(
                pack_dir=pack_dir,
                scenario_id=scenario_id,
                contract=contract,
                report_dir=report_dir,
                report=report,
            )
            errors.extend(training_errors)
            if not training_errors:
                seen_training_proofs.add(scenario_id)

    for scenario_id in sorted(deployable_scenarios - seen_deployable_reports):
        errors.append(
            f"{scenario_id}: deployable scenario has no deployability report sidecars"
        )
    for scenario_id in sorted(pruning_scenarios):
        if scenario_id not in seen_pruning_reports:
            errors.append(
                f"{scenario_id}: active magnitude-prune scenario has no evaluation report"
            )
        elif scenario_id not in seen_pruning_replays:
            errors.append(
                f"{scenario_id}: active magnitude-prune scenario has no pruning replay coverage"
            )
    for scenario_id in sorted(transformation_scenarios):
        if scenario_id not in seen_transformation_reports:
            errors.append(
                f"{scenario_id}: active generated transformation scenario has no evaluation report"
            )
        elif scenario_id not in seen_transformation_replays:
            errors.append(
                f"{scenario_id}: active generated transformation scenario has no transformation replay coverage"
            )
    for scenario_id in sorted(training_scenarios):
        if scenario_id not in seen_training_reports:
            errors.append(
                f"{scenario_id}: active training scenario has no evaluation report"
            )
        elif scenario_id not in seen_training_proofs:
            errors.append(
                f"{scenario_id}: active training scenario has no valid artifact-replay proof"
            )
    return errors
