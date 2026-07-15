from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .editing.implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
    )

EDIT_ARTIFACT_SUMMARY_SCHEMA = "invarlock/evidence-pack-edit-artifact-summary-v1"


def _load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scenario_index(scenarios_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(scenarios_path)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        scenario_id = item.get("id")
        if isinstance(scenario_id, str) and scenario_id:
            result[scenario_id] = item
    return result


def _scenario_from_report_metadata(pack_dir: Path, metadata_path: Path) -> str | None:
    try:
        rel = metadata_path.relative_to(pack_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5 or parts[0] != "reports":
        return None
    if parts[2] == "errors":
        return parts[3] if len(parts) > 3 else None
    return parts[2]


def _first_metadata_by_scenario(pack_dir: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(pack_dir.glob("reports/**/edit_metadata.json")):
        scenario_id = _scenario_from_report_metadata(pack_dir, metadata_path)
        if not scenario_id or scenario_id in observed:
            continue
        try:
            observed[scenario_id] = read_edit_metadata(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return observed


def _first_deployable_validation_by_scenario(
    pack_dir: Path,
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    runtime_paths = sorted(
        pack_dir.glob("reports/**/runtime_deployability_validation.json")
    )
    # The generator's structural receipt binds its sidecars, but cannot stand in
    # for an independent reload/inference proof in public summary output.
    for validation_path in runtime_paths:
        scenario_id = _scenario_from_report_metadata(pack_dir, validation_path)
        if not scenario_id or scenario_id in observed:
            continue
        payload = _load_json_object(validation_path)
        if (
            payload.get("validation_scope") == "runtime_reproof"
            and payload.get("runtime_proof_authoritative") is True
            and isinstance(payload.get("runtime_proof"), dict)
        ):
            observed[scenario_id] = payload
    return observed


def _scenario_artifact_class(spec: dict[str, Any]) -> str:
    artifact_class = spec.get("artifact_class")
    if isinstance(artifact_class, str) and artifact_class:
        return artifact_class
    generation = spec.get("generation")
    kind = generation.get("kind") if isinstance(generation, dict) else ""
    if kind == "error":
        return FAULT_INJECTION_FIXTURE
    if kind == "deployable_edit":
        return DEPLOYABLE_OPTIMIZED_SUBJECT
    if kind == "edit":
        return VALIDATION_SUBJECT_CHECKPOINT
    return "unknown"


def build_edit_artifact_summary(pack_dir: Path, scenarios_path: Path) -> dict[str, Any]:
    scenarios = _scenario_index(scenarios_path)
    observed = _first_metadata_by_scenario(pack_dir)
    deployable_validation = _first_deployable_validation_by_scenario(pack_dir)
    counts: Counter[str] = Counter()
    by_scenario: dict[str, dict[str, Any]] = {}

    for scenario_id, spec in sorted(scenarios.items()):
        artifact_class = _scenario_artifact_class(spec)
        counts[artifact_class] += 1
        metadata = observed.get(scenario_id, {})
        generation = (
            spec.get("generation") if isinstance(spec.get("generation"), dict) else {}
        )
        record: dict[str, Any] = {
            "artifact_class": artifact_class,
            "category": spec.get("category"),
            "failure_class": spec.get("failure_class"),
            "generation_kind": generation.get("kind")
            if isinstance(generation, dict)
            else None,
        }
        for field in (
            "edit_type",
            "optimized_deployment_backend",
            "storage_format",
            "actual_storage_format",
            "packed_quantized_storage",
            "runtime_memory_reduction",
            "backend",
            "edit_provenance",
            "edit_impact",
            "edit_topology",
            "delta_privacy",
        ):
            if field in metadata:
                record[field] = metadata[field]
        if metadata.get("schema") == EDIT_METADATA_SCHEMA:
            record["metadata_present"] = True
        elif metadata:
            record["metadata_present"] = False
        validation = deployable_validation.get(scenario_id, {})
        if validation:
            record["deployable_validation_ok"] = validation.get("ok")
            record["runtime_proof_authoritative"] = validation.get(
                "runtime_proof_authoritative"
            )
            record["load_smoke"] = validation.get("load_smoke")
            record["inference_smoke"] = validation.get("inference_smoke")
            if "backend" not in record and validation.get("backend"):
                record["backend"] = validation.get("backend")
        by_scenario[scenario_id] = record

    lanes = {
        "validation_subjects": counts.get(VALIDATION_SUBJECT_CHECKPOINT, 0) > 0,
        "deployable_subjects": counts.get(DEPLOYABLE_OPTIMIZED_SUBJECT, 0) > 0,
        "fault_injection": counts.get(FAULT_INJECTION_FIXTURE, 0) > 0,
    }
    deployable_records = [
        record
        for record in by_scenario.values()
        if record.get("artifact_class") == DEPLOYABLE_OPTIMIZED_SUBJECT
    ]
    return {
        "schema": EDIT_ARTIFACT_SUMMARY_SCHEMA,
        "counts": dict(sorted(counts.items())),
        "evidence_lanes": lanes,
        "deployable_subjects": {
            "count": len(deployable_records),
            "backends": sorted(
                {
                    str(record.get("backend"))
                    for record in deployable_records
                    if record.get("backend")
                }
            ),
            "all_reload_smokes_passed": bool(deployable_records)
            and all(
                record.get("runtime_proof_authoritative") is True
                and record.get("load_smoke") is True
                for record in deployable_records
            ),
            "all_inference_smokes_passed": bool(deployable_records)
            and all(
                record.get("runtime_proof_authoritative") is True
                and record.get("inference_smoke") is True
                for record in deployable_records
            ),
        },
        "by_scenario": by_scenario,
    }


def _edit_artifact_summary(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_edit_artifact_summary(Path(args.pack_dir), Path(args.scenarios))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _structural_metric_section(
    source_report: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = {}
    if isinstance(source_report, dict):
        raw_metrics = source_report.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
    primary_metric = metrics.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}

    payload: dict[str, Any] = {
        "kind": primary_metric.get("kind") or "ppl_causal",
        "unit": primary_metric.get("unit") or "ppl",
        "direction": primary_metric.get("direction") or "lower",
        "aggregation_scope": primary_metric.get("aggregation_scope") or "token",
        "paired": bool(primary_metric.get("paired", True)),
        "gating_basis": primary_metric.get("gating_basis") or "upper",
        "supports_bootstrap": bool(primary_metric.get("supports_bootstrap", True)),
        "invalid": True,
        "degraded": True,
    }

    preview = primary_metric.get("preview")
    final = primary_metric.get("final")
    if preview is None:
        preview = metrics.get("ppl_preview")
    if final is None:
        final = metrics.get("ppl_final")
    if preview is not None:
        payload["preview"] = preview
    if final is not None:
        payload["final"] = final

    drift_band = primary_metric.get("drift_band")
    if isinstance(drift_band, dict):
        payload["drift_band"] = drift_band

    kind = payload["kind"]
    comparison_field = (
        "delta_vs_baseline_pp" if kind == "accuracy" else "ratio_vs_baseline"
    )
    comparison_value = primary_metric.get(comparison_field)
    if isinstance(comparison_value, int | float) and not isinstance(
        comparison_value, bool
    ):
        payload[comparison_field] = float(comparison_value)

    return payload


def _build_structural_base_report(
    source_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(source_report, dict):
        raise ValueError("source report is required to build a structural failure cert")

    meta = source_report.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    data = source_report.get("data")
    if not isinstance(data, dict):
        data = {}

    edit = source_report.get("edit")
    if not isinstance(edit, dict):
        edit = {}

    artifacts = source_report.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}

    evaluation_windows = source_report.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        evaluation_windows = {}

    def _non_negative_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed >= 0 else 0

    preview_windows = _non_negative_int(data.get("preview_n"))
    final_windows = _non_negative_int(data.get("final_n"))

    return {
        "schema_version": "v1",
        "run_id": str(source_report.get("run_id") or "run"),
        "edit_name": str(edit.get("name") or "unknown"),
        "plugins": {
            "adapters": [],
            "edits": [],
            "guards": [],
        },
        "meta": copy.deepcopy(meta),
        "dataset": {
            "provider": str(data.get("dataset") or "unknown"),
            "seq_len": _non_negative_int(data.get("seq_len")),
            "hash": {
                "preview": "",
                "final": "",
                "dataset": None,
                "preview_tokens": None,
                "final_tokens": None,
                "total_tokens": 0,
                "source": "config_fallback",
            },
            "windows": {
                "preview": preview_windows,
                "final": final_windows,
                "seed": meta.get("seed"),
                "stats": {},
            },
        },
        "primary_metric": _structural_metric_section(source_report),
        "artifacts": copy.deepcopy(artifacts),
        "evaluation_windows": copy.deepcopy(evaluation_windows),
        "flags": copy.deepcopy(source_report.get("flags", {})),
    }


def build_structural_failure_report(
    *,
    error_type: str,
    message: str,
    base_report: dict[str, Any],
    source_report: dict[str, Any] | None,
    source_report_path: str | None,
    edited_report_path: str | None,
    edited_events_path: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_report)
    source_run_id = str(payload.get("run_id") or "run")
    payload["run_id"] = f"{source_run_id}-structural-failure-{error_type}"

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    meta["structural_failure"] = {
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    validation.update(
        {
            "invariants_pass": False,
            "primary_metric_acceptable": False,
            "spectral_stable": False,
            "rmt_stable": False,
            "preview_final_drift_acceptable": False,
            "guard_metric_impact_acceptable": False,
            "primary_metric_tail_acceptable": False,
        }
    )
    payload["validation"] = validation

    payload["guard_metric_impact"] = {
        "degradation_limit": 0.01,
        "evaluated": False,
        "passed": False,
        "checks": {},
        "diagnostics": [
            {
                "kind": "guard_metric_impact_unavailable",
                "severity": "error",
                "message": (
                    "Guard metric impact is unavailable because report generation "
                    "terminated on a structural failure"
                ),
                "details": {"error_type": error_type},
            }
        ],
        "source": "structural_failure",
        "mode": "unevaluated",
    }

    primary_metric = payload.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}
    primary_metric.update(_structural_metric_section(source_report))
    primary_metric["invalid"] = True
    primary_metric["degraded"] = True
    primary_metric["degraded_reason"] = "structural_failure"
    if primary_metric.get("kind") == "accuracy":
        primary_metric.pop("ratio_vs_baseline", None)
    else:
        primary_metric.pop("delta_vs_baseline_pp", None)
    payload["primary_metric"] = primary_metric

    payload["_evidence_pack_structural_failure"] = {
        "format": "evidence-pack-structural-failure-report-v1",
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    payload["invariants"] = {
        "status": "fail",
        "failures": [
            {
                "check": "error_injection",
                "type": "evidence_pack_structural_failure",
                "severity": "fatal",
                "detail": {
                    "error_type": error_type,
                    "message": message,
                },
            }
        ],
    }

    spectral = payload.get("spectral")
    if not isinstance(spectral, dict):
        spectral = {}
    spectral["status"] = "structural_failure"
    payload["spectral"] = spectral

    rmt = payload.get("rmt")
    if not isinstance(rmt, dict):
        rmt = {}
    rmt["status"] = "structural_failure"
    payload["rmt"] = rmt

    return payload


def _write_structural_runtime_manifest(
    *,
    out_path: Path,
    source_runtime_manifest: dict[str, Any] | None,
    error_type: str,
    message: str,
) -> None:
    if not isinstance(source_runtime_manifest, dict):
        return

    manifest_payload = copy.deepcopy(source_runtime_manifest)
    manifest_payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    manifest_payload["report"] = {
        "path": str(out_path.resolve()),
        "filename": out_path.name,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    context = manifest_payload.get("context")
    if not isinstance(context, dict):
        context = {}
    context["evidence_pack_structural_failure"] = {
        "error_type": error_type,
        "message": message,
    }
    manifest_payload["context"] = context
    manifest_path = out_path.parent / "runtime.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _structural_failure_report(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    source_report_path = Path(args.source_report) if args.source_report else None
    source_report = _load_json_optional(source_report_path)
    source_runtime_manifest_path = (
        Path(args.source_runtime_manifest) if args.source_runtime_manifest else None
    )
    source_runtime_manifest = _load_json_optional(source_runtime_manifest_path)
    base_report = _build_structural_base_report(source_report)

    payload = build_structural_failure_report(
        error_type=str(args.error_type),
        message=str(args.message),
        base_report=base_report,
        source_report=source_report,
        source_report_path=str(source_report_path) if source_report_path else None,
        edited_report_path=str(args.edited_report) if args.edited_report else None,
        edited_events_path=str(args.edited_events) if args.edited_events else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_structural_runtime_manifest(
        out_path=out_path,
        source_runtime_manifest=source_runtime_manifest,
        error_type=str(args.error_type),
        message=str(args.message),
    )
    return 0
