from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .editing.metadata import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.metadata import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
    )

SUMMARY_SCHEMA = "invarlock/evidence-pack-edit-artifact-summary-v1"


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
    for validation_path in sorted(
        pack_dir.glob("reports/**/deployable_artifact_validation.json")
    ):
        scenario_id = _scenario_from_report_metadata(pack_dir, validation_path)
        if not scenario_id or scenario_id in observed:
            continue
        payload = _load_json_object(validation_path)
        if payload:
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
        "schema": SUMMARY_SCHEMA,
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
            and all(record.get("load_smoke") is True for record in deployable_records),
            "all_inference_smokes_passed": bool(deployable_records)
            and all(
                record.get("inference_smoke") is True for record in deployable_records
            ),
        },
        "by_scenario": by_scenario,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write edit artifact class summary.")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_edit_artifact_summary(Path(args.pack_dir), Path(args.scenarios))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
