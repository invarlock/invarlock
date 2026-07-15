from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from invarlock.evidence_pack_contracts.probes import (
    PROBE_FILENAMES,
    ProbeValidationError,
    load_probe_snapshot,
)
from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot

try:
    from .report_branding import evidence_pack_text_header
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_branding import evidence_pack_text_header

try:
    from .verdict.generator_helpers import (
        CORE_GUARDS,
        INTERVENTION_SIGNALS,
        SUMMARY_CATEGORIES,
        _build_category_summary,
        _build_guard_intervention_summary,
        _build_guard_signal_summary,
        _build_scenario_signal_summary,
        _load_scenarios_manifest,
        _manifest_root,
        _record_signaled,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verdict.generator_helpers import (
        CORE_GUARDS,
        INTERVENTION_SIGNALS,
        SUMMARY_CATEGORIES,
        _build_category_summary,
        _build_guard_intervention_summary,
        _build_guard_signal_summary,
        _build_scenario_signal_summary,
        _load_scenarios_manifest,
        _manifest_root,
        _record_signaled,
    )


try:
    from .verdict.records import (
        _build_scenario_catalog,
        _collect_baseline_reports,
        _collect_latest_reports,
        _collect_records,
        _scenario_detectors,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    from verdict.records import (
        _build_scenario_catalog,
        _collect_baseline_reports,
        _collect_latest_reports,
        _collect_records,
        _scenario_detectors,
    )


def _scenario_records(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    category: str,
    scenario_id: str,
) -> list[dict[str, Any]]:
    return [
        by_key[(model_name, category, scenario_id)]
        for model_name in model_names
        if (model_name, category, scenario_id) in by_key
    ]


def _display_manifest_path(manifest_path: Path) -> str:
    default_manifest = _manifest_root() / "scenarios.json"
    try:
        if manifest_path.resolve() == default_manifest.resolve():
            return "scripts/evidence_packs/scenarios.json"
    except OSError:
        pass
    if manifest_path.name == "scenarios.json" and manifest_path.parent.name == "state":
        return "state/scenarios.json"
    if manifest_path.name == "scenarios.json":
        return manifest_path.name
    return manifest_path.name


def _report_run_id(payload: dict[str, Any]) -> str | None:
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    meta = payload.get("meta")
    if isinstance(meta, dict):
        run_id = meta.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
    return None


def _pack_report_path(report_path: Path, output_dir: Path) -> str | None:
    try:
        relative = report_path.relative_to(output_dir)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4 or parts[1] != "reports":
        return None
    return Path("reports", parts[0], *parts[2:]).as_posix()


def _report_bindings(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bindings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for report_path in sorted(output_dir.glob("*/reports/**/evaluation.report.json")):
        try:
            relative = report_path.relative_to(output_dir)
        except ValueError:
            continue
        if any(part.startswith(".") and ".tmp." in part for part in relative.parts):
            continue
        packed_path = _pack_report_path(report_path, output_dir)
        if packed_path is None:
            continue
        try:
            report_bytes, payload = read_json_object_snapshot(
                report_path,
                label=f"canonical report {packed_path}",
            )
        except (OSError, StrictJsonError) as exc:
            failures.append(
                {
                    "requirement": "canonical_report_integrity",
                    "message": "Canonical report must be unambiguous immutable JSON",
                    "path": packed_path,
                    "error": str(exc),
                }
            )
            continue
        binding: dict[str, Any] = {
            "path": packed_path,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        }
        probe_bindings: list[dict[str, str]] = []
        for filename in PROBE_FILENAMES:
            probe_path = report_path.parent / filename
            if not probe_path.exists():
                continue
            try:
                probe_bytes, _payload = load_probe_snapshot(probe_path)
            except (OSError, ProbeValidationError) as exc:
                failures.append(
                    {
                        "requirement": "probe_evidence_valid",
                        "message": "Verdict-driving probe must be strict canonical JSON",
                        "path": f"{Path(packed_path).parent.as_posix()}/{filename}",
                        "error": str(exc),
                    }
                )
                continue
            probe_bindings.append(
                {
                    "path": f"{Path(packed_path).parent.as_posix()}/{filename}",
                    "sha256": hashlib.sha256(probe_bytes).hexdigest(),
                }
            )
        if probe_bindings:
            binding["probe_bindings"] = probe_bindings
        run_id = _report_run_id(payload)
        if run_id is not None:
            binding["run_id"] = run_id
        bindings.append(binding)
    return bindings, failures


def _evaluate_coverage_requirements(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    expected_by_category: dict[str, set[str]],
    gating_by_category: dict[str, set[str]],
    primary_guard_required_scenarios: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failed_requirements: list[dict[str, Any]] = []
    missing: dict[str, Any] = {"by_model": {}}

    if not model_names:
        failed_requirements.append(
            {
                "requirement": "evidence_present",
                "message": "No model reports were found; verdict requires evidence.",
                "details": {"report_glob": "*/reports/**/evaluation.report.json"},
            }
        )
        return missing, failed_requirements

    for model_name in model_names:
        missing_model: dict[str, list[str]] = {
            category: [] for category in SUMMARY_CATEGORIES
        }
        for category in SUMMARY_CATEGORIES:
            for scenario_id in sorted(expected_by_category.get(category, set())):
                if (model_name, category, scenario_id) in by_key:
                    continue
                if scenario_id in gating_by_category.get(category, set()) or (
                    scenario_id in primary_guard_required_scenarios
                ):
                    missing_model[category].append(scenario_id)
                else:
                    missing_model.setdefault(f"{category}_informational", []).append(
                        scenario_id
                    )

        gating_categories = {"clean", "trained", "stress", "error_injection"}
        if any(missing_model.get(key) for key in gating_categories):
            missing["by_model"][model_name] = missing_model
            failed_requirements.append(
                {
                    "requirement": "scenario_coverage",
                    "message": "Missing required scenarios for model",
                    "model": model_name,
                    "missing": {
                        key: value
                        for key, value in missing_model.items()
                        if key in gating_categories and value
                    },
                }
            )

    return missing, failed_requirements


def _evaluate_clean_gating(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    scenario_ids: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_ids):
        failed_records = [
            record
            for record in _scenario_records(
                model_names,
                by_key,
                category="clean",
                scenario_id=scenario_id,
            )
            if not bool(record["passed"])
        ]
        if failed_records:
            failures.append(
                {
                    "requirement": "clean_all_pass",
                    "message": "Clean scenarios must PASS",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in failed_records
                    ],
                }
            )
    return failures


def _evaluate_trained_gating(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    scenario_ids: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_ids):
        failed_records = [
            record
            for record in _scenario_records(
                model_names,
                by_key,
                category="trained",
                scenario_id=scenario_id,
            )
            if not bool(record["passed"])
        ]
        if failed_records:
            failures.append(
                {
                    "requirement": "trained_all_pass",
                    "message": "Genuine training scenarios must PASS",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in failed_records
                    ],
                }
            )
    return failures


def _evaluate_stress_gating(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    scenario_ids: set[str],
    scenario_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_ids):
        present = _scenario_records(
            model_names,
            by_key,
            category="stress",
            scenario_id=scenario_id,
        )
        passed_records = [record for record in present if bool(record["passed"])]
        if passed_records:
            failures.append(
                {
                    "requirement": "stress_required_fail",
                    "message": "Required stress scenarios must FAIL",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in passed_records
                    ],
                }
            )

        spec = scenario_index.get(scenario_id, {})
        if not isinstance(spec, dict):
            spec = {}
        detectors_any, _, _ = _scenario_detectors(spec)
        if not detectors_any:
            continue

        missing_detectors = [
            record for record in present if not bool(record["detectors_hit"])
        ]
        if missing_detectors:
            failures.append(
                {
                    "requirement": "stress_expected_detectors",
                    "message": "Stress scenario missing expected detector signal",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "passed": record["passed"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in missing_detectors
                    ],
                }
            )
    return failures


def _evaluate_error_injection_gating(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    scenario_ids: set[str],
    scenario_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_ids):
        present = _scenario_records(
            model_names,
            by_key,
            category="error_injection",
            scenario_id=scenario_id,
        )
        missed = [record for record in present if not bool(record["detectors_hit"])]
        if missed:
            failures.append(
                {
                    "requirement": "error_injection_detected",
                    "message": "Error-injection scenarios must satisfy expected detector outcomes",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "passed": record["passed"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in missed
                    ],
                }
            )

        spec = scenario_index.get(scenario_id, {})
        strictness = str(spec.get("strictness") or "").strip().lower()
        if strictness != "must_pass":
            continue
        failed_records = [record for record in present if not bool(record["passed"])]
        if failed_records:
            failures.append(
                {
                    "requirement": "error_injection_required_pass",
                    "message": "Required-pass error-injection scenarios must PASS",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": record["model"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in failed_records
                    ],
                }
            )
    return failures


def _evaluate_primary_guard_requirements(
    model_names: list[str],
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    scenario_index: dict[str, dict[str, Any]],
    primary_guard_required_scenarios: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(primary_guard_required_scenarios):
        spec = scenario_index.get(scenario_id, {})
        if not isinstance(spec, dict):
            spec = {}
        category = str(spec.get("category") or "").strip().lower()
        primary_guard = str(spec.get("primary_guard") or "").strip().lower()
        if category not in SUMMARY_CATEGORIES or not primary_guard:
            continue

        present = _scenario_records(
            model_names,
            by_key,
            category=category,
            scenario_id=scenario_id,
        )
        if not present:
            continue

        if not any(bool(record.get("primary_guard_hit")) for record in present):
            failures.append(
                {
                    "requirement": "scenario_primary_guard_signal",
                    "message": "Scenario marked primary_guard_required did not trigger its declared primary guard.",
                    "scenario": scenario_id,
                    "category": category,
                    "primary_guard": primary_guard,
                    "failures": [
                        {
                            "model": record["model"],
                            "detectors_hit": record["detectors_hit"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in present
                    ],
                }
            )

        detectors_any, detectors_all, _ = _scenario_detectors(spec)
        if not detectors_any and not detectors_all:
            continue

        missed = [record for record in present if not bool(record.get("detectors_hit"))]
        if missed:
            failures.append(
                {
                    "requirement": "scenario_expected_detectors",
                    "message": "Primary-guard-required scenario missing expected detector signal constraints.",
                    "scenario": scenario_id,
                    "category": category,
                    "primary_guard": primary_guard,
                    "failures": [
                        {
                            "model": record["model"],
                            "detectors_hit": record["detectors_hit"],
                            "reasons": record["reasons"],
                            "path": record["path"],
                        }
                        for record in missed
                    ],
                }
            )
    return failures


def _evaluate_guard_demonstration(
    records: list[dict[str, Any]],
    *,
    primary_guard_required_scenarios: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required_guard_records = [
        record
        for record in records
        if record.get("name") in primary_guard_required_scenarios
    ]
    for guard in CORE_GUARDS:
        guard_records = [
            record
            for record in required_guard_records
            if str(record.get("primary_guard") or "") == guard
        ]
        if not guard_records:
            continue
        if any(bool(record.get("primary_guard_hit")) for record in guard_records):
            continue
        failures.append(
            {
                "requirement": "guard_primary_demonstrated",
                "message": "No primary_guard_required scenario produced a direct hit for guard.",
                "guard": guard,
                "scenarios": sorted(
                    {str(record.get("name") or "") for record in guard_records}
                ),
            }
        )
    return failures


def _evaluate_informational_stress(
    stress_records: list[dict[str, Any]],
    *,
    informational_stress: set[str],
    info_min_signal_fraction: float,
) -> tuple[list[dict[str, Any]], int, int, int]:
    info_stress = [
        record for record in stress_records if record["name"] in informational_stress
    ]
    info_fail = sum(1 for record in info_stress if not record["passed"])
    info_signaled = sum(1 for record in info_stress if _record_signaled(record))
    info_total = len(info_stress)
    failures: list[dict[str, Any]] = []

    if info_total > 0:
        info_signal_fraction = info_signaled / info_total
        if info_signal_fraction < info_min_signal_fraction:
            failures.append(
                {
                    "requirement": "informational_stress_min_signal_fraction",
                    "message": "Informational stress signal fraction below required minimum.",
                    "details": {
                        "required_min": info_min_signal_fraction,
                        "observed": info_signal_fraction,
                        "signaled_count": info_signaled,
                        "total_count": info_total,
                    },
                }
            )

    return failures, info_total, info_fail, info_signaled


def _build_counts(
    *,
    model_names: list[str],
    records: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    stress: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    catastrophic_required: set[str],
    catastrophic_records: list[dict[str, Any]],
    info_total: int,
    info_fail: int,
    info_signaled: int,
    primary_guard_required_scenarios: set[str],
) -> dict[str, Any]:
    primary_guard_required_records = [
        record
        for record in records
        if record["name"] in primary_guard_required_scenarios
    ]
    primary_guard_required_hits = sum(
        1
        for record in primary_guard_required_records
        if bool(record.get("primary_guard_hit"))
    )
    return {
        "models_total": len(model_names),
        "records_total": len(records),
        "clean_total": len(clean),
        "clean_pass": sum(1 for record in clean if record["passed"]),
        "trained_total": len(trained),
        "trained_pass": sum(1 for record in trained if record["passed"]),
        "stress_total": len(stress),
        "stress_fail": sum(1 for record in stress if not record["passed"]),
        "catastrophic_required_total": len(catastrophic_required),
        "catastrophic_required_present": len(
            {record["name"] for record in catastrophic_records}
        ),
        "catastrophic_required_fail": sum(
            1 for record in catastrophic_records if not record["passed"]
        ),
        "error_injection_total": len(errors),
        "error_injection_detected": sum(
            1 for record in errors if record["detectors_hit"]
        ),
        "informational_stress_total": info_total,
        "informational_stress_fail": info_fail,
        "informational_stress_signaled": info_signaled,
        "primary_guard_required_scenarios": len(primary_guard_required_scenarios),
        "primary_guard_required_records": len(primary_guard_required_records),
        "primary_guard_required_hits": primary_guard_required_hits,
        "primary_guard_required_scenarios_hit": len(
            {
                str(record.get("name") or "")
                for record in primary_guard_required_records
                if bool(record.get("primary_guard_hit"))
            }
        ),
    }


def generate_verdict(
    *, output_dir: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = _manifest_root() / "scenarios.json"
    manifest = _load_scenarios_manifest(manifest_path)
    catalog = _build_scenario_catalog(manifest)
    latest = _collect_latest_reports(
        output_dir,
        scenario_index=catalog.scenario_index,
    )
    baseline_reports = _collect_baseline_reports(output_dir)
    records, model_names = _collect_records(
        latest,
        scenario_index=catalog.scenario_index,
        baseline_reports=baseline_reports,
        output_dir=output_dir,
    )
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (record["model"], record["category"], record["name"]): record
        for record in records
    }
    info_min_signal_fraction = 0.5
    missing, failed_requirements = _evaluate_coverage_requirements(
        model_names,
        by_key,
        expected_by_category=catalog.expected_by_category,
        gating_by_category=catalog.gating_by_category,
        primary_guard_required_scenarios=catalog.primary_guard_required_scenarios,
    )
    failed_requirements.extend(
        {
            "requirement": "probe_evidence_valid",
            "message": "Verdict-driving probe evidence must satisfy its exact contract",
            "model": record["model"],
            "scenario": record["name"],
            "errors": record["probe_validation_errors"],
        }
        for record in records
        if record.get("probe_validation_errors")
    )
    failed_requirements.extend(
        _evaluate_clean_gating(
            model_names,
            by_key,
            catalog.gating_by_category.get("clean", set()),
        )
    )
    failed_requirements.extend(
        _evaluate_trained_gating(
            model_names,
            by_key,
            catalog.gating_by_category.get("trained", set()),
        )
    )
    failed_requirements.extend(
        _evaluate_stress_gating(
            model_names,
            by_key,
            scenario_ids=catalog.gating_by_category.get("stress", set()),
            scenario_index=catalog.scenario_index,
        )
    )
    failed_requirements.extend(
        _evaluate_error_injection_gating(
            model_names,
            by_key,
            scenario_ids=catalog.gating_by_category.get("error_injection", set()),
            scenario_index=catalog.scenario_index,
        )
    )
    failed_requirements.extend(
        _evaluate_primary_guard_requirements(
            model_names,
            by_key,
            scenario_index=catalog.scenario_index,
            primary_guard_required_scenarios=catalog.primary_guard_required_scenarios,
        )
    )
    failed_requirements.extend(
        _evaluate_guard_demonstration(
            records,
            primary_guard_required_scenarios=catalog.primary_guard_required_scenarios,
        )
    )

    clean = [record for record in records if record["category"] == "clean"]
    trained = [record for record in records if record["category"] == "trained"]
    stress = [record for record in records if record["category"] == "stress"]
    errors = [record for record in records if record["category"] == "error_injection"]
    info_failures, info_total, info_fail, info_signaled = (
        _evaluate_informational_stress(
            stress,
            informational_stress=catalog.informational_stress,
            info_min_signal_fraction=info_min_signal_fraction,
        )
    )
    failed_requirements.extend(info_failures)

    catastrophic_records = [
        record for record in stress if record["name"] in catalog.catastrophic_required
    ]
    guard_signal_summary = _build_guard_signal_summary(records)
    guard_intervention_summary = _build_guard_intervention_summary(records)
    category_summary = _build_category_summary(
        records,
        expected_by_category=catalog.expected_by_category,
    )
    scenario_signal_summary = _build_scenario_signal_summary(
        records,
        scenario_index=catalog.scenario_index,
    )

    report_bindings, report_binding_failures = _report_bindings(output_dir)
    failed_requirements.extend(report_binding_failures)
    verdict = "PASS" if not failed_requirements else "FAIL"

    return {
        "verdict": verdict,
        "manifest": {
            "path": _display_manifest_path(manifest_path),
            "schema": manifest.get("schema"),
            "schema_version": manifest.get("schema_version"),
        },
        "criteria": {
            "clean_all_pass": True,
            "trained_all_pass": True,
            "stress_required_fail": True,
            "error_injection_detected": True,
            "informational_stress_min_signal_fraction": info_min_signal_fraction,
            "primary_guard_signal_required": True,
        },
        "counts": _build_counts(
            model_names=model_names,
            records=records,
            clean=clean,
            trained=trained,
            stress=stress,
            errors=errors,
            catastrophic_required=catalog.catastrophic_required,
            catastrophic_records=catastrophic_records,
            info_total=info_total,
            info_fail=info_fail,
            info_signaled=info_signaled,
            primary_guard_required_scenarios=catalog.primary_guard_required_scenarios,
        ),
        "core_guard_order": list(CORE_GUARDS),
        "category_summary": category_summary,
        "guard_signal_summary": guard_signal_summary,
        "guard_intervention_summary": guard_intervention_summary,
        "scenario_signal_summary": scenario_signal_summary,
        "report_bindings": report_bindings,
        "records": records,
        "missing": missing,
        "failed_requirements": failed_requirements,
        "timestamp": datetime.now().isoformat(),
    }


def _render_text(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    missing = payload.get("missing") or {}
    failed = payload.get("failed_requirements") or []
    manifest = payload.get("manifest") or {}

    category_summary = payload.get("category_summary") or {}
    guard_signal_summary = payload.get("guard_signal_summary") or {}
    guard_intervention_summary = payload.get("guard_intervention_summary") or {}

    lines = [
        *evidence_pack_text_header("Final Verdict"),
        f"Verdict: {payload.get('verdict')}",
        f"Scenarios manifest: {manifest.get('path')}",
        "",
        "COUNTS:",
        f"  Clean: {counts.get('clean_pass')}/{counts.get('clean_total')} PASS",
        f"  Trained: {counts.get('trained_pass')}/{counts.get('trained_total')} PASS",
        f"  Stress: {counts.get('stress_fail')}/{counts.get('stress_total')} FAIL (expected for stress probes)",
        (
            "  Catastrophic-required stress: "
            f"{counts.get('catastrophic_required_fail')}/{counts.get('catastrophic_required_total')} FAIL"
        ),
        (
            "  Error injection detected: "
            f"{counts.get('error_injection_detected')}/{counts.get('error_injection_total')}"
        ),
        (
            "  Informational stress signaled: "
            f"{counts.get('informational_stress_signaled')}/{counts.get('informational_stress_total')}"
        ),
        (
            "  Primary-guard-required hits: "
            f"{counts.get('primary_guard_required_hits')}/{counts.get('primary_guard_required_records')}"
        ),
        "",
    ]

    if isinstance(category_summary, dict) and category_summary:
        lines.append("CATEGORY SUMMARY:")
        for category in SUMMARY_CATEGORIES:
            row = category_summary.get(category)
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  {category}: scenarios={row.get('scenarios')} reports={row.get('reports')} "
                f"pm_fail={row.get('primary_metric_fail')} "
                f"inv_fail/warn={row.get('invariants_fail')}/{row.get('invariants_warn')} "
                f"any_flag={row.get('any_flag')}"
            )
        lines.append("")

    if isinstance(guard_signal_summary, dict):
        signals = guard_signal_summary.get("signals")
        if isinstance(signals, dict) and signals:
            lines.append("GUARD SIGNALS (flagged / unique):")
            for guard in CORE_GUARDS:
                row = signals.get(guard)
                if not isinstance(row, dict):
                    continue
                lines.append(f"  {guard}: {row.get('flagged')}/{row.get('unique')}")
            lines.append("")

    if isinstance(guard_intervention_summary, dict):
        signals = guard_intervention_summary.get("signals")
        if isinstance(signals, dict) and signals:
            lines.append("GUARD INTERVENTIONS (flagged / unique):")
            for signal in INTERVENTION_SIGNALS:
                row = signals.get(signal)
                if not isinstance(row, dict):
                    continue
                lines.append(f"  {signal}: {row.get('flagged')}/{row.get('unique')}")
            lines.append("")

    missing_by_model = missing.get("by_model")
    if isinstance(missing_by_model, dict) and missing_by_model:
        lines.append("MISSING (required):")
        for model_name in sorted(missing_by_model):
            mm = missing_by_model.get(model_name)
            if not isinstance(mm, dict):
                continue
            clean_missing = mm.get("clean", [])
            stress_missing = mm.get("stress", [])
            error_missing = mm.get("error_injection", [])
            if clean_missing:
                lines.append(f"  {model_name}: clean: {', '.join(clean_missing)}")
            if stress_missing:
                lines.append(f"  {model_name}: stress: {', '.join(stress_missing)}")
            if error_missing:
                lines.append(
                    f"  {model_name}: error_injection: {', '.join(error_missing)}"
                )
        lines.append("")

    if failed:
        lines.append("FAILED REQUIREMENTS:")
        for item in failed:
            lines.append(f"  - {item.get('requirement')}: {item.get('message')}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional scenarios manifest JSON (defaults to scripts/evidence_packs/scenarios.json).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = generate_verdict(output_dir=output_dir, manifest_path=args.manifest)
    _write_json(reports_dir / "final_verdict.json", payload)
    (reports_dir / "final_verdict.txt").write_text(
        _render_text(payload),
        encoding="utf-8",
    )

    _write_json(
        reports_dir / "guard_signal_summary.json",
        payload.get("guard_signal_summary") or {},
    )
    _write_json(
        reports_dir / "guard_intervention_summary.json",
        payload.get("guard_intervention_summary") or {},
    )
    _write_json(
        reports_dir / "category_summary.json",
        payload.get("category_summary") or {},
    )
    _write_json(
        reports_dir / "scenario_signal_summary.json",
        payload.get("scenario_signal_summary") or [],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
