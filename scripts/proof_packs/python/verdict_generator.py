from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from verdict_generator_helpers import (
        CORE_GUARDS,
        INTERVENTION_SIGNALS,
        SUMMARY_CATEGORIES,
        ScenarioCatalog,
        _as_bool,
        _as_float,
        _as_int,
        _build_category_summary,
        _build_guard_intervention_summary,
        _build_guard_signal_summary,
        _build_scenario_signal_summary,
        _classify_report,
        _core_signal_count,
        _detector_matches,
        _edit_family,
        _evaluate_report,
        _extract_run_num,
        _load_scenarios_manifest,
        _manifest_root,
        _record_primary_guard_hit,
        _record_signaled,
        _spectral_caps_applied,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verdict_generator_helpers import (
        CORE_GUARDS,
        INTERVENTION_SIGNALS,
        SUMMARY_CATEGORIES,
        ScenarioCatalog,
        _as_bool,
        _as_float,
        _as_int,
        _build_category_summary,
        _build_guard_intervention_summary,
        _build_guard_signal_summary,
        _build_scenario_signal_summary,
        _classify_report,
        _core_signal_count,
        _detector_matches,
        _edit_family,
        _evaluate_report,
        _extract_run_num,
        _load_scenarios_manifest,
        _manifest_root,
        _record_primary_guard_hit,
        _record_signaled,
        _spectral_caps_applied,
    )


def _build_scenario_catalog(manifest: dict[str, Any]) -> ScenarioCatalog:
    scenario_index: dict[str, dict[str, Any]] = {}
    scenarios = manifest.get("scenarios", [])
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        scenario_id = item.get("id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            continue
        scenario_index[scenario_id] = item

    expected_by_category: dict[str, set[str]] = {
        "clean": set(),
        "stress": set(),
        "error_injection": set(),
    }
    gating_by_category: dict[str, set[str]] = {
        "clean": set(),
        "stress": set(),
        "error_injection": set(),
    }
    catastrophic_required: set[str] = set()
    informational_stress: set[str] = set()
    primary_guard_required_scenarios: set[str] = set()

    for scenario_id, spec in scenario_index.items():
        category = str(spec.get("category") or "").strip().lower()
        strictness = str(spec.get("strictness") or "").strip().lower()
        if category not in expected_by_category:
            continue
        expected_by_category[category].add(scenario_id)
        if strictness in {"must_pass", "must_fail", "must_detect"}:
            gating_by_category[category].add(scenario_id)
        if category == "stress" and strictness == "informational":
            informational_stress.add(scenario_id)
        reqs = spec.get("requirements")
        if isinstance(reqs, dict) and reqs.get("catastrophic_required") is True:
            catastrophic_required.add(scenario_id)
        if isinstance(reqs, dict) and reqs.get("primary_guard_required") is True:
            primary_guard_required_scenarios.add(scenario_id)

    return ScenarioCatalog(
        scenario_index=scenario_index,
        expected_by_category=expected_by_category,
        gating_by_category=gating_by_category,
        catastrophic_required=catastrophic_required,
        informational_stress=informational_stress,
        primary_guard_required_scenarios=primary_guard_required_scenarios,
    )


def _collect_latest_reports(
    output_dir: Path,
) -> dict[tuple[str, str, str], tuple[int, Path]]:
    latest: dict[tuple[str, str, str], tuple[int, Path]] = {}
    for cert_path in sorted(output_dir.glob("*/reports/**/evaluation.report.json")):
        cls = _classify_report(cert_path, output_dir=output_dir)
        if cls is None:
            continue
        model_name, category, scenario_id = cls
        if category not in SUMMARY_CATEGORIES:
            continue
        run_num = _extract_run_num(cert_path, output_dir=output_dir)
        key = (model_name, category, scenario_id)
        prev = latest.get(key)
        if prev is None or run_num >= prev[0]:
            latest[key] = (run_num, cert_path)
    return latest


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_report_with_probes(cert_path: Path) -> dict[str, Any] | None:
    cert = _load_json_object(cert_path)
    if cert is None:
        return None

    for probe_name in ("rmt_probe", "ve_probe"):
        probe_payload = _load_json_object(cert_path.parent / f"{probe_name}.json")
        if probe_payload is not None:
            cert[probe_name] = probe_payload
    return cert


def _scenario_detectors(
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    reqs = spec.get("requirements") if isinstance(spec, dict) else None
    detectors_any: list[dict[str, Any]] = []
    if isinstance(reqs, dict) and isinstance(reqs.get("detectors_any_of"), list):
        detectors_any = [d for d in reqs.get("detectors_any_of") if isinstance(d, dict)]

    detectors_all: list[dict[str, Any]] = []
    if isinstance(reqs, dict) and isinstance(reqs.get("detectors_all_of"), list):
        detectors_all = [d for d in reqs.get("detectors_all_of") if isinstance(d, dict)]

    primary_guard_required = bool(
        isinstance(reqs, dict) and reqs.get("primary_guard_required") is True
    )
    return detectors_any, detectors_all, primary_guard_required


def _detectors_hit(
    cert: dict[str, Any],
    detectors_any: list[dict[str, Any]],
    detectors_all: list[dict[str, Any]],
) -> bool:
    if not detectors_any and not detectors_all:
        return False

    matched = True
    if detectors_any:
        matched = any(_detector_matches(cert, detector) for detector in detectors_any)
    if detectors_all:
        matched = matched and all(
            _detector_matches(cert, detector) for detector in detectors_all
        )
    return matched


def _apply_probe_guard_overrides(
    record: dict[str, Any],
    cert: dict[str, Any],
) -> None:
    rmt_probe = cert.get("rmt_probe")
    if isinstance(rmt_probe, dict):
        record["rmt_probe"] = rmt_probe
        if _as_bool(rmt_probe.get("stable"), default=True) is False:
            record["guard_flags"]["rmt"] = True

    ve_probe = cert.get("ve_probe")
    if isinstance(ve_probe, dict):
        record["ve_probe"] = ve_probe
        if _as_bool(ve_probe.get("signal"), default=False):
            record["guard_flags"]["variance"] = True
        if _as_bool(ve_probe.get("would_enable"), default=False):
            record["guard_flags"]["variance"] = True
        if _as_int(ve_probe.get("proposed_scales"), default=0) > 0:
            record["guard_flags"]["variance"] = True
        gain = _as_float(ve_probe.get("ab_gain"), default=None)
        if gain is not None and gain > 0.0:
            record["guard_flags"]["variance"] = True


def _build_record(
    *,
    cert: dict[str, Any],
    cert_path: Path,
    model_name: str,
    category: str,
    scenario_id: str,
    run_num: int,
    scenario_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    outcome = _evaluate_report(cert)
    spec = scenario_index.get(scenario_id, {})
    if not isinstance(spec, dict):
        spec = {}
    detectors_any, detectors_all, primary_guard_required = _scenario_detectors(spec)

    record: dict[str, Any] = {
        "model": model_name,
        "category": category,
        "name": scenario_id,
        "strictness": str(spec.get("strictness") or ""),
        "intent": str(spec.get("intent") or ""),
        "primary_guard": str(spec.get("primary_guard") or ""),
        "failure_class": str(spec.get("failure_class") or ""),
        "run_num": run_num,
        "family": _edit_family(scenario_id) if category in {"clean", "stress"} else "",
        "passed": outcome.passed,
        "reasons": list(outcome.reasons),
        "detectors_hit": _detectors_hit(cert, detectors_any, detectors_all),
        "detectors": detectors_any,
        "detectors_all_of": detectors_all,
        "primary_guard_required": primary_guard_required,
        "invariants_status": outcome.invariants_status,
        "guard_flags": outcome.guard_flags,
        "spectral_caps_applied": _spectral_caps_applied(cert),
        "path": str(cert_path),
    }
    _apply_probe_guard_overrides(record, cert)
    record["primary_guard_hit"] = _record_primary_guard_hit(record)
    record["any_core_guard_flag"] = _core_signal_count(record) > 0
    return record


def _collect_records(
    latest: dict[tuple[str, str, str], tuple[int, Path]],
    *,
    scenario_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    models: set[str] = set()
    for (model_name, category, scenario_id), (run_num, cert_path) in sorted(
        latest.items()
    ):
        cert = _load_report_with_probes(cert_path)
        if cert is None:
            continue
        models.add(model_name)
        records.append(
            _build_record(
                cert=cert,
                cert_path=cert_path,
                model_name=model_name,
                category=category,
                scenario_id=scenario_id,
                run_num=run_num,
                scenario_index=scenario_index,
            )
        )
    return records, sorted(models)


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
            "clean": [],
            "stress": [],
            "error_injection": [],
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

        if any(missing_model.get(key) for key in SUMMARY_CATEGORIES):
            missing["by_model"][model_name] = missing_model
            failed_requirements.append(
                {
                    "requirement": "scenario_coverage",
                    "message": "Missing required scenarios for model",
                    "model": model_name,
                    "missing": {
                        key: value
                        for key, value in missing_model.items()
                        if key in {"clean", "stress", "error_injection"} and value
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
    scenario_ids: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_ids):
        missed = [
            record
            for record in _scenario_records(
                model_names,
                by_key,
                category="error_injection",
                scenario_id=scenario_id,
            )
            if not bool(record["detectors_hit"])
        ]
        if missed:
            failures.append(
                {
                    "requirement": "error_injection_detected",
                    "message": "Error injections must trigger expected detector signals",
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
    latest = _collect_latest_reports(output_dir)
    records, model_names = _collect_records(
        latest,
        scenario_index=catalog.scenario_index,
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
        _evaluate_clean_gating(
            model_names,
            by_key,
            catalog.gating_by_category.get("clean", set()),
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
            catalog.gating_by_category.get("error_injection", set()),
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

    verdict = "PASS" if not failed_requirements else "FAIL"

    return {
        "verdict": verdict,
        "manifest": {
            "path": str(manifest_path),
            "schema": manifest.get("schema"),
            "schema_version": manifest.get("schema_version"),
        },
        "criteria": {
            "clean_all_pass": True,
            "stress_required_fail": True,
            "error_injection_detected": True,
            "informational_stress_min_signal_fraction": info_min_signal_fraction,
            "primary_guard_signal_required": True,
        },
        "counts": _build_counts(
            model_names=model_names,
            records=records,
            clean=clean,
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
        "INVARLOCK PROOF PACK (ASSURANCE) — FINAL VERDICT",
        f"Verdict: {payload.get('verdict')}",
        f"Scenarios manifest: {manifest.get('path')}",
        "",
        "COUNTS:",
        f"  Clean: {counts.get('clean_pass')}/{counts.get('clean_total')} PASS",
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
        help="Optional scenarios manifest JSON (defaults to scripts/proof_packs/scenarios.json).",
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
