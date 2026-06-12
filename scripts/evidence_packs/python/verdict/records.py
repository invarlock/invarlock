from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generator_helpers import (
    SUMMARY_CATEGORIES,
    ScenarioCatalog,
    _as_bool,
    _as_float,
    _as_int,
    _classify_report,
    _core_signal_count,
    _detector_matches,
    _edit_family,
    _evaluate_report,
    _extract_run_num,
    _record_primary_guard_hit,
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
        if (
            (category == "clean" and strictness == "must_pass")
            or (category == "stress" and strictness == "must_fail")
            or (
                category == "error_injection"
                and strictness in {"must_fail", "must_detect"}
            )
        ):
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
