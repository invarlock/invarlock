from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CORE_GUARDS: tuple[str, ...] = (
    "invariants",
    "spectral",
    "rmt",
    "primary_metric",
)
SUMMARY_CATEGORIES: tuple[str, ...] = ("clean", "stress", "error_injection")


def _manifest_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_scenarios_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read scenarios manifest: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Scenarios manifest must be a JSON object: {path}")
    if payload.get("schema") != "proof_pack_scenarios_v1":
        raise ValueError(f"Unknown scenarios manifest schema: {payload.get('schema')}")
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise ValueError(
            f"Unsupported scenarios manifest version: {payload.get('schema_version')}"
        )
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Scenarios manifest missing scenarios list: {path}")
    return payload


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pass"}
    return bool(value)


def _as_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return default
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return default
    return default


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        v = float(value)
        return v if v == v and abs(v) != float("inf") else default
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except Exception:
            return default
        return v if v == v and abs(v) != float("inf") else default
    return default


def _spectral_caps_applied(cert: dict[str, Any]) -> int:
    spectral = cert.get("spectral")
    if not isinstance(spectral, dict):
        return 0
    return max(0, _as_int(spectral.get("caps_applied"), default=0))


@dataclass(frozen=True)
class ValidationSnapshot:
    invariants_ok: bool
    invariants_status: str
    pm_ok: bool
    pm_degraded: bool
    spectral_ok: bool
    rmt_ok: bool
    drift_ok: bool
    overhead_evaluated: bool
    overhead_ok: bool


@dataclass(frozen=True)
class CertOutcome:
    passed: bool
    reasons: tuple[str, ...]
    guard_flags: dict[str, bool]
    invariants_status: str


# Guard projection is centralized so summaries and verdict gating stay consistent.
def _validation_snapshot(cert: dict[str, Any]) -> ValidationSnapshot:
    validation = cert.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}

    invariants_ok = _as_bool(validation.get("invariants_pass"), default=False)
    pm_ok = _as_bool(validation.get("primary_metric_acceptable"), default=False)
    spectral_ok = _as_bool(validation.get("spectral_stable"), default=False)
    rmt_ok = _as_bool(validation.get("rmt_stable"), default=False)
    drift_ok = _as_bool(validation.get("preview_final_drift_acceptable"), default=True)

    guard_overhead = cert.get("guard_overhead") or {}
    overhead_evaluated = False
    if isinstance(guard_overhead, dict):
        overhead_evaluated = _as_bool(guard_overhead.get("evaluated"), default=False)
    overhead_ok = _as_bool(validation.get("guard_overhead_acceptable"), default=True)

    primary_metric = cert.get("primary_metric") or {}
    pm_degraded = False
    if isinstance(primary_metric, dict):
        pm_degraded = _as_bool(
            primary_metric.get("degraded"), default=False
        ) or _as_bool(primary_metric.get("invalid"), default=False)

    invariants = cert.get("invariants") or {}
    invariants_status = "unknown"
    if isinstance(invariants, dict):
        raw_status = invariants.get("status")
        if isinstance(raw_status, str) and raw_status.strip():
            invariants_status = raw_status.strip().lower()

    return ValidationSnapshot(
        invariants_ok=invariants_ok,
        invariants_status=invariants_status,
        pm_ok=pm_ok,
        pm_degraded=pm_degraded,
        spectral_ok=spectral_ok,
        rmt_ok=rmt_ok,
        drift_ok=drift_ok,
        overhead_evaluated=overhead_evaluated,
        overhead_ok=overhead_ok,
    )


def _guard_flags(snapshot: ValidationSnapshot) -> dict[str, bool]:
    invariants_fail = (not snapshot.invariants_ok) or snapshot.invariants_status in {
        "fail",
        "error",
    }
    invariants_warn = snapshot.invariants_status == "warn"
    return {
        "invariants": invariants_fail or invariants_warn,
        "invariants_fail": invariants_fail,
        "invariants_warn": invariants_warn,
        "primary_metric": (not snapshot.pm_ok) or snapshot.pm_degraded,
        "spectral": not snapshot.spectral_ok,
        "rmt": not snapshot.rmt_ok,
        "drift": not snapshot.drift_ok,
        "overhead": snapshot.overhead_evaluated and (not snapshot.overhead_ok),
    }


def _detector_matches(cert: dict[str, Any], detector: dict[str, Any]) -> bool:
    kind = str(detector.get("kind") or "").strip().lower()
    if kind == "validation_flag":
        flag = detector.get("flag")
        expected = detector.get("expected")
        if not isinstance(flag, str) or expected is None:
            return False
        validation = cert.get("validation")
        if not isinstance(validation, dict):
            return False
        if flag not in validation:
            return False
        return _as_bool(validation.get(flag), default=False) == bool(expected)

    if kind == "primary_metric":
        field = detector.get("field")
        expected = detector.get("expected")
        if not isinstance(field, str) or expected is None:
            return False
        primary_metric = cert.get("primary_metric")
        if not isinstance(primary_metric, dict):
            return False
        if field not in primary_metric:
            return False
        return _as_bool(primary_metric.get(field), default=False) == bool(expected)

    if kind == "invariants_status":
        allowed = detector.get("allowed")
        if not isinstance(allowed, list | tuple | set):
            return False
        allowed_norm = {str(item).strip().lower() for item in allowed if item}
        if not allowed_norm:
            return False
        invariants = cert.get("invariants")
        if not isinstance(invariants, dict):
            return False
        status = invariants.get("status")
        if not isinstance(status, str):
            return False
        return status.strip().lower() in allowed_norm

    if kind == "rmt_probe":
        field = detector.get("field")
        expected = detector.get("expected")
        if not isinstance(field, str) or expected is None:
            return False
        probe = cert.get("rmt_probe")
        if not isinstance(probe, dict):
            return False
        if field not in probe:
            return False
        return _as_bool(probe.get(field), default=False) == bool(expected)

    if kind == "spectral_caps_applied":
        min_caps = detector.get("min")
        if min_caps is None:
            return False
        try:
            min_val = int(min_caps)
        except Exception:
            return False
        if min_val < 0:
            min_val = 0
        return _spectral_caps_applied(cert) >= min_val

    if kind == "ve_probe":
        field = detector.get("field")
        expected = detector.get("expected")
        min_value = detector.get("min")
        if not isinstance(field, str) or not field.strip():
            return False
        probe = cert.get("ve_probe")
        if not isinstance(probe, dict):
            return False
        if field not in probe:
            return False
        if expected is not None:
            return _as_bool(probe.get(field), default=False) == bool(expected)
        if min_value is not None:
            min_val = _as_float(min_value, default=None)
            if min_val is None:
                return False
            actual = _as_float(probe.get(field), default=None)
            if actual is None:
                return False
            return actual >= min_val
        return False

    return False


def _evaluate_report(cert: dict[str, Any]) -> CertOutcome:
    snapshot = _validation_snapshot(cert)
    passed = (
        snapshot.invariants_ok
        and snapshot.pm_ok
        and snapshot.spectral_ok
        and snapshot.rmt_ok
        and snapshot.drift_ok
        and (snapshot.overhead_ok if snapshot.overhead_evaluated else True)
        and not snapshot.pm_degraded
    )

    reasons: list[str] = []
    if snapshot.pm_degraded:
        reasons.append("primary_metric_degraded")
    if not snapshot.invariants_ok:
        reasons.append("invariants_fail")
    if not snapshot.pm_ok:
        reasons.append("primary_metric_fail")
    if not snapshot.spectral_ok:
        reasons.append("spectral_fail")
    if not snapshot.rmt_ok:
        reasons.append("rmt_fail")
    if not snapshot.drift_ok:
        reasons.append("drift_fail")
    if snapshot.overhead_evaluated and not snapshot.overhead_ok:
        reasons.append("overhead_fail")

    return CertOutcome(
        passed=passed,
        reasons=tuple(reasons),
        guard_flags=_guard_flags(snapshot),
        invariants_status=snapshot.invariants_status,
    )


def _edit_family(name: str) -> str:
    n = (name or "").strip().lower()
    if n.startswith("quant_"):
        return "quant"
    if n.startswith("fp8_"):
        return "fp8"
    if n.startswith("prune_"):
        return "prune"
    if n.startswith("svd_"):
        return "svd"
    return "other"


def _classify_report(
    cert_path: Path, *, output_dir: Path
) -> tuple[str, str, str] | None:
    try:
        rel = cert_path.relative_to(output_dir)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 4:
        return None

    model_name = parts[0]
    try:
        idx = parts.index("reports")
    except ValueError:
        return None

    remainder = parts[idx + 1 :]
    if not remainder:
        return None

    head = remainder[0]
    if head == "calibration":
        return model_name, "calibration", head
    if head == "errors":
        error_type = remainder[1] if len(remainder) > 1 else "unknown"
        return model_name, "error_injection", error_type

    edit_name = head
    if edit_name.endswith("_clean"):
        return model_name, "clean", edit_name
    if edit_name.endswith("_stress"):
        return model_name, "stress", edit_name
    return model_name, "other", edit_name


def _extract_run_num(cert_path: Path, *, output_dir: Path) -> int:
    try:
        rel = cert_path.relative_to(output_dir)
    except ValueError:
        return 0
    parts = rel.parts
    try:
        idx = parts.index("reports")
    except ValueError:
        return 0
    remainder = parts[idx + 1 :]
    if not remainder:
        return 0
    if remainder[0] == "errors":
        return 0
    if len(remainder) >= 3:
        run_part = remainder[1]
        if isinstance(run_part, str) and run_part.startswith("run_"):
            try:
                return int(run_part.split("_", 1)[1])
            except Exception:
                return 0
    return 0


def _core_signal_count(record: dict[str, Any]) -> int:
    flags = record.get("guard_flags")
    if not isinstance(flags, dict):
        return 0
    return sum(1 for guard in CORE_GUARDS if bool(flags.get(guard)))


def _record_signaled(record: dict[str, Any]) -> bool:
    if bool(record.get("detectors_hit")):
        return True
    return _core_signal_count(record) > 0


def _record_primary_guard_hit(record: dict[str, Any]) -> bool:
    primary_guard = str(record.get("primary_guard") or "").strip().lower()
    if not primary_guard:
        return False
    flags = record.get("guard_flags")
    if not isinstance(flags, dict):
        flags = {}
    if bool(flags.get(primary_guard)):
        return True

    if primary_guard == "rmt":
        probe = record.get("rmt_probe")
        if isinstance(probe, dict):
            stable = probe.get("stable")
            if stable is not None and _as_bool(stable, default=True) is False:
                return True
    if primary_guard == "spectral":
        if int(record.get("spectral_caps_applied") or 0) > 0:
            return True
    if primary_guard == "variance":
        probe = record.get("ve_probe")
        if isinstance(probe, dict):
            signal = probe.get("signal")
            if signal is not None and _as_bool(signal, default=False) is True:
                return True
            would_enable = probe.get("would_enable")
            if would_enable is not None and _as_bool(would_enable, default=False) is True:
                return True
            scales = _as_int(probe.get("proposed_scales"), default=0)
            gain = _as_float(probe.get("ab_gain"), default=None)
            if scales > 0 and gain is not None and gain > 0.0:
                return True
    return False


def _build_guard_signal_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    signals: dict[str, dict[str, int]] = {}
    for guard in CORE_GUARDS:
        flagged = 0
        unique = 0
        for record in records:
            flags = record.get("guard_flags")
            if not isinstance(flags, dict) or not bool(flags.get(guard)):
                continue
            flagged += 1
            if _core_signal_count(record) == 1:
                unique += 1
        signals[guard] = {"flagged": flagged, "unique": unique}
    return {
        "records_total": len(records),
        "signals": signals,
    }


def _build_category_summary(
    records: list[dict[str, Any]],
    *,
    expected_by_category: dict[str, set[str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for category in SUMMARY_CATEGORIES:
        cat_records = [r for r in records if r.get("category") == category]
        payload[category] = {
            "scenarios": len(expected_by_category.get(category, set())),
            "reports": len(cat_records),
            "primary_metric_fail": sum(
                1
                for r in cat_records
                if bool((r.get("guard_flags") or {}).get("primary_metric"))
            ),
            "invariants_fail": sum(
                1
                for r in cat_records
                if bool((r.get("guard_flags") or {}).get("invariants_fail"))
            ),
            "invariants_warn": sum(
                1
                for r in cat_records
                if bool((r.get("guard_flags") or {}).get("invariants_warn"))
            ),
            "any_flag": sum(
                1 for r in cat_records if bool(r.get("any_core_guard_flag"))
            ),
        }
    return payload


def _build_scenario_signal_summary(
    records: list[dict[str, Any]],
    *,
    scenario_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    category_order = {"clean": 0, "stress": 1, "error_injection": 2}
    rows: list[dict[str, Any]] = []

    for scenario_id, spec in sorted(
        scenario_index.items(),
        key=lambda item: (
            category_order.get(str(item[1].get("category") or ""), 9),
            item[0],
        ),
    ):
        category = str(spec.get("category") or "").strip().lower()
        strictness = str(spec.get("strictness") or "").strip().lower()
        intent = str(spec.get("intent") or "")
        primary_guard = str(spec.get("primary_guard") or "")
        requirements = spec.get("requirements")
        primary_guard_required = bool(
            isinstance(requirements, dict)
            and requirements.get("primary_guard_required") is True
        )

        scenario_records = [
            r
            for r in records
            if r.get("category") == category and r.get("name") == scenario_id
        ]
        detector_hits = sum(1 for r in scenario_records if bool(r.get("detectors_hit")))
        signaled = sum(1 for r in scenario_records if _record_signaled(r))
        primary_guard_hits = sum(
            1 for r in scenario_records if bool(r.get("primary_guard_hit"))
        )

        rows.append(
            {
                "id": scenario_id,
                "category": category,
                "strictness": strictness,
                "intent": intent,
                "primary_guard": primary_guard,
                "primary_guard_required": primary_guard_required,
                "reports": len(scenario_records),
                "passed": sum(1 for r in scenario_records if bool(r.get("passed"))),
                "failed": sum(1 for r in scenario_records if not bool(r.get("passed"))),
                "detector_hits": detector_hits,
                "signaled": signaled,
                "primary_guard_hits": primary_guard_hits,
            }
        )

    return rows


def generate_verdict(*, output_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = _manifest_root() / "scenarios.json"
    manifest = _load_scenarios_manifest(manifest_path)

    scenarios = manifest.get("scenarios", [])
    scenario_index: dict[str, dict[str, Any]] = {}
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

    # Pick the newest run for each (model, category, scenario_id).
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

    records: list[dict[str, Any]] = []
    models: set[str] = set()
    for (model_name, category, scenario_id), (run_num, cert_path) in sorted(
        latest.items()
    ):
        models.add(model_name)
        try:
            cert = json.loads(cert_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cert, dict):
            continue

        probe_path = cert_path.parent / "rmt_probe.json"
        if probe_path.is_file():
            try:
                probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
                if isinstance(probe_payload, dict):
                    cert["rmt_probe"] = probe_payload
            except Exception:
                pass

        ve_probe_path = cert_path.parent / "ve_probe.json"
        if ve_probe_path.is_file():
            try:
                probe_payload = json.loads(ve_probe_path.read_text(encoding="utf-8"))
                if isinstance(probe_payload, dict):
                    cert["ve_probe"] = probe_payload
            except Exception:
                pass

        outcome = _evaluate_report(cert)
        spec = scenario_index.get(scenario_id, {})

        reqs = spec.get("requirements") if isinstance(spec, dict) else None
        detectors_any = None
        if isinstance(reqs, dict) and isinstance(reqs.get("detectors_any_of"), list):
            detectors_any = [
                d for d in reqs.get("detectors_any_of") if isinstance(d, dict)
            ]

        detectors_all = None
        if isinstance(reqs, dict) and isinstance(reqs.get("detectors_all_of"), list):
            detectors_all = [
                d for d in reqs.get("detectors_all_of") if isinstance(d, dict)
            ]

        detectors_hit = False
        if detectors_any or detectors_all:
            detectors_hit = True
            if detectors_any:
                detectors_hit = any(_detector_matches(cert, d) for d in detectors_any)
            if detectors_all:
                detectors_hit = detectors_hit and all(
                    _detector_matches(cert, d) for d in detectors_all
                )
        primary_guard_required = bool(
            isinstance(reqs, dict) and reqs.get("primary_guard_required") is True
        )

        record: dict[str, Any] = {
            "model": model_name,
            "category": category,
            "name": scenario_id,
            "strictness": str(spec.get("strictness") or ""),
            "intent": str(spec.get("intent") or ""),
            "primary_guard": str(spec.get("primary_guard") or ""),
            "failure_class": str(spec.get("failure_class") or ""),
            "run_num": run_num,
            "family": _edit_family(scenario_id)
            if category in {"clean", "stress"}
            else "",
            "passed": outcome.passed,
            "reasons": list(outcome.reasons),
            "detectors_hit": detectors_hit,
            "detectors": detectors_any or [],
            "detectors_all_of": detectors_all or [],
            "primary_guard_required": primary_guard_required,
            "invariants_status": outcome.invariants_status,
            "guard_flags": outcome.guard_flags,
            "spectral_caps_applied": _spectral_caps_applied(cert),
            "path": str(cert_path),
        }
        if isinstance(cert.get("rmt_probe"), dict):
            record["rmt_probe"] = cert["rmt_probe"]
            if _as_bool(record["rmt_probe"].get("stable"), default=True) is False:
                record["guard_flags"]["rmt"] = True
        if isinstance(cert.get("ve_probe"), dict):
            record["ve_probe"] = cert["ve_probe"]
        record["primary_guard_hit"] = _record_primary_guard_hit(record)
        record["any_core_guard_flag"] = _core_signal_count(record) > 0
        records.append(record)

    # Organize by model/category/name
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["model"], r["category"], r["name"]): r for r in records
    }
    model_names = sorted(models)
    info_min_signal_fraction = 0.5

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

    for model_name in model_names:
        missing_model: dict[str, list[str]] = {
            "clean": [],
            "stress": [],
            "error_injection": [],
        }
        for category in SUMMARY_CATEGORIES:
            expected_ids = expected_by_category.get(category, set())
            for scenario_id in sorted(expected_ids):
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
        if any(missing_model.get(k) for k in SUMMARY_CATEGORIES):
            missing["by_model"][model_name] = missing_model
            failed_requirements.append(
                {
                    "requirement": "scenario_coverage",
                    "message": "Missing required scenarios for model",
                    "model": model_name,
                    "missing": {
                        k: v
                        for k, v in missing_model.items()
                        if k in {"clean", "stress", "error_injection"} and v
                    },
                }
            )

    # Evaluate gating scenarios.
    for scenario_id in sorted(gating_by_category.get("clean", set())):
        failures = [
            by_key[(m, "clean", scenario_id)]
            for m in model_names
            if (m, "clean", scenario_id) in by_key
            and not bool(by_key[(m, "clean", scenario_id)]["passed"])
        ]
        if failures:
            failed_requirements.append(
                {
                    "requirement": "clean_all_pass",
                    "message": "Clean scenarios must PASS",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": r["model"],
                            "reasons": r["reasons"],
                            "path": r["path"],
                        }
                        for r in failures
                    ],
                }
            )

    for scenario_id in sorted(gating_by_category.get("stress", set())):
        failures = [
            by_key[(m, "stress", scenario_id)]
            for m in model_names
            if (m, "stress", scenario_id) in by_key
            and bool(by_key[(m, "stress", scenario_id)]["passed"])
        ]
        if failures:
            failed_requirements.append(
                {
                    "requirement": "stress_required_fail",
                    "message": "Required stress scenarios must FAIL",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": r["model"],
                            "reasons": r["reasons"],
                            "path": r["path"],
                        }
                        for r in failures
                    ],
                }
            )

        expected_detectors = scenario_index.get(scenario_id, {}).get("requirements", {})
        detectors_any = (
            expected_detectors.get("detectors_any_of")
            if isinstance(expected_detectors, dict)
            else None
        )
        if detectors_any:
            missing_detectors = [
                by_key[(m, "stress", scenario_id)]
                for m in model_names
                if (m, "stress", scenario_id) in by_key
                and not bool(by_key[(m, "stress", scenario_id)]["detectors_hit"])
            ]
            if missing_detectors:
                failed_requirements.append(
                    {
                        "requirement": "stress_expected_detectors",
                        "message": "Stress scenario missing expected detector signal",
                        "scenario": scenario_id,
                        "failures": [
                            {
                                "model": r["model"],
                                "passed": r["passed"],
                                "reasons": r["reasons"],
                                "path": r["path"],
                            }
                            for r in missing_detectors
                        ],
                    }
                )

    for scenario_id in sorted(gating_by_category.get("error_injection", set())):
        missed = [
            by_key[(m, "error_injection", scenario_id)]
            for m in model_names
            if (m, "error_injection", scenario_id) in by_key
            and not bool(by_key[(m, "error_injection", scenario_id)]["detectors_hit"])
        ]
        if missed:
            failed_requirements.append(
                {
                    "requirement": "error_injection_detected",
                    "message": "Error injections must trigger expected detector signals",
                    "scenario": scenario_id,
                    "failures": [
                        {
                            "model": r["model"],
                            "passed": r["passed"],
                            "reasons": r["reasons"],
                            "path": r["path"],
                        }
                        for r in missed
                    ],
                }
            )

    for scenario_id in sorted(primary_guard_required_scenarios):
        spec = scenario_index.get(scenario_id, {})
        category = str(spec.get("category") or "").strip().lower()
        primary_guard = str(spec.get("primary_guard") or "").strip().lower()
        if category not in SUMMARY_CATEGORIES or not primary_guard:
            continue

        present = [
            by_key[(m, category, scenario_id)]
            for m in model_names
            if (m, category, scenario_id) in by_key
        ]
        if not present:
            continue
        if any(bool(r.get("primary_guard_hit")) for r in present):
            continue

        failed_requirements.append(
            {
                "requirement": "scenario_primary_guard_signal",
                "message": "Scenario marked primary_guard_required did not trigger its declared primary guard.",
                "scenario": scenario_id,
                "category": category,
                "primary_guard": primary_guard,
                "failures": [
                    {
                        "model": r["model"],
                        "detectors_hit": r["detectors_hit"],
                        "reasons": r["reasons"],
                        "path": r["path"],
                    }
                    for r in present
                ],
            }
        )

    required_guard_records = [
        r for r in records if r.get("name") in primary_guard_required_scenarios
    ]
    for guard in CORE_GUARDS:
        guard_records = [
            r
            for r in required_guard_records
            if str(r.get("primary_guard") or "") == guard
        ]
        if not guard_records:
            continue
        if any(bool(r.get("primary_guard_hit")) for r in guard_records):
            continue
        failed_requirements.append(
            {
                "requirement": "guard_primary_demonstrated",
                "message": "No primary_guard_required scenario produced a direct hit for guard.",
                "guard": guard,
                "scenarios": sorted({str(r.get("name") or "") for r in guard_records}),
            }
        )

    clean = [r for r in records if r["category"] == "clean"]
    stress = [r for r in records if r["category"] == "stress"]
    errors = [r for r in records if r["category"] == "error_injection"]

    info_stress = [r for r in stress if r["name"] in informational_stress]
    info_fail = sum(1 for r in info_stress if not r["passed"])
    info_signaled = sum(1 for r in info_stress if _record_signaled(r))
    info_total = len(info_stress)

    if info_total > 0:
        info_signal_fraction = info_signaled / info_total
        if info_signal_fraction < info_min_signal_fraction:
            failed_requirements.append(
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

    primary_guard_required_records = [
        r for r in records if r["name"] in primary_guard_required_scenarios
    ]
    primary_guard_required_hits = sum(
        1 for r in primary_guard_required_records if bool(r.get("primary_guard_hit"))
    )

    catastrophic_records = [r for r in stress if r["name"] in catastrophic_required]
    guard_signal_summary = _build_guard_signal_summary(records)
    category_summary = _build_category_summary(
        records,
        expected_by_category=expected_by_category,
    )
    scenario_signal_summary = _build_scenario_signal_summary(
        records,
        scenario_index=scenario_index,
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
        "counts": {
            "models_total": len(model_names),
            "records_total": len(records),
            "clean_total": len(clean),
            "clean_pass": sum(1 for r in clean if r["passed"]),
            "stress_total": len(stress),
            "stress_fail": sum(1 for r in stress if not r["passed"]),
            "catastrophic_required_total": len(catastrophic_required),
            "catastrophic_required_present": len(
                {r["name"] for r in catastrophic_records}
            ),
            "catastrophic_required_fail": sum(
                1 for r in catastrophic_records if not r["passed"]
            ),
            "error_injection_total": len(errors),
            "error_injection_detected": sum(1 for r in errors if r["detectors_hit"]),
            "informational_stress_total": info_total,
            "informational_stress_fail": info_fail,
            "informational_stress_signaled": info_signaled,
            "primary_guard_required_scenarios": len(primary_guard_required_scenarios),
            "primary_guard_required_records": len(primary_guard_required_records),
            "primary_guard_required_hits": primary_guard_required_hits,
            "primary_guard_required_scenarios_hit": len(
                {
                    str(r.get("name") or "")
                    for r in primary_guard_required_records
                    if bool(r.get("primary_guard_hit"))
                }
            ),
        },
        "core_guard_order": list(CORE_GUARDS),
        "category_summary": category_summary,
        "guard_signal_summary": guard_signal_summary,
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
