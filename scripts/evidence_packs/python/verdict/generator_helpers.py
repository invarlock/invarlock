from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORE_GUARDS: tuple[str, ...] = (
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "primary_metric",
)
INTERVENTION_SIGNALS: tuple[str, ...] = (
    "spectral_caps",
    "ve_signal",
)
SUMMARY_CATEGORIES: tuple[str, ...] = ("clean", "stress", "error_injection")


def _manifest_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_scenarios_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read scenarios manifest: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Scenarios manifest must be a JSON object: {path}")
    if payload.get("schema") != "evidence_pack_scenarios_v1":
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
        except (TypeError, ValueError, OverflowError):
            return default
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (TypeError, ValueError):
            return default
    return default


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        v = float(value)
        return v if math.isfinite(v) else default
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except (TypeError, ValueError, OverflowError):
            return default
        return v if math.isfinite(v) else default
    return default


def _spectral_caps_applied(cert: dict[str, Any]) -> int:
    spectral = cert.get("spectral")
    if not isinstance(spectral, dict):
        return 0
    return max(0, _as_int(spectral.get("caps_applied"), default=0))


def _spectral_cap_modules(cert: dict[str, Any] | None) -> set[tuple[str, str]]:
    if not isinstance(cert, dict):
        return set()
    spectral = cert.get("spectral")
    if not isinstance(spectral, dict):
        return set()
    modules: set[tuple[str, str]] = set()
    violations = spectral.get("violations")
    if not isinstance(violations, list):
        return modules
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        if str(violation.get("type") or "") != "family_z_cap":
            continue
        module = violation.get("module")
        if not isinstance(module, str) or not module.strip():
            continue
        family = violation.get("family")
        family_name = family if isinstance(family, str) else ""
        modules.add((module.strip(), family_name.strip()))
    return modules


def _spectral_baseline_relative_summary(
    cert: dict[str, Any],
    baseline_cert: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_available = isinstance(baseline_cert, dict)
    subject_caps = _spectral_caps_applied(cert)
    baseline_caps = _spectral_caps_applied(baseline_cert or {})
    subject_modules = _spectral_cap_modules(cert)
    baseline_modules = _spectral_cap_modules(baseline_cert)
    new_modules = sorted(
        [
            {"module": module, "family": family}
            for module, family in subject_modules - baseline_modules
        ],
        key=lambda item: (item["family"], item["module"]),
    )
    return {
        "baseline_available": baseline_available,
        "baseline_caps_applied": baseline_caps if baseline_available else None,
        "subject_caps_applied": subject_caps,
        "delta_caps_applied": (
            max(0, subject_caps - baseline_caps) if baseline_available else None
        ),
        "new_caps_applied": len(new_modules) if baseline_available else None,
        "new_cap_modules": new_modules,
    }


def _rmt_signal(cert: dict[str, Any] | None) -> bool:
    if not isinstance(cert, dict):
        return False
    validation = cert.get("validation")
    if isinstance(validation, dict) and "rmt_stable" in validation:
        if _as_bool(validation.get("rmt_stable"), default=True) is False:
            return True
    probe = cert.get("rmt_probe")
    if isinstance(probe, dict) and "stable" in probe:
        if _as_bool(probe.get("stable"), default=True) is False:
            return True
    return False


def _variance_signal(cert: dict[str, Any] | None) -> bool:
    if not isinstance(cert, dict):
        return False
    probe = cert.get("ve_probe")
    if not isinstance(probe, dict):
        return False
    if _as_bool(probe.get("signal"), default=False):
        return True
    if _as_bool(probe.get("would_enable"), default=False):
        return True
    if _as_int(probe.get("proposed_scales"), default=0) > 0:
        return True
    gain = _as_float(probe.get("ab_gain"), default=None)
    return gain is not None and gain > 0.0


def _invariants_signal(cert: dict[str, Any] | None) -> bool:
    if not isinstance(cert, dict):
        return False
    validation = cert.get("validation")
    if isinstance(validation, dict) and "invariants_pass" in validation:
        if _as_bool(validation.get("invariants_pass"), default=False) is False:
            return True
    invariants = cert.get("invariants")
    if isinstance(invariants, dict):
        status = invariants.get("status")
        if isinstance(status, str) and status.strip().lower() in {
            "warn",
            "fail",
            "error",
        }:
            return True
    return False


def _primary_metric_signal(cert: dict[str, Any] | None) -> bool:
    if not isinstance(cert, dict):
        return False
    validation = cert.get("validation")
    if isinstance(validation, dict) and "primary_metric_acceptable" in validation:
        if _as_bool(validation.get("primary_metric_acceptable"), default=True) is False:
            return True
    primary_metric = cert.get("primary_metric")
    if isinstance(primary_metric, dict):
        if _as_bool(primary_metric.get("degraded"), default=False):
            return True
        if _as_bool(primary_metric.get("invalid"), default=False):
            return True
    return False


def _guard_signal(cert: dict[str, Any] | None, guard: str) -> bool:
    guard_name = guard.strip().lower()
    if guard_name == "spectral":
        return _spectral_caps_applied(cert or {}) > 0
    if guard_name == "rmt":
        return _rmt_signal(cert)
    if guard_name == "variance":
        return _variance_signal(cert)
    if guard_name == "invariants":
        return _invariants_signal(cert)
    if guard_name == "primary_metric":
        return _primary_metric_signal(cert)
    return False


def _guard_baseline_relative_summary(
    cert: dict[str, Any],
    baseline_cert: dict[str, Any] | None,
    guard: str,
) -> dict[str, Any]:
    guard_name = guard.strip().lower()
    baseline_available = isinstance(baseline_cert, dict)
    subject_signal = _guard_signal(cert, guard_name)
    baseline_signal = _guard_signal(baseline_cert, guard_name)
    payload: dict[str, Any] = {
        "baseline_available": baseline_available,
        "subject_signal": subject_signal,
        "baseline_signal": baseline_signal if baseline_available else None,
        "relative_signal": baseline_available and subject_signal and not baseline_signal,
    }
    if guard_name == "spectral":
        spectral = _spectral_baseline_relative_summary(cert, baseline_cert)
        payload.update(spectral)
        payload["relative_signal"] = bool(
            spectral.get("baseline_available")
            and (
                _as_int(spectral.get("new_caps_applied"), default=0) > 0
                or _as_int(spectral.get("delta_caps_applied"), default=0) > 0
            )
        )
    return payload


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


@dataclass(frozen=True)
class ScenarioCatalog:
    scenario_index: dict[str, dict[str, Any]]
    expected_by_category: dict[str, set[str]]
    gating_by_category: dict[str, set[str]]
    catastrophic_required: set[str]
    informational_stress: set[str]
    primary_guard_required_scenarios: set[str]


def _validation_snapshot(cert: dict[str, Any]) -> ValidationSnapshot:
    validation = cert.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}

    invariants_ok = _as_bool(validation.get("invariants_pass"), default=False)
    pm_ok = _as_bool(validation.get("primary_metric_acceptable"), default=False)
    spectral_ok = _as_bool(validation.get("spectral_stable"), default=False)
    rmt_ok = _as_bool(validation.get("rmt_stable"), default=False)
    drift_ok = _as_bool(validation.get("preview_final_drift_acceptable"), default=False)

    guard_overhead = cert.get("guard_overhead") or {}
    overhead_evaluated = False
    if isinstance(guard_overhead, dict):
        overhead_evaluated = _as_bool(guard_overhead.get("evaluated"), default=False)
    overhead_ok = _as_bool(validation.get("guard_overhead_acceptable"), default=False)

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
        "variance": False,
        "drift": not snapshot.drift_ok,
        "overhead": snapshot.overhead_evaluated and (not snapshot.overhead_ok),
    }


def _detector_matches(
    cert: dict[str, Any],
    detector: dict[str, Any],
    *,
    baseline_cert: dict[str, Any] | None = None,
) -> bool:
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
        except (TypeError, ValueError, OverflowError):
            return False
        if min_val < 0:
            min_val = 0
        return _spectral_caps_applied(cert) >= min_val

    if kind == "spectral_caps_baseline_relative":
        if not isinstance(baseline_cert, dict):
            return False

        summary = _spectral_baseline_relative_summary(cert, baseline_cert)

        min_new_modules = detector.get("min_new_modules")
        min_delta_count = detector.get("min_delta_count")
        if min_new_modules is None and min_delta_count is None:
            min_new_modules = 1

        if min_new_modules is not None:
            try:
                min_new = int(min_new_modules)
            except (TypeError, ValueError, OverflowError):
                return False
            if min_new < 0:
                min_new = 0
            if _as_int(summary.get("new_caps_applied"), default=0) < min_new:
                return False

        if min_delta_count is not None:
            try:
                min_delta = int(min_delta_count)
            except (TypeError, ValueError, OverflowError):
                return False
            if min_delta < 0:
                min_delta = 0
            if _as_int(summary.get("delta_caps_applied"), default=0) < min_delta:
                return False

        return True

    if kind == "guard_signal_baseline_relative":
        guard = detector.get("guard")
        if not isinstance(guard, str) or not guard.strip():
            return False
        guard_name = guard.strip().lower()
        if guard_name == "spectral":
            if not isinstance(baseline_cert, dict):
                return False
            summary = _guard_baseline_relative_summary(cert, baseline_cert, guard_name)
            min_new_modules = detector.get("min_new_modules")
            min_delta_count = detector.get("min_delta_count")
            if min_new_modules is None and min_delta_count is None:
                return bool(summary.get("relative_signal"))
            if min_new_modules is not None:
                try:
                    min_new = int(min_new_modules)
                except (TypeError, ValueError, OverflowError):
                    return False
                if min_new < 0:
                    min_new = 0
                if _as_int(summary.get("new_caps_applied"), default=0) < min_new:
                    return False
            if min_delta_count is not None:
                try:
                    min_delta = int(min_delta_count)
                except (TypeError, ValueError, OverflowError):
                    return False
                if min_delta < 0:
                    min_delta = 0
                if _as_int(summary.get("delta_caps_applied"), default=0) < min_delta:
                    return False
            return True

        summary = _guard_baseline_relative_summary(cert, baseline_cert, guard_name)
        return bool(summary.get("relative_signal"))

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
            except (TypeError, ValueError):
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

    if bool(record.get("primary_guard_baseline_relative_required")):
        relative_by_guard = record.get("guard_baseline_relative")
        if not isinstance(relative_by_guard, dict):
            return False
        relative = relative_by_guard.get(primary_guard)
        if not isinstance(relative, dict):
            return False
        return bool(relative.get("relative_signal"))

    if bool(flags.get(primary_guard)):
        return True

    if primary_guard == "rmt":
        probe = record.get("rmt_probe")
        if isinstance(probe, dict):
            stable = probe.get("stable")
            if stable is not None and _as_bool(stable, default=True) is False:
                return True
    if primary_guard == "spectral":
        if bool(record.get("spectral_baseline_relative_required")):
            relative = record.get("spectral_baseline_relative")
            if not isinstance(relative, dict):
                return False
            if _as_int(relative.get("new_caps_applied"), default=0) > 0:
                return True
            if _as_int(relative.get("delta_caps_applied"), default=0) > 0:
                return True
            return False
        if int(record.get("spectral_caps_applied") or 0) > 0:
            return True
    if primary_guard == "variance":
        probe = record.get("ve_probe")
        if isinstance(probe, dict):
            signal = probe.get("signal")
            if signal is not None and _as_bool(signal, default=False) is True:
                return True
            would_enable = probe.get("would_enable")
            if (
                would_enable is not None
                and _as_bool(would_enable, default=False) is True
            ):
                return True
            scales = _as_int(probe.get("proposed_scales"), default=0)
            if scales > 0:
                return True
            gain = _as_float(probe.get("ab_gain"), default=None)
            if gain is not None and gain > 0.0:
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


def _intervention_flags(record: dict[str, Any]) -> dict[str, bool]:
    spectral_caps = int(record.get("spectral_caps_applied") or 0) > 0

    ve_signal = False
    probe = record.get("ve_probe")
    if isinstance(probe, dict):
        if _as_bool(probe.get("signal"), default=False):
            ve_signal = True
        if _as_bool(probe.get("would_enable"), default=False):
            ve_signal = True
        scales = _as_int(probe.get("proposed_scales"), default=0)
        gain = _as_float(probe.get("ab_gain"), default=None)
        if scales > 0:
            ve_signal = True
        if gain is not None and gain > 0.0:
            ve_signal = True

    return {
        "spectral_caps": spectral_caps,
        "ve_signal": ve_signal,
    }


def _build_guard_intervention_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    signals: dict[str, dict[str, int]] = {}
    for signal in INTERVENTION_SIGNALS:
        flagged = 0
        unique = 0
        for record in records:
            flags = _intervention_flags(record)
            if not flags.get(signal, False):
                continue
            flagged += 1
            if sum(1 for k in INTERVENTION_SIGNALS if flags.get(k, False)) == 1:
                unique += 1
        signals[signal] = {"flagged": flagged, "unique": unique}
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
            record
            for record in records
            if record.get("category") == category and record.get("name") == scenario_id
        ]
        detector_hits = sum(
            1 for record in scenario_records if bool(record.get("detectors_hit"))
        )
        signaled = sum(1 for record in scenario_records if _record_signaled(record))
        primary_guard_hits = sum(
            1 for record in scenario_records if bool(record.get("primary_guard_hit"))
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
                "passed": sum(
                    1 for record in scenario_records if bool(record.get("passed"))
                ),
                "failed": sum(
                    1 for record in scenario_records if not bool(record.get("passed"))
                ),
                "detector_hits": detector_hits,
                "signaled": signaled,
                "primary_guard_hits": primary_guard_hits,
            }
        )

    return rows
