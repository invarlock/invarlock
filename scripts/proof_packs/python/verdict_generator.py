from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class CertOutcome:
    passed: bool
    reasons: tuple[str, ...]


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

    return False


def _evaluate_certificate(cert: dict[str, Any]) -> CertOutcome:
    validation = cert.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}

    invariants_ok = _as_bool(validation.get("invariants_pass"), default=False)
    pm_ok = _as_bool(validation.get("primary_metric_acceptable"), default=False)
    spectral_ok = _as_bool(validation.get("spectral_stable"), default=False)
    rmt_ok = _as_bool(validation.get("rmt_stable"), default=False)
    drift_ok = _as_bool(validation.get("preview_final_drift_acceptable"), default=True)

    guard_overhead = cert.get("guard_overhead") or {}
    evaluated = False
    if isinstance(guard_overhead, dict):
        evaluated = _as_bool(guard_overhead.get("evaluated"), default=False)
    overhead_ok = _as_bool(validation.get("guard_overhead_acceptable"), default=True)

    primary_metric = cert.get("primary_metric") or {}
    pm_degraded = False
    if isinstance(primary_metric, dict):
        pm_degraded = _as_bool(
            primary_metric.get("degraded"), default=False
        ) or _as_bool(primary_metric.get("invalid"), default=False)

    passed = (
        invariants_ok
        and pm_ok
        and spectral_ok
        and rmt_ok
        and drift_ok
        and (overhead_ok if evaluated else True)
        and not pm_degraded
    )

    reasons: list[str] = []
    if pm_degraded:
        reasons.append("primary_metric_degraded")
    if not invariants_ok:
        reasons.append("invariants_fail")
    if not pm_ok:
        reasons.append("primary_metric_fail")
    if not spectral_ok:
        reasons.append("spectral_fail")
    if not rmt_ok:
        reasons.append("rmt_fail")
    if not drift_ok:
        reasons.append("drift_fail")
    if evaluated and not overhead_ok:
        reasons.append("overhead_fail")

    return CertOutcome(passed=passed, reasons=tuple(reasons))


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


def _classify_certificate(
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
        idx = parts.index("certificates")
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
        idx = parts.index("certificates")
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


def generate_verdict(*, output_dir: Path) -> dict[str, Any]:
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

    # Pick the newest run for each (model, category, scenario_id).
    latest: dict[tuple[str, str, str], tuple[int, Path]] = {}
    for cert_path in sorted(output_dir.glob("*/certificates/**/evaluation.cert.json")):
        cls = _classify_certificate(cert_path, output_dir=output_dir)
        if cls is None:
            continue
        model_name, category, scenario_id = cls
        if category not in {"clean", "stress", "error_injection"}:
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
        outcome = _evaluate_certificate(cert)
        spec = scenario_index.get(scenario_id)
        reqs = spec.get("requirements") if isinstance(spec, dict) else None
        detectors = None
        if isinstance(reqs, dict) and isinstance(reqs.get("detectors_any_of"), list):
            detectors = [d for d in reqs.get("detectors_any_of") if isinstance(d, dict)]
        detectors_hit = False
        if detectors:
            detectors_hit = any(_detector_matches(cert, d) for d in detectors)

        records.append(
            {
                "model": model_name,
                "category": category,
                "name": scenario_id,
                "run_num": run_num,
                "family": _edit_family(scenario_id)
                if category in {"clean", "stress"}
                else "",
                "passed": outcome.passed,
                "reasons": list(outcome.reasons),
                "detectors_hit": detectors_hit,
                "detectors": detectors or [],
                "path": str(cert_path),
            }
        )

    # Organize by model/category/name
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["model"], r["category"], r["name"]): r for r in records
    }
    model_names = sorted(models)

    failed_requirements: list[dict[str, Any]] = []
    missing: dict[str, Any] = {"by_model": {}}

    for model_name in model_names:
        missing_model: dict[str, list[str]] = {
            "clean": [],
            "stress": [],
            "error_injection": [],
        }
        for category in ("clean", "stress", "error_injection"):
            expected_ids = expected_by_category.get(category, set())
            for scenario_id in sorted(expected_ids):
                if (model_name, category, scenario_id) in by_key:
                    continue
                if scenario_id in gating_by_category.get(category, set()):
                    missing_model[category].append(scenario_id)
                else:
                    missing_model.setdefault(f"{category}_informational", []).append(
                        scenario_id
                    )
        if any(missing_model.get(k) for k in ("clean", "stress", "error_injection")):
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

    # Evaluate gating scenarios
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

    verdict = "PASS" if not failed_requirements else "FAIL"

    clean = [r for r in records if r["category"] == "clean"]
    stress = [r for r in records if r["category"] == "stress"]
    errors = [r for r in records if r["category"] == "error_injection"]

    info_stress = [r for r in stress if r["name"] in informational_stress]
    info_fail = sum(1 for r in info_stress if not r["passed"])
    info_total = len(info_stress)

    catastrophic_records = [r for r in stress if r["name"] in catastrophic_required]

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
            "informational_stress_min_fail_fraction": 0.5,
        },
        "counts": {
            "models_total": len(model_names),
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
        },
        "missing": missing,
        "failed_requirements": failed_requirements,
        "timestamp": datetime.now().isoformat(),
    }


def _render_text(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    missing = payload.get("missing") or {}
    failed = payload.get("failed_requirements") or []

    manifest = payload.get("manifest") or {}

    lines = [
        "INVARLOCK PROOF PACK (ASSURANCE) — FINAL VERDICT",
        f"Verdict: {payload.get('verdict')}",
        f"Scenarios manifest: {manifest.get('path')}",
        "",
        "COUNTS:",
        f"  Clean: {counts.get('clean_pass')}/{counts.get('clean_total')} PASS",
        f"  Stress: {counts.get('stress_fail')}/{counts.get('stress_total')} FAIL (expected for stress)",
        (
            "  Catastrophic-required stress: "
            f"{counts.get('catastrophic_required_fail')}/{counts.get('catastrophic_required_total')} FAIL"
        ),
        (
            "  Error injection detected: "
            f"{counts.get('error_injection_detected')}/{counts.get('error_injection_total')}"
        ),
        "",
    ]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = generate_verdict(output_dir=output_dir)
    (reports_dir / "final_verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "final_verdict.txt").write_text(
        _render_text(payload),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
