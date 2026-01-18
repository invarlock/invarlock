from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
        pm_degraded = _as_bool(primary_metric.get("degraded"), default=False) or _as_bool(
            primary_metric.get("invalid"), default=False
        )

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


def generate_verdict(*, output_dir: Path) -> dict[str, Any]:
    expected_clean_families = {"quant", "fp8", "prune", "svd"}
    expected_stress_families = {"quant", "fp8", "prune", "svd"}
    expected_errors = {
        "nan_injection",
        "inf_injection",
        "extreme_quant",
        "scale_explosion",
        "weight_tying_break",
    }

    catastrophic_required = {"prune_50pct_stress", "svd_rank32_stress"}
    stress_informational = {"quant_4bit_stress", "fp8_e5m2_stress"}

    records: list[dict[str, Any]] = []
    for cert_path in sorted(output_dir.glob("*/certificates/**/evaluation.cert.json")):
        cls = _classify_certificate(cert_path, output_dir=output_dir)
        if cls is None:
            continue
        model_name, category, name = cls
        try:
            cert = json.loads(cert_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cert, dict):
            continue
        outcome = _evaluate_certificate(cert)
        records.append(
            {
                "model": model_name,
                "category": category,
                "name": name,
                "family": _edit_family(name) if category in {"clean", "stress"} else "",
                "passed": outcome.passed,
                "reasons": list(outcome.reasons),
                "path": str(cert_path),
            }
        )

    # Aggregate per category
    clean = [r for r in records if r["category"] == "clean"]
    stress = [r for r in records if r["category"] == "stress"]
    errors = [r for r in records if r["category"] == "error_injection"]

    clean_pass = sum(1 for r in clean if r["passed"])
    stress_fail = sum(1 for r in stress if not r["passed"])
    errors_detected = sum(1 for r in errors if not r["passed"])

    clean_families = {r["family"] for r in clean if r["family"]}
    stress_families = {r["family"] for r in stress if r["family"]}
    error_types = {r["name"] for r in errors}

    missing: dict[str, Any] = {
        "clean_families": sorted(expected_clean_families - clean_families),
        "stress_families": sorted(expected_stress_families - stress_families),
        "errors": sorted(expected_errors - error_types),
        "catastrophic_required": [],
    }

    catastrophic_records = {r["name"]: r for r in stress if r["name"] in catastrophic_required}
    missing["catastrophic_required"] = sorted(catastrophic_required - set(catastrophic_records))

    failed_requirements: list[dict[str, Any]] = []
    if missing["clean_families"]:
        failed_requirements.append(
            {
                "requirement": "clean_coverage",
                "message": "Missing clean edit families",
                "missing": missing["clean_families"],
            }
        )
    if missing["stress_families"]:
        failed_requirements.append(
            {
                "requirement": "stress_coverage",
                "message": "Missing stress edit families",
                "missing": missing["stress_families"],
            }
        )
    if missing["errors"]:
        failed_requirements.append(
            {
                "requirement": "error_injection_coverage",
                "message": "Missing error injection scenarios",
                "missing": missing["errors"],
            }
        )
    if missing["catastrophic_required"]:
        failed_requirements.append(
            {
                "requirement": "catastrophic_required_coverage",
                "message": "Missing catastrophic-required stress scenarios",
                "missing": missing["catastrophic_required"],
            }
        )

    clean_failed = [r for r in clean if not r["passed"]]
    if clean_failed:
        failed_requirements.append(
            {
                "requirement": "clean_all_pass",
                "message": "Clean edits must PASS",
                "failures": [
                    {
                        "name": r["name"],
                        "model": r["model"],
                        "reasons": r["reasons"],
                        "path": r["path"],
                    }
                    for r in clean_failed
                ],
            }
        )

    catastrophic_failed = [
        r
        for r in stress
        if r["name"] in catastrophic_required and r["passed"]
    ]
    if catastrophic_failed:
        failed_requirements.append(
            {
                "requirement": "catastrophic_required_fail",
                "message": "Catastrophic-required stress edits must FAIL",
                "failures": [
                    {
                        "name": r["name"],
                        "model": r["model"],
                        "reasons": r["reasons"],
                        "path": r["path"],
                    }
                    for r in catastrophic_failed
                ],
            }
        )

    error_missed = [r for r in errors if r["passed"]]
    if error_missed:
        failed_requirements.append(
            {
                "requirement": "error_injection_detected",
                "message": "Error injections must be detected (not PASS)",
                "failures": [
                    {
                        "name": r["name"],
                        "model": r["model"],
                        "reasons": r["reasons"],
                        "path": r["path"],
                    }
                    for r in error_missed
                ],
            }
        )

    verdict = "PASS" if not failed_requirements else "FAIL"

    # Informational: non-catastrophic stress fail rate.
    info_stress = [r for r in stress if r["name"] in stress_informational]
    info_fail = sum(1 for r in info_stress if not r["passed"])
    info_total = len(info_stress)

    return {
        "verdict": verdict,
        "criteria": {
            "clean_all_pass": True,
            "catastrophic_required_fail": True,
            "error_injection_detected": True,
            "informational_stress_min_fail_fraction": 0.5,
        },
        "counts": {
            "clean_total": len(clean),
            "clean_pass": clean_pass,
            "stress_total": len(stress),
            "stress_fail": stress_fail,
            "catastrophic_required_total": len(catastrophic_required),
            "catastrophic_required_present": len(catastrophic_records),
            "catastrophic_required_fail": sum(
                1
                for r in stress
                if r["name"] in catastrophic_required and not r["passed"]
            ),
            "error_injection_total": len(errors),
            "error_injection_detected": errors_detected,
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

    lines = [
        "INVARLOCK PROOF PACK (ASSURANCE) — FINAL VERDICT",
        f"Verdict: {payload.get('verdict')}",
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
        "MISSING:",
        f"  Clean families: {', '.join(missing.get('clean_families', [])) or 'none'}",
        f"  Stress families: {', '.join(missing.get('stress_families', [])) or 'none'}",
        f"  Errors: {', '.join(missing.get('errors', [])) or 'none'}",
        "",
    ]
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

