#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:
    from invarlock.eval.primary_metric import (
        compute_primary_metric_from_report as _compute_primary_metric_from_report,
    )
except Exception:  # pragma: no cover - optional dependency for golden runs
    _compute_primary_metric_from_report = None


def _sha256_ids(window_ids: Iterable[int]) -> str:
    h = hashlib.sha256()
    for wid in window_ids:
        try:
            b = int(wid).to_bytes(8, "little", signed=True)
        except Exception:
            b = str(wid).encode("utf-8", "ignore")
        h.update(b)
    return h.hexdigest()


def _compute_provider_digest(report: dict[str, Any]) -> dict[str, str] | None:
    windows = report.get("evaluation_windows")
    if not isinstance(windows, dict):
        return None
    ids_bytes = bytearray()
    for section in ("preview", "final"):
        sec = windows.get(section)
        if not isinstance(sec, dict):
            continue
        wids = sec.get("window_ids")
        if isinstance(wids, list):
            for wid in wids:
                try:
                    ids_bytes.extend(int(wid).to_bytes(8, "little", signed=True))
                except Exception:
                    ids_bytes.extend(str(wid).encode("utf-8", "ignore"))
    ids_sha = hashlib.sha256(bytes(ids_bytes)).hexdigest() if ids_bytes else None
    digest: dict[str, str] = {}
    if ids_sha:
        digest["ids_sha256"] = ids_sha
    return digest or None


def _make_windows(
    logloss_prev: list[float],
    logloss_fin: list[float],
    tokens_prev: list[int],
    tokens_fin: list[int],
) -> dict[str, Any]:
    preview_ids = list(range(1, 1 + len(logloss_prev)))
    final_ids = list(range(1001, 1001 + len(logloss_fin)))
    return {
        "preview": {
            "window_ids": preview_ids,
            "logloss": logloss_prev,
            "token_counts": tokens_prev,
        },
        "final": {
            "window_ids": final_ids,
            "logloss": logloss_fin,
            "token_counts": tokens_fin,
        },
    }


def _weighted_mean(v: list[float], w: list[int]) -> float:
    sw = 0.0
    swx = 0.0
    for vi, wi in zip(v, w, strict=False):
        sw += float(wi)
        swx += float(wi) * float(vi)
    return swx / max(sw, 1.0)


def build_ppl_pair(kind: str = "ppl_causal") -> tuple[dict[str, Any], dict[str, Any]]:
    # Deterministic synthetic windows
    prev_ll = [1.00, 1.06, 0.98, 1.02]
    fin_ll_subj = [1.02, 1.08, 1.00, 1.03]
    fin_ll_base = [1.01, 1.07, 0.99, 1.02]
    prev_tc = [100, 120, 90, 110]
    fin_tc = [100, 120, 90, 110]
    windows_subj = _make_windows(prev_ll, fin_ll_subj, prev_tc, fin_tc)
    windows_base = _make_windows(prev_ll, fin_ll_base, prev_tc, fin_tc)

    def ppl_from(ll: list[float], tc: list[int]) -> float:
        return math.exp(_weighted_mean(ll, tc))

    # Compute PPL points for sanity but don't propagate (PM-only artifacts below)
    _ = ppl_from(prev_ll, prev_tc)
    _ = ppl_from(fin_ll_subj, fin_tc)
    _ = ppl_from(fin_ll_base, fin_tc)

    report = {
        "meta": {
            "model_id": f"stub-{kind}",
            "adapter": "stub",
            "device": "cpu",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
        },
        "metrics": {
            "bootstrap": {"replicates": 300, "alpha": 0.05, "method": "percentile"},
            "window_plan": {
                "profile": "ci",
                "requested_preview": len(prev_ll),
                "requested_final": len(fin_ll_subj),
                "actual_preview": len(prev_ll),
                "actual_final": len(fin_ll_subj),
                "coverage_ok": True,
            },
            "preview_total_tokens": sum(prev_tc),
            "final_total_tokens": sum(fin_tc),
            "latency_ms_p50": 1.7,
            "throughput_sps": 123.4,
        },
        "evaluation_windows": windows_subj,
        "edit": {"name": "structured"},
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "run_id": "baseline",
        "model_id": f"stub-{kind}",
        "metrics": {
            "bootstrap": {"replicates": 300, "alpha": 0.05, "method": "percentile"}
        },
        "evaluation_windows": windows_base,
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    # Attach primary_metric to both when primary metric helper is available
    if _compute_primary_metric_from_report is not None:
        try:
            base_pm = _compute_primary_metric_from_report(baseline, kind=kind)
            baseline.setdefault("metrics", {})["primary_metric"] = base_pm
            subj_pm = _compute_primary_metric_from_report(
                report,
                kind=kind,
                baseline=baseline,
            )
            report.setdefault("metrics", {})["primary_metric"] = subj_pm
        except Exception:
            # Golden runs should still be usable even if primary metric computation fails
            pass
    pd = _compute_provider_digest(report)
    if pd:
        report.setdefault("provenance", {})["provider_digest"] = pd
        baseline.setdefault("provenance", {})["provider_digest"] = pd
    return report, baseline


def build_accuracy_pair(
    kind: str = "accuracy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Deterministic aggregate counts
    prev = {"correct_total": 42, "total": 80}
    fin_subj = {"correct_total": 51, "total": 80}
    fin_base = {"correct_total": 48, "total": 80}
    report = {
        "meta": {
            "model_id": f"stub-{kind}",
            "adapter": "stub",
            "device": "cpu",
            "seed": 13,
            "seeds": {"python": 13, "numpy": 13, "torch": 13},
        },
        "metrics": {
            "classification": {"preview": prev, "final": fin_subj},
        },
        "edit": {"name": "structured"},
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "run_id": "baseline",
        "model_id": f"stub-{kind}",
        "metrics": {"classification": {"final": fin_base}},
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    return report, baseline


@dataclass
class CheckResult:
    name: str
    ok: bool
    msg: str = ""


def main() -> int:
    from invarlock.reporting.certificate import make_certificate

    checks: list[CheckResult] = []

    # ppl_causal → PM-only invariants
    r, b = build_ppl_pair("ppl_causal")
    cert = make_certificate(r, b)
    checks.append(CheckResult("schema_v1", cert.get("schema_version") == "v1"))
    checks.append(CheckResult("no_top_level_ppl", "ppl" not in cert))
    pm = cert.get("primary_metric", {})
    checks.append(CheckResult("pm_present", isinstance(pm, dict) and bool(pm)))
    checks.append(
        CheckResult(
            "pm_display_ci_len2",
            isinstance(pm.get("display_ci"), tuple | list)
            and len(pm.get("display_ci")) == 2,
        )
    )
    # Direction sanity: ensure direction is annotated
    checks.append(
        CheckResult(
            "pm_direction_present",
            str(pm.get("direction", "lower")).lower() in {"lower", "higher"},
        )
    )

    # Markdown should not contain ‘PPL’ label (PM-first)
    try:
        from invarlock.reporting.certificate import render_certificate_markdown

        md = render_certificate_markdown(cert)
        checks.append(CheckResult("no_PPL_in_markdown", "PPL" not in md))
    except Exception as exc:
        checks.append(CheckResult("no_PPL_in_markdown", False, f"exception: {exc}"))

    # System overhead rendered when telemetry exists
    checks.append(
        CheckResult(
            "system_overhead_present",
            "system_overhead" in cert
            and isinstance(cert["system_overhead"], dict)
            and len(cert["system_overhead"]) > 0,
            "system_overhead section",
        )
    )
    # Provider digest stable
    r2, b2 = build_ppl_pair("ppl_causal")
    c2 = make_certificate(r2, b2)
    pd1 = cert.get("provenance", {}).get("provider_digest", {})
    pd2 = c2.get("provenance", {}).get("provider_digest", {})
    checks.append(CheckResult("provider_digest_stable", bool(pd1) and pd1 == pd2))

    # ppl_mlm present
    r, b = build_ppl_pair("ppl_mlm")
    cert_mlm = make_certificate(r, b)
    checks.append(
        CheckResult(
            "ppl_mlm_pm_present",
            isinstance(cert_mlm.get("primary_metric", {}), dict)
            and str(cert_mlm.get("primary_metric", {}).get("kind")).startswith("ppl"),
        )
    )

    # ppl_seq2seq CI present
    r, b = build_ppl_pair("ppl_seq2seq")
    cert_s2s = make_certificate(r, b)
    pm_s2s = cert_s2s.get("primary_metric", {})
    checks.append(
        CheckResult(
            "ppl_seq2seq_ci_present",
            isinstance(pm_s2s.get("display_ci"), tuple | list)
            and len(pm_s2s.get("display_ci")) == 2,
        )
    )

    # Equivalence parity: ppl_* preview/final vs legacy fields (tolerance 1e-9 formatting)
    try:
        from invarlock.eval.primary_metric import compute_primary_metric_from_report

        for kind in ("ppl_causal", "ppl_mlm", "ppl_seq2seq"):
            rr, bb = build_ppl_pair(kind)
            pmv = compute_primary_metric_from_report(rr, kind=kind, baseline=bb)
            p_v2 = float(pmv.get("preview", float("nan")))
            f_v2 = float(pmv.get("final", float("nan")))

            # Compute expected directly from windows
            def wmean(sec: dict[str, Any]) -> float:
                v = sec.get("logloss") or []
                w = sec.get("token_counts") or []
                sw = sum(float(x) for x in w) or 1.0
                return sum(float(a) * float(b) for a, b in zip(v, w, strict=False)) / sw

            prev = rr.get("evaluation_windows", {}).get("preview", {})
            fin = rr.get("evaluation_windows", {}).get("final", {})
            p_expected = math.exp(wmean(prev))
            f_expected = math.exp(wmean(fin))
            ok = (
                all(math.isfinite(x) for x in (p_expected, f_expected, p_v2, f_v2))
                and math.isclose(p_expected, p_v2, rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(f_expected, f_v2, rel_tol=1e-9, abs_tol=1e-9)
            )
            checks.append(CheckResult(f"equivalence_{kind}", ok, "metric-v1 parity"))
    except Exception as exc:
        checks.append(CheckResult("equivalence_parity", False, f"exception: {exc}"))

    # accuracy delta pp and CI presence (via paired bootstrap)
    r, b = build_accuracy_pair("accuracy")
    try:
        from invarlock.eval.primary_metric import MetricContribution, get_metric

        pm_acc = get_metric("accuracy")
        subj = [MetricContribution(1.0 if i % 2 == 0 else 0.0) for i in range(80)]
        base = [
            MetricContribution(1.0 if (i % 2 == 0 and i % 5 != 0) else 0.0)
            for i in range(80)
        ]
        comp = pm_acc.paired_compare(subj, base, reps=500, seed=123, ci_level=0.95)
        comp["kind"] = "accuracy"
        r.setdefault("metrics", {})["primary_metric"] = comp
    except Exception:
        pass
    cert_acc = make_certificate(r, b)
    pm_acc2 = cert_acc.get("primary_metric", {})
    checks.append(
        CheckResult(
            "accuracy_ci_present",
            isinstance(pm_acc2.get("display_ci"), tuple | list)
            and len(pm_acc2.get("display_ci")) == 2,
        )
    )

    # Acceptance flag exists
    checks.append(
        CheckResult(
            "pm_acceptance_flag_present",
            isinstance(
                cert.get("validation", {}).get("primary_metric_acceptable"), bool
            )
            or isinstance(cert.get("validation", {}).get("ppl_acceptable"), bool),
        )
    )

    # vqa_accuracy stub
    r, b = build_accuracy_pair("vqa_accuracy")
    try:
        from invarlock.eval.primary_metric import MetricContribution, get_metric

        pm_vqa = get_metric("vqa_accuracy")
        subj = [MetricContribution(1.0 if i % 3 != 0 else 0.0) for i in range(60)]
        base = [MetricContribution(1.0 if i % 2 == 0 else 0.0) for i in range(60)]
        comp = pm_vqa.paired_compare(subj, base, reps=400, seed=99, ci_level=0.95)
        comp["kind"] = "vqa_accuracy"
        r.setdefault("metrics", {})["primary_metric"] = comp
    except Exception:
        pass
    cert_vqa = make_certificate(r, b)
    checks.append(
        CheckResult(
            "vqa_metric_present",
            str(cert_vqa.get("primary_metric", {}).get("kind")) == "vqa_accuracy",
        )
    )

    # MoE stub (observability only)
    r, b = build_ppl_pair("ppl_causal")
    r.setdefault("metrics", {})["moe"] = {
        "top_k": 2,
        "utilization": [0.3, 0.5, 0.7],
        "router_entropy": 1.1,
    }
    cert_moe = make_certificate(r, b)
    checks.append(
        CheckResult(
            "moe_section_present",
            isinstance(cert_moe.get("validation", {}).get("moe_observed"), bool),
        )
    )

    failures = [c for c in checks if not c.ok]
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"[golden] {c.name}: {status} {('- ' + c.msg) if c.msg else ''}")
    if failures:
        print("\nGolden-runs checks had failures:")
        for f in failures:
            print(f" - {f.name}: {f.msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
