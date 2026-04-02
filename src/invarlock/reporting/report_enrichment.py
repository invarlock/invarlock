from __future__ import annotations

import math
from typing import Any

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


def _sanitize_summary_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = "".join(
        ch if ch.isprintable() and ch not in "\r\n\t" else " " for ch in str(value)
    )
    normalized = " ".join(cleaned.split())
    return normalized or None


def attach_quality_overhead(
    evaluation_report: dict[str, Any],
    raw_guard_ctx: Any,
    report: dict[str, Any],
    compute_quality_overhead_from_guard_fn: Any,
) -> None:
    try:
        pm_kind_hint = None
        try:
            pm_try = (
                report.get("metrics", {}).get("primary_metric")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if isinstance(pm_try, dict):
                pm_kind_hint = pm_try.get("kind")
        except _NON_FATAL_EXCEPTIONS:
            pm_kind_hint = None
        quality_overhead = compute_quality_overhead_from_guard_fn(
            raw_guard_ctx, pm_kind_hint
        )
        if (
            isinstance(quality_overhead, dict)
            and "value" in quality_overhead
            and math.isfinite(float(quality_overhead.get("value", float("nan"))))
        ):
            evaluation_report["quality_overhead"] = quality_overhead
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_policy_digest(
    evaluation_report: dict[str, Any],
    auto: dict[str, Any],
    resolved_policy: dict[str, Any],
    baseline_raw: Any,
    baseline_normalized: Any,
    compute_thresholds_payload_fn: Any,
    compute_thresholds_hash_fn: Any,
    policy_version: str,
) -> None:
    try:
        cur_tier = str(auto.get("tier", "balanced")).lower()
    except _NON_FATAL_EXCEPTIONS:
        cur_tier = "balanced"
    thresholds_payload = compute_thresholds_payload_fn(cur_tier, resolved_policy)
    thresholds_hash = compute_thresholds_hash_fn(thresholds_payload)
    base_tier = None
    try:
        if isinstance(baseline_raw, dict):
            baseline_meta = baseline_raw.get("meta")
            if isinstance(baseline_meta, dict):
                baseline_auto = baseline_meta.get("auto")
                if isinstance(baseline_auto, dict) and baseline_auto.get("tier"):
                    base_tier = str(baseline_auto.get("tier")).lower()
        if base_tier is None and isinstance(baseline_normalized, dict):
            base_meta = baseline_normalized.get("meta")
            if isinstance(base_meta, dict):
                base_auto = base_meta.get("auto")
                if isinstance(base_auto, dict) and base_auto.get("tier"):
                    base_tier = str(base_auto.get("tier")).lower()
    except _NON_FATAL_EXCEPTIONS:
        base_tier = None
    baseline_payload = compute_thresholds_payload_fn(
        base_tier or cur_tier, resolved_policy
    )
    baseline_hash = compute_thresholds_hash_fn(baseline_payload)
    changed = bool(
        (base_tier is not None and base_tier != cur_tier)
        or (baseline_hash != thresholds_hash)
    )

    metrics_policy = (
        resolved_policy.get("metrics", {}) if isinstance(resolved_policy, dict) else {}
    )
    if not isinstance(metrics_policy, dict):
        metrics_policy = {}
    ppl_hys = 0.0
    acc_hys = 0.0
    try:
        ppl_hys = float(
            (metrics_policy.get("pm_ratio") or {}).get("hysteresis_ratio", 0.0) or 0.0
        )
        acc_hys = float(
            (metrics_policy.get("accuracy") or {}).get("hysteresis_delta_pp", 0.0)
            or 0.0
        )
    except _NON_FATAL_EXCEPTIONS:
        pass
    min_effective = float(
        (resolved_policy.get("variance") or {}).get("min_effect_lognll", 0.0) or 0.0
    )

    evaluation_report["policy_digest"] = {
        "policy_version": policy_version,
        "tier_policy_name": cur_tier,
        "thresholds_hash": thresholds_hash,
        "hysteresis": {"ppl": ppl_hys, "accuracy_delta_pp": acc_hys},
        "min_effective": min_effective,
        "changed": changed,
    }


def attach_secondary_metrics(
    evaluation_report: dict[str, Any], report: dict[str, Any]
) -> None:
    try:
        if isinstance(report.get("metrics"), dict):
            sec = report["metrics"].get("secondary_metrics")
            if isinstance(sec, list) and sec:
                sanitized: list[dict[str, Any]] = []
                for item in sec:
                    if isinstance(item, dict) and item.get("kind"):
                        payload: dict[str, Any] = {}
                        for key in (
                            "kind",
                            "preview",
                            "final",
                            "ratio_vs_baseline",
                            "unit",
                            "display_ci",
                            "ci",
                        ):
                            if key in item:
                                payload[key] = item[key]
                        sanitized.append(payload)
                if sanitized:
                    evaluation_report["secondary_metrics"] = sanitized
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_classification(
    evaluation_report: dict[str, Any], report: dict[str, Any]
) -> None:
    try:
        cls = (
            report.get("metrics", {}).get("classification")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(cls, dict):
            sub = cls.get("subgroups")
            if isinstance(sub, dict) and all(
                key in sub for key in ("preview", "final")
            ):
                prev = sub.get("preview", {})
                fin = sub.get("final", {})
                pc = prev.get("group_counts", {}) if isinstance(prev, dict) else {}
                pcc = prev.get("correct_counts", {}) if isinstance(prev, dict) else {}
                fc = fin.get("group_counts", {}) if isinstance(fin, dict) else {}
                fcc = fin.get("correct_counts", {}) if isinstance(fin, dict) else {}
                out: dict[str, Any] = {}
                labels = set(list(pc.keys()) + list(fc.keys()))
                for label in labels:
                    try:
                        nprev = float(pc.get(label, 0))
                        nfin = float(fc.get(label, 0))
                        acc_prev = (
                            float(pcc.get(label, 0)) / nprev
                            if nprev > 0
                            else float("nan")
                        )
                        acc_fin = (
                            float(fcc.get(label, 0)) / nfin
                            if nfin > 0
                            else float("nan")
                        )
                        delta_pp = (
                            (acc_fin - acc_prev) * 100.0
                            if (math.isfinite(acc_prev) and math.isfinite(acc_fin))
                            else float("nan")
                        )
                        out[str(label)] = {
                            "preview": acc_prev,
                            "final": acc_fin,
                            "delta_pp": delta_pp,
                            "n_preview": nprev,
                            "n_final": nfin,
                        }
                    except _NON_FATAL_EXCEPTIONS:
                        continue
                if out:
                    evaluation_report["classification"] = {"subgroups": out}
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_system_overhead(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    baseline_raw: Any,
    telemetry: dict[str, Any],
) -> None:
    try:

        def _extract_sys_metrics(
            container: dict[str, Any] | None,
            *,
            fallback_telemetry: dict[str, Any] | None = None,
        ) -> dict[str, float]:
            out: dict[str, float] = {}
            if not isinstance(container, dict):
                return out
            metrics = (
                container.get("metrics", {})
                if isinstance(container.get("metrics"), dict)
                else {}
            )
            telem = fallback_telemetry if isinstance(fallback_telemetry, dict) else {}
            for key in ("latency_ms_p50", "latency_ms_p95", "throughput_sps"):
                val = metrics.get(key)
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out[key] = float(val)
            if "latency_ms_p50" not in out:
                val = metrics.get("latency_ms_per_tok") or telem.get(
                    "latency_ms_per_tok"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["latency_ms_p50"] = float(val)
            if "throughput_sps" not in out:
                val = metrics.get("throughput_tok_per_s") or telem.get(
                    "throughput_tok_per_s"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["throughput_sps"] = float(val)
            return out

        edited_sys = _extract_sys_metrics(report, fallback_telemetry=telemetry)
        base_sys = _extract_sys_metrics(
            baseline_raw if isinstance(baseline_raw, dict) else None,
            fallback_telemetry=None,
        )
        system_overhead: dict[str, Any] = {}
        for metric_key, edited_val in edited_sys.items():
            base_val = base_sys.get(metric_key)
            entry: dict[str, Any] = {"edited": edited_val}
            if isinstance(base_val, int | float) and math.isfinite(float(base_val)):
                entry["baseline"] = float(base_val)
                entry["delta"] = float(edited_val - base_val)
                try:
                    entry["ratio"] = (
                        float(edited_val / base_val) if base_val != 0 else float("nan")
                    )
                except _NON_FATAL_EXCEPTIONS:
                    entry["ratio"] = float("nan")
            system_overhead[metric_key] = entry
        if system_overhead:
            evaluation_report["system_overhead"] = system_overhead
    except _NON_FATAL_EXCEPTIONS:
        pass


def ensure_primary_metric_display_ci(evaluation_report: dict[str, Any]) -> None:
    try:
        pm = (
            evaluation_report.get("primary_metric", {})
            if isinstance(evaluation_report.get("primary_metric"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
            disp = pm.get("display_ci")
            if not (
                isinstance(disp, list | tuple)
                and len(disp) == 2
                and all(isinstance(x, int | float) for x in disp)
            ):
                point = None
                for key in ("ratio_vs_baseline", "final", "preview"):
                    val = pm.get(key)
                    if isinstance(val, int | float) and math.isfinite(float(val)):
                        point = float(val)
                        break
                if isinstance(point, float):
                    pm["display_ci"] = [point, point]
                else:
                    pm["display_ci"] = [1.0, 1.0]
                    pm.setdefault("estimated", True)
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_telemetry_summary_line(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    current_run_id: str,
) -> None:
    try:
        kind = None
        pm_try = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(pm_try, dict):
            kind = pm_try.get("kind")
        if not kind:
            kind = "ppl"
        kind_text = _sanitize_summary_value(kind) or "ppl"
        run_id_text = _sanitize_summary_value(current_run_id) or "unknown"
        windows_cfg = (
            evaluation_report.get("dataset", {}).get("windows", {})
            if isinstance(evaluation_report.get("dataset"), dict)
            else {}
        )
        n_prev = windows_cfg.get("preview")
        n_fin = windows_cfg.get("final")
        tokens_total = None
        try:
            tokens_total = (
                evaluation_report.get("dataset", {}).get("hash", {}).get("total_tokens")
            )
        except _NON_FATAL_EXCEPTIONS:
            tokens_total = None
        ci_lo = None
        ci_hi = None
        ratio = None
        pmc = evaluation_report.get("primary_metric", {})
        rci = pmc.get("display_ci") or pmc.get("ci")
        if isinstance(rci, tuple | list) and len(rci) == 2:
            ci_lo, ci_hi = rci[0], rci[1]
        ratio = pmc.get("ratio_vs_baseline")
        ci_w = None
        try:
            if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
                ci_w = float(ci_hi) - float(ci_lo)
        except _NON_FATAL_EXCEPTIONS:
            ci_w = None
        val = evaluation_report.get("validation", {})
        gate_ok = None
        try:
            gate_ok = bool(val.get("primary_metric_acceptable"))
        except _NON_FATAL_EXCEPTIONS:
            gate_ok = None
        parts = [
            f"run_id={run_id_text}",
            f"metric={kind_text}",
            f"nprev={n_prev}",
            f"nfinal={n_fin}",
            f"tokens={tokens_total}",
        ]
        try:
            split = (evaluation_report.get("provenance", {}) or {}).get("dataset_split")
            if not split:
                split = (report.get("provenance", {}) or {}).get("dataset_split")
            split_fallback = (evaluation_report.get("provenance", {}) or {}).get(
                "split_fallback"
            )
            if split_fallback is None:
                split_fallback = (report.get("provenance", {}) or {}).get(
                    "split_fallback"
                )
            split_text = _sanitize_summary_value(split)
            if split_text:
                parts.append(f"split={split_text}{'*' if split_fallback else ''}")
        except _NON_FATAL_EXCEPTIONS:
            pass
        if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
            parts.append(f"ci={ci_lo:.3f}-{ci_hi:.3f}")
            if isinstance(ci_w, int | float):
                parts.append(f"width={ci_w:.3f}")
        if isinstance(ratio, int | float):
            parts.append(f"ratio={float(ratio):.3f}")
        if isinstance(gate_ok, bool):
            parts.append(f"gate={'pass' if gate_ok else 'fail'}")
        summary_line = "INVARLOCK_TELEMETRY " + " ".join(parts)
        evaluation_report.setdefault("telemetry", {})["summary_line"] = summary_line
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_confidence_label(
    evaluation_report: dict[str, Any], compute_confidence_label_fn: Any
) -> None:
    try:
        evaluation_report["confidence"] = compute_confidence_label_fn(evaluation_report)
    except _NON_FATAL_EXCEPTIONS:
        pass
