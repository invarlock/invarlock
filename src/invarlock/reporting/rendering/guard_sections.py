from __future__ import annotations

import json
import math
from typing import Any

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def _append_pairing_details(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    try:
        stats = (
            evaluation_report.get("dataset", {}).get("windows", {}).get("stats", {})
            or evaluation_report.get("ppl", {}).get("stats", {})
            or {}
        )
        paired_windows = stats.get("paired_windows")
        match_frac = stats.get("window_match_fraction")
        overlap_frac = stats.get("window_overlap_fraction")
        if any(
            value is not None for value in (paired_windows, match_frac, overlap_frac)
        ):
            parts: list[str] = []
            if paired_windows is not None:
                try:
                    parts.append(f"{int(paired_windows)} windows")
                except _PARSE_EXCEPTIONS:
                    parts.append(f"windows={paired_windows}")
            for label, value in (("match", match_frac), ("overlap", overlap_frac)):
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    parts.append(f"{float(value) * 100.0:.1f}% {label}")
                elif value is not None:
                    parts.append(f"{label}={value}")
            lines.extend(("", f"- ✅ Pairing: {', '.join(parts) if parts else 'N/A'}"))
        bootstrap = stats.get("bootstrap") or {}
        if isinstance(bootstrap, dict):
            values = (
                ("replicates", bootstrap.get("replicates")),
                ("seed", bootstrap.get("seed")),
            )
            bits: list[str] = []
            for label, value in values:
                if value is None:
                    continue
                try:
                    rendered = str(int(value))
                except _PARSE_EXCEPTIONS:
                    rendered = str(value)
                bits.append(
                    f"{rendered} replicates"
                    if label == "replicates"
                    else f"seed={rendered}"
                )
            if bits:
                lines.append(f"- ✅ Bootstrap: {', '.join(bits)}")
        delta_ci = evaluation_report.get("primary_metric", {}).get(
            "ci"
        ) or evaluation_report.get("ppl", {}).get("logloss_delta_ci")
        if (
            isinstance(delta_ci, tuple | list)
            and len(delta_ci) == 2
            and all(isinstance(x, int | float) for x in delta_ci)
        ):
            lines.append(
                f"- ℹ️ Log Δ (paired) CI: [{delta_ci[0]:.6f}, {delta_ci[1]:.6f}]"
            )
    except _PARSE_EXCEPTIONS:
        pass


def _append_invariant_notes(lines: list[str], failures: object) -> None:
    if not isinstance(failures, list) or not failures:
        return
    lines.extend(("", "**Invariant Notes**", ""))
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        detail = failure.get("detail", {})
        detail_str = ""
        if isinstance(detail, dict) and detail:
            detail_str = (
                " ("
                + ", ".join(f"{key}={value}" for key, value in detail.items())
                + ")"
            )
        lines.append(
            f"- {failure.get('check', 'unknown')} [{failure.get('severity', 'warning')}]: "
            f"{failure.get('type', 'violation')}{detail_str}"
        )


def _append_spectral_family_details(
    lines: list[str], spectral_info: dict[str, Any], top_scores: object
) -> None:
    caps_by_family = spectral_info.get("caps_applied_by_family") or {}
    quantiles = spectral_info.get("family_z_quantiles") or {}
    family_caps = spectral_info.get("family_caps") or {}
    blocks = (caps_by_family, quantiles, family_caps, top_scores)
    if not any(bool(block) for block in blocks if isinstance(block, dict)):
        return
    lines.extend(("<details>", "<summary>Per-family details</summary>", ""))
    lines.extend(
        (
            "| Family | κ | q95 | Max |z| | Caps Applied |",
            "|--------|---|-----|--------|------------|",
        )
    )
    families: set[str] = set()
    for block in blocks:
        if isinstance(block, dict):
            families.update(str(key) for key in block)
    for family in sorted(families):
        cap_record = (
            family_caps.get(family, {}) if isinstance(family_caps, dict) else {}
        )
        kappa = cap_record.get("kappa") if isinstance(cap_record, dict) else None
        kappa_str = (
            f"{float(kappa):.3f}"
            if isinstance(kappa, int | float) and math.isfinite(float(kappa))
            else "-"
        )
        stats = quantiles.get(family, {}) if isinstance(quantiles, dict) else {}
        q95 = stats.get("q95") if isinstance(stats, dict) else None
        max_z = stats.get("max") if isinstance(stats, dict) else None
        caps = caps_by_family.get(family) if isinstance(caps_by_family, dict) else None
        lines.append(
            f"| {family} | {kappa_str} | {q95:.3f}"
            if isinstance(q95, int | float)
            else f"| {family} | {kappa_str} | -"
            + (f" | {max_z:.3f}" if isinstance(max_z, int | float) else " | -")
            + f" | {int(caps) if isinstance(caps, int | float) else 0} |"
        )
    if isinstance(top_scores, dict) and top_scores:
        lines.extend(("", "Top |z| per family:"))
        for family in sorted(top_scores):
            entries = top_scores[family]
            if not isinstance(entries, list) or not entries:
                continue
            rendered: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                z_val = entry.get("z")
                z_str = (
                    f"{z_val:.3f}"
                    if isinstance(z_val, int | float) and math.isfinite(float(z_val))
                    else "n/a"
                )
                rendered.append(f"{entry.get('module', 'unknown')} (|z|={z_str})")
            lines.append(f"- {family}: {', '.join(rendered)}")
    lines.extend(("", "</details>", ""))


def append_guard_check_details_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    validation_raw = evaluation_report.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}

    lines.append("## Guard Check Details")
    lines.append("")
    lines.append("| Guard Check | Status | Measured | Threshold | Description |")
    lines.append("|--------------|--------|----------|-----------|-------------|")

    inv_summary = evaluation_report.get("invariants", {}) or {}
    inv_status = "✅ PASS" if validation.get("invariants_pass", False) else "❌ FAIL"
    inv_counts = inv_summary.get("summary", {}) if isinstance(inv_summary, dict) else {}
    if not isinstance(inv_counts, dict):
        inv_counts = {}
    inv_measure = (
        str(inv_summary.get("status", "not recorded")).upper()
        if isinstance(inv_summary, dict)
        else "NOT RECORDED"
    )
    fatal_violations = inv_counts.get("fatal_violations") or 0
    warning_violations = (
        inv_counts.get("warning_violations") or inv_counts.get("violations_found") or 0
    )
    if fatal_violations:
        suffix = f"{fatal_violations} fatal"
        if warning_violations:
            suffix += f", {warning_violations} warning"
        inv_measure = f"{inv_measure} ({suffix})"
    elif warning_violations:
        inv_measure = f"{inv_measure} ({warning_violations} warning)"
    lines.append(
        f"| Invariants | {inv_status} | {inv_measure} | pass | Model integrity checks |"
    )
    invariants_failures = (
        inv_summary.get("failures") if isinstance(inv_summary, dict) else []
    ) or []
    if warning_violations and not fatal_violations:
        non_fatal_message = None
        for failure in invariants_failures:
            if isinstance(failure, dict):
                msg = failure.get("message") or failure.get("type")
                if msg:
                    non_fatal_message = msg
                    break
        if not non_fatal_message:
            non_fatal_message = "Non-fatal invariant warnings present."
        lines.append(f"- Non-fatal: {non_fatal_message}")

    spec_status = "✅ PASS" if validation.get("spectral_stable", False) else "❌ FAIL"
    spectral_summary = evaluation_report.get("spectral", {}) or {}
    caps_applied = (
        spectral_summary.get("caps_applied")
        if isinstance(spectral_summary, dict)
        else None
    )
    caps_measure = f"{caps_applied} caps applied" if caps_applied is not None else "N/A"
    spectral_threshold = (
        f"<= {spectral_summary.get('max_caps')}"
        if isinstance(spectral_summary, dict)
        and spectral_summary.get("max_caps") is not None
        else "<= 5"
    )
    lines.append(
        f"| Spectral Stability | {spec_status} | {caps_measure} | {spectral_threshold} | Weight matrix spectral norms |"
    )

    if isinstance(evaluation_report.get("primary_metric"), dict):
        pm_value = validation.get("primary_metric_acceptable")
        pm_status = (
            "✅ PASS"
            if pm_value is True
            else "❌ FAIL"
            if pm_value is False
            else "ℹ️ NOT EVALUATED"
        )
        primary_metric = evaluation_report.get("primary_metric", {})
        pm_kind = str(primary_metric.get("kind") or "").strip().lower()
        pm_ratio = (
            primary_metric.get("ratio_vs_baseline")
            if pm_kind.startswith("ppl")
            else None
        )
        if isinstance(pm_ratio, int | float):
            lines.append(
                f"| Catastrophic Spike Gate (hard stop) | {pm_status} | {pm_ratio:.3f}x | ≤ 2.0x | Hard stop @ 2.0× |"
            )

    rmt_status = "✅ PASS" if validation.get("rmt_stable", False) else "❌ FAIL"
    rmt_state = evaluation_report.get("rmt", {}).get("status", "unknown").title()
    lines.append(
        f"| RMT Health | {rmt_status} | {rmt_state} | ε-rule | Random Matrix Theory guard status |"
    )

    _append_pairing_details(lines, evaluation_report)
    _append_invariant_notes(lines, invariants_failures)

    lines.append("")


def _append_spectral_observability(
    lines: list[str],
    *,
    evaluation_report: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    spectral_info = evaluation_report.get("spectral", {}) or {}
    if not spectral_info:
        return
    lines.append("### Spectral Guard Summary")
    lines.append("")
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")

    spectral_ok = validation.get("spectral_stable") is True
    caps_applied = spectral_info.get("caps_applied")
    max_caps = spectral_info.get("max_caps")
    caps_val = (
        f"{caps_applied}/{max_caps}"
        if caps_applied is not None and max_caps is not None
        else "-"
    )
    lines.append(
        f"| Caps Applied | {caps_val} | {'✅ OK' if spectral_ok else '❌ FAIL'} |"
    )

    summary = spectral_info.get("summary", {}) or {}
    caps_exceeded = summary.get("caps_exceeded")
    if caps_exceeded is not None:
        cap_status = "✅ OK" if not bool(caps_exceeded) else "⚠️ WARN"
        lines.append(f"| Caps Exceeded | {caps_exceeded} | {cap_status} |")

    top_scores = spectral_info.get("top_z_scores") or {}
    max_family: str | None = None
    max_module: str | None = None
    max_abs_z: float | None = None
    if isinstance(top_scores, dict):
        for family, entries in top_scores.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                z_val = entry.get("z")
                if not (
                    isinstance(z_val, (int, float)) and math.isfinite(float(z_val))
                ):
                    continue
                z_abs = abs(float(z_val))
                if max_abs_z is None or z_abs > max_abs_z:
                    max_abs_z = z_abs
                    max_family = str(family)
                    max_module = (
                        str(entry.get("module")) if entry.get("module") else None
                    )

    family_caps = spectral_info.get("family_caps") or {}
    kappa = None
    if max_family and isinstance(family_caps, dict):
        try:
            kappa = (family_caps.get(max_family, {}) or {}).get("kappa")
        except _PARSE_EXCEPTIONS:
            kappa = None
    kappa_f = (
        float(kappa)
        if isinstance(kappa, (int, float)) and math.isfinite(float(kappa))
        else None
    )
    if max_abs_z is not None:
        max_val = f"{max_abs_z:.3f}"
        if max_family:
            max_val += f" ({max_family})"
        if max_module:
            max_val += f" – {max_module}"
        if kappa_f is None:
            max_status = "ℹ️ No κ"
        elif max_abs_z <= kappa_f:
            max_status = f"✅ Within κ={kappa_f:.3f}"
        else:
            max_status = f"⚠️ Above κ={kappa_f:.3f}"
        lines.append(f"| Max |z| | {max_val} | {max_status} |")

    mt_info = spectral_info.get("multiple_testing", {}) or {}
    if isinstance(mt_info, dict) and mt_info:
        parts: list[str] = []
        mt_method = mt_info.get("method")
        mt_alpha = mt_info.get("alpha")
        mt_m = mt_info.get("m")
        if mt_method:
            parts.append(f"method={mt_method}")
        if isinstance(mt_alpha, (int, float)) and math.isfinite(float(mt_alpha)):
            parts.append(f"α={float(mt_alpha):.3g}")
        if isinstance(mt_m, (int, float)) and math.isfinite(float(mt_m)):
            parts.append(f"m={int(mt_m)}")
        lines.append(
            f"| Multiple Testing | {', '.join(parts) if parts else '—'} | ℹ️ INFO |"
        )

    lines.append("")
    _append_spectral_family_details(lines, spectral_info, top_scores)


def _append_rmt_observability(
    lines: list[str],
    *,
    evaluation_report: dict[str, Any],
) -> None:
    rmt_info_raw = evaluation_report.get("rmt", {}) or {}
    rmt_info: dict[str, Any] = rmt_info_raw if isinstance(rmt_info_raw, dict) else {}
    if not rmt_info:
        return
    lines.append("### RMT Guard")
    lines.append("")
    raw_rmt_families = rmt_info.get("families")
    rmt_families: dict[str, Any] = (
        raw_rmt_families if isinstance(raw_rmt_families, dict) else {}
    )
    stable = rmt_info.get("stable")
    status = (
        "✅ OK"
        if stable is True
        else "❌ FAIL"
        if stable is False
        else "ℹ️ NOT EVALUATED"
    )
    mode = rmt_info.get("mode")
    if isinstance(mode, str) and mode.strip():
        lines.append(f"- Mode: `{mode.strip()}`")
    measurement_contract = (
        rmt_info.get("measurement_contract")
        if isinstance(rmt_info.get("measurement_contract"), dict)
        else {}
    )
    if measurement_contract:
        contract_parts: list[str] = []
        estimator = measurement_contract.get("estimator")
        if isinstance(estimator, dict) and estimator:
            contract_parts.append(
                f"estimator={json.dumps(estimator, sort_keys=True, allow_nan=False)}"
            )
        activation_sampling = measurement_contract.get("activation_sampling")
        if isinstance(activation_sampling, dict) and activation_sampling:
            contract_parts.append(
                "activation_sampling="
                + json.dumps(activation_sampling, sort_keys=True, allow_nan=False)
            )
        if contract_parts:
            lines.append(f"- Measurement Contract: {'; '.join(contract_parts)}")
    delta_total = rmt_info.get("delta_total")
    if isinstance(delta_total, int):
        lines.append(f"- Δ total: {delta_total:+d}")
    lines.append(f"- Status: {status}")
    lines.append(f"- Families: {len(rmt_families)}")
    if not rmt_families:
        lines.append("")
        return
    edge_risk_mode = any(
        isinstance(data, dict) and ("edge_base" in data or "edge_cur" in data)
        for data in rmt_families.values()
    )
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>RMT family details</summary>")
    lines.append("")
    if edge_risk_mode:
        lines.append("| Family | ε_f | Edge Base | Edge Cur | Δ |")
        lines.append("|--------|-----|-----------|----------|---|")
    else:
        lines.append("| Family | ε_f | Bare | Guarded | Δ |")
        lines.append("|--------|-----|------|---------|---|")
    for family, data in rmt_families.items():
        epsilon_val = data.get("epsilon")
        epsilon_str = (
            f"{epsilon_val:.3f}" if isinstance(epsilon_val, (int, float)) else "-"
        )
        if edge_risk_mode:
            edge_base = data.get("edge_base")
            edge_cur = data.get("edge_cur")
            delta_val = data.get("delta")
            edge_base_str = (
                f"{edge_base:.3f}" if isinstance(edge_base, (int, float)) else "-"
            )
            edge_cur_str = (
                f"{edge_cur:.3f}" if isinstance(edge_cur, (int, float)) else "-"
            )
            delta_str = (
                f"{delta_val:+.3f}" if isinstance(delta_val, (int, float)) else "-"
            )
            lines.append(
                f"| {family} | {epsilon_str} | {edge_base_str} | {edge_cur_str} | {delta_str} |"
            )
            continue
        bare_count = data.get("bare", 0)
        guarded_count = data.get("guarded", 0)
        try:
            bare_str = str(int(bare_count))
        except _PARSE_EXCEPTIONS:
            bare_str = "-"
        try:
            guarded_str = str(int(guarded_count))
        except _PARSE_EXCEPTIONS:
            guarded_str = "-"
        try:
            delta_count = int(guarded_count) - int(bare_count)
        except _PARSE_EXCEPTIONS:
            delta_count = None
        delta_str = f"{delta_count:+d}" if isinstance(delta_count, int) else "-"
        lines.append(
            f"| {family} | {epsilon_str} | {bare_str} | {guarded_str} | {delta_str} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")


def _append_guard_metric_impact_observability(
    lines: list[str],
    *,
    evaluation_report: dict[str, Any],
) -> None:
    guard_metric_impact_info = evaluation_report.get("guard_metric_impact", {}) or {}
    if not guard_metric_impact_info:
        return
    lines.append("### Guard Metric Impact")
    lines.append("")
    evaluated_flag = guard_metric_impact_info.get("evaluated") is True
    if not evaluated_flag:
        lines.append("- Evaluated: false or not recorded")
    metric_kind = guard_metric_impact_info.get("metric_kind")
    if isinstance(metric_kind, str) and metric_kind:
        lines.append(f"- Metric Kind: {metric_kind}")
    bare_value = guard_metric_impact_info.get("bare_value")
    guarded_value = guard_metric_impact_info.get("guarded_value")
    if (
        isinstance(bare_value, (int, float))
        and not isinstance(bare_value, bool)
        and math.isfinite(float(bare_value))
    ):
        lines.append(f"- Bare Primary Metric: {bare_value:.3f}")
    if (
        isinstance(guarded_value, (int, float))
        and not isinstance(guarded_value, bool)
        and math.isfinite(float(guarded_value))
    ):
        lines.append(f"- Guarded Primary Metric: {guarded_value:.3f}")
    display_value = guard_metric_impact_info.get("display_value")
    display_unit = guard_metric_impact_info.get("display_unit")
    if (
        isinstance(display_value, (int, float))
        and not isinstance(display_value, bool)
        and math.isfinite(float(display_value))
    ):
        if display_unit == "percent":
            lines.append(f"- Degradation: {display_value:+.2f}%")
        elif display_unit == "percentage_points":
            lines.append(f"- Degradation: {display_value:+.2f} pp")
    metric_impact_source = guard_metric_impact_info.get("source")
    if metric_impact_source:
        lines.append(f"- Source: {metric_impact_source}")
    plan_ctx = evaluation_report.get("provenance", {}).get("window_plan", {})
    if isinstance(plan_ctx, dict) and plan_ctx:
        plan_preview = (
            plan_ctx.get("preview_n")
            if plan_ctx.get("preview_n") is not None
            else plan_ctx.get("actual_preview")
        )
        plan_final = (
            plan_ctx.get("final_n")
            if plan_ctx.get("final_n") is not None
            else plan_ctx.get("actual_final")
        )
        plan_profile = plan_ctx.get("profile")
        lines.append(
            f"- Window Plan Used: profile={plan_profile}, preview={plan_preview}, final={plan_final}"
        )
    lines.append("")


def append_guard_observability_sections(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    validation_raw = evaluation_report.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}

    lines.append("## Guard Observability")
    lines.append("")
    _append_spectral_observability(
        lines,
        evaluation_report=evaluation_report,
        validation=validation,
    )
    _append_rmt_observability(lines, evaluation_report=evaluation_report)
    _append_guard_metric_impact_observability(
        lines, evaluation_report=evaluation_report
    )


def append_guard_warnings_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    guard_warnings = evaluation_report.get("guard_warnings")
    if not isinstance(guard_warnings, dict):
        return
    warnings = guard_warnings.get("warnings")
    if not isinstance(warnings, list) or not warnings:
        return
    lines.append("## Guard Warnings")
    lines.append("")
    lines.append(
        "Policy can still pass with guard warnings; these are baseline-relative guard-signal changes, not hard policy failures unless strict warning mode is enabled."
    )
    lines.append("")
    lines.append("| Guard | Kind | Location | Policy | Detail |")
    lines.append("|-------|------|----------|--------|--------|")
    for entry_raw in warnings:
        if not isinstance(entry_raw, dict):
            continue
        guard = entry_raw.get("guard", "guard")
        kind = entry_raw.get("kind", "warning")
        family = entry_raw.get("family")
        module = entry_raw.get("module")
        location_parts = []
        if family:
            location_parts.append(str(family))
        if module:
            location_parts.append(str(module))
        location = " / ".join(location_parts) if location_parts else "-"
        policy = entry_raw.get("policy_gate", "unknown")
        message = str(
            entry_raw.get("message") or "Guard signal changed versus baseline."
        )
        lines.append(f"| {guard} | {kind} | {location} | {policy} | {message} |")
    lines.append("")
