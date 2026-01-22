from __future__ import annotations

import json
import math
from pathlib import Path

# mypy: ignore-errors
from typing import Any

import yaml

# Import certificate module for helper access without creating hard cycles
from . import certificate as C

# Console Validation Block helpers (allow-list driven)
_CONSOLE_LABELS_DEFAULT = [
    "Primary Metric Acceptable",
    "Preview Final Drift Acceptable",
    "Guard Overhead Acceptable",
    "Invariants Pass",
    "Spectral Stable",
    "Rmt Stable",
]


def _load_console_labels() -> list[str]:
    """Load console labels allow-list from contracts with a safe fallback."""
    try:
        root = Path(__file__).resolve().parents[3]
        path = root / "contracts" / "console_labels.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return list(data)
    except Exception:
        pass
    return list(_CONSOLE_LABELS_DEFAULT)


def compute_console_validation_block(certificate: dict[str, Any]) -> dict[str, Any]:
    """Produce a normalized console validation block from a certificate.

    Returns a dict with keys:
    - labels: the canonical label list
    - rows: list of {label, status, evaluated, ok}
    - overall_pass: boolean computed from canonical rows only. Guard Overhead is
      counted only when evaluated.
    """
    labels = _load_console_labels()
    validation = certificate.get("validation", {}) or {}
    guard_ctx = certificate.get("guard_overhead", {}) or {}
    guard_evaluated = (
        bool(guard_ctx.get("evaluated")) if isinstance(guard_ctx, dict) else False
    )

    # Map label → validation key
    def _to_key(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    rows: list[dict[str, Any]] = []
    ok_map: dict[str, bool] = {}
    effective_labels: list[str] = []
    for label in labels:
        key = _to_key(label)
        ok = bool(validation.get(key, False))
        status: str
        evaluated = True
        if key == "guard_overhead_acceptable":
            evaluated = guard_evaluated
            if not evaluated:
                # Omit row entirely when not evaluated (policy/profile skipped)
                continue
            status = "✅ PASS" if ok else "❌ FAIL"
        else:
            status = "✅ PASS" if ok else "❌ FAIL"
        rows.append(
            {"label": label, "status": status, "evaluated": evaluated, "ok": ok}
        )
        effective_labels.append(label)
        ok_map[key] = ok

    # Overall policy from canonical rows only; exclude guard when not evaluated
    keys_for_overall = [
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    ]
    # Include guard overhead only if evaluated
    if guard_evaluated:
        keys_for_overall.append("guard_overhead_acceptable")

    overall_pass = all(ok_map.get(k, False) for k in keys_for_overall)

    return {"labels": effective_labels, "rows": rows, "overall_pass": overall_pass}


def _format_plugin(plugin: dict[str, Any]) -> str:
    """Format a plugin entry for markdown list rendering."""
    name = plugin.get("name", "unknown")
    version = plugin.get("version") or "-"
    module = plugin.get("module") or "unknown"
    entry = plugin.get("entry_point")
    pieces = [f"**{name}** v{version}", f"`{module}`"]
    if entry:
        pieces.append(f"[{entry}]")
    return " ".join(pieces)


def _short_digest(v: str) -> str:
    v = str(v)
    return v if len(v) <= 16 else (v[:8] + "…" + v[-8:])


def _render_executive_dashboard(cert: dict[str, Any]) -> str:
    """Render executive summary dashboard table."""
    lines: list[str] = []
    _append_safety_dashboard_section(lines, cert)
    return "\n".join(lines).rstrip()


def _append_safety_dashboard_section(
    lines: list[str], certificate: dict[str, Any]
) -> None:
    """Append a concise, first-screen dashboard for the certificate."""
    block = compute_console_validation_block(certificate)
    overall_pass = bool(block.get("overall_pass"))
    overall_status = (
        f"{'✅' if overall_pass else '❌'} {'PASS' if overall_pass else 'FAIL'}"
    )

    validation = certificate.get("validation", {}) or {}
    pm = certificate.get("primary_metric", {}) or {}
    auto = certificate.get("auto", {}) or {}
    tier = str(auto.get("tier") or "balanced").lower()

    # Primary metric summary
    pm_kind = str(pm.get("kind", "")).lower()
    pm_basis = pm.get("gating_basis") or pm.get("basis") or "point"
    pm_ok: bool | None
    if isinstance(validation, dict) and "primary_metric_acceptable" in validation:
        pm_ok = bool(validation.get("primary_metric_acceptable"))
    else:
        pm_ok = None
    pm_value = pm.get("ratio_vs_baseline")

    if pm_kind in {"accuracy", "vqa_accuracy"}:
        measured = f"{pm_value:+.2f} pp" if isinstance(pm_value, int | float) else "N/A"
        th_map = {
            "conservative": -0.5,
            "balanced": -1.0,
            "aggressive": -2.0,
            "none": -1.0,
        }
        th = th_map.get(tier, -1.0)
        threshold = f"≥ {th:+.2f} pp ({pm_basis})"
    else:
        measured = f"{pm_value:.3f}×" if isinstance(pm_value, int | float) else "N/A"
        tier_thresholds = {
            "conservative": 1.05,
            "balanced": 1.10,
            "aggressive": 1.20,
            "none": 1.10,
        }
        ratio_limit = tier_thresholds.get(tier, 1.10)
        target_ratio = auto.get("target_pm_ratio")
        if isinstance(target_ratio, int | float) and target_ratio > 0:
            ratio_limit = min(ratio_limit, float(target_ratio))
        threshold = f"≤ {ratio_limit:.2f}× ({pm_basis})"

    pm_status = (
        f"{'✅' if pm_ok else '❌'} {measured}"
        if isinstance(pm_ok, bool)
        else f"🛈 {measured}"
    )

    # Drift summary (final/preview ratio) when preview/final are numeric
    drift_ok: bool | None
    if isinstance(validation, dict) and "preview_final_drift_acceptable" in validation:
        drift_ok = bool(validation.get("preview_final_drift_acceptable"))
    else:
        drift_ok = None
    drift_val = "N/A"
    try:
        pv = (
            float(pm.get("preview"))
            if isinstance(pm.get("preview"), int | float)
            else float("nan")
        )
        fv = (
            float(pm.get("final"))
            if isinstance(pm.get("final"), int | float)
            else float("nan")
        )
        drift = (
            fv / pv
            if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
            else float("nan")
        )
        if math.isfinite(drift):
            drift_val = f"{drift:.3f}×"
    except Exception:
        drift_val = "N/A"
    drift_status = (
        f"{'✅' if drift_ok else '❌'} {drift_val}"
        if isinstance(drift_ok, bool)
        else f"🛈 {drift_val}"
    )

    def _gate_cell(key: str, ok_default: bool | None = None) -> str:
        ok: bool | None
        if not isinstance(validation, dict):
            ok = ok_default
        elif key not in validation:
            ok = ok_default
        else:
            ok = bool(validation.get(key))
        if ok is None:
            return "🛈 N/A"
        return "✅ PASS" if ok else "❌ FAIL"

    overhead_ctx = certificate.get("guard_overhead", {}) or {}
    overhead_evaluated = (
        bool(overhead_ctx.get("evaluated")) if isinstance(overhead_ctx, dict) else False
    )
    overhead_row: tuple[str, str, str] | None = None
    if overhead_evaluated:
        overhead_pct = overhead_ctx.get("overhead_percent")
        overhead_ratio = overhead_ctx.get("overhead_ratio")
        if isinstance(overhead_pct, int | float) and math.isfinite(float(overhead_pct)):
            overhead_measured = f"{float(overhead_pct):+.2f}%"
        elif isinstance(overhead_ratio, int | float) and math.isfinite(
            float(overhead_ratio)
        ):
            overhead_measured = f"{float(overhead_ratio):.3f}×"
        else:
            overhead_measured = "N/A"
        threshold_pct = overhead_ctx.get("threshold_percent")
        if isinstance(threshold_pct, int | float) and math.isfinite(
            float(threshold_pct)
        ):
            threshold_str = f"≤ +{float(threshold_pct):.1f}%"
        else:
            threshold_str = "≤ +1.0%"
        overhead_row = (
            "Overhead",
            f"{'✅' if bool(validation.get('guard_overhead_acceptable', True)) else '❌'} {overhead_measured}"
            if isinstance(validation, dict)
            else f"🛈 {overhead_measured}",
            threshold_str,
        )

    lines.append("## Safety Dashboard")
    lines.append("")
    lines.append("| Check | Status | Quick Summary |")
    lines.append("|-------|--------|---------------|")
    lines.append(f"| Overall | {overall_status} | Canonical gate outcomes |")
    lines.append(f"| Primary Metric | {pm_status} | {threshold} |")
    lines.append(f"| Drift | {drift_status} | 0.95–1.05× band |")
    lines.append(
        f"| Invariants | {_gate_cell('invariants_pass')} | Model integrity checks |"
    )
    lines.append(
        f"| Spectral | {_gate_cell('spectral_stable')} | Weight matrix spectral norms |"
    )
    lines.append(f"| RMT | {_gate_cell('rmt_stable')} | Random Matrix Theory guard |")
    if overhead_row:
        lines.append(f"| {overhead_row[0]} | {overhead_row[1]} | {overhead_row[2]} |")
    lines.append("")


def _append_primary_metric_section(
    lines: list[str], certificate: dict[str, Any]
) -> None:
    """Append the Primary Metric section early for quick triage."""
    pm = certificate.get("primary_metric")
    if not isinstance(pm, dict) or not pm:
        return

    kind = pm.get("kind", "unknown")
    lines.append("## Primary Metric")
    lines.append("")
    unit = pm.get("unit", "-")
    paired = pm.get("paired", False)

    estimated_flag = False
    try:
        if bool(pm.get("estimated")):
            estimated_flag = True
        elif str(pm.get("counts_source", "")).lower() == "pseudo_config":
            estimated_flag = True
    except Exception:
        estimated_flag = False
    est_suffix = " (estimated)" if estimated_flag else ""

    lines.append(f"- Kind: {kind} (unit: {unit}){est_suffix}")
    gating_basis = pm.get("gating_basis") or pm.get("basis")
    if gating_basis:
        lines.append(f"- Basis: {gating_basis}")
    if isinstance(paired, bool):
        lines.append(f"- Paired: {paired}")
    reps = pm.get("reps")
    if isinstance(reps, int | float):
        lines.append(f"- Bootstrap Reps: {int(reps)}")
    ci = pm.get("ci") or pm.get("display_ci")
    if (
        isinstance(ci, list | tuple)
        and len(ci) == 2
        and all(isinstance(x, int | float) for x in ci)
    ):
        lines.append(f"- CI: {ci[0]:.3f}–{ci[1]:.3f}")

    prev = pm.get("preview")
    fin = pm.get("final")
    ratio = pm.get("ratio_vs_baseline")

    lines.append("")
    if estimated_flag and str(kind).lower() in {"accuracy", "vqa_accuracy"}:
        lines.append(
            "- Note: Accuracy derived from pseudo counts (quick dev preset); use a labeled preset for measured accuracy."
        )
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Preview | {_fmt_by_kind(prev, str(kind))} |")
    lines.append(f"| Final | {_fmt_by_kind(fin, str(kind))} |")

    if kind in {"accuracy", "vqa_accuracy"}:
        lines.append(f"| Δ vs Baseline | {_fmt_by_kind(ratio, str(kind))} |")
        try:
            base_pt = pm.get("baseline_point")
            if isinstance(base_pt, int | float) and base_pt < 0.05:
                lines.append("- Note: baseline < 5%; ratio suppressed; showing Δpp")
        except Exception:
            pass
    else:
        try:
            lines.append(f"| Ratio vs Baseline | {float(ratio):.3f} |")
        except Exception:
            lines.append("| Ratio vs Baseline | N/A |")
    lines.append("")

    # Secondary metrics (informational)
    try:
        secs = certificate.get("secondary_metrics")
        if isinstance(secs, list) and secs:
            lines.append("## Secondary Metrics (informational)")
            lines.append("")
            lines.append("| Kind | Preview | Final | vs Baseline | CI |")
            lines.append("|------|---------|-------|-------------|----|")
            for m in secs:
                if not isinstance(m, dict):
                    continue
                k = m.get("kind", "?")
                pv = _fmt_by_kind(m.get("preview"), str(k))
                fv = _fmt_by_kind(m.get("final"), str(k))
                rb = m.get("ratio_vs_baseline")
                try:
                    rb_str = (
                        f"{float(rb):.3f}"
                        if (str(k).startswith("ppl"))
                        else _fmt_by_kind(rb, str(k))
                    )
                except Exception:
                    rb_str = "N/A"
                ci = m.get("display_ci") or m.get("ci")
                if isinstance(ci, tuple | list) and len(ci) == 2:
                    ci_str = f"{float(ci[0]):.3f}-{float(ci[1]):.3f}"
                else:
                    ci_str = "–"
                lines.append(f"| {k} | {pv} | {fv} | {rb_str} | {ci_str} |")
            lines.append("")
    except Exception:
        pass


def _append_policy_configuration_section(
    lines: list[str], certificate: dict[str, Any]
) -> None:
    resolved_policy = certificate.get("resolved_policy")
    policy_provenance = certificate.get("policy_provenance", {}) or {}
    has_prov = isinstance(policy_provenance, dict) and bool(policy_provenance)
    has_resolved = isinstance(resolved_policy, dict) and bool(resolved_policy)
    if not (has_prov or has_resolved):
        return

    lines.append("## Policy Configuration")
    lines.append("")

    tier = None
    if has_prov:
        tier = policy_provenance.get("tier")
    if not tier:
        tier = (certificate.get("auto", {}) or {}).get("tier")
    digest_value = None
    if has_prov:
        digest_value = policy_provenance.get("policy_digest")
    if not digest_value:
        digest_value = (certificate.get("policy_digest", {}) or {}).get(
            "thresholds_hash"
        )

    summary_parts: list[str] = []
    if tier:
        summary_parts.append(f"**Tier:** {tier}")
    if digest_value:
        summary_parts.append(f"**Digest:** `{_short_digest(str(digest_value))}`")
    if summary_parts:
        lines.append(" | ".join(summary_parts))

    if has_prov:
        overrides_list = policy_provenance.get("overrides") or []
        if overrides_list:
            lines.append(f"- **Overrides:** {', '.join(overrides_list)}")
        else:
            lines.append("- **Overrides:** (none)")
        if policy_provenance.get("resolved_at"):
            lines.append(f"- **Resolved At:** {policy_provenance.get('resolved_at')}")

    if has_resolved:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Resolved Policy YAML</summary>")
        lines.append("")
        lines.append("```yaml")
        resolved_yaml = yaml.safe_dump(
            resolved_policy, sort_keys=True, width=80, default_flow_style=False
        ).strip()
        for line in resolved_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append("</details>")

    lines.append("")


def _append_dataset_and_provenance_section(
    lines: list[str], certificate: dict[str, Any]
) -> None:
    dataset = certificate.get("dataset", {}) or {}
    provenance_info = certificate.get("provenance", {}) or {}

    has_dataset = isinstance(dataset, dict) and bool(dataset)
    has_provenance = isinstance(provenance_info, dict) and bool(provenance_info)
    if not (has_dataset or has_provenance):
        return

    lines.append("## Dataset and Provenance")
    lines.append("")

    if has_dataset:
        prov = dataset.get("provider") or "unknown"
        lines.append(f"- **Provider:** {prov}")
        try:
            seq_len_val = (
                int(dataset.get("seq_len"))
                if isinstance(dataset.get("seq_len"), int | float)
                else dataset.get("seq_len")
            )
        except Exception:  # pragma: no cover - defensive
            seq_len_val = dataset.get("seq_len")
        if seq_len_val is not None:
            lines.append(f"- **Sequence Length:** {seq_len_val}")
        windows_blk = (
            dataset.get("windows", {})
            if isinstance(dataset.get("windows"), dict)
            else {}
        )
        win_prev = windows_blk.get("preview")
        win_final = windows_blk.get("final")
        if win_prev is not None and win_final is not None:
            lines.append(f"- **Windows:** {win_prev} preview + {win_final} final")
        if windows_blk.get("seed") is not None:
            lines.append(f"- **Seed:** {windows_blk.get('seed')}")
        hash_blk = (
            dataset.get("hash", {}) if isinstance(dataset.get("hash"), dict) else {}
        )
        if hash_blk.get("preview_tokens") is not None:
            lines.append(f"- **Preview Tokens:** {hash_blk.get('preview_tokens'):,}")
        if hash_blk.get("final_tokens") is not None:
            lines.append(f"- **Final Tokens:** {hash_blk.get('final_tokens'):,}")
        if hash_blk.get("total_tokens") is not None:
            lines.append(f"- **Total Tokens:** {hash_blk.get('total_tokens'):,}")
        if hash_blk.get("dataset"):
            lines.append(f"- **Dataset Hash:** {hash_blk.get('dataset')}")
        tokenizer = dataset.get("tokenizer", {})
        if isinstance(tokenizer, dict) and (
            tokenizer.get("name") or tokenizer.get("hash")
        ):
            vocab_size = tokenizer.get("vocab_size")
            vocab_suffix = (
                f" (vocab {vocab_size})" if isinstance(vocab_size, int) else ""
            )
            lines.append(
                f"- **Tokenizer:** {tokenizer.get('name', 'unknown')}{vocab_suffix}"
            )
            if tokenizer.get("hash"):
                lines.append(f"  - Hash: {tokenizer['hash']}")
            lines.append(
                f"  - BOS/EOS: {tokenizer.get('bos_token')} / {tokenizer.get('eos_token')}"
            )
            if tokenizer.get("pad_token") is not None:
                lines.append(f"  - PAD: {tokenizer.get('pad_token')}")
            if tokenizer.get("add_prefix_space") is not None:
                lines.append(
                    f"  - add_prefix_space: {tokenizer.get('add_prefix_space')}"
                )

    if has_provenance:
        baseline_info = provenance_info.get("baseline", {}) or {}
        edited_info = provenance_info.get("edited", {}) or {}

        if baseline_info or edited_info:
            lines.append("")
        if baseline_info:
            lines.append(f"- **Baseline Run ID:** {baseline_info.get('run_id')}")
            if baseline_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{baseline_info.get('report_hash')}`")
            if baseline_info.get("report_path"):
                lines.append(f"  - Report Path: {baseline_info.get('report_path')}")
        if edited_info:
            lines.append(f"- **Edited Run ID:** {edited_info.get('run_id')}")
            if edited_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{edited_info.get('report_hash')}`")
            if edited_info.get("report_path"):
                lines.append(f"  - Report Path: {edited_info.get('report_path')}")

        provider_digest = provenance_info.get("provider_digest")
        if isinstance(provider_digest, dict) and provider_digest:
            ids_d = provider_digest.get("ids_sha256")
            tok_d = provider_digest.get("tokenizer_sha256")
            mask_d = provider_digest.get("masking_sha256")

            lines.append("- **Provider Digest:**")
            if tok_d:
                lines.append(
                    f"  - tokenizer_sha256: `{_short_digest(tok_d)}` (full in JSON)"
                )
            if ids_d:
                lines.append(f"  - ids_sha256: `{_short_digest(ids_d)}` (full in JSON)")
            if mask_d:
                lines.append(
                    f"  - masking_sha256: `{_short_digest(mask_d)}` (full in JSON)"
                )

        try:
            conf = certificate.get("confidence", {}) or {}
            if isinstance(conf, dict) and conf.get("label"):
                lines.append(f"- **Confidence:** {conf.get('label')}")
        except Exception:
            pass

        try:
            pd = certificate.get("policy_digest", {}) or {}
            if isinstance(pd, dict) and pd:
                pv = pd.get("policy_version")
                th = pd.get("thresholds_hash")
                if pv:
                    lines.append(f"- **Policy Version:** {pv}")
                if isinstance(th, str) and th:
                    short = th if len(th) <= 16 else (th[:8] + "…" + th[-8:])
                    lines.append(f"- **Thresholds Digest:** `{short}` (full in JSON)")
                if pd.get("changed"):
                    lines.append("- Note: policy changed")
        except Exception:
            pass

    lines.append("")


def _fmt_by_kind(x: Any, k: str) -> str:
    try:
        xv = float(x)
    except Exception:
        return "N/A"
    k = str(k).lower()
    if k in {"accuracy", "vqa_accuracy"}:
        return f"{xv * 100.0:.1f}"
    if k.startswith("ppl"):
        return f"{xv:.3g}"
    return f"{xv:.3f}"


def _fmtv(key: str, v: Any) -> str:
    if not (isinstance(v, int | float) and math.isfinite(float(v))):
        return "-"
    if key.startswith("latency_ms_"):
        return f"{float(v):.0f}"
    if key.startswith("throughput_"):
        return f"{float(v):.1f}"
    return f"{float(v):.3f}"


def _p(x: Any) -> str:
    try:
        return f"{float(x) * 100.0:.1f}%"
    except Exception:
        return "N/A"


def _append_system_overhead_section(lines: list[str], sys_over: dict[str, Any]) -> None:
    """Append the System Overhead markdown section to lines given a payload."""
    if not (isinstance(sys_over, dict) and sys_over):
        return
    lines.append("## System Overhead")
    lines.append("")
    lines.append("| Metric | Baseline | Edited | Δ | Ratio |")
    lines.append("|--------|----------|--------|---|-------|")

    mapping = {
        "latency_ms_p50": "Latency p50 (ms)",
        "latency_ms_p95": "Latency p95 (ms)",
        "throughput_sps": "Throughput (samples/s)",
    }
    for key, label in mapping.items():
        ent = sys_over.get(key)
        if not isinstance(ent, dict):
            continue
        b_raw = ent.get("baseline")
        e_raw = ent.get("edited")
        # If both baseline and edited are missing or zero, present N/A to avoid implying measured zeros
        try:
            b_val = float(b_raw)
        except Exception:
            b_val = float("nan")
        try:
            e_val = float(e_raw)
        except Exception:
            e_val = float("nan")
        if (not math.isfinite(b_val) or b_val == 0.0) and (
            not math.isfinite(e_val) or e_val == 0.0
        ):
            b_str = e_str = d_str = r_str = "N/A"
        else:
            b_str = _fmtv(key, b_val)
            e_str = _fmtv(key, e_val)
            d = ent.get("delta")
            r = ent.get("ratio")
            d_str = _fmtv(key, d) if isinstance(d, int | float) else "-"
            r_str = _fmtv(key, r) if isinstance(r, int | float) else "-"
        lines.append(f"| {label} | {b_str} | {e_str} | {d_str} | {r_str} |")
    lines.append("")


def _append_accuracy_subgroups(lines: list[str], subgroups: dict[str, Any]) -> None:
    """Append the Accuracy Subgroups markdown table given a subgroups payload."""
    if not (isinstance(subgroups, dict) and subgroups):
        return
    lines.append("## Accuracy Subgroups (informational)")
    lines.append("")
    lines.append("| Group | n(prev) | n(final) | Acc(prev) | Acc(final) | Δpp |")
    lines.append("|-------|---------|----------|-----------|------------|-----|")
    for g, rec in subgroups.items():
        try:
            npv = int(rec.get("n_preview", 0))
        except Exception:
            npv = 0
        try:
            nfi = int(rec.get("n_final", 0))
        except Exception:
            nfi = 0
        dp = rec.get("delta_pp")
        try:
            dp_str = f"{float(dp):+.1f} pp"
        except Exception:
            dp_str = "N/A"
        lines.append(
            f"| {g} | {npv} | {nfi} | {_p(rec.get('preview'))} | {_p(rec.get('final'))} | {dp_str} |"
        )
    lines.append("")


def _compute_certificate_hash(certificate: dict[str, Any]) -> str:
    """Compute integrity hash for the certificate.

    Hash ignores the `artifacts` section for stability across saves.
    """
    # Create a copy without the artifacts section for stable hashing
    cert_copy = dict(certificate or {})
    cert_copy.pop("artifacts", None)

    # Sort keys for deterministic hashing
    cert_str = json.dumps(cert_copy, sort_keys=True)
    import hashlib as _hash

    return _hash.sha256(cert_str.encode()).hexdigest()[:16]


def build_console_summary_pack(certificate: dict[str, Any]) -> dict[str, Any]:
    """Build a small, reusable console summary pack from a certificate.

    Returns a dict with:
    - overall_pass: bool
    - overall_line: human-friendly overall status line
    - gate_lines: list of "<Label>: <Status>" strings for each evaluated gate
    - labels: the canonical label list used
    """
    block = compute_console_validation_block(certificate)
    overall_pass = bool(block.get("overall_pass"))
    emoji = "✅" if overall_pass else "❌"
    overall_line = f"Overall Status: {emoji} {'PASS' if overall_pass else 'FAIL'}"

    gate_lines: list[str] = []
    for row in block.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "Gate")
        status = row.get("status", "")
        gate_lines.append(f"{label}: {status}")

    return {
        "overall_pass": overall_pass,
        "overall_line": overall_line,
        "gate_lines": gate_lines,
        "labels": block.get("labels", []),
    }


def render_certificate_markdown(certificate: dict[str, Any]) -> str:
    """
    Render a certificate as a formatted Markdown report with pretty tables.

    This implementation is moved from certificate.py to keep that module lean.
    To avoid circular import issues, we alias helpers from the certificate
    module inside the function body.
    """
    # Alias frequently used helpers locally to avoid editing the large body
    validate_certificate = C.validate_certificate

    if not validate_certificate(certificate):
        raise ValueError("Invalid certificate structure")

    lines: list[str] = []
    appendix_lines: list[str] = []
    edit_name = str(certificate.get("edit_name") or "").lower()

    # Header
    lines.append("# InvarLock Evaluation Certificate")
    lines.append("")
    lines.append(
        "> *Basis: “point” gates check the point estimate; “upper” gates check the CI "
        "upper bound; “point & upper” requires both to pass.*"
    )
    lines.append("")
    lines.append(f"**Schema Version:** {certificate['schema_version']}")
    lines.append(f"**Run ID:** `{certificate['run_id']}`")
    lines.append(f"**Generated:** {certificate['artifacts']['generated_at']}")
    lines.append(f"**Edit Type:** {certificate.get('edit_name', 'Unknown')}")
    lines.append("")
    lines.append(
        "> Full evidence: see [`evaluation.cert.json`](evaluation.cert.json) for complete provenance, digests, and raw measurements."
    )
    lines.append("")

    plugins = certificate.get("plugins", {})
    if isinstance(plugins, dict) and plugins:
        lines.append("## Plugin Provenance")
        lines.append("")

        adapter_plugin = plugins.get("adapter")
        if isinstance(adapter_plugin, dict):
            lines.append(f"- Adapter: {_format_plugin(adapter_plugin)}")

        edit_plugin = plugins.get("edit")
        if isinstance(edit_plugin, dict):
            lines.append(f"- Edit: {_format_plugin(edit_plugin)}")

        guard_plugins = plugins.get("guards")
        if isinstance(guard_plugins, list) and guard_plugins:
            guard_entries = [
                _format_plugin(plugin)
                for plugin in guard_plugins
                if isinstance(plugin, dict)
            ]
            if guard_entries:
                lines.append("- Guards:\n  - " + "\n  - ".join(guard_entries))
    lines.append("")

    # Executive Summary with validation status (canonical, from console block)
    lines.append("## Executive Summary")
    lines.append("")
    _block = compute_console_validation_block(certificate)
    overall_pass = bool(_block.get("overall_pass"))
    status_emoji = "✅" if overall_pass else "❌"
    lines.append(
        f"**Overall Status:** {status_emoji} {'PASS' if overall_pass else 'FAIL'}"
    )
    # Window Plan one-liner for quick audit
    try:
        plan_ctx = (
            certificate.get("window_plan")
            or certificate.get("dataset", {}).get("windows", {})
            or certificate.get("ppl", {}).get("window_plan")
        )
        seq_len = certificate.get("dataset", {}).get("seq_len") or certificate.get(
            "dataset", {}
        ).get("sequence_length")
        if isinstance(plan_ctx, dict):
            profile = plan_ctx.get("profile")
            preview_n = (
                plan_ctx.get("preview_n")
                if plan_ctx.get("preview_n") is not None
                else plan_ctx.get("actual_preview")
            )
            final_n = (
                plan_ctx.get("final_n")
                if plan_ctx.get("final_n") is not None
                else plan_ctx.get("actual_final")
            )
            lines.append(
                f"- Window Plan: {profile}, {preview_n}/{final_n}{', seq_len=' + str(seq_len) if seq_len else ''}"
            )
    except Exception:
        pass
    lines.append("")

    dashboard = _render_executive_dashboard(certificate)
    if dashboard:
        lines.extend(dashboard.splitlines())
        lines.append("")

    lines.append("## Contents")
    lines.append("")
    lines.append("- [Safety Dashboard](#safety-dashboard)")
    lines.append("- [Quality Gates](#quality-gates)")
    lines.append("- [Safety Check Details](#safety-check-details)")
    lines.append("- [Primary Metric](#primary-metric)")
    lines.append("- [Guard Observability](#guard-observability)")
    lines.append("- [Model Information](#model-information)")
    lines.append("- [Dataset and Provenance](#dataset-and-provenance)")
    lines.append("- [Policy Configuration](#policy-configuration)")
    lines.append("- [Appendix](#appendix)")
    lines.append("- [Certificate Integrity](#certificate-integrity)")
    lines.append("")

    # Validation table with canonical gates (mirrors console allow-list)
    lines.append("## Quality Gates")
    lines.append("")
    lines.append("| Gate | Status | Measured | Threshold | Basis | Description |")
    lines.append("|------|--------|----------|-----------|-------|-------------|")

    pm_block = certificate.get("primary_metric", {}) or {}
    has_pm = isinstance(pm_block, dict) and bool(pm_block)
    auto_info = certificate.get("auto", {})
    tier = (auto_info.get("tier") or "balanced").lower()

    # Helper to emit Primary Metric Acceptable row
    def _emit_pm_gate_row() -> None:
        pm_kind = str(pm_block.get("kind", "")).lower()
        value = pm_block.get("ratio_vs_baseline")
        gating_basis = pm_block.get("gating_basis") or "point"
        ok = bool(
            certificate.get("validation", {}).get("primary_metric_acceptable", True)
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        if pm_kind in {"accuracy", "vqa_accuracy"}:
            measured = f"{value:+.2f} pp" if isinstance(value, int | float) else "N/A"
            th_map = {
                "conservative": -0.5,
                "balanced": -1.0,
                "aggressive": -2.0,
                "none": -1.0,
            }
            th = th_map.get(tier, -1.0)
            lines.append(
                f"| Primary Metric Acceptable | {status} | {measured} | ≥ {th:+.2f} pp | {gating_basis} | Δ accuracy vs baseline |"
            )
        else:
            tier_thresholds = {
                "conservative": 1.05,
                "balanced": 1.10,
                "aggressive": 1.20,
                "none": 1.10,
            }
            ratio_limit = tier_thresholds.get(tier, 1.10)
            target_ratio = auto_info.get("target_pm_ratio")
            if isinstance(target_ratio, int | float) and target_ratio > 0:
                ratio_limit = min(ratio_limit, float(target_ratio))
            measured = f"{value:.3f}x" if isinstance(value, int | float) else "N/A"
            lines.append(
                f"| Primary Metric Acceptable | {status} | {measured} | ≤ {ratio_limit:.2f}x | {gating_basis} | Ratio vs baseline |"
            )

    # Helper to emit Preview Final Drift Acceptable row
    def _emit_drift_gate_row() -> None:
        ok = bool(
            certificate.get("validation", {}).get(
                "preview_final_drift_acceptable", True
            )
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        drift_min = 0.95
        drift_max = 1.05
        try:
            drift_band = (
                pm_block.get("drift_band") if isinstance(pm_block, dict) else None
            )
            if isinstance(drift_band, dict):
                lo = drift_band.get("min")
                hi = drift_band.get("max")
                if isinstance(lo, int | float) and isinstance(hi, int | float):
                    lo_f = float(lo)
                    hi_f = float(hi)
                    if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                        drift_min = lo_f
                        drift_max = hi_f
            elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
                lo_raw, hi_raw = drift_band[0], drift_band[1]
                if isinstance(lo_raw, int | float) and isinstance(hi_raw, int | float):
                    lo_f = float(lo_raw)
                    hi_f = float(hi_raw)
                    if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                        drift_min = lo_f
                        drift_max = hi_f
        except Exception:
            pass
        # Compute drift from PM preview/final when available
        try:
            pv = (
                float(pm_block.get("preview"))
                if isinstance(pm_block.get("preview"), int | float)
                else float("nan")
            )
            fv = (
                float(pm_block.get("final"))
                if isinstance(pm_block.get("final"), int | float)
                else float("nan")
            )
            drift = (
                fv / pv
                if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
                else float("nan")
            )
        except Exception:
            drift = float("nan")
        measured = f"{drift:.3f}x" if math.isfinite(drift) else "N/A"
        band_label = f"{drift_min:.2f}–{drift_max:.2f}x"
        lines.append(
            f"| Preview Final Drift Acceptable | {status} | {measured} | {band_label} | point | Final/Preview ratio stability |"
        )

    # Helper to emit Guard Overhead Acceptable row (only when evaluated)
    def _emit_overhead_gate_row() -> None:
        guard_overhead = certificate.get("guard_overhead", {}) or {}
        evaluated = bool(guard_overhead.get("evaluated"))
        if not evaluated:
            return
        ok = bool(
            certificate.get("validation", {}).get("guard_overhead_acceptable", True)
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        overhead_pct = guard_overhead.get("overhead_percent")
        overhead_ratio = guard_overhead.get("overhead_ratio")
        if isinstance(overhead_pct, int | float) and math.isfinite(float(overhead_pct)):
            measured = f"{float(overhead_pct):+.2f}%"
        elif isinstance(overhead_ratio, int | float) and math.isfinite(
            float(overhead_ratio)
        ):
            measured = f"{float(overhead_ratio):.3f}x"
        else:
            measured = "N/A"
        threshold_pct = guard_overhead.get("threshold_percent")
        if not (
            isinstance(threshold_pct, int | float)
            and math.isfinite(float(threshold_pct))
        ):
            threshold_val = guard_overhead.get("overhead_threshold", 0.01)
            try:
                threshold_pct = float(threshold_val) * 100.0
            except Exception:
                threshold_pct = 1.0
        lines.append(
            f"| Guard Overhead Acceptable | {status} | {measured} | ≤ +{threshold_pct:.1f}% | point | Guarded vs bare PM overhead |"
        )

    def _emit_pm_tail_gate_row() -> None:
        pm_tail = certificate.get("primary_metric_tail", {}) or {}
        if not isinstance(pm_tail, dict) or not pm_tail:
            return

        evaluated = bool(pm_tail.get("evaluated", False))
        mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
        passed = bool(pm_tail.get("passed", True))
        warned = bool(pm_tail.get("warned", False))

        if not evaluated:
            status = "🛈 INFO"
        elif passed:
            status = "✅ PASS"
        elif mode == "fail":
            status = "❌ FAIL"
        else:
            status = "⚠️ WARN" if warned else "⚠️ WARN"

        policy = (
            pm_tail.get("policy", {}) if isinstance(pm_tail.get("policy"), dict) else {}
        )
        stats = (
            pm_tail.get("stats", {}) if isinstance(pm_tail.get("stats"), dict) else {}
        )

        q = policy.get("quantile", 0.95)
        try:
            qf = float(q)
        except Exception:
            qf = 0.95
        qf = max(0.0, min(1.0, qf))
        q_key = f"q{int(round(100.0 * qf))}"
        q_name = f"P{int(round(100.0 * qf))}"
        q_val = stats.get(q_key)
        mass_val = stats.get("tail_mass")
        eps = policy.get("epsilon", stats.get("epsilon"))

        measured_parts: list[str] = []
        if isinstance(q_val, int | float) and math.isfinite(float(q_val)):
            measured_parts.append(f"{q_name}={float(q_val):.3f}")
        if isinstance(mass_val, int | float) and math.isfinite(float(mass_val)):
            measured_parts.append(f"mass={float(mass_val):.3f}")
        measured = ", ".join(measured_parts) if measured_parts else "N/A"

        thr_parts: list[str] = []
        qmax = policy.get("quantile_max")
        if isinstance(qmax, int | float) and math.isfinite(float(qmax)):
            thr_parts.append(f"{q_name}≤{float(qmax):.3f}")
        mmax = policy.get("mass_max")
        if isinstance(mmax, int | float) and math.isfinite(float(mmax)):
            thr_parts.append(f"mass≤{float(mmax):.3f}")
        if isinstance(eps, int | float) and math.isfinite(float(eps)):
            thr_parts.append(f"ε={float(eps):.1e}")
        threshold = "; ".join(thr_parts) if thr_parts else "policy"

        lines.append(
            f"| Primary Metric Tail | {status} | {measured} | {threshold} | {q_name.lower()} | Tail regression vs baseline (ΔlogNLL) |"
        )

    # Emit canonical gate rows
    if has_pm:
        _emit_pm_gate_row()
        _emit_pm_tail_gate_row()
        _emit_drift_gate_row()
        _emit_overhead_gate_row()

    # Annotate hysteresis usage if applied
    if certificate.get("validation", {}).get("hysteresis_applied"):
        lines.append("- Note: hysteresis applied to gate boundary")

    lines.append("")
    lines.append("## Safety Check Details")
    lines.append("")
    lines.append("| Safety Check | Status | Measured | Threshold | Description |")
    lines.append("|--------------|--------|----------|-----------|-------------|")

    inv_summary = certificate["invariants"]
    validation = certificate.get("validation", {})
    inv_status = "✅ PASS" if validation.get("invariants_pass", False) else "❌ FAIL"
    inv_counts = inv_summary.get("summary", {}) or {}
    inv_measure = inv_summary.get("status", "pass").upper()
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
    invariants_failures = inv_summary.get("failures") or []
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
    caps_applied = certificate["spectral"]["caps_applied"]
    lines.append(
        f"| Spectral Stability | {spec_status} | {caps_applied} violations | < 5 | Weight matrix spectral norms |"
    )

    # Catastrophic spike safety stop row is now driven by primary metric flags
    if isinstance(certificate.get("primary_metric"), dict):
        pm_ok = bool(validation.get("primary_metric_acceptable", True))
        pm_ratio = certificate.get("primary_metric", {}).get("ratio_vs_baseline")
        if isinstance(pm_ratio, int | float):
            lines.append(
                f"| Catastrophic Spike Gate (safety stop) | {'✅ PASS' if pm_ok else '❌ FAIL'} | {pm_ratio:.3f}x | ≤ 2.0x | Hard stop @ 2.0× |"
            )

    # Include RMT Health row for compatibility and clarity
    rmt_status = "✅ PASS" if validation.get("rmt_stable", False) else "❌ FAIL"
    rmt_state = certificate.get("rmt", {}).get("status", "unknown").title()
    lines.append(
        f"| RMT Health | {rmt_status} | {rmt_state} | ε-rule | Random Matrix Theory guard status |"
    )

    # Pairing + Bootstrap snapshot (quick audit surface)
    try:
        stats = (
            certificate.get("dataset", {}).get("windows", {}).get("stats", {})
            or certificate.get("ppl", {}).get("stats", {})
            or {}
        )
        paired_windows = stats.get("paired_windows")
        match_frac = stats.get("window_match_fraction")
        overlap_frac = stats.get("window_overlap_fraction")
        bootstrap = stats.get("bootstrap") or {}
        if (
            paired_windows is not None
            or match_frac is not None
            or overlap_frac is not None
        ):
            lines.append("")
            parts: list[str] = []
            if paired_windows is not None:
                try:
                    parts.append(f"{int(paired_windows)} windows")
                except Exception:
                    parts.append(f"windows={paired_windows}")
            if isinstance(match_frac, int | float) and math.isfinite(float(match_frac)):
                parts.append(f"{float(match_frac) * 100.0:.1f}% match")
            elif match_frac is not None:
                parts.append(f"match={match_frac}")
            if isinstance(overlap_frac, int | float) and math.isfinite(
                float(overlap_frac)
            ):
                parts.append(f"{float(overlap_frac) * 100.0:.1f}% overlap")
            elif overlap_frac is not None:
                parts.append(f"overlap={overlap_frac}")
            lines.append(f"✅ Pairing: {', '.join(parts) if parts else 'N/A'}")
        if isinstance(bootstrap, dict):
            reps = bootstrap.get("replicates")
            bseed = bootstrap.get("seed")
            if reps is not None or bseed is not None:
                bits: list[str] = []
                if reps is not None:
                    try:
                        bits.append(f"{int(reps)} replicates")
                    except Exception:
                        bits.append(f"replicates={reps}")
                if bseed is not None:
                    try:
                        bits.append(f"seed={int(bseed)}")
                    except Exception:
                        bits.append(f"seed={bseed}")
                lines.append(f"✅ Bootstrap: {', '.join(bits) if bits else 'N/A'}")
        # Optional: show log-space paired Δ CI next to ratio CI for clarity
        delta_ci = certificate.get("primary_metric", {}).get("ci") or certificate.get(
            "ppl", {}
        ).get("logloss_delta_ci")
        if (
            isinstance(delta_ci, tuple | list)
            and len(delta_ci) == 2
            and all(isinstance(x, int | float) for x in delta_ci)
        ):
            lines.append(f"🛈 Log Δ (paired) CI: [{delta_ci[0]:.6f}, {delta_ci[1]:.6f}]")
    except Exception:
        pass

    if invariants_failures:
        lines.append("")
        lines.append("**Invariant Notes**")
        lines.append("")
        for failure in invariants_failures:
            severity = failure.get("severity", "warning")
            detail = failure.get("detail", {})
            detail_str = ""
            if isinstance(detail, dict) and detail:
                detail_str = ", ".join(f"{k}={v}" for k, v in detail.items())
                detail_str = f" ({detail_str})"
            lines.append(
                f"- {failure.get('check', 'unknown')} [{severity}]: {failure.get('type', 'violation')}{detail_str}"
            )

    lines.append("")

    _append_primary_metric_section(lines, certificate)

    # Guard observability snapshots
    lines.append("## Guard Observability")
    lines.append("")

    spectral_info = certificate.get("spectral", {}) or {}
    if spectral_info:
        lines.append("### Spectral Guard Summary")
        lines.append("")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")

        spectral_ok = bool(validation.get("spectral_stable", False))
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
                        isinstance(z_val, int | float) and math.isfinite(float(z_val))
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
            except Exception:
                kappa = None
        kappa_f = (
            float(kappa)
            if isinstance(kappa, int | float) and math.isfinite(float(kappa))
            else None
        )

        if max_abs_z is not None:
            max_val = f"{max_abs_z:.3f}"
            if max_family:
                max_val += f" ({max_family})"
            if max_module:
                max_val += f" – {max_module}"
            if kappa_f is None:
                max_status = "🛈 No κ"
            elif max_abs_z <= kappa_f:
                max_status = f"✅ Within κ={kappa_f:.3f}"
            else:
                max_status = f"❌ Exceeds κ={kappa_f:.3f}"
            lines.append(f"| Max |z| | {max_val} | {max_status} |")

        mt_info = spectral_info.get("multiple_testing", {}) or {}
        if isinstance(mt_info, dict) and mt_info:
            mt_method = mt_info.get("method")
            mt_alpha = mt_info.get("alpha")
            mt_m = mt_info.get("m")
            parts: list[str] = []
            if mt_method:
                parts.append(f"method={mt_method}")
            if isinstance(mt_alpha, int | float) and math.isfinite(float(mt_alpha)):
                parts.append(f"α={float(mt_alpha):.3g}")
            if isinstance(mt_m, int | float) and math.isfinite(float(mt_m)):
                parts.append(f"m={int(mt_m)}")
            lines.append(
                f"| Multiple Testing | {', '.join(parts) if parts else '—'} | 🛈 INFO |"
            )

        lines.append("")

        caps_by_family = spectral_info.get("caps_applied_by_family") or {}
        quantiles = spectral_info.get("family_z_quantiles") or {}
        if any(
            bool(x)
            for x in (caps_by_family, quantiles, family_caps, top_scores)
            if isinstance(x, dict)
        ):
            lines.append("<details>")
            lines.append("<summary>Per-family details</summary>")
            lines.append("")
            lines.append("| Family | κ | q95 | Max |z| | Violations |")
            lines.append("|--------|---|-----|--------|------------|")

            families: set[str] = set()
            for block in (caps_by_family, quantiles, family_caps, top_scores):
                if isinstance(block, dict):
                    families.update(str(k) for k in block.keys())

            for family in sorted(families):
                kappa = None
                if isinstance(family_caps, dict):
                    kappa = (family_caps.get(family, {}) or {}).get("kappa")
                kappa_str = (
                    f"{float(kappa):.3f}"
                    if isinstance(kappa, int | float) and math.isfinite(float(kappa))
                    else "-"
                )

                q95 = None
                max_z = None
                if isinstance(quantiles, dict):
                    stats = quantiles.get(family) or {}
                    if isinstance(stats, dict):
                        q95 = stats.get("q95")
                        max_z = stats.get("max")
                q95_str = f"{q95:.3f}" if isinstance(q95, int | float) else "-"
                max_str = f"{max_z:.3f}" if isinstance(max_z, int | float) else "-"

                violations = None
                if isinstance(caps_by_family, dict):
                    violations = caps_by_family.get(family)
                v_str = (
                    str(int(violations)) if isinstance(violations, int | float) else "0"
                )

                lines.append(
                    f"| {family} | {kappa_str} | {q95_str} | {max_str} | {v_str} |"
                )

            if isinstance(top_scores, dict) and top_scores:
                lines.append("")
                lines.append("Top |z| per family:")
                for family in sorted(top_scores.keys()):
                    entries = top_scores[family]
                    if not isinstance(entries, list) or not entries:
                        continue
                    formatted_entries = []
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        module_name = entry.get("module", "unknown")
                        z_val = entry.get("z")
                        if isinstance(z_val, int | float) and math.isfinite(
                            float(z_val)
                        ):
                            z_str = f"{z_val:.3f}"
                        else:
                            z_str = "n/a"
                        formatted_entries.append(f"{module_name} (|z|={z_str})")
                    lines.append(f"- {family}: {', '.join(formatted_entries)}")

            lines.append("")
            lines.append("</details>")
            lines.append("")

    rmt_info = certificate.get("rmt", {}) or {}
    if rmt_info:
        lines.append("### RMT Guard")
        lines.append("")
        families = rmt_info.get("families") or {}
        stable = bool(rmt_info.get("stable", True))
        status = "✅ OK" if stable else "❌ FAIL"
        delta_total = rmt_info.get("delta_total")
        if isinstance(delta_total, int):
            lines.append(f"- Δ total: {delta_total:+d}")
        lines.append(f"- Status: {status}")
        lines.append(f"- Families: {len(families)}")
        if families:
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>RMT family details</summary>")
            lines.append("")
            lines.append("| Family | ε_f | Bare | Guarded | Δ |")
            lines.append("|--------|-----|------|---------|---|")
            for family, data in families.items():
                epsilon_val = data.get("epsilon")
                epsilon_str = (
                    f"{epsilon_val:.3f}"
                    if isinstance(epsilon_val, int | float)
                    else "-"
                )
                bare_count = data.get("bare", 0)
                guarded_count = data.get("guarded", 0)
                delta_val = None
                try:
                    bare_str = str(int(bare_count))
                except (TypeError, ValueError):
                    bare_str = "-"
                try:
                    guarded_str = str(int(guarded_count))
                except (TypeError, ValueError):
                    guarded_str = "-"
                try:
                    delta_val = int(guarded_count) - int(bare_count)  # type: ignore[arg-type]
                except Exception:
                    delta_val = None
                delta_str = f"{delta_val:+d}" if isinstance(delta_val, int) else "-"
                lines.append(
                    f"| {family} | {epsilon_str} | {bare_str} | {guarded_str} | {delta_str} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")
        else:
            lines.append("")

    guard_overhead_info = certificate.get("guard_overhead", {}) or {}
    if guard_overhead_info:
        lines.append("### Guard Overhead")
        lines.append("")
        evaluated_flag = bool(guard_overhead_info.get("evaluated", True))
        if not evaluated_flag:
            # Make explicit when overhead was not evaluated by policy/profile
            lines.append("- Evaluated: false (skipped by policy/profile)")
        bare_ppl = guard_overhead_info.get("bare_ppl")
        guarded_ppl = guard_overhead_info.get("guarded_ppl")
        if isinstance(bare_ppl, int | float) and math.isfinite(float(bare_ppl)):
            lines.append(f"- Bare Primary Metric: {bare_ppl:.3f}")
        if isinstance(guarded_ppl, int | float) and math.isfinite(float(guarded_ppl)):
            lines.append(f"- Guarded Primary Metric: {guarded_ppl:.3f}")
        ratio = guard_overhead_info.get("overhead_ratio")
        percent = guard_overhead_info.get("overhead_percent")
        if (
            isinstance(ratio, int | float)
            and math.isfinite(float(ratio))
            and isinstance(percent, int | float)
            and math.isfinite(float(percent))
        ):
            lines.append(f"- Overhead: {ratio:.4f}x ({percent:+.2f}%)")
        elif isinstance(ratio, int | float) and math.isfinite(float(ratio)):
            lines.append(f"- Overhead: {ratio:.4f}x")
        overhead_source = guard_overhead_info.get("source")
        if overhead_source:
            lines.append(f"- Source: {overhead_source}")
        plan_ctx = certificate.get("provenance", {}).get("window_plan", {})
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

    compression_diag = (
        certificate.get("structure", {}).get("compression_diagnostics", {})
        if isinstance(certificate.get("structure"), dict)
        else {}
    )
    inference_flags = compression_diag.get("inferred") or {}
    inference_sources = compression_diag.get("inference_source") or {}
    inference_log = compression_diag.get("inference_log") or []
    if inference_flags or inference_sources or inference_log:
        appendix_lines.append("### Inference Diagnostics")
        appendix_lines.append("")
        if inference_flags:
            appendix_lines.append("- **Fields Inferred:**")
            for field, flag in inference_flags.items():
                appendix_lines.append(f"  - {field}: {'yes' if flag else 'no'}")
        if inference_sources:
            appendix_lines.append("- **Sources:**")
            for field, source in inference_sources.items():
                appendix_lines.append(f"  - {field}: {source}")
        if inference_log:
            appendix_lines.append("- **Inference Log:**")
            for entry in inference_log:
                appendix_lines.append(f"  - {entry}")
        appendix_lines.append("")

    # Model and Configuration
    lines.append("## Model Information")
    lines.append("")
    meta = certificate["meta"]
    lines.append(f"- **Model ID:** {meta.get('model_id')}")
    lines.append(f"- **Adapter:** {meta.get('adapter')}")
    lines.append(f"- **Device:** {meta.get('device')}")
    lines.append(f"- **Timestamp:** {meta.get('ts')}")
    commit_value = meta.get("commit") or ""
    if commit_value:
        short_sha = str(commit_value)[:12]
        lines.append(f"- **Commit:** {short_sha}")
    else:
        lines.append("- **Commit:** (not set)")
    lines.append(f"- **Seed:** {meta.get('seed')}")
    seeds_map = meta.get("seeds", {})
    if isinstance(seeds_map, dict) and seeds_map:
        lines.append(
            "- **Seeds:** "
            f"python={seeds_map.get('python')}, "
            f"numpy={seeds_map.get('numpy')}, "
            f"torch={seeds_map.get('torch')}"
        )
    invarlock_version = meta.get("invarlock_version")
    if invarlock_version:
        lines.append(f"- **InvarLock Version:** {invarlock_version}")
    env_flags = meta.get("env_flags")
    cuda_flags = meta.get("cuda_flags")

    # Compressed determinism/environment summary for readability
    det_parts: list[str] = []
    for label, keys in (
        ("torch_det", ("torch_deterministic_algorithms", "deterministic_algorithms")),
        ("cudnn_det", ("cudnn_deterministic",)),
        ("cudnn_bench", ("cudnn_benchmark",)),
        ("tf32_matmul", ("cuda_matmul_allow_tf32",)),
        ("tf32_cudnn", ("cudnn_allow_tf32",)),
        ("cublas_ws", ("CUBLAS_WORKSPACE_CONFIG",)),
    ):
        val = None
        for key in keys:
            if isinstance(env_flags, dict) and env_flags.get(key) is not None:
                val = env_flags.get(key)
                break
            if isinstance(cuda_flags, dict) and cuda_flags.get(key) is not None:
                val = cuda_flags.get(key)
                break
        if val is not None:
            det_parts.append(f"{label}={val}")
    if det_parts:
        lines.append(f"- **Determinism:** {', '.join(det_parts)}")

    full_flags: dict[str, Any] = {}
    if isinstance(env_flags, dict) and env_flags:
        full_flags["env_flags"] = env_flags
    if isinstance(cuda_flags, dict) and cuda_flags:
        full_flags["cuda_flags"] = cuda_flags
    if full_flags:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Environment flags (full)</summary>")
        lines.append("")
        lines.append("```yaml")
        flags_yaml = yaml.safe_dump(full_flags, sort_keys=True, width=80).strip()
        for line in flags_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
    lines.append("")

    # Edit Configuration (removed duplicate Edit Information section)

    # Auto-tuning Configuration
    auto = certificate["auto"]
    if auto["tier"] != "none":
        lines.append("## Auto-Tuning Configuration")
        lines.append("")
        lines.append(f"- **Tier:** {auto['tier']}")
        lines.append(f"- **Probes Used:** {auto['probes_used']}")
    if auto.get("target_pm_ratio"):
        lines.append(
            f"- **Auto Policy Target Ratio (informational):** {auto['target_pm_ratio']:.3f}"
        )
        # Tiny relax breadcrumb for dev-only demos
        try:
            if bool(auto.get("tiny_relax")):
                lines.append("- Tiny relax: enabled (dev-only)")
        except Exception:
            pass
        lines.append("")

    _append_dataset_and_provenance_section(lines, certificate)

    # Structural Changes heading is printed with content later; avoid empty header here

    # System Overhead section (latency/throughput)
    sys_over = certificate.get("system_overhead", {}) or {}
    if isinstance(sys_over, dict) and sys_over:
        _append_system_overhead_section(lines, sys_over)

    # Accuracy Subgroups (informational)
    try:
        cls = certificate.get("classification", {})
        sub = cls.get("subgroups") if isinstance(cls, dict) else None
        if isinstance(sub, dict) and sub:
            _append_accuracy_subgroups(lines, sub)
    except Exception:
        pass
    # Structural Changes
    try:
        structure = certificate.get("structure", {}) or {}
        params_changed = int(structure.get("params_changed", 0) or 0)
        layers_modified = int(structure.get("layers_modified", 0) or 0)
        bitwidth_changes = 0
        try:
            bitwidth_changes = int(len(structure.get("bitwidths", []) or []))
        except Exception:
            bitwidth_changes = 0
        # Decide whether to show the section
        has_changes = any(
            v > 0 for v in (params_changed, layers_modified, bitwidth_changes)
        )
        edit_name = str(certificate.get("edit_name", "unknown"))
        if has_changes:
            lines.append("## Structural Changes")
            lines.append("")
            lines.append("| Change Type | Count |")
            lines.append("|-------------|-------|")
            lines.append(f"| Parameters Changed | {params_changed:,} |")
            if edit_name == "quant_rtn":
                # For quantization: prefer a single clear line reconciling target vs applied
                # using diagnostics when available. Fallback to bitwidth-change count.
                try:
                    t_an = (structure.get("compression_diagnostics", {}) or {}).get(
                        "target_analysis", {}
                    )
                except Exception:
                    t_an = {}
                eligible = None
                modified = None
                if isinstance(t_an, dict) and t_an:
                    eligible = t_an.get("modules_eligible")
                    modified = t_an.get("modules_modified")
                if isinstance(modified, int) and isinstance(eligible, int):
                    lines.append(
                        f"| Linear Modules Quantized | {modified} of {eligible} targeted |"
                    )
                else:
                    total_bitwidth_changes = bitwidth_changes
                    if total_bitwidth_changes > 0 and layers_modified > 0:
                        modules_per_layer = total_bitwidth_changes // max(
                            layers_modified, 1
                        )
                        lines.append(
                            f"| Linear Modules Quantized | {total_bitwidth_changes} ({modules_per_layer} per block × {layers_modified} blocks) |"
                        )
                    elif total_bitwidth_changes > 0:
                        lines.append(
                            f"| Linear Modules Quantized | {total_bitwidth_changes} |"
                        )
            else:
                lines.append(f"| Layers Modified | {layers_modified} |")
            lines.append("")
    except Exception:
        # Best-effort; omit section on error
        pass

    # Add detailed breakdowns if available
    if structure.get("bitwidths") and edit_name != "quant_rtn":
        lines.append(f"| Bit-width Changes | {len(structure['bitwidths'])} layers |")
    if structure.get("ranks"):
        lines.append(f"| Rank Changes | {len(structure['ranks'])} layers |")

    lines.append("")

    # Compression Diagnostics
    compression_diag = structure.get("compression_diagnostics", {})
    if edit_name == "noop":
        lines.append("### Compression Diagnostics")
        lines.append("")
        lines.append("Not applicable (no parameters modified).")
        lines.append("")
    elif compression_diag:
        lines.append("### Compression Diagnostics")
        lines.append("")

        # Algorithm execution status
        status = compression_diag.get("execution_status", "unknown")
        status_emoji = (
            "✅" if status == "successful" else "❌" if status == "failed" else "⚠️"
        )
        lines.append(f"**Execution Status:** {status_emoji} {status.upper()}")
        lines.append("")

        # Target module analysis
        target_analysis = compression_diag.get("target_analysis", {})
        if target_analysis:
            lines.append("**Target Module Analysis:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(
                f"| Modules Found | {target_analysis.get('modules_found', 0)} |"
            )
            lines.append(
                f"| Modules Eligible | {target_analysis.get('modules_eligible', 0)} |"
            )
            lines.append(
                f"| Modules Modified | {target_analysis.get('modules_modified', 0)} |"
            )
            try:
                _eligible = int(target_analysis.get("modules_eligible", 0))
                _modified = int(target_analysis.get("modules_modified", 0))
                lines.append(f"| Targets → Applied | {_eligible} → {_modified} |")
            except Exception:
                pass
            lines.append(f"| Scope | {target_analysis.get('scope', 'unknown')} |")
            lines.append("")

        # Parameter effectiveness
        param_analysis = compression_diag.get("parameter_analysis", {})
        if param_analysis:
            lines.append("**Parameter Effectiveness:**")
            lines.append("")
            for param, info in param_analysis.items():
                if isinstance(info, dict):
                    lines.append(
                        f"- **{param}:** {info.get('value', 'N/A')} ({info.get('effectiveness', 'unknown')})"
                    )
                else:
                    lines.append(f"- **{param}:** {info}")
            lines.append("")

        # Algorithm-specific details
        algo_details = compression_diag.get("algorithm_details", {})
        if algo_details:
            lines.append("**Algorithm Details:**")
            lines.append("")
            for key, value in algo_details.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Informational recommendations (non-normative)
        warnings = compression_diag.get("warnings", [])
        if warnings:
            lines.append("**ℹ️ Informational:**")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

    # Variance Guard (Spectral/RMT summaries are already provided above)
    variance = certificate["variance"]
    appendix_lines.append("### Variance Guard")
    appendix_lines.append("")

    # Display whether VE was enabled after A/B test
    appendix_lines.append(f"- **Enabled:** {'Yes' if variance['enabled'] else 'No'}")

    if variance["enabled"]:
        # VE was enabled - show the gain
        gain_value = variance.get("gain", "N/A")
        if isinstance(gain_value, int | float):
            appendix_lines.append(f"- **Gain:** {gain_value:.3f}")
        else:
            appendix_lines.append(f"- **Gain:** {gain_value}")
    else:
        # VE was not enabled - show succinct reason if available, else a clear disabled message
        ppl_no_ve = variance.get("ppl_no_ve")
        ppl_with_ve = variance.get("ppl_with_ve")
        ratio_ci = variance.get("ratio_ci")
        if ppl_no_ve is not None and ppl_with_ve is not None and ratio_ci:
            appendix_lines.append(f"- **Primary metric without VE:** {ppl_no_ve:.3f}")
            appendix_lines.append(f"- **Primary metric with VE:** {ppl_with_ve:.3f}")
            gain_value = variance.get("gain")
            if isinstance(gain_value, int | float):
                appendix_lines.append(f"- **Gain (insufficient):** {gain_value:.3f}")
        else:
            appendix_lines.append(
                "- Variance Guard: Disabled (predictive gate not evaluated for this edit)."
            )
            # Add concise rationale aligned with Balanced predictive gate contract
            try:
                ve_policy = certificate.get("policies", {}).get("variance", {})
                min_effect = ve_policy.get("min_effect_lognll")
                if isinstance(min_effect, int | float):
                    appendix_lines.append(
                        f"- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ {float(min_effect):.4g}."
                    )
                else:
                    appendix_lines.append(
                        "- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ min_effect."
                    )
                appendix_lines.append(
                    "- Predictive Gate: evaluated=false (disabled under current policy/edit)."
                )
            except Exception:
                pass

    if variance.get("ratio_ci"):
        ratio_lo, ratio_hi = variance["ratio_ci"]
        appendix_lines.append(f"- **Ratio CI:** [{ratio_lo:.3f}, {ratio_hi:.3f}]")

    if variance.get("calibration") and variance.get("enabled"):
        calib = variance["calibration"]
        coverage = calib.get("coverage")
        requested = calib.get("requested")
        status = calib.get("status", "unknown")
        appendix_lines.append(
            f"- **Calibration:** {coverage}/{requested} windows ({status})"
        )
    appendix_lines.append("")

    lines.append("")

    # MoE Observability (non-gating)
    moe = certificate.get("moe", {}) if isinstance(certificate.get("moe"), dict) else {}
    if moe:
        lines.append("## MoE Observability")
        lines.append("")
        # Core router fields
        for key in ("top_k", "capacity_factor", "expert_drop_rate"):
            if key in moe:
                lines.append(f"- **{key}:** {moe[key]}")
        # Utilization summary
        if "utilization_count" in moe or "utilization_mean" in moe:
            uc = moe.get("utilization_count")
            um = moe.get("utilization_mean")
            parts = []
            if uc is not None:
                parts.append(f"N={int(uc)}")
            if isinstance(um, int | float):
                parts.append(f"mean={um:.3f}")
            if parts:
                lines.append(f"- **Utilization:** {'; '.join(parts)}")
        # Delta summaries when available
        for key, label in (
            ("delta_load_balance_loss", "Δ load_balance_loss"),
            ("delta_router_entropy", "Δ router_entropy"),
            ("delta_utilization_mean", "Δ utilization mean"),
        ):
            if key in moe and isinstance(moe.get(key), int | float):
                lines.append(f"- **{label}:** {float(moe[key]):+.4f}")
        lines.append("")

    _append_policy_configuration_section(lines, certificate)

    appendix_lines.append("### Artifacts")
    appendix_lines.append("")
    artifacts = certificate["artifacts"]
    if artifacts.get("events_path"):
        appendix_lines.append(f"- **Events Log:** `{artifacts['events_path']}`")
    if artifacts.get("report_path"):
        appendix_lines.append(f"- **Full Report:** `{artifacts['report_path']}`")
    appendix_lines.append(f"- **Certificate Generated:** {artifacts['generated_at']}")
    appendix_lines.append("")

    if appendix_lines:
        lines.append("## Appendix")
        lines.append("")
        lines.extend(appendix_lines)

    # Certificate Hash for Integrity
    cert_hash = _compute_certificate_hash(certificate)
    lines.append("## Certificate Integrity")
    lines.append("")
    lines.append(f"**Certificate Hash:** `{cert_hash}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This InvarLock evaluation certificate provides a comprehensive assessment of model compression safety.*"
    )
    lines.append(
        "*All metrics are compared against the uncompressed baseline model for safety validation.*"
    )

    return "\n".join(lines)
