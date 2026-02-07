#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from invarlock.guards.rmt import RMTGuard


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _extract_windows(
    baseline_report: dict[str, Any], *, max_windows: int
) -> list[dict[str, Any]]:
    windows = baseline_report.get("evaluation_windows")
    if not isinstance(windows, dict):
        raise ValueError("baseline report missing evaluation_windows")

    batches: list[dict[str, Any]] = []
    for section_name in ("preview", "final"):
        section = windows.get(section_name)
        if not isinstance(section, dict):
            continue

        input_ids = section.get("input_ids")
        attention_masks = section.get("attention_masks")
        labels = section.get("labels")

        if not isinstance(input_ids, list):
            continue

        section_batches: list[dict[str, Any]] = []
        for idx, token_row in enumerate(input_ids):
            batch: dict[str, Any] = {"input_ids": token_row}
            if isinstance(attention_masks, list) and idx < len(attention_masks):
                batch["attention_mask"] = attention_masks[idx]
            if isinstance(labels, list) and idx < len(labels):
                batch["labels"] = labels[idx]
            section_batches.append(batch)

        if max_windows > 0 and len(section_batches) > max_windows:
            if max_windows == 1:
                section_batches = [section_batches[0]]
            else:
                # Deterministic evenly spaced slice.
                picks = [
                    int(round(i * (len(section_batches) - 1) / float(max_windows - 1)))
                    for i in range(max_windows)
                ]
                section_batches = [section_batches[i] for i in picks]

        batches.extend(section_batches)

    if not batches:
        raise ValueError("no calibration windows found in baseline report")
    return batches


def _safe_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return v


def _extract_rmt_policy(
    baseline_report: dict[str, Any], *, windows_count: int
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    guards = baseline_report.get("guards")
    if isinstance(guards, list):
        for guard in guards:
            if not isinstance(guard, dict):
                continue
            if str(guard.get("name", "")).strip().lower() != "rmt":
                continue

            metrics = guard.get("metrics")
            guard_policy = guard.get("policy")

            for src in (guard_policy, metrics):
                if not isinstance(src, dict):
                    continue
                eps_default = _safe_float(src.get("epsilon_default"))
                if eps_default is not None:
                    policy["epsilon_default"] = eps_default
                margin = _safe_float(src.get("margin"))
                if margin is not None:
                    policy["margin"] = margin
                deadband = _safe_float(src.get("deadband"))
                if deadband is not None:
                    policy["deadband"] = deadband
                eps_map = src.get("epsilon_by_family")
                if isinstance(eps_map, dict):
                    cleaned: dict[str, float] = {}
                    for family, value in eps_map.items():
                        vv = _safe_float(value)
                        if vv is None:
                            continue
                        cleaned[str(family)] = vv
                    if cleaned:
                        policy["epsilon_by_family"] = cleaned

    # Raise sampling count above default (8) so probes have enough signal.
    policy["activation"] = {
        "sampling": {
            "windows": {
                "count": int(max(windows_count, 8)),
                "indices_policy": "evenly_spaced",
            }
        }
    }
    return policy


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = (name or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def _load_model(
    model_path: Path, *, dtype: torch.dtype, trust_remote_code: bool
) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


def _model_cleanup(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_guard_result(
    result: Any,
) -> tuple[bool, str, dict[str, Any], list[dict[str, Any]]]:
    if isinstance(result, dict):
        passed = bool(result.get("passed", False))
        action = str(result.get("action") or ("continue" if passed else "warn"))
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        violations_raw = result.get("violations") or result.get("errors") or []
        violations = (
            violations_raw if isinstance(violations_raw, list) else [violations_raw]
        )
        norm_violations: list[dict[str, Any]] = []
        for item in violations:
            if isinstance(item, dict):
                norm_violations.append(item)
            else:
                norm_violations.append({"message": str(item)})
        return passed, action, metrics, norm_violations

    passed = bool(getattr(result, "passed", False))
    action = str(getattr(result, "action", "continue" if passed else "warn"))
    metrics = getattr(result, "metrics", {}) or {}
    if not isinstance(metrics, dict):
        metrics = {}
    raw_violations = getattr(result, "violations", []) or []
    norm_violations = []
    for item in raw_violations:
        if isinstance(item, dict):
            norm_violations.append(item)
        else:
            norm_violations.append({"message": str(item)})
    return passed, action, metrics, norm_violations


def _top_k_items(values: dict[str, Any], *, k: int) -> list[dict[str, Any]]:
    if not isinstance(values, dict) or not values:
        return []
    rows: list[tuple[str, float]] = []
    for key, raw in values.items():
        v = _safe_float(raw)
        if v is None:
            continue
        rows.append((str(key), v))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    out: list[dict[str, Any]] = []
    for name, val in rows[: max(int(k), 0)]:
        out.append({"module": name, "value": float(val)})
    return out


def _top_k_deltas(
    base: dict[str, Any], cur: dict[str, Any], *, k: int, eps: float = 1e-12
) -> list[dict[str, Any]]:
    if not isinstance(base, dict):
        base = {}
    if not isinstance(cur, dict):
        cur = {}

    rows: list[tuple[float, str, float, float]] = []
    keys = set(base) | set(cur)
    for key in keys:
        b = _safe_float(base.get(key))
        c = _safe_float(cur.get(key))
        if b is None or c is None:
            continue
        delta = c - b
        frac = delta / max(abs(b), eps)
        rows.append((frac, str(key), b, c))

    rows.sort(key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    for frac, name, b, c in rows[: max(int(k), 0)]:
        out.append(
            {
                "module": name,
                "base": float(b),
                "cur": float(c),
                "delta": float(c - b),
                "delta_frac": float(frac),
            }
        )
    return out


def _classify_family(module_name: str) -> str:
    """Mirror RMTGuard family assignment ({attn, ffn, embed, other})."""
    lower = module_name.lower()
    if any(tok in lower for tok in ("attn", "attention", "self_attn")):
        return "attn"
    if any(
        tok in lower
        for tok in (
            "router",
            "routing",
            "gate",
            "gating",
            "dispatch",
            "switch",
            "experts",
            "expert",
            "moe",
            "mixture_of_experts",
            "mlp",
            "ffn",
            "c_fc",
            "feed_forward",
        )
    ):
        return "ffn"
    if "embed" in lower or "wte" in lower or "wpe" in lower:
        return "embed"
    return "other"


def _delta_max_family_summary(
    base: dict[str, Any], cur: dict[str, Any], *, eps: float = 1e-12
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
    """Summarize per-family edge-risk using the module with max positive delta_frac.

    RMTGuard aggregates edge-risk per family using a max over modules, which can be
    dominated by a single baseline outlier. For cross-model probes we want a stable,
    architecture-robust signal that reflects "something in this family moved" even
    when the absolute family max does not.

    Returns:
      (edge_base_by_family, edge_cur_by_family, peak_by_family)
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(cur, dict):
        cur = {}

    peak_by_family: dict[str, dict[str, Any]] = {}
    keys = set(base) | set(cur)
    for key in keys:
        b = _safe_float(base.get(key))
        c = _safe_float(cur.get(key))
        if b is None or c is None or b <= 0.0:
            continue
        delta_frac = (c - b) / max(abs(b), eps)
        family = _classify_family(str(key))
        prev = peak_by_family.get(family)
        if prev is None or float(delta_frac) > float(prev.get("delta_frac", -1e9)):
            peak_by_family[family] = {
                "module": str(key),
                "base": float(b),
                "cur": float(c),
                "delta_frac": float(delta_frac),
            }

    edge_base_by_family: dict[str, float] = {}
    edge_cur_by_family: dict[str, float] = {}
    for family in ("attn", "ffn", "embed", "other"):
        peak = peak_by_family.get(family)
        if peak is None:
            edge_base_by_family[family] = 0.0
            edge_cur_by_family[family] = 0.0
        else:
            edge_base_by_family[family] = float(peak.get("base", 0.0) or 0.0)
            edge_cur_by_family[family] = float(peak.get("cur", 0.0) or 0.0)

    return edge_base_by_family, edge_cur_by_family, peak_by_family


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    baseline_model_path = Path(args.baseline_model).resolve()
    subject_model_path = Path(args.subject_model).resolve()
    baseline_report_path = Path(args.baseline_report).resolve()
    output_path = Path(args.out).resolve()

    baseline_report = _load_json(baseline_report_path)
    batches = _extract_windows(
        baseline_report, max_windows=max(0, int(args.max_windows_per_split))
    )
    policy = _extract_rmt_policy(
        baseline_report, windows_count=max(8, int(args.activation_windows))
    )
    dtype = _resolve_dtype(args.dtype)

    guard = RMTGuard(correct=False)

    baseline_model = _load_model(
        baseline_model_path, dtype=dtype, trust_remote_code=bool(args.trust_remote_code)
    )
    try:
        guard.prepare(baseline_model, calib=batches, policy=policy)
    finally:
        _model_cleanup(baseline_model)

    subject_model = _load_model(
        subject_model_path, dtype=dtype, trust_remote_code=bool(args.trust_remote_code)
    )
    try:
        guard.after_edit(subject_model)
        result = guard.finalize(subject_model)
    finally:
        _model_cleanup(subject_model)

    passed, action, metrics, violations = _normalize_guard_result(result)
    stable_guard = bool(metrics.get("stable", passed))

    # These are populated even when metrics omit module-level details.
    edge_by_module_base = getattr(guard, "baseline_edge_risk_by_module", {}) or {}
    edge_by_module = getattr(guard, "edge_risk_by_module", {}) or {}

    edge_family_base, edge_family_cur, peak_by_family = _delta_max_family_summary(
        edge_by_module_base, edge_by_module
    )

    epsilon_default = _safe_float(metrics.get("epsilon_default"))
    if epsilon_default is None:
        epsilon_default = _safe_float((policy or {}).get("epsilon_default"))
    if epsilon_default is None:
        epsilon_default = 0.01

    epsilon_by_family_raw = metrics.get("epsilon_by_family") or {}
    epsilon_by_family: dict[str, float] = {}
    if isinstance(epsilon_by_family_raw, dict):
        for family, value in epsilon_by_family_raw.items():
            vv = _safe_float(value)
            if vv is None:
                continue
            epsilon_by_family[str(family)] = vv

    epsilon_violations: list[dict[str, Any]] = []
    for family, base_val in edge_family_base.items():
        base_val = float(base_val)
        cur_val = float(edge_family_cur.get(family, 0.0) or 0.0)
        if base_val <= 0.0:
            continue
        eps_val = float(epsilon_by_family.get(family, epsilon_default))
        allowed = (1.0 + eps_val) * base_val
        if cur_val > allowed:
            peak = peak_by_family.get(family) or {}
            delta = (cur_val / base_val) - 1.0
            epsilon_violations.append(
                {
                    "family": family,
                    "module": peak.get("module"),
                    "edge_base": base_val,
                    "edge_cur": cur_val,
                    "delta": float(delta),
                    "allowed": float(allowed),
                    "epsilon": float(eps_val),
                }
            )

    stable = not epsilon_violations
    if not stable:
        action = "abort"

    payload: dict[str, Any] = {
        "probe": "rmt_cross_model_v2",
        "family_aggregation": "delta_max",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "baseline_model": str(baseline_model_path),
        "subject_model": str(subject_model_path),
        "baseline_report": str(baseline_report_path),
        "profile": str(args.profile),
        "tier": str(args.tier),
        "activation_windows": int(args.activation_windows),
        "calibration_windows_loaded": len(batches),
        "policy": policy,
        "passed": passed,
        "action": action,
        "stable": stable,
        "stable_guard": stable_guard,
        "edge_risk_by_family_base": edge_family_base,
        "edge_risk_by_family": edge_family_cur,
        "epsilon_by_family": epsilon_by_family,
        "epsilon_default": float(epsilon_default),
        "epsilon_violations": epsilon_violations,
        "edge_risk_by_module_count": len(edge_by_module),
        "edge_risk_by_module_base_top": _top_k_items(edge_by_module_base, k=25),
        "edge_risk_by_module_top": _top_k_items(edge_by_module, k=25),
        "edge_risk_by_module_delta_top": _top_k_deltas(
            edge_by_module_base, edge_by_module, k=25
        ),
        "violations": violations,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute cross-model RMT probe using baseline windows."
    )
    parser.add_argument(
        "--baseline-model", required=True, help="Path to baseline model"
    )
    parser.add_argument(
        "--subject-model", required=True, help="Path to edited/error model"
    )
    parser.add_argument(
        "--baseline-report",
        required=True,
        help="Path to baseline report.json containing evaluation_windows",
    )
    parser.add_argument(
        "--out", required=True, help="Output JSON path (rmt_probe.json)"
    )
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--profile", default="ci")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--activation-windows",
        type=int,
        default=64,
        help="RMT activation sampling windows count override",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=0,
        help="Optional cap of windows loaded from each split (0 = all available)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_probe(args)
    status = "UNSTABLE" if payload.get("stable") is False else "STABLE"
    print(
        f"[rmt_probe] status={status} violations={len(payload.get('epsilon_violations') or [])} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
