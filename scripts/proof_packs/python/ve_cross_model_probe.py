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

from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.guards.variance import VarianceGuard

try:
    from runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_tools import require_remote_code_opt_in


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
                picks = [
                    int(round(i * (len(section_batches) - 1) / float(max_windows - 1)))
                    for i in range(max_windows)
                ]
                section_batches = [section_batches[i] for i in picks]

        batches.extend(section_batches)

    if not batches:
        raise ValueError("no calibration windows found in baseline report")
    return batches


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = (name or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def _safe_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return v


def _extract_variance_policy(
    baseline_report: dict[str, Any], *, calibration_windows: int, min_coverage: int
) -> dict[str, Any]:
    policy: dict[str, Any] = {}

    # Prefer resolved policy when it is available (matches evaluation-time guard wiring).
    resolved = baseline_report.get("resolved_policy")
    if isinstance(resolved, dict):
        variance = resolved.get("variance")
        if isinstance(variance, dict):
            policy.update(variance)

    meta = baseline_report.get("meta")
    if isinstance(meta, dict):
        tier_policies = meta.get("tier_policies")
        if isinstance(tier_policies, dict):
            variance = tier_policies.get("variance")
            if isinstance(variance, dict):
                policy.update(variance)

    # Force probe semantics: compute VE even if upstream runs were monitor-only.
    policy["monitor_only"] = False

    calibration = policy.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}
    calibration = dict(calibration)
    calibration["windows"] = int(calibration_windows)
    calibration["min_coverage"] = int(min_coverage)
    if "seed" not in calibration:
        calibration["seed"] = int(policy.get("seed") or 123)
    policy["calibration"] = calibration

    # Keep max_calib consistent with windows so equalise_residual_variance doesn't
    # request more windows than are available.
    max_calib = int(policy.get("max_calib") or 0)
    required = int(calibration_windows) * 10
    if max_calib <= 0 or max_calib < required:
        policy["max_calib"] = required

    # Clamp can come through as YAML tuple or JSON list; normalize to a 2-tuple
    # when possible.
    clamp = policy.get("clamp")
    if isinstance(clamp, list | tuple) and len(clamp) == 2:
        lo = _safe_float(clamp[0])
        hi = _safe_float(clamp[1])
        if lo is not None and hi is not None and lo < hi:
            policy["clamp"] = (lo, hi)

    return policy


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


def _resolve_trust_remote_code(enabled: bool) -> bool:
    if not enabled:
        return False
    return require_remote_code_opt_in("ve_cross_model_probe.py")


def _model_cleanup(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _top_scales(scales: dict[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    items: list[tuple[str, float]] = []
    for name, value in scales.items():
        vv = _safe_float(value)
        if vv is None:
            continue
        items.append((str(name), vv))
    items.sort(key=lambda item: abs(item[1] - 1.0), reverse=True)
    return [{"module": name, "scale": scale} for name, scale in items[:limit]]


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    baseline_model_path = Path(args.baseline_model).resolve()
    subject_model_path = Path(args.subject_model).resolve()
    baseline_report_path = Path(args.baseline_report).resolve()
    output_path = Path(args.out).resolve()

    baseline_report = _load_json(baseline_report_path)
    batches = _extract_windows(
        baseline_report, max_windows=max(0, int(args.max_windows_per_split))
    )

    dtype = _resolve_dtype(args.dtype)
    policy = _extract_variance_policy(
        baseline_report,
        calibration_windows=int(args.calibration_windows),
        min_coverage=int(args.min_coverage),
    )
    trust_remote_code = _resolve_trust_remote_code(bool(args.trust_remote_code))

    guard = VarianceGuard(policy)
    adapter = HF_Causal_Adapter()

    prep_result: dict[str, Any] = {}
    target_resolution: Any = None
    target_module_names: Any = None

    subject_model = _load_model(
        subject_model_path, dtype=dtype, trust_remote_code=trust_remote_code
    )
    try:
        prep_result = guard.prepare(
            subject_model, adapter=adapter, calib=batches, policy=policy
        )
        target_resolution = guard._stats.get("target_resolution")  # noqa: SLF001
        target_module_names = guard._stats.get("target_module_names")  # noqa: SLF001
        proposed_scales_pre = guard._stats.get("proposed_scales_pre_edit", {})  # noqa: SLF001
        proposed_scales = len(guard._scales)  # noqa: SLF001
        ppl_no_ve = guard._ppl_no_ve  # noqa: SLF001
        ppl_with_ve = guard._ppl_with_ve  # noqa: SLF001
        ab_gain = guard._ab_gain  # noqa: SLF001
        ratio_ci = guard._ratio_ci  # noqa: SLF001
        predictive_gate = guard._predictive_gate_state  # noqa: SLF001
        calibration = guard._calibration_stats  # noqa: SLF001

        would_enable = False
        gate_reason = "unknown"
        try:
            would_enable, gate_reason = guard._evaluate_ab_gate()  # noqa: SLF001
        except (RuntimeError, TypeError, ValueError) as exc:
            would_enable = False
            gate_reason = f"gate_error:{exc}"
    finally:
        _model_cleanup(subject_model)

    ppl_no_ve_f = _safe_float(ppl_no_ve)
    ppl_with_ve_f = _safe_float(ppl_with_ve)
    abs_improvement = None
    if ppl_no_ve_f is not None and ppl_with_ve_f is not None:
        abs_improvement = ppl_no_ve_f - ppl_with_ve_f

    reasons: list[str] = []
    if proposed_scales < int(args.min_scales):
        reasons.append("insufficient_scales")
    if ppl_no_ve_f is None or ppl_with_ve_f is None:
        reasons.append("missing_ppl")
    else:
        if abs_improvement is not None and abs_improvement < float(args.min_abs_gain):
            reasons.append("abs_gain_below_threshold")
    ab_gain_f = _safe_float(ab_gain)
    if ab_gain_f is None:
        reasons.append("missing_rel_gain")
    else:
        if ab_gain_f < float(args.min_rel_gain):
            reasons.append("rel_gain_below_threshold")
    if bool(args.require_gate) and not bool(would_enable):
        reasons.append("ab_gate_rejected")

    signal = not reasons

    payload: dict[str, Any] = {
        "probe": "ve_probe_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "baseline_model": str(baseline_model_path),
        "subject_model": str(subject_model_path),
        "baseline_report": str(baseline_report_path),
        "profile": str(args.profile),
        "tier": str(args.tier),
        "calibration_windows_loaded": len(batches),
        "prepare": prep_result,
        "target_resolution": target_resolution,
        "target_module_names": target_module_names,
        "policy": policy,
        "signal": signal,
        "signal_reasons": reasons,
        "would_enable": bool(would_enable),
        "gate_reason": str(gate_reason),
        "proposed_scales": int(proposed_scales),
        "proposed_scales_pre_edit": proposed_scales_pre,
        "top_scales_pre_edit": _top_scales(
            proposed_scales_pre if isinstance(proposed_scales_pre, dict) else {}
        ),
        "ppl_no_ve": ppl_no_ve_f,
        "ppl_with_ve": ppl_with_ve_f,
        "abs_improvement": abs_improvement,
        "ab_gain": ab_gain_f,
        "ratio_ci": ratio_ci,
        "predictive_gate": predictive_gate,
        "calibration": calibration,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute VE probe evidence on a subject model using baseline windows."
    )
    parser.add_argument(
        "--baseline-model", required=True, help="Path to baseline model (metadata only)"
    )
    parser.add_argument(
        "--subject-model", required=True, help="Path to edited/error model"
    )
    parser.add_argument(
        "--baseline-report",
        required=True,
        help="Path to baseline report.json containing evaluation_windows",
    )
    parser.add_argument("--out", required=True, help="Output JSON path (ve_probe.json)")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--profile", default="ci")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help=(
            "Allow remote code for custom model repos; also requires "
            "INVARLOCK_ALLOW_REMOTE_CODE=1"
        ),
    )
    parser.add_argument(
        "--calibration-windows",
        type=int,
        default=12,
        help="How many calibration windows VE should use for A/B evaluation",
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=10,
        help="Minimum calibration windows required for A/B evaluation",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=0,
        help="Optional cap of windows loaded from each split (0 = all available)",
    )
    parser.add_argument(
        "--min-scales",
        type=int,
        default=1,
        help="Minimum proposed scale count required to count as a signal",
    )
    parser.add_argument(
        "--min-abs-gain",
        type=float,
        default=0.05,
        help="Minimum ppl-like absolute improvement (ppl_no_ve - ppl_with_ve)",
    )
    parser.add_argument(
        "--min-rel-gain",
        type=float,
        default=0.001,
        help="Minimum relative gain (ab_gain) required to count as a signal",
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        default=False,
        help="Require the guard's A/B gate to accept (would_enable==true)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_probe(args)
    status = "SIGNAL" if payload.get("signal") else "NO_SIGNAL"
    scales = int(payload.get("proposed_scales") or 0)
    gain = payload.get("ab_gain")
    print(f"[ve_probe] status={status} scales={scales} gain={gain} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
