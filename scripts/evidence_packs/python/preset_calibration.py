from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_YAML_LOAD_ERRORS: tuple[type[BaseException], ...]

try:
    import yaml

    _YAML_AVAILABLE = True
    _YAML_LOAD_ERRORS = (OSError, yaml.YAMLError)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_AVAILABLE = False
    _YAML_LOAD_ERRORS = (OSError,)


def _yaml_safe_load(payload: str) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is unavailable")
    loader = yaml.safe_load
    return loader(payload)


def get_default_guards_order() -> list[str]:
    return ["invariants", "spectral", "rmt", "variance", "invariants"]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_relative_growth(base: Any, current: Any) -> float | None:
    base_val = _safe_float(base)
    current_val = _safe_float(current)
    if (
        base_val is None
        or current_val is None
        or not math.isfinite(base_val)
        or not math.isfinite(current_val)
        or base_val <= 0.0
    ):
        return None
    return max(0.0, (current_val - base_val) / base_val)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    frac = pos - lower
    return values[lower] + (values[upper] - values[lower]) * frac


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_guard_order_and_assurance(
    cal_dir: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    guards_order: list[str] | None = None
    assurance_cfg: dict[str, Any] | None = None
    if _YAML_AVAILABLE:
        cfg_path = next(
            iter(sorted(cal_dir.glob("run_*/calibration_config.yaml"))), None
        )
        if cfg_path is not None:
            try:
                cfg = _yaml_safe_load(cfg_path.read_text())
            except _YAML_LOAD_ERRORS:
                cfg = None
            if isinstance(cfg, dict):
                guards_block = cfg.get("guards") or {}
                if isinstance(guards_block, dict):
                    order = guards_block.get("order")
                    if isinstance(order, list) and order:
                        guards_order = [str(item) for item in order]
                ab = cfg.get("assurance")
                if isinstance(ab, dict) and ab:
                    assurance_cfg = ab

    if guards_order is None:
        guards_order = get_default_guards_order()
    return guards_order, assurance_cfg


def _record_section(rec: dict[str, Any], name: str) -> dict[str, Any]:
    value = rec.get(name)
    return value if isinstance(value, dict) else {}


def _merge_spectral_record(
    rec: dict[str, Any],
    metrics: dict[str, Any],
    policy: dict[str, Any],
    guard: dict[str, Any],
) -> None:
    spectral = _record_section(rec, "spectral")
    for key in ("family_z_quantiles", "family_z_summary", "family_caps", "families"):
        value = metrics.get(key)
        if value:
            spectral.setdefault(key, value)
    for key in ("sigma_quantile", "deadband", "max_caps"):
        value = metrics.get(key)
        if value is not None:
            spectral.setdefault(key, value)
    if metrics.get("family_stats"):
        spectral.setdefault("families", metrics["family_stats"])
    z_scores = guard.get("final_z_scores") or metrics.get("final_z_scores")
    if isinstance(z_scores, dict):
        spectral["final_z_scores"] = z_scores
    family_map = guard.get("module_family_map") or metrics.get("module_family_map")
    if isinstance(family_map, dict):
        spectral["module_family_map"] = family_map
    if policy and not spectral.get("policy"):
        spectral["policy"] = policy
    rec["spectral"] = spectral


def _merge_rmt_record(
    rec: dict[str, Any], metrics: dict[str, Any], policy: dict[str, Any]
) -> None:
    rmt = _record_section(rec, "rmt")
    for key in (
        "outliers_per_family",
        "baseline_outliers_per_family",
        "families",
        "edge_risk_by_family_base",
        "edge_risk_by_family",
    ):
        value = metrics.get(key)
        if isinstance(value, dict) and value:
            rmt.setdefault(key, value)
    epsilon_by_family = metrics.get("epsilon_by_family")
    if epsilon_by_family:
        rmt.setdefault("epsilon_by_family", epsilon_by_family)
    else:
        epsilon = metrics.get("epsilon")
        if isinstance(epsilon, dict):
            rmt.setdefault("epsilon_by_family", epsilon)
        elif epsilon is not None:
            rmt.setdefault("epsilon_default", epsilon)
    for source, target in (
        ("epsilon_default", "epsilon_default"),
        ("margin_used", "margin"),
        ("deadband_used", "deadband"),
    ):
        if metrics.get(source) is not None:
            rmt.setdefault(target, metrics[source])
    if policy and not rmt.get("policy"):
        rmt["policy"] = policy
    rec["rmt"] = rmt


def _merge_variance_record(
    rec: dict[str, Any], metrics: dict[str, Any], policy: dict[str, Any]
) -> None:
    variance = _record_section(rec, "variance")
    for key in (
        "predictive_gate",
        "ab_windows_used",
        "deadband",
        "min_gain",
        "min_effect_lognll",
        "calibration",
        "calibration_stats",
    ):
        if metrics.get(key) is not None:
            variance.setdefault(key, metrics[key])
    if policy and not variance.get("policy"):
        variance["policy"] = policy
    rec["variance"] = variance


def _merge_record(cert: Any, report: Any) -> dict[str, Any] | None:
    rec: dict[str, Any] = {}
    if isinstance(cert, dict):
        rec = json.loads(json.dumps(cert))
    if not isinstance(report, dict):
        return rec or None

    metrics = report.get("metrics", {}) or {}
    pm = metrics.get("primary_metric", {}) or {}
    if not pm and "ppl_final" in metrics:
        pm = {"final": metrics.get("ppl_final"), "preview": metrics.get("ppl_preview")}
        try:
            pm["ratio_vs_baseline"] = float(pm["final"]) / max(
                float(pm["preview"]), 1e-10
            )
        except (TypeError, ValueError, OverflowError):
            pass
    if pm and not rec.get("primary_metric"):
        rec["primary_metric"] = pm

    guards = report.get("guards", []) or []
    for guard in guards:
        if not isinstance(guard, dict):
            continue
        name = str(guard.get("name", "")).lower()
        gmetrics = guard.get("metrics", {}) or {}
        gpolicy = guard.get("policy", {}) or {}

        if name == "spectral":
            _merge_spectral_record(rec, gmetrics, gpolicy, guard)
        elif name == "rmt":
            _merge_rmt_record(rec, gmetrics, gpolicy)
        elif name == "variance":
            _merge_variance_record(rec, gmetrics, gpolicy)

    return rec or None


def load_records(*, cal_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(cal_dir.glob("run_*")):
        cert = None
        report = None

        cert_path = run_dir / "evaluation.report.json"
        if cert_path.exists():
            cert = _load_json(cert_path)

        report_path = run_dir / "baseline_report.json"
        if not report_path.exists():
            report_files = list(run_dir.glob("**/report*.json"))
            if report_files:
                report_path = report_files[0]
        if report_path.exists():
            report = _load_json(report_path)

        record = _merge_record(cert, report)
        if record:
            records.append(record)
    return records


def calibrate_drift(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    for rec in recs:
        pm = rec.get("primary_metric", {}) or {}
        ratio = None

        def _finite(val: Any) -> float | None:
            try:
                if val is None:
                    return None
                parsed = float(val)
            except (TypeError, ValueError, OverflowError):
                return None
            return parsed if math.isfinite(parsed) else None

        ratio = _finite(pm.get("preview_final_ratio"))
        if ratio is None:
            ratio = _finite(pm.get("drift"))
        if ratio is None:
            preview = _finite(pm.get("preview"))
            final = _finite(pm.get("final"))
            if preview is not None and final is not None and preview > 0:
                ratio = float(final) / max(float(preview), 1e-10)
        if ratio is not None:
            ratios.append(float(ratio))

    ratios = [r for r in ratios if math.isfinite(r)]
    single_run_margin = 0.01
    if len(ratios) < 2:
        base = ratios[0] if ratios else 1.0
        compatible = 0.95 <= float(base) <= 1.05
        lo = max(float(base) - single_run_margin, 1e-6)
        hi = float(base) + single_run_margin
        return {
            "mean": round(float(base), 4),
            "std": 0.0,
            "min": round(float(base), 4),
            "max": round(float(base), 4),
            "suggested_band": [round(lo, 4), round(hi, 4)],
            "band_compatible": compatible,
        }

    mean = statistics.mean(ratios)
    std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    band = [round(mean - 3.0 * std, 4), round(mean + 3.0 * std, 4)]
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(ratios), 4),
        "max": round(max(ratios), 4),
        "suggested_band": band,
        "band_compatible": 0.95 <= mean <= 1.05,
    }


def _spectral_margin(tier_name: str) -> float:
    return 0.10 if tier_name == "conservative" else 0.05


def _default_max_caps(tier_name: str) -> int:
    if tier_name == "conservative":
        return 3
    if tier_name == "aggressive":
        return 8
    return 5


def _allocate_budget(counts: dict[str, int], budget: int) -> dict[str, int]:
    if not counts or budget <= 0:
        return dict.fromkeys(counts, 0)
    total = sum(counts.values())
    if total <= 0:
        return dict.fromkeys(counts, 0)
    raw = {fam: budget * count / total for fam, count in counts.items()}
    alloc = {fam: int(round(val)) for fam, val in raw.items()}
    diff = budget - sum(alloc.values())
    if diff > 0:
        for fam in sorted(raw, key=lambda name: raw[name], reverse=True):
            if diff == 0:
                break
            alloc[fam] += 1
            diff -= 1
    elif diff < 0:
        for fam in sorted(raw, key=lambda name: raw[name]):
            if diff == 0:
                break
            if alloc.get(fam, 0) > 0:
                alloc[fam] -= 1
                diff += 1
    return alloc


def _proposed_spectral_caps(
    *,
    per_run_caps: dict[str, list[float]],
    q99_values: dict[str, list[float]],
    max_values: dict[str, list[float]],
    existing_caps: dict[str, float],
    margin: float,
) -> dict[str, dict[str, float]]:
    proposed: dict[str, dict[str, float]] = {}
    observed_families = sorted(set(q99_values) | set(max_values))
    if per_run_caps:
        for family, candidates in per_run_caps.items():
            if not candidates:
                continue
            base = max(candidates)
            if family in existing_caps:
                base = max(base, existing_caps[family])
            proposed[family] = {"kappa": round(base + margin, 3)}
    if per_run_caps or q99_values or max_values:
        for family in observed_families:
            if family in proposed:
                continue
            observed = q99_values.get(family, []) + max_values.get(family, [])
            if not observed:
                continue
            base = max(observed)
            if family in existing_caps:
                base = max(base, existing_caps[family])
            proposed[family] = {"kappa": round(base + margin, 3)}
        return proposed
    return {family: {"kappa": kappa} for family, kappa in existing_caps.items()}


def calibrate_spectral(
    recs: list[dict[str, Any]], *, tier: str
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    per_run_caps: dict[str, list[float]] = defaultdict(list)
    q99_values: dict[str, list[float]] = defaultdict(list)
    max_values: dict[str, list[float]] = defaultdict(list)
    existing_caps: dict[str, float] = {}
    sigma_quantile: float | None = None
    deadband: float | None = None
    max_caps: int | None = None

    for rec in recs:
        spec = rec.get("spectral", {}) or {}
        if not isinstance(spec, dict):
            continue
        policy = spec.get("policy", {}) if isinstance(spec.get("policy"), dict) else {}

        if sigma_quantile is None:
            sq = (
                policy.get("sigma_quantile")
                or policy.get("contraction")
                or policy.get("kappa")
                or spec.get("sigma_quantile")
                or (spec.get("summary") or {}).get("sigma_quantile")
            )
            sq_val = _safe_float(sq)
            if sq_val is not None:
                sigma_quantile = sq_val

        if deadband is None:
            db = (
                policy.get("deadband")
                or spec.get("deadband")
                or (spec.get("summary") or {}).get("deadband")
            )
            db_val = _safe_float(db)
            if db_val is not None:
                deadband = db_val

        if max_caps is None:
            mc = (
                policy.get("max_caps")
                or spec.get("max_caps")
                or (spec.get("summary") or {}).get("max_caps")
            )
            try:
                if mc is not None:
                    max_caps = int(mc)
            except (TypeError, ValueError, OverflowError):
                pass

        fam_caps = spec.get("family_caps", {})
        if not fam_caps and isinstance(policy.get("family_caps"), dict):
            fam_caps = policy.get("family_caps", {})
        if isinstance(fam_caps, dict):
            for fam, cap in fam_caps.items():
                try:
                    if isinstance(cap, dict):
                        cap = cap.get("kappa")
                    existing_caps[str(fam)] = float(cap)
                except (TypeError, ValueError, OverflowError):
                    pass

        z_map = spec.get("final_z_scores")
        fam_map = spec.get("module_family_map")
        if isinstance(z_map, dict) and isinstance(fam_map, dict):
            z_by_family: dict[str, list[float]] = defaultdict(list)
            for module, z in z_map.items():
                fam = fam_map.get(module)
                if fam is None:
                    continue
                z_val = _safe_float(z)
                if z_val is None:
                    continue
                z_by_family[str(fam)].append(abs(z_val))
            if z_by_family:
                counts = {fam: len(vals) for fam, vals in z_by_family.items() if vals}
                budget = (
                    max_caps
                    if isinstance(max_caps, int) and max_caps >= 0
                    else _default_max_caps(tier)
                )
                alloc = _allocate_budget(counts, budget)
                for fam, values in z_by_family.items():
                    if not values:
                        continue
                    values_sorted = sorted(values, reverse=True)
                    idx = max(0, min(alloc.get(fam, 1) - 1, len(values_sorted) - 1))
                    per_run_caps[fam].append(values_sorted[idx])

        fq = spec.get("family_z_quantiles", {})
        if not fq and isinstance(spec.get("family_z_summary"), dict):
            fq = spec.get("family_z_summary", {})
        if isinstance(fq, dict):
            for fam, stats in fq.items():
                if not isinstance(stats, dict):
                    continue
                val_q99 = _safe_float(stats.get("q99"))
                val_max = _safe_float(stats.get("max"))
                if val_q99 is not None:
                    q99_values[str(fam)].append(val_q99)
                if val_max is not None:
                    max_values[str(fam)].append(val_max)

    summary = {
        "families_seen": sorted(
            set(per_run_caps) | set(q99_values) | set(existing_caps)
        ),
        "sigma_quantile": sigma_quantile,
        "deadband": deadband,
        "max_caps": max_caps,
    }

    proposed_caps = _proposed_spectral_caps(
        per_run_caps=per_run_caps,
        q99_values=q99_values,
        max_values=max_values,
        existing_caps=existing_caps,
        margin=_spectral_margin(tier),
    )

    return summary, proposed_caps


def _rmt_quantile_for_tier(tier_name: str) -> float:
    if tier_name == "conservative":
        return 0.95
    return 0.9


def calibrate_rmt(
    recs: list[dict[str, Any]], *, tier: str
) -> tuple[dict[str, Any], dict[str, float]]:
    epsilon_samples: dict[str, list[float]] = defaultdict(list)
    observed_growths: dict[str, list[float]] = defaultdict(list)
    existing_eps: dict[str, float] = {}
    margin: float | None = None
    deadband: float | None = None

    for rec in recs:
        rmt = rec.get("rmt", {}) or {}
        if not isinstance(rmt, dict):
            continue
        policy = rmt.get("policy", {}) if isinstance(rmt.get("policy"), dict) else {}

        if margin is None:
            margin = _safe_float(policy.get("margin") or rmt.get("margin"))
        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or rmt.get("deadband"))

        eps_map = rmt.get("epsilon_by_family")
        if isinstance(eps_map, dict) and eps_map:
            for fam, eps in eps_map.items():
                eps_val = _safe_float(eps)
                if eps_val is not None:
                    existing_eps[str(fam)] = eps_val

        families = rmt.get("families") or {}
        if isinstance(families, dict):
            for fam, fam_block in families.items():
                if not isinstance(fam_block, dict):
                    continue
                eps_val = _safe_float(
                    fam_block.get("epsilon")
                    or fam_block.get("epsilon_default")
                    or fam_block.get("eps")
                )
                if eps_val is not None:
                    epsilon_samples[str(fam)].append(eps_val)

        base_risk = rmt.get("edge_risk_by_family_base") or {}
        current_risk = rmt.get("edge_risk_by_family") or {}
        if isinstance(base_risk, dict) and isinstance(current_risk, dict):
            for fam in set(base_risk) | set(current_risk):
                growth = _positive_relative_growth(
                    base_risk.get(fam),
                    current_risk.get(fam),
                )
                if growth is not None:
                    observed_growths[str(fam)].append(growth)

    summary: dict[str, Any] = {
        "margin": margin,
        "deadband": deadband,
        "growth_quantile": _rmt_quantile_for_tier(tier),
    }
    proposed_eps: dict[str, float] = {}
    q = summary["growth_quantile"]
    families_seen = set(epsilon_samples) | set(observed_growths) | set(existing_eps)
    for fam in families_seen:
        eps_val = float(existing_eps.get(fam, 0.0))

        samples = epsilon_samples.get(fam, [])
        vals = [x for x in samples if isinstance(x, float) and math.isfinite(x)]
        if vals:
            proposed = _quantile(vals, q)
            if proposed is not None:
                eps_val = max(eps_val, float(proposed))

        growth_samples = observed_growths.get(fam, [])
        growth_vals = [
            x for x in growth_samples if isinstance(x, float) and math.isfinite(x)
        ]
        if growth_vals:
            proposed_growth = _quantile(growth_vals, q)
            if proposed_growth is not None:
                eps_val = max(eps_val, float(proposed_growth))

        if eps_val > 0.0:
            proposed_eps[fam] = round(eps_val, 6)

    if not proposed_eps:
        proposed_eps = existing_eps

    return summary, proposed_eps


def calibrate_variance(recs: list[dict[str, Any]]) -> dict[str, Any]:
    deadband: float | None = None
    min_gain: float | None = None
    policy_min_effect: float | None = None
    min_effect_samples: list[float] = []
    variance_changes: list[float] = []

    for rec in recs:
        var = rec.get("variance", {}) or {}
        if not isinstance(var, dict):
            continue
        policy = var.get("policy", {}) if isinstance(var.get("policy"), dict) else {}

        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or var.get("deadband"))
        if min_gain is None:
            min_gain = _safe_float(
                policy.get("min_gain")
                or policy.get("min_rel_gain")
                or var.get("min_gain")
            )
        if policy_min_effect is None:
            policy_min_effect = _safe_float(
                policy.get("min_effect_lognll") or var.get("min_effect_lognll")
            )

        predictive = var.get("predictive_gate", {}) or {}
        delta_ci = predictive.get("delta_ci")
        if isinstance(delta_ci, (list, tuple)) and len(delta_ci) == 2:
            lo = _safe_float(delta_ci[0])
            hi = _safe_float(delta_ci[1])
            if lo is not None and hi is not None:
                width = abs(hi - lo) / 2.0
                if width > 0:
                    min_effect_samples.append(width)

        calib = var.get("calibration") or var.get("calibration_stats") or {}
        if isinstance(calib, dict):
            vchange = (
                calib.get("variance_change")
                or calib.get("delta")
                or calib.get("max_delta")
            )
            vchange_val = _safe_float(vchange)
            if vchange_val is not None:
                variance_changes.append(abs(vchange_val))

    result: dict[str, Any] = {}
    if deadband is None and variance_changes:
        result["deadband"] = round(max(variance_changes) * 1.1 + 0.01, 3)
    elif deadband is not None:
        result["deadband"] = deadband

    if min_effect_samples:
        proposed = _quantile(min_effect_samples, 0.95)
        if proposed is not None:
            result["min_effect_lognll"] = max(round(proposed, 4), 0.0009)
    elif policy_min_effect is not None:
        result["min_effect_lognll"] = policy_min_effect

    if min_gain is not None:
        result["min_gain"] = min_gain

    return result


def _spectral_max_caps_for_edit_type(edit_type: str) -> int:
    et = (edit_type or "").strip().lower()
    if et == "quant_rtn":
        return 15
    return 10


def _coerce_int_env(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def get_spectral_max_caps(edit_type: str) -> int:
    env_override = _coerce_int_env("PACK_SPECTRAL_MAX_CAPS")
    if env_override is not None:
        return env_override
    return _spectral_max_caps_for_edit_type(edit_type)


def _apply_spectral_max_caps(
    preset: dict[str, Any], *, edit_type: str | None, tier: str
) -> None:
    guards = preset.get("guards")
    if not isinstance(guards, dict):
        return
    spectral = guards.get("spectral")
    if not isinstance(spectral, dict):
        return

    base = spectral.get("max_caps")
    try:
        base_int = int(base) if base is not None else None
    except (TypeError, ValueError, OverflowError):
        base_int = None

    override = _coerce_int_env("PACK_SPECTRAL_MAX_CAPS")
    suggested = get_spectral_max_caps(edit_type or "")

    if override is not None:
        final = override
    else:
        final = suggested if base_int is None else max(base_int, suggested)
    spectral["max_caps"] = int(final)
