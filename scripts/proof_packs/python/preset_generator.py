from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_AVAILABLE = False


def get_default_guards_order() -> list[str]:
    return ["invariants", "spectral", "rmt", "variance", "invariants"]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    except Exception:
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
                cfg = yaml.safe_load(cfg_path.read_text())  # type: ignore[attr-defined]
            except Exception:
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
        except Exception:
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
            spec = (
                rec.get("spectral", {}) if isinstance(rec.get("spectral"), dict) else {}
            )
            if gmetrics.get("family_z_quantiles"):
                spec.setdefault(
                    "family_z_quantiles", gmetrics.get("family_z_quantiles")
                )
            if gmetrics.get("family_z_summary"):
                spec.setdefault("family_z_summary", gmetrics.get("family_z_summary"))
            if gmetrics.get("family_caps"):
                spec.setdefault("family_caps", gmetrics.get("family_caps"))
            if gmetrics.get("sigma_quantile") is not None:
                spec.setdefault("sigma_quantile", gmetrics.get("sigma_quantile"))
            if gmetrics.get("deadband") is not None:
                spec.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("max_caps") is not None:
                spec.setdefault("max_caps", gmetrics.get("max_caps"))
            if gmetrics.get("families"):
                spec.setdefault("families", gmetrics.get("families"))
            if gmetrics.get("family_stats"):
                spec.setdefault("families", gmetrics.get("family_stats"))
            z_scores = guard.get("final_z_scores") or gmetrics.get("final_z_scores")
            if isinstance(z_scores, dict):
                spec["final_z_scores"] = z_scores
            fam_map = guard.get("module_family_map") or gmetrics.get(
                "module_family_map"
            )
            if isinstance(fam_map, dict):
                spec["module_family_map"] = fam_map
            if gpolicy and not spec.get("policy"):
                spec["policy"] = gpolicy
            rec["spectral"] = spec

        elif name == "rmt":
            rmt = rec.get("rmt", {}) if isinstance(rec.get("rmt"), dict) else {}
            for key in (
                "outliers_per_family",
                "baseline_outliers_per_family",
                "families",
            ):
                val = gmetrics.get(key)
                if isinstance(val, dict) and val:
                    rmt.setdefault(key, val)
            epsilon_by_family = gmetrics.get("epsilon_by_family")
            if epsilon_by_family:
                rmt.setdefault("epsilon_by_family", epsilon_by_family)
            else:
                epsilon = gmetrics.get("epsilon")
                if epsilon is not None:
                    if isinstance(epsilon, dict):
                        rmt.setdefault("epsilon_by_family", epsilon)
                    else:
                        rmt.setdefault("epsilon_default", epsilon)
            if gmetrics.get("epsilon_default") is not None:
                rmt.setdefault("epsilon_default", gmetrics.get("epsilon_default"))
            if gmetrics.get("margin_used") is not None:
                rmt.setdefault("margin", gmetrics.get("margin_used"))
            if gmetrics.get("deadband_used") is not None:
                rmt.setdefault("deadband", gmetrics.get("deadband_used"))
            if gpolicy and not rmt.get("policy"):
                rmt["policy"] = gpolicy
            rec["rmt"] = rmt

        elif name == "variance":
            var = (
                rec.get("variance", {}) if isinstance(rec.get("variance"), dict) else {}
            )
            if gmetrics.get("predictive_gate") is not None:
                var.setdefault("predictive_gate", gmetrics.get("predictive_gate"))
            if gmetrics.get("ab_windows_used") is not None:
                var.setdefault("ab_windows_used", gmetrics.get("ab_windows_used"))
            if gmetrics.get("deadband") is not None:
                var.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("min_gain") is not None:
                var.setdefault("min_gain", gmetrics.get("min_gain"))
            if gmetrics.get("min_effect_lognll") is not None:
                var.setdefault("min_effect_lognll", gmetrics.get("min_effect_lognll"))
            if gmetrics.get("calibration") is not None:
                var.setdefault("calibration", gmetrics.get("calibration"))
            if gmetrics.get("calibration_stats") is not None:
                var.setdefault("calibration_stats", gmetrics.get("calibration_stats"))
            if gpolicy and not var.get("policy"):
                var["policy"] = gpolicy
            rec["variance"] = var

    return rec or None


def load_records(*, cal_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(cal_dir.glob("run_*")):
        cert = None
        report = None

        cert_path = run_dir / "evaluation.cert.json"
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
        ratio = pm.get("ratio_vs_baseline") or pm.get("drift")
        if ratio is None:
            preview = pm.get("preview")
            final = pm.get("final")
            if preview is not None and final is not None:
                try:
                    ratio = float(final) / max(float(preview), 1e-10)
                except Exception:
                    ratio = None
        if ratio is not None:
            try:
                ratios.append(float(ratio))
            except Exception:
                pass

    ratios = [r for r in ratios if math.isfinite(r)]
    if len(ratios) < 2:
        base = ratios[0] if ratios else 1.0
        return {
            "mean": float(base),
            "std": 0.0,
            "min": float(base),
            "max": float(base),
            "suggested_band": [0.95, 1.05],
            "band_compatible": True,
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
        for fam in sorted(raw, key=raw.get, reverse=True):
            if diff == 0:
                break
            alloc[fam] += 1
            diff -= 1
    elif diff < 0:
        for fam in sorted(raw, key=raw.get):
            if diff == 0:
                break
            if alloc.get(fam, 0) > 0:
                alloc[fam] -= 1
                diff += 1
    return alloc


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
            except Exception:
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
                except Exception:
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

    proposed_caps: dict[str, dict[str, float]] = {}
    margin = _spectral_margin(tier)
    if per_run_caps:
        for fam, candidates in per_run_caps.items():
            if not candidates:
                continue
            base = max(candidates)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
        for fam in sorted(set(q99_values) | set(max_values)):
            if fam in proposed_caps:
                continue
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    elif q99_values or max_values:
        for fam in sorted(set(q99_values) | set(max_values)):
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    else:
        for fam, kappa in existing_caps.items():
            proposed_caps[fam] = {"kappa": kappa}

    return summary, proposed_caps


def _rmt_quantile_for_tier(tier_name: str) -> float:
    if tier_name == "conservative":
        return 0.95
    return 0.9


def calibrate_rmt(
    recs: list[dict[str, Any]], *, tier: str
) -> tuple[dict[str, Any], dict[str, float]]:
    epsilon_samples: dict[str, list[float]] = defaultdict(list)
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

    summary: dict[str, Any] = {"margin": margin, "deadband": deadband}
    proposed_eps: dict[str, float] = {}
    q = _rmt_quantile_for_tier(tier)
    for fam, samples in epsilon_samples.items():
        vals = [x for x in samples if isinstance(x, float) and math.isfinite(x)]
        if not vals:
            continue
        proposed = _quantile(vals, q)
        if proposed is None:
            continue
        eps_val = round(float(proposed), 6)
        if fam in existing_eps:
            eps_val = max(eps_val, float(existing_eps[fam]))
        proposed_eps[fam] = eps_val

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
    if et in {"quant_rtn", "fp8_quant"}:
        return 15
    if et in {"lowrank_svd"}:
        return 25
    return 10


def _coerce_int_env(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
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
    except Exception:
        base_int = None

    override = _coerce_int_env("PACK_SPECTRAL_MAX_CAPS")
    suggested = get_spectral_max_caps(edit_type or "")

    if override is not None:
        final = override
    else:
        final = suggested if base_int is None else max(base_int, suggested)
    spectral["max_caps"] = int(final)


def generate_preset(
    *,
    cal_dir: Path,
    preset_file: Path,
    model_name: str,
    model_path: str,
    tier: str,
    dataset_provider: str,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    edit_types: list[str],
) -> tuple[Path, Path, list[Path]]:
    guards_order, assurance_cfg = _load_guard_order_and_assurance(cal_dir)
    enabled_guards = set(guards_order)

    records = load_records(cal_dir=cal_dir)
    if not records:
        raise SystemExit(
            "ERROR: No calibration records found; cannot create valid preset"
        )

    drift_stats = calibrate_drift(records)
    spectral_summary, spectral_caps = calibrate_spectral(records, tier=tier)
    rmt_summary, rmt_epsilon = calibrate_rmt(records, tier=tier)
    variance_config = calibrate_variance(records)

    preset: dict[str, Any] = {
        "_calibration_meta": {
            "model_name": model_name,
            "num_runs": len(records),
            "tier": tier,
            "drift_mean": drift_stats.get("mean"),
            "drift_std": drift_stats.get("std"),
            "drift_band_compatible": drift_stats.get("band_compatible"),
            "suggested_drift_band": drift_stats.get("suggested_band"),
        },
        "model": {"id": model_path},
        "dataset": {
            "provider": dataset_provider,
            "split": "validation",
            "seq_len": int(seq_len),
            "stride": int(stride),
            "preview_n": int(preview_n),
            "final_n": int(final_n),
            "seed": 42,
        },
        "guards": {"order": guards_order},
    }

    if isinstance(assurance_cfg, dict) and assurance_cfg:
        preset["assurance"] = assurance_cfg

    spectral: dict[str, Any] = {}
    if spectral_caps:
        spectral["family_caps"] = spectral_caps
    if spectral_summary.get("sigma_quantile") is not None:
        spectral["sigma_quantile"] = spectral_summary["sigma_quantile"]
    if spectral_summary.get("deadband") is not None:
        spectral["deadband"] = spectral_summary["deadband"]
    if spectral_summary.get("max_caps") is not None:
        spectral["max_caps"] = spectral_summary["max_caps"]
    if "spectral" in enabled_guards and spectral:
        preset["guards"]["spectral"] = spectral

    rmt: dict[str, Any] = {}
    if rmt_epsilon:
        rmt["epsilon_by_family"] = rmt_epsilon
    if rmt_summary.get("margin") is not None:
        rmt["margin"] = rmt_summary["margin"]
    if rmt_summary.get("deadband") is not None:
        rmt["deadband"] = rmt_summary["deadband"]
    if "rmt" in enabled_guards and rmt:
        preset["guards"]["rmt"] = rmt

    if "variance" in enabled_guards and variance_config:
        preset["guards"]["variance"] = variance_config

    stats_path = cal_dir / "calibration_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "guards_order": guards_order,
                "assurance": assurance_cfg,
                "drift": drift_stats,
                "spectral": {**spectral_summary, "family_caps": spectral_caps},
                "rmt": {**rmt_summary, "epsilon_by_family": rmt_epsilon},
                "variance": variance_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    preset_file.parent.mkdir(parents=True, exist_ok=True)
    if _YAML_AVAILABLE and preset_file.suffix.lower() in {".yaml", ".yml"}:
        preset_file.write_text(
            yaml.safe_dump(preset, sort_keys=False)  # type: ignore[attr-defined]
        )
    else:
        preset_file = preset_file.with_suffix(".json")
        preset_file.write_text(json.dumps(preset, indent=2) + "\n")

    derived_files: list[Path] = []
    for edit_type in edit_types:
        derived = copy.deepcopy(preset)
        meta = derived.get("_calibration_meta")
        if isinstance(meta, dict):
            meta["edit_type"] = edit_type
        _apply_spectral_max_caps(derived, edit_type=edit_type, tier=tier)
        out = preset_file.with_name(
            f"{preset_file.stem}__{edit_type}{preset_file.suffix}"
        )
        if _YAML_AVAILABLE and out.suffix.lower() in {".yaml", ".yml"}:
            out.write_text(
                yaml.safe_dump(derived, sort_keys=False)  # type: ignore[attr-defined]
            )
        else:
            out = out.with_suffix(".json")
            out.write_text(json.dumps(derived, indent=2) + "\n")
        derived_files.append(out)

    return preset_file, stats_path, derived_files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibrated proof-pack presets"
    )
    parser.add_argument("--cal-dir", required=True)
    parser.add_argument("--preset-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tier", default=os.environ.get("INVARLOCK_TIER", "balanced"))
    parser.add_argument(
        "--dataset-provider", default=os.environ.get("INVARLOCK_DATASET", "wikitext2")
    )
    parser.add_argument(
        "--seq-len", type=int, default=int(os.environ.get("PRESET_SEQ_LEN", "1024"))
    )
    parser.add_argument(
        "--stride", type=int, default=int(os.environ.get("PRESET_STRIDE", "512"))
    )
    parser.add_argument(
        "--preview-n", type=int, default=int(os.environ.get("PRESET_PREVIEW_N", "40"))
    )
    parser.add_argument(
        "--final-n", type=int, default=int(os.environ.get("PRESET_FINAL_N", "40"))
    )
    parser.add_argument(
        "--edit-types",
        default=os.environ.get("PACK_PRESET_EDIT_TYPES", ""),
        help="Comma-separated edit types for derived presets (default: core types).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    edit_types = [e.strip() for e in str(args.edit_types).split(",") if e.strip()]
    if not edit_types:
        edit_types = ["quant_rtn", "fp8_quant", "magnitude_prune", "lowrank_svd"]

    preset_file, stats_path, derived_files = generate_preset(
        cal_dir=Path(args.cal_dir),
        preset_file=Path(args.preset_file),
        model_name=str(args.model_name),
        model_path=str(args.model_path),
        tier=str(args.tier).strip().lower(),
        dataset_provider=str(args.dataset_provider),
        seq_len=int(args.seq_len),
        stride=int(args.stride),
        preview_n=int(args.preview_n),
        final_n=int(args.final_n),
        edit_types=edit_types,
    )
    print(f"Saved preset to {preset_file}")
    print(f"Saved stats to {stats_path}")
    for path in derived_files:
        print(f"Saved derived preset to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
