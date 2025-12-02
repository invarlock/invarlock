from __future__ import annotations

import ast
import math
from collections.abc import Iterable
from typing import Any


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value - round(value)) > 1e-9:
            return None
        return int(round(value))
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sanitize_seed_bundle(bundle: Any, fallback: int | None) -> dict[str, int | None]:
    fallback_int = _coerce_int(fallback)
    sanitized = {
        "python": fallback_int,
        "numpy": fallback_int,
        "torch": fallback_int,
    }
    if isinstance(bundle, dict):
        for key in ("python", "numpy", "torch"):
            coerced = _coerce_int(bundle.get(key))
            if coerced is None and bundle.get(key) is None:
                sanitized[key] = None
            elif coerced is not None:
                sanitized[key] = coerced
    return sanitized


def _infer_scope_from_modules(modules: Iterable[str]) -> str:
    modules = list(modules)
    if not modules:
        return "unknown"
    families: set[str] = set()
    for module in modules:
        name = str(module).lower()
        if ".attn" in name or "attention" in name:
            families.add("attn")
        if any(token in name for token in (".mlp", ".ffn", "feed_forward", "fc")):
            families.add("ffn")
        if any(token in name for token in (".wte", ".embed", "embedding")):
            families.add("embed")
    if not families:
        return "all"
    if len(families) == 1:
        return next(iter(families))
    return "+".join(sorted(families))


def _coerce_interval(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return float("nan"), float("nan")
        value = parsed
    if isinstance(value, list | tuple) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return float("nan"), float("nan")
    return float("nan"), float("nan")


def _weighted_mean(values: list[Any], weights: list[Any]) -> float:
    # Types guarantee lists; validate lengths and contents for robustness
    if len(values) != len(weights) or not values:
        return float("nan")
    sw = 0.0
    swx = 0.0
    for v, w in zip(values, weights, strict=False):
        try:
            vf = float(v)
            wf = float(w)
        except Exception:
            continue
        if not math.isfinite(vf) or not math.isfinite(wf) or wf <= 0:
            continue
        sw += wf
        swx += wf * vf
    if sw <= 0.0:
        return float("nan")
    return float(swx / sw)


def _get_section(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _get_mapping(source: Any, key: str) -> dict[str, Any]:
    value = _get_section(source, key)
    return value if isinstance(value, dict) else {}


def _iter_guard_entries(report: Any) -> list[dict[str, Any]]:
    guards = _get_section(report, "guards")
    if isinstance(guards, list):
        return [g for g in guards if isinstance(g, dict)]
    if isinstance(guards, dict):
        entries: list[dict[str, Any]] = []
        for name, payload in guards.items():
            entry: dict[str, Any] = {"name": name}
            if isinstance(payload, dict):
                entry.update(payload)
            entries.append(entry)
        return entries
    return []


def _pair_logloss_windows(
    run_windows: Any,
    baseline_windows: Any,
) -> tuple[list[float], list[float]] | None:
    # Defensive input checks
    if not isinstance(run_windows, dict) or not isinstance(baseline_windows, dict):
        return None
    # Inputs are dicts; guard for expected list payloads
    run_ids = run_windows.get("window_ids")
    baseline_ids = baseline_windows.get("window_ids")
    run_logloss = run_windows.get("logloss")
    baseline_logloss = baseline_windows.get("logloss")
    if not (
        isinstance(run_ids, list)
        and isinstance(baseline_ids, list)
        and isinstance(run_logloss, list)
        and isinstance(baseline_logloss, list)
    ):
        return None
    baseline_map = {
        int(b_id): float(log_val)
        for b_id, log_val in zip(baseline_ids, baseline_logloss, strict=False)
        if isinstance(b_id, int | float) and isinstance(log_val, int | float)
    }
    paired_run: list[float] = []
    paired_baseline: list[float] = []
    for r_id, log_val in zip(run_ids, run_logloss, strict=False):
        if not isinstance(r_id, int | float) or not isinstance(log_val, int | float):
            continue
        key = int(r_id)
        if key in baseline_map:
            paired_run.append(float(log_val))
            paired_baseline.append(baseline_map[key])
    if len(paired_run) >= 2 and len(paired_run) == len(paired_baseline):
        return paired_run, paired_baseline
    return None


__all__ = [
    "_coerce_int",
    "_sanitize_seed_bundle",
    "_infer_scope_from_modules",
    "_coerce_interval",
    "_weighted_mean",
    "_get_section",
    "_get_mapping",
    "_iter_guard_entries",
    "_pair_logloss_windows",
]
