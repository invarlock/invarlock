from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

DEFAULT_CORE_GUARDS: tuple[str, ...] = (
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "primary_metric",
)
DEFAULT_CATEGORIES: tuple[str, ...] = ("clean", "stress", "error_injection")
_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_JSON_READ_ERRORS = (OSError, TypeError, ValueError)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pass"}
    return bool(value)


def _as_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except _COERCE_ERRORS:
            return default
    if isinstance(value, str):
        try:
            return int(value.strip())
        except _COERCE_ERRORS:
            return default
    return default


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except _COERCE_ERRORS:
            return default
        if not math.isfinite(v):
            return default
        return v
    return default


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except _JSON_READ_ERRORS as exc:
        raise ValueError(f"Failed to read JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _core_guard_order(verdict: dict[str, Any]) -> list[str]:
    raw = verdict.get("core_guard_order")
    if isinstance(raw, list) and all(isinstance(x, str) and x.strip() for x in raw):
        return [x.strip().lower() for x in raw]
    return list(DEFAULT_CORE_GUARDS)


def _record_flags(record: dict[str, Any], *, core_guards: list[str]) -> dict[str, bool]:
    flags = record.get("guard_flags")
    if not isinstance(flags, dict):
        flags = {}
    return {guard: _as_bool(flags.get(guard), default=False) for guard in core_guards}


def _core_signal_count(flags: dict[str, bool]) -> int:
    return sum(1 for v in flags.values() if v)


def _spectral_caps(record: dict[str, Any]) -> bool:
    return _as_int(record.get("spectral_caps_applied"), default=0) > 0


def _ve_signal(record: dict[str, Any]) -> bool:
    probe = record.get("ve_probe")
    if not isinstance(probe, dict):
        return False
    if _as_bool(probe.get("signal"), default=False):
        return True
    if _as_bool(probe.get("would_enable"), default=False):
        return True
    if _as_int(probe.get("proposed_scales"), default=0) > 0:
        return True
    gain = _as_float(probe.get("ab_gain"), default=None)
    return gain is not None and gain > 0.0


def _build_core_guard_table(
    records: list[dict[str, Any]], *, core_guards: list[str]
) -> list[dict[str, Any]]:
    pm = "primary_metric"
    rows: list[dict[str, Any]] = []
    for guard in core_guards:
        flagged = 0
        unique = 0
        flagged_without_pm: int | None = 0
        flagged_with_pm: int | None = 0
        for record in records:
            flags = _record_flags(record, core_guards=core_guards)
            if not flags.get(guard, False):
                continue
            flagged += 1
            if _core_signal_count(flags) == 1:
                unique += 1
            if guard == pm:
                flagged_without_pm = None
                flagged_with_pm = None
            else:
                if flags.get(pm, False):
                    assert flagged_with_pm is not None
                    flagged_with_pm += 1
                else:
                    assert flagged_without_pm is not None
                    flagged_without_pm += 1
        rows.append(
            {
                "guard": guard,
                "flagged": flagged,
                "unique": unique,
                "flagged_without_pm": flagged_without_pm,
                "flagged_with_pm": flagged_with_pm,
            }
        )
    return rows


def _build_intervention_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spectral_flags = [_spectral_caps(r) for r in records]
    ve_flags = [_ve_signal(r) for r in records]
    rows = []
    for name, flags, other in (
        ("spectral_caps", spectral_flags, ve_flags),
        ("ve_signal", ve_flags, spectral_flags),
    ):
        flagged = sum(1 for v in flags if v)
        unique = sum(1 for v, o in zip(flags, other, strict=True) if v and not o)
        rows.append({"signal": name, "flagged": flagged, "unique": unique})
    return rows


def _build_non_pm_without_pm(records: list[dict[str, Any]], *, core_guards: list[str]):
    pm = "primary_metric"
    non_pm = [g for g in core_guards if g != pm]
    counts_by_k: Counter[int] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    any_non_pm_without_pm = 0
    multi_non_pm_without_pm = 0
    for record in records:
        flags = _record_flags(record, core_guards=core_guards)
        if flags.get(pm, False):
            continue
        k = sum(1 for g in non_pm if flags.get(g, False))
        if k:
            any_non_pm_without_pm += 1
            counts_by_k[k] += 1
        if k >= 2:
            multi_non_pm_without_pm += 1
            present = [g for g in non_pm if flags.get(g, False)]
            for a, b in combinations(present, 2):
                pair_counts[(a, b)] += 1
    return {
        "any_non_pm_without_pm": any_non_pm_without_pm,
        "multi_non_pm_without_pm": multi_non_pm_without_pm,
        "by_k_non_pm_without_pm": {str(k): v for k, v in sorted(counts_by_k.items())},
        "pairs_non_pm_without_pm": {
            f"{a}+{b}": v for (a, b), v in sorted(pair_counts.items())
        },
    }


def _build_by_category(
    records: list[dict[str, Any]], *, core_guards: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for category in DEFAULT_CATEGORIES:
        cat_records = [r for r in records if r.get("category") == category]
        out[category] = {
            "records_total": len(cat_records),
            "core_guards": _build_core_guard_table(
                cat_records, core_guards=core_guards
            ),
            "interventions": _build_intervention_table(cat_records),
            "non_pm_without_pm": _build_non_pm_without_pm(
                cat_records, core_guards=core_guards
            ),
        }
    return out


def _build_by_scenario(
    records: list[dict[str, Any]], *, core_guards: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = record.get("category")
        name = record.get("name")
        if not isinstance(category, str) or not isinstance(name, str):
            continue
        groups[(category, name)].append(record)

    rows: list[dict[str, Any]] = []
    for (category, name), recs in sorted(groups.items()):
        guard_counts: dict[str, int] = dict.fromkeys(core_guards, 0)
        spectral_caps = 0
        ve_signal = 0
        passed = 0
        for record in recs:
            flags = _record_flags(record, core_guards=core_guards)
            for guard, value in flags.items():
                guard_counts[guard] += 1 if value else 0
            spectral_caps += 1 if _spectral_caps(record) else 0
            ve_signal += 1 if _ve_signal(record) else 0
            passed += 1 if _as_bool(record.get("passed"), default=False) else 0
        rows.append(
            {
                "category": category,
                "scenario": name,
                "records_total": len(recs),
                "passed": passed,
                "core_guards_flagged": guard_counts,
                "interventions_flagged": {
                    "spectral_caps": spectral_caps,
                    "ve_signal": ve_signal,
                },
            }
        )
    return rows


def _render_markdown(payload: dict[str, Any]) -> str:
    core_rows = payload["core_guards"]
    intervention_rows = payload["interventions"]
    non_pm = payload["non_pm_without_pm"]

    lines: list[str] = []
    meta = payload.get("meta") or {}
    lines.append(f"# Proof Pack Verdict Tables ({meta.get('verdict')})")
    if meta.get("source"):
        lines.append(f"- Source: `{meta.get('source')}`")
    lines.append(f"- Records: {meta.get('records_total')}")
    lines.append("")

    lines.append("## Core Guards (Signals)")
    lines.append("")
    lines.append("| guard | flagged | unique | flagged_without_pm | flagged_with_pm |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in core_rows:
        guard = row["guard"]
        fwopm = row.get("flagged_without_pm")
        fwp = row.get("flagged_with_pm")
        lines.append(
            f"| {guard} | {row['flagged']} | {row['unique']} |"
            f" {fwopm if fwopm is not None else '-'} | {fwp if fwp is not None else '-'} |"
        )
    lines.append("")

    lines.append("## Interventions (Mitigations)")
    lines.append("")
    lines.append("| signal | flagged | unique |")
    lines.append("|---|---:|---:|")
    for row in intervention_rows:
        lines.append(f"| {row['signal']} | {row['flagged']} | {row['unique']} |")
    lines.append("")

    lines.append("## Non-PM Signals Without PM")
    lines.append("")
    lines.append(f"- any_non_pm_without_pm: {non_pm['any_non_pm_without_pm']}")
    lines.append(f"- multi_non_pm_without_pm: {non_pm['multi_non_pm_without_pm']}")
    if non_pm.get("by_k_non_pm_without_pm"):
        lines.append(
            f"- by_k_non_pm_without_pm: {json.dumps(non_pm['by_k_non_pm_without_pm'])}"
        )
    lines.append("")

    lines.append("## By Category")
    lines.append("")
    by_cat = payload.get("by_category") or {}
    for cat in DEFAULT_CATEGORIES:
        cat_payload = by_cat.get(cat) or {}
        lines.append(f"### {cat} (records={cat_payload.get('records_total', 0)})")
        lines.append("")
        lines.append("| guard | flagged | unique |")
        lines.append("|---|---:|---:|")
        for row in cat_payload.get("core_guards", []):
            lines.append(f"| {row['guard']} | {row['flagged']} | {row['unique']} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate summary tables from proof-pack final_verdict.json"
    )
    parser.add_argument(
        "--verdict", type=Path, required=True, help="Path to final_verdict.json"
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--out-md", type=Path, help="Optional output Markdown path")
    args = parser.parse_args(argv)

    verdict = _load_json(args.verdict)
    records = verdict.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Verdict missing records list: {args.verdict}")
    core_guards = _core_guard_order(verdict)

    payload = {
        "meta": {
            "source": str(args.verdict),
            "verdict": verdict.get("verdict"),
            "records_total": len(records),
            "models_total": ((verdict.get("counts") or {}).get("models_total")),
        },
        "core_guards": _build_core_guard_table(records, core_guards=core_guards),
        "interventions": _build_intervention_table(records),
        "non_pm_without_pm": _build_non_pm_without_pm(records, core_guards=core_guards),
        "by_category": _build_by_category(records, core_guards=core_guards),
        "by_scenario": _build_by_scenario(records, core_guards=core_guards),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(payload), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
