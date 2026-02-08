#!/usr/bin/env python3
"""
summarize_artifacts.py
=====================

Summarize verifier-carrying artifacts (S1) into analysis-friendly JSONL/CSV.

Goal: make it easy to run F4 analyses without hand-parsing deeply nested JSON.
This script intentionally avoids heavy dependencies; compute-heavy stats live
outside this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_artifact(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("schema_version") == "verifier_carrying_artifact.v1"
    )


def _get(obj: Any, *path: str) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _as_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _as_int(val: Any) -> int | None:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _iter_artifact_paths(inputs: list[Path]) -> Iterable[Path]:
    for p in inputs:
        if p.is_dir():
            yield from sorted(p.rglob("*.json"))
        else:
            yield p


def _load_evaluation_report(
    artifact: dict[str, Any], *, strict: bool
) -> dict[str, Any] | None:
    embedded = _get(
        artifact, "guard_evidence", "invarlock", "evaluation_report", "embedded"
    )
    if isinstance(embedded, dict):
        return embedded

    path_str = _get(
        artifact, "guard_evidence", "invarlock", "evaluation_report", "path"
    )
    if isinstance(path_str, str) and path_str:
        p = Path(path_str)
        if p.exists():
            obj = _read_json(p)
            return obj if isinstance(obj, dict) else None

    if strict:
        raise ValueError(
            "artifact missing embedded evaluation report and readable path"
        )
    return None


def _trace_groups(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group traces by (verifier.name, prompt_set.digest_sha256) preserving order
    within the artifact. For each group, interpret trace[0] as baseline and
    trace[1] as edited when present.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for tr in traces:
        name = _get(tr, "verifier", "name")
        digest = _get(tr, "trace_contract", "prompt_set", "digest_sha256")
        if not (isinstance(name, str) and isinstance(digest, str)):
            continue
        key = (name, digest)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(tr)
    return [
        {"verifier_name": k[0], "prompt_digest": k[1], "traces": groups[k]}
        for k in order
    ]


def _extract_invarlock_features(eval_report: dict[str, Any] | None) -> dict[str, Any]:
    if not eval_report:
        return {}
    pm = eval_report.get("primary_metric") if isinstance(eval_report, dict) else None
    v = eval_report.get("validation") if isinstance(eval_report, dict) else None
    spectral = eval_report.get("spectral") if isinstance(eval_report, dict) else None
    rmt = eval_report.get("rmt") if isinstance(eval_report, dict) else None
    invariants = (
        eval_report.get("invariants") if isinstance(eval_report, dict) else None
    )
    variance = eval_report.get("variance") if isinstance(eval_report, dict) else None

    out: dict[str, Any] = {
        "invarlock_model_id": _get(eval_report, "meta", "model_id"),
        "invarlock_profile": _get(eval_report, "meta", "profile"),
        "invarlock_adapter": _get(eval_report, "meta", "adapter"),
        "invarlock_edit_name": eval_report.get("edit_name")
        or _get(eval_report, "edit", "name"),
        "primary_metric_kind": _get(pm, "kind") if isinstance(pm, dict) else None,
        "primary_metric_ratio_vs_baseline": _as_float(_get(pm, "ratio_vs_baseline")),
        "primary_metric_degraded": bool(_get(pm, "degraded"))
        if isinstance(pm, dict)
        else None,
        "gate_primary_metric_acceptable": _get(v, "primary_metric_acceptable")
        if isinstance(v, dict)
        else None,
        "gate_invariants_pass": _get(v, "invariants_pass")
        if isinstance(v, dict)
        else None,
        "gate_spectral_stable": _get(v, "spectral_stable")
        if isinstance(v, dict)
        else None,
        "gate_rmt_stable": _get(v, "rmt_stable") if isinstance(v, dict) else None,
        "invariants_status": _get(invariants, "status")
        if isinstance(invariants, dict)
        else None,
        "spectral_status": _get(spectral, "summary", "status")
        if isinstance(spectral, dict)
        else None,
        "spectral_stability_score": _as_float(
            _get(spectral, "summary", "stability_score")
        )
        if isinstance(spectral, dict)
        else None,
        "rmt_status": _get(rmt, "status") if isinstance(rmt, dict) else None,
        "rmt_max_edge_ratio": _as_float(_get(rmt, "max_edge_ratio"))
        if isinstance(rmt, dict)
        else None,
        "variance_enabled": _get(variance, "enabled")
        if isinstance(variance, dict)
        else None,
    }
    return out


def _extract_trace_summary(tr: dict[str, Any]) -> dict[str, Any]:
    summary = _get(tr, "results", "summary")
    return {
        "trace_model_revision": _get(tr, "trace_contract", "model", "revision"),
        "metric_name": _get(summary, "metric_name"),
        "k": _as_int(_get(summary, "k")),
        "n_samples_per_case": _as_int(_get(summary, "n_samples_per_case")),
        "n_total": _as_int(_get(summary, "n_total")),
        "n_pass": _as_int(_get(summary, "n_pass")),
        "pass_rate": _as_float(_get(summary, "pass_rate")),
    }


def _rows_for_artifact(
    artifact_path: Path, artifact: dict[str, Any], *, strict: bool
) -> list[dict[str, Any]]:
    eval_report = _load_evaluation_report(artifact, strict=strict)
    base = {
        "artifact_path": str(artifact_path),
        **_extract_invarlock_features(eval_report),
    }

    traces_raw = artifact.get("verifier_traces")
    traces: list[dict[str, Any]] = (
        [t for t in traces_raw if isinstance(t, dict)]
        if isinstance(traces_raw, list)
        else []
    )
    out_rows: list[dict[str, Any]] = []
    for g in _trace_groups(traces):
        rows = dict(base)
        rows["verifier_name"] = g["verifier_name"]
        rows["prompt_digest"] = g["prompt_digest"]

        trs = g["traces"]
        b = _extract_trace_summary(trs[0]) if len(trs) >= 1 else {}
        e = _extract_trace_summary(trs[1]) if len(trs) >= 2 else {}

        rows["baseline_model_revision"] = b.get("trace_model_revision")
        rows["edited_model_revision"] = e.get("trace_model_revision")
        rows["baseline_pass_rate"] = b.get("pass_rate")
        rows["edited_pass_rate"] = e.get("pass_rate")
        rows["delta_pass_rate"] = (
            (e.get("pass_rate") - b.get("pass_rate"))
            if isinstance(b.get("pass_rate"), float)
            and isinstance(e.get("pass_rate"), float)
            else None
        )
        # Include the (shared) metric labels from baseline trace.
        rows["verifier_metric_name"] = b.get("metric_name")
        rows["verifier_k"] = b.get("k")
        rows["verifier_n_samples_per_case"] = b.get("n_samples_per_case")
        rows["verifier_n_total"] = b.get("n_total")

        out_rows.append(rows)

    return out_rows


def _write_jsonl(out_fp: Any, rows: list[dict[str, Any]]) -> None:  # noqa: ANN401
    for r in rows:
        out_fp.write(json.dumps(r, ensure_ascii=True) + "\n")


def _write_csv(out_fp: Any, rows: list[dict[str, Any]]) -> None:  # noqa: ANN401
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    w = csv.DictWriter(out_fp, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Artifact JSON file(s) or directory(ies) to scan recursively for *.json.",
    )
    p.add_argument("--out", type=Path, help="Output path (default: stdout).")
    p.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on artifacts missing embedded evaluation report and readable path.",
    )
    args = p.parse_args(argv)

    artifacts: list[Path] = []
    for ap in _iter_artifact_paths(list(args.inputs)):
        if ap.name.endswith(".example.json"):
            continue
        artifacts.append(ap)
    if not artifacts:
        print("No candidate *.json artifact files found.", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for ap in artifacts:
        try:
            obj = _read_json(ap)
        except Exception:
            continue
        if not _is_artifact(obj):
            continue
        try:
            rows.extend(_rows_for_artifact(ap, obj, strict=bool(args.strict)))
        except Exception as exc:
            if args.strict:
                raise
            print(f"Skipping artifact {ap}: {exc}", file=sys.stderr)
            continue

    if not rows:
        print("No verifier-carrying artifacts found in inputs.", file=sys.stderr)
        return 2

    out_fp = (
        sys.stdout
        if args.out is None
        else args.out.open("w", encoding="utf-8", newline="")
    )
    try:
        if args.format == "jsonl":
            _write_jsonl(out_fp, rows)
        else:
            _write_csv(out_fp, rows)
    finally:
        if args.out is not None:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
