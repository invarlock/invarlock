from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_base_report(source_report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source_report, dict):
        raise ValueError("source report is required to build a structural failure cert")

    meta = source_report.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    data = source_report.get("data")
    if not isinstance(data, dict):
        data = {}

    edit = source_report.get("edit")
    if not isinstance(edit, dict):
        edit = {}

    artifacts = source_report.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}

    evaluation_windows = source_report.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        evaluation_windows = {}

    def _non_negative_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed >= 0 else 0

    preview_windows = _non_negative_int(data.get("preview_n"))
    final_windows = _non_negative_int(data.get("final_n"))

    return {
        "schema_version": "v1",
        "run_id": str(source_report.get("run_id") or "run"),
        "edit_name": str(edit.get("name") or "unknown"),
        "plugins": {
            "adapters": [],
            "edits": [],
            "guards": [],
        },
        "meta": copy.deepcopy(meta),
        "dataset": {
            "provider": str(data.get("dataset") or "unknown"),
            "seq_len": _non_negative_int(data.get("seq_len")),
            "hash": {
                "preview": "",
                "final": "",
                "dataset": None,
                "preview_tokens": None,
                "final_tokens": None,
                "total_tokens": 0,
                "source": "config_fallback",
            },
            "windows": {
                "preview": preview_windows,
                "final": final_windows,
                "seed": meta.get("seed"),
                "stats": {},
            },
        },
        "primary_metric": _metric_section(source_report),
        "artifacts": copy.deepcopy(artifacts),
        "evaluation_windows": copy.deepcopy(evaluation_windows),
        "flags": copy.deepcopy(source_report.get("flags", {})),
    }


def _metric_section(source_report: dict[str, Any] | None) -> dict[str, Any]:
    metrics = {}
    if isinstance(source_report, dict):
        raw_metrics = source_report.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
    primary_metric = metrics.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}

    payload: dict[str, Any] = {
        "kind": primary_metric.get("kind") or "ppl_causal",
        "unit": primary_metric.get("unit") or "ppl",
        "direction": primary_metric.get("direction") or "lower",
        "aggregation_scope": primary_metric.get("aggregation_scope") or "token",
        "paired": bool(primary_metric.get("paired", True)),
        "gating_basis": primary_metric.get("gating_basis") or "upper",
        "supports_bootstrap": bool(primary_metric.get("supports_bootstrap", True)),
        "invalid": True,
        "degraded": True,
    }

    preview = primary_metric.get("preview")
    final = primary_metric.get("final")
    if preview is None:
        preview = metrics.get("ppl_preview")
    if final is None:
        final = metrics.get("ppl_final")
    if preview is not None:
        payload["preview"] = preview
    if final is not None:
        payload["final"] = final

    drift_band = primary_metric.get("drift_band")
    if isinstance(drift_band, dict):
        payload["drift_band"] = drift_band

    return payload


def build_structural_failure_report(
    *,
    error_type: str,
    message: str,
    base_report: dict[str, Any],
    source_report: dict[str, Any] | None,
    source_report_path: str | None,
    edited_report_path: str | None,
    edited_events_path: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_report)
    source_run_id = str(payload.get("run_id") or "run")
    payload["run_id"] = f"{source_run_id}-structural-failure-{error_type}"

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    meta["structural_failure"] = {
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    validation.update(
        {
            "invariants_pass": False,
            "primary_metric_acceptable": False,
            "spectral_stable": False,
            "rmt_stable": False,
            "preview_final_drift_acceptable": False,
            "guard_overhead_acceptable": True,
            "primary_metric_tail_acceptable": False,
        }
    )
    payload["validation"] = validation

    guard_overhead = payload.get("guard_overhead")
    if not isinstance(guard_overhead, dict):
        guard_overhead = {}
    guard_overhead["evaluated"] = bool(guard_overhead.get("evaluated", True))
    payload["guard_overhead"] = guard_overhead

    primary_metric = payload.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}
    primary_metric.update(_metric_section(source_report))
    primary_metric["invalid"] = True
    primary_metric["degraded"] = True
    primary_metric["degraded_reason"] = "structural_failure"
    primary_metric.pop("ratio_vs_baseline", None)
    payload["primary_metric"] = primary_metric

    payload["_evidence_pack_structural_failure"] = {
        "format": "evidence-pack-structural-failure-report-v1",
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    payload["invariants"] = {
        "status": "fail",
        "failures": [
            {
                "check": "error_injection",
                "type": "evidence_pack_structural_failure",
                "severity": "fatal",
                "detail": {
                    "error_type": error_type,
                    "message": message,
                },
            }
        ],
    }

    spectral = payload.get("spectral")
    if not isinstance(spectral, dict):
        spectral = {}
    spectral["status"] = "structural_failure"
    payload["spectral"] = spectral

    rmt = payload.get("rmt")
    if not isinstance(rmt, dict):
        rmt = {}
    rmt["status"] = "structural_failure"
    payload["rmt"] = rmt

    return payload


def _write_runtime_manifest(
    *,
    out_path: Path,
    source_runtime_manifest: dict[str, Any] | None,
    error_type: str,
    message: str,
) -> None:
    if not isinstance(source_runtime_manifest, dict):
        return

    manifest_payload = copy.deepcopy(source_runtime_manifest)
    manifest_payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    manifest_payload["report"] = {
        "path": str(out_path.resolve()),
        "filename": out_path.name,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    context = manifest_payload.get("context")
    if not isinstance(context, dict):
        context = {}
    context["evidence_pack_structural_failure"] = {
        "error_type": error_type,
        "message": message,
    }
    manifest_payload["context"] = context
    manifest_path = out_path.parent / "runtime.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a minimal evidence-pack evaluation.report.json when "
            "structural error evaluation fails before the canonical report exists."
        )
    )
    parser.add_argument("--error-type", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--source-report")
    parser.add_argument("--source-runtime-manifest")
    parser.add_argument("--edited-report")
    parser.add_argument("--edited-events")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    source_report_path = Path(args.source_report) if args.source_report else None
    source_report = _load_json(source_report_path)
    source_runtime_manifest_path = (
        Path(args.source_runtime_manifest) if args.source_runtime_manifest else None
    )
    source_runtime_manifest = _load_json(source_runtime_manifest_path)
    base_report = _build_base_report(source_report)

    payload = build_structural_failure_report(
        error_type=str(args.error_type),
        message=str(args.message),
        base_report=base_report,
        source_report=source_report,
        source_report_path=str(source_report_path) if source_report_path else None,
        edited_report_path=str(args.edited_report) if args.edited_report else None,
        edited_events_path=str(args.edited_events) if args.edited_events else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_runtime_manifest(
        out_path=out_path,
        source_runtime_manifest=source_runtime_manifest,
        error_type=str(args.error_type),
        message=str(args.message),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
