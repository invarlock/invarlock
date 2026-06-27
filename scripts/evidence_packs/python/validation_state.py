#!/usr/bin/env python3
"""State and manifest helpers for evidence-pack shell orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_TIMING_KEYS = (
    "load_model",
    "load_dataset",
    "prepare",
    "prepare_guards",
    "edit",
    "guards",
    "eval",
    "finalize",
    "execute",
    "total",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_values(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _truthy(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_deployable(scenario: dict[str, Any]) -> bool:
    generation = scenario.get("generation")
    generation_kind = generation.get("kind") if isinstance(generation, dict) else None
    return (
        scenario.get("artifact_class") == "deployable_optimized_subject"
        or generation_kind == "deployable_edit"
        or str(scenario.get("category") or "").startswith("deployable_")
    )


def _suites_ok(scenario: dict[str, Any], suite: str) -> bool:
    suites = scenario.get("suites")
    return not isinstance(suites, list) or not suites or suite in suites


def render_scenarios(
    src: Path,
    out: Path,
    *,
    suite: str,
    scenario_ids_csv: str,
    include_deployable: str,
    deploy_backends_csv: str,
) -> None:
    payload = _load_json(src)
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["applied_suite"] = suite

    requested_ids = _csv_values(scenario_ids_csv)
    requested_id_set = set(requested_ids)
    if requested_ids:
        meta["scenario_ids_filter"] = requested_ids
    payload["_meta"] = meta

    deploy_backends = set(_csv_values(deploy_backends_csv))
    include_deployable_flag = _truthy(include_deployable)
    selected: list[dict[str, Any]] = []
    for scenario in payload.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("id") or "")
        explicit = scenario_id in requested_id_set
        generation = scenario.get("generation")
        backend = ""
        if isinstance(generation, dict):
            backend = str(generation.get("backend") or "")
        backend = backend or str(scenario.get("backend") or "")
        deployable = _is_deployable(scenario)
        deploy_enabled = (
            deployable
            and include_deployable_flag
            and (not deploy_backends or backend in deploy_backends)
        )
        if requested_ids and not explicit:
            continue
        if not (explicit or _suites_ok(scenario, suite) or deploy_enabled):
            continue
        if deployable and not (explicit or deploy_enabled):
            continue
        selected.append(scenario)

    payload["scenarios"] = selected
    _write_json(out, payload)


def non_runnable_deployable_ids(path: Path) -> str:
    payload = _load_json(path)
    ids: list[str] = []
    for scenario in payload.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        generation = scenario.get("generation")
        if (
            isinstance(generation, dict)
            and generation.get("kind") == "deployable_edit"
            and scenario.get("runnable") is False
        ):
            scenario_id = scenario.get("id")
            if isinstance(scenario_id, str) and scenario_id:
                ids.append(scenario_id)
    return ",".join(ids)


def final_verdict(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    value = payload.get("verdict")
    return value.strip().upper() if isinstance(value, str) else "MISSING"


def count_edit_scenarios(path: Path, source_label: str) -> str:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return ""
    clean = 0
    stress = 0
    for scenario in payload.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        generation = scenario.get("generation")
        if not isinstance(generation, dict) or generation.get("kind") != "edit":
            continue
        if generation.get("version") == "clean":
            clean += 1
        elif generation.get("version") == "stress":
            stress += 1
    return f"{clean}|{stress}|{source_label}"


def count_generation_kind(path: Path, kind: str) -> str:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return ""
    total = 0
    for scenario in payload.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        generation = scenario.get("generation")
        if isinstance(generation, dict) and generation.get("kind") == kind:
            total += 1
    return str(total)


def sanitize_model_name(model_id: str) -> str:
    value = model_id.lower().replace("/", "__").replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]", "", value)


def _numeric_or_raw(value: str) -> int | float | str:
    raw = str(value)
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def write_disk_pressure(
    path: Path,
    *,
    free_gb: str,
    min_gb: str,
    output_dir: str,
) -> None:
    payload = {
        "detected_at": _utc_now(),
        "free_gb": _numeric_or_raw(free_gb),
        "min_free_gb": _numeric_or_raw(min_gb),
        "output_dir": output_dir,
    }
    _write_json(path, payload)


def estimate_model_params(model_path: Path) -> str:
    config_file = model_path / "config.json"
    if not config_file.is_file():
        return "7"
    try:
        config = _load_json(config_file)
        hidden = float(config.get("hidden_size", 4096))
        layers = float(config.get("num_hidden_layers", 32))
        vocab = float(config.get("vocab_size", 32000))
        intermediate = float(config.get("intermediate_size", hidden * 4))
        experts = int(config.get("num_local_experts", 1) or 1)
        if experts == 1:
            experts = int(config.get("num_experts", 1) or 1)

        embedding_params = vocab * hidden
        attention_per_layer = 4 * hidden * hidden
        ffn_per_layer = 3 * hidden * intermediate
        lm_head = hidden * vocab
        if experts > 1:
            return "moe"
        base_params = (
            embedding_params + layers * (attention_per_layer + ffn_per_layer) + lm_head
        ) / 1e9
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "7"

    if base_params > 55:
        return "70"
    if base_params > 28:
        return "40"
    if base_params > 18:
        return "30"
    if base_params > 10:
        return "13"
    return "7"


def reset_task_for_resume(path: Path) -> None:
    payload = _load_json(path)
    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
    params.update({"retry_after": None, "last_error_type": None})
    payload.update(
        {
            "status": "pending",
            "retries": 0,
            "assigned_gpus": None,
            "started_at": None,
            "completed_at": None,
            "error_msg": None,
            "params": params,
        }
    )
    _write_json(path, payload)


def _extract_ratio(cert: dict[str, Any]) -> float | None:
    verdict = cert.get("verdict") or {}
    metrics = cert.get("metrics") or {}
    if not isinstance(verdict, dict):
        verdict = {}
    if not isinstance(metrics, dict):
        metrics = {}
    for candidate in (
        verdict.get("primary_metric_ratio"),
        verdict.get("primary_metric_ratio_raw"),
        verdict.get("primary_metric_ratio_mean"),
        metrics.get("primary_metric_ratio"),
        metrics.get("primary_metric_ratio_mean"),
    ):
        if isinstance(candidate, int | float):
            return float(candidate)
    return None


def write_determinism_repeats_summary(
    *,
    out_path: Path,
    model_id: str,
    edit_name: str,
    requested: int,
    mode: str,
    suite: str,
    cert_paths: list[Path],
) -> None:
    hashes: list[str] = []
    ratios: list[float] = []
    errors: list[str] = []

    for path in cert_paths:
        try:
            raw = path.read_bytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue

        if isinstance(data, dict):
            ratio = _extract_ratio(data)
            if ratio is not None:
                ratios.append(ratio)

    ratio_summary = None
    if ratios:
        ratio_summary = {
            "min": min(ratios),
            "max": max(ratios),
            "delta": max(ratios) - min(ratios),
        }

    payload: dict[str, object] = {
        "requested": requested,
        "completed": len(cert_paths),
        "mode": str(mode),
        "suite": str(suite),
        "model_id": str(model_id),
        "edit_name": str(edit_name),
        "cert_hashes_match": bool(hashes) and len(set(hashes)) == 1,
        "cert_hashes": hashes,
        "primary_metric_ratio": ratio_summary,
        "errors": errors,
        "generated_at": _utc_now(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def preflight_models(out_file: Path, model_ids: list[str]) -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - depends on optional deps
        print(
            "ERROR: huggingface_hub is required for preflight; install it before running with --net 1.",
            file=sys.stderr,
        )
        print(f"       Details: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, object] = {
        "generated_at": _utc_now(),
        "suite": str(os.environ.get("PACK_SUITE", "")),
        "model_list": list(model_ids),
        "models": {},
    }

    api = HfApi(token=False)

    errors: list[str] = []
    models_out: dict[str, dict[str, object]] = {}
    for model_id in model_ids:
        try:
            info = api.model_info(model_id, token=False)
        except HfHubHTTPError as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            msg = (
                "requires authentication (gated/private)"
                if status in (401, 403)
                else str(err)
            )
            print(
                f"ERROR: {model_id} is not publicly accessible ({msg})",
                file=sys.stderr,
            )
            errors.append(model_id)
            continue
        except (OSError, RuntimeError, ValueError) as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            msg = (
                "requires authentication (gated/private)"
                if status in (401, 403)
                else str(err)
            )
            print(
                f"ERROR: {model_id} is not publicly accessible ({msg})",
                file=sys.stderr,
            )
            errors.append(model_id)
            continue

        gated = bool(getattr(info, "gated", False))
        private = bool(getattr(info, "private", False))
        if gated or private:
            print(
                f"ERROR: {model_id} is gated/private; evidence packs require ungated models.",
                file=sys.stderr,
            )
            errors.append(model_id)
            continue

        models_out[model_id] = {
            "revision": str(getattr(info, "sha", "") or ""),
            "resolved_at": _utc_now(),
            "gated": gated,
            "private": private,
        }

    payload["models"] = models_out

    if errors:
        return 2

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote model revisions to {out_file}")
    return 0


def _get_tuned_edit_entry(
    *,
    data: dict[str, Any],
    defaults: dict[str, Any],
    models_map: dict[str, Any],
    model_id: str,
    model_name: str,
    edit_type: str,
) -> dict[str, Any]:
    entry_map: dict[str, Any] = {}
    if isinstance(models_map, dict):
        entry = models_map.get(model_id) or models_map.get(model_name) or {}
        entry_map = entry if isinstance(entry, dict) else {}
    if not entry_map and isinstance(data.get(edit_type), dict):
        entry_map = data
    entry = entry_map.get(edit_type) or defaults.get(edit_type) or {}
    return entry if isinstance(entry, dict) else {}


def _describe_tuned_entry_diff(
    entry: dict[str, Any], canonical_entry: dict[str, Any]
) -> str:
    diff_keys = sorted(set(entry) | set(canonical_entry))
    parts: list[str] = []
    for key in diff_keys:
        if entry.get(key) == canonical_entry.get(key):
            continue
        parts.append(
            f"{key}={json.dumps(entry.get(key), sort_keys=True)}"
            f"!=canonical={json.dumps(canonical_entry.get(key), sort_keys=True)}"
        )
    return "; ".join(parts) if parts else "entry!=canonical"


def validate_tuned_edit_params(
    *,
    path: Path,
    models_csv: str,
    model_names_csv: str,
    edit_types_csv: str,
    canonical_file: Path | None = None,
    allow_noncanonical: bool = False,
) -> int:
    if not path.is_file():
        raise SystemExit(f"Tuned edit preset file not found: {path}")

    models = _csv_values(models_csv)
    model_names = _csv_values(model_names_csv)
    required = sorted(set(_csv_values(edit_types_csv)))

    data = _load_json(path)
    if not isinstance(data, dict):
        raise SystemExit("Invalid tuned edit preset file (expected JSON object).")

    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    models_map = data.get("models") if isinstance(data.get("models"), dict) else {}
    canonical: dict[str, Any] = {}
    canonical_defaults: dict[str, Any] = {}
    canonical_models_map: dict[str, Any] = {}
    if canonical_file is not None:
        if not canonical_file.is_file():
            raise SystemExit(
                f"Canonical tuned edit preset file not found: {canonical_file}"
            )
        canonical = _load_json(canonical_file)
        if not isinstance(canonical, dict):
            raise SystemExit(
                "Invalid canonical tuned edit preset file (expected JSON object)."
            )
        canonical_defaults = (
            canonical.get("defaults")
            if isinstance(canonical.get("defaults"), dict)
            else {}
        )
        canonical_models_map = (
            canonical.get("models") if isinstance(canonical.get("models"), dict) else {}
        )

    missing: list[str] = []
    noncanonical: list[str] = []
    for idx, model_id in enumerate(models):
        model_name = model_names[idx] if idx < len(model_names) else ""
        for edit_type in required:
            entry = _get_tuned_edit_entry(
                data=data,
                defaults=defaults,
                models_map=models_map,
                model_id=model_id,
                model_name=model_name,
                edit_type=edit_type,
            )
            status = str(entry.get("status") or "missing")
            if status != "selected":
                missing.append(f"{model_id}:{edit_type}:{status}")
                continue
            if allow_noncanonical or canonical_file is None:
                continue
            canonical_entry = _get_tuned_edit_entry(
                data=canonical,
                defaults=canonical_defaults,
                models_map=canonical_models_map,
                model_id=model_id,
                model_name=model_name,
                edit_type=edit_type,
            )
            canonical_status = str(canonical_entry.get("status") or "missing")
            if canonical_status != "selected":
                continue
            if entry != canonical_entry:
                diff = _describe_tuned_entry_diff(entry, canonical_entry)
                noncanonical.append(f"{model_id}:{edit_type}:{diff}")

    if missing:
        msg = "Missing tuned edit presets: " + ", ".join(missing)
        raise SystemExit(msg)
    if noncanonical:
        msg = "Noncanonical tuned edit presets: " + ", ".join(noncanonical)
        raise SystemExit(msg)

    return 0


def _sum_timing(payloads: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in payloads:
        timings = payload.get("timings_seconds")
        if not isinstance(timings, dict):
            continue
        value = timings.get(key)
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _sum_run_timing(payloads: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in payloads:
        aggregate = payload.get("aggregate_run_timings_seconds")
        if isinstance(aggregate, dict):
            try:
                total += float(aggregate.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            continue
        run_timings = payload.get("run_timings_seconds")
        if not isinstance(run_timings, dict):
            continue
        for side_payload in run_timings.values():
            if not isinstance(side_payload, dict):
                continue
            try:
                total += float(side_payload.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _collect_evaluate_timings(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("**/evaluate_timing.json")):
        if "/results/analysis/" in str(path):
            continue
        payload = _read_json_optional(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def build_evaluation_optimization_summary(run_dir: Path) -> dict[str, Any]:
    timings = _collect_evaluate_timings(run_dir)
    return {
        "schema": "invarlock/evidence-pack-evaluation-optimization-summary-v1",
        "run_dir": ".",
        "path_scope": "run_root_relative",
        "controls": {
            "PACK_DEFER_REPORT_RENDERING": os.environ.get("PACK_DEFER_REPORT_RENDERING")
            or os.environ.get("PACK_DEFER_OPTIONAL_REPORT_RENDERING")
            or "0",
        },
        "evaluation_reports_timed": len(timings),
        "baseline_report_reuse_count": sum(
            1 for item in timings if bool(item.get("baseline_report_reused"))
        ),
        "deferred_rendering_count": sum(
            1 for item in timings if bool(item.get("defer_report_rendering"))
        ),
        "timing_totals_seconds": {
            "plan": _sum_timing(timings, "plan"),
            "baseline": _sum_timing(timings, "baseline"),
            "subject": _sum_timing(timings, "subject"),
            "evaluation_report": _sum_timing(timings, "evaluation_report"),
            "total": _sum_timing(timings, "total"),
        },
        "run_timing_totals_seconds": {
            key: _sum_run_timing(timings, key) for key in _RUN_TIMING_KEYS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_render = subparsers.add_parser("render-scenarios")
    p_render.add_argument("--src", required=True)
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--suite", required=True)
    p_render.add_argument("--scenario-ids", default="")
    p_render.add_argument("--include-deployable", default="0")
    p_render.add_argument("--deploy-backends", default="")

    p_non_runnable = subparsers.add_parser("non-runnable-deployable-ids")
    p_non_runnable.add_argument("path")

    p_verdict = subparsers.add_parser("final-verdict")
    p_verdict.add_argument("path")

    p_count_edit = subparsers.add_parser("count-edit-scenarios")
    p_count_edit.add_argument("path")
    p_count_edit.add_argument("--source-label", default="scenarios.json")

    p_count_kind = subparsers.add_parser("count-generation-kind")
    p_count_kind.add_argument("path")
    p_count_kind.add_argument("--kind", required=True)

    p_sanitize = subparsers.add_parser("sanitize-model-name")
    p_sanitize.add_argument("model_id")

    p_disk = subparsers.add_parser("write-disk-pressure")
    p_disk.add_argument("--path", required=True)
    p_disk.add_argument("--free-gb", required=True)
    p_disk.add_argument("--min-gb", required=True)
    p_disk.add_argument("--output-dir", required=True)

    p_estimate = subparsers.add_parser("estimate-model-params")
    p_estimate.add_argument("model_path")

    p_reset_task = subparsers.add_parser("reset-task-for-resume")
    p_reset_task.add_argument("path")

    p_determinism = subparsers.add_parser("determinism-repeats-summary")
    p_determinism.add_argument("out_path")
    p_determinism.add_argument("model_id")
    p_determinism.add_argument("edit_name")
    p_determinism.add_argument("requested_repeats")
    p_determinism.add_argument("mode")
    p_determinism.add_argument("suite")
    p_determinism.add_argument("cert_paths", nargs="*")

    p_preflight = subparsers.add_parser("preflight-models")
    p_preflight.add_argument("out_file")
    p_preflight.add_argument("model_ids", nargs="+")

    p_eval_optimization = subparsers.add_parser("evaluation-optimization-summary")
    p_eval_optimization.add_argument("run_dir")
    p_eval_optimization.add_argument("--out", required=True)

    p_tuned = subparsers.add_parser("validate-tuned-edit-params")
    p_tuned.add_argument("--file", required=True)
    p_tuned.add_argument("--models", required=True)
    p_tuned.add_argument("--model-names", required=True)
    p_tuned.add_argument("--edit-types", required=True)
    p_tuned.add_argument("--canonical-file")
    p_tuned.add_argument("--allow-noncanonical", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "render-scenarios":
        render_scenarios(
            Path(args.src),
            Path(args.out),
            suite=args.suite,
            scenario_ids_csv=args.scenario_ids,
            include_deployable=args.include_deployable,
            deploy_backends_csv=args.deploy_backends,
        )
        return 0
    if args.command == "non-runnable-deployable-ids":
        print(non_runnable_deployable_ids(Path(args.path)))
        return 0
    if args.command == "final-verdict":
        print(final_verdict(Path(args.path)))
        return 0
    if args.command == "count-edit-scenarios":
        print(count_edit_scenarios(Path(args.path), args.source_label))
        return 0
    if args.command == "count-generation-kind":
        print(count_generation_kind(Path(args.path), args.kind))
        return 0
    if args.command == "sanitize-model-name":
        print(sanitize_model_name(args.model_id))
        return 0
    if args.command == "write-disk-pressure":
        write_disk_pressure(
            Path(args.path),
            free_gb=args.free_gb,
            min_gb=args.min_gb,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "estimate-model-params":
        print(estimate_model_params(Path(args.model_path)))
        return 0
    if args.command == "reset-task-for-resume":
        reset_task_for_resume(Path(args.path))
        return 0
    if args.command == "determinism-repeats-summary":
        try:
            requested = int(args.requested_repeats)
        except (TypeError, ValueError):
            requested = 0
        write_determinism_repeats_summary(
            out_path=Path(args.out_path),
            model_id=args.model_id,
            edit_name=args.edit_name,
            requested=requested,
            mode=args.mode,
            suite=args.suite,
            cert_paths=[Path(path) for path in args.cert_paths],
        )
        return 0
    if args.command == "preflight-models":
        return preflight_models(Path(args.out_file), list(args.model_ids))
    if args.command == "evaluation-optimization-summary":
        summary = build_evaluation_optimization_summary(Path(args.run_dir))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-tuned-edit-params":
        return validate_tuned_edit_params(
            path=Path(args.file),
            models_csv=args.models,
            model_names_csv=args.model_names,
            edit_types_csv=args.edit_types,
            canonical_file=Path(args.canonical_file) if args.canonical_file else None,
            allow_noncanonical=bool(args.allow_noncanonical),
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
