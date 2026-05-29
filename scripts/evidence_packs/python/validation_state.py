#!/usr/bin/env python3
"""State and manifest helpers for evidence-pack shell orchestration."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


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
        "detected_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
