from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .editing.implementations import resolve_edit_spec
    from .task_tools_errors import _create_error_model
    from .task_tools_model import (
        _download_baseline,
        _repair_missing_tensors_config,
        _write_model_profile,
        download_snapshot,
        model_supports_flash_attention,
        sanitize_generation_config,
        write_model_profile,
    )
    from .task_tools_preset import (
        _normalize_staged_preset,
        _parse_window_candidate,
        _plan_effective_windows,
        _validate_baseline_report,
        normalize_staged_preset,
        schedule_from_baseline_report,
    )
    from .task_tools_reports import (
        EDIT_ARTIFACT_SUMMARY_SCHEMA,
        _edit_artifact_summary,
        _structural_failure_report,
        build_edit_artifact_summary,
        build_structural_failure_report,
    )
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.implementations import resolve_edit_spec
    from task_tools_errors import _create_error_model
    from task_tools_model import (
        _download_baseline,
        _repair_missing_tensors_config,
        _write_model_profile,
        download_snapshot,
        model_supports_flash_attention,
        sanitize_generation_config,
        write_model_profile,
    )
    from task_tools_preset import (
        _normalize_staged_preset,
        _parse_window_candidate,
        _plan_effective_windows,
        _validate_baseline_report,
        normalize_staged_preset,
        schedule_from_baseline_report,
    )
    from task_tools_reports import (
        EDIT_ARTIFACT_SUMMARY_SCHEMA,
        _edit_artifact_summary,
        _structural_failure_report,
        build_edit_artifact_summary,
        build_structural_failure_report,
    )

__all__ = [
    "EDIT_ARTIFACT_SUMMARY_SCHEMA",
    "build_edit_artifact_summary",
    "build_structural_failure_report",
    "download_snapshot",
    "main",
    "model_supports_flash_attention",
    "normalize_staged_preset",
    "sanitize_generation_config",
    "schedule_from_baseline_report",
    "write_model_profile",
]


def _resolve_adapter(args: argparse.Namespace) -> int:
    model_id = str(args.model_id_or_path).strip()
    if not model_id:
        return 0

    from invarlock.adapters.auto import resolve_auto_adapter

    print(resolve_auto_adapter(model_id))
    return 0


def _resolve_edit_params(args: argparse.Namespace) -> int:
    resolved = resolve_edit_spec(
        model_output_dir=Path(args.model_output_dir),
        edit_spec=str(args.edit_spec),
        version_hint=str(args.version_hint or ""),
    )
    print(json.dumps(resolved.to_shell_payload()))
    return 0


def _model_revision(args: argparse.Namespace) -> int:
    path = Path(args.revisions_json)
    model_id = str(args.model_id)

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(data, dict):
        return 0

    revision = ""
    models = data.get("models")
    if isinstance(models, dict):
        entry = models.get(model_id)
        if isinstance(entry, dict):
            revision = str(entry.get("revision") or "")

    if revision:
        print(revision)
    return 0


def _evaluation_report(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    out_path = Path(args.out)

    try:
        from invarlock.reporting.report_make import make_report
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        evaluation_report = make_report(report, report)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evaluation_report, indent=2) + "\n")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-pack task helper tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-baseline",
        help="Download a pinned baseline model with evidence-pack storage policy.",
    )
    download_parser.add_argument("--model-id", required=True)
    download_parser.add_argument("--output-dir", required=True)
    download_parser.add_argument("--success-marker", default="")
    download_parser.set_defaults(func=_download_baseline)

    normalize_parser = subparsers.add_parser(
        "normalize-staged-preset",
        help="Normalize a staged preset for evaluate runtime.",
    )
    normalize_parser.add_argument("--preset", required=True)
    normalize_parser.add_argument("--baseline-report")
    normalize_parser.add_argument("--seq-len", type=int)
    normalize_parser.add_argument("--stride", type=int)
    normalize_parser.add_argument("--preview-n", type=int)
    normalize_parser.add_argument("--final-n", type=int)
    normalize_parser.add_argument("--skip-overhead-check", action="store_true")
    normalize_parser.set_defaults(func=_normalize_staged_preset)

    error_model_parser = subparsers.add_parser(
        "create-error-model",
        help="Create an evidence-pack structural error model.",
    )
    error_model_parser.add_argument("baseline_path")
    error_model_parser.add_argument("output_path")
    error_model_parser.add_argument("error_type")
    error_model_parser.set_defaults(func=_create_error_model)

    adapter_parser = subparsers.add_parser(
        "resolve-adapter",
        help="Resolve the InvarLock adapter for a model id or local path.",
    )
    adapter_parser.add_argument("model_id_or_path")
    adapter_parser.set_defaults(func=_resolve_adapter)

    edit_parser = subparsers.add_parser(
        "resolve-edit-params",
        help="Resolve an edit spec to shell-friendly JSON.",
    )
    edit_parser.add_argument("model_output_dir")
    edit_parser.add_argument("edit_spec")
    edit_parser.add_argument("version_hint", nargs="?", default="")
    edit_parser.set_defaults(func=_resolve_edit_params)

    revision_parser = subparsers.add_parser(
        "model-revision",
        help="Read a model revision from state/model_revisions.json.",
    )
    revision_parser.add_argument("revisions_json")
    revision_parser.add_argument("model_id")
    revision_parser.set_defaults(func=_model_revision)

    report_parser = subparsers.add_parser(
        "evaluation-report",
        help="Generate evaluation.report.json from report.json.",
    )
    report_parser.add_argument("--report", required=True)
    report_parser.add_argument("--out", required=True)
    report_parser.set_defaults(func=_evaluation_report)

    baseline_parser = subparsers.add_parser(
        "validate-baseline-report",
        help="Validate a generated baseline report contract.",
    )
    baseline_parser.add_argument("baseline_report")
    baseline_parser.add_argument("expected_adapter")
    baseline_parser.add_argument("expected_profile")
    baseline_parser.add_argument("expected_tier")
    baseline_parser.set_defaults(func=_validate_baseline_report)

    profile_parser = subparsers.add_parser(
        "write-model-profile",
        help="Write model_profile.json next to a downloaded baseline config.",
    )
    profile_parser.add_argument("baseline_dir")
    profile_parser.add_argument("model_id")
    profile_parser.set_defaults(func=_write_model_profile)

    repair_parser = subparsers.add_parser(
        "repair-missing-tensors-config",
        help="Repair legacy missing_tensors layer-drop config metadata.",
    )
    repair_parser.add_argument("baseline_config")
    repair_parser.add_argument("error_config")
    repair_parser.set_defaults(func=_repair_missing_tensors_config)

    plan_parser = subparsers.add_parser(
        "plan-effective-windows",
        help="Plan CI window schedules using effective post-dedupe token counts.",
    )
    plan_parser.add_argument("--model-path", required=True)
    plan_parser.add_argument("--dataset-provider", default="wikitext2")
    plan_parser.add_argument("--split", default="validation")
    plan_parser.add_argument("--seed", type=int, default=42)
    plan_parser.add_argument("--tier", default="balanced")
    plan_parser.add_argument("--profile", default="ci")
    plan_parser.add_argument("--headroom-ratio", type=float, default=1.05)
    plan_parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_window_candidate,
        default=[],
        help="Candidate schedule as seq_len:preview_n:final_n",
    )
    plan_parser.set_defaults(func=_plan_effective_windows)

    structural_parser = subparsers.add_parser(
        "structural-failure-report",
        help="Emit an evaluation.report.json for structural error evaluation failures.",
    )
    structural_parser.add_argument("--error-type", required=True)
    structural_parser.add_argument("--out", required=True)
    structural_parser.add_argument("--message", required=True)
    structural_parser.add_argument("--source-report")
    structural_parser.add_argument("--source-runtime-manifest")
    structural_parser.add_argument("--edited-report")
    structural_parser.add_argument("--edited-events")
    structural_parser.set_defaults(func=_structural_failure_report)

    summary_parser = subparsers.add_parser(
        "edit-artifact-summary",
        help="Write edit artifact class summary.",
    )
    summary_parser.add_argument("--pack-dir", required=True)
    summary_parser.add_argument("--scenarios", required=True)
    summary_parser.add_argument("--out", required=True)
    summary_parser.set_defaults(func=_edit_artifact_summary)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
