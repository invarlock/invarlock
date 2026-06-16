#!/usr/bin/env python3
"""Run the shipped-model evidence sweep for supported experimental lanes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evidence_workflows.workflow_plan import (
    WorkflowCommandStep,
    WorkflowLanePlan,
    WorkflowSweepPlan,
)
from evidence_workflows.workflow_runner import (
    WorkflowLaneExecutionRequest,
    WorkflowSweepExecutionRequest,
    execute_workflow_lane,
    execute_workflow_sweep,
    workflow_return_code,
)
from evidence_workflows.workflow_state import (
    WorkflowLaneResult as LaneResult,
)
from evidence_workflows.workflow_state import (
    WorkflowLaneRunState,
    WorkflowRunMetadata,
    sha256_file,
    write_summary_files,
)
from evidence_workflows.workflow_state import (
    capture_artifacts as _capture_workflow_artifacts,
)
from evidence_workflows.workflow_state import (
    write_artifact_manifest as _write_workflow_artifact_manifest,
)
from model_evidence_lanes import (  # noqa: E402
    CURRENT_PUBLISHED_BASIS_LANES,
    CURRENT_SUPPORTED_EXPERIMENTAL_LANES,
    DEFAULT_SUITE,
    DOCUMENTED_SMOKE_CANARY_LANES,
    EXECUTION_MODES,
    MODEL_CATALOG_GPU_LANES,
    MODEL_CATALOG_GPU_SUITE,
    PROMOTION_GAP_GPU_LANES,
    PROMOTION_GAP_GPU_SUITE,
    REPO_MENTIONED_GPU_SUITE,
    REPO_ROOT,
    SUITES,
    SUPPORT_MATRIX_BACKLOG_GPU_LANES,
    SUPPORT_MATRIX_BACKLOG_GPU_SUITE,
    EvidenceLane,
    lane_requires_remote_code,
    lane_resource_estimate,
    manifest_lane_ids,
    select_specs,
    supported_experimental_lane_ids,
    validate_manifest_coverage,
)

RETRYABLE_EVALUATE_RETURNCODES = {-15}
_sha256_file = sha256_file

__all__ = [
    "CURRENT_PUBLISHED_BASIS_LANES",
    "CURRENT_SUPPORTED_EXPERIMENTAL_LANES",
    "DEFAULT_SUITE",
    "DOCUMENTED_SMOKE_CANARY_LANES",
    "EXECUTION_MODES",
    "MODEL_CATALOG_GPU_LANES",
    "MODEL_CATALOG_GPU_SUITE",
    "PROMOTION_GAP_GPU_LANES",
    "PROMOTION_GAP_GPU_SUITE",
    "REPO_MENTIONED_GPU_SUITE",
    "SUITES",
    "SUPPORT_MATRIX_BACKLOG_GPU_LANES",
    "SUPPORT_MATRIX_BACKLOG_GPU_SUITE",
    "EvidenceLane",
    "LaneResult",
    "manifest_lane_ids",
    "select_specs",
    "supported_experimental_lane_ids",
]

MODEL_EVIDENCE_ARTIFACT_PATTERNS = (
    "manifest.json",
    "summary.json",
    "summary.tsv",
    "status.log",
    "model_revisions.json",
    "logs/*.log",
    "eval/*/dataset/manifest.jsonl",
    "eval/*/dataset/materialization_summary.json",
    "eval/*/dataset/images/*",
    "eval/*/prepared_preset.yaml",
    "eval/*/report/evaluation.report.json",
    "eval/*/verify.json",
)


def write_summary(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    shard_index: int,
    shard_count: int,
    results: Sequence[LaneResult],
) -> None:
    write_summary_files(
        output_root,
        metadata=WorkflowRunMetadata(
            suite=suite,
            execution_mode=execution_mode,
            shard_index=shard_index,
            shard_count=shard_count,
        ),
        results=results,
    )


def write_manifest(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    specs: Sequence[EvidenceLane],
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "execution_mode": execution_mode,
        "lanes": [spec.to_manifest_entry() for spec in specs],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_artifacts(output_root: Path) -> list[dict[str, object]]:
    return _capture_workflow_artifacts(
        output_root,
        patterns=MODEL_EVIDENCE_ARTIFACT_PATTERNS,
    )


def _default_hf_home() -> Path:
    return REPO_ROOT / "tmp" / "model_evidence_hf_home"


def _ensure_hf_cache_env(env: dict[str, str]) -> None:
    hf_home = env.get("HF_HOME")
    if hf_home is None or not hf_home.strip():
        hf_home = str(_default_hf_home())
        env["HF_HOME"] = hf_home
    if not env.get("HF_HUB_CACHE", "").strip():
        env["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
    if not env.get("HF_DATASETS_CACHE", "").strip():
        env["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        Path(env[name]).expanduser().mkdir(parents=True, exist_ok=True)


def _hf_cache_dir_from_env(env: dict[str, str]) -> Path | None:
    hub_cache = env.get("HF_HUB_CACHE")
    if hub_cache and hub_cache.strip():
        return Path(hub_cache).expanduser()
    hf_home = env.get("HF_HOME")
    if hf_home and hf_home.strip():
        return Path(hf_home).expanduser() / "hub"
    return None


def _collect_hf_model_revisions(
    specs: Sequence[EvidenceLane],
    *,
    env: dict[str, str],
) -> list[dict[str, object]]:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:  # pragma: no cover - environment-bound.
        return [
            {
                "slug": spec.slug,
                "model_id": spec.model_id,
                "status": "unavailable",
                "reason": f"huggingface_hub_unavailable:{type(exc).__name__}",
            }
            for spec in specs
        ]

    cache_dir = _hf_cache_dir_from_env(env)
    try:
        cache_info = (
            scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
        )
    except Exception as exc:  # pragma: no cover - cache state is host-specific.
        return [
            {
                "slug": spec.slug,
                "model_id": spec.model_id,
                "status": "unavailable",
                "reason": f"cache_scan_failed:{type(exc).__name__}",
                "cache_dir": str(cache_dir) if cache_dir else None,
            }
            for spec in specs
        ]

    repos = {repo.repo_id: repo for repo in cache_info.repos}
    revisions: list[dict[str, object]] = []
    for spec in specs:
        repo = repos.get(spec.model_id)
        if repo is None:
            revisions.append(
                {
                    "slug": spec.slug,
                    "model_id": spec.model_id,
                    "status": "missing",
                    "revisions": [],
                    "cache_dir": str(cache_dir) if cache_dir else None,
                }
            )
            continue
        repo_revisions = sorted(
            (
                {
                    "commit_hash": revision.commit_hash,
                    "refs": sorted(revision.refs),
                    "snapshot_path": str(revision.snapshot_path),
                    "size_on_disk": revision.size_on_disk,
                }
                for revision in repo.revisions
            ),
            key=lambda item: str(item["commit_hash"]),
        )
        revisions.append(
            {
                "slug": spec.slug,
                "model_id": spec.model_id,
                "status": "observed" if repo_revisions else "missing",
                "revisions": repo_revisions,
                "cache_dir": str(cache_dir) if cache_dir else None,
            }
        )
    return revisions


def write_model_revisions(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    specs: Sequence[EvidenceLane],
    env: dict[str, str],
) -> None:
    payload = {
        "schema": "invarlock/model-evidence-model-revisions-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "execution_mode": execution_mode,
        "models": _collect_hf_model_revisions(specs, env=env),
    }
    (output_root / "model_revisions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_artifact_manifest(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    shard_index: int,
    shard_count: int,
    results: Sequence[LaneResult],
) -> None:
    _write_workflow_artifact_manifest(
        output_root,
        schema="invarlock/model-evidence-artifact-manifest-v1",
        metadata=WorkflowRunMetadata(
            suite=suite,
            execution_mode=execution_mode,
            shard_index=shard_index,
            shard_count=shard_count,
        ),
        results=results,
        artifact_patterns=MODEL_EVIDENCE_ARTIFACT_PATTERNS,
    )


def default_output_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / "model_evidence" / stamp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shipped-model evidence sweep for current supported "
            "experimental lanes."
        )
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        choices=sorted(SUITES),
        help="Named lane suite to run.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Restrict to one or more manifest slugs.",
    )
    parser.add_argument(
        "--lane-id",
        action="append",
        default=[],
        help="Restrict to one or more support_matrix lane_ids.",
    )
    parser.add_argument(
        "--preset-override",
        action="append",
        default=[],
        metavar="SLUG=PATH",
        help=(
            "Use PATH instead of the lane preset for one manifest slug. "
            "Repeat for multiple lanes."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Destination root for logs, reports, and summaries.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional global profile override for evaluate and verify.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed through to evaluate.",
    )
    parser.add_argument(
        "--execution-mode",
        default="container",
        choices=EXECUTION_MODES,
        help=(
            "How to execute model-loading commands. 'container' keeps the "
            "default runtime-container path; 'host' adds the "
            "explicit host-bypass and verify override flags."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for `python -m invarlock ...` subprocesses.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for partial sweeps.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards for partial sweeps.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first evaluate or verify failure.",
    )
    parser.add_argument(
        "--list-json",
        action="store_true",
        help="Print the resolved lane manifest as JSON and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run and exit.",
    )
    return parser.parse_args(argv)


def build_evaluate_command(
    spec: EvidenceLane,
    *,
    python_exe: str,
    profile: str,
    device: str,
    execution_mode: str,
    lane_root: Path,
    preset_arg_override: str | None = None,
) -> list[str]:
    preset_arg = preset_arg_override or spec.preset_arg(execution_mode=execution_mode)
    command = [
        python_exe,
        "-m",
        "invarlock",
        "evaluate",
        "--baseline",
        spec.model_id,
        "--subject",
        spec.model_id,
        "--baseline-adapter",
        spec.adapter,
        "--subject-adapter",
        spec.adapter,
        "--preset",
        preset_arg,
        "--profile",
        profile,
        "--allow-network",
        "--device",
        device,
        "--out",
        _command_path(lane_root / "runs", execution_mode=execution_mode),
        "--report-out",
        _command_path(lane_root / "report", execution_mode=execution_mode),
    ]
    if execution_mode == "host":
        command.extend(["--execution-mode", "host"])
    if lane_requires_remote_code(spec):
        command.append("--allow-remote-code")
    if profile == "dev":
        command.extend(["--assurance", "off"])
    return command


def parse_preset_overrides(raw_items: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError("--preset-override entries must use SLUG=PATH")
        slug, path = raw.split("=", 1)
        slug = slug.strip()
        path = path.strip()
        if not slug or not path:
            raise ValueError("--preset-override entries must use non-empty SLUG=PATH")
        if slug in overrides:
            raise ValueError(f"duplicate --preset-override for slug: {slug}")
        overrides[slug] = path
    return overrides


def build_vision_text_materialize_command(
    spec: EvidenceLane,
    *,
    python_exe: str,
    lane_root: Path,
    execution_mode: str,
) -> list[str] | None:
    materialization = spec.vision_text_materialization
    if not materialization:
        return None
    output_dir = lane_root / "dataset"
    command = [
        python_exe,
        "scripts/model_evidence/materialize_vision_text_dataset.py",
        "--dataset",
        str(materialization["dataset"]),
        "--split",
        str(materialization.get("split", "validation")),
        "--output-dir",
        _command_path(output_dir, execution_mode=execution_mode),
        "--max-samples",
        str(materialization.get("max_samples", 64)),
        "--image-field",
        str(materialization.get("image_field", "image")),
        "--prompt-field",
        str(materialization.get("prompt_field", "question")),
        "--image-format",
        str(materialization.get("image_format", "png")),
        "--overwrite",
    ]
    optional_flags = (
        ("revision", "--revision"),
        ("config_name", "--config-name"),
        ("answer_field", "--answer-field"),
        ("answers_field", "--answers-field"),
        ("id_field", "--id-field"),
        ("prompt_template", "--prompt-template"),
    )
    for key, flag in optional_flags:
        value = materialization.get(key)
        if value is not None and str(value) != "":
            command.extend([flag, str(value)])
    if bool(materialization.get("shuffle", False)):
        command.append("--shuffle")
    if "seed" in materialization:
        command.extend(["--seed", str(materialization["seed"])])
    return command


def write_prepared_preset(
    spec: EvidenceLane,
    *,
    lane_root: Path,
    execution_mode: str,
) -> Path | None:
    if not spec.vision_text_materialization:
        return None
    preset_data = yaml.safe_load(spec.preset_path.read_text(encoding="utf-8"))
    if not isinstance(preset_data, dict):
        raise ValueError(f"Preset must be a mapping: {spec.preset_relpath}")
    dataset = preset_data.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError(
            f"Preset dataset section must be a mapping: {spec.preset_relpath}"
        )
    provider = dataset.setdefault("provider", {})
    if not isinstance(provider, dict):
        raise ValueError(
            f"Preset dataset.provider section must be a mapping: {spec.preset_relpath}"
        )
    provider["kind"] = "vision_text"
    provider["path"] = _command_path(
        lane_root / "dataset" / "manifest.jsonl",
        execution_mode=execution_mode,
    )
    prepared_path = lane_root / "prepared_preset.yaml"
    prepared_path.write_text(
        yaml.safe_dump(preset_data, sort_keys=False),
        encoding="utf-8",
    )
    return prepared_path


def _prefetch_adapter_name(spec: EvidenceLane) -> str:
    adapter_name = spec.adapter
    if adapter_name in {"auto", "auto_hf"}:
        preset = spec.preset_relpath.lower()
        lane = spec.lane_id.lower()
        model_id = spec.model_id.lower()
        if "masked_lm" in preset or "masked" in lane or "bert" in model_id:
            adapter_name = "hf_mlm"
        elif "seq2seq" in preset or "t5" in model_id:
            adapter_name = "hf_seq2seq"
        else:
            adapter_name = "hf_causal"
    return adapter_name


def build_prefetch_command(
    spec: EvidenceLane,
    *,
    python_exe: str,
) -> list[str]:
    adapter_name = _prefetch_adapter_name(spec)
    prefetch_code = (
        "from huggingface_hub import snapshot_download; "
        "from invarlock.model_profile import detect_model_profile; "
        "import sys; "
        "model_id = sys.argv[1]; "
        f"detect_model_profile(model_id, adapter={adapter_name!r}).make_tokenizer(); "
        "snapshot_download(model_id)"
    )
    return [python_exe, "-c", prefetch_code, spec.model_id]


def build_verify_command(
    *,
    python_exe: str,
    profile: str,
    execution_mode: str,
    report_path: Path,
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "invarlock",
        "verify",
        "--profile",
        profile,
        "--json",
        str(report_path),
    ]
    if execution_mode == "host":
        command[4:4] = ["--runtime-provenance", "host"]
    return command


def resolve_lane_profile(
    *, profile_override: str | None, execution_mode: str, spec: EvidenceLane
) -> str:
    if profile_override:
        return profile_override
    if execution_mode == "host":
        return "dev"
    return spec.verify_profile


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    _ensure_hf_cache_env(env)
    return env


def _accelerator_requested(device: str) -> bool:
    normalized = str(device or "").strip().lower()
    return normalized in {"auto", "cuda"} or normalized.startswith("cuda:")


def visible_cuda_device_count(env: dict[str, str]) -> int | None:
    raw_value = env.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None or raw_value.strip() == "":
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"all", "gpu-all"}:
        return None
    if normalized in {"-1", "none", "void"}:
        return 0
    return len([item for item in raw_value.split(",") if item.strip()])


def lane_resource_preflight(
    spec: EvidenceLane,
    *,
    env: dict[str, str],
    device: str,
) -> dict[str, object] | None:
    estimate = lane_resource_estimate(spec.model_id)
    if estimate is None:
        return None

    visible_gpus = (
        visible_cuda_device_count(env) if _accelerator_requested(device) else 0
    )
    recommended = int(estimate["recommended_min_gpus_80gb"])
    payload: dict[str, object] = {
        "resource_estimate": estimate,
        "requested_device": device,
        "visible_cuda_devices": visible_gpus,
        "recommended_min_gpus_80gb": recommended,
        "ok": True,
    }
    if visible_gpus is not None and visible_gpus < recommended:
        payload["ok"] = False
        payload["warning"] = (
            f"visible CUDA device count {visible_gpus} is below the "
            f"recommended minimum {recommended} for {spec.slug}; use --gpu-group "
            "or CUDA_VISIBLE_DEVICES to expose enough devices for this lane"
        )
    return payload


def _is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def _execution_root(output_root: Path, *, execution_mode: str) -> Path:
    if execution_mode != "container" or _is_within_repo(output_root):
        return output_root
    suffix = hashlib.sha256(
        output_root.resolve().as_posix().encode("utf-8")
    ).hexdigest()[:16]
    return REPO_ROOT / "tmp" / "model_evidence_container" / suffix


def _command_path(path: Path, *, execution_mode: str) -> str:
    if execution_mode == "container":
        try:
            return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def _publish_lane_artifacts(source: Path, destination: Path) -> None:
    if source == destination or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _cleanup_execution_lane(
    lane_root: Path,
    *,
    execution_root: Path,
    output_root: Path,
) -> None:
    if execution_root == output_root:
        return
    scratch_root = REPO_ROOT / "tmp" / "model_evidence_container"
    try:
        lane_resolved = lane_root.resolve()
        lane_resolved.relative_to(scratch_root.resolve())
    except ValueError:
        return
    shutil.rmtree(lane_resolved, ignore_errors=True)
    for candidate in (lane_resolved.parent, execution_root):
        try:
            candidate.rmdir()
        except OSError:
            pass


def _publish_and_cleanup_lane_artifacts(
    *,
    lane_root: Path,
    published_lane_root: Path,
    execution_root: Path,
    output_root: Path,
) -> None:
    _publish_lane_artifacts(lane_root, published_lane_root)
    _cleanup_execution_lane(
        lane_root,
        execution_root=execution_root,
        output_root=output_root,
    )


def build_lane_plan(
    spec: EvidenceLane,
    *,
    output_root: Path,
    execution_root: Path,
    python_exe: str,
    profile: str | None,
    device: str,
    execution_mode: str,
    env: dict[str, str],
    preset_overrides: dict[str, str] | None = None,
) -> WorkflowLanePlan:
    lane_root = execution_root / "eval" / spec.slug
    published_lane_root = output_root / "eval" / spec.slug
    lane_profile = resolve_lane_profile(
        profile_override=profile,
        execution_mode=execution_mode,
        spec=spec,
    )

    materialize_cmd = build_vision_text_materialize_command(
        spec,
        python_exe=python_exe,
        lane_root=lane_root,
        execution_mode=execution_mode,
    )
    prepared_preset = None
    preset_arg_override = None
    steps: list[WorkflowCommandStep] = []
    if materialize_cmd is not None:
        prepared_preset = _command_path(
            lane_root / "prepared_preset.yaml",
            execution_mode=execution_mode,
        )
        preset_arg_override = prepared_preset
        steps.append(
            WorkflowCommandStep(
                name="materialize_dataset",
                command=tuple(materialize_cmd),
                log_mode="w",
            )
        )
    elif preset_overrides:
        preset_arg_override = preset_overrides.get(spec.slug)

    if execution_mode == "host":
        steps.append(
            WorkflowCommandStep(
                name="prefetch",
                command=tuple(build_prefetch_command(spec, python_exe=python_exe)),
                log_mode="a" if steps else "w",
            )
        )

    evaluate_cmd = build_evaluate_command(
        spec,
        python_exe=python_exe,
        profile=lane_profile,
        device=device,
        execution_mode=execution_mode,
        lane_root=lane_root,
        preset_arg_override=preset_arg_override,
    )
    steps.append(
        WorkflowCommandStep(
            name="evaluate",
            command=tuple(evaluate_cmd),
            log_mode="a" if steps else "w",
            retry_returncodes=tuple(sorted(RETRYABLE_EVALUATE_RETURNCODES)),
            retry_message="evaluate exited with {returncode}; retrying once.",
        )
    )
    report_path = lane_root / "report" / "evaluation.report.json"
    verify_path = lane_root / "verify.json"
    steps.append(
        WorkflowCommandStep(
            name="verify",
            command=tuple(
                build_verify_command(
                    python_exe=python_exe,
                    profile=lane_profile,
                    execution_mode=execution_mode,
                    report_path=report_path,
                )
            ),
            output_path=verify_path,
            requires_report=True,
        )
    )
    return WorkflowLanePlan(
        slug=spec.slug,
        lane_id=spec.lane_id,
        model_id=spec.model_id,
        execution_mode=execution_mode,
        preset=spec.preset_relpath,
        lane_root=lane_root,
        published_lane_root=published_lane_root,
        report_path=report_path,
        verify_path=verify_path,
        profile=lane_profile,
        steps=tuple(steps),
        resource_preflight=lane_resource_preflight(spec, env=env, device=device),
        prepared_preset=prepared_preset,
    )


def build_sweep_plan(
    *,
    args: argparse.Namespace,
    output_root: Path,
    execution_root: Path,
    specs: Sequence[EvidenceLane],
    env: dict[str, str],
    preset_overrides: dict[str, str],
) -> WorkflowSweepPlan:
    metadata = WorkflowRunMetadata(
        suite=args.suite,
        execution_mode=args.execution_mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    lanes = tuple(
        build_lane_plan(
            spec,
            output_root=output_root,
            execution_root=execution_root,
            python_exe=args.python,
            profile=args.profile,
            device=args.device,
            execution_mode=args.execution_mode,
            env=env,
            preset_overrides=preset_overrides,
        )
        for spec in specs
    )
    return WorkflowSweepPlan(
        metadata=metadata,
        output_root=output_root,
        execution_root=execution_root,
        lanes=lanes,
    )


def _classify_failure(
    *,
    log_path: Path,
    evaluate_exit: int,
    verify_exit: int | None,
    phase: str,
) -> tuple[str, str | None]:
    if evaluate_exit == 0 and verify_exit == 0:
        return ("ok", None)
    if evaluate_exit == -9:
        return ("failed", "resource_killed")
    try:
        text = log_path.read_text(encoding="utf-8").lower()
    except OSError:
        text = ""
    if phase == "prefetch" and (
        "gatedrepoerror" in text
        or "cannot access gated repo" in text
        or "you are trying to access a gated repo" in text
    ):
        return ("skipped", "gated_repo")
    if phase == "prefetch" and (
        "trust_remote_code=true" in text
        or "contains custom code which must be executed" in text
        or "loading this model requires you to execute custom code" in text
    ):
        return ("skipped", "remote_code_required")
    if (
        "invalid baseline metrics.ppl_final" in text
        or "primary metric degraded or non-finite" in text
    ):
        return ("failed", "invalid_primary_metric")
    if evaluate_exit == 125 and (
        "docker:" in text
        and (
            "error from registry: denied" in text
            or "unable to find image" in text
            or "pull access denied" in text
        )
    ):
        return ("failed", "container_image_pull_denied")
    if evaluate_exit != 0:
        return ("failed", f"{phase}_failed")
    if verify_exit not in {None, 0}:
        return ("failed", "verify_failed")
    return ("failed", None)


def _lane_env_for_spec(spec: EvidenceLane, env: dict[str, str]) -> dict[str, str]:
    lane_env = dict(env)
    if lane_requires_remote_code(spec):
        lane_env["INVARLOCK_ALLOW_REMOTE_CODE"] = "1"
    return lane_env


def _after_successful_model_step(
    spec: EvidenceLane,
    plan: WorkflowLanePlan,
    step: WorkflowCommandStep,
) -> None:
    if step.name != "materialize_dataset":
        return
    write_prepared_preset(
        spec,
        lane_root=plan.lane_root,
        execution_mode=plan.execution_mode,
    )


def _after_model_lane(
    plan: WorkflowLanePlan,
    *,
    output_root: Path,
    execution_root: Path,
) -> None:
    _publish_and_cleanup_lane_artifacts(
        lane_root=plan.lane_root,
        published_lane_root=plan.published_lane_root,
        execution_root=execution_root,
        output_root=output_root,
    )


def _model_lane_result_from_state(
    plan: WorkflowLanePlan,
    state: WorkflowLaneRunState,
    log_path: Path,
) -> LaneResult:
    base = state.to_lane_result()
    published_report_path = (
        plan.published_lane_root / "report" / "evaluation.report.json"
    )
    published_verify_path = plan.published_lane_root / "verify.json"
    failed_phase = next((phase for phase in state.phases if not phase.ok), None)

    if failed_phase is None:
        status, detail = _classify_failure(
            log_path=log_path,
            evaluate_exit=base.evaluate_exit,
            verify_exit=base.verify_exit,
            phase="evaluate",
        )
    elif failed_phase.name == "materialize_dataset":
        status, detail = ("failed", "dataset_materialize_failed")
    elif failed_phase.name == "verify" and failed_phase.detail == "report_missing":
        status, detail = ("failed", "report_missing")
    else:
        phase_exit = (
            base.evaluate_exit
            if failed_phase.returncode is None
            else int(failed_phase.returncode)
        )
        status, detail = _classify_failure(
            log_path=log_path,
            evaluate_exit=phase_exit,
            verify_exit=base.verify_exit,
            phase=failed_phase.name,
        )

    return LaneResult(
        slug=plan.slug,
        lane_id=plan.lane_id,
        model_id=plan.model_id,
        preset=plan.preset,
        evaluate_exit=base.evaluate_exit,
        verify_exit=base.verify_exit,
        report_path=str(published_report_path),
        verify_path=(
            str(published_verify_path) if published_verify_path.is_file() else None
        ),
        status=status,
        detail=detail,
    )


def run_lane(
    spec: EvidenceLane,
    *,
    output_root: Path,
    execution_root: Path,
    python_exe: str,
    profile: str | None,
    device: str,
    execution_mode: str,
    env: dict[str, str],
    preset_overrides: dict[str, str] | None = None,
    plan: WorkflowLanePlan | None = None,
) -> LaneResult:
    if plan is None:
        plan = build_lane_plan(
            spec,
            output_root=output_root,
            execution_root=execution_root,
            python_exe=python_exe,
            profile=profile,
            device=device,
            execution_mode=execution_mode,
            env=env,
            preset_overrides=preset_overrides,
        )
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.slug}.log"
    return execute_workflow_lane(
        WorkflowLaneExecutionRequest(
            plan=plan,
            cwd=REPO_ROOT,
            env=_lane_env_for_spec(spec, env),
            log_path=log_path,
        ),
        after_successful_step=lambda lane_plan, step, _run: (
            _after_successful_model_step(spec, lane_plan, step)
        ),
        after_lane=lambda lane_plan, _state: _after_model_lane(
            lane_plan,
            output_root=output_root,
            execution_root=execution_root,
        ),
        lane_result=_model_lane_result_from_state,
    )


def run_sweep(args: argparse.Namespace) -> int:
    if args.execution_mode == "host" and args.profile in {"ci", "release"}:
        print(
            "--execution-mode host is incompatible with --profile ci/release; "
            "omit --profile or use --profile dev for host-side evidence runs.",
            file=sys.stderr,
        )
        return 2

    validate_manifest_coverage(CURRENT_SUPPORTED_EXPERIMENTAL_LANES)
    try:
        preset_overrides = parse_preset_overrides(args.preset_override)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    specs = select_specs(
        args.suite,
        slugs=args.slug,
        lane_ids=args.lane_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not specs:
        print("No evidence lanes selected.", file=sys.stderr)
        return 2

    if args.list_json:
        print(json.dumps([spec.to_manifest_entry() for spec in specs], indent=2))
        return 0

    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else default_output_root()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    execution_root = _execution_root(output_root, execution_mode=args.execution_mode)
    if execution_root != output_root:
        execution_root.mkdir(parents=True, exist_ok=True)
    env = runtime_env()
    plan = build_sweep_plan(
        args=args,
        output_root=output_root,
        execution_root=execution_root,
        specs=specs,
        env=env,
        preset_overrides=preset_overrides,
    )
    write_manifest(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        specs=specs,
    )
    if args.dry_run:
        print(json.dumps(plan.to_dry_run_payload(), indent=2))
        return 0

    specs_by_slug = {spec.slug: spec for spec in specs}
    results = execute_workflow_sweep(
        WorkflowSweepExecutionRequest(
            plan=plan,
            cwd=REPO_ROOT,
            env=env,
            fail_fast=bool(args.fail_fast),
            status_log_path=output_root / "status.log",
            log_dir=output_root / "logs",
        ),
        lane_env=lambda lane_plan, base_env: _lane_env_for_spec(
            specs_by_slug[lane_plan.slug],
            dict(base_env),
        ),
        after_successful_step=lambda lane_plan, step, _run: (
            _after_successful_model_step(specs_by_slug[lane_plan.slug], lane_plan, step)
        ),
        after_lane=lambda lane_plan, _state: _after_model_lane(
            lane_plan,
            output_root=output_root,
            execution_root=execution_root,
        ),
        lane_result=_model_lane_result_from_state,
    )
    write_summary(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        results=results,
    )
    write_model_revisions(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        specs=specs,
        env=env,
    )
    write_artifact_manifest(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        results=results,
    )
    return workflow_return_code(results)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        return run_sweep(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
