#!/usr/bin/env python3
"""Run the shipped-model evidence sweep for supported experimental lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
    EvidenceLane,
    lane_requires_remote_code,
    manifest_lane_ids,
    select_specs,
    supported_experimental_lane_ids,
    validate_manifest_coverage,
)

RETRYABLE_EVALUATE_RETURNCODES = {-15}

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
    "EvidenceLane",
    "manifest_lane_ids",
    "select_specs",
    "supported_experimental_lane_ids",
]


@dataclass(frozen=True)
class LaneResult:
    slug: str
    lane_id: str
    model_id: str
    preset: str
    evaluate_exit: int
    verify_exit: int | None
    report_path: str
    verify_path: str | None
    status: str = "failed"
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_summary_entry(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def write_summary(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    shard_index: int,
    shard_count: int,
    results: Sequence[LaneResult],
) -> None:
    summary_tsv = output_root / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "slug\tlane_id\tstatus\tdetail\tevaluate_exit\tverify_exit\treport\n"
        )
        for result in results:
            verify_exit = (
                "NA" if result.verify_exit is None else str(result.verify_exit)
            )
            handle.write(
                f"{result.slug}\t{result.lane_id}\t{result.status}\t"
                f"{result.detail or ''}\t{result.evaluate_exit}\t"
                f"{verify_exit}\t{result.report_path}\n"
            )

    payload = {
        "suite": suite,
        "execution_mode": execution_mode,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "ok": all(result.ok for result in results),
        "results": [result.to_summary_entry() for result in results],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
) -> list[str]:
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
        spec.preset_arg(execution_mode=execution_mode),
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
    if profile == "dev":
        command.extend(["--assurance", "off"])
    return command


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
    return env


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
    if evaluate_exit != 0:
        return ("failed", f"{phase}_failed")
    if verify_exit not in {None, 0}:
        return ("failed", "verify_failed")
    return ("failed", None)


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
) -> LaneResult:
    lane_root = execution_root / "eval" / spec.slug
    lane_root.mkdir(parents=True, exist_ok=True)
    published_lane_root = output_root / "eval" / spec.slug
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.slug}.log"
    report_path = lane_root / "report" / "evaluation.report.json"
    verify_path = lane_root / "verify.json"
    lane_env = dict(env)
    if lane_requires_remote_code(spec):
        lane_env["INVARLOCK_ALLOW_REMOTE_CODE"] = "1"

    log_mode = "w"
    eval_returncode: int | None = None
    lane_profile = resolve_lane_profile(
        profile_override=profile,
        execution_mode=execution_mode,
        spec=spec,
    )
    if execution_mode == "host":
        prefetch_cmd = build_prefetch_command(spec, python_exe=python_exe)
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(prefetch_cmd) + "\n")
            prefetch_proc = subprocess.run(
                prefetch_cmd,
                cwd=REPO_ROOT,
                env=lane_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        log_mode = "a"
        if prefetch_proc.returncode != 0:
            status, detail = _classify_failure(
                log_path=log_path,
                evaluate_exit=prefetch_proc.returncode,
                verify_exit=None,
                phase="prefetch",
            )
            _publish_lane_artifacts(
                lane_root, published_lane_root := output_root / "eval" / spec.slug
            )
            published_report_path = (
                published_lane_root / "report" / "evaluation.report.json"
            )
            published_verify_path = published_lane_root / "verify.json"
            return LaneResult(
                slug=spec.slug,
                lane_id=spec.lane_id,
                model_id=spec.model_id,
                preset=spec.preset_relpath,
                evaluate_exit=prefetch_proc.returncode,
                verify_exit=None,
                report_path=str(published_report_path),
                verify_path=(
                    str(published_verify_path)
                    if published_verify_path.is_file()
                    else None
                ),
                status=status,
                detail=detail,
            )

    evaluate_cmd = build_evaluate_command(
        spec,
        python_exe=python_exe,
        profile=lane_profile,
        device=device,
        execution_mode=execution_mode,
        lane_root=lane_root,
    )
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(evaluate_cmd) + "\n")
        eval_proc = subprocess.run(
            evaluate_cmd,
            cwd=REPO_ROOT,
            env=lane_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    eval_returncode = eval_proc.returncode
    if eval_returncode in RETRYABLE_EVALUATE_RETURNCODES:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[WARN] evaluate exited with {eval_returncode}; retrying once.\n"
            )
            log_file.write("$ " + " ".join(evaluate_cmd) + "\n")
            eval_proc = subprocess.run(
                evaluate_cmd,
                cwd=REPO_ROOT,
                env=lane_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        eval_returncode = eval_proc.returncode

    verify_exit: int | None = None
    if eval_returncode == 0 and report_path.is_file():
        verify_cmd = build_verify_command(
            python_exe=python_exe,
            profile=lane_profile,
            execution_mode=execution_mode,
            report_path=report_path,
        )
        with (
            verify_path.open("w", encoding="utf-8") as verify_file,
            log_path.open("a", encoding="utf-8") as log_file,
        ):
            log_file.write("\n$ " + " ".join(verify_cmd) + "\n")
            verify_proc = subprocess.run(
                verify_cmd,
                cwd=REPO_ROOT,
                env=lane_env,
                stdout=verify_file,
                stderr=log_file,
                text=True,
                check=False,
            )
            verify_exit = verify_proc.returncode

    _publish_lane_artifacts(lane_root, published_lane_root)
    published_report_path = published_lane_root / "report" / "evaluation.report.json"
    published_verify_path = published_lane_root / "verify.json"
    status, detail = _classify_failure(
        log_path=log_path,
        evaluate_exit=eval_returncode,
        verify_exit=verify_exit,
        phase="evaluate",
    )

    return LaneResult(
        slug=spec.slug,
        lane_id=spec.lane_id,
        model_id=spec.model_id,
        preset=spec.preset_relpath,
        evaluate_exit=eval_returncode,
        verify_exit=verify_exit,
        report_path=str(published_report_path),
        verify_path=(
            str(published_verify_path) if published_verify_path.is_file() else None
        ),
        status=status,
        detail=detail,
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
    write_manifest(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        specs=specs,
    )
    if args.dry_run:
        payload = []
        for spec in specs:
            lane_root = execution_root / "eval" / spec.slug
            item = {
                "slug": spec.slug,
                "execution_mode": args.execution_mode,
                "evaluate": build_evaluate_command(
                    spec,
                    python_exe=args.python,
                    profile=resolve_lane_profile(
                        profile_override=args.profile,
                        execution_mode=args.execution_mode,
                        spec=spec,
                    ),
                    device=args.device,
                    execution_mode=args.execution_mode,
                    lane_root=lane_root,
                ),
                "verify": build_verify_command(
                    python_exe=args.python,
                    profile=resolve_lane_profile(
                        profile_override=args.profile,
                        execution_mode=args.execution_mode,
                        spec=spec,
                    ),
                    execution_mode=args.execution_mode,
                    report_path=lane_root / "report" / "evaluation.report.json",
                ),
            }
            if args.execution_mode == "host":
                item["prefetch"] = build_prefetch_command(
                    spec,
                    python_exe=args.python,
                )
            payload.append(item)
        print(json.dumps(payload, indent=2))
        return 0

    status_log = output_root / "status.log"
    env = runtime_env()
    results: list[LaneResult] = []
    with status_log.open("w", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now(UTC).isoformat()}] START\n")
        for spec in specs:
            handle.write(f"[{datetime.now(UTC).isoformat()}] START {spec.slug}\n")
            handle.flush()
            result = run_lane(
                spec,
                output_root=output_root,
                execution_root=execution_root,
                python_exe=args.python,
                profile=args.profile,
                device=args.device,
                execution_mode=args.execution_mode,
                env=env,
            )
            results.append(result)
            verify_repr = (
                "NA" if result.verify_exit is None else str(result.verify_exit)
            )
            handle.write(
                f"[{datetime.now(UTC).isoformat()}] DONE {result.slug} "
                f"status={result.status} detail={result.detail or '-'} "
                f"eval={result.evaluate_exit} verify={verify_repr}\n"
            )
            handle.flush()
            if args.fail_fast and not result.ok:
                break
        handle.write(f"[{datetime.now(UTC).isoformat()}] ALL_TASKS_COMPLETE\n")
    write_summary(
        output_root,
        suite=args.suite,
        execution_mode=args.execution_mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        results=results,
    )
    return 0 if results and all(result.ok for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        return run_sweep(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
