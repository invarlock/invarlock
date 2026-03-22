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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX_PATH = REPO_ROOT / "contracts" / "support_matrix.json"
DEFAULT_SUITE = "current-supported-experimental"
EXECUTION_MODES = ("container", "host")


@dataclass(frozen=True)
class EvidenceLane:
    slug: str
    lane_id: str
    family: str
    model_id: str
    preset_relpath: str
    adapter: str = "auto"
    verify_profile: str = "ci"

    @property
    def preset_path(self) -> Path:
        return REPO_ROOT / self.preset_relpath

    def preset_arg(self, *, execution_mode: str) -> str:
        if execution_mode == "container":
            return self.preset_relpath
        return str(self.preset_path)

    def to_manifest_entry(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "lane_id": self.lane_id,
            "family": self.family,
            "model_id": self.model_id,
            "preset": self.preset_relpath,
            "adapter": self.adapter,
            "verify_profile": self.verify_profile,
        }


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

    @property
    def ok(self) -> bool:
        return self.evaluate_exit == 0 and self.verify_exit == 0

    def to_summary_entry(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


CURRENT_SUPPORTED_EXPERIMENTAL_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="mistral_7b",
        lane_id="mistral-7b-causal-hf",
        family="Mistral 7B causal LM",
        model_id="mistralai/Mistral-7B-v0.1",
        preset_relpath="configs/presets/causal_lm/mistral_7b_512.yaml",
    ),
    EvidenceLane(
        slug="qwen2_7b",
        lane_id="qwen2-7b-causal-hf",
        family="Qwen2 7B causal LM",
        model_id="Qwen/Qwen2-7B",
        preset_relpath="configs/presets/causal_lm/qwen2_7b_512.yaml",
    ),
    EvidenceLane(
        slug="qwen3_8b",
        lane_id="qwen3-causal-hf",
        family="Qwen3 causal LM",
        model_id="Qwen/Qwen3-8B",
        preset_relpath="configs/presets/causal_lm/qwen3_8b_512.yaml",
    ),
    EvidenceLane(
        slug="qwq_32b",
        lane_id="qwq-32b-reasoning-causal-hf",
        family="QwQ-32B reasoning causal LM",
        model_id="Qwen/QwQ-32B",
        preset_relpath="configs/presets/causal_lm/qwq_32b_512.yaml",
    ),
    EvidenceLane(
        slug="deepseek_r1_distill_qwen_7b",
        lane_id="deepseek-r1-distill-qwen-causal-hf",
        family="DeepSeek-R1-Distill-Qwen causal LM",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        preset_relpath="configs/presets/causal_lm/deepseek_r1_distill_qwen_7b_512.yaml",
    ),
    EvidenceLane(
        slug="phi4_reasoning_plus",
        lane_id="phi-4-text-causal-hf",
        family="Phi-4 causal LM (text-only eval)",
        model_id="microsoft/Phi-4-reasoning-plus",
        preset_relpath="configs/presets/causal_lm/phi4_reasoning_plus_512.yaml",
    ),
    EvidenceLane(
        slug="tinyllama_1_1b",
        lane_id="tinyllama-1-1b-causal-hf",
        family="TinyLlama 1.1B causal LM",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        preset_relpath="configs/presets/causal_lm/tinyllama_1_1b_512.yaml",
    ),
    EvidenceLane(
        slug="olmo2_7b",
        lane_id="olmo-2-causal-hf",
        family="OLMo 2 causal LM",
        model_id="allenai/OLMo-2-1124-7B",
        preset_relpath="configs/presets/causal_lm/olmo2_7b_512.yaml",
    ),
    EvidenceLane(
        slug="olmo2_13b",
        lane_id="olmo-2-causal-hf",
        family="OLMo 2 causal LM",
        model_id="allenai/OLMo-2-1124-13B-Instruct",
        preset_relpath="configs/presets/causal_lm/olmo2_13b_512.yaml",
    ),
    EvidenceLane(
        slug="qwen3_5_9b",
        lane_id="qwen3-5-causal-hf",
        family="Qwen3.5 causal LM",
        model_id="Qwen/Qwen3.5-9B",
        preset_relpath="configs/presets/causal_lm/qwen3_5_9b_512.yaml",
    ),
)

SUITES: dict[str, tuple[EvidenceLane, ...]] = {
    DEFAULT_SUITE: CURRENT_SUPPORTED_EXPERIMENTAL_LANES,
}


def _load_support_matrix(path: Path = SUPPORT_MATRIX_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Support matrix must be a JSON object")
    return payload


def supported_experimental_lane_ids(
    support_matrix: dict[str, object] | None = None,
) -> set[str]:
    payload = support_matrix or _load_support_matrix()
    lanes = payload.get("lanes") or []
    if not isinstance(lanes, list):
        raise ValueError("support_matrix.lanes must be a list")
    lane_ids: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if lane.get("support_tier") != "supported_experimental":
            continue
        lane_id = lane.get("lane_id")
        if isinstance(lane_id, str) and lane_id:
            lane_ids.add(lane_id)
    return lane_ids


def manifest_lane_ids(specs: tuple[EvidenceLane, ...] | list[EvidenceLane]) -> set[str]:
    return {spec.lane_id for spec in specs}


def validate_manifest_coverage(
    specs: tuple[EvidenceLane, ...] | list[EvidenceLane],
    support_matrix: dict[str, object] | None = None,
) -> None:
    expected = supported_experimental_lane_ids(support_matrix)
    actual = manifest_lane_ids(specs)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing lane_ids: " + ", ".join(missing))
        if extra:
            parts.append("unexpected lane_ids: " + ", ".join(extra))
        raise ValueError("Model evidence manifest drift: " + "; ".join(parts))


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
        default="ci",
        help="Profile to pass to evaluate and verify.",
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
            "secure-default runtime-container path; 'host' adds the explicit "
            "host-bypass and verify override flags."
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


def select_specs(
    suite: str,
    *,
    slugs: list[str],
    lane_ids: list[str],
    shard_index: int,
    shard_count: int,
) -> list[EvidenceLane]:
    if shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must be within [0, shard-count)")

    specs = list(SUITES[suite])
    if slugs:
        slug_set = set(slugs)
        specs = [spec for spec in specs if spec.slug in slug_set]
        missing = sorted(slug_set - {spec.slug for spec in specs})
        if missing:
            raise ValueError("Unknown slugs: " + ", ".join(missing))
    if lane_ids:
        lane_id_set = set(lane_ids)
        specs = [spec for spec in specs if spec.lane_id in lane_id_set]
        missing = sorted(lane_id_set - {spec.lane_id for spec in specs})
        if missing:
            raise ValueError("Unknown lane_ids: " + ", ".join(missing))

    return [spec for idx, spec in enumerate(specs) if idx % shard_count == shard_index]


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
        "--adapter",
        spec.adapter,
        "--preset",
        spec.preset_arg(execution_mode=execution_mode),
        "--profile",
        profile,
        "--allow-network",
        "--device",
        device,
        "--out",
        str(lane_root / "runs"),
        "--report-out",
        str(lane_root / "report"),
    ]
    if execution_mode == "host":
        command.append("--allow-host-execution")
    return command


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
        command.insert(-1, "--allow-unattested-artifacts")
    return command


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


def _publish_lane_artifacts(source: Path, destination: Path) -> None:
    if source == destination or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def run_lane(
    spec: EvidenceLane,
    *,
    output_root: Path,
    execution_root: Path,
    python_exe: str,
    profile: str,
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

    evaluate_cmd = build_evaluate_command(
        spec,
        python_exe=python_exe,
        profile=profile,
        device=device,
        execution_mode=execution_mode,
        lane_root=lane_root,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(evaluate_cmd) + "\n")
        eval_proc = subprocess.run(
            evaluate_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    verify_exit: int | None = None
    if eval_proc.returncode == 0 and report_path.is_file():
        verify_cmd = build_verify_command(
            python_exe=python_exe,
            profile=profile,
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
                env=env,
                stdout=verify_file,
                stderr=log_file,
                text=True,
                check=False,
            )
            verify_exit = verify_proc.returncode

    _publish_lane_artifacts(lane_root, published_lane_root)
    published_report_path = published_lane_root / "report" / "evaluation.report.json"
    published_verify_path = published_lane_root / "verify.json"

    return LaneResult(
        slug=spec.slug,
        lane_id=spec.lane_id,
        model_id=spec.model_id,
        preset=spec.preset_relpath,
        evaluate_exit=eval_proc.returncode,
        verify_exit=verify_exit,
        report_path=str(published_report_path),
        verify_path=str(published_verify_path) if published_verify_path.is_file() else None,
    )


def write_summary(
    output_root: Path,
    *,
    suite: str,
    execution_mode: str,
    shard_index: int,
    shard_count: int,
    results: list[LaneResult],
) -> None:
    summary_tsv = output_root / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8") as handle:
        handle.write("slug\tlane_id\tevaluate_exit\tverify_exit\treport\n")
        for result in results:
            verify_exit = (
                "NA" if result.verify_exit is None else str(result.verify_exit)
            )
            handle.write(
                f"{result.slug}\t{result.lane_id}\t{result.evaluate_exit}\t"
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
    specs: list[EvidenceLane],
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


def run_sweep(args: argparse.Namespace) -> int:
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
            payload.append(
                {
                    "slug": spec.slug,
                    "execution_mode": args.execution_mode,
                    "evaluate": build_evaluate_command(
                        spec,
                        python_exe=args.python,
                        profile=args.profile,
                        device=args.device,
                        execution_mode=args.execution_mode,
                        lane_root=lane_root,
                    ),
                    "verify": build_verify_command(
                        python_exe=args.python,
                        profile=args.profile,
                        execution_mode=args.execution_mode,
                        report_path=lane_root / "report" / "evaluation.report.json",
                    ),
                }
            )
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
