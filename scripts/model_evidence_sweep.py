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
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX_PATH = REPO_ROOT / "contracts" / "support_matrix.json"
MODEL_FAMILY_CATALOG_PATH = REPO_ROOT / "contracts" / "model_family_catalog.json"
DEFAULT_SUITE = "current-supported-experimental"
REPO_MENTIONED_GPU_SUITE = "repo-mentioned-gpu"
MODEL_CATALOG_GPU_SUITE = "model-catalog-gpu"
EXECUTION_MODES = ("container", "host")
RETRYABLE_EVALUATE_RETURNCODES = {-15}


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
    status: str = "failed"
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

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
        slug="qwen2_5_14b",
        lane_id="qwen2-5-14b-causal-hf",
        family="Qwen2.5 14B causal LM",
        model_id="Qwen/Qwen2.5-14B",
        preset_relpath="configs/presets/causal_lm/qwen2_5_14b_512.yaml",
    ),
    EvidenceLane(
        slug="qwen3_8b",
        lane_id="qwen3-causal-hf",
        family="Qwen3 causal LM",
        model_id="Qwen/Qwen3-8B",
        preset_relpath="configs/presets/causal_lm/qwen3_8b_512.yaml",
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
    EvidenceLane(
        slug="gemma4_e2b",
        lane_id="gemma4-e2b-text-causal-hf",
        family="Gemma 4 E2B causal LM (text-only eval)",
        model_id="google/gemma-4-E2B-it",
        preset_relpath="configs/presets/causal_lm/gemma4_e2b_512.yaml",
        adapter="hf_causal",
    ),
    EvidenceLane(
        slug="ministral3_8b",
        lane_id="ministral-3-text-causal-hf",
        family="Ministral 3 causal LM (text-only eval)",
        model_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        preset_relpath="configs/presets/causal_lm/ministral3_8b_512.yaml",
    ),
    EvidenceLane(
        slug="ministral3_14b",
        lane_id="ministral-3-text-causal-hf",
        family="Ministral 3 causal LM (text-only eval)",
        model_id="mistralai/Ministral-3-14B-Instruct-2512-BF16",
        preset_relpath="configs/presets/causal_lm/ministral3_14b_512.yaml",
    ),
)

CURRENT_PUBLISHED_BASIS_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="gpt2_public",
        lane_id="published-gpt2-causal-hf",
        family="GPT-2 causal LM",
        model_id="gpt2",
        preset_relpath="configs/presets/causal_lm/wikitext2_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="bert_base_uncased_public",
        lane_id="published-bert-base-uncased-mlm-hf",
        family="BERT / RoBERTa MLM",
        model_id="bert-base-uncased",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="roberta_base_public",
        lane_id="published-roberta-base-mlm-hf",
        family="BERT / RoBERTa MLM",
        model_id="roberta-base",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
)

DOCUMENTED_SMOKE_CANARY_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="tiny_gpt2_canary",
        lane_id="smoke-tiny-gpt2-causal-hf",
        family="GPT-2 causal LM smoke canary",
        model_id="sshleifer/tiny-gpt2",
        preset_relpath="configs/presets/causal_lm/wikitext2_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="bert_tiny_canary",
        lane_id="smoke-bert-tiny-mlm-hf",
        family="BERT MLM smoke canary",
        model_id="prajjwal1/bert-tiny",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
)

MODEL_FAMILY_CATALOG_SECTIONS = (
    "declared_support",
    "implemented_coverage",
    "usage_only",
    "recommended_additions",
)


def _load_model_family_catalog(
    path: Path = MODEL_FAMILY_CATALOG_PATH,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model family catalog must be a JSON object")
    return payload


def _catalog_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "_")
    for old, new in ((".", "_"), ("-", "_"), ("+", "_")):
        slug = slug.replace(old, new)
    return slug


def _catalog_lane_defaults(model_id: str) -> tuple[str, str]:
    model_lower = model_id.lower()
    if any(
        keyword in model_lower
        for keyword in (
            "bert",
            "roberta",
            "deberta",
            "distilbert",
            "albert",
            "electra",
        )
    ):
        return ("configs/presets/masked_lm/wikitext2_128.yaml", "hf_mlm")
    if any(
        keyword in model_lower
        for keyword in ("t5", "bart", "mbart", "pegasus", "marian", "opus-mt")
    ):
        return ("configs/presets/seq2seq/synth_128.yaml", "hf_seq2seq")
    if model_lower == "google/gemma-4-e4b-it":
        return (
            "configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml",
            "hf_multimodal",
        )
    return ("configs/presets/causal_lm/wikitext2_512.yaml", "auto")


def _build_model_catalog_gpu_lanes(
    payload: dict[str, object] | None = None,
) -> tuple[EvidenceLane, ...]:
    catalog = payload or _load_model_family_catalog()
    lanes: list[EvidenceLane] = []
    seen: set[str] = set()
    for section in MODEL_FAMILY_CATALOG_SECTIONS:
        families = catalog.get(section) or []
        if not isinstance(families, list):
            raise ValueError(f"model_family_catalog.{section} must be a list")
        for family in families:
            if not isinstance(family, dict):
                continue
            display_name = family.get("display_name")
            family_label = display_name if isinstance(display_name, str) else section
            models = family.get("representative_models") or []
            if not isinstance(models, list):
                continue
            for model_id in models:
                if not isinstance(model_id, str) or not model_id or model_id in seen:
                    continue
                preset_relpath, adapter = _catalog_lane_defaults(model_id)
                lanes.append(
                    EvidenceLane(
                        slug=_catalog_slug(model_id),
                        lane_id=f"catalog::{_catalog_slug(model_id)}",
                        family=family_label,
                        model_id=model_id,
                        preset_relpath=preset_relpath,
                        adapter=adapter,
                        verify_profile="dev",
                    )
                )
                seen.add(model_id)
    return tuple(lanes)


MODEL_CATALOG_GPU_LANES = _build_model_catalog_gpu_lanes()

SUITES: dict[str, tuple[EvidenceLane, ...]] = {
    DEFAULT_SUITE: CURRENT_SUPPORTED_EXPERIMENTAL_LANES,
    REPO_MENTIONED_GPU_SUITE: (
        CURRENT_PUBLISHED_BASIS_LANES
        + DOCUMENTED_SMOKE_CANARY_LANES
        + CURRENT_SUPPORTED_EXPERIMENTAL_LANES
    ),
    MODEL_CATALOG_GPU_SUITE: MODEL_CATALOG_GPU_LANES,
}


@cache
def _preset_model_config(preset_path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(preset_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    model_cfg = data.get("model")
    return model_cfg if isinstance(model_cfg, dict) else {}


def lane_requires_remote_code(spec: EvidenceLane) -> bool:
    model_cfg = _preset_model_config(str(spec.preset_path))
    return bool(model_cfg.get("trust_remote_code") is True)


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
        _command_path(lane_root / "runs", execution_mode=execution_mode),
        "--report-out",
        _command_path(lane_root / "report", execution_mode=execution_mode),
    ]
    if execution_mode == "host":
        command.extend(["--assurance", "trusted-local"])
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
        command[4:4] = ["--assurance", "trusted-local"]
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


def _command_path(path: Path, *, execution_mode: str) -> str:
    if execution_mode == "container" and _is_within_repo(path):
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
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
    lane_env = dict(env)
    if lane_requires_remote_code(spec):
        lane_env["INVARLOCK_ALLOW_REMOTE_CODE"] = "1"

    log_mode = "w"
    eval_returncode: int | None = None
    lane_profile = profile or spec.verify_profile
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
                verify_path=str(published_verify_path)
                if published_verify_path.is_file()
                else None,
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
        verify_path=str(published_verify_path)
        if published_verify_path.is_file()
        else None,
        status=status,
        detail=detail,
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
            item = {
                "slug": spec.slug,
                "execution_mode": args.execution_mode,
                "evaluate": build_evaluate_command(
                    spec,
                    python_exe=args.python,
                    profile=args.profile or spec.verify_profile,
                    device=args.device,
                    execution_mode=args.execution_mode,
                    lane_root=lane_root,
                ),
                "verify": build_verify_command(
                    python_exe=args.python,
                    profile=args.profile or spec.verify_profile,
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
