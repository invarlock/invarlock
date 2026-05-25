"""Command construction helpers for model evidence sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def command_path(path: Path, *, execution_mode: str, repo_root: Path) -> str:
    if execution_mode == "container":
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def build_evaluate_command(
    spec: Any,
    *,
    python_exe: str,
    profile: str,
    device: str,
    execution_mode: str,
    lane_root: Path,
    repo_root: Path,
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
        command_path(
            lane_root / "runs", execution_mode=execution_mode, repo_root=repo_root
        ),
        "--report-out",
        command_path(
            lane_root / "report", execution_mode=execution_mode, repo_root=repo_root
        ),
    ]
    if execution_mode == "host":
        command.extend(["--execution-mode", "host"])
    return command


def prefetch_adapter_name(spec: Any) -> str:
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
    spec: Any,
    *,
    python_exe: str,
) -> list[str]:
    adapter_name = prefetch_adapter_name(spec)
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
    *, profile_override: str | None, execution_mode: str, spec: Any
) -> str:
    if profile_override:
        return profile_override
    if execution_mode == "host":
        return "dev"
    return spec.verify_profile


__all__ = [
    "build_evaluate_command",
    "build_prefetch_command",
    "build_verify_command",
    "command_path",
    "prefetch_adapter_name",
    "resolve_lane_profile",
]
