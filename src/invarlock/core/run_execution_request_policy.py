from __future__ import annotations

import os
from typing import Mapping, Protocol

from invarlock.core.run_orchestrator import RunExecutionRequest


class SupportsRunExecutionRequest(Protocol):
    config: str
    device: str | None
    profile: str | None
    out: str | None
    edit: str | None
    edit_label: str | None
    tier: str | None
    metric_kind: str | None
    probes: int | None
    until_pass: bool
    max_attempts: int
    timeout: int | None
    baseline: str | None
    no_cleanup: bool
    timing: bool
    progress: bool
    telemetry: bool
    prefer_local_files_only: bool


def env_flag(name: str, *, environ: Mapping[str, str] | None = None) -> bool:
    source = environ if environ is not None else os.environ
    return str(source.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def env_text(name: str, *, environ: Mapping[str, str] | None = None) -> str | None:
    source = environ if environ is not None else os.environ
    value = source.get(name)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_run_execution_request(
    request: SupportsRunExecutionRequest,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunExecutionRequest:
    return RunExecutionRequest(
        config=request.config,
        device=request.device,
        profile=request.profile,
        out=request.out,
        edit=request.edit,
        edit_label=request.edit_label,
        tier=request.tier,
        metric_kind=request.metric_kind,
        probes=request.probes,
        until_pass=bool(request.until_pass),
        max_attempts=int(request.max_attempts),
        timeout=request.timeout,
        baseline=request.baseline,
        no_cleanup=bool(request.no_cleanup),
        capture_timings=bool(request.timing or request.progress),
        telemetry=bool(request.telemetry),
        prefer_local_files_only=bool(request.prefer_local_files_only),
        eval_device_override=env_text("INVARLOCK_EVAL_DEVICE", environ=environ),
        determinism_mode=env_text("PACK_DETERMINISM", environ=environ)
        or env_text("INVARLOCK_DETERMINISM", environ=environ),
        determinism_warn_only=env_flag(
            "INVARLOCK_DETERMINISM_WARN_ONLY", environ=environ
        ),
        tiny_relax_enabled=env_flag("INVARLOCK_TINY_RELAX", environ=environ),
        export_model_requested=env_flag("INVARLOCK_EXPORT_MODEL", environ=environ),
        export_dir=env_text("INVARLOCK_EXPORT_DIR", environ=environ),
    )


__all__ = [
    "SupportsRunExecutionRequest",
    "build_run_execution_request",
    "env_flag",
    "env_text",
]
