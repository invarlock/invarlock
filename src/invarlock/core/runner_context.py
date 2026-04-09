from __future__ import annotations

import os
from typing import Any

from .api import RunConfig

_CUDA_FLAG_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
    TypeError,
    ValueError,
)

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return None


def env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return coerce_bool(raw)


def collect_cuda_flags() -> dict[str, Any]:
    """Capture deterministic CUDA configuration for provenance."""
    flags: dict[str, Any] = {}
    try:
        import torch

        flags["deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        if hasattr(torch.backends, "cudnn"):
            flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            matmul = torch.backends.cuda.matmul
            if hasattr(matmul, "allow_tf32"):
                flags["cuda_matmul_allow_tf32"] = bool(matmul.allow_tf32)
    except _CUDA_FLAG_ERRORS:  # pragma: no cover - fallback when torch missing
        pass

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace:
        flags["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return flags


def serialize_config(config: RunConfig) -> dict[str, Any]:
    """Serialize RunConfig for storage in report."""
    return {
        "device": config.device,
        "max_pm_ratio": config.max_pm_ratio,
        "checkpoint_interval": config.checkpoint_interval,
        "dry_run": config.dry_run,
        "verbose": config.verbose,
        "guards": config.context.get("guards", {}) if config.context else {},
    }


def resolve_policy_flags(config: RunConfig | None) -> dict[str, bool]:
    run_ctx: dict[str, Any] = {}
    eval_ctx: dict[str, Any] = {}
    if config and isinstance(config.context, dict):
        run_ctx = (
            config.context.get("run", {})
            if isinstance(config.context.get("run"), dict)
            else {}
        )
        eval_ctx = (
            config.context.get("eval", {})
            if isinstance(config.context.get("eval"), dict)
            else {}
        )

    def resolve_flag(
        *,
        run_key: str,
        eval_keys: tuple[str, ...],
        env_key: str | None,
        default: bool,
    ) -> bool:
        val = coerce_bool(run_ctx.get(run_key))
        if val is None:
            for key in eval_keys:
                val = coerce_bool(eval_ctx.get(key))
                if val is not None:
                    break
        if env_key:
            env_val = env_flag(env_key)
            if env_val is not None:
                val = env_val
        return default if val is None else bool(val)

    return {
        "strict_eval": resolve_flag(
            run_key="strict_eval",
            eval_keys=("strict_errors", "strict"),
            env_key=None,
            default=True,
        ),
        "strict_guard_prepare": resolve_flag(
            run_key="strict_guard_prepare",
            eval_keys=(),
            env_key=None,
            default=True,
        ),
        "allow_calibration_materialize": resolve_flag(
            run_key="allow_calibration_materialize",
            eval_keys=("materialize_calibration", "allow_iterable_calibration"),
            env_key="INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE",
            default=False,
        ),
    }


__all__ = [
    "coerce_bool",
    "collect_cuda_flags",
    "env_flag",
    "resolve_policy_flags",
    "serialize_config",
]
