"""Model export helpers for run orchestrator attempt execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _should_export_model(
    *,
    output_cfg: Any,
    export_model_requested: bool,
) -> bool:
    save_model_cfg = False
    try:
        if isinstance(output_cfg, dict):
            save_model_cfg = bool(output_cfg.get("save_model", False))
        else:
            save_model_cfg = bool(getattr(output_cfg, "save_model", False))
    except (AttributeError, TypeError):
        save_model_cfg = False
    return bool(export_model_requested) or save_model_cfg


def _resolve_export_model_dir(
    *,
    output_cfg: Any,
    run_dir: Path,
    export_dir_override: str | None,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> Path:
    export_dir: Path | None = None
    try:
        model_dir_cfg = None
        if isinstance(output_cfg, dict):
            model_dir_cfg = output_cfg.get("model_dir") or output_cfg.get("model_path")
        elif output_cfg is not None:
            model_dir_cfg = getattr(output_cfg, "model_dir", None) or getattr(
                output_cfg,
                "model_path",
                None,
            )
        if model_dir_cfg:
            candidate = Path(str(model_dir_cfg))
            export_dir = candidate if candidate.is_absolute() else (run_dir / candidate)
    except optional_runtime_exceptions:
        export_dir = None
    if export_dir is None and isinstance(export_dir_override, str):
        if export_dir_override.strip():
            candidate = Path(export_dir_override.strip())
            export_dir = candidate if candidate.is_absolute() else (run_dir / candidate)
    if export_dir is not None:
        return export_dir
    try:
        if isinstance(output_cfg, dict):
            resolved_export_subdir = str(output_cfg.get("model_subdir", "model"))
        else:
            resolved_export_subdir = str(getattr(output_cfg, "model_subdir", "model"))
    except optional_runtime_exceptions:
        resolved_export_subdir = "model"
    return run_dir / resolved_export_subdir


def _maybe_export_model_artifacts(
    *,
    cfg: Any,
    run_dir: Path,
    report: dict[str, Any],
    adapter: Any,
    model: Any | None,
    tokenizer: Any | None,
    export_model_requested: bool,
    export_dir_override: str | None,
    cfg_value: Any,
    emit_diagnostic: Any,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> None:
    output_cfg = cfg_value(cfg, "output") or {}
    if not _should_export_model(
        output_cfg=output_cfg,
        export_model_requested=export_model_requested,
    ):
        return
    try:
        export_dir = _resolve_export_model_dir(
            output_cfg=output_cfg,
            run_dir=run_dir,
            export_dir_override=export_dir_override,
            optional_runtime_exceptions=optional_runtime_exceptions,
        )
        ok = False
        if hasattr(adapter, "save_pretrained") and model is not None:
            ok = bool(adapter.save_pretrained(model, export_dir))
        if not ok:
            emit_diagnostic(code="export_adapter_directory_missing")
            return
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if callable(save_tokenizer):
            try:
                save_tokenizer(str(export_dir))
            except optional_runtime_exceptions:
                emit_diagnostic(code="export_tokenizer_missing")
        report["artifacts"]["checkpoint_path"] = str(export_dir)
    except optional_runtime_exceptions:
        emit_diagnostic(code="export_failed")


__all__ = [
    "_maybe_export_model_artifacts",
    "_resolve_export_model_dir",
    "_should_export_model",
]
