"""Artifact/provenance helpers for `cli.commands.run`."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def persist_ref_masks(core_report: Any, run_dir: Path) -> Path | None:
    """Persist reference keep indices to artifact if present."""
    edit_section = (
        core_report.get("edit")
        if isinstance(core_report, dict)
        else getattr(core_report, "edit", None)
    )
    if not isinstance(edit_section, dict):
        return None

    artifacts_section = edit_section.get("artifacts")
    if not isinstance(artifacts_section, dict):
        return None

    mask_payload = artifacts_section.get("mask_payload")
    if not isinstance(mask_payload, dict) or not mask_payload:
        return None

    payload_copy = copy.deepcopy(mask_payload)
    meta_section = payload_copy.setdefault("meta", {})
    meta_section.setdefault("generated_at", datetime.now().isoformat())

    target_dir = run_dir / "artifacts" / "edit_masks"
    target_dir.mkdir(parents=True, exist_ok=True)
    mask_path = target_dir / "masks.json"
    with mask_path.open("w", encoding="utf-8") as handle:
        json.dump(payload_copy, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return mask_path


def resolve_exit_code(
    exc: Exception,
    *,
    profile: str | None,
    config_error_cls: type[BaseException],
    validation_error_cls: type[BaseException],
    data_error_cls: type[BaseException],
    invarlock_error_cls: type[BaseException],
) -> int:
    """Resolve CLI exit code based on error class and active profile."""
    try:
        prof = (profile or "").strip().lower()
    except Exception:
        prof = ""
    if isinstance(exc, config_error_cls | validation_error_cls | data_error_cls):
        return 2
    if isinstance(exc, ValueError) and "Invalid RunReport" in str(exc):
        return 2
    if isinstance(exc, invarlock_error_cls) and prof in {"ci", "release"}:
        return 3
    return 1
