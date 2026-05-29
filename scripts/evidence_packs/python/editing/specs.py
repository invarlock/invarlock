from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_FILE_READ_ERRORS = (OSError, TypeError, ValueError)


@dataclass(frozen=True)
class ResolvedEditSpec:
    status: str
    edit_type: str
    param1: str = ""
    param2: str = ""
    scope: str = ""
    version: str = ""
    edit_dir_name: str = ""
    reason: str = ""

    @property
    def skip(self) -> bool:
        return self.status == "skipped"

    @property
    def selected(self) -> bool:
        return self.status == "selected"

    def to_shell_payload(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reason": self.reason,
            "edit_type": self.edit_type,
            "param1": self.param1,
            "param2": self.param2,
            "scope": self.scope,
            "version": self.version,
            "edit_dir_name": self.edit_dir_name,
        }

    def to_batch_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.edit_type,
            "status": self.status,
            "reason": self.reason,
            "scope": self.scope,
            "edit_dir_name": self.edit_dir_name,
            "version": self.version,
        }
        if self.edit_type == "quant_rtn":
            payload["bits"] = (
                int(self.param1) if _safe_int(self.param1) is not None else 0
            )
            payload["group_size"] = (
                int(self.param2) if _safe_int(self.param2) is not None else 0
            )
        elif self.edit_type == "fp8_quant":
            payload["format"] = self.param1
        elif self.edit_type == "magnitude_prune":
            payload["ratio"] = (
                float(self.param1) if _safe_float(self.param1) is not None else 0.0
            )
        elif self.edit_type == "lowrank_svd":
            payload["rank"] = (
                int(self.param1) if _safe_int(self.param1) is not None else 0
            )
        return payload


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except _COERCE_ERRORS:
        return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except _COERCE_ERRORS:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except _FILE_READ_ERRORS:
        return None
    return payload if isinstance(payload, dict) else None


def _model_id_for(model_output_dir: Path) -> str:
    model_id_path = model_output_dir / ".model_id"
    if not model_id_path.exists():
        return ""
    try:
        return model_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_tuned_entry(
    *,
    tuned_path: str,
    model_key: str,
    model_id: str,
    model_output_dir_name: str,
    edit_type: str,
) -> tuple[dict[str, Any], str, str]:
    if not tuned_path:
        return {}, "missing", "missing_tuned_edit_params_file"

    payload = _load_json_object(Path(tuned_path))
    if payload is None:
        if Path(tuned_path).exists():
            return {}, "invalid", "invalid_tuned_edit_params_file"
        return {}, "missing", "missing_tuned_edit_params_file"

    entry_map: dict[str, Any] = {}
    models = payload.get("models")
    if isinstance(models, dict):
        entry_map = (
            models.get(model_key)
            or models.get(model_id)
            or models.get(model_output_dir_name)
            or {}
        )

    if not entry_map and isinstance(payload.get(edit_type), dict):
        entry_map = payload

    defaults = payload.get("defaults")
    entry = (
        (entry_map.get(edit_type) if isinstance(entry_map, dict) else None)
        or (defaults.get(edit_type) if isinstance(defaults, dict) else None)
        or {}
    )
    if not isinstance(entry, dict):
        entry = {}

    return entry, str(entry.get("status") or "missing"), str(entry.get("reason") or "")


def _normalize_non_quant_scope(
    edit_type: str,
    param1: str,
    param2: str,
    scope: str,
) -> tuple[str, str]:
    if edit_type != "quant_rtn" and not scope:
        scope = param2
        param2 = ""
    return param2, scope


def _normalize_quant_scope(param1: str, param2: str, scope: str) -> tuple[str, str]:
    if not scope and param1 and param2:
        scope = param2
        param2 = ""
    return param2, scope


def _default_edit_dir_name(
    *,
    edit_type: str,
    param1: str,
    param2: str,
    version: str,
) -> str:
    if not version:
        return ""
    if edit_type == "quant_rtn":
        return f"quant_{param1}bit_{version}"
    if edit_type == "fp8_quant":
        return f"fp8_{param1}_{version}"
    if edit_type == "magnitude_prune":
        try:
            pct = int(float(param1) * 100)
        except _COERCE_ERRORS:
            pct = 0
        return f"prune_{pct}pct_{version}"
    if edit_type == "lowrank_svd":
        return f"svd_rank{param1}_{version}"
    return f"{edit_type}_{version}"


def resolve_edit_spec(
    *,
    model_output_dir: Path,
    edit_spec: str,
    version_hint: str = "",
    tuned_path: str | None = None,
) -> ResolvedEditSpec:
    parts = edit_spec.split(":") if edit_spec else []
    edit_type = parts[0] if parts else ""
    param1 = parts[1] if len(parts) > 1 else ""
    param2 = parts[2] if len(parts) > 2 else ""
    scope = parts[3] if len(parts) > 3 else ""

    param2, scope = _normalize_non_quant_scope(edit_type, param1, param2, scope)
    if edit_type == "quant_rtn":
        param2, scope = _normalize_quant_scope(param1, param2, scope)

    clean_spec = param1 == "clean"
    status = "selected"
    reason = ""
    edit_dir_name = ""

    if clean_spec:
        resolved_tuned_path = (
            tuned_path or os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or ""
        ).strip()
        model_id = _model_id_for(model_output_dir)
        model_key = model_id or model_output_dir.name
        entry, status, reason = _load_tuned_entry(
            tuned_path=resolved_tuned_path,
            model_key=model_key,
            model_id=model_id,
            model_output_dir_name=model_output_dir.name,
            edit_type=edit_type,
        )
        if status == "selected":
            if edit_type == "quant_rtn":
                param1 = str(entry.get("bits", ""))
                param2 = str(entry.get("group_size", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "fp8_quant":
                param1 = str(entry.get("format", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "magnitude_prune":
                param1 = str(entry.get("sparsity", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "lowrank_svd":
                param1 = str(entry.get("rank", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            edit_dir_name = str(entry.get("edit_dir_name") or "")
    else:
        if edit_type == "quant_rtn":
            if _safe_int(param1) is None or _safe_int(param2) is None:
                status = "invalid"
                reason = "invalid_quant_params"
        elif edit_type == "magnitude_prune":
            if _safe_float(param1) is None:
                status = "invalid"
                reason = "invalid_prune_sparsity"
        elif edit_type == "lowrank_svd":
            if _safe_int(param1) is None:
                status = "invalid"
                reason = "invalid_lowrank_rank"
        elif edit_type == "fp8_quant":
            if not param1:
                status = "invalid"
                reason = "invalid_fp_format"

    version = version_hint or ("clean" if clean_spec else "")
    if status == "selected" and not edit_dir_name:
        edit_dir_name = _default_edit_dir_name(
            edit_type=edit_type,
            param1=param1,
            param2=param2,
            version=version,
        )

    return ResolvedEditSpec(
        status=status,
        reason=reason,
        edit_type=edit_type,
        param1=param1,
        param2=param2,
        scope=scope,
        version=version,
        edit_dir_name=edit_dir_name,
    )


def parse_edit_specs_json(raw_payload: str) -> list[object]:
    try:
        edit_specs = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid edit_specs JSON: {exc}") from exc

    if not isinstance(edit_specs, list):
        raise ValueError("edit_specs_json must be a JSON list")
    return edit_specs


def resolve_batch_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
    tuned_path: str | None = None,
) -> ResolvedEditSpec | None:
    if not isinstance(spec_entry, dict):
        return None
    spec_str = str(spec_entry.get("spec", ""))
    version = str(spec_entry.get("version", "clean"))
    return resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec=spec_str,
        version_hint=version,
        tuned_path=tuned_path,
    )


__all__ = [
    "ResolvedEditSpec",
    "parse_edit_specs_json",
    "resolve_batch_entry",
    "resolve_edit_spec",
]
