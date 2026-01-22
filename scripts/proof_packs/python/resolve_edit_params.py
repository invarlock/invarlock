from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _load_tuned_entry(
    tuned_path: str,
    model_key: str,
    model_id: str,
    model_output_dir_name: str,
    edit_type: str,
) -> tuple[dict[str, Any], str, str]:
    if not tuned_path:
        return {}, "missing", "missing_tuned_edit_params_file"

    path = Path(tuned_path)
    if not path.exists():
        return {}, "missing", "missing_tuned_edit_params_file"

    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}, "invalid", "invalid_tuned_edit_params_file"

    if not isinstance(data, dict):
        return {}, "invalid", "invalid_tuned_edit_params_file"

    entry_map: dict[str, Any] = {}
    models = data.get("models")
    if isinstance(models, dict):
        entry_map = (
            models.get(model_key)
            or models.get(model_id)
            or models.get(model_output_dir_name)
            or {}
        )

    if not entry_map and isinstance(data.get(edit_type), dict):
        entry_map = data

    defaults = data.get("defaults")
    entry = (
        (entry_map.get(edit_type) if isinstance(entry_map, dict) else None)
        or (defaults.get(edit_type) if isinstance(defaults, dict) else None)
        or {}
    )
    if not isinstance(entry, dict):
        entry = {}

    status = str(entry.get("status") or "missing")
    reason = str(entry.get("reason") or "")
    return entry, status, reason


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: resolve_edit_params.py <model_output_dir> <edit_spec> [version_hint]",
            file=sys.stderr,
        )
        return 2

    model_output_dir = Path(argv[0])
    edit_spec = argv[1] if len(argv) > 1 else ""
    version_hint = argv[2] if len(argv) > 2 else ""

    parts = edit_spec.split(":") if edit_spec else []
    edit_type = parts[0] if parts else ""
    param1 = parts[1] if len(parts) > 1 else ""
    param2 = parts[2] if len(parts) > 2 else ""
    scope = parts[3] if len(parts) > 3 else ""

    if edit_type != "quant_rtn" and not scope:
        scope = param2
        param2 = ""

    if edit_type == "quant_rtn" and not scope:
        if param1 and param2:
            scope = param2
            param2 = ""

    clean_spec = param1 == "clean"
    status = "selected"
    reason = ""
    edit_dir_name = ""

    if clean_spec:
        tuned_path = (os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or "").strip()
        model_id_path = model_output_dir / ".model_id"
        model_id = ""
        if model_id_path.exists():
            try:
                model_id = model_id_path.read_text().strip()
            except Exception:
                model_id = ""
        model_key = model_id or model_output_dir.name

        entry, status, reason = _load_tuned_entry(
            tuned_path=tuned_path,
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
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "magnitude_prune":
                param1 = str(entry.get("sparsity", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "lowrank_svd":
                param1 = str(entry.get("rank", ""))
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
        if edit_type == "quant_rtn":
            edit_dir_name = f"quant_{param1}bit_{version}" if version else ""
        elif edit_type == "fp8_quant":
            edit_dir_name = f"fp8_{param1}_{version}" if version else ""
        elif edit_type == "magnitude_prune":
            try:
                pct = int(float(param1) * 100)
            except Exception:
                pct = 0
            edit_dir_name = f"prune_{pct}pct_{version}" if version else ""
        elif edit_type == "lowrank_svd":
            edit_dir_name = f"svd_rank{param1}_{version}" if version else ""
        else:
            edit_dir_name = f"{edit_type}_{version}" if version else ""

    payload = {
        "status": status,
        "reason": reason,
        "edit_type": edit_type,
        "param1": param1,
        "param2": param2,
        "scope": scope,
        "version": version,
        "edit_dir_name": edit_dir_name,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
