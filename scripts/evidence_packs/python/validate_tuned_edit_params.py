from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a tuned_edit_params.json file for required models/edit types."
    )
    parser.add_argument("--file", required=True, help="Path to tuned_edit_params.json")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model IDs (Hugging Face IDs).",
    )
    parser.add_argument(
        "--model-names",
        required=True,
        help="Comma-separated sanitized model names (evidence-pack directory names).",
    )
    parser.add_argument(
        "--edit-types",
        required=True,
        help="Comma-separated edit type keys (e.g., quant_rtn, fp8_quant).",
    )
    parser.add_argument(
        "--canonical-file",
        help="Optional canonical tuned_edit_params.json to compare against.",
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Allow requested entries to differ from the canonical file.",
    )
    return parser.parse_args(argv)


def _split_csv(value: str) -> list[str]:
    return [item for item in (value or "").split(",") if item]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_entry(
    *,
    data: dict,
    defaults: dict,
    models_map: dict,
    model_id: str,
    model_name: str,
    edit_type: str,
) -> dict:
    entry_map: dict = {}
    if isinstance(models_map, dict):
        entry_map = models_map.get(model_id) or models_map.get(model_name) or {}
    if not entry_map and isinstance(data.get(edit_type), dict):
        entry_map = data
    entry = entry_map.get(edit_type) or defaults.get(edit_type) or {}
    return entry if isinstance(entry, dict) else {}


def _describe_entry_diff(entry: dict, canonical_entry: dict) -> str:
    diff_keys = sorted(set(entry) | set(canonical_entry))
    parts: list[str] = []
    for key in diff_keys:
        if entry.get(key) == canonical_entry.get(key):
            continue
        parts.append(
            f"{key}={json.dumps(entry.get(key), sort_keys=True)}"
            f"!=canonical={json.dumps(canonical_entry.get(key), sort_keys=True)}"
        )
    return "; ".join(parts) if parts else "entry!=canonical"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Tuned edit preset file not found: {path}")

    models = _split_csv(str(args.models))
    model_names = _split_csv(str(args.model_names))
    required = sorted(set(_split_csv(str(args.edit_types))))

    data = _load_json(path)
    if not isinstance(data, dict):
        raise SystemExit("Invalid tuned edit preset file (expected JSON object).")

    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    models_map = data.get("models") if isinstance(data.get("models"), dict) else {}
    canonical: dict = {}
    canonical_defaults: dict = {}
    canonical_models_map: dict = {}
    if args.canonical_file:
        canonical_path = Path(args.canonical_file)
        if not canonical_path.is_file():
            raise SystemExit(
                f"Canonical tuned edit preset file not found: {canonical_path}"
            )
        canonical = _load_json(canonical_path)
        if not isinstance(canonical, dict):
            raise SystemExit(
                "Invalid canonical tuned edit preset file (expected JSON object)."
            )
        canonical_defaults = (
            canonical.get("defaults")
            if isinstance(canonical.get("defaults"), dict)
            else {}
        )
        canonical_models_map = (
            canonical.get("models") if isinstance(canonical.get("models"), dict) else {}
        )

    missing: list[str] = []
    noncanonical: list[str] = []
    for idx, model_id in enumerate(models):
        model_name = model_names[idx] if idx < len(model_names) else ""
        for edit_type in required:
            entry = _get_entry(
                data=data,
                defaults=defaults,
                models_map=models_map,
                model_id=model_id,
                model_name=model_name,
                edit_type=edit_type,
            )
            status = str(entry.get("status") or "missing")
            if status != "selected":
                missing.append(f"{model_id}:{edit_type}:{status}")
                continue
            if args.allow_noncanonical or not args.canonical_file:
                continue
            canonical_entry = _get_entry(
                data=canonical if isinstance(canonical, dict) else {},
                defaults=canonical_defaults,
                models_map=canonical_models_map,
                model_id=model_id,
                model_name=model_name,
                edit_type=edit_type,
            )
            canonical_status = str(canonical_entry.get("status") or "missing")
            if canonical_status != "selected":
                continue
            if entry != canonical_entry:
                diff = _describe_entry_diff(entry, canonical_entry)
                noncanonical.append(f"{model_id}:{edit_type}:{diff}")

    if missing:
        msg = "Missing tuned edit presets: " + ", ".join(missing)
        raise SystemExit(msg)
    if noncanonical:
        msg = "Noncanonical tuned edit presets: " + ", ".join(noncanonical)
        raise SystemExit(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
