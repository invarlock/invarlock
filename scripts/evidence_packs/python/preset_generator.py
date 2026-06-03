from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from runtime_tools import env_truthy
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_tools import env_truthy

try:
    import yaml

    _YAML_AVAILABLE = True
    _YAML_LOAD_ERRORS = (OSError, yaml.YAMLError)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_AVAILABLE = False
    _YAML_LOAD_ERRORS = (OSError,)


def _yaml_safe_load(payload: str) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is unavailable")
    loader = yaml.safe_load
    return loader(payload)


def _yaml_safe_dump(payload: Any, *, sort_keys: bool) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is unavailable")
    dumper = yaml.safe_dump
    return dumper(payload, sort_keys=sort_keys)


try:
    from .preset_calibration import (
        _apply_spectral_max_caps,
        _load_guard_order_and_assurance,
        calibrate_drift,
        calibrate_rmt,
        calibrate_spectral,
        calibrate_variance,
        get_default_guards_order,
        get_spectral_max_caps,
        load_records,
    )
except ImportError:  # pragma: no cover - direct script execution
    from preset_calibration import (
        _apply_spectral_max_caps,
        _load_guard_order_and_assurance,
        calibrate_drift,
        calibrate_rmt,
        calibrate_spectral,
        calibrate_variance,
        get_default_guards_order,
        get_spectral_max_caps,
        load_records,
    )

__all__ = [
    "calibrate_drift",
    "calibrate_rmt",
    "calibrate_spectral",
    "calibrate_variance",
    "generate_preset",
    "get_default_guards_order",
    "get_spectral_max_caps",
    "load_records",
]


def _resolve_dataset_provider_spec(
    kind: str,
) -> str | dict[str, Any]:
    """Resolve dataset.provider to either a string or a mapping.

    Evidence packs historically used a string provider name (e.g. "wikitext2").
    For providers that require extra parameters (hf_text/local_jsonl), we emit a
    mapping under dataset.provider so `invarlock run/evaluate` can pass those
    kwargs to the provider constructor.
    """
    kind_norm = str(kind or "").strip()
    if not kind_norm:
        kind_norm = "wikitext2"

    raw_yaml = os.environ.get("INVARLOCK_DATASET_PROVIDER_YAML")
    if raw_yaml:
        if not _YAML_AVAILABLE:
            raise SystemExit(
                "INVARLOCK_DATASET_PROVIDER_YAML is set but PyYAML is unavailable"
            )
        parsed = _yaml_safe_load(raw_yaml)
        if not isinstance(parsed, dict):
            raise SystemExit("INVARLOCK_DATASET_PROVIDER_YAML must parse to a mapping")
        provider = dict(parsed)
        provider.setdefault("kind", kind_norm)
        return provider

    raw_json = os.environ.get("INVARLOCK_DATASET_PROVIDER_JSON")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"INVARLOCK_DATASET_PROVIDER_JSON is not valid JSON ({exc})"
            ) from exc
        if not isinstance(parsed, dict):
            raise SystemExit("INVARLOCK_DATASET_PROVIDER_JSON must be a JSON object")
        provider = dict(parsed)
        provider.setdefault("kind", kind_norm)
        return provider

    if kind_norm == "hf_text":
        dataset_name = os.environ.get("INVARLOCK_HF_DATASET_NAME") or os.environ.get(
            "INVARLOCK_HF_DATASET"
        )
        if not dataset_name:
            dataset_name = "allenai/c4"
        # Migrate legacy "c4" to "allenai/c4" (script-based c4 deprecated in datasets 4.x)
        if str(dataset_name) == "c4":
            dataset_name = "allenai/c4"
        config_name = os.environ.get("INVARLOCK_HF_CONFIG_NAME") or os.environ.get(
            "INVARLOCK_HF_DATASET_CONFIG_NAME"
        )
        if not config_name and str(dataset_name) == "allenai/c4":
            config_name = "en"
        text_field = os.environ.get("INVARLOCK_HF_TEXT_FIELD") or "text"
        try:
            max_samples = int(os.environ.get("INVARLOCK_HF_MAX_SAMPLES") or "2000")
        except (TypeError, ValueError):
            max_samples = 2000
        cache_dir = os.environ.get("INVARLOCK_HF_CACHE_DIR") or os.environ.get(
            "HF_DATASETS_CACHE"
        )
        # trust_remote_code only if explicitly set (not needed for allenai/c4 Parquet)
        trust_raw = os.environ.get("INVARLOCK_HF_TRUST_REMOTE_CODE")
        trust_remote_code: bool | None = None
        if trust_raw is not None:
            norm = trust_raw.strip().lower()
            if norm in {"1", "true", "yes", "y", "on"}:
                if not env_truthy("INVARLOCK_ALLOW_REMOTE_CODE"):
                    raise ValueError(
                        "INVARLOCK_HF_TRUST_REMOTE_CODE=true requires "
                        "INVARLOCK_ALLOW_REMOTE_CODE=1."
                    )
                trust_remote_code = True
            elif norm in {"0", "false", "no", "n", "off"}:
                trust_remote_code = False
        provider: dict[str, Any] = {
            "kind": "hf_text",
            "dataset_name": str(dataset_name),
            "text_field": str(text_field),
            "max_samples": int(max_samples),
        }
        if config_name:
            provider["config_name"] = str(config_name)
        if trust_remote_code is not None:
            provider["trust_remote_code"] = bool(trust_remote_code)
        if cache_dir:
            provider["cache_dir"] = str(cache_dir)
        return provider

    if kind_norm == "local_jsonl":
        file = os.environ.get("INVARLOCK_LOCAL_JSONL_FILE")
        path = os.environ.get("INVARLOCK_LOCAL_JSONL_PATH")
        data_files = os.environ.get("INVARLOCK_LOCAL_JSONL_DATA_FILES")
        text_field = os.environ.get("INVARLOCK_LOCAL_JSONL_TEXT_FIELD") or "text"
        try:
            max_samples = int(
                os.environ.get("INVARLOCK_LOCAL_JSONL_MAX_SAMPLES") or "2000"
            )
        except (TypeError, ValueError):
            max_samples = 2000
        provider = {
            "kind": "local_jsonl",
            "text_field": str(text_field),
            "max_samples": int(max_samples),
        }
        if file:
            provider["file"] = str(file)
        elif path:
            provider["path"] = str(path)
        elif data_files:
            provider["data_files"] = str(data_files)
        return provider

    return kind_norm


def generate_preset(
    *,
    cal_dir: Path,
    preset_file: Path,
    model_name: str,
    model_path: str,
    tier: str,
    dataset_provider: str | dict[str, Any],
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    edit_types: list[str],
) -> tuple[Path, Path, list[Path]]:
    guards_order, assurance_cfg = _load_guard_order_and_assurance(cal_dir)
    enabled_guards = set(guards_order)

    records = load_records(cal_dir=cal_dir)
    if not records:
        raise SystemExit(
            "ERROR: No calibration records found; cannot create valid preset"
        )

    drift_stats = calibrate_drift(records)
    spectral_summary, spectral_caps = calibrate_spectral(records, tier=tier)
    rmt_summary, rmt_epsilon = calibrate_rmt(records, tier=tier)
    variance_config = calibrate_variance(records)

    drift_band_cfg: dict[str, float] | None = None
    band = drift_stats.get("suggested_band")
    if isinstance(band, list | tuple) and len(band) == 2:
        try:
            lo = float(band[0])
            hi = float(band[1])
        except (TypeError, ValueError, OverflowError):
            lo, hi = float("nan"), float("nan")
        if math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi:
            drift_band_cfg = {"min": lo, "max": hi}

    preset: dict[str, Any] = {
        "_calibration_meta": {
            "model_name": model_name,
            "num_runs": len(records),
            "tier": tier,
            "drift_mean": drift_stats.get("mean"),
            "drift_std": drift_stats.get("std"),
            "drift_band_compatible": drift_stats.get("band_compatible"),
            "suggested_drift_band": drift_stats.get("suggested_band"),
        },
        "model": {"id": model_path},
        "dataset": {
            "provider": dataset_provider,
            "split": "validation",
            "seq_len": int(seq_len),
            "stride": int(stride),
            "preview_n": int(preview_n),
            "final_n": int(final_n),
            "seed": 42,
        },
        "guards": {"order": guards_order},
    }
    if drift_band_cfg is not None:
        preset["primary_metric"] = {"drift_band": drift_band_cfg}

    if isinstance(assurance_cfg, dict) and assurance_cfg:
        preset["assurance"] = assurance_cfg

    spectral: dict[str, Any] = {}
    if spectral_caps:
        spectral["family_caps"] = spectral_caps
    if spectral_summary.get("sigma_quantile") is not None:
        spectral["sigma_quantile"] = spectral_summary["sigma_quantile"]
    if spectral_summary.get("deadband") is not None:
        spectral["deadband"] = spectral_summary["deadband"]
    if spectral_summary.get("max_caps") is not None:
        spectral["max_caps"] = spectral_summary["max_caps"]
    if "spectral" in enabled_guards and spectral:
        preset["guards"]["spectral"] = spectral

    rmt: dict[str, Any] = {}
    if rmt_epsilon:
        rmt["epsilon_by_family"] = rmt_epsilon
    if rmt_summary.get("margin") is not None:
        rmt["margin"] = rmt_summary["margin"]
    if rmt_summary.get("deadband") is not None:
        rmt["deadband"] = rmt_summary["deadband"]
    if "rmt" in enabled_guards and rmt:
        preset["guards"]["rmt"] = rmt

    if "variance" in enabled_guards and variance_config:
        preset["guards"]["variance"] = variance_config

    stats_path = cal_dir / "calibration_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "guards_order": guards_order,
                "assurance": assurance_cfg,
                "drift": drift_stats,
                "spectral": {**spectral_summary, "family_caps": spectral_caps},
                "rmt": {**rmt_summary, "epsilon_by_family": rmt_epsilon},
                "variance": variance_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    preset_file.parent.mkdir(parents=True, exist_ok=True)
    if _YAML_AVAILABLE and preset_file.suffix.lower() in {".yaml", ".yml"}:
        preset_file.write_text(_yaml_safe_dump(preset, sort_keys=False))
    else:
        preset_file = preset_file.with_suffix(".json")
        preset_file.write_text(json.dumps(preset, indent=2) + "\n")

    derived_files: list[Path] = []
    for edit_type in edit_types:
        derived = copy.deepcopy(preset)
        meta = derived.get("_calibration_meta")
        if isinstance(meta, dict):
            meta["edit_type"] = edit_type
        _apply_spectral_max_caps(derived, edit_type=edit_type, tier=tier)
        out = preset_file.with_name(
            f"{preset_file.stem}__{edit_type}{preset_file.suffix}"
        )
        if _YAML_AVAILABLE and out.suffix.lower() in {".yaml", ".yml"}:
            out.write_text(_yaml_safe_dump(derived, sort_keys=False))
        else:
            out = out.with_suffix(".json")
            out.write_text(json.dumps(derived, indent=2) + "\n")
        derived_files.append(out)

    return preset_file, stats_path, derived_files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibrated evidence-pack presets"
    )
    parser.add_argument("--cal-dir", required=True)
    parser.add_argument("--preset-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tier", default=os.environ.get("INVARLOCK_TIER", "balanced"))
    parser.add_argument(
        "--dataset-provider", default=os.environ.get("INVARLOCK_DATASET", "wikitext2")
    )
    parser.add_argument(
        "--seq-len", type=int, default=int(os.environ.get("PRESET_SEQ_LEN", "1024"))
    )
    parser.add_argument(
        "--stride", type=int, default=int(os.environ.get("PRESET_STRIDE", "512"))
    )
    parser.add_argument(
        "--preview-n", type=int, default=int(os.environ.get("PRESET_PREVIEW_N", "40"))
    )
    parser.add_argument(
        "--final-n", type=int, default=int(os.environ.get("PRESET_FINAL_N", "40"))
    )
    parser.add_argument(
        "--edit-types",
        default=os.environ.get("PACK_PRESET_EDIT_TYPES", ""),
        help="Comma-separated edit types for derived presets (default: core types).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    edit_types = [e.strip() for e in str(args.edit_types).split(",") if e.strip()]
    if not edit_types:
        edit_types = ["quant_rtn", "fp8_quant", "magnitude_prune", "lowrank_svd"]

    dataset_provider = _resolve_dataset_provider_spec(str(args.dataset_provider))

    preset_file, stats_path, derived_files = generate_preset(
        cal_dir=Path(args.cal_dir),
        preset_file=Path(args.preset_file),
        model_name=str(args.model_name),
        model_path=str(args.model_path),
        tier=str(args.tier).strip().lower(),
        dataset_provider=dataset_provider,
        seq_len=int(args.seq_len),
        stride=int(args.stride),
        preview_n=int(args.preview_n),
        final_n=int(args.final_n),
        edit_types=edit_types,
    )
    print(f"Saved preset to {preset_file}")
    print(f"Saved stats to {stats_path}")
    for path in derived_files:
        print(f"Saved derived preset to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
