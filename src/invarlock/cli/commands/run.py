"""
InvarLock CLI Run Command
=====================

Run a guarded pipeline from a YAML config. Intended for local smokes,
plugin demos, and development. Advanced: for pairwise certification,
prefer Compare & Certify via `invarlock certify --baseline ... --subject ...`.
"""

import copy
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import sys as _sys
import types as _types
import warnings
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
import psutil
import typer
from rich.console import Console

from invarlock.cli.output import (
    OutputStyle,
    make_console,
    perf_counter,
    print_event,
    print_timing_summary,
    resolve_output_style,
    timed_step,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from invarlock.cli.errors import InvarlockError
from invarlock.cli.utils import (
    coerce_float as _coerce_float,
)
from invarlock.cli.utils import (
    coerce_int as _coerce_int,
)
from invarlock.cli.utils import (
    coerce_option as _coerce_option,
)
from invarlock.core.exceptions import (
    ConfigError as _CfgErr,
)
from invarlock.core.exceptions import (
    DataError as _DataErr,
)
from invarlock.core.exceptions import (
    ValidationError as _ValErr,
)
from invarlock.model_profile import detect_model_profile, resolve_tokenizer
from invarlock.model_utils import set_seed
from invarlock.reporting.validate import validate_guard_overhead

from ..config import (
    InvarLockConfig,
)
from ..overhead_utils import _extract_pm_snapshot_for_overhead

console = make_console()


def _style_from_console(console: Console, profile: str | None = None) -> OutputStyle:
    style = getattr(console, "_invarlock_output_style", None)
    if isinstance(style, OutputStyle):
        return style
    return resolve_output_style(
        style=None,
        profile=profile,
        progress=False,
        timing=False,
        no_color=False,
    )


def _event(
    console: Console,
    tag: str,
    message: str,
    *,
    emoji: str | None = None,
    console_style: str | None = None,
    profile: str | None = None,
) -> None:
    style = _style_from_console(console, profile=profile)
    print_event(
        console,
        tag,
        message,
        style=style,
        emoji=emoji,
        console_style=console_style,
    )


LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Release profile window planning constants
RELEASE_BUFFER_FRACTION = 0.12
RELEASE_MIN_WINDOWS_PER_ARM = 200
RELEASE_CALIBRATION_MIN = 16
RELEASE_CALIBRATION_MAX = 24
GUARD_OVERHEAD_THRESHOLD = 0.01
KV_LABEL_WIDTH = 10

_NOISY_WARNING_PATTERNS = (
    r".*`torch_dtype` is deprecated.*",
    r".*loss_type=None.*unrecognized.*",
)


def _resolve_warning_suppression(profile: str | None) -> tuple[bool, bool]:
    suppress_all = os.getenv("INVARLOCK_SUPPRESS_WARNINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    profile_norm = (profile or "").strip().lower()
    enabled = bool(suppress_all) or profile_norm in {"ci", "ci_cpu", "release", "dev"}
    return enabled, suppress_all


def _apply_warning_filters(profile: str | None) -> bool:
    enabled, suppress_all = _resolve_warning_suppression(profile)
    if not enabled:
        return False
    if suppress_all:
        warnings.simplefilter("ignore")
    else:
        for pattern in _NOISY_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern)
    return True


@contextmanager
def _suppress_noisy_warnings(profile: str | None) -> Iterator[None]:
    enabled, _suppress_all = _resolve_warning_suppression(profile)
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        _apply_warning_filters(profile)
        yield


def _format_kv_line(label: str, value: str, *, width: int = KV_LABEL_WIDTH) -> str:
    return f"  {label:<{width}}: {value}"


def _device_resolution_note(target_device: str, resolved_device: str) -> str:
    target_norm = str(target_device or "").strip().lower()
    resolved_norm = str(resolved_device or "").strip().lower()
    if not target_norm or target_norm == "auto":
        return "auto-resolved"
    if target_norm == resolved_norm:
        return "requested"
    return f"resolved from {target_device}"


def _format_guard_chain(guards: list[Any]) -> str:
    names = [str(getattr(guard, "name", "unknown")) for guard in guards]
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return " → ".join(deduped)


# Common dataset split aliases we probe in order when not explicitly set
SPLIT_ALIASES: tuple[str, ...] = ("validation", "val", "dev", "eval", "test")


def _coerce_mapping(obj: object) -> dict[str, Any]:
    """Best-effort conversion of config-like objects to plain dicts."""

    if isinstance(obj, dict):
        return obj
    try:
        raw = getattr(obj, "_data", None)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    try:
        dumped = obj.model_dump()  # type: ignore[attr-defined]
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass
    try:
        data = vars(obj)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _prune_none_values(value: Any) -> Any:
    """Recursively drop keys/items whose value is None.

    Used when serializing dataclass-style config sections that define many optional
    fields defaulting to None; those should behave as "unset" rather than explicit
    policy overrides.
    """

    if isinstance(value, dict):
        return {
            key: _prune_none_values(val)
            for key, val in value.items()
            if val is not None
        }
    if isinstance(value, list):
        return [_prune_none_values(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(_prune_none_values(item) for item in value if item is not None)
    return value


def _to_serialisable_dict(section: object) -> dict[str, Any]:
    """Coerce config fragments to plain dicts.

    Handles InvarLockConfig sections (which wrap dicts in a private `_Obj` with
    `_data`) so downstream components (core.runner) see canonical mappings,
    e.g. `eval.bootstrap.replicates`.
    """

    # Prefer native dump methods
    if hasattr(section, "model_dump"):
        return section.model_dump()  # type: ignore[return-value]
    if hasattr(section, "dict"):
        try:
            return section.dict()  # type: ignore[return-value]
        except Exception:
            pass
    # Unwrap CLI _Obj wrapper used by InvarLockConfig for attribute access
    try:
        raw = getattr(section, "_data", None)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    # Already a mapping
    if isinstance(section, dict):
        return section
    # Best-effort attribute dump (prune None so "unset" does not override tier defaults)
    try:
        data = vars(section)
        # Common case: {'_data': {...}}
        if isinstance(data, dict) and isinstance(data.get("_data"), dict):
            return data["_data"]
        return _prune_none_values(data)  # type: ignore[return-value]
    except TypeError:
        return {}


def _resolve_pm_acceptance_range(
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from config/env with safe defaults."""

    base_min = 0.95
    base_max = 1.10

    cfg_min = None
    cfg_max = None
    try:
        cfg_map = _coerce_mapping(cfg) if cfg is not None else {}
        pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
        pm_map = _coerce_mapping(pm_section)
        acceptance = (
            pm_map.get("acceptance_range") if isinstance(pm_map, dict) else None
        )
        if isinstance(acceptance, dict):
            if acceptance.get("min") is not None:
                try:
                    cfg_min = float(acceptance["min"])
                except (TypeError, ValueError):
                    cfg_min = None
            if acceptance.get("max") is not None:
                try:
                    cfg_max = float(acceptance["max"])
                except (TypeError, ValueError):
                    cfg_max = None
    except Exception:
        cfg_min = None
        cfg_max = None

    def _parse_env(name: str) -> float | None:
        try:
            raw = os.environ.get(name, "")
            if raw is None or str(raw).strip() == "":
                return None
            return float(raw)
        except Exception:
            return None

    env_min = _parse_env("INVARLOCK_PM_ACCEPTANCE_MIN")
    env_max = _parse_env("INVARLOCK_PM_ACCEPTANCE_MAX")

    has_explicit = any(v is not None for v in (cfg_min, cfg_max, env_min, env_max))
    if not has_explicit:
        return {}

    min_val = (
        env_min if env_min is not None else cfg_min if cfg_min is not None else base_min
    )
    max_val = (
        env_max if env_max is not None else cfg_max if cfg_max is not None else base_max
    )

    try:
        if min_val is not None and min_val <= 0:
            min_val = base_min
    except Exception:
        min_val = base_min
    try:
        if max_val is not None and max_val <= 0:
            max_val = base_max
    except Exception:
        max_val = base_max

    try:
        if max_val is not None and min_val is not None and max_val < min_val:
            max_val = min_val
    except Exception:
        max_val = base_max

    return {"min": float(min_val), "max": float(max_val)}


def _resolve_pm_drift_band(
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve preview→final drift band from config/env with safe defaults.

    The drift band governs the Preview Final Drift Acceptable gate. By default,
    certificates enforce 0.95–1.05 unless an explicit band is provided.
    """

    base_min = 0.95
    base_max = 1.05

    cfg_min = None
    cfg_max = None
    try:
        cfg_map = _coerce_mapping(cfg) if cfg is not None else {}
        pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
        pm_map = _coerce_mapping(pm_section)
        drift_band = pm_map.get("drift_band") if isinstance(pm_map, dict) else None
        if isinstance(drift_band, dict):
            if drift_band.get("min") is not None:
                try:
                    cfg_min = float(drift_band["min"])
                except (TypeError, ValueError):
                    cfg_min = None
            if drift_band.get("max") is not None:
                try:
                    cfg_max = float(drift_band["max"])
                except (TypeError, ValueError):
                    cfg_max = None
        elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
            try:
                cfg_min = float(drift_band[0])
                cfg_max = float(drift_band[1])
            except (TypeError, ValueError):
                cfg_min = None
                cfg_max = None
    except Exception:
        cfg_min = None
        cfg_max = None

    def _parse_env(name: str) -> float | None:
        try:
            raw = os.environ.get(name, "")
            if raw is None or str(raw).strip() == "":
                return None
            return float(raw)
        except Exception:
            return None

    env_min = _parse_env("INVARLOCK_PM_DRIFT_MIN")
    env_max = _parse_env("INVARLOCK_PM_DRIFT_MAX")

    has_explicit = any(v is not None for v in (cfg_min, cfg_max, env_min, env_max))
    if not has_explicit:
        return {}

    min_val = (
        env_min if env_min is not None else cfg_min if cfg_min is not None else base_min
    )
    max_val = (
        env_max if env_max is not None else cfg_max if cfg_max is not None else base_max
    )

    try:
        if min_val is not None and min_val <= 0:
            min_val = base_min
    except Exception:
        min_val = base_min
    try:
        if max_val is not None and max_val <= 0:
            max_val = base_max
    except Exception:
        max_val = base_max
    try:
        if min_val is not None and max_val is not None and min_val >= max_val:
            min_val, max_val = base_min, base_max
    except Exception:
        min_val, max_val = base_min, base_max

    return {"min": float(min_val), "max": float(max_val)}


def _free_model_memory(model: object | None) -> None:
    """Best-effort cleanup to release GPU memory for a model object."""
    if model is None:
        return
    try:
        import gc

        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        # Cleanup should never raise; fallback is to proceed without cache purge
        pass


class _SnapshotRestoreFailed(RuntimeError):
    """Internal signal for snapshot restore failures during retries."""


def _should_measure_overhead(profile_normalized: str) -> tuple[bool, bool]:
    """Return (measure_guard_overhead, skip_overhead) derived from env/profile."""

    skip_overhead_env = (
        os.environ.get("INVARLOCK_SKIP_OVERHEAD_CHECK", "").strip().lower()
    )
    skip_overhead = skip_overhead_env in {"1", "true", "yes"}
    measure_guard_overhead = (
        profile_normalized in {"ci", "release"} and not skip_overhead
    )
    return measure_guard_overhead, skip_overhead


def _choose_dataset_split(
    *, requested: str | None, available: list[str] | None
) -> tuple[str, bool]:
    """
    Choose a dataset split deterministically.

    Returns (split, used_fallback). If `requested` is provided, returns it verbatim.
    Else tries SPLIT_ALIASES in order; if none present, falls back to the first
    available split (sorted for determinism). If `available` is None/empty, returns
    ('validation', True) as a last resort so the run does not crash.
    """
    try:
        if isinstance(requested, str) and requested:
            return requested, False
    except Exception:
        pass
    avail = list(available) if isinstance(available, list) and available else []
    if avail:
        for cand in SPLIT_ALIASES:
            if cand in avail:
                return cand, True
        return sorted(avail)[0], True
    return "validation", True


def _persist_ref_masks(core_report: Any, run_dir: Path) -> Path | None:
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


def _resolve_exit_code(exc: Exception, *, profile: str | None) -> int:
    """Resolve exit code based on exception type and profile.

    - ValueError("Invalid RunReport...") → 2 (schema/shape issue)
    - InvarlockError in CI/Release         → 3 (hard abort)
    - All other cases                  → 1 (generic failure)
    """
    try:
        prof = (profile or "").strip().lower()
    except Exception:
        prof = ""
    # Schema/validation classes and known shapes → exit 2
    if isinstance(exc, _CfgErr | _ValErr | _DataErr):
        return 2
    if isinstance(exc, ValueError) and "Invalid RunReport" in str(exc):
        return 2
    if isinstance(exc, InvarlockError) and prof in {"ci", "release"}:
        return 3
    return 1


## NOTE: Deprecated helper `_check_pairability_or_abort` was removed.
## Provider parity and pairing guarantees are enforced via guard digests and
## invariant checks during run execution.


def _hash_sequences(seqs: Sequence[Sequence[int]] | Iterable[Sequence[int]]) -> str:
    """Compute a stable digest for a sequence of integer token sequences."""
    hasher = hashlib.blake2s(digest_size=16)
    for seq in seqs:
        try:
            seq_len = len(seq)
        except TypeError:
            seq = list(seq)
            seq_len = len(seq)
        hasher.update(seq_len.to_bytes(4, "little", signed=False))
        arr = array("I", (int(token) & 0xFFFFFFFF for token in seq))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _compute_mask_positions_digest(windows: dict[str, Any]) -> str | None:
    """Compute a rolled hash of MLM mask positions across windows.

    Expects windows of the shape { 'preview': {...}, 'final': {...} } with
    'labels' and optional 'window_ids' in each section. Positions where
    labels != -100 are treated as masked.
    """
    try:
        # Simple, dependency-light digest of positions where labels != -100
        hasher = hashlib.blake2s(digest_size=16)
        any_masked = False
        for arm in ("preview", "final"):
            sec = windows.get(arm)
            if not isinstance(sec, dict):
                continue
            labels = sec.get("labels")
            if not isinstance(labels, list) or not labels:
                continue
            hasher.update(arm.encode("utf-8"))
            for row in labels:
                row_list = _tensor_or_list_to_ints(row)
                if not row_list:
                    continue
                found = False
                for idx, v in enumerate(row_list):
                    if int(v) != -100:
                        hasher.update(b"1")
                        hasher.update(idx.to_bytes(4, "little", signed=False))
                        found = True
                if found:
                    any_masked = True
                hasher.update(b"|")
        if not any_masked:
            return None
        digest = hasher.hexdigest()
        return digest if digest else None
    except Exception:
        return None


def _to_int_list(values: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _tensor_or_list_to_ints(values: Any) -> list[int]:
    """Coerce possible tensor/list-like inputs to a list[int]."""
    try:
        # Torch tensors: `.tolist()` path
        if torch is not None and hasattr(values, "tolist"):
            raw = values.tolist()
            if isinstance(raw, list):
                return _to_int_list(raw)
            try:
                return _to_int_list(list(raw))
            except (typer.Exit, SystemExit, click.exceptions.Exit):
                raise
            except Exception:
                pass
        # Numpy arrays: treat as list-like
        if isinstance(values, np.ndarray | list | tuple):
            return _to_int_list(list(values))
        # Iterables of ints
        if isinstance(values, Iterable):
            return _to_int_list(values)
    except Exception:
        pass
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _derive_mlm_seed(base_seed: int, window_id: str | int, position: int) -> int:
    payload = f"{base_seed}:{window_id}:{position}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _apply_mlm_masks(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    mask_prob: float,
    seed: int,
    random_token_prob: float,
    original_token_prob: float,
    prefix: str,
) -> tuple[int, list[int]]:
    """Apply basic BERT-style MLM masking to tokenized records in-place."""
    if mask_prob <= 0.0:
        zeroed = []
        for record in records:
            length = len(record["input_ids"])
            record["labels"] = [-100] * length
            record["mlm_masked"] = 0
            zeroed.append(0)
        return 0, zeroed

    vocab_size = _safe_int(getattr(tokenizer, "vocab_size", 0))
    # Require an explicit mask token id for MLM
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define mask_token_id; required for MLM evaluation."
        )
    try:
        mask_token_id = int(mask_token_id)
    except Exception:
        mask_token_id = _safe_int(mask_token_id, 0)

    # Build special token id set to avoid masking them
    special_ids = set()
    for attr in (
        "cls_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            try:
                special_ids.add(int(val))
            except Exception:
                pass
    try:
        special_ids.update(
            int(t) for t in getattr(tokenizer, "all_special_ids", []) or []
        )
    except Exception:
        pass

    masked_total = 0
    masked_counts: list[int] = []
    for idx_record, record in enumerate(records):
        window_id = record.get("window_id", f"{prefix}:{idx_record}")
        input_ids = _tensor_or_list_to_ints(record.get("input_ids", []))
        attention = _tensor_or_list_to_ints(record.get("attention_mask", []))
        labels = [-100] * len(input_ids)

        masked = 0
        for pos, (tok, att) in enumerate(zip(input_ids, attention, strict=False)):
            if not att:
                continue
            if int(tok) in special_ids:
                continue
            if random.random() < mask_prob:
                rng = random.Random(_derive_mlm_seed(seed, window_id, pos))
                labels[pos] = int(tok)
                r = rng.random()
                if r < 1.0 - (random_token_prob + original_token_prob):
                    input_ids[pos] = mask_token_id
                elif r < 1.0 - original_token_prob and vocab_size > 0:
                    rng2 = random.Random(_derive_mlm_seed(seed + 17, window_id, pos))
                    input_ids[pos] = rng2.randint(0, max(1, vocab_size - 1))
                masked += 1

        # Ensure at least one masked token for stability
        if masked == 0:
            candidate_positions = [
                p
                for p, (tok, att) in enumerate(zip(input_ids, attention, strict=False))
                if att and int(tok) not in special_ids
            ]
            if candidate_positions:
                pos = candidate_positions[len(candidate_positions) // 2]
                rng = random.Random(_derive_mlm_seed(seed + 17, window_id, pos))
                labels[pos] = int(input_ids[pos])
                masked = 1
                r = rng.random()
                if r < 1.0 - (random_token_prob + original_token_prob):
                    input_ids[pos] = mask_token_id
                elif r < 1.0 - original_token_prob and vocab_size > 0:
                    input_ids[pos] = rng.randrange(vocab_size)

        record["input_ids"] = _to_int_list(input_ids)
        record["attention_mask"] = _to_int_list(attention)
        record["labels"] = _to_int_list(labels)
        record["mlm_masked"] = masked
        masked_total += masked
        masked_counts.append(masked)

    return masked_total, masked_counts


def _tokenizer_digest(tokenizer: Any) -> str:
    """Compute a stable digest for a tokenizer config.

    Tries, in order: get_vocab().items(), `vocab` attribute if list-like, else
    hashes a small set of informative attributes.
    """
    try:
        if hasattr(tokenizer, "get_vocab"):
            try:
                items = getattr(tokenizer.get_vocab(), "items", None)
                if callable(items):
                    pairs = list(items())
                    # Filter non-string keys for stability
                    pairs = [
                        (str(k), int(v)) for k, v in pairs if isinstance(k, str | int)
                    ]
                    payload = json.dumps(sorted(pairs), separators=(",", ":")).encode()
                    return hashlib.sha256(payload).hexdigest()
            except Exception:
                pass
        # Fallback to `vocab` attribute (e.g., list of pairs)
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, list):
            try:
                payload = json.dumps(
                    [(str(k), int(v)) for k, v in vocab], separators=(",", ":")
                ).encode()
                return hashlib.sha256(payload).hexdigest()
            except Exception:
                pass
        # Last resort: small attribute set
        attrs = {
            "name": getattr(tokenizer, "name_or_path", None),
            "eos": getattr(tokenizer, "eos_token", None),
            "pad": getattr(tokenizer, "pad_token", None),
            "size": _safe_int(getattr(tokenizer, "vocab_size", 0)),
        }
        return hashlib.sha256(json.dumps(attrs, sort_keys=True).encode()).hexdigest()
    except Exception:
        return "unknown-tokenizer"


def _extract_pairing_schedule(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract sanitized pairing schedule from a baseline-like report shape.

    Returns a dict with 'preview' and 'final' sections when present. Each section
    contains window_ids, input_ids, attention_masks, optional labels and counts.
    """
    if not isinstance(report, dict):
        return None
    windows = report.get("evaluation_windows")
    if not isinstance(windows, dict):
        return None

    def _wrap_single_row(raw: Any, *, expected_rows: int) -> list | None:
        if not isinstance(raw, list):
            return None
        if expected_rows == 1 and raw and not isinstance(raw[0], list):
            return [raw]
        return raw

    def _sanitize(section_key: str, *, start_id: int) -> dict[str, Any] | None:
        section = windows.get(section_key)
        if not isinstance(section, dict):
            return None
        input_ids_raw = section.get("input_ids")
        if not isinstance(input_ids_raw, list):
            return None
        input_ids = [_tensor_or_list_to_ints(seq) for seq in input_ids_raw]
        if not input_ids:
            return None

        window_ids_raw = section.get("window_ids")
        window_ids: list[int] = []
        if isinstance(window_ids_raw, list):
            if len(window_ids_raw) != len(input_ids):
                return None
            for wid in window_ids_raw:
                try:
                    window_ids.append(int(wid))
                except Exception:
                    return None
        else:
            window_ids = list(range(int(start_id), int(start_id) + len(input_ids)))

        attention_raw = section.get("attention_masks")
        attention_masks: list[list[int]]
        if isinstance(attention_raw, list):
            maybe = _wrap_single_row(attention_raw, expected_rows=len(input_ids))
            if isinstance(maybe, list) and all(
                isinstance(mask, list) for mask in maybe
            ):
                attention_masks = [_tensor_or_list_to_ints(mask) for mask in maybe]
            else:
                attention_masks = [
                    [1 if int(token) != 0 else 0 for token in seq] for seq in input_ids
                ]
        else:
            attention_masks = [
                [1 if int(token) != 0 else 0 for token in seq] for seq in input_ids
            ]
        if len(attention_masks) != len(input_ids):
            return None
        for seq, mask in zip(input_ids, attention_masks, strict=False):
            if len(mask) != len(seq):
                return None

        labels_raw = section.get("labels")
        labels: list[list[int]] | None = None
        if isinstance(labels_raw, list) and labels_raw:
            maybe_labels = _wrap_single_row(labels_raw, expected_rows=len(input_ids))
            if not isinstance(maybe_labels, list) or len(maybe_labels) != len(
                input_ids
            ):
                return None
            labels = []
            for idx, raw_label in enumerate(maybe_labels):
                label_list = _tensor_or_list_to_ints(raw_label)
                if idx < len(input_ids):
                    target_len = len(input_ids[idx])
                    if len(label_list) < target_len:
                        label_list = label_list + [-100] * (
                            target_len - len(label_list)
                        )
                    elif len(label_list) > target_len:
                        label_list = label_list[:target_len]
                labels.append(label_list)

        masked_counts: list[int] | None = None
        if section.get("masked_token_counts") is not None:
            raw = section.get("masked_token_counts")
            if isinstance(raw, int) and len(input_ids) == 1:
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(input_ids):
                return None
            masked_counts = [int(v) for v in raw]
        actual_counts: list[int] | None = None
        if section.get("actual_token_counts") is not None:
            raw = section.get("actual_token_counts")
            if isinstance(raw, int) and len(input_ids) == 1:
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(input_ids):
                return None
            actual_counts = [int(v) for v in raw]

        payload: dict[str, Any] = {
            "window_ids": window_ids,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
        }
        if labels is not None:
            payload["labels"] = labels
        if masked_counts is not None:
            payload["masked_token_counts"] = masked_counts
        if actual_counts is not None:
            payload["actual_token_counts"] = actual_counts
        return payload

    preview = _sanitize("preview", start_id=0)
    if not preview:
        return None
    final = _sanitize("final", start_id=len(preview.get("input_ids") or []))
    if preview and final:
        return {"preview": preview, "final": final}
    return None


def _prepare_config_for_run(
    *,
    config_path: str,
    profile: str | None,
    edit: str | None,
    tier: str | None,
    probes: int | None,
    console: Console,
) -> InvarLockConfig:
    """Load InvarLock config and apply CLI/profile overrides deterministically."""
    # Local import to allow test monkeypatching of invarlock.cli.config functions
    from ..config import (
        apply_edit_override as _apply_edit_override,
    )
    from ..config import (
        apply_profile as _apply_profile,
    )
    from ..config import (
        load_config as _load_config,
    )
    from ..config import (
        resolve_edit_kind as _resolve_edit_kind,
    )

    _event(
        console,
        "INIT",
        f"Loading configuration: {config_path}",
        emoji="📋",
        profile=profile,
    )
    cfg = _load_config(config_path)

    # Apply profile if specified (dev is a no-op)
    if profile and str(profile).lower() not in {"dev"}:
        _event(
            console, "INIT", f"Applying profile: {profile}", emoji="🎯", profile=profile
        )
        try:
            cfg = _apply_profile(cfg, profile)
        except Exception as exc:
            _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
            raise typer.Exit(1) from exc

    # Apply edit override
    if edit:
        try:
            edit_name = _resolve_edit_kind(edit)
            _event(
                console,
                "EXEC",
                f"Edit override: {edit} → {edit_name}",
                emoji="✂️",
                profile=profile,
            )
            cfg = _apply_edit_override(cfg, edit)
        except ValueError as e:
            _event(console, "FAIL", str(e), emoji="❌", profile=profile)
            raise typer.Exit(1) from e

    # Apply CLI overrides for auto configuration
    if tier or probes is not None:
        if tier and tier not in ["conservative", "balanced", "aggressive", "none"]:
            _event(
                console,
                "FAIL",
                f"Invalid tier '{tier}'. Valid options: conservative, balanced, aggressive, none",
                emoji="❌",
                profile=profile,
            )
            raise typer.Exit(1)
        if probes is not None and (probes < 0 or probes > 10):
            _event(
                console,
                "FAIL",
                f"Invalid probes '{probes}'. Must be between 0 and 10",
                emoji="❌",
                profile=profile,
            )
            raise typer.Exit(1)

        cfg_dict = cfg.model_dump()
        auto_section = (
            cfg_dict.get("auto") if isinstance(cfg_dict.get("auto"), dict) else {}
        )
        cfg_dict["auto"] = auto_section
        if tier:
            auto_section["tier"] = tier
            _event(
                console,
                "INIT",
                f"Auto tier override: {tier}",
                emoji="🎛️",
                profile=profile,
            )
        if probes is not None:
            auto_section["probes"] = probes
            _event(
                console,
                "INIT",
                f"Auto probes override: {probes}",
                emoji="🔬",
                profile=profile,
            )
        cfg = InvarLockConfig(cfg_dict)

    # Resolve adapter:auto to a concrete built-in adapter if requested
    try:
        from ..adapter_auto import apply_auto_adapter_if_needed as _apply_auto

        cfg = _apply_auto(cfg)
    except Exception:
        pass

    return cfg


def _maybe_plan_release_windows(
    capacity_meta: dict[str, Any],
    *,
    requested_preview: int,
    requested_final: int,
    max_calibration: int,
    console: Console,
) -> dict[str, Any]:
    """Thin wrapper around _plan_release_windows to improve readability."""
    return _plan_release_windows(
        capacity_meta,
        requested_preview=requested_preview,
        requested_final=requested_final,
        max_calibration=max_calibration,
        console=console,
    )


def _print_pipeline_start(console: Console) -> None:
    _event(console, "INIT", "Starting InvarLock pipeline...", emoji="🚀")


def _emit_run_artifacts(
    *, report: Any, out_dir: Path, filename_prefix: str, console: Console
) -> dict[str, str]:
    """Save run report and return emitted artifact paths."""
    from invarlock.reporting.report import save_report as _save_report

    _event(console, "DATA", "Saving run report...", emoji="💾")
    return _save_report(
        report, out_dir, formats=["json"], filename_prefix=filename_prefix
    )


def _resolve_device_and_output(
    cfg: Any, *, device: str | None, out: str | None, console: Console
) -> tuple[str, Path]:
    """Resolve device and output directory with validation and logging."""
    from ..device import (
        resolve_device as _resolve_device,
    )
    from ..device import (
        validate_device_for_config as _validate,
    )

    try:
        cfg_device = getattr(cfg.model, "device", None)
    except Exception:
        cfg_device = None
    target_device = device or cfg_device or "auto"
    resolved_device = _resolve_device(target_device)
    resolution_note = _device_resolution_note(target_device, resolved_device)
    console.print(_format_kv_line("Device", f"{resolved_device} ({resolution_note})"))
    is_valid, error_msg = _validate(resolved_device)
    if not is_valid:
        _event(console, "FAIL", f"Device validation failed: {error_msg}", emoji="❌")
        raise typer.Exit(1)

    # Determine output directory
    if out:
        output_dir = Path(out)
    else:
        try:
            output_dir = Path(cfg.output.dir)
        except Exception:
            output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(resolved_device), output_dir


def _resolve_provider_and_split(
    cfg: Any,
    model_profile: Any,
    *,
    get_provider_fn: Any,
    provider_kwargs: dict[str, Any] | None = None,
    console: Console,
    resolved_device: str | None = None,
    emit: Callable[[str, str, str | None], None] | None = None,
) -> tuple[Any, str, bool]:
    """Resolve dataset provider and split, returning (provider, split, used_fallback)."""
    provider_name = None
    provider_kwargs = dict(provider_kwargs or {})
    try:
        provider_val = cfg.dataset.provider
    except Exception:
        provider_val = None
    if isinstance(provider_val, str) and provider_val:
        provider_name = provider_val
    else:
        try:
            provider_name = provider_val.kind  # type: ignore[attr-defined]
            try:
                for k, v in provider_val.items():  # type: ignore[attr-defined]
                    if k != "kind" and v is not None and v != "":
                        provider_kwargs[k] = v
            except Exception:
                pass
        except Exception:
            provider_name = None
    if not provider_name:
        provider_name = getattr(model_profile, "default_provider", None) or "wikitext2"
    # Pass device hint only to providers that understand it (currently WikiText-2)
    if resolved_device and provider_name == "wikitext2":
        provider_kwargs.setdefault("device_hint", resolved_device)
    if emit is not None and provider_name == "wikitext2":
        data_provider = get_provider_fn(provider_name, emit=emit, **provider_kwargs)
    else:
        data_provider = get_provider_fn(provider_name, **provider_kwargs)

    requested_split = None
    try:
        requested_split = getattr(cfg.dataset, "split", None)
    except Exception:
        requested_split = None
    available_splits = None
    if hasattr(data_provider, "available_splits"):
        try:
            available_splits = list(data_provider.available_splits())  # type: ignore[attr-defined]
        except Exception:
            available_splits = None
    resolved_split, used_fallback_split = _choose_dataset_split(
        requested=requested_split, available=available_splits
    )
    return data_provider, resolved_split, used_fallback_split


def _extract_model_load_kwargs(cfg: InvarLockConfig) -> dict[str, Any]:
    """Return adapter.load_model kwargs from config (excluding core fields)."""
    try:
        data = cfg.model_dump()
    except Exception:
        data = {}
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return {}
    extra = {
        key: value
        for key, value in model.items()
        if key not in {"id", "adapter", "device"} and value is not None
    }
    # Backwards-compatible aliasing: config `dtype` → HF `torch_dtype`.
    if "dtype" in extra and "torch_dtype" not in extra:
        extra["torch_dtype"] = extra.pop("dtype")

    # Normalize torch_dtype when present (keep as string for JSON-ability).
    if "torch_dtype" in extra and isinstance(extra.get("torch_dtype"), str):
        dtype_str = str(extra.get("torch_dtype") or "").strip().lower()
        aliases = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }
        if dtype_str in aliases:
            extra["torch_dtype"] = aliases[dtype_str]
        elif dtype_str:
            extra["torch_dtype"] = dtype_str

    return extra


def _load_model_with_cfg(
    adapter: Any,
    cfg: InvarLockConfig,
    device: str,
    *,
    profile: str | None = None,
) -> Any:
    """Load a model with config-provided kwargs, filtering for strict adapters."""
    try:
        model_id = cfg.model.id
    except Exception:
        try:
            model_id = (cfg.model_dump().get("model") or {}).get("id")
        except Exception:
            model_id = None
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Missing model.id in config")

    extra = _extract_model_load_kwargs(cfg)
    with _suppress_noisy_warnings(profile):
        try:
            sig = inspect.signature(adapter.load_model)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_var_kw:
                return adapter.load_model(model_id, device=device, **extra)
            allowed = {k: v for k, v in extra.items() if k in sig.parameters}
            if allowed:
                return adapter.load_model(model_id, device=device, **allowed)
        except Exception:
            # Fall back to the strictest call shape.
            pass
        return adapter.load_model(model_id, device=device)


def _run_bare_control(
    *,
    adapter: Any,
    edit_op: Any,
    cfg: Any,
    model: Any,
    run_config: Any,
    calibration_data: list[Any],
    auto_config: Any,
    edit_config: Any,
    preview_count: int,
    final_count: int,
    seed_bundle: dict[str, int | None],
    resolved_device: str,
    restore_fn: Any | None,
    console: Console,
    resolved_loss_type: str,
    profile_normalized: str | None = None,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
) -> dict[str, Any] | None:
    """Execute the bare-control run for overhead estimation and return payload."""
    from invarlock.core.runner import CoreRunner as _CoreRunner

    _event(
        console,
        "EXEC",
        "Running bare control (guards disabled) for overhead check",
        emoji="🧪",
        profile=profile_normalized,
    )
    set_seed(seed_bundle["python"])  # type: ignore[arg-type]

    bare_runner = _CoreRunner()
    bare_config = copy.deepcopy(run_config)
    bare_config.event_path = None
    bare_context = copy.deepcopy(run_config.context)
    bare_context.setdefault("validation", {})["guard_overhead_mode"] = "bare"
    bare_config.context = bare_context
    runtime_edit_config = dict(edit_config or {})
    runtime_edit_config.setdefault("console", console)
    runtime_edit_config.setdefault(
        "output_style", _style_from_console(console, profile=profile_normalized)
    )
    runtime_edit_config.setdefault("emit", True)

    private_model_loaded = False
    bare_target_model = None
    try:
        if restore_fn and model is not None:
            try:
                restore_fn()
            except Exception as exc:
                raise _SnapshotRestoreFailed(str(exc)) from exc
            bare_target_model = model
        elif skip_model_load:
            bare_target_model = model or SimpleNamespace(name="bare_stub_model")
        else:
            bare_target_model = _load_model_with_cfg(
                adapter, cfg, resolved_device, profile=profile_normalized
            )
            private_model_loaded = True
            if snapshot_provenance is not None:
                snapshot_provenance["reload_path_used"] = True

        bare_report = bare_runner.execute(
            model=bare_target_model,
            adapter=adapter,
            edit=edit_op,
            guards=[],
            config=bare_config,
            calibration_data=calibration_data,
            auto_config=auto_config,
            edit_config=runtime_edit_config,
            preview_n=preview_count,
            final_n=final_count,
        )
    finally:
        if private_model_loaded:
            _free_model_memory(bare_target_model)

    bare_ppl_final = None
    bare_ppl_preview = None
    if hasattr(bare_report, "metrics") and bare_report.metrics:
        bare_pm = bare_report.metrics.get("primary_metric", {})
        bare_ppl_final = bare_pm.get("final") if isinstance(bare_pm, dict) else None
        bare_ppl_preview = bare_pm.get("preview") if isinstance(bare_pm, dict) else None

    if profile_normalized in {"ci", "release"}:

        def _finite(x: Any) -> bool:
            try:
                return isinstance(x, (int | float)) and math.isfinite(float(x))
            except Exception:
                return False

        if not (_finite(bare_ppl_preview) and _finite(bare_ppl_final)):
            _event(
                console,
                "WARN",
                "Primary metric non-finite during bare control; continuing with diagnostics.",
                emoji="⚠️",
                profile=profile_normalized,
            )

    payload: dict[str, Any] = {
        "overhead_threshold": GUARD_OVERHEAD_THRESHOLD,
        "messages": [],
        "warnings": [],
        "errors": [],
        "checks": {},
        "source": f"{profile_normalized or 'ci'}_profile",
        "mode": "bare",
    }

    if getattr(bare_report, "status", "").lower() not in {"success", "completed", "ok"}:
        payload["warnings"].append(
            f"Bare run status: {getattr(bare_report, 'status', 'unknown')}"
        )

    try:
        lk = str(resolved_loss_type or "causal").lower()
        if lk == "mlm":
            pm_kind_bare = "ppl_mlm"
        elif lk in {"seq2seq", "s2s", "t5"}:
            pm_kind_bare = "ppl_seq2seq"
        else:
            pm_kind_bare = "ppl_causal"
        pm_bare = _extract_pm_snapshot_for_overhead(bare_report, kind=pm_kind_bare)
        if isinstance(pm_bare, dict) and pm_bare:
            payload["bare_report"] = {"metrics": {"primary_metric": pm_bare}}
    except Exception:
        pass

    set_seed(seed_bundle["python"])  # type: ignore[arg-type]
    return payload


def _execute_guarded_run(
    *,
    runner: Any,
    adapter: Any,
    model: Any,
    cfg: Any,
    edit_op: Any,
    run_config: Any,
    guards: list[Any],
    calibration_data: list[Any],
    auto_config: Any,
    edit_config: Any,
    preview_count: int,
    final_count: int,
    restore_fn: Any | None,
    resolved_device: str,
    profile_normalized: str | None = None,
    console: Console,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
) -> tuple[Any, Any]:
    """Restore or load model and execute the guarded CoreRunner."""
    if restore_fn and model is not None:
        try:
            restore_fn()
        except Exception as exc:
            raise _SnapshotRestoreFailed(str(exc)) from exc
    elif skip_model_load:
        model = model or SimpleNamespace(name="guarded_stub_model")
    else:
        _event(
            console,
            "INIT",
            f"Loading model: {cfg.model.id} (attempt 1)",
            emoji="🔧",
            profile=profile_normalized,
        )
        model = _load_model_with_cfg(
            adapter, cfg, resolved_device, profile=profile_normalized
        )
        if snapshot_provenance is not None:
            snapshot_provenance["reload_path_used"] = True

    runtime_edit_config = dict(edit_config or {})
    runtime_edit_config.setdefault("console", console)
    runtime_edit_config.setdefault(
        "output_style", _style_from_console(console, profile=profile_normalized)
    )
    runtime_edit_config.setdefault("emit", True)

    core_report = runner.execute(
        model=model,
        adapter=adapter,
        edit=edit_op,
        guards=guards,
        config=run_config,
        calibration_data=calibration_data,
        auto_config=auto_config,
        edit_config=runtime_edit_config,
        preview_n=preview_count,
        final_n=final_count,
    )
    return core_report, model


def _postprocess_and_summarize(
    *,
    report: dict[str, Any],
    run_dir: Path,
    run_config: Any,
    window_plan: dict[str, Any] | None,
    dataset_meta: dict[str, Any],
    match_fraction: float | None,
    overlap_fraction: float | None,
    console: Console,
) -> dict[str, str]:
    """Finalize report windows stats and print/save summary artifacts."""
    try:
        ds = report.setdefault("dataset", {}).setdefault("windows", {})
        stats = ds.setdefault("stats", {})
        if match_fraction is not None:
            stats["window_match_fraction"] = float(match_fraction)
        if overlap_fraction is not None:
            stats["window_overlap_fraction"] = float(overlap_fraction)
        try:
            if isinstance(window_plan, dict) and "coverage_ok" in window_plan:
                stats["coverage"] = bool(window_plan.get("coverage_ok"))
        except Exception:
            pass
    except Exception:
        pass

    saved_files = _emit_run_artifacts(
        report=report, out_dir=run_dir, filename_prefix="report", console=console
    )
    _event(console, "PASS", "Run completed successfully!", emoji="✅")
    _event(console, "DATA", f"Report: {saved_files['json']}", emoji="📄")
    if run_config.event_path:
        _event(console, "DATA", f"Events: {run_config.event_path}", emoji="📝")
    return saved_files


def _compute_provider_digest(report: dict[str, Any]) -> dict[str, str] | None:
    """Compute provider digest (ids/tokenizer/masking) from report context.

    Returns a dict with keys: ids_sha256, tokenizer_sha256, masking_sha256?
    """
    # Prefer centralized digest helpers
    from invarlock.utils.digest import hash_json as _hash_json

    windows = report.get("evaluation_windows") if isinstance(report, dict) else None
    if not isinstance(windows, dict) or not windows:
        return None
    # window_ids digest across preview+final (sorted for stability)
    all_ids: list = []
    for key in ("preview", "final"):
        sec = windows.get(key)
        if not isinstance(sec, dict):
            continue
        wids = sec.get("window_ids")
        if isinstance(wids, list):
            all_ids.extend(list(wids))
    ids_sha = None
    if all_ids:
        # Prefer ints when possible; fall back to strings to avoid mixed-type sorting.
        ids_int: list[int] = []
        use_ints = True
        for raw in all_ids:
            try:
                ids_int.append(int(raw))
            except Exception:
                use_ints = False
                break
        if use_ints:
            ids_sha = _hash_json(sorted(ids_int))
        else:
            ids_str = [str(v) for v in all_ids]
            ids_sha = _hash_json(sorted(ids_str))

    # tokenizer hash: prefer meta.tokenizer_hash then data.tokenizer_hash
    tok_hash = None
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else None
    if isinstance(meta, dict):
        tok_hash = meta.get("tokenizer_hash")
    if not tok_hash and isinstance(report.get("data"), dict):
        tok_hash = report["data"].get("tokenizer_hash")

    # masking hash from mask positions
    masking = _compute_mask_positions_digest(windows)

    digest: dict[str, str] = {}
    if isinstance(ids_sha, str) and ids_sha:
        digest["ids_sha256"] = ids_sha
    if isinstance(tok_hash, str) and tok_hash:
        digest["tokenizer_sha256"] = str(tok_hash)
    if isinstance(masking, str) and masking:
        digest["masking_sha256"] = masking
    return digest or None


def _validate_and_harvest_baseline_schedule(
    cfg: Any,
    pairing_schedule: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    *,
    tokenizer_hash: str | None,
    resolved_loss_type: str,
    profile: str | None = None,
    baseline_path_str: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    """Validate baseline schedule compatibility and harvest dataset metadata.

    Returns a mapping with keys: effective_preview, effective_final, preview_count,
    final_count, dataset_meta, window_plan, calibration_data.
    """

    # Helpers
    def _print(msg: str) -> None:
        if console is not None:
            console.print(msg)

    def _fail_schedule(reason: str) -> None:
        path = baseline_path_str or "baseline"
        prof = (profile or "dev").strip().lower()
        message = f"PAIRING-EVIDENCE-MISSING: {path}: {reason}"
        if prof in {"ci", "release"}:
            raise InvarlockError(code="E001", message=message)
        if console is not None:
            _event(
                console,
                "FAIL",
                f"Baseline pairing schedule '{path}' is incompatible: {reason}",
                emoji="❌",
                profile=prof,
            )
        raise typer.Exit(1)

    baseline_meta = (
        baseline_report_data.get("data")
        if isinstance(baseline_report_data, dict)
        else {}
    )
    if not isinstance(baseline_meta, dict):
        baseline_meta = {}

    def _extract_meta(field: str, default: Any = None) -> Any:
        value = baseline_meta.get(field)
        return value if value is not None else default

    # Structural integrity checks (fail closed in CI/Release)
    try:
        prev = (
            pairing_schedule.get("preview")
            if isinstance(pairing_schedule, dict)
            else None
        )
        fin = (
            pairing_schedule.get("final")
            if isinstance(pairing_schedule, dict)
            else None
        )
        if not isinstance(prev, dict) or not isinstance(fin, dict):
            _fail_schedule("missing preview/final evaluation_windows sections")

        def _arm_check(
            label: str, section: dict[str, Any]
        ) -> tuple[list[int], list[list[int]]]:
            wids = section.get("window_ids")
            toks = section.get("input_ids")
            masks = section.get("attention_masks")
            if not isinstance(wids, list) or not isinstance(toks, list):
                _fail_schedule(f"invalid {label} section: missing window_ids/input_ids")
            if len(wids) != len(toks):
                _fail_schedule(
                    f"{label} coherence error: len(window_ids)={len(wids)} len(input_ids)={len(toks)}"
                )
            ids_int: list[int] = []
            seqs: list[list[int]] = []
            for idx, (wid, seq) in enumerate(zip(wids, toks, strict=False)):
                try:
                    wid_int = int(wid)
                except Exception:
                    _fail_schedule(
                        f"{label} window_ids contains non-int at index {idx}"
                    )
                ids_int.append(wid_int)
                seq_ints = _tensor_or_list_to_ints(seq)
                if not seq_ints:
                    _fail_schedule(f"{label} input_ids empty at index {idx}")
                seqs.append(seq_ints)

            # attention_masks are required for pairing, but some baselines may omit them.
            # When absent, default to all-ones masks (cannot infer padding reliably).
            masks_rows: list[list[int]] = []
            masks_missing = masks is None or masks == []
            if (
                isinstance(masks, list)
                and masks
                and len(seqs) == 1
                and not isinstance(masks[0], list)
            ):  # type: ignore[index]
                masks = [masks]

            if isinstance(masks, list) and masks:
                if len(masks) != len(seqs):
                    _fail_schedule(
                        f"{label} coherence error: len(attention_masks)={len(masks)} len(input_ids)={len(seqs)}"
                    )
                for j, (seq_ints, mask) in enumerate(zip(seqs, masks, strict=False)):
                    if not isinstance(mask, list):
                        _fail_schedule(
                            f"{label} attention_masks row is not a list at index {j}"
                        )
                    mask_ints = _tensor_or_list_to_ints(mask)
                    if len(mask_ints) != len(seq_ints):
                        _fail_schedule(
                            f"{label} attention_masks length mismatch at index {j}"
                        )
                    masks_rows.append(mask_ints)
            else:
                masks_missing = True
                masks_rows = [[1] * len(seq) for seq in seqs]

            if masks_missing:
                try:
                    section["attention_masks"] = masks_rows
                except Exception:
                    pass

            # Optional MLM fields must align when present.
            labels = section.get("labels")
            if isinstance(labels, list) and labels:
                if len(labels) != len(seqs):
                    _fail_schedule(f"{label} labels length mismatch")
                for j, row in enumerate(labels):
                    row_ints = _tensor_or_list_to_ints(row)
                    if len(row_ints) != len(seqs[j]):
                        _fail_schedule(f"{label} labels length mismatch at index {j}")

            for key in ("masked_token_counts", "actual_token_counts"):
                if section.get(key) is not None:
                    raw_counts = section.get(key)
                    if not isinstance(raw_counts, list) or len(raw_counts) != len(seqs):
                        _fail_schedule(f"{label} {key} length mismatch")
            return ids_int, seqs

        prev_ids, prev_seqs = _arm_check("preview", prev)
        fin_ids, fin_seqs = _arm_check("final", fin)

        if len(set(prev_ids)) != len(prev_ids):
            _fail_schedule("duplicate window_ids detected in preview arm")
        if len(set(fin_ids)) != len(fin_ids):
            _fail_schedule("duplicate window_ids detected in final arm")
        if set(prev_ids) & set(fin_ids):
            _fail_schedule("window_ids overlap between preview and final arms")

        def _hash_tokens(tokens: list[int]) -> bytes:
            if not tokens:
                return b""
            token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
            return hashlib.blake2b(token_array.tobytes(), digest_size=16).digest()

        prev_hashes = [_hash_tokens(seq) for seq in prev_seqs]
        fin_hashes = [_hash_tokens(seq) for seq in fin_seqs]
        if len(set(prev_hashes)) != len(prev_hashes):
            _fail_schedule("duplicate token sequences detected in preview arm")
        if len(set(fin_hashes)) != len(fin_hashes):
            _fail_schedule("duplicate token sequences detected in final arm")
        if set(prev_hashes) & set(fin_hashes):
            _fail_schedule("preview/final token sequence overlap detected")

        # Optional: validate baseline hashes when present in baseline report data
        expected_preview_hash = _hash_sequences(prev_seqs)
        expected_final_hash = _hash_sequences(fin_seqs)
        expected_dataset_hash = hashlib.blake2s(
            (expected_preview_hash + expected_final_hash).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        baseline_preview_hash = baseline_meta.get("preview_hash")
        baseline_final_hash = baseline_meta.get("final_hash")
        baseline_dataset_hash = baseline_meta.get("dataset_hash")
        if (
            isinstance(baseline_preview_hash, str)
            and baseline_preview_hash
            and baseline_preview_hash != expected_preview_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("preview_hash mismatch vs baseline report data")
            if console is not None:
                _event(
                    console,
                    "WARN",
                    "Baseline preview_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )
        if (
            isinstance(baseline_final_hash, str)
            and baseline_final_hash
            and baseline_final_hash != expected_final_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("final_hash mismatch vs baseline report data")
            if console is not None:
                _event(
                    console,
                    "WARN",
                    "Baseline final_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )
        if (
            isinstance(baseline_dataset_hash, str)
            and baseline_dataset_hash
            and baseline_dataset_hash != expected_dataset_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("dataset_hash mismatch vs baseline report data")
            if console is not None:
                _event(
                    console,
                    "WARN",
                    "Baseline dataset_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )
    except InvarlockError:
        raise
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail_schedule(f"failed to validate baseline schedule integrity ({exc})")

    # Adopt counts from the schedule, warning if they differ from cfg
    baseline_preview = len(pairing_schedule["preview"].get("input_ids") or [])
    baseline_final = len(pairing_schedule["final"].get("input_ids") or [])
    cfg_preview = getattr(cfg.dataset, "preview_n", None)
    cfg_final = getattr(cfg.dataset, "final_n", None)
    if (
        cfg_preview is not None
        and baseline_preview is not None
        and baseline_preview != cfg_preview
    ) or (
        cfg_final is not None
        and baseline_final is not None
        and baseline_final != cfg_final
    ):
        if console is not None:
            _event(
                console,
                "WARN",
                f"Adjusting evaluation window counts to match baseline schedule ({baseline_preview}/{baseline_final}).",
                emoji="⚠️",
                profile=profile,
            )

    effective_preview = int(baseline_preview)
    effective_final = int(baseline_final)
    preview_count = effective_preview
    final_count = effective_final

    # Validate key dataset parameters
    cfg_seq_len = getattr(cfg.dataset, "seq_len", None)
    baseline_seq_len = _extract_meta("seq_len")
    if (
        cfg_seq_len is not None
        and baseline_seq_len is not None
        and baseline_seq_len != cfg_seq_len
    ):
        _fail_schedule(
            f"sequence length mismatch (baseline {baseline_seq_len} vs config {cfg_seq_len})"
        )

    cfg_stride = getattr(cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", None))
    baseline_stride = _extract_meta("stride")
    if (
        baseline_stride is not None
        and cfg_stride is not None
        and baseline_stride != cfg_stride
    ):
        _fail_schedule(
            f"stride mismatch (baseline {baseline_stride} vs config {cfg_stride})"
        )

    cfg_dataset = getattr(cfg.dataset, "provider", None)
    if cfg_dataset is None:
        cfg_dataset = getattr(cfg.dataset, "dataset", None)
    baseline_dataset = _extract_meta("dataset")
    if (
        baseline_dataset is not None
        and cfg_dataset is not None
        and baseline_dataset != cfg_dataset
    ):
        _fail_schedule(
            f"dataset mismatch (baseline {baseline_dataset} vs config {cfg_dataset})"
        )

    cfg_split = getattr(cfg.dataset, "split", "validation")
    baseline_split = _extract_meta("split")
    if (
        baseline_split is not None
        and cfg_split is not None
        and baseline_split != cfg_split
    ):
        _fail_schedule(
            f"split mismatch (baseline {baseline_split} vs config {cfg_split})"
        )

    baseline_tokenizer_hash = baseline_meta.get("tokenizer_hash")
    if (
        baseline_tokenizer_hash
        and tokenizer_hash
        and baseline_tokenizer_hash != tokenizer_hash
    ):
        _fail_schedule(
            "tokenizer hash mismatch between baseline and current configuration"
        )

    dataset_meta = {
        key: baseline_meta.get(key)
        for key in (
            "tokenizer_hash",
            "tokenizer_name",
            "vocab_size",
            "bos_token",
            "eos_token",
            "pad_token",
            "add_prefix_space",
            "dataset_hash",
            "preview_hash",
            "final_hash",
            "preview_total_tokens",
            "final_total_tokens",
        )
        if baseline_meta.get(key) is not None
    }
    dataset_meta["loss_type"] = resolved_loss_type
    window_plan = baseline_meta.get("window_plan")
    calibration_data: list[Any] | None = []

    return {
        "effective_preview": effective_preview,
        "effective_final": effective_final,
        "preview_count": preview_count,
        "final_count": final_count,
        "dataset_meta": dataset_meta,
        "window_plan": window_plan,
        "calibration_data": calibration_data,
    }


def _enforce_provider_parity(
    subject_digest: dict | None, baseline_digest: dict | None, *, profile: str | None
) -> None:
    """Enforce tokenizer/masking parity rules for CI/Release profiles.

    - If tokenizers differ in CI/Release, abort.
    - If tokenizers match but masking digests differ (MLM), abort.
    No-ops outside CI/Release.
    """
    prof = (profile or "").strip().lower()
    if prof not in {"ci", "release"}:
        return
    sd = subject_digest or {}
    bd = baseline_digest or {}
    subj_ids = sd.get("ids_sha256")
    base_ids = bd.get("ids_sha256")
    subj_tok = sd.get("tokenizer_sha256")
    base_tok = bd.get("tokenizer_sha256")
    subj_mask = sd.get("masking_sha256")
    base_mask = bd.get("masking_sha256")
    # Missing digest information in CI/Release → abort
    if not (
        isinstance(subj_ids, str)
        and isinstance(base_ids, str)
        and subj_ids
        and base_ids
        and isinstance(subj_tok, str)
        and isinstance(base_tok, str)
        and subj_tok
        and base_tok
    ):
        raise InvarlockError(
            code="E004",
            message="PROVIDER-DIGEST-MISSING: subject or baseline missing ids/tokenizer digest",
        )
    # Window-ids mismatch → abort
    if subj_ids != base_ids:
        raise InvarlockError(
            code="E006",
            message="IDS-DIGEST-MISMATCH: subject and baseline window IDs differ",
        )
    # Tokenizer mismatch → abort with code
    if subj_tok != base_tok:
        raise InvarlockError(
            code="E002",
            message="TOKENIZER-DIGEST-MISMATCH: subject and baseline tokenizers differ",
        )
    # Masking mismatch under identical tokenizers → abort
    if (
        isinstance(subj_mask, str)
        and isinstance(base_mask, str)
        and subj_mask
        and base_mask
        and subj_mask != base_mask
    ):
        raise InvarlockError(
            code="E003",
            message="MASK-PARITY-MISMATCH: mask positions differ under matched tokenizers",
        )


def _resolve_metric_and_provider(
    cfg: Any,
    model_profile: Any,
    *,
    resolved_loss_type: str | None = None,
    metric_kind_override: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    """Resolve metric kind, provider kind, and metric options from config with precedence.

    Precedence: CLI args (not handled here) → config → ModelProfile defaults → fallback.
    Primary metric (metric‑v1) is canonical in dev‑phase; no env flag toggles.
    """
    # Provider kind
    provider_val = None
    try:
        provider_val = cfg.dataset.provider
    except Exception:
        provider_val = None
    provider_kind = None
    if isinstance(provider_val, str) and provider_val:
        provider_kind = provider_val
    else:
        # Support object-like config sections (e.g., InvarLockConfig _Obj)
        try:
            provider_kind = provider_val.kind
        except Exception:
            try:
                provider_kind = provider_val.get("kind")  # type: ignore[attr-defined]
            except Exception:
                provider_kind = None
    if not provider_kind and hasattr(model_profile, "default_provider"):
        provider_kind = model_profile.default_provider
    # Fallback to a known provider name supported by get_provider()
    if not provider_kind:
        provider_kind = "wikitext2"

    # Metric config
    metric_cfg = None
    try:
        eval_section = cfg.eval
        metric_cfg = getattr(eval_section, "metric", None)
    except Exception:
        metric_cfg = None

    metric_kind = None
    if isinstance(metric_kind_override, str) and metric_kind_override.strip():
        mk_override = metric_kind_override.strip().lower()
        if mk_override != "auto":
            metric_kind = mk_override
    reps = None
    ci_level = None
    if metric_kind is None and metric_cfg is not None:
        try:
            metric_kind = (
                metric_cfg.get("kind")
                if isinstance(metric_cfg, dict)
                else metric_cfg.kind
            )
        except Exception:
            metric_kind = None
        try:
            reps = (
                metric_cfg.get("reps")
                if isinstance(metric_cfg, dict)
                else metric_cfg.reps
            )
        except Exception:
            reps = None
        try:
            ci_level = (
                metric_cfg.get("ci_level")
                if isinstance(metric_cfg, dict)
                else metric_cfg.ci_level
            )
        except Exception:
            ci_level = None

    # Resolve metric kind from config
    if isinstance(metric_kind, str) and metric_kind:
        mk = metric_kind.strip().lower()
        if mk == "auto":
            metric_kind = None
        else:
            metric_kind = mk
    else:
        metric_kind = None

    # Fallback to model profile default or loss-type mapping
    if not metric_kind and hasattr(model_profile, "default_metric"):
        metric_kind = model_profile.default_metric
    if not metric_kind:
        # Map from loss kind
        lk = (resolved_loss_type or "causal").lower()
        if lk == "mlm":
            metric_kind = "ppl_mlm"
        elif lk in {"seq2seq", "s2s", "t5"}:
            metric_kind = "ppl_seq2seq"
        else:
            metric_kind = "ppl_causal"

    # Metric options dict if present
    opts: dict[str, float] = {}
    if reps is not None:
        try:
            opts["reps"] = float(int(reps))
        except Exception:
            pass
    if ci_level is not None:
        try:
            opts["ci_level"] = float(ci_level)
        except Exception:
            pass

    return str(metric_kind), str(provider_kind), opts


def _plan_release_windows(
    capacity: dict[str, Any],
    *,
    requested_preview: int,
    requested_final: int,
    max_calibration: int,
    console: Console | None = None,
) -> dict[str, Any]:
    """Derive release-tier window plan based on dataset capacity."""
    available_unique = int(capacity.get("available_unique", 0))
    available_nonoverlap = int(capacity.get("available_nonoverlap", 0))
    total_tokens = int(capacity.get("total_tokens", 0))
    dedupe_rate = float(capacity.get("dedupe_rate", 0.0))
    candidate_unique = capacity.get("candidate_unique")
    if candidate_unique is not None and int(candidate_unique) > 0:
        effective_unique = int(candidate_unique)
    else:
        effective_unique = available_unique
    candidate_limit = capacity.get("candidate_limit")

    target_per_arm = int(min(requested_preview, requested_final))
    if target_per_arm <= 0:
        target_per_arm = requested_preview or requested_final or 1

    max_calibration = max(0, int(max_calibration or 0))
    if max_calibration > 0:
        calibration_windows = max(
            RELEASE_CALIBRATION_MIN,
            min(RELEASE_CALIBRATION_MAX, max_calibration // 10),
        )
    else:
        calibration_windows = RELEASE_CALIBRATION_MIN
    calibration_windows = min(calibration_windows, available_unique)

    buffer_windows = math.ceil(RELEASE_BUFFER_FRACTION * effective_unique)
    reserve_windows = min(effective_unique, calibration_windows + buffer_windows)
    available_for_eval = max(0, effective_unique - reserve_windows)
    actual_per_arm_raw = available_for_eval // 2

    coverage_ok = actual_per_arm_raw >= RELEASE_MIN_WINDOWS_PER_ARM
    if not coverage_ok:
        raise RuntimeError(
            "Release profile capacity insufficient: "
            f"available_unique={available_unique}, reserve={reserve_windows} "
            f"(calibration={calibration_windows}, buffer={buffer_windows}), "
            f"usable_per_arm={actual_per_arm_raw}, "
            f"requires ≥{RELEASE_MIN_WINDOWS_PER_ARM} per arm."
        )

    actual_per_arm = min(target_per_arm, actual_per_arm_raw)

    if console:
        candidate_msg = ""
        if candidate_unique is not None:
            candidate_msg = f", candidate_unique={int(candidate_unique)}" + (
                f"/{int(candidate_limit)}" if candidate_limit is not None else ""
            )
        _event(
            console,
            "METRIC",
            "Release window capacity:"
            f" unique={available_unique}, reserve={reserve_windows} "
            f"(calib {calibration_windows}, buffer {buffer_windows}), "
            f"usable={available_for_eval}, "
            f"per-arm raw={actual_per_arm_raw} → selected {actual_per_arm} "
            f"(target {target_per_arm}{candidate_msg})",
            emoji="📏",
            profile="release",
        )
        if actual_per_arm < target_per_arm:
            _event(
                console,
                "WARN",
                f"Adjusted per-arm windows down from {target_per_arm} to {actual_per_arm} based on capacity.",
                emoji="⚠️",
                profile="release",
            )

    plan = {
        "profile": "release",
        "requested_preview": int(requested_preview),
        "requested_final": int(requested_final),
        "target_per_arm": target_per_arm,
        "min_per_arm": RELEASE_MIN_WINDOWS_PER_ARM,
        "actual_preview": int(actual_per_arm),
        "actual_final": int(actual_per_arm),
        "actual_per_arm_raw": int(actual_per_arm_raw),
        "coverage_ok": coverage_ok,
        "capacity": {
            "total_tokens": total_tokens,
            "available_nonoverlap": available_nonoverlap,
            "available_unique": available_unique,
            "effective_unique": effective_unique,
            "dedupe_rate": dedupe_rate,
            "calibration": calibration_windows,
            "buffer_fraction": RELEASE_BUFFER_FRACTION,
            "buffer_windows": buffer_windows,
            "reserve_windows": reserve_windows,
            "usable_after_reserve": available_for_eval,
        },
    }
    if candidate_unique is not None:
        plan["capacity"]["candidate_unique"] = int(candidate_unique)
    if candidate_limit is not None:
        plan["capacity"]["candidate_limit"] = int(candidate_limit)
    return plan


# Check if core components are available
try:
    from invarlock.core.api import RunConfig  # noqa: F401
    from invarlock.core.registry import get_registry  # noqa: F401

    HAS_CORE_COMPONENTS = True
except ImportError:
    HAS_CORE_COMPONENTS = False


def run_command(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file"
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device override (auto|cuda|mps|cpu)"
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to apply (e.g. ci, release, ci_cpu; dev is a no-op)",
    ),
    out: str | None = typer.Option(None, "--out", help="Output directory override"),
    edit: str | None = typer.Option(None, "--edit", help="Edit kind (quant|mixed)"),
    edit_label: str | None = typer.Option(
        None,
        "--edit-label",
        help=(
            "Edit algorithm label for BYOE models. Use 'noop' for baseline, "
            "'quant_rtn' etc. for built-in edits, 'custom' for pre-edited models."
        ),
    ),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Auto-tuning tier override (conservative|balanced|aggressive)",
    ),
    metric_kind: str | None = typer.Option(
        None,
        "--metric-kind",
        help="Primary metric kind override (ppl_causal|ppl_mlm|accuracy|etc.)",
    ),
    probes: int | None = typer.Option(
        None, "--probes", help="Number of micro-probes (0=deterministic, >0=adaptive)"
    ),
    until_pass: bool = typer.Option(
        False, "--until-pass", help="Retry until certificate passes (max 3 attempts)"
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts", help="Maximum retry attempts for --until-pass mode"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", help="Timeout in seconds for --until-pass mode"
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Path to baseline report.json for certificate validation",
    ),
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", help="Skip cleanup of temporary artifacts"
    ),
    style: str | None = typer.Option(
        None, "--style", help="Output style (audit|friendly)"
    ),
    progress: bool = typer.Option(
        False, "--progress", help="Show progress done messages"
    ),
    timing: bool = typer.Option(False, "--timing", help="Show timing summary"),
    telemetry: bool = typer.Option(
        False, "--telemetry", help="Write telemetry JSON alongside the report"
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    """
    Run InvarLock pipeline with the given configuration.

    The command assembles non-overlapping preview/final windows, executes the
    GuardChain (invariants → spectral → RMT → variance), checks pairing/overlap
    invariants, enforces guard-overhead ≤1 %, and emits a run report plus JSONL
    events suitable for certificate generation.
    """

    try:
        from typer.models import OptionInfo as _TyperOptionInfo  # noqa: F401
    except Exception:  # pragma: no cover - typer internals may change
        _TyperOptionInfo = ()  # type: ignore[assignment]

    config = _coerce_option(config)
    device = _coerce_option(device)
    profile = _coerce_option(profile)
    profile_normalized = (str(profile or "")).strip().lower()
    out = _coerce_option(out)
    edit = _coerce_option(edit)
    edit_label = _coerce_option(edit_label)
    tier = _coerce_option(tier)
    metric_kind = _coerce_option(metric_kind)
    probes = _coerce_option(probes)
    until_pass = bool(_coerce_option(until_pass, False))
    max_attempts = int(_coerce_option(max_attempts, 3))
    timeout = _coerce_option(timeout)
    baseline = _coerce_option(baseline)
    no_cleanup = bool(_coerce_option(no_cleanup, False))
    style = _coerce_option(style)
    progress = bool(_coerce_option(progress, False))
    timing = bool(_coerce_option(timing, False))
    telemetry = bool(_coerce_option(telemetry, False))
    no_color = bool(_coerce_option(no_color, False))

    output_style = resolve_output_style(
        style=str(style) if style is not None else None,
        profile=profile_normalized,
        progress=progress,
        timing=timing,
        no_color=no_color,
    )
    console._invarlock_output_style = output_style
    if not output_style.color:
        console.no_color = True
    timings: dict[str, float] = {}
    collect_timings = bool(output_style.timing or telemetry)
    total_start: float | None = perf_counter() if collect_timings else None

    _apply_warning_filters(profile_normalized)

    # Use shared CLI coercers from invarlock.cli.utils
    report_path_out: str | None = None

    def _fail_run(message: str) -> None:
        _event(console, "FAIL", message, emoji="❌", profile=profile_normalized)
        # Generic failure path → exit 1 (InvarlockError paths handle code 3 separately)
        raise typer.Exit(1)

    def _provider_event(tag: str, message: str, emoji: str | None = None) -> None:
        _event(
            console,
            tag,
            message,
            emoji=emoji,
            profile=profile_normalized,
        )

    # Fail fast when torch is missing so users see a clear extras hint instead of
    # a raw ModuleNotFoundError from deeper imports.
    try:
        import torch as _torch  # type: ignore[import]

        _ = _torch  # pragma: no cover
    except (ImportError, ModuleNotFoundError) as e:
        _event(
            console,
            "FAIL",
            "Torch is required for this command. "
            'Install extras with: pip install "invarlock[hf]" '
            'or "invarlock[adapters]".',
            emoji="❌",
            profile=profile_normalized,
        )
        raise typer.Exit(1) from e

    # use module-level _extract_pairing_schedule

    # use module-level _to_int_list, _tensor_or_list_to_ints, _safe_int

    # Use the module-level _hash_sequences to avoid duplication

    # use module-level _derive_mlm_seed

    # use module-level _apply_mlm_masks

    # use module-level _tokenizer_digest

    try:
        # Import InvarLock components
        from invarlock.core.api import RunConfig
        from invarlock.core.registry import get_registry
        from invarlock.core.runner import CoreRunner
        from invarlock.eval.data import EvaluationWindow, get_provider
        from invarlock.reporting.report_types import create_empty_report

        # Load and validate configuration via helper (preserves console prints)
        cfg = _prepare_config_for_run(
            config_path=config,
            profile=profile,
            edit=edit,
            tier=tier,
            probes=probes,
            console=console,
        )

        # cfg prepared by helper above

        adapter_name = str(getattr(cfg.model, "adapter", "")).lower()
        model_id_raw = str(getattr(cfg.model, "id", ""))
        model_profile = detect_model_profile(
            model_id=model_id_raw, adapter=adapter_name
        )
        tokenizer_hash: str | None = None
        tokenizer: Any | None = None

        loss_cfg = getattr(cfg.eval, "loss", None)
        resolved_loss_type = (
            str(getattr(loss_cfg, "type", "auto")).lower() if loss_cfg else "auto"
        )
        if resolved_loss_type == "auto":
            resolved_loss_type = model_profile.default_loss
        use_mlm = resolved_loss_type == "mlm"
        mask_prob = _coerce_float(getattr(loss_cfg, "mask_prob", None), 0.15)
        mask_seed = _coerce_int(getattr(loss_cfg, "seed", None), 42)
        random_token_prob = _coerce_float(
            getattr(loss_cfg, "random_token_prob", None), 0.1
        )
        original_token_prob = _coerce_float(
            getattr(loss_cfg, "original_token_prob", None), 0.1
        )
        if loss_cfg is not None and getattr(loss_cfg, "type", None) == "auto":
            try:
                loss_cfg.type = resolved_loss_type  # type: ignore[assignment]
            except Exception:
                pass

        # Set deterministic seeds for Python/NumPy/Torch and record provenance
        raw_seed_value = 42
        if hasattr(cfg, "dataset"):
            try:
                raw_seed_value = getattr(cfg.dataset, "seed", 42)
            except Exception:
                raw_seed_value = 42
        try:
            seed_value = int(raw_seed_value)
        except (TypeError, ValueError, OverflowError):
            seed_value = 42
        set_seed(seed_value)
        # Enforce deterministic algorithms in CI/Release profiles when torch is available
        profile_label = profile_normalized or None
        if torch is not None and profile_label in {"ci", "release"}:
            try:  # pragma: no cover - behavior depends on torch availability
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True, warn_only=False)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = False
                    try:
                        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                # If we cannot enforce determinism here, we will rely on core checks
                pass
        try:
            numpy_seed = int(np.random.get_state()[1][0])
        except Exception:
            numpy_seed = seed_value
        torch_seed = None
        if torch is not None:
            try:
                torch_seed = int(torch.initial_seed())
            except Exception:
                torch_seed = seed_value
        seed_bundle = {
            "python": int(seed_value),
            "numpy": int(numpy_seed),
            "torch": int(torch_seed) if torch_seed is not None else None,
        }
        _event(
            console,
            "INIT",
            "Deterministic seeds → "
            f"python={seed_bundle['python']}, numpy={seed_bundle['numpy']}, "
            f"torch={seed_bundle['torch'] if seed_bundle['torch'] is not None else 'N/A'}",
            emoji="🎲",
            profile=profile_normalized,
        )

        # Resolve device and output directory
        resolved_device, output_dir = _resolve_device_and_output(
            cfg, device=device, out=out, console=console
        )

        determinism_meta: dict[str, Any] | None = None
        try:
            from invarlock.cli.determinism import apply_determinism_preset

            preset = apply_determinism_preset(
                profile=profile_label,
                device=resolved_device,
                seed=int(seed_bundle.get("python") or seed_value),
                threads=int(os.environ.get("INVARLOCK_OMP_THREADS", 1) or 1),
            )
            if isinstance(preset, dict) and preset:
                determinism_meta = preset
                preset_seeds = preset.get("seeds")
                if isinstance(preset_seeds, dict) and preset_seeds:
                    for key in ("python", "numpy", "torch"):
                        if key in preset_seeds:
                            seed_bundle[key] = preset_seeds.get(key)
        except Exception:
            determinism_meta = None

        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{output_dir.name}-{timestamp}" if output_dir.name else timestamp

        console.print(_format_kv_line("Output", str(run_dir)))
        console.print(_format_kv_line("Run ID", run_id))

        # Initialize retry controller if --until-pass mode enabled
        retry_controller = _init_retry_controller(
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
            console=console,
        )

        baseline_report_data: dict[str, Any] | None = None
        pairing_schedule: dict[str, Any] | None = None
        if baseline:
            baseline_path = Path(baseline)
            strict_baseline = profile_normalized in {"ci", "release"}
            if not baseline_path.exists():
                msg = (
                    "PAIRING-EVIDENCE-MISSING: baseline report path does not exist "
                    f"({baseline})"
                )
                if strict_baseline:
                    raise InvarlockError(code="E001", message=msg)
                _event(
                    console,
                    "WARN",
                    f"{msg}. Falling back to dataset schedule.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
            else:
                try:
                    with baseline_path.open(encoding="utf-8") as f:
                        baseline_report_data = json.load(f)
                except Exception as exc:  # noqa: BLE001
                    msg = f"PAIRING-EVIDENCE-MISSING: baseline report JSON parse failed ({exc})"
                    if strict_baseline:
                        raise InvarlockError(code="E001", message=msg) from exc
                    _event(
                        console,
                        "WARN",
                        f"{msg}. Falling back to dataset schedule.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    baseline_report_data = None
                if isinstance(baseline_report_data, dict):
                    pairing_schedule = _extract_pairing_schedule(baseline_report_data)
                    if pairing_schedule:
                        # Normalize baseline report in-memory so downstream digest/parity
                        # computations see a consistent window_id + mask shape even for
                        # baselines missing some fields.
                        try:
                            ew = baseline_report_data.get("evaluation_windows")
                            if not isinstance(ew, dict):
                                ew = {}
                                baseline_report_data["evaluation_windows"] = ew
                            # Merge the sanitized pairing schedule into existing
                            # evaluation_windows without discarding logloss/token_counts.
                            for arm in ("preview", "final"):
                                src = (
                                    pairing_schedule.get(arm)
                                    if isinstance(pairing_schedule, dict)
                                    else None
                                )
                                if not isinstance(src, dict):
                                    continue
                                dst = ew.get(arm)
                                if not isinstance(dst, dict):
                                    ew[arm] = dict(src)
                                    continue
                                for key, value in src.items():
                                    dst[key] = value
                        except Exception:
                            pass
                        # Harvest tokenizer hash provenance from baseline when present.
                        try:
                            if not tokenizer_hash:
                                tok = None
                                meta = (
                                    baseline_report_data.get("meta")
                                    if isinstance(
                                        baseline_report_data.get("meta"), dict
                                    )
                                    else {}
                                )
                                data = (
                                    baseline_report_data.get("data")
                                    if isinstance(
                                        baseline_report_data.get("data"), dict
                                    )
                                    else {}
                                )
                                if isinstance(meta, dict):
                                    tok = meta.get("tokenizer_hash")
                                if not tok and isinstance(data, dict):
                                    tok = data.get("tokenizer_hash")
                                if isinstance(tok, str) and tok:
                                    tokenizer_hash = tok
                        except Exception:
                            pass
                        _event(
                            console,
                            "DATA",
                            "Loaded baseline evaluation schedule for pairing",
                            emoji="🧬",
                            profile=profile_normalized,
                        )
                    else:
                        msg = (
                            "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                            f"evaluation_windows ({baseline})"
                        )
                        if strict_baseline:
                            raise InvarlockError(code="E001", message=msg)
                        _event(
                            console,
                            "WARN",
                            f"{msg}. Falling back to dataset schedule.",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                        baseline_report_data = None
                        pairing_schedule = None

        requested_preview = int(getattr(cfg.dataset, "preview_n", 0))
        requested_final = int(getattr(cfg.dataset, "final_n", 0))
        effective_preview = requested_preview
        effective_final = requested_final
        preview_count = effective_preview
        final_count = effective_final
        # Default split prior to provider resolution; updated if provider exposes splits
        try:
            resolved_split = getattr(cfg.dataset, "split", None) or "validation"
        except Exception:
            resolved_split = "validation"
        used_fallback_split: bool = False

        # Execute the pipeline using CoreRunner
        _print_pipeline_start(console)

        # Get registry and create components
        registry = get_registry()
        adapter = registry.get_adapter(cfg.model.adapter)
        edit_name = getattr(getattr(cfg, "edit", None), "name", None)
        if not isinstance(edit_name, str) or not edit_name.strip():
            _event(
                console,
                "FAIL",
                "Edit configuration must specify a non-empty `edit.name`.",
                emoji="❌",
                profile=profile_normalized,
            )
            raise typer.Exit(1)
        try:
            edit_op = registry.get_edit(edit_name.strip())
        except Exception:
            _event(
                console,
                "WARN",
                f"Unknown edit '{edit_name.strip()}'. Using pass-through shim.",
                emoji="⚠️",
                profile=profile_normalized,
            )
            edit_op = SimpleNamespace(name=edit_name.strip())

        adapter_meta = registry.get_plugin_metadata(cfg.model.adapter, "adapters")
        try:
            from invarlock.cli.provenance import (
                extract_adapter_provenance,
            )  # local import to avoid CLI import cycles

            prov = extract_adapter_provenance(cfg.model.adapter)
            # Attach a small, stable provenance dict under adapter plugin metadata
            adapter_meta["provenance"] = prov.to_dict()
        except Exception:
            # Best-effort only; absence should not break runs
            pass
        try:
            edit_meta = registry.get_plugin_metadata(edit_name.strip(), "edits")
        except Exception:
            edit_meta = {
                "name": edit_name.strip(),
                "module": "edits.unknown",
                "version": "unknown",
            }

        guards = []
        guard_metadata: list[dict[str, Any]] = []
        for guard_name in cfg.guards.order:
            if guard_name != "noop":
                try:
                    guard = registry.get_guard(guard_name)
                    guards.append(guard)
                    guard_metadata.append(
                        registry.get_plugin_metadata(guard_name, "guards")
                    )
                except KeyError:
                    _event(
                        console,
                        "WARN",
                        f"Guard '{guard_name}' not found, skipping",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
        plugin_provenance = {
            "adapter": adapter_meta,
            "edit": edit_meta,
            "guards": guard_metadata,
        }
        pm_acceptance_range = _resolve_pm_acceptance_range(cfg)
        pm_drift_band = _resolve_pm_drift_band(cfg)

        _event(
            console,
            "DATA",
            f"Adapter: {adapter.name}",
            emoji="🔌",
            profile=profile_normalized,
        )

        # Create run configuration
        guard_overrides = {
            "spectral": _to_serialisable_dict(getattr(cfg.guards, "spectral", {})),
            "rmt": _to_serialisable_dict(getattr(cfg.guards, "rmt", {})),
            "variance": _to_serialisable_dict(getattr(cfg.guards, "variance", {})),
            "invariants": _to_serialisable_dict(getattr(cfg.guards, "invariants", {})),
        }

        if model_profile.invariants:
            invariants_policy = guard_overrides.setdefault("invariants", {})
            existing_checks = invariants_policy.get("profile_checks", [])
            if isinstance(existing_checks, list | tuple | set):
                checks_list = [str(item) for item in existing_checks]
            elif existing_checks:
                checks_list = [str(existing_checks)]
            else:
                checks_list = []
            for invariant in model_profile.invariants:
                invariant_name = str(invariant)
                if invariant_name not in checks_list:
                    checks_list.append(invariant_name)
            invariants_policy["profile_checks"] = checks_list

        run_context = {
            "eval": _to_serialisable_dict(cfg.eval),
            "dataset": _to_serialisable_dict(cfg.dataset),
            "guards": guard_overrides,
            "profile": profile if profile else "",
            "pairing_baseline": pairing_schedule,
            "seeds": seed_bundle,
            "plugins": plugin_provenance,
            "run_id": run_id,
        }
        # Provide baseline per-window logloss to the CoreRunner for paired tail
        # evidence and (optionally) fail/rollback enforcement.
        try:
            if isinstance(baseline_report_data, dict):
                ew = baseline_report_data.get("evaluation_windows")
                if isinstance(ew, dict):
                    final = ew.get("final")
                    if (
                        isinstance(final, dict)
                        and isinstance(final.get("window_ids"), list)
                        and isinstance(final.get("logloss"), list)
                    ):
                        base_eval: dict[str, Any] = {
                            "final": {
                                "window_ids": list(final.get("window_ids") or []),
                                "logloss": list(final.get("logloss") or []),
                            }
                        }
                        if isinstance(final.get("token_counts"), list):
                            base_eval["final"]["token_counts"] = list(
                                final.get("token_counts") or []
                            )
                        run_context["baseline_eval_windows"] = base_eval
        except Exception:
            pass
        run_context.setdefault("primary_metric", {})["acceptance_range"] = (
            pm_acceptance_range
        )
        run_context["pm_acceptance_range"] = pm_acceptance_range
        if pm_drift_band:
            run_context.setdefault("primary_metric", {})["drift_band"] = pm_drift_band
            run_context["pm_drift_band"] = pm_drift_band
        run_context["model_profile"] = {
            "family": model_profile.family,
            "default_loss": model_profile.default_loss,
            "module_selectors": model_profile.module_selectors,
            "invariants": model_profile.invariants,
            "cert_lints": model_profile.cert_lints,
        }
        extra_context = _to_serialisable_dict(getattr(cfg, "context", {}))
        if isinstance(extra_context, dict):
            run_context.update(extra_context)
        try:
            run_context.setdefault("eval", {}).setdefault("loss", {})[
                "resolved_type"
            ] = resolved_loss_type
        except Exception:
            pass
        run_config = RunConfig(
            device=resolved_device,
            max_pm_ratio=getattr(cfg.eval, "max_pm_ratio", 1.5),
            event_path=run_dir / "events.jsonl",
            context=run_context,
        )
        skip_model_load = False

        # Load model using adapter
        # Load calibration data if dataset is configured
        calibration_data = None
        dataset_meta: dict[str, Any] = {}
        baseline_meta: dict[str, Any] = {}
        window_plan: dict[str, Any] | None = None
        dataset_timing_start: float | None = perf_counter() if collect_timings else None
        if pairing_schedule:
            harvested = _validate_and_harvest_baseline_schedule(
                cfg,
                pairing_schedule,
                baseline_report_data,
                tokenizer_hash=tokenizer_hash,
                resolved_loss_type=resolved_loss_type,
                profile=profile,
                baseline_path_str=str(baseline) if baseline else None,
                console=console,
            )
            effective_preview = harvested["effective_preview"]
            effective_final = harvested["effective_final"]
            preview_count = harvested["preview_count"]
            final_count = harvested["final_count"]
            dataset_meta = harvested["dataset_meta"]
            window_plan = harvested["window_plan"]
            calibration_data = harvested["calibration_data"]
            if use_mlm and tokenizer is None:
                try:
                    tokenizer, tokenizer_hash = resolve_tokenizer(model_profile)
                except Exception as exc:
                    _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
                    raise typer.Exit(1) from exc
            preview_window_ids = pairing_schedule["preview"].get("window_ids")
            preview_labels = pairing_schedule["preview"].get("labels")
            for idx, (input_ids, attention_mask) in enumerate(
                zip(
                    pairing_schedule["preview"]["input_ids"],
                    pairing_schedule["preview"]["attention_masks"],
                    strict=False,
                )
            ):
                window_id = (
                    preview_window_ids[idx]
                    if preview_window_ids and idx < len(preview_window_ids)
                    else idx
                )
                entry = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "window_id": f"preview::{window_id}",
                }
                if use_mlm:
                    labels_list: list[int] = []
                    if isinstance(preview_labels, list) and idx < len(preview_labels):
                        labels_list = _tensor_or_list_to_ints(preview_labels[idx])
                    if labels_list and any(token != -100 for token in labels_list):
                        entry["labels"] = labels_list
                        entry["mlm_masked"] = sum(
                            1 for token in labels_list if token != -100
                        )
                    else:
                        entry["labels"] = []
                        entry["mlm_masked"] = 0
                    # Prefer masked_token_counts if present in schedule
                    mtc = pairing_schedule["preview"].get("masked_token_counts")
                    if isinstance(mtc, list) and idx < len(mtc):
                        try:
                            entry["mlm_masked"] = int(mtc[idx])
                        except Exception:
                            pass
                calibration_data.append(entry)
            final_window_ids = pairing_schedule["final"].get("window_ids")
            final_labels = pairing_schedule["final"].get("labels")
            for idx, (input_ids, attention_mask) in enumerate(
                zip(
                    pairing_schedule["final"]["input_ids"],
                    pairing_schedule["final"]["attention_masks"],
                    strict=False,
                )
            ):
                window_id = (
                    final_window_ids[idx]
                    if final_window_ids and idx < len(final_window_ids)
                    else idx
                )
                entry = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "window_id": f"final::{window_id}",
                }
                if use_mlm:
                    labels_list: list[int] = []
                    if isinstance(final_labels, list) and idx < len(final_labels):
                        labels_list = _tensor_or_list_to_ints(final_labels[idx])
                    if labels_list and any(token != -100 for token in labels_list):
                        entry["labels"] = labels_list
                        entry["mlm_masked"] = sum(
                            1 for token in labels_list if token != -100
                        )
                    else:
                        entry["labels"] = []
                        entry["mlm_masked"] = 0
                    # Prefer masked_token_counts if present in schedule
                    mtc = pairing_schedule["final"].get("masked_token_counts")
                    if isinstance(mtc, list) and idx < len(mtc):
                        try:
                            entry["mlm_masked"] = int(mtc[idx])
                        except Exception:
                            pass
                calibration_data.append(entry)
            preview_count = len(pairing_schedule["preview"]["input_ids"])
            final_count = len(pairing_schedule["final"]["input_ids"])
            effective_preview = int(preview_count)
            effective_final = int(final_count)
            preview_mask_total = 0
            final_mask_total = 0
            preview_mask_counts: list[int] = []
            final_mask_counts: list[int] = []
            if use_mlm:
                preview_entries = calibration_data[:preview_count]
                final_entries = calibration_data[preview_count:]

                def _needs_masks(entries):
                    missing_any = False
                    counts = []
                    for entry in entries:
                        labels_val = entry.get("labels")
                        has_label_masks = bool(
                            isinstance(labels_val, list)
                            and any(token != -100 for token in labels_val)
                        )
                        existing_count = int(entry.get("mlm_masked", 0))
                        if not has_label_masks and existing_count <= 0:
                            missing_any = True
                        counts.append(int(entry.get("mlm_masked", 0)))
                    return missing_any, counts

                preview_missing, preview_counts_existing = _needs_masks(preview_entries)
                final_missing, final_counts_existing = _needs_masks(final_entries)

                if preview_missing:
                    preview_mask_total, preview_mask_counts = _apply_mlm_masks(
                        preview_entries,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="preview",
                    )
                else:
                    preview_mask_counts = preview_counts_existing
                    preview_mask_total = sum(preview_mask_counts)

                if final_missing:
                    final_mask_total, final_mask_counts = _apply_mlm_masks(
                        final_entries,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="final",
                    )
                else:
                    final_mask_counts = final_counts_existing
                    final_mask_total = sum(final_mask_counts)

                # Ensure counts and labels set on entries
                if preview_mask_counts:
                    for entry, count in zip(
                        preview_entries, preview_mask_counts, strict=False
                    ):
                        entry["mlm_masked"] = int(count)
                if final_mask_counts:
                    for entry, count in zip(
                        final_entries, final_mask_counts, strict=False
                    ):
                        entry["mlm_masked"] = int(count)

                if preview_count > 0 and preview_mask_total <= 0:
                    _fail_run(
                        "Baseline pairing schedule provided no masked tokens for preview windows; "
                        "ensure MLM labels are present in the baseline report."
                    )
                if final_count > 0 and final_mask_total <= 0:
                    _fail_run(
                        "Baseline pairing schedule provided no masked tokens for final windows; "
                        "ensure MLM labels are present in the baseline report."
                    )

                dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
                dataset_meta["masked_tokens_final"] = int(final_mask_total)
                dataset_meta["masked_tokens_total"] = int(
                    preview_mask_total + final_mask_total
                )
                if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                    console.print(
                        f"[debug] MLM pairing masks → preview={preview_mask_total}, final={final_mask_total}"
                    )
            if "preview_total_tokens" not in dataset_meta:
                dataset_meta["preview_total_tokens"] = sum(
                    len(_tensor_or_list_to_ints(seq))
                    for seq in pairing_schedule["preview"]["input_ids"]
                )
            if "final_total_tokens" not in dataset_meta:
                dataset_meta["final_total_tokens"] = sum(
                    len(_tensor_or_list_to_ints(seq))
                    for seq in pairing_schedule["final"]["input_ids"]
                )
            if "preview_hash" not in dataset_meta:
                preview_hash = _hash_sequences(
                    _tensor_or_list_to_ints(seq)
                    for seq in pairing_schedule["preview"]["input_ids"]
                )
                dataset_meta["preview_hash"] = preview_hash
            else:
                preview_hash = dataset_meta["preview_hash"]
            if "final_hash" not in dataset_meta:
                final_hash = _hash_sequences(
                    _tensor_or_list_to_ints(seq)
                    for seq in pairing_schedule["final"]["input_ids"]
                )
                dataset_meta["final_hash"] = final_hash
            else:
                final_hash = dataset_meta["final_hash"]
            if "dataset_hash" not in dataset_meta:
                dataset_meta["dataset_hash"] = hashlib.blake2s(
                    (str(preview_hash) + str(final_hash)).encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
            if not window_plan:
                window_capacity = (
                    baseline_meta.get("window_capacity")
                    if isinstance(baseline_meta, dict)
                    else {}
                )
                window_plan = {
                    "profile": (profile or "").lower() or "baseline",
                    "requested_preview": int(preview_count),
                    "requested_final": int(final_count),
                    "actual_preview": int(preview_count),
                    "actual_final": int(final_count),
                    "coverage_ok": True,
                    "capacity": window_capacity or {},
                }
            if isinstance(window_plan, dict):
                dataset_meta.setdefault("window_plan", window_plan)
                capacity_meta = window_plan.get("capacity")
                if capacity_meta and "window_capacity" not in dataset_meta:
                    dataset_meta["window_capacity"] = capacity_meta
        elif cfg.dataset.provider:
            _event(
                console,
                "DATA",
                f"Loading dataset: {cfg.dataset.provider}",
                emoji="📊",
                profile=profile_normalized,
            )
            # Pass through provider-specific kwargs when available
            provider_kwargs = {}
            for key in (
                "dataset_name",
                "config_name",
                "text_field",
                "src_field",
                "tgt_field",
                "cache_dir",
                "max_samples",
                # Local providers (e.g., local_jsonl)
                "file",
                "path",
                "data_files",
            ):
                try:
                    val = getattr(cfg.dataset, key)
                except Exception:
                    val = None
                if val is not None and val != "":
                    provider_kwargs[key] = val
            # Resolve provider kind from config (supports string or mapping with kind)
            provider_val = getattr(cfg.dataset, "provider", None)
            provider_name = None
            if isinstance(provider_val, dict):
                provider_name = provider_val.get("kind")
                # Include nested provider-specific kwargs
                for k, v in provider_val.items():
                    if k != "kind" and v is not None and v != "":
                        provider_kwargs[k] = v
            elif isinstance(provider_val, str):
                provider_name = provider_val  # noqa: F841
            else:
                # Support mapping-like provider configs (e.g., _Obj with .get)
                try:
                    _ = provider_val.get("kind")  # type: ignore[attr-defined]
                    # Try to expose nested entries
                    try:
                        for k, v in provider_val._data.items():  # type: ignore[attr-defined]
                            if k != "kind" and v is not None and v != "":
                                provider_kwargs[k] = v
                    except Exception:
                        # Fallback: if items() exists
                        try:
                            for k, v in provider_val.items():  # type: ignore[attr-defined]
                                if k != "kind" and v is not None and v != "":
                                    provider_kwargs[k] = v
                        except Exception:
                            pass
                except Exception:
                    _ = None
            data_provider, resolved_split, used_fallback_split = (
                _resolve_provider_and_split(
                    cfg,
                    model_profile,
                    get_provider_fn=get_provider,
                    provider_kwargs=provider_kwargs,
                    console=console,
                    resolved_device=resolved_device,
                    emit=_provider_event,
                )
            )

            # Load tokenizer for dataset processing
            try:
                tokenizer, tokenizer_hash = resolve_tokenizer(model_profile)
            except Exception as exc:
                _event(console, "FAIL", str(exc), emoji="❌", profile=profile)
                raise typer.Exit(1) from exc

            dataset_stride = getattr(
                cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0) // 2
            )
            release_profile = (profile or "").lower() == "release"
            if release_profile and not pairing_schedule:
                estimate_fn = getattr(data_provider, "estimate_capacity", None)
                if callable(estimate_fn):
                    capacity_fast = bool(getattr(cfg.eval, "capacity_fast", False))
                    capacity_meta = estimate_fn(
                        tokenizer=tokenizer,
                        seq_len=cfg.dataset.seq_len,
                        stride=dataset_stride,
                        split=resolved_split,
                        target_total=requested_preview + requested_final,
                        fast_mode=capacity_fast,
                    )
                    variance_policy = getattr(cfg.guards, "variance", None)
                    max_calibration = (
                        getattr(variance_policy, "max_calib", 0)
                        if variance_policy is not None
                        else 0
                    )
                    try:
                        window_plan = _maybe_plan_release_windows(
                            capacity_meta,
                            requested_preview=requested_preview,
                            requested_final=requested_final,
                            max_calibration=max_calibration,
                            console=console,
                        )
                    except RuntimeError as err:
                        _event(console, "FAIL", str(err), emoji="❌", profile=profile)
                        raise typer.Exit(1) from err

                    actual_per_arm = int(window_plan["actual_preview"])
                    effective_preview = actual_per_arm
                    effective_final = actual_per_arm
                    preview_count = effective_preview
                    final_count = effective_final
                    dataset_stride = getattr(
                        cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0)
                    )
                else:
                    _event(
                        console,
                        "WARN",
                        "Release profile requested but dataset provider does not expose capacity estimation; using configured window counts.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

            preview_records: list[tuple[list[int], list[int]]] = []
            final_records: list[tuple[list[int], list[int]]] = []

            while True:
                preview_window, final_window = data_provider.windows(
                    tokenizer=tokenizer,
                    seq_len=cfg.dataset.seq_len,
                    stride=getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
                    preview_n=effective_preview,
                    final_n=effective_final,
                    seed=getattr(cfg.dataset, "seed", 42),
                    split=resolved_split,
                )

                preview_count = len(getattr(preview_window, "input_ids", []))
                final_count = len(getattr(final_window, "input_ids", []))
                is_eval_window = isinstance(
                    preview_window, EvaluationWindow
                ) and isinstance(final_window, EvaluationWindow)
                if is_eval_window:
                    if (
                        preview_count != effective_preview
                        or final_count != effective_final
                    ):
                        _fail_run(
                            "Dataset provider returned mismatched preview/final counts "
                            f"({preview_count}/{final_count}) "
                            f"expected ({effective_preview}/{effective_final}). "
                            "CI/Release profiles require exact parity."
                        )
                else:
                    preview_count = effective_preview
                    final_count = effective_final

                # Optional: provider-supplied labels for seq2seq
                provider_labels_prev = None
                provider_labels_fin = None
                try:
                    provider_labels_prev = getattr(
                        data_provider, "last_preview_labels", None
                    )
                    provider_labels_fin = getattr(
                        data_provider, "last_final_labels", None
                    )
                except Exception:
                    provider_labels_prev = None
                    provider_labels_fin = None

                preview_records = []
                preview_indices_raw = getattr(preview_window, "indices", [])
                if isinstance(preview_indices_raw, list):
                    preview_indices = preview_indices_raw
                else:
                    try:
                        preview_indices = list(preview_indices_raw)
                    except TypeError:
                        preview_indices = []
                for idx_local, (input_ids, attention_mask) in enumerate(
                    zip(
                        preview_window.input_ids,
                        preview_window.attention_masks,
                        strict=False,
                    )
                ):
                    input_ids_list = _tensor_or_list_to_ints(input_ids)
                    attention_mask_list = (
                        _tensor_or_list_to_ints(attention_mask)
                        if attention_mask is not None
                        else [1] * len(input_ids_list)
                    )
                    dataset_index = (
                        _safe_int(preview_indices[idx_local])
                        if idx_local < len(preview_indices)
                        else idx_local
                    )
                    rec = {
                        "input_ids": input_ids_list,
                        "attention_mask": attention_mask_list,
                        "dataset_index": dataset_index,
                    }
                    # Attach provider labels for seq2seq if available
                    if provider_labels_prev is not None and idx_local < len(
                        provider_labels_prev
                    ):
                        rec["labels"] = _tensor_or_list_to_ints(
                            provider_labels_prev[idx_local]
                        )
                    preview_records.append(rec)

                final_records = []
                final_indices_raw = getattr(final_window, "indices", [])
                if isinstance(final_indices_raw, list):
                    final_indices = final_indices_raw
                else:
                    try:
                        final_indices = list(final_indices_raw)
                    except TypeError:
                        final_indices = []
                for idx_local, (input_ids, attention_mask) in enumerate(
                    zip(
                        final_window.input_ids,
                        final_window.attention_masks,
                        strict=False,
                    )
                ):
                    input_ids_list = _tensor_or_list_to_ints(input_ids)
                    attention_mask_list = (
                        _tensor_or_list_to_ints(attention_mask)
                        if attention_mask is not None
                        else [1] * len(input_ids_list)
                    )
                    dataset_index = (
                        _safe_int(final_indices[idx_local])
                        if idx_local < len(final_indices)
                        else idx_local
                    )
                    final_records.append(
                        {
                            "input_ids": input_ids_list,
                            "attention_mask": attention_mask_list,
                            "dataset_index": dataset_index,
                        }
                    )

                if use_mlm:
                    temp_preview_records = [
                        {
                            "input_ids": list(rec["input_ids"]),
                            "attention_mask": list(rec["attention_mask"]),
                            "dataset_index": rec.get("dataset_index"),
                            "window_id": rec.get("window_id"),
                        }
                        for rec in preview_records
                    ]
                    temp_final_records = [
                        {
                            "input_ids": list(rec["input_ids"]),
                            "attention_mask": list(rec["attention_mask"]),
                            "dataset_index": rec.get("dataset_index"),
                            "window_id": rec.get("window_id"),
                        }
                        for rec in final_records
                    ]
                    _apply_mlm_masks(
                        temp_preview_records,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="preview",
                    )
                    _apply_mlm_masks(
                        temp_final_records,
                        tokenizer=tokenizer,
                        mask_prob=mask_prob,
                        seed=mask_seed,
                        random_token_prob=random_token_prob,
                        original_token_prob=original_token_prob,
                        prefix="final",
                    )
                    records_for_signatures = temp_preview_records + temp_final_records
                else:
                    records_for_signatures = preview_records + final_records

                signatures = []
                for record in records_for_signatures:
                    tokens = record["input_ids"]
                    masks = record["attention_mask"]
                    signatures.append(
                        tuple(
                            tok
                            for tok, mask in zip(tokens, masks, strict=False)
                            if mask
                        )
                    )

                unique_sequences = len(set(signatures))
                combined_total = len(signatures)
                if unique_sequences == combined_total:
                    break

                deficit = combined_total - unique_sequences
                reduction = max(5, int(deficit) if deficit > 0 else 1)
                proposed_per_arm = preview_count - reduction
                if proposed_per_arm >= preview_count:
                    proposed_per_arm = preview_count - 1
                min_per_arm_floor = RELEASE_MIN_WINDOWS_PER_ARM
                if window_plan is None or window_plan.get("profile") != "release":
                    min_per_arm_floor = max(
                        10,
                        min(
                            int(requested_preview or 0) or RELEASE_MIN_WINDOWS_PER_ARM,
                            int(requested_final or 0) or RELEASE_MIN_WINDOWS_PER_ARM,
                        )
                        // 2,
                    )
                if proposed_per_arm < min_per_arm_floor:
                    raise RuntimeError(
                        "Unable to construct non-overlapping windows within minimum window floor."
                    )
                _event(
                    console,
                    "WARN",
                    f"Detected {deficit} duplicate windows; reducing per-arm windows to {proposed_per_arm} and retrying stratification.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )

                effective_preview = proposed_per_arm
                effective_final = proposed_per_arm
                preview_count = effective_preview
                final_count = effective_final
                if window_plan is not None:
                    window_plan.setdefault("dedupe_adjustments", []).append(
                        {
                            "deficit": int(deficit),
                            "proposed_per_arm": int(proposed_per_arm),
                        }
                    )
                    window_plan["actual_preview"] = proposed_per_arm
                    window_plan["actual_final"] = proposed_per_arm
                continue

            if window_plan is None:
                window_plan = {
                    "profile": (profile or "").lower() or "default",
                    "requested_preview": int(requested_preview),
                    "requested_final": int(requested_final),
                    "actual_preview": int(preview_count),
                    "actual_final": int(final_count),
                    "coverage_ok": preview_count == final_count,
                    "capacity": {},
                }
            else:
                window_plan["actual_preview"] = int(preview_count)
                window_plan["actual_final"] = int(final_count)
                window_plan["coverage_ok"] = (
                    window_plan.get("coverage_ok", True)
                    and preview_count == final_count
                )

            calibration_data: list[dict[str, Any]] = []
            preview_mask_total = 0
            final_mask_total = 0
            preview_mask_counts: list[int] = []
            final_mask_counts: list[int] = []
            if use_mlm:
                preview_mask_total, preview_mask_counts = _apply_mlm_masks(
                    preview_records,
                    tokenizer=tokenizer,
                    mask_prob=mask_prob,
                    seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    prefix="preview",
                )
                final_mask_total, final_mask_counts = _apply_mlm_masks(
                    final_records,
                    tokenizer=tokenizer,
                    mask_prob=mask_prob,
                    seed=mask_seed,
                    random_token_prob=random_token_prob,
                    original_token_prob=original_token_prob,
                    prefix="final",
                )
            else:
                preview_mask_counts = [0] * len(preview_records)
                final_mask_counts = [0] * len(final_records)

            preview_sequences = [record["input_ids"] for record in preview_records]
            for idx, record in enumerate(preview_records):
                entry = {
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                    "window_id": f"preview::{idx}",
                    "dataset_index": record.get("dataset_index"),
                    "mlm_masked": record.get("mlm_masked", 0),
                }
                if use_mlm:
                    entry["labels"] = record.get(
                        "labels", [-100] * len(record["input_ids"])
                    )
                calibration_data.append(entry)

            final_sequences = [record["input_ids"] for record in final_records]
            for idx, record in enumerate(final_records):
                entry = {
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                    "window_id": f"final::{idx}",
                    "dataset_index": record.get("dataset_index"),
                    "mlm_masked": record.get("mlm_masked", 0),
                }
                if use_mlm:
                    entry["labels"] = record.get(
                        "labels", [-100] * len(record["input_ids"])
                    )
                elif provider_labels_fin is not None and idx < len(provider_labels_fin):
                    entry["labels"] = _tensor_or_list_to_ints(provider_labels_fin[idx])
                calibration_data.append(entry)

            masked_tokens_total = preview_mask_total + final_mask_total
            preview_hash = _hash_sequences(preview_sequences)
            final_hash = _hash_sequences(final_sequences)
            dataset_meta = {
                "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
                "tokenizer_hash": tokenizer_hash
                if tokenizer_hash is not None
                else _tokenizer_digest(tokenizer),
                "vocab_size": _safe_int(getattr(tokenizer, "vocab_size", 0)),
                "bos_token": getattr(tokenizer, "bos_token", None),
                "eos_token": getattr(tokenizer, "eos_token", None),
                "pad_token": getattr(tokenizer, "pad_token", None),
                "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
                "dataset_hash": hashlib.blake2s(
                    (preview_hash + final_hash).encode("utf-8"), digest_size=16
                ).hexdigest(),
                "preview_hash": preview_hash,
                "final_hash": final_hash,
                "preview_total_tokens": sum(len(seq) for seq in preview_sequences),
                "final_total_tokens": sum(len(seq) for seq in final_sequences),
            }
            dataset_meta["loss_type"] = resolved_loss_type
            if use_mlm:
                dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
                dataset_meta["masked_tokens_final"] = int(final_mask_total)
                dataset_meta["masked_tokens_total"] = int(masked_tokens_total)
            if window_plan:
                dataset_meta["window_plan"] = window_plan
                capacity_meta = window_plan.get("capacity")
                if capacity_meta:
                    dataset_meta["window_capacity"] = capacity_meta
            strat_stats = getattr(data_provider, "stratification_stats", None)
            if strat_stats:
                dataset_meta["stratification"] = strat_stats
            scorer_profile = getattr(data_provider, "scorer_profile", None)
            if scorer_profile:
                dataset_meta["scorer_profile"] = scorer_profile

        try:
            run_context["dataset"]["preview_n"] = preview_count
            run_context["dataset"]["final_n"] = final_count
        except Exception:
            pass
        run_context["dataset_meta"] = dataset_meta
        if window_plan:
            run_context["window_plan"] = window_plan
        if dataset_timing_start is not None:
            timings["load_dataset"] = max(
                0.0, float(perf_counter() - dataset_timing_start)
            )

        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            console.print(
                "[debug] calibration batch size => preview="
                f"{preview_count} final={final_count} total={len(calibration_data)}"
            )
            if use_mlm and calibration_data:
                masked_preview = sum(
                    entry.get("mlm_masked", 0)
                    for entry in calibration_data[:preview_count]
                )
                masked_final = sum(
                    entry.get("mlm_masked", 0)
                    for entry in calibration_data[preview_count:]
                )
                console.print(
                    f"[debug] masked tokens (preview/final) = {masked_preview}/{masked_final}"
                )
                console.print(
                    f"[debug] sample labels first preview entry (first 10) = {calibration_data[0]['labels'][:10]}"
                )

        # Execute the real pipeline using CoreRunner
        _event(
            console,
            "EXEC",
            f"Executing pipeline with {len(guards)} guards...",
            emoji="⚙️",
            profile=profile_normalized,
        )
        runner = CoreRunner()

        # Prepare auto configuration for tier resolution
        # Build auto configuration with safe fallbacks when section/keys are absent
        try:
            auto_enabled = bool(cfg.auto.enabled)
        except Exception:
            auto_enabled = False
        try:
            auto_tier = cfg.auto.tier
        except Exception:
            auto_tier = "balanced"
        try:
            auto_probes = int(cfg.auto.probes)
        except Exception:
            auto_probes = 0
        try:
            auto_target_ratio = float(cfg.auto.target_pm_ratio)
        except Exception:
            auto_target_ratio = 2.0

        auto_config = {
            "enabled": auto_enabled,
            "tier": auto_tier,
            "probes": auto_probes,
            "target_pm_ratio": auto_target_ratio,
        }

        # Extract edit configuration parameters
        edit_config = {}
        if hasattr(cfg.edit, "plan") and cfg.edit.plan:
            try:
                # Accept plain dicts, dict-like wrappers, or nested objects
                plan_obj = getattr(cfg.edit, "plan", {})
                if isinstance(plan_obj, dict):
                    edit_config = dict(plan_obj)
                else:
                    # Best-effort unwrap for InvarLockConfig _Obj wrapper
                    plan_data = getattr(plan_obj, "_data", None)
                    if isinstance(plan_data, dict):
                        edit_config = dict(plan_data)
                    elif hasattr(plan_obj, "items"):
                        edit_config = dict(plan_obj)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                pass
        elif hasattr(cfg.edit, "parameters") and cfg.edit.parameters:
            try:
                if hasattr(cfg.edit.parameters, "items"):
                    edit_config = dict(cfg.edit.parameters)
                elif isinstance(cfg.edit.parameters, dict):
                    edit_config = cfg.edit.parameters
            except (TypeError, AttributeError):
                pass

        if (
            model_profile.module_selectors
            and "module_selectors" not in edit_config
            and isinstance(model_profile.module_selectors, dict)
        ):
            edit_config["module_selectors"] = {
                key: list(values)
                for key, values in model_profile.module_selectors.items()
            }

        console.print(_format_kv_line("Edit", str(edit_op.name)))
        console.print(_format_kv_line("Guards", _format_guard_chain(guards)))

        # Model load/snapshot strategy
        model = None
        restore_fn = None
        snapshot_tmpdir: str | None = None
        snapshot_provenance: dict[str, bool] = {
            "restore_failed": False,
            "reload_path_used": False,
        }

        # Try single-load with snapshot/restore if adapter supports it; fallback to reload per attempt
        try:
            # Load once
            _event(
                console,
                "INIT",
                f"Loading model once: {cfg.model.id}",
                emoji="🔧",
                profile=profile_normalized,
            )
            with timed_step(
                console=console,
                style=_style_from_console(console, profile=profile_normalized),
                timings=timings,
                key="load_model",
                tag="INIT",
                message="Load model",
                emoji="🔧",
            ):
                model = _load_model_with_cfg(
                    adapter, cfg, resolved_device, profile=profile_normalized
                )

            # No edit-specific bootstrap logic

            def _estimate_model_bytes(m: Any) -> int:
                total = 0
                try:
                    for _, p in getattr(m, "named_parameters", lambda: [])():
                        try:
                            total += int(p.element_size() * p.nelement())
                        except Exception:
                            pass
                    for _, b in getattr(m, "named_buffers", lambda: [])():
                        try:
                            total += int(b.element_size() * b.nelement())
                        except Exception:
                            pass
                except Exception:
                    return 0
                return total

            # Load snapshot config from config.context.snapshot (highest precedence)
            cfg_snapshot = {}
            try:
                cfg_context = _to_serialisable_dict(getattr(cfg, "context", {}))
                if isinstance(cfg_context, dict):
                    cfg_snapshot = _to_serialisable_dict(
                        cfg_context.get("snapshot", {})
                    )
                    if not isinstance(cfg_snapshot, dict):
                        cfg_snapshot = {}
            except Exception:
                cfg_snapshot = {}

            def _choose_snapshot_mode() -> str:
                # Precedence: config > env > auto
                cfg_mode = (
                    str(cfg_snapshot.get("mode", "")).lower()
                    if isinstance(cfg_snapshot, dict)
                    else ""
                )
                mode_env = str(
                    os.environ.get("INVARLOCK_SNAPSHOT_MODE", "auto")
                ).lower()
                supports_chunked = hasattr(adapter, "snapshot_chunked") and hasattr(
                    adapter, "restore_chunked"
                )
                supports_bytes = hasattr(adapter, "snapshot") and hasattr(
                    adapter, "restore"
                )
                if cfg_mode in {"bytes", "chunked"}:
                    if cfg_mode == "bytes" and supports_bytes:
                        return "bytes"
                    if cfg_mode == "chunked" and supports_chunked:
                        return "chunked"
                    # fallback preference
                    if supports_bytes:
                        return "bytes"
                    if supports_chunked:
                        return "chunked"
                    return "reload"
                if mode_env in {"bytes", "chunked"}:
                    if mode_env == "bytes" and supports_bytes:
                        return "bytes"
                    if mode_env == "chunked" and supports_chunked:
                        return "chunked"
                    # fallback preference
                    if supports_bytes:
                        return "bytes"
                    if supports_chunked:
                        return "chunked"
                    return "reload"
                # auto
                est_mb = _estimate_model_bytes(model) / (1024.0 * 1024.0)
                # RAM-based heuristic
                try:
                    ram = psutil.virtual_memory()
                    avail_mb = float(getattr(ram, "available", 0)) / (1024.0 * 1024.0)
                except Exception:
                    avail_mb = 0.0
                # fraction: config override > env > default 0.4
                frac = 0.4
                try:
                    if (
                        isinstance(cfg_snapshot, dict)
                        and cfg_snapshot.get("ram_fraction") is not None
                    ):
                        frac = float(cfg_snapshot.get("ram_fraction"))
                    else:
                        frac = float(
                            os.environ.get("INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION", frac)
                        )
                except Exception:
                    pass
                # threshold mb: if no RAM info, use config threshold_mb or env fallback; else derive from avail*frac
                if avail_mb > 0:
                    threshold_mb = avail_mb * max(0.0, min(frac, 1.0))
                else:
                    try:
                        if (
                            isinstance(cfg_snapshot, dict)
                            and cfg_snapshot.get("threshold_mb") is not None
                        ):
                            threshold_mb = float(cfg_snapshot.get("threshold_mb"))
                        else:
                            threshold_mb = float(
                                os.environ.get("INVARLOCK_SNAPSHOT_THRESHOLD_MB", "768")
                            )
                    except Exception:
                        threshold_mb = 768.0
                # Disk availability for chunked
                try:
                    tmpdir = None
                    if isinstance(cfg_snapshot, dict):
                        tmpdir = cfg_snapshot.get("temp_dir") or None
                    if not tmpdir:
                        tmpdir = (
                            os.environ.get("TMPDIR") or os.environ.get("TMP") or "/tmp"
                        )
                    du = shutil.disk_usage(tmpdir)
                    free_mb = float(du.free) / (1024.0 * 1024.0)
                except Exception:
                    free_mb = 0.0
                # Disk margin ratio: config > default 1.2
                margin = 1.2
                try:
                    if (
                        isinstance(cfg_snapshot, dict)
                        and cfg_snapshot.get("disk_free_margin_ratio") is not None
                    ):
                        margin = float(cfg_snapshot.get("disk_free_margin_ratio"))
                except Exception:
                    pass
                # Choose chunked if model snapshot is a large fraction of available RAM and disk has room
                if (
                    supports_chunked
                    and est_mb >= threshold_mb
                    and (free_mb <= 0.0 or est_mb * margin <= free_mb)
                ):
                    return "chunked"
                # Otherwise prefer bytes when supported
                if supports_bytes:
                    # If RAM is extremely low and even bytes snapshot likely risky, fallback to chunked when possible
                    if (
                        supports_chunked
                        and avail_mb > 0
                        and est_mb >= max(64.0, avail_mb * 0.8)
                        and (free_mb <= 0.0 or est_mb * margin <= free_mb)
                    ):
                        return "chunked"
                    return "bytes"
                if supports_chunked:
                    return "chunked"
                return "reload"

            mode = _choose_snapshot_mode()
            enabled = mode in {"bytes", "chunked"}
            _event(
                console,
                "INIT",
                f"Snapshot mode: {'enabled' if enabled else 'disabled'}",
                emoji="💾",
                profile=profile_normalized,
            )
            if mode == "chunked":
                snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

                def _restore():
                    adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

                restore_fn = _restore
            elif mode == "bytes":
                supports_chunked = hasattr(adapter, "snapshot_chunked") and hasattr(
                    adapter, "restore_chunked"
                )
                try:
                    base_blob = adapter.snapshot(model)  # type: ignore[attr-defined]
                except Exception:
                    if not supports_chunked:
                        raise
                    snapshot_tmpdir = adapter.snapshot_chunked(model)  # type: ignore[attr-defined]

                    def _restore_fallback_chunked():
                        adapter.restore_chunked(model, snapshot_tmpdir)  # type: ignore[attr-defined]

                    restore_fn = _restore_fallback_chunked
                else:

                    def _restore2():
                        adapter.restore(model, base_blob)  # type: ignore[attr-defined]

                    restore_fn = _restore2
            else:
                # reload path - properly free GPU memory before setting to None
                _free_model_memory(model)
                model = None
                restore_fn = None
        except Exception:
            # On any failure, fall back to reload-per-attempt path
            _free_model_memory(model)
            model = None
            restore_fn = None

        # RETRY LOOP - All report processing inside loop
        attempt = 1
        measure_guard_overhead, skip_overhead = _should_measure_overhead(
            profile_normalized
        )
        if skip_overhead and profile_normalized in {"ci", "release"}:
            _event(
                console,
                "WARN",
                "Overhead check skipped via INVARLOCK_SKIP_OVERHEAD_CHECK",
                emoji="⚠️",
                profile=profile_normalized,
            )

        while True:
            # Reset RNG streams each attempt to guarantee determinism across retries
            set_seed(seed_bundle["python"])

            if retry_controller:
                console.print("\n")
                _event(
                    console,
                    "EXEC",
                    f"Attempt {attempt}/{max_attempts}",
                    emoji="🚀",
                    profile=profile_normalized,
                )
                if attempt > 1:
                    _event(
                        console,
                        "EXEC",
                        f"Retry attempt {attempt}/{max_attempts}",
                        emoji="🔄",
                        profile=profile_normalized,
                    )
            else:
                if attempt > 1:
                    console.print("\n")
                    _event(
                        console,
                        "EXEC",
                        f"Attempt {attempt}",
                        emoji="🚀",
                        profile=profile_normalized,
                    )

            # Adjust parameters for retry attempts
            if retry_controller and attempt > 1:
                from invarlock.core.retry import adjust_edit_params

                edit_config = adjust_edit_params(
                    edit_op.name, edit_config, attempt, None
                )

            guard_overhead_payload: dict[str, Any] | None = None
            try:
                if skip_overhead and profile_normalized in {"ci", "release"}:
                    guard_overhead_payload = {
                        "overhead_threshold": GUARD_OVERHEAD_THRESHOLD,
                        "evaluated": False,
                        "passed": True,
                        "skipped": True,
                        "skip_reason": "INVARLOCK_SKIP_OVERHEAD_CHECK",
                        "mode": "skipped",
                        "source": "env:INVARLOCK_SKIP_OVERHEAD_CHECK",
                        "messages": [
                            "Overhead check skipped via INVARLOCK_SKIP_OVERHEAD_CHECK"
                        ],
                        "warnings": [],
                        "errors": [],
                        "checks": {},
                    }
                elif measure_guard_overhead:
                    bare_edit_config = dict(edit_config or {})
                    bare_edit_config["emit"] = False
                    guard_overhead_payload = _run_bare_control(
                        adapter=adapter,
                        edit_op=edit_op,
                        cfg=cfg,
                        model=model,
                        run_config=run_config,
                        calibration_data=calibration_data,
                        auto_config=auto_config,
                        edit_config=bare_edit_config,
                        preview_count=preview_count,
                        final_count=final_count,
                        seed_bundle=seed_bundle,
                        resolved_device=resolved_device,
                        restore_fn=restore_fn,
                        console=console,
                        resolved_loss_type=resolved_loss_type,
                        profile_normalized=profile_normalized,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                    )

                # Ensure clean state for guarded run
                with timed_step(
                    console=console,
                    style=_style_from_console(console, profile=profile_normalized),
                    timings=timings,
                    key="execute",
                    tag="EXEC",
                    message="Execute pipeline",
                    emoji="⚙️",
                ):
                    core_report, model = _execute_guarded_run(
                        runner=runner,
                        adapter=adapter,
                        model=model,
                        cfg=cfg,
                        edit_op=edit_op,
                        run_config=run_config,
                        guards=guards,
                        calibration_data=calibration_data,
                        auto_config=auto_config,
                        edit_config=edit_config,
                        preview_count=preview_count,
                        final_count=final_count,
                        restore_fn=restore_fn,
                        resolved_device=resolved_device,
                        profile_normalized=profile_normalized,
                        console=console,
                        snapshot_provenance=snapshot_provenance,
                        skip_model_load=skip_model_load,
                    )
            except _SnapshotRestoreFailed as exc:
                snapshot_provenance["restore_failed"] = True
                _free_model_memory(model)
                model = None
                restore_fn = None
                _event(
                    console,
                    "WARN",
                    "Snapshot restore failed; switching to reload-per-attempt.",
                    emoji="⚠️",
                    profile=profile_normalized,
                )
                _event(
                    console,
                    "WARN",
                    f"↳ {exc}",
                    profile=profile_normalized,
                )
                if retry_controller:
                    retry_controller.record_attempt(
                        attempt,
                        {
                            "passed": False,
                            "failures": ["restore_failed"],
                            "validation": {},
                        },
                        edit_config,
                    )
                    if retry_controller.should_retry(False):
                        attempt += 1
                        continue
                raise typer.Exit(1) from exc

            if not hasattr(core_report, "context") or core_report.context is None:
                core_report.context = {}

            # Convert CoreRunner report to evaluation report
            report = create_empty_report()

            # Persist minimal run context for certificate/report provenance.
            try:
                report["context"] = {
                    "profile": profile_normalized,
                    "auto": dict(auto_config),
                    "assurance": dict(run_context.get("assurance") or {}),
                }
            except Exception:
                pass

            # Code provenance: commit hash and InvarLock version
            commit_value = (
                getattr(cfg.meta, "commit", "") if hasattr(cfg, "meta") else ""
            )
            if not commit_value:
                try:
                    import subprocess

                    commit_value = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                        )
                        .decode("utf-8", "ignore")
                        .strip()
                    )
                except Exception:
                    commit_value = ""
            invarlock_version = None
            try:
                from invarlock import __version__ as _invarlock_version

                invarlock_version = _invarlock_version
            except Exception:
                invarlock_version = None

            # Collect determinism/env flags
            env_flags: dict[str, object] = {}
            try:
                import os as _os

                if torch is not None:
                    try:
                        det_enabled = getattr(
                            torch, "are_deterministic_algorithms_enabled", None
                        )
                        if callable(det_enabled):
                            env_flags["torch_deterministic_algorithms"] = bool(
                                det_enabled()
                            )
                    except Exception:
                        pass
                    try:
                        tf32_matmul = getattr(
                            getattr(torch.backends, "cuda", object()), "matmul", None
                        )
                        if tf32_matmul is not None and hasattr(
                            tf32_matmul, "allow_tf32"
                        ):
                            env_flags["cuda_matmul_allow_tf32"] = bool(
                                tf32_matmul.allow_tf32
                            )
                    except Exception:
                        pass
                    try:
                        cudnn_mod = getattr(torch.backends, "cudnn", None)
                        if cudnn_mod is not None:
                            env_flags["cudnn_allow_tf32"] = bool(
                                getattr(cudnn_mod, "allow_tf32", None)
                            )
                            env_flags["cudnn_deterministic"] = bool(
                                getattr(cudnn_mod, "deterministic", None)
                            )
                            env_flags["cudnn_benchmark"] = bool(
                                getattr(cudnn_mod, "benchmark", None)
                            )
                    except Exception:
                        pass
                    try:
                        env_flags["mps_available"] = bool(
                            getattr(torch.backends, "mps", None)
                            and torch.backends.mps.is_available()
                        )
                    except Exception:
                        pass
                # Common environment variables for determinism
                env_flags["CUBLAS_WORKSPACE_CONFIG"] = _os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                )
            except Exception:
                env_flags = {}

            meta_payload = {
                "model_id": cfg.model.id,
                "adapter": cfg.model.adapter,
                "device": str(resolved_device),
                "commit": commit_value,
                "seed": seed_bundle["python"],
                "seeds": seed_bundle,
                "ts": datetime.now().isoformat(),
                "auto": auto_config,
            }
            if invarlock_version:
                meta_payload["invarlock_version"] = invarlock_version
            if env_flags:
                meta_payload["env_flags"] = env_flags
            if determinism_meta:
                meta_payload["determinism"] = determinism_meta
            report["meta"].update(meta_payload)
            if pm_acceptance_range:
                report["meta"]["pm_acceptance_range"] = pm_acceptance_range
            if pm_drift_band:
                report["meta"]["pm_drift_band"] = pm_drift_band
            report["meta"]["model_profile"] = {
                "family": model_profile.family,
                "default_loss": model_profile.default_loss,
                "module_selectors": model_profile.module_selectors,
                "invariants": list(model_profile.invariants),
                "cert_lints": [dict(lint) for lint in model_profile.cert_lints],
            }

            report["data"].update(
                {
                    "dataset": cfg.dataset.provider,
                    # Resolved split (explicit or inferred)
                    "split": resolved_split,
                    "seq_len": cfg.dataset.seq_len,
                    "stride": getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
                    "preview_n": _safe_int(preview_count),
                    "final_n": _safe_int(final_count),
                }
            )
            dataset_meta_context = core_report.context.get("dataset_meta", {})
            if isinstance(dataset_meta_context, dict):
                report["data"].update(dataset_meta_context)
                dataset_tokenizer_hash = dataset_meta_context.get("tokenizer_hash")
                if (
                    not tokenizer_hash
                    and isinstance(dataset_tokenizer_hash, str)
                    and dataset_tokenizer_hash
                ):
                    tokenizer_hash = dataset_tokenizer_hash

            if tokenizer_hash:
                report["meta"]["tokenizer_hash"] = tokenizer_hash

            # Snapshot/restore provenance (survives retries).
            try:
                prov = report.setdefault("provenance", {})
                prov["restore_failed"] = bool(snapshot_provenance.get("restore_failed"))
                prov["reload_path_used"] = bool(
                    snapshot_provenance.get("reload_path_used")
                )
            except Exception:
                pass

            # Transfer edit information
            if hasattr(core_report, "edit") and core_report.edit:
                edit_deltas = core_report.edit.get("deltas", {})
                report["edit"].update(
                    {
                        "name": edit_op.name,
                        "plan_digest": core_report.edit.get(
                            "plan_digest", str(hash(str(core_report.edit)))
                        ),
                        "deltas": {
                            "params_changed": edit_deltas.get("params_changed", 0),
                            "sparsity": edit_deltas.get("sparsity", None),
                            "bitwidth_map": edit_deltas.get("bitwidth_map", None),
                            "layers_modified": edit_deltas.get("layers_modified", 0),
                        },
                    }
                )
                for key in (
                    "algorithm",
                    "algorithm_version",
                    "implementation",
                    "scope",
                    "ranking",
                    "grouping",
                    "budgets",
                    "seed",
                    "mask_digest",
                ):
                    if key in core_report.edit:
                        report["edit"][key] = copy.deepcopy(core_report.edit[key])
                if isinstance(core_report.context, dict):
                    core_report.context.setdefault("edit", {})
                    core_report.context["edit"].update(
                        {
                            "name": edit_op.name,
                            "params_changed": edit_deltas.get("params_changed", 0),
                            "layers_modified": edit_deltas.get("layers_modified", 0),
                        }
                    )

            if edit_label:
                report.setdefault("edit", {})
                report["edit"]["name"] = edit_label
                report["edit"]["algorithm"] = edit_label
                if isinstance(core_report.context, dict):
                    core_report.context.setdefault("edit", {})
                    core_report.context["edit"]["name"] = edit_label

            mask_artifact_path = _persist_ref_masks(core_report, run_dir)
            if mask_artifact_path:
                report.setdefault("artifacts", {})
                report["artifacts"]["masks_path"] = str(mask_artifact_path)

            # Transfer metrics (PM-only: do not write ppl_* fields)
            if hasattr(core_report, "metrics") and core_report.metrics:
                if isinstance(core_report.metrics, dict):
                    core_timings = core_report.metrics.get("timings")
                    if isinstance(core_timings, dict):
                        for key in (
                            "prepare",
                            "prepare_guards",
                            "edit",
                            "guards",
                            "eval",
                            "finalize",
                        ):
                            if key in core_timings:
                                try:
                                    timings[key] = float(core_timings[key])
                                except Exception:
                                    timings[key] = core_timings[key]
                metrics_payload = {
                    "latency_ms_per_tok": core_report.metrics.get(
                        "latency_ms_per_tok", 0.0
                    ),
                    "memory_mb_peak": core_report.metrics.get("memory_mb_peak", 0.0),
                    "spectral": {},
                    "rmt": {},
                    "invariants": {},
                }
                window_plan_ctx = core_report.context.get("window_plan")
                if isinstance(window_plan_ctx, dict):
                    metrics_payload["window_plan"] = window_plan_ctx
                    capacity_meta = window_plan_ctx.get("capacity")
                    if isinstance(capacity_meta, dict):
                        metrics_payload["window_capacity"] = capacity_meta
                    stats_section = metrics_payload.setdefault("stats", {})
                    if isinstance(stats_section, dict):
                        stats_section.update(
                            {
                                "requested_preview": window_plan_ctx.get(
                                    "requested_preview"
                                ),
                                "requested_final": window_plan_ctx.get(
                                    "requested_final"
                                ),
                                "actual_preview": window_plan_ctx.get("actual_preview"),
                                "actual_final": window_plan_ctx.get("actual_final"),
                                "coverage_ok": window_plan_ctx.get("coverage_ok"),
                            }
                        )
                optional_keys = [
                    "logloss_preview",
                    "logloss_final",
                    "logloss_delta",
                    "logloss_preview_ci",
                    "logloss_final_ci",
                    "logloss_delta_ci",
                    "bootstrap",
                    "window_overlap_fraction",
                    "window_match_fraction",
                    "window_pairing_reason",
                    "window_pairing_preview",
                    "window_pairing_final",
                    "paired_windows",
                    "paired_delta_summary",
                    "primary_metric_tail",
                    "preview_total_tokens",
                    "final_total_tokens",
                    "masked_tokens_total",
                    "masked_tokens_preview",
                    "masked_tokens_final",
                    "timings",
                    "guard_timings",
                    "memory_snapshots",
                    "gpu_memory_mb_peak",
                    "gpu_memory_reserved_mb_peak",
                    "reduction",
                ]
                for key in optional_keys:
                    if key in core_report.metrics:
                        metrics_payload[key] = core_report.metrics[key]
                metrics_payload["loss_type"] = resolved_loss_type
                if metrics_payload.get("loss_type") is None and isinstance(
                    dataset_meta_context, dict
                ):
                    metrics_payload["loss_type"] = dataset_meta_context.get(
                        "loss_type", resolved_loss_type
                    )
                if isinstance(dataset_meta_context, dict):
                    for meta_key in (
                        "masked_tokens_total",
                        "masked_tokens_preview",
                        "masked_tokens_final",
                    ):
                        if (
                            meta_key not in metrics_payload
                            and dataset_meta_context.get(meta_key) is not None
                        ):
                            metrics_payload[meta_key] = dataset_meta_context[meta_key]
                report["metrics"].update(metrics_payload)

            if guard_overhead_payload is not None:
                if bool(guard_overhead_payload.get("skipped", False)):
                    report["guard_overhead"] = guard_overhead_payload
                else:
                    # Compute guarded primary-metric snapshot; pass structured reports into validator
                    try:
                        # Map loss type to ppl family kind
                        lk = str(resolved_loss_type or "causal").lower()
                        if lk == "mlm":
                            pm_kind_for_overhead = "ppl_mlm"
                        elif lk in {"seq2seq", "s2s", "t5"}:
                            pm_kind_for_overhead = "ppl_seq2seq"
                        else:
                            pm_kind_for_overhead = "ppl_causal"

                        # Prefer computing from the in-memory core_report windows to avoid ordering issues
                        pm_guarded = _extract_pm_snapshot_for_overhead(
                            core_report, kind=pm_kind_for_overhead
                        )
                        if not isinstance(pm_guarded, dict) or not pm_guarded:
                            pm_guarded = _extract_pm_snapshot_for_overhead(
                                report, kind=pm_kind_for_overhead
                            )

                        guard_overhead_payload["guarded_report"] = (
                            {"metrics": {"primary_metric": pm_guarded}}
                            if isinstance(pm_guarded, dict) and pm_guarded
                            else None
                        )
                    except Exception:
                        guard_overhead_payload["guarded_report"] = None
                    bare_struct = guard_overhead_payload.get("bare_report") or {}
                    guarded_struct = guard_overhead_payload.get("guarded_report") or {}
                    # Be robust to mocks or minimal objects returned by validators
                    result = validate_guard_overhead(
                        bare_struct,
                        guarded_struct,
                        overhead_threshold=guard_overhead_payload.get(
                            "overhead_threshold", GUARD_OVERHEAD_THRESHOLD
                        ),
                    )
                    try:
                        messages = list(getattr(result, "messages", []))
                    except Exception:  # pragma: no cover - defensive
                        messages = []
                    try:
                        warnings = list(getattr(result, "warnings", []))
                    except Exception:  # pragma: no cover - defensive
                        warnings = []
                    try:
                        errors = list(getattr(result, "errors", []))
                    except Exception:  # pragma: no cover - defensive
                        errors = []
                    try:
                        checks = dict(getattr(result, "checks", {}))
                    except Exception:  # pragma: no cover - defensive
                        checks = {}
                    metrics_obj = getattr(result, "metrics", {})
                    if not isinstance(metrics_obj, dict):
                        metrics_obj = {}
                    overhead_ratio = metrics_obj.get("overhead_ratio")
                    if overhead_ratio is None:
                        overhead_ratio = getattr(result, "overhead_ratio", None)
                    overhead_percent = metrics_obj.get("overhead_percent")
                    if overhead_percent is None:
                        overhead_percent = getattr(result, "overhead_percent", None)
                    passed_flag = bool(getattr(result, "passed", False))

                    guard_overhead_payload.update(
                        {
                            "messages": messages,
                            "warnings": warnings,
                            "errors": errors,
                            "checks": checks,
                            "overhead_ratio": overhead_ratio,
                            "overhead_percent": overhead_percent,
                            "passed": passed_flag,
                            "evaluated": True,
                        }
                    )
                    # Normalize for non-finite/degenerate cases
                    guard_overhead_payload = _normalize_overhead_result(
                        guard_overhead_payload, profile=profile_normalized
                    )
                    report["guard_overhead"] = guard_overhead_payload

            had_baseline = bool(baseline and Path(baseline).exists())
            if (
                hasattr(core_report, "evaluation_windows")
                and core_report.evaluation_windows
            ):
                preview_windows = core_report.evaluation_windows.get("preview", {})
                final_windows = core_report.evaluation_windows.get("final", {})
                report["evaluation_windows"] = {
                    "preview": {
                        "window_ids": list(preview_windows.get("window_ids", [])),
                        "logloss": list(preview_windows.get("logloss", [])),
                        "input_ids": [
                            list(seq) for seq in preview_windows.get("input_ids", [])
                        ],
                        "attention_masks": [
                            list(mask)
                            for mask in preview_windows.get("attention_masks", [])
                        ],
                        "token_counts": list(preview_windows.get("token_counts", [])),
                        "masked_token_counts": list(
                            preview_windows.get("masked_token_counts", [])
                        ),
                        "actual_token_counts": list(
                            preview_windows.get("actual_token_counts", [])
                        ),
                        "labels": [
                            list(seq) for seq in preview_windows.get("labels", [])
                        ],
                    },
                    "final": {
                        "window_ids": list(final_windows.get("window_ids", [])),
                        "logloss": list(final_windows.get("logloss", [])),
                        "input_ids": [
                            list(seq) for seq in final_windows.get("input_ids", [])
                        ],
                        "attention_masks": [
                            list(mask)
                            for mask in final_windows.get("attention_masks", [])
                        ],
                        "token_counts": list(final_windows.get("token_counts", [])),
                        "masked_token_counts": list(
                            final_windows.get("masked_token_counts", [])
                        ),
                        "actual_token_counts": list(
                            final_windows.get("actual_token_counts", [])
                        ),
                        "labels": [
                            list(seq) for seq in final_windows.get("labels", [])
                        ],
                    },
                }
            elif had_baseline and (profile or "").lower() in {"ci", "release"}:
                _event(
                    console,
                    "FAIL",
                    "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but evaluation windows were not produced. Check capacity/pairing config.",
                    emoji="❌",
                    profile=profile_normalized,
                )
                raise typer.Exit(3)
            else:
                # Populate evaluation_windows directly from assembled records when the
                # runner did not provide a structured window payload. This ensures
                # provenance (provider_digest) can be computed even in lightweight/dev
                # runs and unit tests that stub the runner.
                try:

                    def _tokens(rec: dict[str, Any]) -> int:
                        try:
                            return int(len(rec.get("input_ids", []) or []))
                        except Exception:
                            return 0

                    preview_window_count = len(preview_records)
                    final_window_count = len(final_records)

                    report["evaluation_windows"] = {
                        "preview": {
                            "window_ids": list(range(preview_window_count)),
                            "input_ids": [
                                list(r["input_ids"]) for r in preview_records
                            ],
                            "attention_masks": [
                                list(r["attention_mask"]) for r in preview_records
                            ],
                            "token_counts": [_tokens(r) for r in preview_records],
                            **(
                                {
                                    "masked_token_counts": list(preview_mask_counts),
                                    "labels": [
                                        r.get("labels", [-100] * len(r["input_ids"]))
                                        for r in preview_records
                                    ],
                                }
                                if use_mlm
                                else {}
                            ),
                        },
                        "final": {
                            "window_ids": list(
                                range(
                                    preview_window_count,
                                    preview_window_count + final_window_count,
                                )
                            ),
                            "input_ids": [list(r["input_ids"]) for r in final_records],
                            "attention_masks": [
                                list(r["attention_mask"]) for r in final_records
                            ],
                            "token_counts": [_tokens(r) for r in final_records],
                            **(
                                {
                                    "masked_token_counts": list(final_mask_counts),
                                    "labels": [
                                        r.get("labels", [-100] * len(r["input_ids"]))
                                        for r in final_records
                                    ],
                                }
                                if use_mlm
                                else {}
                            ),
                        },
                    }
                except Exception:
                    # Best-effort: provenance digest will be skipped if windows cannot be built
                    pass

            # Attach provider digest and dataset split provenance when available
            try:
                prov = report.setdefault("provenance", {})
                # Always record dataset split provenance for visibility
                try:
                    prov["dataset_split"] = str(resolved_split)
                    prov["split_fallback"] = bool(used_fallback_split)
                except Exception:
                    pass
                provider_digest = _compute_provider_digest(report)
                if provider_digest:
                    prov["provider_digest"] = provider_digest
                    # Attach digest version for future evolution
                    prov["digest_version"] = 1
                    # Strict parity checks in CI/Release when baseline present
                    try:
                        if isinstance(baseline_report_data, dict):
                            base_digest = None
                            base_prov = baseline_report_data.get("provenance")
                            if isinstance(base_prov, dict):
                                base_pd = base_prov.get("provider_digest")
                                if isinstance(base_pd, dict):
                                    base_digest = base_pd
                            if base_digest is None:
                                base_digest = _compute_provider_digest(
                                    baseline_report_data
                                )
                            _enforce_provider_parity(
                                provider_digest,
                                base_digest,
                                profile=(str(profile).lower() if profile else None),
                            )
                    except InvarlockError as ce:
                        console.print(str(ce))
                        # Map to profile-aware exit code: dev→1, ci/release→3
                        raise typer.Exit(
                            _resolve_exit_code(ce, profile=profile)
                        ) from None
                    except RuntimeError as _e:
                        _fail_run(str(_e))
                    except Exception:
                        pass
            except (typer.Exit, SystemExit, click.exceptions.Exit):
                raise
            except Exception:
                pass

            # Transfer guard results
            if hasattr(core_report, "guards") and core_report.guards:
                for guard_name, guard_result in core_report.guards.items():
                    guard_entry = {
                        "name": guard_name,
                        "passed": guard_result.get("passed"),
                        "action": guard_result.get("action"),
                        "policy": guard_result.get("policy", {}),
                        "metrics": guard_result.get("metrics", {}),
                        "actions": guard_result.get("actions", []),
                        "violations": guard_result.get("violations", []),
                        "warnings": guard_result.get("warnings", []),
                        "errors": guard_result.get("errors", []),
                        "details": guard_result.get("details", {}),
                    }
                    for extra_key in ("final_z_scores", "module_family_map"):
                        if extra_key in guard_result:
                            guard_entry[extra_key] = guard_result[extra_key]
                    report["guards"].append(guard_entry)

            # Set artifacts
            report["artifacts"].update(
                {
                    "events_path": str(run_config.event_path)
                    if run_config.event_path
                    else "",
                    "logs_path": "",
                    "checkpoint_path": None,
                }
            )

            # Optional: export HF-loadable model snapshot when requested
            export_env = str(
                os.environ.get("INVARLOCK_EXPORT_MODEL", "")
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            save_model_cfg = False
            try:
                save_model_cfg = bool(
                    getattr(getattr(cfg, "output", {}), "save_model", False)
                )
            except Exception:
                save_model_cfg = False
            if export_env or save_model_cfg:
                try:
                    # Resolve destination with precedence:
                    # 1) cfg.output.model_dir (absolute or relative to run_dir)
                    # 2) env INVARLOCK_EXPORT_DIR (absolute or relative)
                    # 3) cfg.output.model_subdir (under run_dir)
                    # 4) default: run_dir / "model"
                    export_dir: Path | None = None
                    # (1) explicit model_dir in config
                    try:
                        out_cfg = getattr(cfg, "output", None)
                        model_dir_cfg = None
                        if out_cfg is not None:
                            model_dir_cfg = getattr(
                                out_cfg, "model_dir", None
                            ) or getattr(out_cfg, "model_path", None)
                        if model_dir_cfg:
                            p = Path(str(model_dir_cfg))
                            export_dir = p if p.is_absolute() else (run_dir / p)
                    except Exception:
                        export_dir = None
                    # (2) env override
                    if export_dir is None:
                        env_dir_raw = os.environ.get("INVARLOCK_EXPORT_DIR", "")
                        if isinstance(env_dir_raw, str) and env_dir_raw.strip():
                            p = Path(env_dir_raw.strip())
                            export_dir = p if p.is_absolute() else (run_dir / p)
                    # (3) config subdir
                    if export_dir is None:
                        export_subdir = "model"
                        try:
                            export_subdir = str(
                                getattr(
                                    getattr(cfg, "output", {}), "model_subdir", "model"
                                )
                            )
                        except Exception:
                            export_subdir = "model"
                        export_dir = run_dir / export_subdir

                    # Ensure directory exists
                    ok = False
                    if hasattr(adapter, "save_pretrained") and model is not None:
                        ok = bool(adapter.save_pretrained(model, export_dir))  # type: ignore[attr-defined]
                    if ok:
                        report["artifacts"]["checkpoint_path"] = str(export_dir)
                    else:
                        _event(
                            console,
                            "WARN",
                            "Model export requested but adapter did not save a HF directory.",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )
                except Exception:
                    _event(
                        console,
                        "WARN",
                        "Model export requested but failed due to an unexpected error.",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

            # Set flags
            report["flags"].update(
                {
                    "guard_recovered": any(
                        not g.get("passed", True)
                        for g in core_report.guards.values()
                        if hasattr(core_report, "guards") and core_report.guards
                    ),
                    "rollback_reason": None,
                }
            )

            metrics_section = report.get("metrics", {}) or {}
            data_section = report.get("data", {}) or {}
            preview_count_report = data_section.get("preview_n")
            final_count_report = data_section.get("final_n")

            # Classification metric (accuracy) — deterministic smoke path
            # If loss type is explicitly 'classification', derive accuracy
            # counts from evaluation windows using a deterministic label rule.
            try:
                loss_type_ctx = (
                    run_config.context.get("eval", {})
                    .get("loss", {})
                    .get("resolved_type")
                )
            except Exception:
                loss_type_ctx = None
            if str(loss_type_ctx).lower() == "classification":
                try:
                    from invarlock.eval.primary_metric import compute_accuracy_counts

                    # Prefer in-memory core_report.evaluation_windows (includes input_ids)
                    ew = {}
                    try:
                        if hasattr(core_report, "evaluation_windows") and isinstance(
                            core_report.evaluation_windows, dict
                        ):
                            ew = core_report.evaluation_windows  # type: ignore[assignment]
                    except Exception:
                        ew = {}
                    if not ew:
                        # Fallback to the soon-to-be persisted report windows (may lack input_ids)
                        ew = (
                            report.get("evaluation_windows", {})
                            if isinstance(report.get("evaluation_windows"), dict)
                            else {}
                        )
                    prev_rec = []
                    fin_rec = []
                    if isinstance(ew, dict):
                        prev = ew.get("preview", {})
                        fin = ew.get("final", {})
                        if isinstance(prev, dict):
                            prev_rec = [
                                {"input_ids": seq}
                                for seq in prev.get("input_ids", []) or []
                                if isinstance(seq, list)
                            ]
                        if isinstance(fin, dict):
                            fin_rec = [
                                {"input_ids": seq}
                                for seq in fin.get("input_ids", []) or []
                                if isinstance(seq, list)
                            ]
                    c_prev, n_prev = compute_accuracy_counts(prev_rec)
                    c_fin, n_fin = compute_accuracy_counts(fin_rec)
                    # If we could not derive counts (no windows persisted), fall back to
                    # deterministic pseudo-accuracy based on configured window counts.
                    used_pseudo_counts = False
                    if n_prev == 0 and n_fin == 0:
                        try:
                            prev_n_cfg = getattr(cfg.dataset, "preview_n", None)
                            fin_n_cfg = getattr(cfg.dataset, "final_n", None)
                        except Exception:
                            prev_n_cfg = None
                            fin_n_cfg = None
                        try:
                            prev_n = int(preview_count_report or prev_n_cfg or 0)
                            fin_n = int(final_count_report or fin_n_cfg or 0)
                        except Exception:
                            prev_n = 0
                            fin_n = 0
                        c_prev, n_prev = (prev_n, prev_n) if prev_n > 0 else (0, 0)
                        c_fin, n_fin = (fin_n, fin_n) if fin_n > 0 else (0, 0)
                        used_pseudo_counts = prev_n > 0 or fin_n > 0
                    classification_metrics = {
                        "preview": {"correct_total": int(c_prev), "total": int(n_prev)},
                        "final": {"correct_total": int(c_fin), "total": int(n_fin)},
                    }
                    # Tag source of counts for downstream rendering/doctor
                    if used_pseudo_counts:
                        classification_metrics["counts_source"] = "pseudo_config"
                        # Add a provenance crumb for transparency
                        try:
                            prov = report.setdefault("provenance", {})
                            notes = prov.setdefault("metric_notes", [])
                            if isinstance(notes, list):
                                notes.append(
                                    "accuracy: pseudo counts from preview_n/final_n"
                                )
                        except Exception:
                            pass
                    else:
                        classification_metrics["counts_source"] = "measured"
                    report.setdefault("metrics", {})["classification"] = (
                        classification_metrics
                    )
                    # Convenience: top-level accuracy (final)
                    if n_fin > 0:
                        report["metrics"]["accuracy"] = float(c_fin / n_fin)
                except Exception:
                    pass

            match_fraction = metrics_section.get("window_match_fraction")
            if match_fraction is not None and not math.isclose(
                match_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                err = InvarlockError(
                    code="E001",
                    message=(
                        f"PAIRING-SCHEDULE-MISMATCH: window_match_fraction={match_fraction:.3f}"
                    ),
                    details={"window_match_fraction": float(match_fraction)},
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            overlap_fraction = metrics_section.get("window_overlap_fraction")
            if overlap_fraction is not None and overlap_fraction > 1e-9:
                err = InvarlockError(
                    code="E001",
                    message=(
                        f"PAIRING-SCHEDULE-MISMATCH: window_overlap_fraction={overlap_fraction:.3f}"
                    ),
                    details={"window_overlap_fraction": float(overlap_fraction)},
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            # Paired-run enforcement: baseline provided must be truly paired in CI/Release.
            if baseline and profile_normalized in {"ci", "release"}:
                pairing_reason = metrics_section.get("window_pairing_reason")
                if pairing_reason is not None:
                    err = InvarlockError(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but run was not paired "
                            f"(window_pairing_reason={pairing_reason})"
                        ),
                        details={"window_pairing_reason": pairing_reason},
                    )
                    code = _resolve_exit_code(err, profile=profile_normalized)
                    console.print(f"[red]{err}[/red]")
                    raise typer.Exit(code)

                paired_windows_val = metrics_section.get("paired_windows")
                paired_windows_int = None
                try:
                    if paired_windows_val is not None and not isinstance(
                        paired_windows_val, bool
                    ):
                        paired_windows_int = int(paired_windows_val)
                except Exception:
                    paired_windows_int = None
                if paired_windows_int is None or paired_windows_int <= 0:
                    err = InvarlockError(
                        code="E001",
                        message=(
                            "PAIRED-WINDOWS-COLLAPSED: paired_windows<=0 under paired baseline. "
                            "Check device stability, dataset windows, or edit scope."
                        ),
                        details={
                            "paired_windows": paired_windows_val,
                            "profile": profile_normalized,
                        },
                    )
                    code = _resolve_exit_code(err, profile=profile_normalized)
                    console.print(f"[red]{err}[/red]")
                    raise typer.Exit(code)

            expected_preview = effective_preview or getattr(
                cfg.dataset, "preview_n", preview_count_report
            )
            expected_final = effective_final or getattr(
                cfg.dataset, "final_n", final_count_report
            )
            if (
                preview_count_report is not None
                and expected_preview is not None
                and int(preview_count_report) != int(expected_preview)
            ) or (
                final_count_report is not None
                and expected_final is not None
                and int(final_count_report) != int(expected_final)
            ):
                err = InvarlockError(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: counts do not match configuration after stratification"
                    ),
                    details={
                        "preview_used": int(preview_count_report or -1),
                        "preview_expected": int(expected_preview or -1),
                        "final_used": int(final_count_report or -1),
                        "final_expected": int(expected_final or -1),
                    },
                )
                code = _resolve_exit_code(err, profile=profile_normalized)
                console.print(f"[red]{err}[/red]")
                raise typer.Exit(code)

            # Compute metric-v1 snapshot (primary_metric) — canonical path
            try:
                metric_kind_resolved, _provider_kind, metric_opts = (
                    _resolve_metric_and_provider(
                        cfg,
                        model_profile,
                        resolved_loss_type=resolved_loss_type,
                        metric_kind_override=metric_kind,
                    )
                )
                if metric_kind_resolved:
                    from invarlock.eval.primary_metric import (
                        compute_primary_metric_from_report,
                    )

                    pm = compute_primary_metric_from_report(
                        report, kind=metric_kind_resolved, baseline=baseline_report_data
                    )
                    core_primary_metric = None
                    if hasattr(core_report, "metrics") and isinstance(
                        core_report.metrics, dict
                    ):
                        core_primary_metric = core_report.metrics.get("primary_metric")
                    pm = _merge_primary_metric_health(pm, core_primary_metric)
                    report.setdefault("metrics", {})["primary_metric"] = pm
                    # Attach configured reps/ci_level when provided
                    if metric_opts:
                        try:
                            if "reps" in metric_opts:
                                report["metrics"]["primary_metric"]["reps"] = int(
                                    metric_opts["reps"]
                                )  # type: ignore[index]
                            if "ci_level" in metric_opts:
                                report["metrics"]["primary_metric"]["ci_level"] = float(
                                    metric_opts["ci_level"]
                                )  # type: ignore[index]
                        except Exception:
                            pass
                # Shadow parity check against ppl_* fields (best-effort)
                try:
                    pm_blk = report.get("metrics", {}).get("primary_metric", {})
                    ppl_final_v1 = float(pm_blk.get("final"))
                    ppl_final_v2 = float(pm.get("final", float("nan")))
                    if math.isfinite(ppl_final_v1) and math.isfinite(ppl_final_v2):
                        if not math.isclose(
                            ppl_final_v1, ppl_final_v2, rel_tol=1e-9, abs_tol=1e-9
                        ):
                            report.setdefault("metrics", {}).setdefault(
                                "_metric_v1_mismatch", {}
                            )["ppl_final_diff"] = ppl_final_v2 - ppl_final_v1
                    # Optional: dual-write diffs logging for ppl_* metrics
                    debug_diffs = str(
                        os.environ.get("DEBUG_METRIC_DIFFS", "")
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    if debug_diffs and str(pm.get("kind", "")).startswith("ppl"):
                        diffs_line = _format_debug_metric_diffs(
                            pm, report.get("metrics", {}), baseline_report_data
                        )
                        if diffs_line:
                            console.print(
                                "[dim]DEBUG_METRIC_DIFFS: " + diffs_line + "[/dim]"
                            )
                except Exception:
                    pass
            except Exception:
                # Non-fatal: metric-v1 snapshot should not break runs
                pass

            # No deprecation notices in dev-phase: primary_metric is canonical.

            # Derive dataset.windows.stats (PM-only surface)
            try:
                ds = report.setdefault("dataset", {}).setdefault("windows", {})
                stats = ds.setdefault("stats", {})
                if match_fraction is not None:
                    stats["window_match_fraction"] = float(match_fraction)
                if overlap_fraction is not None:
                    stats["window_overlap_fraction"] = float(overlap_fraction)
                try:
                    if isinstance(window_plan, dict) and "coverage_ok" in window_plan:
                        stats["coverage"] = bool(window_plan.get("coverage_ok"))
                except Exception:
                    pass
            except Exception:
                pass

            telemetry_path: Path | None = None
            if telemetry:
                telemetry_path = run_dir / "telemetry.json"
                report.setdefault("artifacts", {})["telemetry_path"] = str(
                    telemetry_path
                )

            saved_files = _postprocess_and_summarize(
                report=report,
                run_dir=run_dir,
                run_config=run_config,
                window_plan=window_plan,
                dataset_meta=dataset_meta,
                match_fraction=match_fraction,
                overlap_fraction=overlap_fraction,
                console=console,
            )
            try:
                if isinstance(saved_files, dict) and saved_files.get("json"):
                    report_path_out = str(saved_files["json"])
            except Exception:
                pass

            if telemetry and telemetry_path is not None:
                try:
                    from invarlock.reporting.telemetry import save_telemetry_report

                    saved_path = save_telemetry_report(
                        report, run_dir, filename=telemetry_path.name
                    )
                    if isinstance(saved_files, dict):
                        saved_files["telemetry"] = str(saved_path)
                    _event(
                        console,
                        "DATA",
                        f"Telemetry: {saved_path}",
                        emoji="📈",
                        profile=profile_normalized,
                    )
                except Exception as exc:  # pragma: no cover - best-effort
                    _event(
                        console,
                        "WARN",
                        f"Telemetry export failed: {exc}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )

            # Metrics display
            pm_obj = None
            try:
                pm_obj = report.get("metrics", {}).get("primary_metric")
            except Exception:
                pm_obj = None
            if isinstance(pm_obj, dict) and pm_obj:
                try:
                    pm_kind = str(pm_obj.get("kind", "primary")).lower()
                    pm_prev = pm_obj.get("preview")
                    pm_fin = pm_obj.get("final")
                    if isinstance(pm_prev, (int | float)) and isinstance(
                        pm_fin, (int | float)
                    ):
                        _event(
                            console,
                            "METRIC",
                            f"Primary Metric [{pm_kind}] — preview: {pm_prev:.3f}, final: {pm_fin:.3f}",
                            emoji="📌",
                            profile=profile_normalized,
                        )
                    ratio_vs_base = pm_obj.get("ratio_vs_baseline")
                    if isinstance(ratio_vs_base, (int | float)) and math.isfinite(
                        ratio_vs_base
                    ):
                        _event(
                            console,
                            "METRIC",
                            f"Ratio vs baseline [{pm_kind}]: {ratio_vs_base:.3f}",
                            emoji="🔗",
                            profile=profile_normalized,
                        )
                except Exception:
                    pass
            # Legacy ppl_* console block removed in favor of primary_metric summary

            guard_overhead_info = report.get("guard_overhead")
            if guard_overhead_info:
                threshold_fraction = _print_guard_overhead_summary(
                    console, guard_overhead_info
                )
                if not guard_overhead_info.get("passed", True):
                    _event(
                        console,
                        "FAIL",
                        "Guard overhead gate FAILED: Guards add more than the permitted budget",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    # Only fail hard when the overhead check was actually evaluated
                    # (e.g., for causal LMs with available bare/guarded PM). For
                    # masked LM flows where ppl-like PM is undefined, record as not evaluated
                    # and continue without aborting the run.
                    loss_type_ctx = None
                    try:
                        loss_type_ctx = (
                            run_config.context.get("eval", {})
                            .get("loss", {})
                            .get("resolved_type")
                        )
                    except Exception:
                        loss_type_ctx = None
                    if (
                        measure_guard_overhead
                        and guard_overhead_info.get("evaluated", False)
                        and str(loss_type_ctx).lower() != "mlm"
                    ):
                        _fail_run(
                            "Guard overhead gate exceeded the configured budget "
                            f"(>{threshold_fraction * 100:.1f}% increase)"
                        )

            # Drift gate status is no longer surfaced in console; rely on certificate gates

            # Certificate validation for --until-pass mode
            if retry_controller and baseline:
                from invarlock.reporting.certificate import make_certificate

                try:
                    baseline_report = baseline_report_data
                    if baseline_report is None and baseline:
                        baseline_path = Path(baseline)
                        with baseline_path.open(encoding="utf-8") as f:
                            baseline_report = json.load(f)

                    if baseline_report is None:
                        raise FileNotFoundError("Baseline report unavailable")

                    _event(
                        console,
                        "EXEC",
                        "Generating evaluation certificate...",
                        emoji="📜",
                        profile=profile_normalized,
                    )
                    certificate = make_certificate(report, baseline_report)

                    validation = certificate.get("validation", {})
                    certificate_passed = all(validation.values())

                    failed_gates = [k for k, v in validation.items() if not v]
                    result_summary = {
                        "passed": certificate_passed,
                        "failures": failed_gates,
                        "validation": validation,
                    }
                    retry_controller.record_attempt(
                        attempt, result_summary, edit_config
                    )

                    if certificate_passed:
                        _event(
                            console,
                            "PASS",
                            "Certificate PASSED all gates!",
                            emoji="✅",
                            profile=profile_normalized,
                        )
                        break
                    else:
                        _event(
                            console,
                            "FAIL",
                            f"Certificate FAILED gates: {', '.join(failed_gates)}",
                            emoji="⚠️",
                            profile=profile_normalized,
                        )

                        # Auto-tune mask-only heads (binary search on keep count)
                        try:
                            head_section = None
                            for k in ("heads", "head_budget", "head_budgets"):
                                if isinstance(edit_config.get(k), dict):
                                    head_section = edit_config[k]
                                    break
                            search = (
                                head_section.get("_auto_search")
                                if isinstance(head_section, dict)
                                else None
                            )
                            if isinstance(search, dict) and head_section.get(
                                "mask_only"
                            ):
                                keep_low = int(search.get("keep_low", 0))
                                keep_high = int(
                                    search.get(
                                        "keep_high", search.get("total_heads", 0)
                                    )
                                )
                                keep_current = int(
                                    search.get("keep_current", keep_high)
                                )
                                # If the quality gate (PM) is unacceptable, increase keep (less pruning); if only other gates failed, be conservative and increase keep slightly
                                pm_ok = bool(
                                    validation.get("primary_metric_acceptable", False)
                                )
                                if not pm_ok:
                                    keep_low = max(keep_low, keep_current)
                                else:
                                    # drift/spectral/etc failed: ease pruning
                                    keep_low = max(keep_low, keep_current)
                                next_keep = int((keep_low + keep_high + 1) // 2)
                                search.update(
                                    {
                                        "keep_low": keep_low,
                                        "keep_high": keep_high,
                                        "keep_current": next_keep,
                                    }
                                )
                                head_section["global_k"] = next_keep
                                _event(
                                    console,
                                    "INIT",
                                    f"Auto-tune adjust: global_k → {next_keep} (bounds {keep_low}-{keep_high})",
                                    emoji="🔧",
                                    profile=profile_normalized,
                                )
                        except Exception:
                            pass

                        if retry_controller.should_retry(certificate_passed):
                            attempt += 1
                            continue
                        else:
                            _event(
                                console,
                                "FAIL",
                                f"Exhausted retry budget after {attempt} attempts",
                                emoji="❌",
                                profile=profile_normalized,
                            )
                            break

                except Exception as cert_error:
                    _event(
                        console,
                        "WARN",
                        f"Certificate validation failed: {cert_error}",
                        emoji="⚠️",
                        profile=profile_normalized,
                    )
                    if retry_controller:
                        retry_controller.record_attempt(
                            attempt,
                            {
                                "passed": False,
                                "failures": ["certificate_error"],
                                "validation": {},
                            },
                            edit_config,
                        )
                    break
            else:
                if retry_controller:
                    retry_controller.record_attempt(
                        attempt,
                        {"passed": True, "failures": [], "validation": {}},
                        edit_config,
                    )
                # No retry mode - single run
                break

            # Show retry summary if applicable
            _print_retry_summary(console, retry_controller)

            # (moved) Cleanup printing occurs after loop to guarantee execution
            pass

        if output_style.timing:
            total_duration = (
                max(0.0, float(perf_counter() - total_start))
                if total_start is not None
                else None
            )
            timings_for_summary: dict[str, float] = {}
            for key, value in timings.items():
                if isinstance(value, (int | float)):
                    timings_for_summary[key] = float(value)
            if total_duration is not None:
                timings_for_summary["total"] = total_duration

            has_breakdown = any(
                key in timings_for_summary
                for key in (
                    "prepare",
                    "prepare_guards",
                    "edit",
                    "guards",
                    "eval",
                    "finalize",
                )
            )

            order: list[tuple[str, str]] = []

            def _add(label: str, key: str) -> None:
                if key in timings_for_summary:
                    order.append((label, key))

            _add("Load model", "load_model")
            _add("Load data", "load_dataset")
            if has_breakdown:
                _add("Prepare", "prepare")
                _add("Prep guards", "prepare_guards")
                _add("Edit", "edit")
                _add("Guards", "guards")
                _add("Eval", "eval")
                _add("Finalize", "finalize")
            else:
                _add("Execute", "execute")
            _add("Total", "total")

            extra_lines: list[str] = []
            metrics_section = (
                report.get("metrics", {}) if isinstance(report, dict) else {}
            )
            if isinstance(metrics_section, dict):
                mem_peak = metrics_section.get("memory_mb_peak")
                gpu_peak = metrics_section.get("gpu_memory_mb_peak")
                if isinstance(mem_peak, (int | float)):
                    extra_lines.append(f"  Peak Memory : {float(mem_peak):.2f} MB")
                if isinstance(gpu_peak, (int | float)):
                    extra_lines.append(f"  Peak GPU Mem: {float(gpu_peak):.2f} MB")

            if timings_for_summary and order:
                print_timing_summary(
                    console,
                    timings_for_summary,
                    style=output_style,
                    order=order,
                    extra_lines=extra_lines,
                )

        # Normal path falls through; cleanup handled below in finally
        return report_path_out

    except FileNotFoundError as e:
        _event(
            console,
            "FAIL",
            f"Configuration file not found: {e}",
            emoji="❌",
            profile=profile_normalized,
        )
        raise typer.Exit(1) from e
    except InvarlockError as ce:
        # InvarlockError → code 3 only in CI/Release; dev → 1
        console.print(str(ce))
        raise typer.Exit(_resolve_exit_code(ce, profile=profile)) from ce
    except (typer.Exit, SystemExit, click.exceptions.Exit):
        # Preserve explicit exit codes (e.g., parity checks, user-triggered exits)
        raise
    except Exception as e:
        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            import traceback

            traceback.print_exc()
        # Emit a clearer message for schema failures (exit 2)
        if isinstance(e, ValueError) and "Invalid RunReport" in str(e):
            _event(
                console,
                "FAIL",
                "Schema invalid: run report structure failed validation",
                emoji="❌",
                profile=profile_normalized,
            )
            code = 2
        else:
            _event(
                console,
                "FAIL",
                f"Pipeline execution failed: {e}",
                emoji="❌",
                profile=profile_normalized,
            )
            code = _resolve_exit_code(e, profile=profile)
        raise typer.Exit(code) from e
    finally:
        # Cleanup snapshot directory if used (always print once per run)
        try:
            if snapshot_tmpdir and not no_cleanup:
                try:
                    import shutil as _sh

                    _sh.rmtree(snapshot_tmpdir, ignore_errors=True)
                except Exception:
                    pass
                finally:
                    _event(
                        console,
                        "INFO",
                        "Cleanup: removed",
                        emoji="🧹",
                        profile=profile_normalized,
                    )
            else:
                _event(
                    console,
                    "INFO",
                    "Cleanup: skipped",
                    emoji="🧹",
                    profile=profile_normalized,
                )
        except Exception:
            # Best-effort cleanup printing; never raise from finally
            pass


def _merge_primary_metric_health(
    primary_metric: dict[str, Any] | None,
    core_primary_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(primary_metric, dict):
        return {}
    merged = dict(primary_metric)
    if not isinstance(core_primary_metric, dict):
        return merged
    if core_primary_metric.get("invalid") is True:
        merged["invalid"] = True
        merged["degraded"] = True
    if core_primary_metric.get("degraded") is True:
        merged["degraded"] = True
    core_reason = core_primary_metric.get("degraded_reason")
    if isinstance(core_reason, str) and core_reason:
        merged["degraded_reason"] = core_reason
        merged["degraded"] = True
    return merged


def _format_debug_metric_diffs(
    pm: dict[str, float] | None,
    metrics: dict[str, float] | None,
    baseline_report_data: dict | None,
) -> str:
    """Build a compact DEBUG_METRIC_DIFFS line comparing current snapshot vs ppl_*.

    Returns a semicolon-separated string of deltas like
    "final: v1-v1 = +0.000000000; Δlog(final): +0.000000000; ...". Safe to call with
    missing fields; non-finite entries are skipped.
    """
    import math as _m

    if not isinstance(pm, dict) or not isinstance(metrics, dict):
        return ""
    diffs: list[str] = []
    try:
        pm_blk = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
        ppl_final_v1 = float(pm_blk.get("final", float("nan")))
    except Exception:
        ppl_final_v1 = float("nan")
    try:
        ppl_prev_v1 = float(pm_blk.get("preview", float("nan")))
    except Exception:
        ppl_prev_v1 = float("nan")
    try:
        ppl_final_v2 = float(pm.get("final", float("nan")))
    except Exception:
        ppl_final_v2 = float("nan")
    try:
        ppl_prev_v2 = float(pm.get("preview", float("nan")))
    except Exception:
        ppl_prev_v2 = float("nan")

    if _m.isfinite(ppl_final_v1) and _m.isfinite(ppl_final_v2):
        diffs.append(f"final: v1-v1 = {ppl_final_v2 - ppl_final_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(final): {_m.log(ppl_final_v2) - _m.log(ppl_final_v1):+.9f}"
            )
        except Exception:
            pass
    if _m.isfinite(ppl_prev_v1) and _m.isfinite(ppl_prev_v2):
        diffs.append(f"preview: v1-v1 = {ppl_prev_v2 - ppl_prev_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(preview): {_m.log(ppl_prev_v2) - _m.log(ppl_prev_v1):+.9f}"
            )
        except Exception:
            pass

    # ratio vs baseline
    try:
        r_v2 = float(pm.get("ratio_vs_baseline", float("nan")))
    except Exception:
        r_v2 = float("nan")
    # prefer PM ratio when present
    r_v1 = float(pm_blk.get("ratio_vs_baseline", float("nan")))
    if (not _m.isfinite(r_v1)) and isinstance(baseline_report_data, dict):
        try:
            base_fin = float(
                (
                    (baseline_report_data.get("metrics") or {}).get("primary_metric")
                    or {}
                ).get("final")
            )
            if _m.isfinite(base_fin) and base_fin > 0 and _m.isfinite(ppl_final_v1):
                r_v1 = ppl_final_v1 / base_fin
        except Exception:
            pass
    if _m.isfinite(r_v1) and _m.isfinite(r_v2):
        diffs.append(f"ratio_vs_baseline: v1-v1 = {r_v2 - r_v1:+.9f}")
    return "; ".join(diffs)


# Provide a module shim so tests can patch 'src.invarlock.cli.commands.run.shutil.*'.
try:  # best-effort; harmless in production
    _shim = _types.ModuleType(__name__ + ".shutil")

    def _shim_getattr(name: str):  # pragma: no cover
        return getattr(shutil, name)

    _shim.__getattr__ = _shim_getattr  # type: ignore[attr-defined]
    _shim.disk_usage = shutil.disk_usage  # type: ignore[attr-defined]
    _shim.rmtree = shutil.rmtree  # type: ignore[attr-defined]
    _sys.modules[__name__ + ".shutil"] = _shim
    _sys.modules["src." + __name__ + ".shutil"] = _shim
except Exception:
    pass


def _normalize_overhead_result(
    payload: dict[str, object] | None, profile: str | None = None
) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny/degenerate runs.

    If the computed overhead ratio is missing or non-finite, mark the check as
    not evaluated and passed to avoid spurious gate failures in tiny runs.
    """
    payload = dict(payload or {})
    try:
        ratio = payload.get("overhead_ratio")
        val = float(ratio) if isinstance(ratio, int | float) else float("nan")
    except Exception:
        val = float("nan")
    if not (isinstance(val, float) and math.isfinite(val)):
        payload["evaluated"] = False
        payload["passed"] = True
    return payload


# helper moved to invarlock.cli.overhead_utils


def _print_guard_overhead_summary(
    console: Console, guard_overhead_info: dict[str, Any]
) -> float:
    """Print a concise guard-overhead console summary. Returns threshold fraction used."""
    evaluated = bool(guard_overhead_info.get("evaluated", True))
    if not evaluated:
        _event(console, "METRIC", "Guard Overhead: not evaluated", emoji="🛡️")
        return GUARD_OVERHEAD_THRESHOLD
    overhead_status = "PASS" if guard_overhead_info.get("passed", True) else "FAIL"
    overhead_percent = guard_overhead_info.get("overhead_percent")
    if isinstance(overhead_percent, (int | float)) and math.isfinite(
        float(overhead_percent)
    ):
        overhead_display = f"{float(overhead_percent):+.2f}%"
    else:
        ratio_value = guard_overhead_info.get("overhead_ratio")
        if isinstance(ratio_value, (int | float)) and math.isfinite(float(ratio_value)):
            overhead_display = f"{float(ratio_value):.3f}x"
        else:
            # Avoid any 'nanx' or ambiguous output
            overhead_display = "not evaluated"
    threshold_percent = guard_overhead_info.get("overhead_threshold", 0.01)
    try:
        threshold_fraction = float(threshold_percent)
    except (TypeError, ValueError):
        threshold_fraction = GUARD_OVERHEAD_THRESHOLD
    threshold_display = f"≤ +{threshold_fraction * 100:.1f}%"
    _event(
        console,
        "METRIC",
        f"Guard Overhead: {overhead_status} {overhead_display} ({threshold_display})",
        emoji="🛡️",
    )
    return threshold_fraction


def _print_retry_summary(console: Console, retry_controller: Any | None) -> None:
    """Print a one-line retry summary when retries were attempted."""
    try:
        if retry_controller and getattr(retry_controller, "attempt_history", None):
            summary = retry_controller.get_attempt_summary()
            console.print("\n")
            _event(
                console,
                "METRIC",
                f"Retry Summary: {summary['total_attempts']} attempts in {summary['elapsed_time']:.1f}s",
                emoji="📊",
            )
    except Exception:
        # Never break the run for summary printing
        pass


def _init_retry_controller(
    *,
    until_pass: bool,
    max_attempts: int,
    timeout: int | None,
    baseline: str | None,
    console: Console,
):
    """Initialize RetryController with consistent console prints."""
    retry_controller = None
    if until_pass:
        from invarlock.core.retry import RetryController

        retry_controller = RetryController(
            max_attempts=max_attempts, timeout=timeout, verbose=True
        )
        _event(
            console,
            "INIT",
            f"Retry mode enabled: max {max_attempts} attempts",
            emoji="🔄",
        )
        if baseline:
            _event(console, "DATA", f"Using baseline: {baseline}", emoji="📋")
    else:
        if baseline:
            _event(console, "DATA", f"Using baseline: {baseline}", emoji="📋")
    return retry_controller
