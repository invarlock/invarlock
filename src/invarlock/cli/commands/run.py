"""
InvarLock CLI Run Command
=====================

Run a guarded pipeline from a YAML config. Intended for local smokes,
plugin demos, and development. Advanced: for pairwise evaluation,
prefer Compare & Evaluate via `invarlock evaluate --baseline ... --subject ...`.
"""

import copy
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys as _sys
import warnings
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
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
from invarlock.cli.run_artifacts import (
    persist_ref_masks as _persist_ref_masks_impl,
)
from invarlock.cli.run_command_impl import (
    run_command_impl as _run_command_impl,
)
from invarlock.cli.run_config import (
    extract_model_load_kwargs as _extract_model_load_kwargs_impl,
)
from invarlock.cli.run_config import (
    prepare_config_for_run as _prepare_config_for_run_impl,
)
from invarlock.cli.run_config import (
    resolve_device_and_output as _resolve_device_and_output_impl,
)
from invarlock.cli.run_config import (
    resolve_provider_and_split as _resolve_provider_and_split_impl,
)
from invarlock.cli.run_overhead import (
    plan_release_windows as _plan_release_windows_impl,
)
from invarlock.cli.run_pairing import (
    compute_provider_digest as _compute_provider_digest_impl,
)
from invarlock.cli.run_pairing import (
    enforce_provider_parity as _enforce_provider_parity_impl,
)
from invarlock.cli.run_pairing import (
    extract_pairing_schedule as _extract_pairing_schedule_impl,
)
from invarlock.cli.run_pairing import (
    resolve_metric_and_provider as _resolve_metric_and_provider_impl,
)
from invarlock.cli.run_pairing import (
    validate_and_harvest_baseline_schedule as _validate_and_harvest_baseline_schedule_impl,
)
from invarlock.cli.utils import (
    coerce_float as _coerce_float,
)
from invarlock.cli.utils import (
    coerce_int as _coerce_int,
)
from invarlock.cli.utils import (
    coerce_option as _coerce_option,
)
from invarlock.core.auto_tuning import (
    resolve_tier_policies as _resolve_tier_policies,
)
from invarlock.core.config_execution import (
    RuntimeDelegationError,
    run_from_config,
)
from invarlock.core.exceptions import (
    ConfigError as _CfgErr,
)
from invarlock.core.exceptions import (
    InvarlockError,
)
from invarlock.core.exit_codes import (
    resolve_command_exit_code as _resolve_command_exit_code,
)
from invarlock.core.run_baseline_evidence import (
    load_baseline_pairing_evidence as _load_baseline_pairing_evidence_impl,
)
from invarlock.core.run_baseline_evidence import (
    materialize_baseline_pairing_schedule as _materialize_baseline_pairing_schedule_impl,
)
from invarlock.core.run_evaluation_windows_policy import (
    build_fallback_evaluation_windows as _build_fallback_evaluation_windows_impl,
)
from invarlock.core.run_evaluation_windows_policy import (
    serialize_evaluation_windows as _serialize_evaluation_windows_impl,
)
from invarlock.core.run_execution_context_policy import (
    build_run_context_payload as _build_run_context_payload_impl,
)
from invarlock.core.run_execution_context_policy import (
    build_run_execution_config_payloads as _build_run_execution_config_payloads_impl,
)
from invarlock.core.run_guard_overhead_policy import (
    build_guard_overhead_summary as _build_guard_overhead_summary_impl,
)
from invarlock.core.run_guard_overhead_policy import (
    finalize_guard_overhead_payload as _finalize_guard_overhead_payload_impl,
)
from invarlock.core.run_guard_overhead_policy import (
    normalize_guard_overhead_result as _normalize_overhead_result_impl,
)
from invarlock.core.run_guard_overhead_policy import (
    prepare_guard_overhead_report as _prepare_guard_overhead_report_impl,
)
from invarlock.core.run_policy import (
    choose_dataset_split as _choose_dataset_split_impl,
)
from invarlock.core.run_policy import (
    coerce_bool_like as _coerce_bool_like_impl,
)
from invarlock.core.run_policy import (
    coerce_mapping as _coerce_mapping_impl,
)
from invarlock.core.run_policy import (
    resolve_guard_overhead_threshold as _resolve_guard_overhead_threshold_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_acceptance_range as _resolve_pm_acceptance_range_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_drift_band as _resolve_pm_drift_band_impl,
)
from invarlock.core.run_policy import (
    resolve_skip_overhead_policy as _resolve_skip_overhead_policy_impl,
)
from invarlock.core.run_policy import (
    should_measure_overhead as _should_measure_overhead_impl,
)
from invarlock.core.run_provider_dataset_plan import (
    build_provider_dataset_plan as _build_provider_dataset_plan_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_artifacts_payload as _build_artifacts_payload_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_edit_payload as _build_edit_payload_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_flags_payload as _build_flags_payload_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_guard_entries as _build_guard_entries_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_metrics_payload as _build_metrics_payload_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_run_report_context as _build_run_report_context_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_run_report_data as _build_run_report_data_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_run_report_meta as _build_run_report_meta_impl,
)
from invarlock.core.run_report_payload_policy import (
    build_snapshot_provenance as _build_snapshot_provenance_impl,
)
from invarlock.core.run_report_payload_policy import (
    merge_core_timing_metrics as _merge_core_timing_metrics_impl,
)
from invarlock.core.run_retry_policy import (
    build_restore_failure_attempt_summary as _build_restore_failure_attempt_summary_impl,
)
from invarlock.core.run_retry_policy import (
    build_retry_result_summary as _build_retry_result_summary_impl,
)
from invarlock.core.run_retry_policy import (
    decide_failed_retry_transition as _decide_failed_retry_transition_impl,
)
from invarlock.core.run_retry_policy import (
    record_retry_attempt as _record_retry_attempt_impl,
)
from invarlock.core.run_retry_policy import (
    resolve_retry_validation_transition as _resolve_retry_validation_transition_impl,
)
from invarlock.core.run_snapshot_policy import (
    choose_snapshot_mode as _choose_snapshot_mode_impl,
)
from invarlock.core.run_snapshot_policy import (
    estimate_model_bytes as _estimate_model_bytes_impl,
)
from invarlock.core.run_snapshot_policy import (
    resolve_snapshot_config as _resolve_snapshot_config_impl,
)
from invarlock.core.run_timing_policy import (
    build_timing_summary_payload as _build_timing_summary_payload_impl,
)
from invarlock.eval.window_planning import (
    resolve_effective_windows as _resolve_effective_windows_impl,
)
from invarlock.model_utils import set_seed
from invarlock.reporting.run_metric_utils import (
    format_debug_metric_diffs as _format_debug_metric_diffs_impl,
)
from invarlock.reporting.run_metric_utils import (
    merge_primary_metric_health as _merge_primary_metric_health_impl,
)
from invarlock.reporting.run_pairing_contract import (
    build_dataset_window_stats as _build_dataset_window_stats_impl,
)
from invarlock.reporting.run_pairing_contract import (
    validate_pairing_report_metrics as _validate_pairing_report_metrics_impl,
)
from invarlock.reporting.run_provenance_contract import (
    finalize_run_provenance as _finalize_run_provenance_impl,
)
from invarlock.reporting.run_report_metrics_contract import (
    enrich_run_report_metrics as _enrich_run_report_metrics_impl,
)
from invarlock.reporting.run_retry_validation import (
    validate_retry_evaluation_report as _validate_retry_evaluation_report_impl,
)

from ...core.config_runtime import InvarLockConfig
from ..overhead_utils import _extract_pm_snapshot_for_overhead

console = make_console()
_IMPORT_UNSET = object()
_psutil_module: Any = _IMPORT_UNSET
_torch_module: Any = _IMPORT_UNSET


class _LazyImportProxy:
    """Expose a patch-friendly module surface while deferring the real import."""

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader

    def _target(self) -> Any:
        return self._loader()

    def __getattr__(self, name: str) -> Any:
        target = self._target()
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)

    def __bool__(self) -> bool:
        return self._target() is not None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        target = self._target()
        if target is None:
            return "<lazy-missing-module>"
        return repr(target)


def _load_psutil_module() -> Any:
    global _psutil_module
    if _psutil_module is _IMPORT_UNSET:
        try:
            import psutil as _psutil
        except ImportError:
            _psutil_module = None
        else:
            _psutil_module = _psutil
    return None if _psutil_module is _IMPORT_UNSET else _psutil_module


def _load_torch_module() -> Any:
    global _torch_module
    if _torch_module is _IMPORT_UNSET:
        try:
            import torch as _torch
        except ImportError:
            _torch_module = None
        else:
            _torch_module = _torch
    return None if _torch_module is _IMPORT_UNSET else _torch_module


def _get_psutil() -> Any:
    return psutil


def _get_torch() -> Any:
    return torch


psutil: Any = _LazyImportProxy(_load_psutil_module)
torch: Any = _LazyImportProxy(_load_torch_module)


def _reset_optional_runtime_caches() -> None:
    global _psutil_module, _torch_module
    if isinstance(psutil, _LazyImportProxy):
        _psutil_module = _IMPORT_UNSET
    if isinstance(torch, _LazyImportProxy):
        _torch_module = _IMPORT_UNSET


def detect_model_profile(model_id: str, adapter: str | None = None) -> Any:
    from invarlock.model_profile import detect_model_profile as _detect_model_profile

    return _detect_model_profile(model_id=model_id, adapter=adapter)


def resolve_tokenizer(profile: Any) -> tuple[Any, str]:
    from invarlock.model_profile import resolve_tokenizer as _resolve_tokenizer

    return _resolve_tokenizer(profile)


def validate_guard_overhead(*args: Any, **kwargs: Any) -> Any:
    from invarlock.reporting.validate import (
        validate_guard_overhead as _validate_guard_overhead,
    )

    return _validate_guard_overhead(*args, **kwargs)


# Keep these names available on this module for delegated run_command_impl
# runtime lookup and test monkeypatch compatibility.
_RUN_COMMAND_IMPL_EXPORTS = (
    psutil,
    perf_counter,
    print_timing_summary,
    timed_step,
    _coerce_float,
    _coerce_int,
    _coerce_option,
    detect_model_profile,
    resolve_tokenizer,
    validate_guard_overhead,
)


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


def _canonical_dataset_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        value = getattr(value, "_data", value)
    except AttributeError:
        pass
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


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

_NOISY_WARNING_PATTERNS = (r".*loss_type=None.*unrecognized.*",)


def _resolve_warning_suppression(profile: str | None) -> tuple[bool, bool]:
    suppress_all = os.getenv("INVARLOCK_SUPPRESS_WARNINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    profile_norm = (profile or "").strip().lower()
    enabled = bool(suppress_all) or profile_norm in {"ci", "ci_cpu", "release"}
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
def _suppress_noisy_warnings(
    profile: str | None,
    *,
    event_path: Path | None = None,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    enabled, suppress_all = _resolve_warning_suppression(profile)
    if not enabled:
        yield
        return

    prev_tf_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    transformers_logger = logging.getLogger("transformers")
    prev_tf_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)

    patterns = [re.compile(p) for p in _NOISY_WARNING_PATTERNS]
    suppressed: list[str] = []

    class _NoisyLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            try:
                message = record.getMessage()
            except Exception:
                return True
            if any(p.search(message) for p in patterns):
                suppressed.append(message)
                return False
            return True

    def _iter_handlers() -> list[logging.Handler]:
        handlers: list[logging.Handler] = []
        seen: set[int] = set()
        for logger in (
            logging.getLogger(),
            logging.getLogger("transformers"),
            logging.getLogger("huggingface_hub"),
            logging.getLogger("datasets"),
        ):
            for handler in getattr(logger, "handlers", []) or []:
                if id(handler) in seen:
                    continue
                seen.add(id(handler))
                handlers.append(handler)
        return handlers

    log_filter = _NoisyLogFilter()
    handlers = _iter_handlers()

    def _append_suppressed_warnings() -> None:
        if not suppressed or event_path is None:
            return
        try:
            path = Path(event_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now().isoformat(),
                "component": "warnings",
                "operation": "suppressed",
                "level": "WARNING",
                "data": {
                    "count": len(suppressed),
                    "messages": suppressed[:50],
                    "profile": profile or "",
                    **(context or {}),
                },
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except (OSError, TypeError, ValueError):
            # Best-effort: suppressed warnings are non-fatal and logging must not
            # impact model loading.
            return

    for handler in handlers:
        handler.addFilter(log_filter)

    try:
        with warnings.catch_warnings():
            from contextlib import redirect_stderr, redirect_stdout

            class _FilteredStream:
                def __init__(self, raw: Any) -> None:
                    self._raw = raw

                def __getattr__(self, name: str) -> object:
                    return getattr(self._raw, name)

                def write(self, s: object) -> int:
                    try:
                        if isinstance(s, bytes):
                            text = s.decode("utf-8", errors="replace")
                        else:
                            text = str(s)
                    except (TypeError, ValueError, UnicodeDecodeError):
                        return int(self._raw.write(s))

                    # Preserve progress bars (carriage returns) by passing through
                    # all non-matching chunks immediately.
                    pieces = text.splitlines(keepends=True)
                    for piece in pieces:
                        if any(p.search(piece) for p in patterns):
                            suppressed.append(piece.rstrip("\n"))
                            continue
                        self._raw.write(piece)
                    return len(text)

                def flush(self) -> None:
                    try:
                        self._raw.flush()
                    except (AttributeError, OSError, ValueError):
                        pass

            stdout_proxy = _FilteredStream(_sys.stdout)
            stderr_proxy = _FilteredStream(_sys.stderr)

            with redirect_stdout(stdout_proxy), redirect_stderr(stderr_proxy):
                if suppress_all:
                    warnings.simplefilter("ignore")
                    yield
                else:
                    original_showwarning = warnings.showwarning

                    def _showwarning(
                        message: Warning | str,
                        category: type[Warning],
                        filename: str,
                        lineno: int,
                        file: object | None = None,
                        line: str | None = None,
                    ) -> None:
                        try:
                            rendered = warnings.formatwarning(
                                message, category, filename, lineno, line
                            )
                        except (TypeError, ValueError):
                            rendered = str(message)
                        if any(p.search(rendered) for p in patterns):
                            suppressed.append(str(message))
                            return
                        original_showwarning(
                            message,
                            category,
                            filename,
                            lineno,
                            file=file,
                            line=line,
                        )

                    warnings.showwarning = _showwarning  # type: ignore[assignment]
                    try:
                        yield
                    finally:
                        warnings.showwarning = original_showwarning  # type: ignore[assignment]
    finally:
        for handler in handlers:
            try:
                handler.removeFilter(log_filter)
            except ValueError:
                pass
        try:
            transformers_logger.setLevel(prev_tf_level)
        except (TypeError, ValueError):
            pass
        if prev_tf_verbosity is None:
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        else:
            os.environ["TRANSFORMERS_VERBOSITY"] = prev_tf_verbosity
        _append_suppressed_warnings()


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
    return " → ".join(names)


# Common dataset split aliases we probe in order when not explicitly set
SPLIT_ALIASES: tuple[str, ...] = ("validation", "val", "dev", "eval", "test")


def _coerce_mapping(obj: object) -> dict[str, Any]:
    """Best-effort conversion of config-like objects to plain dicts."""
    return _coerce_mapping_impl(obj)


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
    """Resolve primary-metric acceptance bounds from config with safe defaults."""
    return _resolve_pm_acceptance_range_impl(cfg, coerce_mapping_fn=_coerce_mapping)


def _resolve_pm_drift_band(
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve preview→final drift band from config with safe defaults."""
    return _resolve_pm_drift_band_impl(cfg, coerce_mapping_fn=_coerce_mapping)


def _resolve_guard_overhead_threshold(
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> float:
    """Resolve guard-overhead threshold from config with safe default fallback."""
    return _resolve_guard_overhead_threshold_impl(
        cfg,
        default_threshold=GUARD_OVERHEAD_THRESHOLD,
        coerce_mapping_fn=_coerce_mapping,
    )


def _coerce_bool_like(value: Any) -> bool | None:
    """Best-effort bool coercion used for config policy toggles."""
    return _coerce_bool_like_impl(value)


def _resolve_skip_overhead_policy(
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Resolve overhead-skip policy from run/eval config context."""
    return _resolve_skip_overhead_policy_impl(cfg, coerce_mapping_fn=_coerce_mapping)


def _free_model_memory(model: object | None) -> None:
    """Best-effort cleanup to release GPU memory for a model object."""
    if model is None:
        return
    try:
        import gc

        torch_mod = _get_torch()
        del model
        gc.collect()
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        # Cleanup should never raise; fallback is to proceed without cache purge
        pass


class _SnapshotRestoreFailed(RuntimeError):
    """Internal signal for snapshot restore failures during retries."""


def _should_measure_overhead(
    profile_normalized: str,
    cfg: InvarLockConfig | dict[str, Any] | None,
) -> tuple[bool, bool, str | None]:
    """Return overhead check policy resolved from profile + config context."""
    return _should_measure_overhead_impl(
        profile_normalized,
        cfg,
        coerce_mapping_fn=_coerce_mapping,
    )


def _resolve_snapshot_config(context: object | None) -> dict[str, Any]:
    return _resolve_snapshot_config_impl(
        context,
        to_serialisable_dict_fn=_to_serialisable_dict,
    )


def _estimate_model_bytes(model: object | None) -> int:
    return _estimate_model_bytes_impl(model)


def _choose_snapshot_mode(
    *,
    snapshot_config: Mapping[str, Any] | None,
    env_mode: str | None,
    supports_bytes: bool,
    supports_chunked: bool,
    estimated_model_mb: float,
    available_ram_mb: float,
    disk_free_mb: float,
    env_ram_fraction: str | None = None,
    env_threshold_mb: str | None = None,
) -> str:
    return _choose_snapshot_mode_impl(
        snapshot_config=snapshot_config,
        env_mode=env_mode,
        supports_bytes=supports_bytes,
        supports_chunked=supports_chunked,
        estimated_model_mb=estimated_model_mb,
        available_ram_mb=available_ram_mb,
        disk_free_mb=disk_free_mb,
        env_ram_fraction=env_ram_fraction,
        env_threshold_mb=env_threshold_mb,
    )


def _choose_dataset_split(
    *, requested: str | None, available: list[str] | None
) -> tuple[str, bool]:
    """Choose a dataset split deterministically."""
    return _choose_dataset_split_impl(
        requested=requested,
        available=available,
        split_aliases=SPLIT_ALIASES,
    )


def _persist_ref_masks(core_report: Any, run_dir: Path) -> Path | None:
    """Persist reference keep indices to artifact if present."""
    return _persist_ref_masks_impl(core_report, run_dir)


def _build_retry_result_summary(
    validation: Mapping[str, Any] | None,
) -> dict[str, object]:
    return _build_retry_result_summary_impl(validation)


def _decide_failed_retry_transition(
    retry_controller: Any,
    *,
    attempt: int,
    attempt_summary: Mapping[str, Any] | None,
    edit_config: Mapping[str, Any] | None,
    passed: bool = False,
) -> Any:
    return _decide_failed_retry_transition_impl(
        retry_controller,
        attempt=attempt,
        attempt_summary=attempt_summary,
        edit_config=edit_config,
        passed=passed,
    )


def _record_retry_attempt(
    retry_controller: Any,
    *,
    attempt: int,
    attempt_summary: Mapping[str, Any] | None,
    edit_config: Mapping[str, Any] | None,
) -> None:
    _record_retry_attempt_impl(
        retry_controller,
        attempt=attempt,
        attempt_summary=attempt_summary,
        edit_config=edit_config,
    )


def _build_restore_failure_attempt_summary() -> dict[str, Any]:
    return _build_restore_failure_attempt_summary_impl()


def _resolve_retry_validation_transition(
    retry_controller: Any,
    *,
    attempt: int,
    validation_result: Any,
    edit_config: Mapping[str, Any] | None,
) -> Any:
    return _resolve_retry_validation_transition_impl(
        retry_controller,
        attempt=attempt,
        validation_result=validation_result,
        edit_config=edit_config,
    )


def _build_timing_summary_payload(
    *,
    timings: Mapping[str, Any] | None,
    total_duration: float | None,
    report: Mapping[str, Any] | None,
) -> object | None:
    return _build_timing_summary_payload_impl(
        timings=timings,
        total_duration=total_duration,
        report=report,
    )


def _serialize_evaluation_windows(
    evaluation_windows: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    return _serialize_evaluation_windows_impl(evaluation_windows)


def _build_fallback_evaluation_windows(
    preview_records: Sequence[Mapping[str, Any]],
    final_records: Sequence[Mapping[str, Any]],
    *,
    use_mlm: bool,
    preview_mask_counts: Sequence[int] | None = None,
    final_mask_counts: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    return _build_fallback_evaluation_windows_impl(
        preview_records,
        final_records,
        use_mlm=use_mlm,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
    )


def _finalize_guard_overhead_payload(
    payload: Mapping[str, Any] | None,
    result: Any,
) -> dict[str, Any]:
    return _finalize_guard_overhead_payload_impl(payload, result)


def _prepare_guard_overhead_report(
    guard_overhead_payload: Mapping[str, Any] | None,
    *,
    resolved_loss_type: str | None,
    core_report: Any,
    report: Mapping[str, Any] | None,
    default_threshold: float,
) -> dict[str, Any]:
    return _prepare_guard_overhead_report_impl(
        guard_overhead_payload,
        resolved_loss_type=resolved_loss_type,
        core_report=core_report,
        report=report,
        default_threshold=default_threshold,
        extract_pm_snapshot_for_overhead_fn=_extract_pm_snapshot_for_overhead,
        validate_guard_overhead_fn=validate_guard_overhead,
    )


def _validate_pairing_report_metrics(
    metrics_section: Mapping[str, Any] | None,
    *,
    baseline_requested: bool,
    profile: str | None,
    preview_count_report: Any,
    final_count_report: Any,
    expected_preview: Any,
    expected_final: Any,
) -> list[Any]:
    return _validate_pairing_report_metrics_impl(
        metrics_section,
        baseline_requested=baseline_requested,
        profile=profile,
        preview_count_report=preview_count_report,
        final_count_report=final_count_report,
        expected_preview=expected_preview,
        expected_final=expected_final,
    )


def _build_dataset_window_stats(
    *,
    match_fraction: Any,
    overlap_fraction: Any,
    window_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _build_dataset_window_stats_impl(
        match_fraction=match_fraction,
        overlap_fraction=overlap_fraction,
        window_plan=window_plan,
    )


def _build_provider_dataset_plan(
    *,
    cfg: Any,
    model_profile: Any,
    console: Console,
    resolved_device: str | None,
    profile: str | None,
    profile_normalized: str | None,
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
    pairing_schedule_present: bool,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    resolved_loss_type: str,
    tier: str | None,
    get_provider_fn: Any,
) -> Any:
    return _build_provider_dataset_plan_impl(
        cfg=cfg,
        model_profile=model_profile,
        console=console,
        resolved_device=resolved_device,
        profile=profile,
        profile_normalized=profile_normalized,
        requested_preview=requested_preview,
        requested_final=requested_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
        pairing_schedule_present=pairing_schedule_present,
        use_mlm=use_mlm,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        resolved_loss_type=resolved_loss_type,
        tier=tier,
        get_provider_fn=get_provider_fn,
        resolve_provider_and_split_fn=_resolve_provider_and_split,
        resolve_tokenizer_fn=resolve_tokenizer,
        maybe_plan_release_windows_fn=_maybe_plan_release_windows,
        resolve_effective_windows_fn=_resolve_effective_windows,
        apply_mlm_masks_fn=_apply_mlm_masks,
        resolve_pm_min_tokens_target_fn=_resolve_pm_min_tokens_target,
        hash_sequences_fn=_hash_sequences,
        tokenizer_digest_fn=_tokenizer_digest,
        safe_int_fn=_safe_int,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
    )


def _build_run_context_payload(
    *,
    cfg: Any,
    profile: str | None,
    pairing_schedule: dict[str, Any] | None,
    seed_bundle: Mapping[str, Any],
    plugin_provenance: Mapping[str, Any],
    run_id: str,
    baseline_report_data: Mapping[str, Any] | None,
    pm_acceptance_range: tuple[float, float] | None,
    pm_drift_band: tuple[float, float] | None,
    guard_overhead_threshold: float,
    model_profile: Any,
    resolved_loss_type: str,
    tiny_relax_enabled: bool,
) -> dict[str, Any]:
    return _build_run_context_payload_impl(
        cfg=cfg,
        profile=profile,
        pairing_schedule=pairing_schedule,
        seed_bundle=seed_bundle,
        plugin_provenance=plugin_provenance,
        run_id=run_id,
        baseline_report_data=baseline_report_data,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        guard_overhead_threshold=guard_overhead_threshold,
        model_profile=model_profile,
        resolved_loss_type=resolved_loss_type,
        tiny_relax_enabled=tiny_relax_enabled,
        to_serialisable_dict_fn=_to_serialisable_dict,
    )


def _build_run_execution_config_payloads(
    *,
    cfg: Any,
    model_profile: Any,
) -> Any:
    return _build_run_execution_config_payloads_impl(
        cfg=cfg,
        model_profile=model_profile,
    )


def _enrich_run_report_metrics(
    *,
    report: dict[str, Any],
    core_report: Any,
    run_config: Any,
    cfg: Any,
    model_profile: Any,
    baseline_requested: bool,
    baseline_report_data: Mapping[str, Any] | None,
    metric_kind: str | None,
    resolved_loss_type: str,
    effective_preview: Any,
    effective_final: Any,
    profile_normalized: str | None,
    window_plan: Mapping[str, Any] | None,
    debug_metric_diffs_enabled: bool,
) -> Any:
    return _enrich_run_report_metrics_impl(
        report=report,
        core_report=core_report,
        run_config=run_config,
        cfg=cfg,
        model_profile=model_profile,
        baseline_requested=baseline_requested,
        baseline_report_data=baseline_report_data,
        metric_kind=metric_kind,
        resolved_loss_type=resolved_loss_type,
        effective_preview=effective_preview,
        effective_final=effective_final,
        profile_normalized=profile_normalized,
        window_plan=window_plan,
        debug_metric_diffs_enabled=debug_metric_diffs_enabled,
        resolve_metric_and_provider_fn=_resolve_metric_and_provider,
    )


def _validate_retry_evaluation_report(
    *,
    report: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    baseline_path: Path | None,
) -> Any:
    from invarlock.reporting.report_builder import make_report as _make_report
    from invarlock.reporting.report_telemetry import (
        telemetry_output_enabled as _telemetry_output_enabled,
    )
    from invarlock.reporting.report_telemetry import (
        telemetry_summary_line as _telemetry_summary_line,
    )

    return _validate_retry_evaluation_report_impl(
        report=report,
        baseline_report_data=baseline_report_data,
        baseline_path=baseline_path,
        build_retry_result_summary_fn=_build_retry_result_summary,
        make_report_fn=_make_report,
        telemetry_output_enabled_fn=_telemetry_output_enabled,
        telemetry_summary_line_fn=_telemetry_summary_line,
    )


def _adjust_edit_params(
    edit_name: str,
    edit_params: dict[str, Any],
    attempt: int,
    report_result: dict[str, Any] | None = None,
) -> Any:
    from invarlock.core.retry import adjust_edit_params

    return adjust_edit_params(edit_name, edit_params, attempt, report_result)


def _resolve_exit_code(exc: Exception, *, profile: str | None) -> int:
    """Resolve exit code based on exception type and profile.

    - ValueError("Invalid RunReport...") → 2 (schema/shape issue)
    - InvarlockError in CI/Release         → 3 (hard abort)
    - All other cases                  → 1 (generic failure)
    """
    return _resolve_command_exit_code(exc, profile=profile)


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
        torch_mod = _get_torch()
        if torch_mod is not None and hasattr(values, "tolist"):
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
    except (TypeError, ValueError, OverflowError):
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
    return _extract_pairing_schedule_impl(
        report,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
    )


def _load_baseline_pairing_evidence(
    *,
    baseline_path: Path,
    tokenizer_hash: str | None,
):
    return _load_baseline_pairing_evidence_impl(
        baseline_path=baseline_path,
        tokenizer_hash=tokenizer_hash,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )


def _materialize_baseline_pairing_schedule(
    *,
    pairing_schedule: dict[str, Any],
    calibration_data: list[dict[str, Any]] | None,
    dataset_meta: dict[str, Any],
    window_plan: dict[str, Any] | None,
    tokenizer: Any,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    resolved_tier: str | None,
    profile: str | None,
) -> Any:
    return _materialize_baseline_pairing_schedule_impl(
        pairing_schedule=pairing_schedule,
        calibration_data=calibration_data,
        dataset_meta=dataset_meta,
        window_plan=window_plan,
        tokenizer=tokenizer,
        use_mlm=use_mlm,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        resolved_tier=resolved_tier,
        profile=profile,
        apply_mlm_masks_fn=_apply_mlm_masks,
        resolve_pm_min_tokens_target_fn=_resolve_pm_min_tokens_target,
        hash_sequences_fn=_hash_sequences,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
    )


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
    from ...core.config_runtime import apply_edit_override as _apply_edit_override
    from ...core.config_runtime import apply_profile as _apply_profile
    from ...core.config_runtime import load_config as _load_config
    from ...core.config_runtime import resolve_edit_kind as _resolve_edit_kind

    try:
        from ...core.adapter_auto import apply_auto_adapter_if_needed as _apply_auto
    except Exception:  # pragma: no cover - optional adapter path
        _apply_auto = None

    return _prepare_config_for_run_impl(
        config_path=config_path,
        profile=profile,
        edit=edit,
        tier=tier,
        probes=probes,
        console=console,
        event_fn=_event,
        invarlock_config_cls=InvarLockConfig,
        load_config_fn=_load_config,
        apply_profile_fn=_apply_profile,
        resolve_edit_kind_fn=_resolve_edit_kind,
        apply_edit_override_fn=_apply_edit_override,
        apply_auto_adapter_fn=_apply_auto,
    )


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


def _resolve_effective_windows(
    *,
    data_provider: Any,
    tokenizer: Any,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    seed: int,
    split: str,
    requested_preview: int | None = None,
    requested_final: int | None = None,
    profile: str | None = None,
    signature_transform: Callable[
        [list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]
    ]
    | None = None,
    event_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    return _resolve_effective_windows_impl(
        data_provider=data_provider,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        seed=seed,
        split=split,
        requested_preview=requested_preview,
        requested_final=requested_final,
        profile=profile,
        release_min_windows_per_arm=RELEASE_MIN_WINDOWS_PER_ARM,
        signature_transform=signature_transform,
        event_fn=event_fn,
    )


def _resolve_pm_min_tokens_target(
    *,
    tier: str | None,
    profile: str | None,
) -> int:
    resolved = _resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        return int(pm_ratio.get("min_tokens", 0) or 0)
    except Exception:
        return 0


def _print_pipeline_start(console: Console) -> None:
    _event(console, "INIT", "Starting InvarLock pipeline...", emoji="🚀")


def _emit_run_artifacts(
    *, report: Any, out_dir: Path, filename_prefix: str, console: Console
) -> dict[str, str]:
    """Save run report and return emitted artifact paths."""
    from invarlock.reporting.report_files import save_report as _save_report

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

    return _resolve_device_and_output_impl(
        cfg,
        device=device,
        out=out,
        console=console,
        event_fn=_event,
        format_kv_line_fn=_format_kv_line,
        device_resolution_note_fn=_device_resolution_note,
        resolve_device_fn=_resolve_device,
        validate_device_fn=_validate,
    )


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
    return _resolve_provider_and_split_impl(
        cfg,
        model_profile,
        get_provider_fn=get_provider_fn,
        choose_dataset_split_fn=_choose_dataset_split,
        provider_kwargs=provider_kwargs,
        resolved_device=resolved_device,
        emit=emit,
    )


def _extract_model_load_kwargs(cfg: InvarLockConfig) -> dict[str, Any]:
    """Return adapter.load_model kwargs from config (excluding core fields)."""
    return _extract_model_load_kwargs_impl(cfg, invarlock_error_cls=InvarlockError)


def _load_model_with_cfg(
    adapter: Any,
    cfg: InvarLockConfig,
    device: str,
    *,
    profile: str | None = None,
    event_path: Path | None = None,
    warning_context: dict[str, Any] | None = None,
    prefer_local_files_only: bool = False,
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
    with _suppress_noisy_warnings(
        profile,
        event_path=event_path,
        context=warning_context,
    ):
        strict_accepts_local_files_only = False
        try:
            sig = inspect.signature(adapter.load_model)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_var_kw:
                allowed = dict(extra)
                if prefer_local_files_only:
                    allowed["prefer_local_files_only"] = True
                return adapter.load_model(model_id, device=device, **allowed)
            allowed = {k: v for k, v in extra.items() if k in sig.parameters}
            strict_accepts_local_files_only = (
                "prefer_local_files_only" in sig.parameters
            )
            if prefer_local_files_only and strict_accepts_local_files_only:
                allowed["prefer_local_files_only"] = True
            if allowed:
                return adapter.load_model(model_id, device=device, **allowed)
        except Exception:
            # Fall back to the strictest call shape.
            pass
        if prefer_local_files_only and strict_accepts_local_files_only:
            return adapter.load_model(
                model_id, device=device, prefer_local_files_only=True
            )
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
    overhead_threshold: float = GUARD_OVERHEAD_THRESHOLD,
    profile_normalized: str | None = None,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
    prefer_local_files_only: bool = False,
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
                adapter,
                cfg,
                resolved_device,
                profile=profile_normalized,
                prefer_local_files_only=prefer_local_files_only,
            )
            private_model_loaded = True
            if snapshot_provenance is not None:
                snapshot_provenance["reload_path_used"] = True

        with _suppress_noisy_warnings(
            profile_normalized,
            event_path=getattr(run_config, "event_path", None),
            context={"phase": "guard_overhead_bare"},
        ):
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
        "overhead_threshold": float(overhead_threshold),
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
    prefer_local_files_only: bool = False,
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
        warning_context: dict[str, Any] = {"phase": "load_model"}
        try:
            if hasattr(run_config, "context") and isinstance(run_config.context, dict):
                rid = run_config.context.get("run_id")
                if isinstance(rid, str) and rid:
                    warning_context["run_id"] = rid
        except Exception:
            pass
        model = _load_model_with_cfg(
            adapter,
            cfg,
            resolved_device,
            profile=profile_normalized,
            event_path=getattr(run_config, "event_path", None),
            warning_context=warning_context,
            prefer_local_files_only=prefer_local_files_only,
        )
        if snapshot_provenance is not None:
            snapshot_provenance["reload_path_used"] = True

    runtime_edit_config = dict(edit_config or {})
    runtime_edit_config.setdefault("console", console)
    runtime_edit_config.setdefault(
        "output_style", _style_from_console(console, profile=profile_normalized)
    )
    runtime_edit_config.setdefault("emit", True)

    with _suppress_noisy_warnings(
        profile_normalized,
        event_path=getattr(run_config, "event_path", None),
        context={"phase": "core_runner_execute"},
    ):
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
    console: Console,
) -> dict[str, str]:
    saved_files = _emit_run_artifacts(
        report=report, out_dir=run_dir, filename_prefix="report", console=console
    )
    _event(console, "PASS", "Run completed successfully!", emoji="✅")
    _event(console, "DATA", f"Report: {saved_files['json']}", emoji="📄")
    if run_config.event_path:
        _event(console, "DATA", f"Events: {run_config.event_path}", emoji="📝")
    return saved_files


def _compute_provider_digest(report: dict[str, Any]) -> dict[str, str] | None:
    return _compute_provider_digest_impl(
        report,
        compute_mask_positions_digest_fn=_compute_mask_positions_digest,
    )


def _finalize_run_provenance(
    *,
    report: dict[str, Any],
    core_report: Any,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    use_mlm: bool,
    preview_mask_counts: list[int] | None,
    final_mask_counts: list[int] | None,
    had_baseline: bool,
    profile: str | None,
    resolved_split: str | None,
    used_fallback_split: bool,
    baseline_report_data: dict[str, Any] | None,
) -> Any:
    return _finalize_run_provenance_impl(
        report=report,
        core_report=core_report,
        preview_records=preview_records,
        final_records=final_records,
        use_mlm=use_mlm,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        had_baseline=had_baseline,
        profile=profile,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
        baseline_report_data=baseline_report_data,
        serialize_evaluation_windows_fn=_serialize_evaluation_windows,
        build_fallback_evaluation_windows_fn=_build_fallback_evaluation_windows,
        compute_provider_digest_fn=_compute_provider_digest,
        enforce_provider_parity_fn=_enforce_provider_parity,
    )


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
    return _validate_and_harvest_baseline_schedule_impl(
        cfg,
        pairing_schedule,
        baseline_report_data,
        tokenizer_hash=tokenizer_hash,
        resolved_loss_type=resolved_loss_type,
        profile=profile,
        baseline_path_str=baseline_path_str,
        console=console,
        event_fn=_event,
        canonical_dataset_id_fn=_canonical_dataset_id,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
        hash_sequences_fn=_hash_sequences,
        invarlock_error_cls=InvarlockError,
    )


def _enforce_provider_parity(
    subject_digest: dict | None, baseline_digest: dict | None, *, profile: str | None
) -> None:
    _enforce_provider_parity_impl(
        subject_digest,
        baseline_digest,
        profile=profile,
        invarlock_error_cls=InvarlockError,
    )


def _resolve_metric_and_provider(
    cfg: Any,
    model_profile: Any,
    *,
    resolved_loss_type: str | None = None,
    metric_kind_override: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    return _resolve_metric_and_provider_impl(
        cfg,
        model_profile,
        resolved_loss_type=resolved_loss_type,
        metric_kind_override=metric_kind_override,
    )


def _plan_release_windows(
    capacity: dict[str, Any],
    *,
    requested_preview: int,
    requested_final: int,
    max_calibration: int,
    console: Console | None = None,
) -> dict[str, Any]:
    return _plan_release_windows_impl(
        capacity,
        requested_preview=requested_preview,
        requested_final=requested_final,
        max_calibration=max_calibration,
        console=console,
        event_fn=_event,
    )


# Check if core components are available
try:
    from invarlock.core.api import RunConfig  # noqa: F401
    from invarlock.core.registry import get_registry  # noqa: F401

    HAS_CORE_COMPONENTS = True
except ImportError:
    HAS_CORE_COMPONENTS = False


def _build_run_command_deps() -> dict[str, Any]:
    """Build explicit dependencies for run_command_impl.

    Passing an explicit map avoids dynamic module globals mutation while keeping
    monkeypatch behavior stable (resolved at call time).
    """

    _reset_optional_runtime_caches()

    return {
        "ConfigError": _CfgErr,
        "InvarlockError": InvarlockError,
        "Path": Path,
        "RELEASE_MIN_WINDOWS_PER_ARM": RELEASE_MIN_WINDOWS_PER_ARM,
        "_SnapshotRestoreFailed": _SnapshotRestoreFailed,
        "_apply_mlm_masks": _apply_mlm_masks,
        "_apply_warning_filters": _apply_warning_filters,
        "_build_artifacts_payload": _build_artifacts_payload_impl,
        "_build_provider_dataset_plan": _build_provider_dataset_plan,
        "_build_run_context_payload": _build_run_context_payload,
        "_build_run_execution_config_payloads": _build_run_execution_config_payloads,
        "_enrich_run_report_metrics": _enrich_run_report_metrics,
        "_validate_retry_evaluation_report": _validate_retry_evaluation_report,
        "_build_dataset_window_stats": _build_dataset_window_stats,
        "_canonical_dataset_id": _canonical_dataset_id,
        "_coerce_float": _coerce_float,
        "_coerce_int": _coerce_int,
        "_coerce_option": _coerce_option,
        "_compute_provider_digest": _compute_provider_digest,
        "_finalize_run_provenance": _finalize_run_provenance,
        "_build_edit_payload": _build_edit_payload_impl,
        "_enforce_provider_parity": _enforce_provider_parity,
        "_event": _event,
        "_execute_guarded_run": _execute_guarded_run,
        "_extract_pairing_schedule": _extract_pairing_schedule,
        "_load_baseline_pairing_evidence": _load_baseline_pairing_evidence,
        "_materialize_baseline_pairing_schedule": _materialize_baseline_pairing_schedule,
        "_extract_pm_snapshot_for_overhead": _extract_pm_snapshot_for_overhead,
        "_format_debug_metric_diffs": _format_debug_metric_diffs,
        "_format_guard_chain": _format_guard_chain,
        "_format_kv_line": _format_kv_line,
        "_free_model_memory": _free_model_memory,
        "_hash_sequences": _hash_sequences,
        "_init_retry_controller": _init_retry_controller,
        "_load_model_with_cfg": _load_model_with_cfg,
        "_maybe_plan_release_windows": _maybe_plan_release_windows,
        "_build_flags_payload": _build_flags_payload_impl,
        "_build_guard_entries": _build_guard_entries_impl,
        "_build_metrics_payload": _build_metrics_payload_impl,
        "_build_run_report_context": _build_run_report_context_impl,
        "_build_run_report_data": _build_run_report_data_impl,
        "_build_run_report_meta": _build_run_report_meta_impl,
        "_resolve_effective_windows": _resolve_effective_windows,
        "_resolve_pm_min_tokens_target": _resolve_pm_min_tokens_target,
        "_merge_primary_metric_health": _merge_primary_metric_health,
        "_merge_core_timing_metrics": _merge_core_timing_metrics_impl,
        "_normalize_overhead_result": _normalize_overhead_result,
        "_build_timing_summary_payload": _build_timing_summary_payload,
        "_build_restore_failure_attempt_summary": _build_restore_failure_attempt_summary,
        "_record_retry_attempt": _record_retry_attempt,
        "_decide_failed_retry_transition": _decide_failed_retry_transition,
        "_resolve_retry_validation_transition": _resolve_retry_validation_transition,
        "_adjust_edit_params": _adjust_edit_params,
        "_build_fallback_evaluation_windows": _build_fallback_evaluation_windows,
        "_choose_snapshot_mode": _choose_snapshot_mode,
        "_estimate_model_bytes": _estimate_model_bytes,
        "_finalize_guard_overhead_payload": _finalize_guard_overhead_payload,
        "_persist_ref_masks": _persist_ref_masks,
        "_postprocess_and_summarize": _postprocess_and_summarize,
        "_prepare_guard_overhead_report": _prepare_guard_overhead_report,
        "_prepare_config_for_run": _prepare_config_for_run,
        "_print_guard_overhead_summary": _print_guard_overhead_summary,
        "_print_pipeline_start": _print_pipeline_start,
        "_print_retry_summary": _print_retry_summary,
        "_resolve_device_and_output": _resolve_device_and_output,
        "_resolve_exit_code": _resolve_exit_code,
        "_resolve_guard_overhead_threshold": _resolve_guard_overhead_threshold,
        "_resolve_metric_and_provider": _resolve_metric_and_provider,
        "_resolve_pm_acceptance_range": _resolve_pm_acceptance_range,
        "_resolve_pm_drift_band": _resolve_pm_drift_band,
        "_resolve_provider_and_split": _resolve_provider_and_split,
        "_resolve_snapshot_config": _resolve_snapshot_config,
        "_run_bare_control": _run_bare_control,
        "_safe_int": _safe_int,
        "_serialize_evaluation_windows": _serialize_evaluation_windows,
        "_should_measure_overhead": _should_measure_overhead,
        "_style_from_console": _style_from_console,
        "_tensor_or_list_to_ints": _tensor_or_list_to_ints,
        "_to_serialisable_dict": _to_serialisable_dict,
        "_tokenizer_digest": _tokenizer_digest,
        "_build_snapshot_provenance": _build_snapshot_provenance_impl,
        "_validate_pairing_report_metrics": _validate_pairing_report_metrics,
        "_validate_and_harvest_baseline_schedule": _validate_and_harvest_baseline_schedule,
        "click": click,
        "console": console,
        "copy": copy,
        "datetime": datetime,
        "detect_model_profile": detect_model_profile,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": np,
        "os": os,
        "perf_counter": perf_counter,
        "print_timing_summary": print_timing_summary,
        "get_psutil": _get_psutil,
        "resolve_output_style": resolve_output_style,
        "resolve_tokenizer": resolve_tokenizer,
        "set_seed": set_seed,
        "shutil": shutil,
        "timed_step": timed_step,
        "get_torch": _get_torch,
        "typer": typer,
        "validate_guard_overhead": validate_guard_overhead,
    }


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
    edit: str | None = typer.Option(
        None,
        "--edit",
        help="Edit name override (canonical plugin name, e.g. quant_rtn)",
    ),
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
        False,
        "--until-pass",
        help="Retry until evaluation report passes gates (max 3 attempts)",
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
        help="Path to baseline report.json for evaluation report validation",
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
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Explicitly allow outbound network access for this command.",
    ),
    allow_host_execution: bool = typer.Option(
        False,
        "--allow-host-execution",
        help="Run on the host instead of auto-delegating to the runtime container.",
    ),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Enable third-party entry-point plugin discovery for this command.",
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="Allow trust_remote_code-style model loading for this command.",
    ),
    prefer_local_files_only: bool = typer.Option(
        False,
        hidden=True,
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    """
    Run InvarLock pipeline with the given configuration.

    The command assembles non-overlapping preview/final windows, executes the
    GuardChain (invariants → spectral → RMT → variance), checks pairing/overlap
    invariants, enforces the configured guard-overhead budget (default ≤1 %),
    and emits a run report plus JSONL
    events suitable for evaluation report generation.
    """
    allow_network = bool(_coerce_option(allow_network, False))
    allow_host_execution = bool(_coerce_option(allow_host_execution, False))
    allow_third_party_plugins = bool(_coerce_option(allow_third_party_plugins, False))
    allow_remote_code = bool(_coerce_option(allow_remote_code, False))
    prefer_local_files_only = bool(_coerce_option(prefer_local_files_only, False))
    try:
        return run_from_config(
            config=config,
            device=device,
            profile=profile,
            out=out,
            edit=edit,
            edit_label=edit_label,
            tier=tier,
            metric_kind=metric_kind,
            probes=probes,
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
            no_cleanup=no_cleanup,
            style=style,
            progress=progress,
            timing=timing,
            telemetry=telemetry,
            no_color=no_color,
            allow_network=allow_network,
            allow_host_execution=allow_host_execution,
            allow_third_party_plugins=allow_third_party_plugins,
            allow_remote_code=allow_remote_code,
            prefer_local_files_only=prefer_local_files_only,
            command_name="run",
            run_impl=_run_command_impl,
            deps_builder=_build_run_command_deps,
        )
    except RuntimeDelegationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _merge_primary_metric_health(
    primary_metric: dict[str, Any] | None,
    core_primary_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    return _merge_primary_metric_health_impl(primary_metric, core_primary_metric)


def _format_debug_metric_diffs(
    pm: dict[str, float] | None,
    metrics: dict[str, float] | None,
    baseline_report_data: dict | None,
) -> str:
    return _format_debug_metric_diffs_impl(pm, metrics, baseline_report_data)


def _normalize_overhead_result(
    payload: dict[str, object] | None, profile: str | None = None
) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny/degenerate runs."""
    _ = profile
    return _normalize_overhead_result_impl(payload)


# helper moved to invarlock.cli.overhead_utils


def _print_guard_overhead_summary(
    console: Console,
    guard_overhead_info: dict[str, Any],
    *,
    default_threshold: float = GUARD_OVERHEAD_THRESHOLD,
) -> float:
    """Print a concise guard-overhead console summary. Returns threshold fraction used."""
    summary = _build_guard_overhead_summary_impl(
        guard_overhead_info,
        default_threshold=default_threshold,
    )
    if not summary.evaluated:
        _event(console, "METRIC", "Guard Overhead: not evaluated", emoji="🛡️")
        return summary.threshold_fraction
    _event(
        console,
        "METRIC",
        f"Guard Overhead: {summary.status} {summary.overhead_display} ({summary.threshold_display})",
        emoji="🛡️",
    )
    return summary.threshold_fraction


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
