"""Runtime-only execution helpers for config-driven run commands."""

from __future__ import annotations

import builtins
import copy
import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys as _sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from ctypes import CDLL, c_int, c_size_t
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from invarlock.core.backend_inventory import (
    BACKEND_INVENTORY_FILENAME,
    build_backend_inventory_for_adapter,
)
from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_policy import GUARD_OVERHEAD_THRESHOLD
from invarlock.core.run_snapshot_contract import (
    build_snapshot_execution_plan as _build_snapshot_execution_plan_impl,
)
from invarlock.core.run_snapshot_contract import (
    choose_snapshot_mode as _choose_snapshot_mode_impl,
)
from invarlock.core.run_snapshot_contract import (
    estimate_model_bytes as _estimate_model_bytes_impl,
)

_NOISY_WARNING_PATTERNS = (r".*loss_type=None.*unrecognized.*",)
_LOG_MESSAGE_ERRORS = (RuntimeError, TypeError, ValueError)


def _import_optional_module(name: str) -> Any:
    try:
        return builtins.__import__(name)
    except ImportError:
        return None


def get_psutil() -> Any:
    global psutil
    if psutil is None or isinstance(psutil, ModuleType):
        psutil = _import_optional_module("psutil")
    return psutil


def get_torch() -> Any:
    global torch
    if torch is None or isinstance(torch, ModuleType):
        torch = _import_optional_module("torch")
    return torch


psutil: Any = _import_optional_module("psutil")
torch: Any = _import_optional_module("torch")


def reset_optional_runtime_caches() -> None:
    global psutil, torch
    if psutil is None:
        psutil = _import_optional_module("psutil")
    if torch is None:
        torch = _import_optional_module("torch")


def _malloc_trim() -> bool:
    try:
        libc = CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return False
        trim.argtypes = [c_size_t]
        trim.restype = c_int
        return bool(trim(0))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def release_process_memory() -> None:
    """Best-effort process-wide memory trim after heavyweight model work."""
    try:
        gc.collect()
    except (RuntimeError, TypeError, ValueError):
        pass
    try:
        torch_mod = get_torch()
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except (RuntimeError, TypeError, ValueError, AttributeError):
        pass
    try:
        _malloc_trim()
    except (RuntimeError, TypeError, ValueError, AttributeError, OSError):
        pass


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


def free_model_memory(model: object | None) -> None:
    """Best-effort cleanup to release GPU memory for a model object."""
    if model is None:
        return
    try:
        del model
        release_process_memory()
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        # Cleanup should never raise; fallback is to proceed without cache purge.
        return


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


class FilteredWarningStream:
    """File-like wrapper that suppresses matched warning chunks."""

    def __init__(self, raw: Any, patterns: Sequence[re.Pattern[str]], sink: list[str]):
        self._raw = raw
        self._patterns = patterns
        self._sink = sink

    @property
    def encoding(self) -> str | None:
        return getattr(self._raw, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._raw, "errors", None)

    @property
    def buffer(self) -> Any:
        return self._raw.buffer

    @property
    def closed(self) -> bool:
        return bool(getattr(self._raw, "closed", False))

    def fileno(self) -> int:
        return int(self._raw.fileno())

    def isatty(self) -> bool:
        return bool(self._raw.isatty())

    def writable(self) -> bool:
        return bool(getattr(self._raw, "writable", lambda: True)())

    def write(self, s: object) -> int:
        try:
            if isinstance(s, bytes):
                text = s.decode("utf-8", errors="replace")
            else:
                text = str(s)
        except (TypeError, ValueError, UnicodeDecodeError):
            return int(self._raw.write(s))

        pieces = text.splitlines(keepends=True)
        for piece in pieces:
            if any(pattern.search(piece) for pattern in self._patterns):
                self._sink.append(piece.rstrip("\n"))
                continue
            self._raw.write(piece)
        return len(text)

    def flush(self) -> None:
        try:
            self._raw.flush()
        except (AttributeError, OSError, ValueError):
            pass


@contextmanager
def suppress_noisy_warnings(
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
            except _LOG_MESSAGE_ERRORS:
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

    def _append_suppressed_warnings() -> None:
        if not suppressed or event_path is None:
            return
        try:
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
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except (OSError, TypeError, ValueError):
            return

    log_filter = _NoisyLogFilter()
    handlers = _iter_handlers()
    for handler in handlers:
        handler.addFilter(log_filter)

    try:
        with warnings.catch_warnings():
            from contextlib import redirect_stderr, redirect_stdout

            stdout_proxy = FilteredWarningStream(_sys.stdout, patterns, suppressed)
            stderr_proxy = FilteredWarningStream(_sys.stderr, patterns, suppressed)

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

                    showwarning_fn: Any = _showwarning
                    warnings.showwarning = showwarning_fn
                    try:
                        yield
                    finally:
                        warnings.showwarning = original_showwarning
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


class SnapshotRestoreFailed(RuntimeError):
    """Internal signal for snapshot restore failures during retries."""


_SNAPSHOT_RESTORE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _require_snapshot_reuse_model(*, model: Any, phase: str) -> Any:
    if model is None:
        raise SnapshotRestoreFailed(
            f"Snapshot reuse requested for {phase} without a live model instance."
        )
    return model


def _capture_backend_inventory(
    *,
    adapter: Any,
    cfg: Any,
    model: Any,
    run_config: Any,
) -> None:
    try:
        from invarlock.cli.run_config import extract_model_load_kwargs

        load_kwargs = extract_model_load_kwargs(
            cfg,
            invarlock_error_cls=InvarlockError,
        )
    except (AttributeError, KeyError, TypeError, ValueError, InvarlockError):
        load_kwargs = {}
    quantization_config = load_kwargs.get("quantization_config")
    adapter_name = str(getattr(adapter, "name", "") or "")
    inventory = build_backend_inventory_for_adapter(
        adapter=adapter_name,
        quantization_config=(
            quantization_config if isinstance(quantization_config, dict) else {}
        ),
        model=model,
        load_smoke=True,
        inference_smoke=False,
    )
    if inventory is None:
        return
    context = getattr(run_config, "context", None)
    if isinstance(context, dict):
        context["_backend_inventory"] = inventory
    event_path = getattr(run_config, "event_path", None)
    if event_path is None:
        return
    try:
        output_dir = Path(event_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / BACKEND_INVENTORY_FILENAME).write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        return


def build_snapshot_execution_plan(
    *,
    adapter: Any,
    model: Any,
    cfg_snapshot: dict[str, Any] | None,
    direct_reuse_loaded_model: bool,
    skip_overhead_source: str | None,
) -> Any:
    return _build_snapshot_execution_plan_impl(
        adapter=adapter,
        model=model,
        cfg_snapshot=cfg_snapshot,
        direct_reuse_loaded_model=direct_reuse_loaded_model,
        skip_overhead_source=skip_overhead_source,
        choose_snapshot_mode_fn=_choose_snapshot_mode_impl,
        estimate_model_bytes_fn=_estimate_model_bytes_impl,
        psutil_module=get_psutil(),
        environ=os.environ,
        disk_usage_fn=shutil.disk_usage,
        free_model_memory_fn=free_model_memory,
    )


def load_model_with_cfg(
    adapter: Any,
    cfg: Any,
    device: str,
    *,
    profile: str | None = None,
    event_path: Any | None = None,
    warning_context: dict[str, Any] | None = None,
    prefer_local_files_only: bool = False,
) -> Any:
    """Load a model with config-provided kwargs, filtering for strict adapters."""
    try:
        model_id = cfg.model.id
    except (AttributeError, KeyError, TypeError):
        try:
            model_id = (cfg.model_dump().get("model") or {}).get("id")
        except (AttributeError, KeyError, TypeError, ValueError):
            model_id = None
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Missing model.id in config")

    from invarlock.cli.run_config import extract_model_load_kwargs

    extra = extract_model_load_kwargs(cfg, invarlock_error_cls=InvarlockError)
    with suppress_noisy_warnings(
        profile,
        event_path=event_path,
        context=warning_context,
    ):
        try:
            sig = inspect.signature(adapter.load_model)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            strict_accepts_local_files_only = (
                "prefer_local_files_only" in sig.parameters
            )
            if accepts_var_kw:
                allowed = dict(extra)
                if prefer_local_files_only:
                    allowed["prefer_local_files_only"] = True
                return adapter.load_model(model_id, device=device, **allowed)
            allowed = {k: v for k, v in extra.items() if k in sig.parameters}
            if prefer_local_files_only and strict_accepts_local_files_only:
                allowed["prefer_local_files_only"] = True
            if allowed:
                return adapter.load_model(model_id, device=device, **allowed)
            if prefer_local_files_only and strict_accepts_local_files_only:
                return adapter.load_model(
                    model_id, device=device, prefer_local_files_only=True
                )

        if prefer_local_files_only and sig is None:
            return adapter.load_model(
                model_id, device=device, prefer_local_files_only=True
            )
        return adapter.load_model(model_id, device=device)


def init_retry_controller(
    *,
    until_pass: bool,
    max_attempts: int,
    timeout: int | None,
    baseline: str | None,
) -> Any:
    """Initialize RetryController for owner-managed retry handling."""
    del baseline
    retry_controller = None
    if until_pass:
        from invarlock.core.retry import RetryController

        retry_controller = RetryController(
            max_attempts=max_attempts, timeout=timeout, verbose=True
        )
    return retry_controller


def run_bare_control(
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
    resolved_loss_type: str,
    overhead_threshold: float = GUARD_OVERHEAD_THRESHOLD,
    profile_normalized: str | None = None,
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
    prefer_local_files_only: bool = False,
) -> dict[str, Any] | None:
    """Execute the bare-control run for overhead estimation and return payload."""
    from invarlock.cli.run_overhead import _extract_pm_snapshot_for_overhead
    from invarlock.core.api import EditRuntime
    from invarlock.core.determinism_policy import set_seed
    from invarlock.core.runner import CoreRunner

    python_seed = seed_bundle.get("python")
    if isinstance(python_seed, int):
        set_seed(python_seed)

    bare_runner = CoreRunner()
    bare_config = copy.deepcopy(run_config)
    bare_config.event_path = None
    bare_context = copy.deepcopy(run_config.context)
    bare_context.setdefault("validation", {})["guard_overhead_mode"] = "bare"
    bare_config.context = bare_context
    edit_runtime = EditRuntime(
        profile=profile_normalized,
        verbose=bool(getattr(run_config, "verbose", False)),
    )
    _capture_backend_inventory(
        adapter=adapter,
        cfg=cfg,
        model=model,
        run_config=run_config,
    )

    private_model_loaded = False
    bare_target_model = None
    try:
        if restore_fn and model is not None:
            try:
                restore_fn()
            except _SNAPSHOT_RESTORE_EXCEPTIONS as exc:
                raise SnapshotRestoreFailed(str(exc)) from exc
            bare_target_model = model
        elif skip_model_load:
            bare_target_model = _require_snapshot_reuse_model(
                model=model,
                phase="bare control",
            )
        else:
            bare_target_model = load_model_with_cfg(
                adapter,
                cfg,
                resolved_device,
                profile=profile_normalized,
                prefer_local_files_only=prefer_local_files_only,
            )
            private_model_loaded = True
            if snapshot_provenance is not None:
                snapshot_provenance["reload_path_used"] = True

        with suppress_noisy_warnings(
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
                edit_config=dict(edit_config or {}),
                edit_runtime=edit_runtime,
                preview_n=preview_count,
                final_n=final_count,
            )
    finally:
        if private_model_loaded:
            free_model_memory(bare_target_model)

    bare_ppl_final = None
    bare_ppl_preview = None
    if hasattr(bare_report, "metrics") and bare_report.metrics:
        bare_pm = bare_report.metrics.get("primary_metric", {})
        bare_ppl_final = bare_pm.get("final") if isinstance(bare_pm, dict) else None
        bare_ppl_preview = bare_pm.get("preview") if isinstance(bare_pm, dict) else None

    payload: dict[str, Any] = {
        "overhead_threshold": float(overhead_threshold),
        "diagnostics": [],
        "checks": {},
        "source": f"{profile_normalized or 'ci'}_profile",
        "mode": "bare",
    }

    if profile_normalized in {"ci", "release"}:

        def _finite(x: Any) -> bool:
            try:
                return isinstance(x, int | float) and math.isfinite(float(x))
            except (TypeError, ValueError):
                return False

        if not (_finite(bare_ppl_preview) and _finite(bare_ppl_final)):
            payload["diagnostics"].append(
                {
                    "kind": "guard_overhead_warning",
                    "severity": "warning",
                    "message": (
                        "Primary metric non-finite during bare control; continuing with "
                        "diagnostics."
                    ),
                    "details": {},
                }
            )

    if getattr(bare_report, "status", "").lower() not in {"success", "completed", "ok"}:
        payload["diagnostics"].append(
            {
                "kind": "guard_overhead_warning",
                "severity": "warning",
                "message": f"Bare run status: {getattr(bare_report, 'status', 'unknown')}",
                "details": {},
            }
        )

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
    else:
        payload["diagnostics"].append(
            {
                "kind": "guard_overhead_warning",
                "severity": "warning",
                "message": "Bare control primary metric unavailable for overhead diagnostics.",
                "details": {},
            }
        )

    if isinstance(python_seed, int):
        set_seed(python_seed)
    return payload


def execute_guarded_run(
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
    snapshot_provenance: dict[str, bool] | None = None,
    skip_model_load: bool = False,
    prefer_local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Restore or load model and execute the guarded CoreRunner."""
    from invarlock.core.api import EditRuntime

    if restore_fn and model is not None:
        try:
            restore_fn()
        except _SNAPSHOT_RESTORE_EXCEPTIONS as exc:
            raise SnapshotRestoreFailed(str(exc)) from exc
    elif skip_model_load:
        model = _require_snapshot_reuse_model(
            model=model,
            phase="guarded execution",
        )
    else:
        warning_context: dict[str, Any] = {"phase": "load_model"}
        try:
            if hasattr(run_config, "context") and isinstance(run_config.context, dict):
                rid = run_config.context.get("run_id")
                if isinstance(rid, str) and rid:
                    warning_context["run_id"] = rid
        except (AttributeError, TypeError):
            pass
        model = load_model_with_cfg(
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

    edit_runtime = EditRuntime(
        profile=profile_normalized,
        verbose=bool(getattr(run_config, "verbose", False)),
    )
    _capture_backend_inventory(
        adapter=adapter,
        cfg=cfg,
        model=model,
        run_config=run_config,
    )

    with suppress_noisy_warnings(
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
            edit_config=dict(edit_config or {}),
            edit_runtime=edit_runtime,
            preview_n=preview_count,
            final_n=final_count,
        )
    return core_report, model


__all__ = [
    "FilteredWarningStream",
    "GUARD_OVERHEAD_THRESHOLD",
    "SnapshotRestoreFailed",
    "_apply_warning_filters",
    "_resolve_warning_suppression",
    "build_snapshot_execution_plan",
    "detect_model_profile",
    "execute_guarded_run",
    "free_model_memory",
    "get_psutil",
    "get_torch",
    "init_retry_controller",
    "load_model_with_cfg",
    "release_process_memory",
    "reset_optional_runtime_caches",
    "resolve_tokenizer",
    "run_bare_control",
    "suppress_noisy_warnings",
    "validate_guard_overhead",
]
