"""Deterministic run-policy and config-resolution helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.core.exceptions import ConfigError, InvarlockError
from invarlock.core.run_policy_timing import (
    TimingSummaryPayload as TimingSummaryPayload,
)
from invarlock.core.run_policy_timing import (
    build_timing_summary_payload as build_timing_summary_payload,
)
from invarlock.core.run_policy_windows import (
    _fallback_window_payload as _fallback_window_payload,
)
from invarlock.core.run_policy_windows import (
    _is_sequence_payload as _is_sequence_payload,
)
from invarlock.core.run_policy_windows import _list_payload as _list_payload
from invarlock.core.run_policy_windows import (
    _nested_list_payload as _nested_list_payload,
)
from invarlock.core.run_policy_windows import _token_count as _token_count
from invarlock.core.run_policy_windows import _window_payload as _window_payload
from invarlock.core.run_policy_windows import (
    build_fallback_evaluation_windows as build_fallback_evaluation_windows,
)
from invarlock.core.run_policy_windows import (
    serialize_evaluation_windows as serialize_evaluation_windows,
)

GUARD_OVERHEAD_THRESHOLD = 0.01

ToSerialisableDictFn = Callable[[Any], dict[str, Any]]


def enforce_provider_parity(
    subject_digest: dict | None,
    baseline_digest: dict | None,
    *,
    profile: str | None,
    invarlock_error_cls: type[InvarlockError] = InvarlockError,
) -> None:
    """Enforce tokenizer/masking parity rules for CI and release profiles."""

    prof = (profile or "").strip().lower()
    if prof not in {"ci", "release"}:
        return

    subject = subject_digest or {}
    baseline = baseline_digest or {}
    subj_ids = subject.get("ids_sha256")
    base_ids = baseline.get("ids_sha256")
    subj_tok = subject.get("tokenizer_sha256")
    base_tok = baseline.get("tokenizer_sha256")
    subj_proc = subject.get("processor_sha256")
    base_proc = baseline.get("processor_sha256")
    subj_mask = subject.get("masking_sha256")
    base_mask = baseline.get("masking_sha256")
    subject_surface = subj_tok if isinstance(subj_tok, str) and subj_tok else subj_proc
    baseline_surface = base_tok if isinstance(base_tok, str) and base_tok else base_proc

    if not (
        isinstance(subj_ids, str)
        and isinstance(base_ids, str)
        and subj_ids
        and base_ids
        and isinstance(subject_surface, str)
        and isinstance(baseline_surface, str)
        and subject_surface
        and baseline_surface
    ):
        raise invarlock_error_cls(
            code="E004",
            message="PROVIDER-DIGEST-MISSING: subject or baseline missing ids/model-surface digest",
        )

    if subj_ids != base_ids:
        raise invarlock_error_cls(
            code="E006",
            message="IDS-DIGEST-MISMATCH: subject and baseline window IDs differ",
        )

    if subject_surface != baseline_surface:
        raise invarlock_error_cls(
            code="E002",
            message=(
                "TOKENIZER-DIGEST-MISMATCH: subject and baseline tokenization/processor "
                "surfaces differ"
            ),
        )

    if (
        isinstance(subj_mask, str)
        and isinstance(base_mask, str)
        and subj_mask
        and base_mask
        and subj_mask != base_mask
    ):
        raise invarlock_error_cls(
            code="E003",
            message="MASK-PARITY-MISMATCH: mask positions differ under matched tokenizers",
        )


@dataclass(frozen=True)
class RunExecutionRequest:
    """Typed request contract for config-driven run execution."""

    config: str
    device: str | None = None
    profile: str | None = None
    out: str | None = None
    edit: str | None = None
    edit_label: str | None = None
    tier: str | None = None
    metric_kind: str | None = None
    probes: int | None = None
    until_pass: bool = False
    max_attempts: int = 3
    timeout: int | None = None
    baseline: str | None = None
    no_cleanup: bool = False
    capture_timings: bool = False
    telemetry: bool = False
    prefer_local_files_only: bool = False
    eval_device_override: str | None = None
    determinism_mode: str | None = None
    determinism_warn_only: bool = False
    tiny_relax_enabled: bool = False
    export_model_requested: bool = False
    export_dir: str | None = None


@dataclass(frozen=True)
class RunExecutionConfigPayloads:
    auto_config: dict[str, Any]
    edit_config: dict[str, Any]


class SupportsRunExecutionRequest(Protocol):
    config: str
    device: str | None
    profile: str | None
    out: str | None
    edit: str | None
    edit_label: str | None
    tier: str | None
    metric_kind: str | None
    probes: int | None
    until_pass: bool
    max_attempts: int
    timeout: int | None
    baseline: str | None
    no_cleanup: bool
    timing: bool
    progress: bool
    telemetry: bool
    prefer_local_files_only: bool


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _raise_config_error(path: str, value: Any, message: str) -> None:
    raise ConfigError(
        code="E002",
        message=message,
        details={"path": path, "value": value},
    )


def coerce_mapping(obj: object) -> dict[str, Any]:
    """Convert config-like objects to plain dicts without hiding programming errors."""
    if isinstance(obj, dict):
        return obj
    raw = getattr(obj, "_data", None)
    if isinstance(raw, dict):
        return raw
    dumped = getattr(obj, "model_dump", None)
    if callable(dumped):
        result = dumped()
        if isinstance(result, dict):
            return result
    try:
        data = obj.__dict__
    except (AttributeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_pm_acceptance_range(
    cfg: object | None,
    *,
    coerce_mapping_fn: Any | None = None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from config with safe defaults."""
    if coerce_mapping_fn is None:
        coerce_mapping_fn = coerce_mapping
    base_min = 0.95
    base_max = 1.10

    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    acceptance = pm_map.get("acceptance_range") if isinstance(pm_map, dict) else None
    if acceptance is None:
        return {}
    if not isinstance(acceptance, dict):
        _raise_config_error(
            "primary_metric.acceptance_range",
            acceptance,
            "primary_metric.acceptance_range must be a mapping with optional min/max bounds.",
        )
    cfg_min = None
    cfg_max = None
    if "min" in acceptance:
        cfg_min = _coerce_optional_float(acceptance.get("min"))
        if cfg_min is None:
            _raise_config_error(
                "primary_metric.acceptance_range.min",
                acceptance.get("min"),
                "primary_metric.acceptance_range.min must be a positive finite number.",
            )
    if "max" in acceptance:
        cfg_max = _coerce_optional_float(acceptance.get("max"))
        if cfg_max is None:
            _raise_config_error(
                "primary_metric.acceptance_range.max",
                acceptance.get("max"),
                "primary_metric.acceptance_range.max must be a positive finite number.",
            )

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        _raise_config_error(
            "primary_metric.acceptance_range.min",
            min_val,
            "primary_metric.acceptance_range.min must be greater than zero.",
        )
    if max_val <= 0:
        _raise_config_error(
            "primary_metric.acceptance_range.max",
            max_val,
            "primary_metric.acceptance_range.max must be greater than zero.",
        )

    if max_val < min_val:
        _raise_config_error(
            "primary_metric.acceptance_range",
            acceptance,
            "primary_metric.acceptance_range.max must be greater than or equal to min.",
        )

    return {"min": float(min_val), "max": float(max_val)}


def resolve_pm_drift_band(
    cfg: object | None,
    *,
    coerce_mapping_fn: Any | None = None,
) -> dict[str, float]:
    """Resolve preview→final drift band from config with safe defaults."""
    if coerce_mapping_fn is None:
        coerce_mapping_fn = coerce_mapping
    base_min = 0.95
    base_max = 1.05

    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    drift_band = pm_map.get("drift_band") if isinstance(pm_map, dict) else None
    if drift_band is None:
        return {}
    cfg_min = None
    cfg_max = None
    if isinstance(drift_band, dict):
        if "min" in drift_band:
            cfg_min = _coerce_optional_float(drift_band.get("min"))
            if cfg_min is None:
                _raise_config_error(
                    "primary_metric.drift_band.min",
                    drift_band.get("min"),
                    "primary_metric.drift_band.min must be a positive finite number.",
                )
        if "max" in drift_band:
            cfg_max = _coerce_optional_float(drift_band.get("max"))
            if cfg_max is None:
                _raise_config_error(
                    "primary_metric.drift_band.max",
                    drift_band.get("max"),
                    "primary_metric.drift_band.max must be a positive finite number.",
                )
    elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
        cfg_min = _coerce_optional_float(drift_band[0])
        cfg_max = _coerce_optional_float(drift_band[1])
        if cfg_min is None or cfg_max is None:
            _raise_config_error(
                "primary_metric.drift_band",
                drift_band,
                "primary_metric.drift_band list form must contain two positive finite numbers.",
            )
    else:
        _raise_config_error(
            "primary_metric.drift_band",
            drift_band,
            "primary_metric.drift_band must be a mapping or a two-item list/tuple.",
        )

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        _raise_config_error(
            "primary_metric.drift_band.min",
            min_val,
            "primary_metric.drift_band.min must be greater than zero.",
        )
    if max_val <= 0:
        _raise_config_error(
            "primary_metric.drift_band.max",
            max_val,
            "primary_metric.drift_band.max must be greater than zero.",
        )
    if min_val >= max_val:
        _raise_config_error(
            "primary_metric.drift_band",
            drift_band,
            "primary_metric.drift_band.min must be less than max.",
        )

    return {"min": float(min_val), "max": float(max_val)}


def resolve_guard_overhead_threshold(
    cfg: object | None,
    *,
    default_threshold: float = GUARD_OVERHEAD_THRESHOLD,
    coerce_mapping_fn=coerce_mapping,
) -> float:
    """Resolve guard-overhead threshold from config with safe default fallback."""
    threshold = float(default_threshold)
    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    candidate = pm_map.get("overhead_threshold") if isinstance(pm_map, dict) else None
    if candidate is None:
        return float(threshold)
    parsed = _coerce_optional_float(candidate)
    if parsed is None or not math.isfinite(parsed) or parsed < 0.0:
        _raise_config_error(
            "primary_metric.overhead_threshold",
            candidate,
            "primary_metric.overhead_threshold must be a non-negative finite number.",
        )
    assert parsed is not None
    threshold = float(parsed)
    return float(threshold)


def coerce_bool_like(value: Any) -> bool | None:
    """Best-effort bool coercion used for config policy toggles."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def resolve_skip_overhead_policy(
    cfg: object | None,
    *,
    coerce_mapping_fn=coerce_mapping,
) -> tuple[bool, str | None]:
    """Resolve overhead-skip policy from run/eval config context."""
    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    if not isinstance(cfg_map, dict):
        return False, None
    ctx = coerce_mapping_fn(cfg_map.get("context"))
    run_ctx = coerce_mapping_fn(ctx.get("run")) if isinstance(ctx, dict) else {}
    eval_ctx = coerce_mapping_fn(ctx.get("eval")) if isinstance(ctx, dict) else {}

    run_val = coerce_bool_like(run_ctx.get("skip_overhead_check"))
    if run_val is not None:
        return bool(run_val), "config:context.run.skip_overhead_check"

    eval_val = coerce_bool_like(eval_ctx.get("skip_overhead_check"))
    if eval_val is not None:
        return bool(eval_val), "config:context.eval.skip_overhead_check"

    return False, None


def should_measure_overhead(
    profile_normalized: str,
    cfg: object | None,
    *,
    coerce_mapping_fn=coerce_mapping,
) -> tuple[bool, bool, str | None]:
    """Return overhead check policy resolved from profile + config context."""
    skip_overhead_cfg, skip_source = resolve_skip_overhead_policy(
        cfg, coerce_mapping_fn=coerce_mapping_fn
    )
    enforce_profile = profile_normalized in {"ci", "release"}
    skip_overhead = bool(skip_overhead_cfg and enforce_profile)
    measure_guard_overhead = bool(enforce_profile and not skip_overhead)
    source = skip_source if skip_overhead else None
    return measure_guard_overhead, skip_overhead, source


def resolve_pm_min_tokens_target(
    *,
    tier: str | None,
    profile: str | None,
) -> int:
    """Resolve the minimum PM token target from tier policy."""
    resolved = resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        min_tokens = int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError):
        _raise_config_error(
            "tier_policies.metrics.pm_ratio.min_tokens",
            pm_ratio.get("min_tokens"),
            "Resolved tier policy min_tokens must be an integer.",
        )
    if min_tokens < 0:
        _raise_config_error(
            "tier_policies.metrics.pm_ratio.min_tokens",
            min_tokens,
            "Resolved tier policy min_tokens must be non-negative.",
        )
    return min_tokens


def choose_dataset_split(
    *,
    requested: str | None,
    available: list[str] | None,
    split_aliases: Sequence[str] = ("validation", "val", "dev", "eval", "test"),
) -> tuple[str, bool]:
    """Choose a dataset split deterministically."""
    if isinstance(requested, str):
        requested_text = str(requested)
        if requested_text:
            return requested_text, False
    avail = list(available) if isinstance(available, list) and available else []
    if avail:
        for cand in split_aliases:
            if cand in avail:
                return cand, True
        return sorted(avail)[0], True
    return "validation", True


def env_flag(name: str, *, environ: Mapping[str, str] | None = None) -> bool:
    source = environ if environ is not None else os.environ
    return str(source.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def env_text(name: str, *, environ: Mapping[str, str] | None = None) -> str | None:
    source = environ if environ is not None else os.environ
    value = source.get(name)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_run_execution_request(
    request: SupportsRunExecutionRequest,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunExecutionRequest:
    return RunExecutionRequest(
        config=request.config,
        device=request.device,
        profile=request.profile,
        out=request.out,
        edit=request.edit,
        edit_label=request.edit_label,
        tier=request.tier,
        metric_kind=request.metric_kind,
        probes=request.probes,
        until_pass=bool(request.until_pass),
        max_attempts=int(request.max_attempts),
        timeout=request.timeout,
        baseline=request.baseline,
        no_cleanup=bool(request.no_cleanup),
        capture_timings=bool(request.timing or request.progress),
        telemetry=bool(request.telemetry),
        prefer_local_files_only=bool(request.prefer_local_files_only),
        eval_device_override=env_text("INVARLOCK_EVAL_DEVICE", environ=environ),
        determinism_mode=env_text("PACK_DETERMINISM", environ=environ)
        or env_text("INVARLOCK_DETERMINISM", environ=environ),
        determinism_warn_only=env_flag(
            "INVARLOCK_DETERMINISM_WARN_ONLY", environ=environ
        ),
        tiny_relax_enabled=env_flag("INVARLOCK_TINY_RELAX", environ=environ),
        export_model_requested=env_flag("INVARLOCK_EXPORT_MODEL", environ=environ),
        export_dir=env_text("INVARLOCK_EXPORT_DIR", environ=environ),
    )


def _normalize_profile_checks(existing_checks: Any) -> list[str]:
    if isinstance(existing_checks, list | tuple | set):
        return [str(item) for item in existing_checks]
    if existing_checks:
        return [str(existing_checks)]
    return []


def _section_dict(cfg: Any, name: str) -> dict[str, Any]:
    section_fn = getattr(cfg, "section", None)
    if callable(section_fn):
        try:
            section = section_fn(name)
        except (AttributeError, KeyError, TypeError):
            section = None
        if isinstance(section, dict):
            return section
    try:
        value = getattr(cfg, name)
    except (AttributeError, KeyError, TypeError):
        value = None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
    return {}


def _baseline_eval_windows(
    baseline_report_data: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(baseline_report_data, Mapping):
        return None
    evaluation_windows = baseline_report_data.get("evaluation_windows")
    if not isinstance(evaluation_windows, Mapping):
        return None
    final = evaluation_windows.get("final")
    if not isinstance(final, Mapping):
        return None
    window_ids = final.get("window_ids")
    logloss = final.get("logloss")
    if not isinstance(window_ids, list) or not isinstance(logloss, list):
        return None
    payload: dict[str, Any] = {
        "final": {
            "window_ids": list(window_ids or []),
            "logloss": list(logloss or []),
        }
    }
    token_counts = final.get("token_counts")
    if isinstance(token_counts, list):
        payload["final"]["token_counts"] = list(token_counts or [])
    return payload


def build_run_context_payload(
    *,
    cfg: Any,
    profile: str | None,
    pairing_schedule: dict[str, Any] | None,
    seed_bundle: Mapping[str, Any],
    plugin_provenance: Mapping[str, Any],
    run_id: str,
    baseline_report_data: Mapping[str, Any] | None,
    pm_acceptance_range: Mapping[str, float] | tuple[float, float] | None,
    pm_drift_band: Mapping[str, float] | tuple[float, float] | None,
    guard_overhead_threshold: float,
    model_profile: Any,
    resolved_loss_type: str,
    tiny_relax_enabled: bool,
    to_serialisable_dict_fn: ToSerialisableDictFn,
) -> dict[str, Any]:
    guards_section = _section_dict(cfg, "guards")
    eval_section = _section_dict(cfg, "eval")
    guard_overrides = {
        "spectral": to_serialisable_dict_fn(guards_section.get("spectral", {})),
        "rmt": to_serialisable_dict_fn(guards_section.get("rmt", {})),
        "variance": to_serialisable_dict_fn(guards_section.get("variance", {})),
        "invariants": to_serialisable_dict_fn(guards_section.get("invariants", {})),
    }

    if getattr(model_profile, "invariants", None):
        invariants_policy = guard_overrides.setdefault("invariants", {})
        checks_list = _normalize_profile_checks(
            invariants_policy.get("profile_checks", [])
        )
        for invariant in model_profile.invariants:
            invariant_name = str(invariant)
            if invariant_name not in checks_list:
                checks_list.append(invariant_name)
        invariants_policy["profile_checks"] = checks_list

    run_context: dict[str, Any] = {
        "eval": to_serialisable_dict_fn(eval_section),
        "dataset": to_serialisable_dict_fn(cfg.dataset),
        "guards": guard_overrides,
        "profile": profile if profile else "",
        "pairing_baseline": pairing_schedule,
        "seeds": dict(seed_bundle),
        "plugins": dict(plugin_provenance),
        "run_id": run_id,
    }
    guard_order = guards_section.get("order")
    if isinstance(guard_order, list) and all(
        isinstance(item, str) for item in guard_order
    ):
        run_context["guard_chain_observed"] = list(guard_order)
    if tiny_relax_enabled:
        run_context["run"] = {"tiny_relax": True}

    baseline_eval = _baseline_eval_windows(baseline_report_data)
    if baseline_eval is not None:
        run_context["baseline_eval_windows"] = baseline_eval

    primary_metric: dict[str, Any] = {}
    run_context["primary_metric"] = primary_metric
    primary_metric["acceptance_range"] = pm_acceptance_range
    run_context["pm_acceptance_range"] = pm_acceptance_range
    if pm_drift_band:
        primary_metric["drift_band"] = pm_drift_band
        run_context["pm_drift_band"] = pm_drift_band
    primary_metric["overhead_threshold"] = guard_overhead_threshold
    run_context["guard_overhead_threshold"] = guard_overhead_threshold
    run_context["model_profile"] = {
        "family": getattr(model_profile, "family", ""),
        "default_loss": getattr(model_profile, "default_loss", ""),
        "module_selectors": getattr(model_profile, "module_selectors", {}),
        "invariants": getattr(model_profile, "invariants", []),
        "cert_lints": getattr(model_profile, "cert_lints", []),
    }
    extra_context = to_serialisable_dict_fn(_section_dict(cfg, "context"))
    if isinstance(extra_context, dict):
        run_context.update(extra_context)
    assurance_section = to_serialisable_dict_fn(_section_dict(cfg, "assurance"))
    if isinstance(assurance_section, dict) and assurance_section:
        run_context["assurance"] = assurance_section
    eval_context = run_context.get("eval")
    if isinstance(eval_context, dict):
        loss_context = eval_context.setdefault("loss", {})
        if isinstance(loss_context, dict):
            loss_context["resolved_type"] = resolved_loss_type

    return run_context


def _normalize_edit_plan(plan_obj: Any) -> dict[str, Any]:
    if isinstance(plan_obj, dict):
        return dict(plan_obj)
    plan_data = getattr(plan_obj, "_data", None)
    if isinstance(plan_data, dict):
        return dict(plan_data)
    if hasattr(plan_obj, "items"):
        try:
            return dict(plan_obj)
        except (TypeError, ValueError):
            return {}
    return {}


def build_run_execution_config_payloads(
    *,
    cfg: Any,
    model_profile: Any,
) -> RunExecutionConfigPayloads:
    auto_section = _section_dict(cfg, "auto")
    try:
        auto_enabled = bool(auto_section.get("enabled"))
    except (AttributeError, TypeError, ValueError):
        auto_enabled = False
    try:
        auto_tier = auto_section.get("tier") or "balanced"
    except (AttributeError, TypeError, ValueError):
        auto_tier = "balanced"
    try:
        auto_probes = int(auto_section.get("probes", 0))
    except (AttributeError, TypeError, ValueError):
        auto_probes = 0
    try:
        auto_target_ratio = float(auto_section.get("target_pm_ratio", 2.0))
    except (AttributeError, TypeError, ValueError):
        auto_target_ratio = 2.0

    auto_config = {
        "enabled": auto_enabled,
        "tier": auto_tier,
        "probes": auto_probes,
        "target_pm_ratio": auto_target_ratio,
    }

    edit_config: dict[str, Any] = {}
    try:
        plan_obj = getattr(cfg.edit, "plan", {})
        if plan_obj:
            edit_config = _normalize_edit_plan(plan_obj)
    except (AttributeError, TypeError):
        edit_config = {}

    module_selectors = getattr(model_profile, "module_selectors", None)
    if (
        isinstance(module_selectors, dict)
        and module_selectors
        and "module_selectors" not in edit_config
    ):
        edit_config["module_selectors"] = {
            key: list(values) for key, values in module_selectors.items()
        }

    return RunExecutionConfigPayloads(
        auto_config=auto_config,
        edit_config=edit_config,
    )
