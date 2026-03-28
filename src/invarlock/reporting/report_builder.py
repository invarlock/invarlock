"""
InvarLock Evaluation Report Generation
=====================================

Generate standardized evaluation reports from RunReport and baseline
comparison.
Evaluation reports are standalone, portable artifacts that record statistical
gates and evidence for CI/CD checks and audits (not formal verification).
"""

from __future__ import annotations

## Core evaluation report building and analysis orchestration lives here.
# mypy: ignore-errors
import copy
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable
from typing import Any

# Optional JSON Schema validation support
try:  # pragma: no cover - exercised in integration
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.eval.primary_metric import compute_primary_metric_from_report, get_metric
from invarlock.public_contracts import load_json_contract
from invarlock.utils.digest import hash_json

from . import report_schema as _report_schema
from .report_overhead import (
    compute_quality_overhead_from_guard as _compute_quality_overhead_from_guard_impl,
)
from .report_overhead import (
    prepare_guard_overhead_section as _prepare_guard_overhead_section_impl,
)
from .report_policy import (
    resolve_pm_acceptance_range_from_report as _resolve_pm_acceptance_range_from_report_impl,
)
from .report_policy import (
    resolve_pm_drift_band_from_report as _resolve_pm_drift_band_from_report_impl,
)
from .report_policy import (
    resolve_tiny_relax_from_report as _resolve_tiny_relax_from_report_impl,
)
from .report_provenance import (
    build_provenance_block as _build_provenance_block_impl,
)
from .report_schema import (
    REPORT_JSON_SCHEMA,
    REPORT_SCHEMA_VERSION,
)
from .report_types import RunReport
from .report_types import validate_report as validate_run_report
from .report_validation import (
    compute_validation_flags as _compute_validation_flags_impl,
)

# Expose compute_window_hash for tests that monkeypatch it
# compute_window_hash used to be exposed via the evaluation report builder; tests now patch
# dataset_hashing.compute_window_hash directly, so this import is no longer needed.
from .utils import (
    _coerce_int,
    _get_mapping,
    _infer_scope_from_modules,
    _sanitize_seed_bundle,
)
from .validate import validate_guard_overhead

# Policy digest semantic version (bumped when thresholds basis changes)
POLICY_VERSION = "policy-v1"

# Canonical base ratio limits per tier
TIER_RATIO_LIMITS: dict[str, float] = {
    "conservative": 1.05,
    "balanced": 1.10,
    "aggressive": 1.20,
    "none": 1.10,
}

# Canonical preview→final drift band used when not explicitly configured.
PM_DRIFT_BAND_DEFAULT: tuple[float, float] = (0.95, 1.05)


def _is_ppl_kind(name: Any) -> bool:
    """Return True if a primary_metric kind denotes a ppl-like metric.

    Supports alternate names to stay resilient across schema variants.
    """
    try:
        n = str(name or "").lower()
    except Exception:  # pragma: no cover
        n = ""
    return n in {
        "ppl",
        "perplexity",
        "ppl_causal",
        "causal_ppl",
        "ppl_mlm",
        "mlm_ppl",
        "ppl_masked",
        "ppl_seq2seq",
        "seq2seq_ppl",
    }


## NOTE: Deprecated helper `_get_ppl_final` was removed; callers should
## use the normalized primary_metric block directly via make_report or
## report processing utilities.


def _compute_edit_digest(report: dict) -> dict:
    """Compute a minimal, non-leaky edit breadcrumb for provenance.

    If `quant_rtn` is detected as the edit name, tag as quantization and
    hash the name+config. Otherwise, treat as cert_only with a stable hash.
    """
    try:
        edits = report.get("edit")
        if not isinstance(edits, dict):
            provenance = report.get("provenance")
            edits = provenance.get("edits") if isinstance(provenance, dict) else {}
    except Exception:  # pragma: no cover
        edits = {}
    family = "cert_only"
    impl_hash = hash_json({"family": "cert_only"})
    try:
        if isinstance(edits, dict) and str(edits.get("name", "")) == "quant_rtn":
            family = "quantization"
            cfg = (
                edits.get("config", {}) if isinstance(edits.get("config"), dict) else {}
            )
            impl_hash = hash_json({"name": "quant_rtn", "config": cfg})
    except Exception:  # pragma: no cover
        pass
    return {"family": family, "impl_hash": impl_hash, "version": 1}


def _compute_confidence_label(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    """Compute evaluation report confidence label based on stability and CI width.

    Heuristics:
    - High: ppl_acceptable=True, unstable=False, width <= 0.03 (ratio) or <= 1.0 pp for accuracy
    - Medium: floors met but unstable=True or width borderline (<= 2x threshold)
    - Low: otherwise (floors unmet, failure, or missing bounds)
    Returns a dict with label, basis, width and threshold for transparency.
    """
    validation = evaluation_report.get("validation", {}) or {}
    pm_ok = bool(validation.get("primary_metric_acceptable", False))
    # Basis label shown in confidence block:
    #  - For ppl-like metrics, use 'ppl_ratio' to reflect ratio width threshold
    #  - For accuracy-like metrics, use their kind ('accuracy' or 'vqa_accuracy')
    #  - Fall back to 'primary_metric' when unknown
    basis = "primary_metric"
    lo = hi = float("nan")
    try:
        pm = evaluation_report.get("primary_metric", {}) or {}
        kind = str(pm.get("kind", "") or "").lower()
        if isinstance(pm, dict) and pm and pm.get("display_ci"):
            dci = pm.get("display_ci")
            if isinstance(dci, tuple | list) and len(dci) == 2:
                lo, hi = float(dci[0]), float(dci[1])
                # Map kind → confidence basis label
                if kind.startswith("ppl"):
                    basis = "ppl_ratio"
                elif kind in {"accuracy", "vqa_accuracy"}:
                    basis = kind
                else:
                    basis = basis if basis else (kind or "primary_metric")
    except (TypeError, ValueError):  # pragma: no cover
        pass

    width = hi - lo if (math.isfinite(lo) and math.isfinite(hi)) else float("nan")
    # Thresholds (policy-configurable; fallback to defaults)
    thr_ratio = 0.03  # 3% width for ratio
    thr_pp = 1.0  # 1.0 percentage point for accuracy kinds
    try:
        pol = evaluation_report.get("resolved_policy")
        if isinstance(pol, dict):
            conf_pol = pol.get("confidence")
            if isinstance(conf_pol, dict):
                rr = conf_pol.get("ppl_ratio_width_max")
                if isinstance(rr, int | float):
                    thr_ratio = float(rr)
                ap = conf_pol.get("accuracy_delta_pp_width_max")
                if isinstance(ap, int | float):
                    thr_pp = float(ap)
    except (TypeError, ValueError):  # pragma: no cover
        pass
    is_acc = basis in {"accuracy", "vqa_accuracy"}
    thr = thr_pp if is_acc else thr_ratio

    # Unstable hint from primary metric (if provided)
    try:
        unstable = bool((evaluation_report.get("primary_metric") or {}).get("unstable"))
    except (AttributeError, TypeError, ValueError):  # pragma: no cover
        unstable = False

    label = "Low"
    if pm_ok:
        if (not unstable) and math.isfinite(width) and width <= thr:
            label = "High"
        else:
            # Floors met, but unstable or borderline width
            if math.isfinite(width) and width <= 2 * thr:
                label = "Medium"
            else:
                label = "Medium" if unstable else "Low"
    else:
        label = "Low"

    return {
        "label": label,
        "basis": basis,
        "width": width,
        "threshold": thr,
        "unstable": unstable,
    }


# Minimal JSON Schema describing the canonical shape of an evaluation report.
# This focuses on structural validity; numerical thresholds are validated
# separately in metric-specific logic.
# JSON Schema is provided by report_schema; no duplication here.


# Mirror jsonschema and structural validator for test monkeypatching compatibility.
jsonschema = getattr(_report_schema, "jsonschema", None)


def _validate_with_jsonschema(evaluation_report: dict[str, Any]) -> bool:
    if jsonschema is None:
        return True
    try:
        jsonschema.validate(instance=evaluation_report, schema=REPORT_JSON_SCHEMA)
        return True
    except Exception:  # pragma: no cover
        return False


def validate_report(evaluation_report: dict[str, Any]) -> bool:
    """Validate that an evaluation report has all required fields and valid data."""
    try:
        if evaluation_report.get("schema_version") != REPORT_SCHEMA_VERSION:
            return False
        # Prefer JSON Schema structural validation; if unavailable or too strict,
        # fall back to a lenient minimal check used by unit tests.
        if not _validate_with_jsonschema(evaluation_report):
            # Minimal fallback: require schema version + run_id + primary_metric
            run_id_ok = isinstance(evaluation_report.get("run_id"), str) and bool(
                evaluation_report.get("run_id")
            )
            pm = evaluation_report.get("primary_metric")
            pm_ok = isinstance(pm, dict) and (
                isinstance(pm.get("final"), int | float)
                or (isinstance(pm.get("kind"), str) and bool(pm.get("kind")))
            )
            if not (run_id_ok and pm_ok):
                return False

        validation = evaluation_report.get("validation", {})
        for flag in [
            "preview_final_drift_acceptable",
            "primary_metric_acceptable",
            "invariants_pass",
            "spectral_stable",
            "rmt_stable",
            "guard_overhead_acceptable",
        ]:
            if flag in validation and not isinstance(validation.get(flag), bool):
                return False

        return True
    except (KeyError, TypeError, ValueError):
        return False


VARIANCE_CANONICAL_KEYS = (
    "deadband",
    "min_abs_adjust",
    "max_scale_step",
    "min_effect_lognll",
    "predictive_one_sided",
    "topk_backstop",
    "max_adjusted_modules",
)


## Helpers are imported from invarlock.reporting.utils


def _collect_backend_versions() -> dict[str, Any]:
    """Collect backend/library versions for provenance.env_flags.

    Best-effort and resilient to missing libraries. Includes torch/cuda/cudnn/nccl
    when available, as well as Python/platform basics.
    """
    info: dict[str, Any] = {}
    # Python/platform
    try:
        info["python"] = platform.python_version()
        info["platform"] = platform.platform()
        info["machine"] = platform.machine()
    except Exception:  # pragma: no cover
        pass
    # Torch + CUDA libs (best-effort)
    try:  # pragma: no cover - depends on torch availability
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        tv = getattr(torch, "version", None)
        if tv is not None:
            info["torch_cuda"] = getattr(tv, "cuda", None)
            info["torch_cudnn"] = getattr(tv, "cudnn", None)
            info["torch_git"] = getattr(tv, "git_version", None)
        # Device and driver meta
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["device_name"] = getattr(props, "name", None)
                try:
                    maj = getattr(props, "major", None)
                    minr = getattr(props, "minor", None)
                    if maj is not None and minr is not None:
                        info["sm_capability"] = f"{int(maj)}.{int(minr)}"
                except Exception:  # pragma: no cover
                    pass
        except Exception:  # pragma: no cover
            pass
        # cuDNN runtime version
        try:
            if hasattr(torch.backends, "cudnn") and hasattr(
                torch.backends.cudnn, "version"
            ):
                v = torch.backends.cudnn.version()
                info["cudnn_runtime"] = int(v) if v is not None else None
        except Exception:  # pragma: no cover
            pass
        # NCCL version
        try:
            nccl_mod = getattr(torch.cuda, "nccl", None)
            if nccl_mod is not None and hasattr(nccl_mod, "version"):
                info["nccl"] = str(nccl_mod.version())
        except Exception:  # pragma: no cover
            pass
        # TF32 status (duplicated from meta.cuda_flags for convenience)
        try:
            tf32 = {}
            if hasattr(torch.backends, "cudnn") and hasattr(
                torch.backends.cudnn, "allow_tf32"
            ):
                tf32["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
            if hasattr(torch.backends, "cuda") and hasattr(
                torch.backends.cuda, "matmul"
            ):
                matmul = torch.backends.cuda.matmul
                if hasattr(matmul, "allow_tf32"):
                    tf32["cuda_matmul_allow_tf32"] = bool(matmul.allow_tf32)
            if tf32:
                info["tf32"] = tf32
        except Exception:  # pragma: no cover
            pass
    except Exception:  # pragma: no cover
        # torch not available
        pass
    # Environment variable hints
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    except Exception:  # pragma: no cover
        pass
    return {k: v for k, v in info.items() if v is not None}


## Pairing helper available from invarlock.reporting.utils


def _compute_variance_policy_digest(policy: dict[str, Any]) -> str:
    from .policy_utils import _compute_variance_policy_digest as _impl

    return _impl(policy)


def _compute_thresholds_payload(
    tier: str, resolved_policy: dict[str, Any]
) -> dict[str, Any]:
    from .policy_utils import _compute_thresholds_payload as _impl

    return _impl(tier, resolved_policy)


def _compute_thresholds_hash(payload: dict[str, Any]) -> str:
    from .policy_utils import _compute_thresholds_hash as _impl

    return _impl(payload)


# Allow-list loader with safe defaults for validation keys
_VALIDATION_ALLOWLIST_DEFAULT = {
    "primary_metric_acceptable",
    "primary_metric_tail_acceptable",
    "preview_final_drift_acceptable",
    "guard_overhead_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    # Compatibility keys were removed; PM-only surface
    "hysteresis_applied",
    "moe_observed",
    "moe_identity_ok",
}


def _load_validation_allowlist_with_source() -> tuple[set[str], str]:
    """Load validation key allow-list and report the source explicitly."""
    try:
        data = load_json_contract("validation_keys.json")
        if isinstance(data, list):
            return {str(k) for k in data}, "contracts"
        return (
            set(_VALIDATION_ALLOWLIST_DEFAULT),
            "fallback:invalid-contract-validation-keys",
        )
    except Exception:  # pragma: no cover
        return set(_VALIDATION_ALLOWLIST_DEFAULT), "fallback:load-error"


def _load_validation_allowlist() -> set[str]:
    """Load validation key allow-list from contracts with fail-closed fallback."""
    keys, _ = _load_validation_allowlist_with_source()
    return keys


def _apply_validation_allowlist_schema(validation_keys: set[str]) -> None:
    """Apply allow-list constraints to report schema (fail closed on shape drift)."""
    schema_properties = REPORT_JSON_SCHEMA.get("properties")
    if not isinstance(schema_properties, dict):
        raise RuntimeError(
            "REPORT_JSON_SCHEMA.properties must be a mapping to enforce validation "
            "allow-list constraints."
        )
    validation_spec = schema_properties.get("validation")
    if not isinstance(validation_spec, dict):
        raise RuntimeError(
            "REPORT_JSON_SCHEMA.properties.validation must be a mapping to enforce "
            "validation allow-list constraints."
        )
    validation_spec["properties"] = {k: {"type": "boolean"} for k in validation_keys}
    validation_spec["additionalProperties"] = False


# Tighten JSON Schema: populate validation.properties from allow-list and
# disallow unknown validation keys at schema level.
_VALIDATION_ALLOWLIST_KEYS, _VALIDATION_ALLOWLIST_SOURCE = (
    _load_validation_allowlist_with_source()
)
_apply_validation_allowlist_schema(_VALIDATION_ALLOWLIST_KEYS)


## Note: helpers like _get_section/_get_mapping/_iter_guard_entries,
## and policy helpers are provided by invarlock.reporting.utils and policy_utils.
## Import those directly in callers/tests instead of through this module.


def _normalize_and_validate_report(report: RunReport | dict[str, Any]) -> RunReport:
    """Normalize a possibly-minimal report and validate its structure.

    Uses the local normalizer when available, then checks `validate_run_report`.
    Raises ValueError on invalid input. Returns the normalized RunReport.
    """
    try:
        from .normalizer import normalize_run_report as _norm

        if isinstance(report, dict):
            report = _norm(report)
    except ImportError:  # pragma: no cover
        pass
    except (TypeError, ValueError, AttributeError):  # pragma: no cover
        pass
    if not validate_run_report(report):
        raise ValueError("Invalid RunReport structure")
    return report


def _extract_report_meta(report: RunReport) -> dict[str, Any]:
    """Extract the evaluation report metadata block with a full seed bundle."""
    meta_section = (
        report.get("meta", {}) if isinstance(report.get("meta"), dict) else {}
    )
    seed_value = _coerce_int(meta_section.get("seed"))
    seeds_bundle = _sanitize_seed_bundle(meta_section.get("seeds"), seed_value)
    primary_seed = (
        seeds_bundle.get("python") if isinstance(seeds_bundle, dict) else None
    )
    if primary_seed is None:
        primary_seed = 0
    return {
        "model_id": meta_section.get("model_id", "unknown"),
        "adapter": meta_section.get("adapter", "unknown"),
        "device": meta_section.get("device", "unknown"),
        "ts": meta_section.get("ts"),
        "commit": meta_section.get("commit"),
        "seed": primary_seed,
        "seeds": seeds_bundle,
    }


def _enforce_drift_ratio_identity(
    paired_windows: int,
    delta_mean: Any,
    drift_ratio: float,
    window_plan_profile: str | None,
) -> float | None:
    """Ensure exp(delta_mean) aligns with observed drift ratio."""
    if (
        paired_windows > 0
        and isinstance(delta_mean, (int | float))
        and math.isfinite(delta_mean)
        and isinstance(drift_ratio, (int | float))
        and math.isfinite(drift_ratio)
    ):
        ratio_from_delta = math.exp(float(delta_mean))
        tolerance = 1e-3 * max(1.0, abs(drift_ratio))
        if abs(ratio_from_delta - drift_ratio) > tolerance:
            profile = (window_plan_profile or "dev").lower()
            if profile in {"ci", "release"}:
                raise ValueError(
                    "Paired ΔlogNLL mean is inconsistent with reported drift ratio."
                )
        return ratio_from_delta
    return None


def _enforce_ratio_ci_alignment(
    ratio_ci_source: str,
    ratio_ci: Any,
    logloss_delta_ci: Any,
) -> None:
    """Validate that ratio_ci matches exp(logloss_delta_ci) when paired."""
    if ratio_ci_source != "paired_baseline":
        return
    if not (
        isinstance(logloss_delta_ci, tuple | list)
        and len(logloss_delta_ci) == 2
        and isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
    ):
        return
    expected_bounds = tuple(math.exp(bound) for bound in logloss_delta_ci)
    for observed, expected in zip(ratio_ci, expected_bounds, strict=False):
        if not (
            isinstance(observed, (int | float))
            and math.isfinite(observed)
            and isinstance(expected, (int | float))
            and math.isfinite(expected)
        ):
            continue
        tolerance = 5e-4 * max(1.0, abs(expected))
        if abs(float(observed) - float(expected)) > tolerance:
            raise ValueError(
                "Paired ΔlogNLL CI mismatch: ratio bounds do not match exp(Δlog bounds)."
            )


def _enforce_display_ci_alignment(
    ratio_ci_source: str,
    primary_metric: Any,
    logloss_delta_ci: Any,
    window_plan_profile: str | None,
) -> None:
    """Ensure display_ci matches exp(ci) for ppl-like metrics when paired."""
    if ratio_ci_source != "paired_baseline":
        return
    if not isinstance(primary_metric, dict) or not primary_metric:
        return
    try:
        kind = str(primary_metric.get("kind", "")).lower()
    except Exception:
        return
    if not kind.startswith("ppl"):
        return

    def _finite_bounds(bounds: Any) -> bool:
        return (
            isinstance(bounds, tuple | list)
            and len(bounds) == 2
            and all(isinstance(v, int | float) and math.isfinite(v) for v in bounds)
        )

    ci = primary_metric.get("ci")
    if not _finite_bounds(ci):
        if _finite_bounds(logloss_delta_ci):
            primary_metric["ci"] = (
                float(logloss_delta_ci[0]),
                float(logloss_delta_ci[1]),
            )
            ci = primary_metric["ci"]
        else:
            profile = (window_plan_profile or "dev").lower()
            if profile in {"ci", "release"}:
                raise ValueError(
                    "primary_metric.ci missing for ppl-like metric under paired baseline."
                )
            return

    expected = tuple(math.exp(float(bound)) for bound in ci)
    display_ci = primary_metric.get("display_ci")
    if not _finite_bounds(display_ci):
        profile = (window_plan_profile or "dev").lower()
        if profile in {"ci", "release"}:
            raise ValueError(
                "primary_metric.display_ci missing for ppl-like metric under paired baseline."
            )
        primary_metric["display_ci"] = [expected[0], expected[1]]
        return

    for observed, exp_val in zip(display_ci, expected, strict=False):
        tolerance = 5e-4 * max(1.0, abs(exp_val))
        if abs(float(observed) - float(exp_val)) > tolerance:
            profile = (window_plan_profile or "dev").lower()
            if profile in {"ci", "release"}:
                raise ValueError(
                    "primary_metric.display_ci mismatch: bounds do not match exp(ci)."
                )
            primary_metric["display_ci"] = [expected[0], expected[1]]
            break


def _enforce_pairing_and_coverage(
    stats: dict[str, Any] | None,
    window_plan_profile: str | None,
    tier: str | None,
) -> None:
    """Enforce pairing and coverage contracts for CI/Release profiles."""
    profile = (window_plan_profile or "dev").lower()
    if profile not in {"ci", "release"}:
        return
    if not isinstance(stats, dict):
        raise ValueError("Missing dataset window stats for CI/Release enforcement.")

    pairing_reason = stats.get("window_pairing_reason")
    if pairing_reason is not None:
        raise ValueError(
            "CI/Release requires paired baseline evidence "
            f"(window_pairing_reason={pairing_reason!r})."
        )

    match_fraction = stats.get("window_match_fraction")
    overlap_fraction = stats.get("window_overlap_fraction")
    if not (
        isinstance(match_fraction, (int | float))
        and math.isfinite(float(match_fraction))
    ):
        raise ValueError("CI/Release requires window_match_fraction.")
    if float(match_fraction) < 0.999999:
        raise ValueError(
            f"CI/Release requires perfect pairing (window_match_fraction={float(match_fraction):.6f})."
        )

    if not (
        isinstance(overlap_fraction, (int | float))
        and math.isfinite(float(overlap_fraction))
    ):
        raise ValueError("CI/Release requires window_overlap_fraction.")
    if float(overlap_fraction) > 1e-9:
        raise ValueError(
            f"CI/Release requires non-overlapping windows (window_overlap_fraction={float(overlap_fraction):.6f})."
        )

    def _coerce_count(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val < 0:
            return None
        if abs(val - round(val)) > 1e-9:
            return None
        return int(round(val))

    paired_windows = _coerce_count(stats.get("paired_windows"))
    if paired_windows is None:
        raise ValueError("CI/Release requires paired_windows metric.")
    if paired_windows == 0:
        raise ValueError("CI/Release requires paired_windows > 0.")

    actual_preview = _coerce_count(stats.get("actual_preview"))
    actual_final = _coerce_count(stats.get("actual_final"))
    if actual_preview is None or actual_final is None:
        coverage = stats.get("coverage")
        if isinstance(coverage, dict):
            if actual_preview is None:
                actual_preview = _coerce_count(coverage.get("preview", {}).get("used"))
            if actual_final is None:
                actual_final = _coerce_count(coverage.get("final", {}).get("used"))

    if actual_preview is None or actual_final is None:
        raise ValueError("CI/Release requires preview/final window counts.")
    if actual_preview != actual_final:
        raise ValueError(
            f"CI/Release requires matching preview/final counts "
            f"(preview={actual_preview}, final={actual_final})."
        )

    from invarlock.core.runner_pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS

    tier_key = str(tier or "balanced").lower()
    floors = BOOTSTRAP_COVERAGE_REQUIREMENTS.get(
        tier_key, BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]
    )
    preview_floor = int(floors.get("preview", 0))
    final_floor = int(floors.get("final", 0))
    replicates_floor = int(floors.get("replicates", 0))

    coverage = stats.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("CI/Release requires bootstrap coverage stats.")

    preview_used = _coerce_count(coverage.get("preview", {}).get("used"))
    final_used = _coerce_count(coverage.get("final", {}).get("used"))
    replicates_used = _coerce_count(coverage.get("replicates", {}).get("used"))

    if replicates_used is None:
        bootstrap = stats.get("bootstrap")
        if isinstance(bootstrap, dict):
            replicates_used = _coerce_count(
                bootstrap.get("replicates", bootstrap.get("n"))
            )

    if preview_used is None or final_used is None or replicates_used is None:
        raise ValueError("CI/Release requires preview/final/replicates coverage stats.")

    if preview_used < preview_floor or final_used < final_floor:
        raise ValueError(
            "CI/Release requires preview/final coverage at or above tier floors "
            f"(preview={preview_used}/{preview_floor}, final={final_used}/{final_floor})."
        )
    if replicates_used < replicates_floor:
        raise ValueError(
            "CI/Release requires bootstrap replicates at or above tier floors "
            f"(replicates={replicates_used}/{replicates_floor})."
        )


def _fallback_paired_windows(
    paired_windows: int, coverage_summary: dict[str, Any]
) -> int:
    """Use coverage preview counts when explicit pairing is unavailable."""
    if paired_windows > 0 or not isinstance(coverage_summary, dict):
        return paired_windows
    try:
        cprev = coverage_summary.get("preview")
        used = cprev.get("used") if isinstance(cprev, dict) else None
        if isinstance(used, int | float) and used >= 0:
            return int(used)
    except Exception:  # pragma: no cover
        pass
    return paired_windows


def make_report(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
) -> dict[str, Any]:
    """Generate an evaluation report from a RunReport and baseline comparison."""
    from .report_make_impl import make_report_impl

    return make_report_impl(report, baseline)


# Console Validation Block helpers and render_report_markdown live in
# invarlock.reporting.render. This module owns report construction helpers only.
## Private helper functions


def _normalize_baseline(baseline: RunReport | dict[str, Any]) -> dict[str, Any]:
    """Normalize baseline input to a consistent dictionary format."""
    if isinstance(baseline, dict):

        def _guard_metrics_block(guard_name: str) -> dict[str, Any]:
            for guard in baseline.get("guards", []) or []:
                if str(guard.get("name", "")).lower() != guard_name:
                    continue
                metrics = guard.get("metrics")
                if isinstance(metrics, dict) and metrics:
                    return dict(metrics)
                return {}
            return {}

        def _merged_guard_metrics(
            guard_name: str, metrics_value: Any
        ) -> dict[str, Any]:
            merged = dict(metrics_value) if isinstance(metrics_value, dict) else {}
            guard_metrics = _guard_metrics_block(guard_name)
            if guard_metrics:
                merged.update(guard_metrics)
            return merged

        def _coerce_valid_ppl(value: Any, *, label: str) -> float:
            if not (isinstance(value, int | float) and math.isfinite(float(value))):
                raise ValueError(
                    f"Invalid baseline {label}: expected finite numeric value."
                )
            out = float(value)
            if out <= 0.0:
                raise ValueError(
                    f"Invalid baseline {label}: expected value > 0.0, observed {out}."
                )
            return out

        def _normalize_kind(value: Any) -> str:
            try:
                return str(value or "").strip().lower()
            except Exception:
                return ""

        def _is_ppl_kind(value: Any) -> bool:
            kind = _normalize_kind(value)
            # Treat unknown kinds as ppl-like for backward compatibility.
            return not kind or kind.startswith("ppl")

        def _derive_ppl_from_logloss_block(block: Any) -> float | None:
            if not isinstance(block, dict):
                return None
            ll = block.get("logloss")
            if not isinstance(ll, list) or not ll:
                return None
            vals = [float(x) for x in ll if isinstance(x, int | float)]
            if not vals:
                return None
            token_counts = block.get("token_counts")
            mean_ll: float
            if (
                isinstance(token_counts, list)
                and token_counts
                and len(token_counts) == len(vals)
            ):
                try:
                    num = 0.0
                    den = 0.0
                    for lval, tval in zip(vals, token_counts, strict=False):
                        if not isinstance(tval, int | float):
                            continue
                        tf = float(tval)
                        if tf <= 0.0:
                            continue
                        num += float(lval) * tf
                        den += tf
                    if den > 0.0:
                        mean_ll = num / den
                    else:
                        mean_ll = float(sum(vals) / len(vals))
                except Exception:
                    mean_ll = float(sum(vals) / len(vals))
            else:
                mean_ll = float(sum(vals) / len(vals))
            if not math.isfinite(mean_ll):
                return None
            ppl = math.exp(mean_ll)
            return float(ppl) if math.isfinite(ppl) else None

        # Check if it's a baseline schema (v1 only)
        if baseline.get("schema_version") in {"baseline-v1"}:
            metrics_blk = baseline.get("metrics", {}) or {}
            pm = (
                metrics_blk.get("primary_metric", {})
                if isinstance(metrics_blk, dict)
                else {}
            )
            pm_kind = _normalize_kind(pm.get("kind")) if isinstance(pm, dict) else ""
            pm_is_ppl = _is_ppl_kind(pm_kind)
            ppl_final_raw = (
                metrics_blk.get("ppl_final") if isinstance(metrics_blk, dict) else None
            )
            if (
                not (
                    isinstance(ppl_final_raw, int | float)
                    and math.isfinite(float(ppl_final_raw))
                )
                and pm_is_ppl
            ):
                if isinstance(pm, dict):
                    pf = pm.get("final")
                    if isinstance(pf, int | float):
                        ppl_final_raw = float(pf)
            if not (
                isinstance(ppl_final_raw, int | float)
                and math.isfinite(float(ppl_final_raw))
            ):
                eval_windows = baseline.get("evaluation_windows")
                if isinstance(eval_windows, dict):
                    ppl_final_raw = _derive_ppl_from_logloss_block(
                        eval_windows.get("final")
                    )
            if (
                not (
                    isinstance(ppl_final_raw, int | float)
                    and math.isfinite(float(ppl_final_raw))
                )
                and pm_kind
                and not pm_kind.startswith("ppl")
            ):
                out: dict[str, Any] = {
                    "run_id": baseline.get("meta", {}).get("commit_sha", "unknown")[
                        :16
                    ],
                    "model_id": baseline.get("meta", {}).get("model_id", "unknown"),
                    "spectral": baseline.get("spectral_base", {}),
                    "rmt": baseline.get("rmt_base", {}),
                    "invariants": baseline.get("invariants", {}),
                }
                if isinstance(pm, dict) and pm:
                    out["primary_metric"] = copy.deepcopy(pm)
                return out
            ppl_final = _coerce_valid_ppl(
                ppl_final_raw,
                label="metrics.ppl_final",
            )
            return {
                "run_id": baseline.get("meta", {}).get("commit_sha", "unknown")[:16],
                "model_id": baseline.get("meta", {}).get("model_id", "unknown"),
                "ppl_final": ppl_final,
                "spectral": baseline.get("spectral_base", {}),
                "rmt": baseline.get("rmt_base", {}),
                "invariants": baseline.get("invariants", {}),
            }
        # Check if it's a RunReport structure
        elif "meta" in baseline and "metrics" in baseline and "edit" in baseline:
            # Accept both ppl_* metrics and PM-first reports
            metrics_blk = baseline.get("metrics", {}) or {}
            pm = (
                metrics_blk.get("primary_metric", {})
                if isinstance(metrics_blk, dict)
                else {}
            )
            pm_kind = _normalize_kind(pm.get("kind")) if isinstance(pm, dict) else ""
            pm_is_ppl = _is_ppl_kind(pm_kind)
            ppl_final = metrics_blk.get("ppl_final")
            ppl_preview = metrics_blk.get("ppl_preview")
            if (
                not (
                    isinstance(ppl_final, int | float)
                    and math.isfinite(float(ppl_final))
                )
                and pm_is_ppl
            ):
                # Fallback: derive from primary_metric when present.
                try:
                    pf = pm.get("final")
                    pp = pm.get("preview", pf)
                    if isinstance(pf, int | float):
                        ppl_final = float(pf)
                    if isinstance(pp, int | float):
                        ppl_preview = float(pp)
                except Exception:  # pragma: no cover
                    # Leave as None; downstream validation will handle
                    pass
            eval_windows = baseline.get("evaluation_windows", {})
            final_windows = (
                eval_windows.get("final", {}) if isinstance(eval_windows, dict) else {}
            )
            preview_windows = (
                eval_windows.get("preview", {})
                if isinstance(eval_windows, dict)
                else {}
            )
            if ppl_final is None:
                ppl_final = _derive_ppl_from_logloss_block(final_windows)
            if ppl_preview is None:
                ppl_preview = _derive_ppl_from_logloss_block(preview_windows)
            if ppl_preview is None:
                ppl_preview = ppl_final

            non_ppl_without_ppl_metrics = (
                not (
                    isinstance(ppl_final, int | float)
                    and math.isfinite(float(ppl_final))
                )
                and pm_kind
                and not pm_kind.startswith("ppl")
            )

            # Fail closed for invalid baseline ppl metrics; this prevents silent
            # hardcoded baseline substitution that can mask broken evidence inputs.
            baseline_eval_windows = {
                "final": {
                    "window_ids": list(final_windows.get("window_ids", [])),
                    "logloss": [
                        float(x)
                        for x in final_windows.get("logloss", [])
                        if isinstance(x, int | float)
                    ],
                }
            }
            bootstrap_info = (
                baseline["metrics"].get("bootstrap", {})
                if isinstance(baseline.get("metrics"), dict)
                else {}
            )
            window_overlap = baseline["metrics"].get(
                "window_overlap_fraction", float("nan")
            )
            window_match = baseline["metrics"].get(
                "window_match_fraction", float("nan")
            )

            # Try to capture tokenizer hash from baseline report when available
            baseline_tokenizer_hash = None
            try:
                baseline_tokenizer_hash = baseline.get("meta", {}).get(
                    "tokenizer_hash"
                ) or baseline.get("data", {}).get("tokenizer_hash")
            except Exception:  # pragma: no cover
                baseline_tokenizer_hash = None

            baseline_out: dict[str, Any] = {
                "run_id": _generate_run_id(baseline),
                "model_id": baseline["meta"]["model_id"],
                "spectral": _merged_guard_metrics(
                    "spectral", baseline["metrics"].get("spectral", {})
                ),
                "rmt": _merged_guard_metrics("rmt", baseline["metrics"].get("rmt", {})),
                "invariants": baseline["metrics"].get("invariants", {}),
                "moe": baseline["metrics"].get("moe", {}),
                "evaluation_windows": baseline_eval_windows,
                "bootstrap": bootstrap_info,
                "window_overlap_fraction": window_overlap,
                "window_match_fraction": window_match,
                "tokenizer_hash": baseline_tokenizer_hash,
            }
            if isinstance(pm, dict) and pm:
                baseline_out["primary_metric"] = copy.deepcopy(pm)

            if non_ppl_without_ppl_metrics:
                return baseline_out

            ppl_final = _coerce_valid_ppl(ppl_final, label="metrics.ppl_final")
            ppl_preview = _coerce_valid_ppl(
                ppl_preview if ppl_preview is not None else ppl_final,
                label="metrics.ppl_preview",
            )
            baseline_out["ppl_final"] = ppl_final
            baseline_out["ppl_preview"] = ppl_preview
            return baseline_out
        else:
            # Assume it's already normalized, but recover ppl fields from PM-first
            # baseline payloads when needed.
            baseline_out = baseline.copy()
            metrics_blk = baseline_out.get("metrics", {})
            pm_kind = ""
            if isinstance(metrics_blk, dict):
                pm_metrics = metrics_blk.get("primary_metric")
                if isinstance(pm_metrics, dict):
                    pm_kind = _normalize_kind(pm_metrics.get("kind"))
            if not pm_kind:
                pm_top = baseline_out.get("primary_metric", {})
                if isinstance(pm_top, dict):
                    pm_kind = _normalize_kind(pm_top.get("kind"))
            pm_is_non_ppl = bool(pm_kind) and not pm_kind.startswith("ppl")

            ppl_final = baseline_out.get("ppl_final")
            ppl_preview = baseline_out.get("ppl_preview")

            if not (
                isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final))
            ):
                if isinstance(metrics_blk, dict):
                    pf_direct = metrics_blk.get("ppl_final")
                    pp_direct = metrics_blk.get("ppl_preview", pf_direct)
                    if isinstance(pf_direct, int | float) and math.isfinite(
                        float(pf_direct)
                    ):
                        ppl_final = float(pf_direct)
                    if isinstance(pp_direct, int | float) and math.isfinite(
                        float(pp_direct)
                    ):
                        ppl_preview = float(pp_direct)
                    if (
                        not (
                            isinstance(ppl_final, int | float)
                            and math.isfinite(float(ppl_final))
                        )
                        and not pm_is_non_ppl
                    ):
                        pm = metrics_blk.get("primary_metric", {})
                        if isinstance(pm, dict):
                            pf = pm.get("final")
                            pp = pm.get("preview", pf)
                            if isinstance(pf, int | float):
                                ppl_final = float(pf)
                            if isinstance(pp, int | float):
                                ppl_preview = float(pp)
                if (
                    not (
                        isinstance(ppl_final, int | float)
                        and math.isfinite(float(ppl_final))
                    )
                    and not pm_is_non_ppl
                ):
                    pm_top = baseline_out.get("primary_metric", {})
                    if isinstance(pm_top, dict):
                        pf = pm_top.get("final")
                        pp = pm_top.get("preview", pf)
                        if isinstance(pf, int | float):
                            ppl_final = float(pf)
                        if isinstance(pp, int | float):
                            ppl_preview = float(pp)
                if not (
                    isinstance(ppl_final, int | float)
                    and math.isfinite(float(ppl_final))
                ):
                    eval_windows = baseline_out.get("evaluation_windows")
                    if isinstance(eval_windows, dict):
                        ppl_final = _derive_ppl_from_logloss_block(
                            eval_windows.get("final")
                        )
                        if ppl_preview is None:
                            ppl_preview = _derive_ppl_from_logloss_block(
                                eval_windows.get("preview")
                            )

            if (
                not (
                    isinstance(ppl_final, int | float)
                    and math.isfinite(float(ppl_final))
                )
                and pm_is_non_ppl
            ):
                if "ppl_final" in baseline_out:
                    try:
                        pf = float(baseline_out.get("ppl_final"))
                    except Exception:
                        baseline_out.pop("ppl_final", None)
                    else:
                        if not math.isfinite(pf) or pf <= 0.0:
                            baseline_out.pop("ppl_final", None)
                if "ppl_preview" in baseline_out:
                    try:
                        pp = float(baseline_out.get("ppl_preview"))
                    except Exception:
                        baseline_out.pop("ppl_preview", None)
                    else:
                        if not math.isfinite(pp) or pp <= 0.0:
                            baseline_out.pop("ppl_preview", None)
                return baseline_out

            ppl_final = _coerce_valid_ppl(ppl_final, label="ppl_final")
            if ppl_preview is None:
                ppl_preview = ppl_final
            else:
                ppl_preview = _coerce_valid_ppl(ppl_preview, label="ppl_preview")

            baseline_out["ppl_final"] = ppl_final
            if "ppl_preview" in baseline_out or not math.isclose(
                float(ppl_preview), float(ppl_final), rel_tol=0.0, abs_tol=0.0
            ):
                baseline_out["ppl_preview"] = ppl_preview
            return baseline_out
    else:
        raise ValueError(
            "Baseline must be a RunReport dict or normalized baseline dict"
        )


## Dataset hashing helpers live in invarlock.reporting.dataset_hashing


## Guard extractors moved to invarlock.reporting.guards_analysis and imported above


def _extract_structural_deltas(report: RunReport) -> dict[str, Any]:
    """Extract structural parameter changes with compression diagnostics."""
    edit_section = report.get("edit", {}) if isinstance(report, dict) else {}
    deltas = edit_section.get("deltas", {}) if isinstance(edit_section, dict) else {}
    # Try to get edit configuration from plan first, fallback to config
    primary_config = None
    if isinstance(edit_section, dict):
        if isinstance(edit_section.get("plan"), dict):
            primary_config = edit_section["plan"]
        elif isinstance(edit_section.get("config"), dict):
            primary_config = edit_section["config"]
    if primary_config is None:
        edit_config: dict[str, Any] = {}
    else:
        edit_config = dict(primary_config)

    inference_record = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }

    def _infer(field: str, value: Any, source: str) -> bool:
        if value in (None, "unknown"):
            return False
        current = edit_config.get(field)
        if current not in (None, "unknown"):
            return False
        edit_config[field] = value
        inference_record["flags"][field] = True
        inference_record["sources"][field] = source
        inference_record["log"].append(f"{field} inferred from {source}: {value}")
        return True

    if isinstance(edit_section, dict):
        for key, value in edit_section.items():
            if key in {"plan", "config", "deltas"}:
                continue
            if value is None or isinstance(value, dict):
                continue
            edit_config.setdefault(key, value)

    if isinstance(edit_section, dict):
        plan_digest = str(edit_section.get("plan_digest", "")).lower()
        if "energy" in plan_digest:
            _infer("rank_policy", "energy", "plan_digest")

        if "energy_" in plan_digest and not edit_config.get("frac"):
            try:
                fraction_str = plan_digest.split("energy_")[1].split("_")[0]
                _infer("frac", float(fraction_str), "plan_digest")
            except (IndexError, ValueError):
                pass
        if not edit_config.get("scope"):
            if "ffn" in plan_digest:
                _infer("scope", "ffn", "plan_digest")
            elif "attn" in plan_digest:
                _infer("scope", "attn", "plan_digest")
            elif "embed" in plan_digest or "embedding" in plan_digest:
                _infer("scope", "embed", "plan_digest")
    try:
        edit_name = (report.get("edit", {}) or {}).get("name", "unknown")  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        edit_name = "unknown"

    structure = {
        "params_changed": deltas.get("params_changed", 0),
        "layers_modified": deltas.get("layers_modified", 0),
    }

    # Add optional fields if present
    if deltas.get("sparsity") is not None:
        structure["sparsity"] = deltas["sparsity"]

    if deltas.get("bitwidth_map"):
        structure["bitwidths"] = deltas["bitwidth_map"]
        # Extract bitwidth analysis
        bitwidth_summary = _analyze_bitwidth_map(deltas["bitwidth_map"])
        structure["bitwidth_analysis"] = bitwidth_summary

    # Extract rank information for SVD-based edits
    if "rank" in edit_name.lower() or "svd" in edit_name.lower():
        structure["ranks"] = _extract_rank_information(edit_config, deltas)
        structure["savings"] = _compute_savings_summary(deltas)
    else:
        structure["ranks"] = {}

    # Add compression diagnostics
    compression_diag = _extract_compression_diagnostics(
        edit_name, edit_config, deltas, structure, inference_record
    )
    structure["compression_diagnostics"] = compression_diag

    target_analysis = compression_diag.get("target_analysis", {})
    algo_details = compression_diag.setdefault("algorithm_details", {})

    fallback_scope = (
        edit_section.get("scope") if isinstance(edit_section, dict) else None
    )
    if _infer("scope", fallback_scope, "report.edit.scope"):
        target_analysis["scope"] = fallback_scope
    elif fallback_scope and target_analysis.get("scope") in (None, "unknown"):
        target_analysis["scope"] = fallback_scope

    if isinstance(edit_section, dict):
        edit_seed = edit_section.get("seed")
        _infer("seed", edit_seed, "report.edit.seed")

    if not inference_record["flags"].get("seed"):
        meta = report.get("meta", {}) if isinstance(report, dict) else {}
        meta_seed = None
        seeds_bundle = meta.get("seeds")
        if isinstance(seeds_bundle, dict):
            meta_seed = seeds_bundle.get("python")
        if meta_seed is None:
            meta_seed = meta.get("seed")
        _infer("seed", meta_seed, "report.meta.seeds")

    target_analysis["scope"] = edit_config.get(
        "scope", target_analysis.get("scope", "unknown")
    )
    algo_details["scope_targeting"] = target_analysis.get("scope", "unknown")

    final_seed = edit_config.get("seed", algo_details.get("seed", "unknown"))
    algo_details["seed"] = final_seed

    compression_diag["inferred"] = inference_record["flags"]
    if inference_record.get("sources"):
        compression_diag["inference_source"] = inference_record["sources"]
    if inference_record.get("log"):
        compression_diag["inference_log"] = inference_record["log"]

    return structure


def _extract_edit_metadata(
    report: RunReport, plugin_provenance: dict[str, Any]
) -> dict[str, Any]:
    """Extract edit-level provenance and configuration metadata for the evaluation report."""

    edit_section = _get_mapping(report, "edit")
    if not edit_section:
        return {}

    edit_name = str(edit_section.get("name", "") or "")

    plugin_edit = {}
    if isinstance(plugin_provenance, dict):
        candidate = plugin_provenance.get("edit")
        if isinstance(candidate, dict):
            plugin_edit = candidate

    # Prefer explicit metadata when provided, otherwise infer sensible defaults.
    algorithm = edit_section.get("algorithm")
    if not algorithm:
        algorithm = edit_name or ""
    # Sanitize algorithm identifiers to purge unsupported edit labels
    try:
        alg_lower = str(algorithm).strip().lower()
    except Exception:  # pragma: no cover
        alg_lower = ""
    allowed_algorithms = {"quant_rtn", "noop", "custom"}
    if alg_lower not in allowed_algorithms:
        algorithm = ""

    algorithm_version = (
        edit_section.get("algorithm_version") or plugin_edit.get("version") or ""
    )

    implementation = (
        edit_section.get("implementation") or plugin_edit.get("module") or ""
    )
    # Sanitize implementation identifiers
    if isinstance(implementation, str) and (
        "structured" in implementation.lower() or "lowrank" in implementation.lower()
    ):
        implementation = ""

    # Capture the resolved plan configuration (either top-level plan or config.plan).
    plan_dict: dict[str, Any] = {}
    raw_plan = edit_section.get("plan")
    if isinstance(raw_plan, dict):
        plan_dict = copy.deepcopy(raw_plan)
    else:
        config_section = edit_section.get("config")
        if isinstance(config_section, dict):
            config_plan = config_section.get("plan")
            if isinstance(config_plan, dict):
                plan_dict = copy.deepcopy(config_plan)

    if not isinstance(plan_dict, dict):
        plan_dict = {}

    scope = plan_dict.get("scope") or edit_section.get("scope")

    ranking = plan_dict.get("ranking") or edit_section.get("ranking") or ""
    grouping = plan_dict.get("grouping") or edit_section.get("grouping")

    budgets: dict[str, Any] = {}
    for key in (
        "head_budget",
        "mlp_budget",
        "heads",
        "mlp",
        "neuron_budget",
        "ffn_budget",
    ):
        value = plan_dict.get(key)
        if isinstance(value, dict):
            budgets[key] = copy.deepcopy(value)

    target_sparsity = plan_dict.get("target_sparsity")
    if isinstance(target_sparsity, int | float):
        budgets["target_sparsity"] = float(target_sparsity)

    if not scope:
        if "head_budget" in budgets and "mlp_budget" in budgets:
            scope = "heads+ffn"
        elif "head_budget" in budgets:
            scope = "heads"
        elif "mlp_budget" in budgets:
            scope = "ffn"
        else:
            scope = ""

    if not grouping:
        grouping = "auto" if scope == "heads" else ("none" if scope else "")

    seed_candidate = plan_dict.get("seed", edit_section.get("seed"))
    if seed_candidate is None:
        meta_section = _get_mapping(report, "meta")
        seed_candidate = meta_section.get("seed")
    seed_value = _coerce_int(seed_candidate)

    edit_metadata: dict[str, Any] = {
        "name": edit_name,
        "algorithm": algorithm,
        "algorithm_version": str(algorithm_version),
        "implementation": str(implementation),
        "scope": scope,
        "ranking": ranking,
        "grouping": grouping,
        "budgets": budgets,
        "seed": seed_value,
        "plan_digest": str(edit_section.get("plan_digest") or ""),
        "mask_digest": str(edit_section.get("mask_digest") or ""),
    }

    if not budgets:
        edit_metadata.pop("budgets")
    if seed_value is None:
        edit_metadata.pop("seed")
    if not scope:
        edit_metadata.pop("scope")
    if not ranking:
        edit_metadata.pop("ranking")
    if not grouping:
        edit_metadata.pop("grouping")

    return edit_metadata


def _extract_effective_policies(report: RunReport) -> dict[str, Any]:
    from .policy_utils import _extract_effective_policies as _impl

    return _impl(report)


def _normalize_override_entry(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if item is not None]
    return []


def _extract_policy_overrides(report: RunReport) -> list[str]:
    from .policy_utils import _extract_policy_overrides as _impl

    return _impl(report)


def _format_family_caps(caps: Any) -> dict[str, dict[str, float]]:
    from .policy_utils import _format_family_caps as _impl

    return _impl(caps)


def _format_epsilon_map(epsilon_map: Any) -> dict[str, float]:
    from .policy_utils import _format_epsilon_map as _impl

    return _impl(epsilon_map)


def _build_resolved_policies(
    tier: str,
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    variance: dict[str, Any],
    *,
    profile: str | None = None,
    explicit_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from .policy_utils import _build_resolved_policies as _impl

    return _impl(
        tier,
        spectral,
        rmt,
        variance,
        profile=profile,
        explicit_overrides=explicit_overrides,
    )


def _compute_policy_digest(policy: dict[str, Any]) -> str:
    from .policy_utils import _compute_policy_digest as _impl

    return _impl(policy)


def _compute_report_digest(report: RunReport | dict[str, Any] | None) -> str | None:
    if not isinstance(report, dict):
        return None
    meta = report.get("meta", {}) if isinstance(report.get("meta"), dict) else {}
    edit = report.get("edit", {}) if isinstance(report.get("edit"), dict) else {}
    metrics = (
        report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    )
    spectral_metrics = metrics.get("spectral", {})
    rmt_metrics = metrics.get("rmt", {})
    subset = {
        "meta": {
            "model_id": meta.get("model_id"),
            "adapter": meta.get("adapter"),
            "commit": meta.get("commit"),
            "ts": meta.get("ts"),
        },
        "edit": {
            "name": edit.get("name"),
            "plan_digest": edit.get("plan_digest"),
        },
        "metrics": {
            # Legacy PPL fields removed in PM-only surface
            "spectral_caps": spectral_metrics.get("caps_applied")
            if isinstance(spectral_metrics, dict)
            else None,
            "rmt_outliers": rmt_metrics.get("outliers")
            if isinstance(rmt_metrics, dict)
            else None,
        },
    }
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _prepare_guard_overhead_section(
    raw: Any,
) -> tuple[dict[str, Any], bool]:
    """Normalize guard overhead payload and determine whether it passes the gate."""

    return _prepare_guard_overhead_section_impl(
        raw,
        validate_guard_overhead_fn=validate_guard_overhead,
    )


def _compute_quality_overhead_from_guard(
    raw_guard: Any,
    pm_kind_hint: str | None = None,
) -> dict[str, Any] | None:
    """Compute PM-aware quality overhead from guard context when possible."""

    return _compute_quality_overhead_from_guard_impl(
        raw_guard,
        pm_kind_hint,
        compute_primary_metric_from_report_fn=compute_primary_metric_from_report,
        get_metric_fn=get_metric,
    )


def _propagate_pairing_stats(
    evaluation_report: dict[str, Any], ppl_analysis: dict[str, Any] | None
) -> None:
    """Surface pairing statistics inside evaluation_report.dataset.windows.stats."""
    if not isinstance(evaluation_report, dict):
        return
    ds = evaluation_report.get("dataset", {})
    if not isinstance(ds, dict):
        return
    windows = ds.get("windows", {})
    if not isinstance(windows, dict):
        windows = {}
    stats = windows.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    pairing = None
    paired_windows_out = None
    pa_stats = ppl_analysis.get("stats", {}) if isinstance(ppl_analysis, dict) else {}
    try:
        pairing = pa_stats.get("pairing")
        paired_windows_out = pa_stats.get("paired_windows")
        passthrough_keys = (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        )
        for key in passthrough_keys:
            if key in pa_stats:
                stats[key] = pa_stats[key]
        coverage = pa_stats.get("coverage")
        if isinstance(coverage, dict) and coverage:
            stats["coverage"] = coverage
        bootstrap = pa_stats.get("bootstrap")
        if isinstance(bootstrap, dict) and bootstrap:
            stats["bootstrap"] = bootstrap
        paired_delta_summary = pa_stats.get("paired_delta_summary")
        if isinstance(paired_delta_summary, dict) and paired_delta_summary:
            stats["paired_delta_summary"] = paired_delta_summary
        wmf = pa_stats.get("window_match_fraction")
        if wmf is not None:
            stats["window_match_fraction"] = wmf
        wof = pa_stats.get("window_overlap_fraction")
        if wof is not None:
            stats["window_overlap_fraction"] = wof
        wpr = pa_stats.get("window_pairing_reason")
        if wpr is not None:
            stats["window_pairing_reason"] = wpr
    except Exception:  # pragma: no cover
        pairing = None
        paired_windows_out = None
    if pairing is not None:
        stats["pairing"] = pairing
    if paired_windows_out is not None:
        stats.setdefault("paired_windows", paired_windows_out)
    if stats is not windows.get("stats"):
        windows["stats"] = stats
    if windows is not ds.get("windows"):
        ds["windows"] = windows
    evaluation_report["dataset"] = ds


def _build_provenance_block(
    report: RunReport,
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any],
    artifacts_payload: dict[str, Any],
    policy_provenance: dict[str, Any],
    schedule_digest: str | None,
    ppl_analysis: dict[str, Any],
    current_run_id: str,
) -> dict[str, Any]:
    return _build_provenance_block_impl(
        report,
        baseline_raw,
        baseline_ref,
        artifacts_payload,
        policy_provenance,
        schedule_digest,
        ppl_analysis,
        current_run_id,
        compute_report_digest_fn=_compute_report_digest,
        collect_backend_versions_fn=_collect_backend_versions,
        compute_edit_digest_fn=_compute_edit_digest,
    )


def _resolve_pm_acceptance_range_from_report(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from report context/meta."""

    return _resolve_pm_acceptance_range_from_report_impl(report)


def _resolve_pm_drift_band_from_report(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve preview→final drift band from report context/meta."""

    return _resolve_pm_drift_band_from_report_impl(
        report,
        drift_band_default=PM_DRIFT_BAND_DEFAULT,
    )


def _resolve_tiny_relax_from_report(report: dict[str, Any] | None) -> bool:
    """Resolve tiny-relax mode from report context/meta policy fields."""

    return _resolve_tiny_relax_from_report_impl(report)


def _compute_validation_flags(
    ppl: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    invariants: dict[str, Any],
    tier: str = "balanced",
    _ppl_metrics: dict[str, Any] | None = None,
    target_ratio: float | None = None,
    guard_overhead: dict[str, Any] | None = None,
    primary_metric: dict[str, Any] | None = None,
    moe: dict[str, Any] | None = None,
    dataset_capacity: dict[str, Any] | None = None,
    pm_acceptance_range: dict[str, float] | None = None,
    pm_drift_band: dict[str, float] | None = None,
    pm_tail: dict[str, Any] | None = None,
    tiny_relax: bool = False,
) -> dict[str, bool]:
    """Compute validation flags for the evaluation report including canonical gates."""

    return _compute_validation_flags_impl(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics=_ppl_metrics,
        target_ratio=target_ratio,
        guard_overhead=guard_overhead,
        primary_metric=primary_metric,
        moe=moe,
        dataset_capacity=dataset_capacity,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        pm_tail=pm_tail,
        tiny_relax=tiny_relax,
        pm_drift_band_default=PM_DRIFT_BAND_DEFAULT,
        get_tier_policies_fn=get_tier_policies,
    )


def _generate_run_id(report: RunReport) -> str:
    """Generate a unique run ID from report metadata."""
    if isinstance(report, dict):
        meta = report.get("meta", {})
    else:
        meta = getattr(report, "meta", {})

    if isinstance(meta, dict):
        existing = meta.get("run_id")
        if isinstance(existing, str) and existing:
            return existing
        timestamp = str(meta.get("ts", meta.get("start_time", "")))
        model_id = str(meta.get("model_id", "unknown"))
        commit = str(meta.get("commit", meta.get("commit_sha", "")))[:16]
        base_str = f"{timestamp}{model_id}{commit}"
    else:
        base_str = str(meta or report)

    return hashlib.sha256(base_str.encode()).hexdigest()[:16]


## NOTE: _compute_report_hash moved to invarlock.reporting.render and is re-exported below.


def _analyze_bitwidth_map(bitwidth_map: dict[str, Any]) -> dict[str, Any]:
    """Analyze bitwidth changes for compression diagnostics."""
    if not bitwidth_map:
        return {}

    # Extract bitwidth statistics
    bitwidths = []
    for module_info in bitwidth_map.values():
        if isinstance(module_info, dict) and "bitwidth" in module_info:
            bitwidths.append(module_info["bitwidth"])

    if not bitwidths:
        return {}

    return {
        "total_modules": len(bitwidths),
        "bitwidths_used": list(set(bitwidths)),
        "avg_bitwidth": sum(bitwidths) / len(bitwidths),
        "min_bitwidth": min(bitwidths),
        "max_bitwidth": max(bitwidths),
    }


def _compute_savings_summary(deltas: dict[str, Any]) -> dict[str, Any]:
    """Compute realized vs theoretical savings summary for edits."""
    summary = _get_mapping(deltas, "savings")
    rank_map = _get_mapping(deltas, "rank_map")
    deploy_mode: str | None = summary.get("deploy_mode") if summary else None

    def _accumulate(value: Any) -> int:
        coerced = _coerce_int(value)
        return coerced if coerced is not None else 0

    if rank_map:
        total_realized = 0
        total_theoretical = 0
        for info in rank_map.values():
            total_realized += _accumulate(info.get("realized_params_saved"))
            total_theoretical += _accumulate(info.get("theoretical_params_saved"))
            if deploy_mode is None:
                mode_candidate = info.get("deploy_mode")
                if isinstance(mode_candidate, str):
                    deploy_mode = mode_candidate
    else:
        total_realized = (
            _accumulate(summary.get("total_realized_params_saved")) if summary else 0
        )
        total_theoretical = (
            _accumulate(summary.get("total_theoretical_params_saved")) if summary else 0
        )

    mode = "none"
    if total_realized > 0:
        mode = "realized"
    elif total_theoretical > 0:
        mode = "theoretical"
    elif deploy_mode == "recompose" and any(
        isinstance(info, dict) and not info.get("skipped", False)
        for info in rank_map.values()
    ):
        mode = "theoretical"

    result = {
        "mode": mode,
        "total_realized_params_saved": total_realized,
        "total_theoretical_params_saved": total_theoretical,
    }
    if deploy_mode:
        result["deploy_mode"] = deploy_mode
    return result


def _extract_rank_information(
    edit_config: dict[str, Any], deltas: dict[str, Any]
) -> dict[str, Any]:
    """Extract rank information for SVD-based compression."""
    rank_info = {}

    # Extract from config
    if "frac" in edit_config:
        rank_info["target_fraction"] = edit_config["frac"]
    if "rank_policy" in edit_config:
        rank_info["rank_policy"] = edit_config["rank_policy"]

    rank_map = deltas.get("rank_map")
    if isinstance(rank_map, dict) and rank_map:
        per_module = {}
        skipped = []
        for module_name, info in rank_map.items():
            per_module[module_name] = {
                "rank": info.get("rank"),
                "params_saved": info.get("params_saved"),
                "energy_retained": info.get("energy_retained"),
                "deploy_mode": info.get("deploy_mode"),
                "savings_mode": info.get("savings_mode"),
                "realized_params_saved": info.get("realized_params_saved"),
                "theoretical_params_saved": info.get("theoretical_params_saved"),
                "realized_params": info.get("realized_params"),
                "theoretical_params": info.get("theoretical_params"),
            }
            if info.get("skipped"):
                skipped.append(module_name)

        rank_info["per_module"] = per_module
        if skipped:
            rank_info["skipped_modules"] = skipped
        rank_info["savings_summary"] = _compute_savings_summary(deltas)

    else:
        summary = _get_mapping(deltas, "savings")
        if summary:
            rank_info["savings_summary"] = _compute_savings_summary(deltas)

    return rank_info


def _extract_compression_diagnostics(
    edit_name: str,
    edit_config: dict[str, Any],
    deltas: dict[str, Any],
    structure: dict[str, Any],
    inference_record: dict[str, Any],
) -> dict[str, Any]:
    """Extract comprehensive compression diagnostics."""
    diagnostics = {}

    if inference_record is None:
        inference_record = {
            "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
            "sources": {},
            "log": [],
        }

    def mark(field: str, value: Any, source: str) -> bool:
        if value in (None, "unknown"):
            return False
        current = edit_config.get(field)
        if current not in (None, "unknown"):
            return False
        edit_config[field] = value
        if not inference_record["flags"].get(field):
            inference_record["flags"][field] = True
            inference_record.setdefault("sources", {})[field] = source
            inference_record.setdefault("log", []).append(
                f"{field} inferred from {source}: {value}"
            )
        return True

    # Determine execution status
    params_changed = deltas.get("params_changed", 0)
    if params_changed > 0:
        diagnostics["execution_status"] = "successful"
    else:
        diagnostics["execution_status"] = "no_modifications"

    # Enhanced target module analysis with detailed extraction
    bitwidth_map = deltas.get("bitwidth_map", {})
    num_quantized_modules = len(bitwidth_map) if bitwidth_map else 0

    diagnostics["target_analysis"] = {
        # Without a separate planned target list, treat "found/eligible" as the
        # set of modules that satisfied selection and were considered by the
        # algorithm in this run; "modified" reflects the modules actually
        # quantized (bitwidth_map entries).
        "modules_found": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "modules_eligible": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "modules_modified": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "scope": edit_config.get("scope", "unknown"),
    }
    existing_scope = edit_config.get("scope")
    if existing_scope not in (None, "unknown"):
        diagnostics["target_analysis"]["scope"] = existing_scope
    else:
        module_iter: Iterable[str]
        source_label = "modules"
        if isinstance(bitwidth_map, dict) and bitwidth_map:
            module_iter = bitwidth_map.keys()
            source_label = "bitwidth_map"
        elif isinstance(deltas.get("rank_map"), dict) and deltas["rank_map"]:
            module_iter = deltas["rank_map"].keys()
            source_label = "rank_map"
        else:
            module_iter = []
        inferred_scope = _infer_scope_from_modules(module_iter)
        if inferred_scope != "unknown" and mark("scope", inferred_scope, source_label):
            diagnostics["target_analysis"]["scope"] = inferred_scope
    diagnostics["target_analysis"]["scope"] = edit_config.get(
        "scope", diagnostics["target_analysis"].get("scope", "unknown")
    )

    # Enhanced parameter effectiveness analysis
    param_analysis = {}

    if deltas.get("rank_map"):
        rank_map = deltas["rank_map"]
        modules_modified = [
            name for name, info in rank_map.items() if not info.get("skipped", False)
        ]
        diagnostics["rank_summary"] = {
            "modules": rank_map,
            "modules_modified": len(modules_modified),
            "skipped_modules": [
                name for name, info in rank_map.items() if info.get("skipped", False)
            ],
        }
        diagnostics["target_analysis"]["modules_modified"] = len(modules_modified)
        if modules_modified:
            diagnostics["execution_status"] = (
                "partial"
                if len(modules_modified) < len(rank_map)
                else diagnostics["execution_status"]
            )

    if "quant" in edit_name.lower():
        # Extract actual bitwidth from bitwidth_map or config
        actual_bitwidth: Any = "unknown"
        if bitwidth_map:
            # Get bitwidth from first module in bitwidth_map
            first_module: dict[str, Any] = next(iter(bitwidth_map.values()), {})
            actual_bitwidth = first_module.get("bitwidth", edit_config.get("bitwidth", "unknown"))
        else:
            actual_bitwidth = edit_config.get("bitwidth", "unknown")

        param_analysis["bitwidth"] = {
            "value": actual_bitwidth,
            "effectiveness": "applied" if params_changed > 0 else "ineffective",
        }

        # Extract group_size info
        if bitwidth_map:
            first_module = next(iter(bitwidth_map.values()), {})
            group_size_used = first_module.get("group_size")
            param_analysis["group_size"] = {
                "value": group_size_used,
                "effectiveness": "used" if group_size_used else "per_channel",
            }
        elif edit_config.get("group_size") not in (None, "unknown"):
            group_size_cfg = edit_config["group_size"]
            param_analysis["group_size"] = {
                "value": group_size_cfg,
                "effectiveness": "used" if group_size_cfg else "per_channel",
            }

        # Extract clamp_ratio
        if edit_config.get("clamp_ratio") not in (None, "unknown"):
            param_analysis["clamp_ratio"] = {
                "value": edit_config["clamp_ratio"],
                "effectiveness": "applied"
                if edit_config["clamp_ratio"] > 0
                else "disabled",
            }

    elif "svd" in edit_name.lower() or "rank" in edit_name.lower():
        # SVD-specific analysis
        param_analysis["frac"] = {
            "value": edit_config.get("frac", "unknown"),
            "effectiveness": "applied" if params_changed > 0 else "too_conservative",
        }
        param_analysis["rank_policy"] = {
            "value": edit_config.get("rank_policy", "unknown"),
            "effectiveness": "used",
        }

    diagnostics["parameter_analysis"] = param_analysis

    # Enhanced algorithm-specific details
    algo_details = {}
    algo_details["scope_targeting"] = edit_config.get("scope", "unknown")
    algo_details["seed"] = edit_config.get("seed", "unknown")

    # Add quantization-specific details
    if "quant" in edit_name.lower() and bitwidth_map:
        algo_details["modules_quantized"] = len(bitwidth_map)
        algo_details["quantization_type"] = (
            "per_channel"
            if not any(m.get("group_size") for m in bitwidth_map.values())
            else "grouped"
        )

        # Calculate total params quantized
        total_quantized_params = sum(m.get("params", 0) for m in bitwidth_map.values())
        algo_details["total_params_quantized"] = total_quantized_params

        # Memory estimate (rough)
        memory_saved_bytes = 0
        if isinstance(actual_bitwidth, int) and actual_bitwidth < 32:
            memory_saved_bytes = total_quantized_params * (32 - actual_bitwidth) / 8

        algo_details["estimated_memory_saved_mb"] = round(
            memory_saved_bytes / (1024 * 1024), 2
        )

    diagnostics["algorithm_details"] = algo_details

    # Generate warnings based on analysis (fewer and non-prescriptive for successful runs)
    warnings = []
    if params_changed == 0:
        warnings.append(
            "No parameters were modified - algorithm may be too conservative"
        )
        warnings.append("Check scope configuration and parameter thresholds")

        if edit_config.get("scope") == "ffn":
            warnings.append(
                "FFN scope may not match model architecture - try 'all' scope"
            )

        if "frac" in edit_config and edit_config["frac"] < 0.1:
            warnings.append(
                f"Fraction {edit_config['frac']} may be too small for meaningful compression"
            )
    else:
        # Success case – keep diagnostics descriptive only, avoid suggesting
        # specific alternative edit parameters to remain edit-agnostic.
        pass

    diagnostics["warnings"] = warnings

    diagnostics["inferred"] = inference_record["flags"]
    if inference_record.get("sources"):
        diagnostics["inference_source"] = inference_record["sources"]
    if inference_record.get("log"):
        diagnostics["inference_log"] = inference_record["log"]

    return diagnostics


## Note: compute_window_hashes is available under invarlock.reporting.dataset_hashing.

__all__ = [
    "make_report",
    "validate_report",
    "_validate_with_jsonschema",
    "jsonschema",
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
]
