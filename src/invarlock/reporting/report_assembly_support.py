from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from typing import Any

from invarlock.public_contracts import load_json_contract
from invarlock.utils.digest import hash_json

from .report_schema import REPORT_JSON_SCHEMA

POLICY_VERSION = "policy-v1"

_VALIDATION_ALLOWLIST_DEFAULT = {
    "primary_metric_acceptable",
    "primary_metric_tail_acceptable",
    "preview_final_drift_acceptable",
    "guard_overhead_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    "hysteresis_applied",
    "moe_observed",
    "moe_identity_ok",
}


def is_ppl_kind(name: Any) -> bool:
    """Return True if a primary-metric kind denotes a ppl-like metric."""
    try:
        normalized = str(name or "").lower()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        normalized = ""
    return normalized in {
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


def compute_edit_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Compute a minimal, non-leaky edit breadcrumb for provenance."""
    edits: dict[str, Any] = {}
    try:
        raw_edit = report.get("edit")
        if isinstance(raw_edit, dict):
            edits = raw_edit
        else:
            provenance = report.get("provenance")
            if isinstance(provenance, dict):
                raw_provenance_edit = provenance.get("edits")
                if isinstance(raw_provenance_edit, dict):
                    edits = raw_provenance_edit
    except (AttributeError, RuntimeError, TypeError, ValueError):
        edits = {}

    family = "cert_only"
    impl_hash = hash_json({"family": family})
    try:
        if str(edits.get("name", "")) == "quant_rtn":
            family = "quantization"
            cfg = edits.get("config", {})
            if not isinstance(cfg, dict):
                cfg = {}
            impl_hash = hash_json({"name": "quant_rtn", "config": cfg})
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return {"family": family, "impl_hash": impl_hash, "version": 1}


def compute_confidence_label(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    """Compute evaluation-report confidence label based on stability and CI width."""
    validation = evaluation_report.get("validation", {}) or {}
    pm_ok = bool(validation.get("primary_metric_acceptable", False))
    basis = "primary_metric"
    lo = hi = float("nan")
    try:
        pm = evaluation_report.get("primary_metric", {}) or {}
        kind = str(pm.get("kind", "") or "").lower()
        display_ci = pm.get("display_ci")
        if isinstance(display_ci, tuple | list) and len(display_ci) == 2:
            lo, hi = float(display_ci[0]), float(display_ci[1])
            if kind.startswith("ppl"):
                basis = "ppl_ratio"
            elif kind in {"accuracy", "vqa_accuracy"}:
                basis = kind
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    width = hi - lo if math.isfinite(lo) and math.isfinite(hi) else float("nan")
    thr_ratio = 0.03
    thr_pp = 1.0
    try:
        pol = evaluation_report.get("resolved_policy")
        if isinstance(pol, dict):
            conf_pol = pol.get("confidence")
            if isinstance(conf_pol, dict):
                rr = conf_pol.get("ppl_ratio_width_max")
                if isinstance(rr, (int, float)):
                    thr_ratio = float(rr)
                ap = conf_pol.get("accuracy_delta_pp_width_max")
                if isinstance(ap, (int, float)):
                    thr_pp = float(ap)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    thr = thr_pp if basis in {"accuracy", "vqa_accuracy"} else thr_ratio

    try:
        unstable = bool((evaluation_report.get("primary_metric") or {}).get("unstable"))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        unstable = False

    label = "Low"
    if pm_ok:
        if (not unstable) and math.isfinite(width) and width <= thr:
            label = "High"
        elif math.isfinite(width) and width <= 2 * thr:
            label = "Medium"
        else:
            label = "Medium" if unstable else "Low"

    return {
        "label": label,
        "basis": basis,
        "width": width,
        "threshold": thr,
        "unstable": unstable,
    }


def collect_backend_versions() -> dict[str, Any]:
    """Collect backend/library versions for provenance.env_flags."""
    info: dict[str, Any] = {}
    try:
        info["python"] = platform.python_version()
        info["platform"] = platform.platform()
        info["machine"] = platform.machine()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

    try:  # pragma: no cover - depends on torch availability
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        tv = getattr(torch, "version", None)
        if tv is not None:
            info["torch_cuda"] = getattr(tv, "cuda", None)
            info["torch_cudnn"] = getattr(tv, "cudnn", None)
            info["torch_git"] = getattr(tv, "git_version", None)
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["device_name"] = getattr(props, "name", None)
                maj = getattr(props, "major", None)
                minr = getattr(props, "minor", None)
                if maj is not None and minr is not None:
                    info["sm_capability"] = f"{int(maj)}.{int(minr)}"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            if hasattr(torch.backends, "cudnn") and hasattr(
                torch.backends.cudnn, "version"
            ):
                v = torch.backends.cudnn.version()
                info["cudnn_runtime"] = int(v) if v is not None else None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            nccl_mod = getattr(torch.cuda, "nccl", None)
            if nccl_mod is not None and hasattr(nccl_mod, "version"):
                info["nccl"] = str(nccl_mod.version())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            tf32: dict[str, Any] = {}
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
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    except ImportError:  # pragma: no cover - torch not available
        pass

    try:
        cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas:
            info["cublas_workspace_config"] = cublas
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

    return {k: v for k, v in info.items() if v is not None}


def _load_validation_allowlist_default() -> set[str]:
    return set(_VALIDATION_ALLOWLIST_DEFAULT)


def load_validation_allowlist_with_source() -> tuple[set[str], str]:
    """Load validation key allow-list and report the source explicitly."""
    try:
        data = load_json_contract("validation_keys.json")
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return _load_validation_allowlist_default(), "fallback:load-error"
    if isinstance(data, list):
        return {str(k) for k in data}, "contracts"
    return (
        _load_validation_allowlist_default(),
        "fallback:invalid-contract-validation-keys",
    )


def load_validation_allowlist() -> set[str]:
    """Load validation key allow-list from contracts with fail-closed fallback."""
    keys, _ = load_validation_allowlist_with_source()
    return keys


def apply_validation_allowlist_schema(validation_keys: set[str]) -> None:
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


def fallback_paired_windows(
    paired_windows: int, coverage_summary: dict[str, Any]
) -> int:
    """Use coverage preview counts when explicit pairing is unavailable."""
    if paired_windows > 0 or not isinstance(coverage_summary, dict):
        return paired_windows
    cprev = coverage_summary.get("preview")
    if isinstance(cprev, dict):
        used = cprev.get("used")
        if isinstance(used, (int, float)) and used >= 0:
            return int(used)
    return paired_windows


def propagate_pairing_stats(
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
    if isinstance(pa_stats, dict):
        pairing = pa_stats.get("pairing")
        paired_windows_out = pa_stats.get("paired_windows")
        for key in (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        ):
            if key in pa_stats:
                stats[key] = pa_stats[key]
        for key in ("coverage", "bootstrap", "paired_delta_summary"):
            value = pa_stats.get(key)
            if isinstance(value, dict) and value:
                stats[key] = value
        for key in ("window_match_fraction", "window_overlap_fraction"):
            value = pa_stats.get(key)
            if value is not None:
                stats[key] = value
        value = pa_stats.get("window_pairing_reason")
        if value is not None:
            stats["window_pairing_reason"] = value

    if pairing is not None:
        stats["pairing"] = pairing
    if paired_windows_out is not None:
        stats.setdefault("paired_windows", paired_windows_out)
    if stats is not windows.get("stats"):
        windows["stats"] = stats
    if windows is not ds.get("windows"):
        ds["windows"] = windows
    evaluation_report["dataset"] = ds


def enforce_drift_ratio_identity(
    paired_windows: int,
    delta_mean: Any,
    drift_ratio: float,
    window_plan_profile: str | None,
) -> float | None:
    """Ensure exp(delta_mean) aligns with observed drift ratio."""
    if not (
        paired_windows > 0
        and isinstance(delta_mean, (int, float))
        and math.isfinite(delta_mean)
        and isinstance(drift_ratio, (int, float))
        and math.isfinite(drift_ratio)
    ):
        return None
    ratio_from_delta = math.exp(float(delta_mean))
    tolerance = 1e-3 * max(1.0, abs(drift_ratio))
    if abs(ratio_from_delta - drift_ratio) > tolerance:
        profile = (window_plan_profile or "dev").lower()
        if profile in {"ci", "release"}:
            raise ValueError(
                "Paired ΔlogNLL mean is inconsistent with reported drift ratio."
            )
    return ratio_from_delta


def enforce_ratio_ci_alignment(
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
            isinstance(observed, (int, float))
            and math.isfinite(observed)
            and isinstance(expected, (int, float))
            and math.isfinite(expected)
        ):
            continue
        tolerance = 5e-4 * max(1.0, abs(expected))
        if abs(float(observed) - float(expected)) > tolerance:
            raise ValueError(
                "Paired ΔlogNLL CI mismatch: ratio bounds do not match exp(Δlog bounds)."
            )


def enforce_display_ci_alignment(
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
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return
    if not kind.startswith("ppl"):
        return

    def _finite_bounds(bounds: Any) -> bool:
        return (
            isinstance(bounds, tuple | list)
            and len(bounds) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds)
        )

    try:
        ci = primary_metric.get("ci")
        display_ci = primary_metric.get("display_ci")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return
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


def enforce_pairing_and_coverage(
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
        isinstance(match_fraction, (int, float))
        and math.isfinite(float(match_fraction))
    ):
        raise ValueError("CI/Release requires window_match_fraction.")
    if float(match_fraction) < 0.999999:
        raise ValueError(
            f"CI/Release requires perfect pairing (window_match_fraction={float(match_fraction):.6f})."
        )

    if not (
        isinstance(overlap_fraction, (int, float))
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


def compute_report_digest(report: dict[str, Any] | None) -> str | None:
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
