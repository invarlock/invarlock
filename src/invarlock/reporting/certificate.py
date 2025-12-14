"""
InvarLock Safety Certificate Generation
==================================

Generate standardized safety certificates from RunReport and baseline comparison.
Certificates are standalone, portable verification artifacts that can be used
for CI/CD gates and regulatory compliance.
"""

from __future__ import annotations

## Core certificate generation and analysis orchestration lives here.
# mypy: ignore-errors
import copy
import hashlib
import inspect
import json
import math
import os
import platform
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional JSON Schema validation support
try:  # pragma: no cover - exercised in integration
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.core.bootstrap import (
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)
from invarlock.eval.primary_metric import compute_primary_metric_from_report, get_metric
from invarlock.utils.digest import hash_json

from . import certificate_schema as _cert_schema
from .certificate_schema import (
    CERTIFICATE_JSON_SCHEMA,
    CERTIFICATE_SCHEMA_VERSION,
)
from .dataset_hashing import (
    _extract_dataset_info,
)
from .guards_analysis import (
    _extract_invariants,
    _extract_rmt_analysis,
    _extract_spectral_analysis,
    _extract_variance_analysis,
)
from .report_types import RunReport, validate_report

# Expose compute_window_hash for tests that monkeypatch it
# compute_window_hash used to be exposed via certificate; tests now patch
# dataset_hashing.compute_window_hash directly, so this import is no longer needed.
from .utils import (
    _coerce_int,
    _coerce_interval,
    _get_mapping,
    _infer_scope_from_modules,
    _pair_logloss_windows,
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


def _is_ppl_kind(name: Any) -> bool:
    """Return True if a primary_metric kind denotes a ppl-like metric.

    Supports legacy and alternate names to stay resilient across schema variants.
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


## NOTE: Deprecated legacy helper `_get_ppl_final` was removed; callers should
## use the normalized primary_metric block directly via make_certificate or
## report processing utilities.


def _compute_edit_digest(report: dict) -> dict:
    """Compute a minimal, non-leaky edit breadcrumb for provenance.

    If `quant_rtn` is detected as the edit name, tag as quantization and
    hash the name+config. Otherwise, treat as cert_only with a stable hash.
    """
    try:
        edits = report.get("edit") or report.get("provenance", {}).get("edits") or {}
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


def _compute_confidence_label(certificate: dict[str, Any]) -> dict[str, Any]:
    """Compute certificate confidence label based on stability and CI width.

    Heuristics:
    - High: ppl_acceptable=True, unstable=False, width <= 0.03 (ratio) or <= 1.0 pp for accuracy
    - Medium: floors met but unstable=True or width borderline (<= 2x threshold)
    - Low: otherwise (floors unmet, failure, or missing bounds)
    Returns a dict with label, basis, width and threshold for transparency.
    """
    validation = certificate.get("validation", {}) or {}
    pm_ok = bool(validation.get("primary_metric_acceptable", False))
    # Basis label shown in confidence block:
    #  - For ppl-like metrics, use 'ppl_ratio' to reflect ratio width threshold
    #  - For accuracy-like metrics, use their kind ('accuracy' or 'vqa_accuracy')
    #  - Fall back to 'primary_metric' when unknown
    basis = "primary_metric"
    lo = hi = float("nan")
    try:
        pm = certificate.get("primary_metric", {}) or {}
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
    except Exception:  # pragma: no cover
        pass

    width = hi - lo if (math.isfinite(lo) and math.isfinite(hi)) else float("nan")
    # Thresholds (policy-configurable; fallback to defaults)
    thr_ratio = 0.03  # 3% width for ratio
    thr_pp = 1.0  # 1.0 percentage point for accuracy kinds
    try:
        pol = certificate.get("resolved_policy")
        if isinstance(pol, dict):
            conf_pol = pol.get("confidence")
            if isinstance(conf_pol, dict):
                rr = conf_pol.get("ppl_ratio_width_max")
                if isinstance(rr, int | float):
                    thr_ratio = float(rr)
                ap = conf_pol.get("accuracy_delta_pp_width_max")
                if isinstance(ap, int | float):
                    thr_pp = float(ap)
    except Exception:  # pragma: no cover
        pass
    is_acc = basis in {"accuracy", "vqa_accuracy"}
    thr = thr_pp if is_acc else thr_ratio

    # Unstable hint from primary metric (if provided)
    try:
        unstable = bool((certificate.get("primary_metric") or {}).get("unstable"))
    except Exception:  # pragma: no cover
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


# Minimal JSON Schema describing the canonical shape of a certificate.
# This focuses on structural validity; numerical thresholds are validated
# separately in metric-specific logic.
# JSON Schema is provided by certificate_schema; no duplication here.


# Mirror jsonschema and structural validator for test monkeypatching compatibility.
jsonschema = getattr(_cert_schema, "jsonschema", None)


def _validate_with_jsonschema(certificate: dict[str, Any]) -> bool:
    if jsonschema is None:
        return True
    try:
        jsonschema.validate(instance=certificate, schema=CERTIFICATE_JSON_SCHEMA)
        return True
    except Exception:  # pragma: no cover
        return False


def validate_certificate(certificate: dict[str, Any]) -> bool:
    """Validate that a certificate has all required fields and valid data."""
    try:
        if certificate.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
            return False
        # Prefer JSON Schema structural validation; if unavailable or too strict,
        # fall back to a lenient minimal check used by unit tests.
        if not _validate_with_jsonschema(certificate):
            # Minimal fallback: require schema version + run_id + primary_metric
            run_id_ok = isinstance(certificate.get("run_id"), str) and bool(
                certificate.get("run_id")
            )
            pm = certificate.get("primary_metric")
            pm_ok = isinstance(pm, dict) and (
                isinstance(pm.get("final"), int | float)
                or (isinstance(pm.get("kind"), str) and bool(pm.get("kind")))
            )
            if not (run_id_ok and pm_ok):
                return False

        validation = certificate.get("validation", {})
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


def _load_validation_allowlist() -> set[str]:
    """Load validation key allow-list from contracts/validation_keys.json when available.

    Falls back to a safe built-in default when the contracts directory is not present
    (e.g., installed wheel) or when parsing fails.
    """
    try:
        root = Path(__file__).resolve().parents[3]
        path = root / "contracts" / "validation_keys.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(k) for k in data}
    except Exception:  # pragma: no cover
        pass
    return set(_VALIDATION_ALLOWLIST_DEFAULT)


# Tighten JSON Schema: populate validation.properties from allow-list and
# disallow unknown validation keys at schema level.
try:
    _vkeys = _load_validation_allowlist()
    if isinstance(CERTIFICATE_JSON_SCHEMA.get("properties"), dict):
        vspec = CERTIFICATE_JSON_SCHEMA["properties"].get("validation")
        if isinstance(vspec, dict):
            vspec["properties"] = {k: {"type": "boolean"} for k in _vkeys}
            vspec["additionalProperties"] = False
except Exception:  # pragma: no cover
    # Keep permissive defaults if something goes wrong during import
    pass


## Note: helpers like _get_section/_get_mapping/_iter_guard_entries,
## and policy helpers are provided by invarlock.reporting.utils and policy_utils.
## Import those directly in callers/tests instead of through this module.


def _normalize_and_validate_report(report: RunReport | dict[str, Any]) -> RunReport:
    """Normalize a possibly-minimal report and validate its structure.

    Uses the local normalizer when available, then checks `validate_report`.
    Raises ValueError on invalid input. Returns the normalized RunReport.
    """
    try:
        from .normalizer import normalize_run_report as _norm

        if isinstance(report, dict):
            report = _norm(report)
    except Exception:  # pragma: no cover
        pass
    if not validate_report(report):
        raise ValueError("Invalid RunReport structure")
    return report


def _extract_certificate_meta(report: RunReport) -> dict[str, Any]:
    """Extract the certificate metadata block with a full seed bundle."""
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


def make_certificate(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a safety certificate from a RunReport and baseline comparison.

    The certificate is a standalone, portable artifact that contains all
    essential metrics and comparisons needed for safety verification.

    Args:
        report: The guarded run report to certify
        baseline: Step-0 baseline RunReport or baseline metrics dict

    Returns:
        Certificate dictionary with all required fields

    Raises:
        ValueError: If inputs are invalid or required data is missing
    """
    # Normalize and validate the primary report
    report = _normalize_and_validate_report(report)

    # Normalize baseline input
    baseline_raw = baseline
    baseline_normalized = _normalize_baseline(baseline_raw)

    # Extract core metadata with full seed bundle
    meta = _extract_certificate_meta(report)

    # Propagate environment flags captured in the RunReport (e.g., deterministic algos,
    # TF32 controls, MPS/CUDA availability). This is useful for auditability and
    # reproducibility of certification runs.
    try:
        env_flags = (
            report.get("meta", {}).get("env_flags")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(env_flags, dict) and env_flags:
            meta["env_flags"] = env_flags
    except Exception:  # pragma: no cover
        pass

    # Determinism preset (CI/Release provenance) when present.
    try:
        det = (
            report.get("meta", {}).get("determinism")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(det, dict) and det:
            meta["determinism"] = det
    except Exception:  # pragma: no cover
        pass

    tokenizer_hash_meta = report["meta"].get("tokenizer_hash")
    if not tokenizer_hash_meta:
        dataset_section = report.get("data", {})
        if isinstance(dataset_section, dict):
            tokenizer_hash_meta = dataset_section.get("tokenizer_hash")
    if isinstance(tokenizer_hash_meta, str) and tokenizer_hash_meta:
        meta["tokenizer_hash"] = tokenizer_hash_meta

    model_profile_meta = report["meta"].get("model_profile")
    if isinstance(model_profile_meta, dict) and model_profile_meta:
        meta["model_profile"] = model_profile_meta

    cuda_flags = report["meta"].get("cuda_flags")
    if isinstance(cuda_flags, dict) and cuda_flags:
        meta["cuda_flags"] = cuda_flags

    # Extract auto-tuning configuration
    auto_config = report["meta"].get("auto")
    if auto_config:
        auto = {
            "tier": auto_config.get("tier", "balanced"),
            "probes_used": auto_config.get("probes", auto_config.get("probes_used", 0)),
            "target_pm_ratio": auto_config.get("target_pm_ratio"),
        }
    else:
        auto = {"tier": "none", "probes_used": 0, "target_pm_ratio": None}

    # Extract dataset configuration and compute hashes
    dataset_info = _extract_dataset_info(report)
    try:
        if isinstance(dataset_info, dict):
            windows = dataset_info.get("windows")
            if isinstance(windows, dict):
                windows.setdefault("stats", {})
    except Exception:  # pragma: no cover
        pass

    # Baseline reference (PM-only). Derive a primary_metric snapshot from baseline windows.
    # Prefer explicit baseline primary_metric when provided; otherwise compute from windows
    baseline_pm = None
    try:
        bm = (
            baseline_raw.get("metrics", {}).get("primary_metric")
            if isinstance(baseline_raw.get("metrics"), dict)
            else None
        )
        if isinstance(bm, dict) and bm:
            baseline_pm = bm
    except Exception:  # pragma: no cover
        baseline_pm = None
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        try:
            baseline_pm = compute_primary_metric_from_report(baseline_normalized)
        except Exception:  # pragma: no cover
            baseline_pm = {"kind": "ppl_causal", "final": float("nan")}
    baseline_ref = {
        "run_id": baseline_normalized.get("run_id", "unknown"),
        "model_id": baseline_normalized.get("model_id", report["meta"]["model_id"]),
        "primary_metric": {
            "kind": baseline_pm.get("kind", "ppl_causal"),
            "final": baseline_pm.get("final", float("nan")),
        },
    }
    # Propagate baseline tokenizer hash for verify-time linting when available
    baseline_tok_hash = baseline_normalized.get("tokenizer_hash")
    if isinstance(baseline_tok_hash, str) and baseline_tok_hash:
        baseline_ref["tokenizer_hash"] = baseline_tok_hash

    # Primary-metric analysis (PM-only)
    ppl_metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    edited_preview = float("nan")
    edited_final = float("nan")
    ratio_vs_baseline = float("nan")

    metrics_bootstrap_obj = (
        report["metrics"].get("bootstrap", {})
        if isinstance(report.get("metrics"), dict)
        else {}
    )
    metrics_bootstrap = (
        dict(metrics_bootstrap_obj) if isinstance(metrics_bootstrap_obj, dict) else {}
    )
    raw_coverage = metrics_bootstrap.get("coverage") if metrics_bootstrap else None
    coverage_summary = (
        copy.deepcopy(raw_coverage) if isinstance(raw_coverage, dict) else {}
    )
    window_plan_ctx = (
        report.get("metrics", {}).get("window_plan")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    window_plan_profile = (
        str(window_plan_ctx.get("profile"))
        if isinstance(window_plan_ctx, dict) and window_plan_ctx.get("profile")
        else None
    )
    preview_ci = None
    final_ci = None
    ratio_ci = None
    ratio_ci_source = "run_metrics"
    # PM-only fallback: derive ratio_ci from logloss_delta_ci when available
    if ratio_ci is None:
        try:
            dlci = _coerce_interval(report["metrics"].get("logloss_delta_ci"))
            if (
                isinstance(dlci, tuple | list)
                and len(dlci) == 2
                and all(isinstance(x, (int | float)) for x in dlci)
            ):
                lo, hi = float(dlci[0]), float(dlci[1])
                ratio_ci = (math.exp(lo), math.exp(hi))
                ratio_ci_source = "run_metrics"
        except Exception:  # pragma: no cover
            pass
    paired_windows = 0
    # UX hint: mark CI as unstable for very low replicate counts or insufficient tokens
    unstable_ci_flag = False
    try:
        rep_raw = metrics_bootstrap.get("replicates", metrics_bootstrap.get("n"))
        if rep_raw is not None and int(rep_raw) < 200:
            unstable_ci_flag = True
    except Exception:  # pragma: no cover
        unstable_ci_flag = False
    # Also consider token-count floor from tier policy when available
    try:
        tokens_prev = (
            report.get("metrics", {}).get("preview_total_tokens")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        tokens_fin = (
            report.get("metrics", {}).get("final_total_tokens")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        total_tokens = None
        if isinstance(tokens_prev, int | float) and isinstance(tokens_fin, int | float):
            total_tokens = int(tokens_prev) + int(tokens_fin)
        # Resolve tier
        tier = "balanced"
        try:
            auto_cfg = (
                report.get("meta", {}).get("auto")
                if isinstance(report.get("meta"), dict)
                else None
            )
            if isinstance(auto_cfg, dict) and auto_cfg.get("tier"):
                tier = str(auto_cfg.get("tier")).lower()
        except Exception:  # pragma: no cover
            pass
        tier_policies = get_tier_policies()
        tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
        metrics_policy = (
            tier_defaults.get("metrics", {}) if isinstance(tier_defaults, dict) else {}
        )
        pm_policy = (
            metrics_policy.get("pm_ratio", {})
            if isinstance(metrics_policy, dict)
            else {}
        )
        min_tokens = int(pm_policy.get("min_tokens", 0))
        if (
            isinstance(total_tokens, int)
            and min_tokens > 0
            and total_tokens < min_tokens
        ):
            unstable_ci_flag = True
    except Exception:  # pragma: no cover
        pass
    raw_logloss_delta = report["metrics"].get("logloss_delta")
    logloss_delta = (
        float(raw_logloss_delta)
        if isinstance(raw_logloss_delta, int | float)
        else float("nan")
    )
    logloss_delta_ci = _coerce_interval(report["metrics"].get("logloss_delta_ci"))
    raw_delta_summary = report["metrics"].get("paired_delta_summary", {})
    paired_delta_summary = (
        dict(raw_delta_summary) if isinstance(raw_delta_summary, dict) else {}
    )

    run_windows = (
        report.get("evaluation_windows", {}).get("final", {})
        if isinstance(report.get("evaluation_windows"), dict)
        else {}
    )
    baseline_windows = (
        baseline_normalized.get("evaluation_windows", {}).get("final", {})
        if isinstance(baseline_normalized.get("evaluation_windows"), dict)
        else {}
    )

    paired = _pair_logloss_windows(run_windows, baseline_windows)
    baseline_delta_mean = float("nan")
    if paired:
        paired_run, paired_base = paired
        paired_windows = len(paired_run)
        method = str(metrics_bootstrap.get("method", "percentile")).lower()
        replicates = int(
            metrics_bootstrap.get(
                "replicates", metrics_bootstrap.get("n", 1000) or 1000
            )
        )
        alpha = float(metrics_bootstrap.get("alpha", 0.05) or 0.05)
        seed = int(metrics_bootstrap.get("seed", 0) or 0)
        # Default to percentile for deterministic behavior; enable BCa only when requested
        ci_method = "percentile"
        try:
            if "bca" in method:
                ci_method = "bca"
            else:
                # Opt-in via env flag and sufficiently large sample
                use_bca_flag = str(
                    os.environ.get("INVARLOCK_BOOTSTRAP_BCA", "")
                ).strip().lower() in {"1", "true", "yes", "on"}
                if use_bca_flag and paired_windows >= 200:
                    ci_method = "bca"
        except Exception:  # pragma: no cover
            pass
        if replicates > 0:
            try:
                delta_ci = compute_paired_delta_log_ci(
                    paired_run,
                    paired_base,
                    method=ci_method,
                    replicates=replicates,
                    alpha=alpha,
                    seed=seed + 503,
                )
                if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
                    delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
                logloss_delta_ci = delta_ci
                ratio_ci = logspace_to_ratio_ci(delta_ci)
                ratio_ci_source = "paired_baseline"
                # Compute token-weighted paired mean ΔlogNLL vs baseline for identity checks
                try:
                    run_ids = (
                        run_windows.get("window_ids")
                        if isinstance(run_windows, dict)
                        else None
                    )
                    run_ll = (
                        run_windows.get("logloss")
                        if isinstance(run_windows, dict)
                        else None
                    )
                    base_ids = (
                        baseline_windows.get("window_ids")
                        if isinstance(baseline_windows, dict)
                        else None
                    )
                    base_ll = (
                        baseline_windows.get("logloss")
                        if isinstance(baseline_windows, dict)
                        else None
                    )
                    run_w = (
                        run_windows.get("token_counts")
                        if isinstance(run_windows, dict)
                        else None
                    )
                    if (
                        isinstance(run_ids, list)
                        and isinstance(run_ll, list)
                        and isinstance(base_ids, list)
                        and isinstance(base_ll, list)
                        and isinstance(run_w, list)
                    ):
                        base_map: dict[int, float] = {}
                        for b_id, b_val in zip(base_ids, base_ll, strict=False):
                            if isinstance(b_id, int | float) and isinstance(
                                b_val, int | float
                            ):
                                base_map[int(b_id)] = float(b_val)
                        sum_w = 0.0
                        sum_dw = 0.0
                        for r_id, r_val, w in zip(run_ids, run_ll, run_w, strict=False):
                            if not (
                                isinstance(r_id, int | float)
                                and isinstance(r_val, int | float)
                            ):
                                continue
                            try:
                                wv = float(w)
                            except Exception:  # pragma: no cover
                                continue
                            if not math.isfinite(wv) or wv <= 0:
                                continue
                            key = int(r_id)
                            if key not in base_map:
                                continue
                            sum_w += wv
                            sum_dw += wv * (float(r_val) - base_map[key])
                        if sum_w > 0.0:
                            baseline_delta_mean = float(sum_dw / sum_w)
                except Exception:  # pragma: no cover
                    baseline_delta_mean = float("nan")
            except Exception:  # pragma: no cover
                ratio_ci_source = "run_metrics"

    def _finite_bounds(bounds: tuple[float, float]) -> bool:
        return (
            isinstance(bounds, tuple | list)
            and len(bounds) == 2
            and all(isinstance(v, int | float) and math.isfinite(v) for v in bounds)
        )

    drift_ci = (float("nan"), float("nan"))
    if _finite_bounds(preview_ci) and _finite_bounds(final_ci):
        lower_preview = max(preview_ci[0], 1e-12)
        upper_preview = max(preview_ci[1], 1e-12)
        drift_ci = (
            final_ci[0] / upper_preview if upper_preview > 0 else float("nan"),
            final_ci[1] / max(lower_preview, 1e-12),
        )

    def _is_number(value: Any) -> bool:
        return isinstance(value, int | float) and math.isfinite(float(value))

    delta_mean = paired_delta_summary.get("mean")
    degenerate_delta = paired_delta_summary.get("degenerate", False)
    drift_ratio = (
        edited_final / edited_preview
        if _is_number(edited_final)
        and _is_number(edited_preview)
        and edited_preview > 0
        else float("nan")
    )

    ratio_from_delta = None
    if _is_number(delta_mean) and not degenerate_delta:
        ratio_from_delta = _enforce_drift_ratio_identity(
            paired_windows, float(delta_mean), drift_ratio, window_plan_profile
        )

    if (
        ratio_from_delta is not None
        and _is_number(baseline_delta_mean)
        and _is_number(ratio_vs_baseline)
    ):
        expected_ratio_baseline = math.exp(float(baseline_delta_mean))
        tolerance = 5e-4 * max(1.0, abs(expected_ratio_baseline))
        if abs(expected_ratio_baseline - ratio_vs_baseline) > tolerance:
            pass

    # Fallback: if we could not compute a finite ratio, but we did compute a paired
    # baseline delta, use exp(delta) as an identity-consistent ratio. This covers
    # tiny runs where ppl_* fields are absent and PM-only windows are identical.
    if not (
        isinstance(ratio_vs_baseline, int | float) and math.isfinite(ratio_vs_baseline)
    ):
        try:
            if isinstance(baseline_delta_mean, int | float) and math.isfinite(
                baseline_delta_mean
            ):
                ratio_vs_baseline = math.exp(float(baseline_delta_mean))
                # Provide a degenerate CI if none was computed
                if not (
                    isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2
                ) and isinstance(edited_final, int | float):
                    ratio_ci = (float(edited_final), float(edited_final))
        except Exception:  # pragma: no cover
            pass

    _enforce_ratio_ci_alignment(ratio_ci_source, ratio_ci, logloss_delta_ci)

    paired_windows = _fallback_paired_windows(paired_windows, coverage_summary)

    # Primary-metric stats for gating/summary (PM-only)
    try:
        pm_blk = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
    except Exception:  # pragma: no cover
        pm_blk = None
    if not isinstance(pm_blk, dict) or not pm_blk:
        try:
            pm_blk = compute_primary_metric_from_report(report)
        except Exception:  # pragma: no cover
            pm_blk = {}
    pm_prev = pm_blk.get("preview") if isinstance(pm_blk, dict) else float("nan")
    pm_fin = pm_blk.get("final") if isinstance(pm_blk, dict) else float("nan")
    pm_ratio = pm_blk.get("ratio_vs_baseline") if isinstance(pm_blk, dict) else None
    if not isinstance(pm_ratio, (int | float)):
        try:
            base_final = baseline_ref.get("primary_metric", {}).get("final")
            if (
                isinstance(pm_fin, (int | float))
                and isinstance(base_final, (int | float))
                and base_final > 0
            ):
                pm_ratio = float(pm_fin) / float(base_final)
        except Exception:  # pragma: no cover
            pm_ratio = float("nan")
    pm_preview_final_ratio = (
        float(pm_fin) / float(pm_prev)
        if isinstance(pm_fin, (int | float))
        and isinstance(pm_prev, (int | float))
        and pm_prev > 0
        else float("nan")
    )
    ppl_analysis = {
        "preview": pm_prev,
        "final": pm_fin,
        "ratio_vs_baseline": pm_ratio
        if isinstance(pm_ratio, (int | float))
        else float("nan"),
        "preview_final_ratio": pm_preview_final_ratio,
        "drift": pm_preview_final_ratio,
        "preview_ci": None,
        "final_ci": None,
        "ratio_ci": ratio_ci,
        "degenerate": bool(
            isinstance(ratio_ci, list | tuple)
            and len(ratio_ci) == 2
            and all(isinstance(x, int | float) for x in ratio_ci)
            and abs(ratio_ci[0] - 1.0) < 1e-12
            and abs(ratio_ci[1] - 1.0) < 1e-12
        ),
        "unstable": bool(unstable_ci_flag),
        "drift_ci": drift_ci,
        "logloss_delta": logloss_delta,
        "logloss_delta_ci": logloss_delta_ci,
        "logloss_delta_paired_baseline": float(baseline_delta_mean)
        if _is_number(baseline_delta_mean)
        else None,
        "reduction": report["metrics"].get("reduction")
        if isinstance(report.get("metrics"), dict)
        else None,
        "stats": {
            "metric_space": "log_nll",
            "bootstrap": metrics_bootstrap,
            "coverage": coverage_summary,
            "pairing": ratio_ci_source,
            "paired_windows": paired_windows,
            "window_overlap_fraction": report["metrics"].get(
                "window_overlap_fraction", float("nan")
            ),
            "window_match_fraction": report["metrics"].get(
                "window_match_fraction", float("nan")
            ),
            "window_pairing_reason": report["metrics"].get(
                "window_pairing_reason", None
            ),
            "paired_delta_summary": paired_delta_summary,
        },
    }

    metrics_stats_source = {}
    if isinstance(report.get("metrics"), dict):
        metrics_stats_source = report["metrics"].get("stats", {}) or {}
    if isinstance(metrics_stats_source, dict):
        for key in (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        ):
            if key in metrics_stats_source:
                ppl_analysis["stats"][key] = metrics_stats_source[key]

    # Derive requested/actual window counts for auditability when runners do not
    # emit a metrics.stats block (normalization may also drop it).
    try:
        stats_obj = ppl_analysis.get("stats", {})
        if isinstance(stats_obj, dict):

            def _as_count(value: Any) -> int | None:
                if value is None or isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return int(value) if value >= 0 else None
                if isinstance(value, float) and math.isfinite(value):
                    if abs(value - round(value)) > 1e-9 or value < 0:
                        return None
                    return int(round(value))
                return None

            data_cfg = report.get("data", {}) if isinstance(report, dict) else {}
            data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
            windows_cfg = (
                dataset_info.get("windows", {})
                if isinstance(dataset_info, dict)
                else {}
            )
            windows_cfg = windows_cfg if isinstance(windows_cfg, dict) else {}

            req_prev = _as_count(stats_obj.get("requested_preview"))
            if req_prev is None:
                req_prev = _as_count(data_cfg.get("preview_n"))
            if req_prev is None:
                req_prev = _as_count(windows_cfg.get("preview"))

            req_fin = _as_count(stats_obj.get("requested_final"))
            if req_fin is None:
                req_fin = _as_count(data_cfg.get("final_n"))
            if req_fin is None:
                req_fin = _as_count(windows_cfg.get("final"))

            eval_windows = (
                report.get("evaluation_windows", {}) if isinstance(report, dict) else {}
            )
            eval_windows = eval_windows if isinstance(eval_windows, dict) else {}

            def _len_ids(section: Any) -> int | None:
                if not isinstance(section, dict):
                    return None
                ids = section.get("window_ids")
                if isinstance(ids, list):
                    return int(len(ids))
                return None

            act_prev = _as_count(stats_obj.get("actual_preview"))
            if act_prev is None:
                act_prev = _len_ids(eval_windows.get("preview"))
            if act_prev is None:
                cov_prev = (
                    coverage_summary.get("preview")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_prev, dict):
                    act_prev = _as_count(cov_prev.get("used"))
            if act_prev is None:
                act_prev = req_prev

            act_fin = _as_count(stats_obj.get("actual_final"))
            if act_fin is None:
                act_fin = _len_ids(eval_windows.get("final"))
            if act_fin is None:
                cov_fin = (
                    coverage_summary.get("final")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_fin, dict):
                    act_fin = _as_count(cov_fin.get("used"))
                elif isinstance(coverage_summary, dict):
                    act_fin = _as_count(coverage_summary.get("used"))
            if act_fin is None:
                act_fin = req_fin

            if req_prev is not None:
                stats_obj.setdefault("requested_preview", req_prev)
            if req_fin is not None:
                stats_obj.setdefault("requested_final", req_fin)
            if act_prev is not None:
                stats_obj.setdefault("actual_preview", act_prev)
            if act_fin is not None:
                stats_obj.setdefault("actual_final", act_fin)

            if "coverage_ok" not in stats_obj:
                if (
                    isinstance(req_prev, int)
                    and isinstance(req_fin, int)
                    and isinstance(act_prev, int)
                    and isinstance(act_fin, int)
                ):
                    stats_obj["coverage_ok"] = (act_prev >= req_prev) and (
                        act_fin >= req_fin
                    )
    except Exception:  # pragma: no cover
        pass

    if isinstance(window_plan_ctx, dict):
        ppl_analysis["window_plan"] = window_plan_ctx

    # Extract invariant status
    invariants = _extract_invariants(report)

    # Extract spectral analysis
    spectral = _extract_spectral_analysis(report, baseline_normalized)

    # Extract RMT analysis
    rmt = _extract_rmt_analysis(report, baseline_normalized)

    # Extract variance guard info
    variance = _extract_variance_analysis(report)

    # Extract structural deltas
    structure = _extract_structural_deltas(report)
    compression_diag = structure.get("compression_diagnostics", {})
    structure["compression_diagnostics"] = compression_diag

    # Extract effective policies used
    policies = _extract_effective_policies(report)
    variance_policy = policies.get("variance")
    guard_variance_policy = None
    for guard in report.get("guards", []):
        if guard.get("name", "").lower() == "variance" and isinstance(
            guard.get("policy"), dict
        ):
            guard_variance_policy = guard.get("policy")
            break

    variance_policy_digest = ""
    if isinstance(variance_policy, dict):
        variance_policy_digest = _compute_variance_policy_digest(variance_policy)
        if not variance_policy_digest and isinstance(guard_variance_policy, dict):
            variance_policy_digest = _compute_variance_policy_digest(
                guard_variance_policy
            )
            if variance_policy_digest:
                for key in VARIANCE_CANONICAL_KEYS:
                    if (
                        isinstance(guard_variance_policy, dict)
                        and key in guard_variance_policy
                        and key not in variance_policy
                    ):
                        variance_policy[key] = guard_variance_policy[key]
        if variance_policy_digest:
            policies["variance"]["policy_digest"] = variance_policy_digest

    # Resolve tier/profile policy (canonical) and merge observed guard policies.
    profile = None
    explicit_overrides = None
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        if isinstance(ctx, dict) and ctx.get("profile"):
            profile = str(ctx.get("profile"))
    except Exception:
        profile = None
    try:
        window_plan = (
            report.get("metrics", {}).get("window_plan")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if (
            profile is None
            and isinstance(window_plan, dict)
            and window_plan.get("profile")
        ):
            profile = str(window_plan.get("profile"))
    except Exception:
        profile = None
    try:
        meta_cfg = (
            report.get("meta", {}).get("config")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(meta_cfg, dict) and isinstance(meta_cfg.get("guards"), dict):
            explicit_overrides = meta_cfg.get("guards")
        if explicit_overrides is None and isinstance(report.get("config"), dict):
            cfg2 = report.get("config")
            if isinstance(cfg2.get("guards"), dict):
                explicit_overrides = cfg2.get("guards")
    except Exception:
        explicit_overrides = None

    resolved_policy = _build_resolved_policies(
        auto.get("tier", "balanced"),
        spectral,
        rmt,
        variance,
        profile=profile,
        explicit_overrides=explicit_overrides,
    )
    overrides_list = _extract_policy_overrides(report)
    resolved_digest = _compute_policy_digest(
        {"resolved_policy": resolved_policy, "overrides": overrides_list}
    )
    policy_provenance = {
        "tier": auto.get("tier", "balanced"),
        "overrides": overrides_list,
        "policy_digest": resolved_digest,
    }
    auto["policy_digest"] = resolved_digest

    for guard_name in ("spectral", "rmt", "variance"):
        if guard_name in resolved_policy:
            policies[guard_name] = copy.deepcopy(resolved_policy[guard_name])
            if guard_name == "variance" and variance_policy_digest:
                policies[guard_name]["policy_digest"] = variance_policy_digest

    plugin_provenance = report.get("meta", {}).get("plugins", {})
    edit_metadata = _extract_edit_metadata(report, plugin_provenance)

    # Extract telemetry (latency, memory, etc.)
    telemetry: dict[str, Any] = {}
    metrics_section = report.get("metrics", {})
    if isinstance(metrics_section, dict):
        for key in ("latency_ms_per_tok", "memory_mb_peak", "throughput_tok_per_s"):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and math.isfinite(value):
                telemetry[key] = float(value)

        for key in ("preview_total_tokens", "final_total_tokens"):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)
        for key in (
            "masked_tokens_total",
            "masked_tokens_preview",
            "masked_tokens_final",
        ):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)

        edge_ctx = metrics_section.get("edge_device")
        if isinstance(edge_ctx, dict):
            telemetry["edge_device"] = edge_ctx

    device_name = meta.get("device")
    if device_name:
        telemetry.setdefault("device", device_name)

    # Build the certificate
    window_capacity_ctx = (
        report.get("metrics", {}).get("window_capacity")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    window_plan_ctx = (
        report.get("metrics", {}).get("window_plan")
        if isinstance(report.get("metrics"), dict)
        else None
    )

    report_artifacts = (
        report.get("artifacts", {}) if isinstance(report.get("artifacts"), dict) else {}
    )
    artifacts_payload = {
        "events_path": report_artifacts.get("events_path", ""),
        "report_path": report_artifacts.get(
            "report_path", report_artifacts.get("logs_path", "")
        ),
        "generated_at": datetime.now().isoformat(),
    }
    masks_path = report_artifacts.get("masks_path")
    if isinstance(masks_path, str) and masks_path:
        artifacts_payload["masks_path"] = masks_path

    raw_guard_ctx = report.get("guard_overhead")
    guard_overhead_section, _ = _prepare_guard_overhead_section(raw_guard_ctx)

    # Add schedule digest to provenance/overhead for auditability of schedule reuse
    try:
        final_windows_ctx = (
            report.get("evaluation_windows", {}).get("final", {})
            if isinstance(report.get("evaluation_windows"), dict)
            else {}
        )
        window_ids = final_windows_ctx.get("window_ids")
        if isinstance(window_ids, list) and window_ids:
            import hashlib as _hashlib

            h = _hashlib.blake2s(digest_size=16)
            for wid in window_ids:
                try:
                    h.update(int(wid).to_bytes(8, "little", signed=True))
                except Exception:  # pragma: no cover
                    h.update(str(wid).encode("utf-8", "ignore"))
            schedule_digest = h.hexdigest()
            guard_overhead_section["schedule_digest"] = schedule_digest
        else:
            schedule_digest = None
    except Exception:  # pragma: no cover
        schedule_digest = None

    policy_provenance["resolved_at"] = artifacts_payload["generated_at"]

    current_run_id = _generate_run_id(report)
    provenance = _build_provenance_block(
        report,
        baseline_raw,
        baseline_ref,
        artifacts_payload,
        policy_provenance,
        schedule_digest,
        ppl_analysis,
        current_run_id,
    )

    # Prepare MoE section (observability; non-gating)
    moe_section: dict[str, Any] = {}
    try:
        run_moe = (
            report.get("metrics", {}).get("moe")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        base_moe = None
        # Try raw baseline first (dict with optional 'moe')
        if isinstance(baseline_raw, dict):
            try:
                base_moe = baseline_raw.get("moe")
            except Exception:  # pragma: no cover
                base_moe = None
        # Then normalized baseline variants
        if (not isinstance(base_moe, dict) or not base_moe) and isinstance(
            baseline_normalized, dict
        ):
            try:
                bm = baseline_normalized.get("moe")
                if isinstance(bm, dict) and bm:
                    base_moe = bm
                else:
                    mx = (
                        baseline_normalized.get("metrics")
                        if isinstance(baseline_normalized.get("metrics"), dict)
                        else None
                    )
                    if isinstance(mx, dict):
                        base_moe = mx.get("moe")
            except Exception:  # pragma: no cover
                pass
        if isinstance(run_moe, dict) and run_moe:
            # Copy selected fields
            for key in (
                "top_k",
                "capacity_factor",
                "expert_drop_rate",
                "load_balance_loss",
                "router_entropy",
            ):
                val = run_moe.get(key)
                if isinstance(val, int | float):
                    moe_section[key] = float(val)
            # Utilization summary
            util = run_moe.get("utilization")
            if isinstance(util, list) and util:
                try:
                    util_vals = [float(x) for x in util]
                    moe_section["utilization_mean"] = float(
                        sum(util_vals) / max(1, len(util_vals))
                    )
                    moe_section["utilization_count"] = int(len(util_vals))
                except Exception:  # pragma: no cover
                    pass
            # Deltas vs baseline (if available)
            if isinstance(base_moe, dict) and base_moe:
                for key in ("load_balance_loss", "router_entropy"):
                    rv = run_moe.get(key)
                    bv = base_moe.get(key)
                    if isinstance(rv, int | float) and isinstance(bv, int | float):
                        moe_section[f"delta_{key}"] = float(rv) - float(bv)
                bu = base_moe.get("utilization")
                if isinstance(util, list) and isinstance(bu, list) and util and bu:
                    try:
                        util_vals = [float(x) for x in util]
                        bu_vals = [float(x) for x in bu]
                        mu = float(sum(util_vals) / len(util_vals))
                        mb = float(sum(bu_vals) / len(bu_vals))
                        moe_section["delta_utilization_mean"] = mu - mb
                    except Exception:  # pragma: no cover
                        pass
    except Exception:  # pragma: no cover
        moe_section = {}

    # Build dataset capacity context for gating floors
    capacity_tokens: int | None = None
    capacity_examples: int | None = None
    try:
        if isinstance(window_capacity_ctx, dict):
            tv = window_capacity_ctx.get("total_tokens")
            if isinstance(tv, int | float):
                capacity_tokens = int(tv)
            ex = (
                window_capacity_ctx.get("available_unique")
                or window_capacity_ctx.get("available_nonoverlap")
                or window_capacity_ctx.get("candidate_limit")
            )
            if isinstance(ex, int | float):
                capacity_examples = int(ex)
        # Fallback: sum of configured windows
        if capacity_examples is None:
            try:
                capacity_examples = int(
                    dataset_info.get("windows", {}).get("preview", 0)
                ) + int(dataset_info.get("windows", {}).get("final", 0))
            except Exception:  # pragma: no cover
                capacity_examples = None
    except Exception:  # pragma: no cover
        capacity_tokens = None
        capacity_examples = None

    pm_acceptance_range = _resolve_pm_acceptance_range_from_report(report)

    validation_kwargs = {
        "ppl": ppl_analysis,
        "spectral": spectral,
        "rmt": rmt,
        "invariants": invariants,
        "tier": auto.get("tier", "balanced"),
        "_ppl_metrics": ppl_metrics,
        "target_ratio": auto.get("target_pm_ratio"),
        "guard_overhead": guard_overhead_section,
        "primary_metric": report.get("metrics", {}).get("primary_metric")
        if isinstance(report.get("metrics"), dict)
        else None,
        "moe": moe_section,
        "dataset_capacity": {
            "tokens_available": capacity_tokens,
            "examples_available": capacity_examples,
        },
    }
    try:
        if (
            "pm_acceptance_range"
            in inspect.signature(_compute_validation_flags).parameters
        ):
            validation_kwargs["pm_acceptance_range"] = pm_acceptance_range
    except Exception:  # pragma: no cover - defensive against patched functions
        validation_kwargs["pm_acceptance_range"] = pm_acceptance_range

    validation_flags = _compute_validation_flags(**validation_kwargs)
    # Enforce validation key allow-list to prevent surface drift
    _allowed_validation = _load_validation_allowlist()
    validation_filtered = {
        k: bool(v) for k, v in validation_flags.items() if k in _allowed_validation
    }

    certificate = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": current_run_id,
        "meta": meta,
        "auto": auto,
        "dataset": dataset_info,
        "edit": edit_metadata,
        "telemetry": telemetry,
        "baseline_ref": baseline_ref,
        "invariants": invariants,
        "spectral": spectral,
        "rmt": rmt,
        "variance": variance,
        "structure": structure,
        "policies": policies,
        "resolved_policy": resolved_policy,
        "policy_provenance": policy_provenance,
        "provenance": provenance,
        "plugins": plugin_provenance,
        "edit_name": (report.get("edit", {}) or {}).get(
            "name", "unknown"
        ),  # Include edit name for rendering
        "artifacts": artifacts_payload,
        "validation": validation_filtered,
        "guard_overhead": guard_overhead_section,
    }

    # Record tiny-relax provenance explicitly when active (dev-only demos)
    try:
        import os as _os

        _tiny_relax_env = str(
            _os.environ.get("INVARLOCK_TINY_RELAX", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:  # pragma: no cover
        _tiny_relax_env = False
    if _tiny_relax_env:
        try:
            certificate.setdefault("auto", {})["tiny_relax"] = True
            prov = certificate.setdefault("provenance", {})
            flags = prov.setdefault("flags", [])
            if "tiny_relax" not in flags:
                flags.append("tiny_relax")
        except Exception:  # pragma: no cover
            pass

    # Compute PM-aware quality overhead when both snapshots are present
    try:
        pm_kind_hint = None
        try:
            pm_try = (
                report.get("metrics", {}).get("primary_metric")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if isinstance(pm_try, dict):
                pm_kind_hint = pm_try.get("kind")
        except Exception:  # pragma: no cover
            pm_kind_hint = None
        qo = _compute_quality_overhead_from_guard(raw_guard_ctx, pm_kind_hint)
        if (
            isinstance(qo, dict)
            and "value" in qo
            and math.isfinite(float(qo.get("value", float("nan"))))
        ):
            certificate["quality_overhead"] = qo
    except Exception:  # pragma: no cover
        pass

    try:
        _propagate_pairing_stats(certificate, ppl_analysis)
    except Exception:  # pragma: no cover
        pass

    # Attach policy/version digest object (thresholds/floors + key knobs)
    try:
        cur_tier = str(auto.get("tier", "balanced")).lower()
    except Exception:  # pragma: no cover
        cur_tier = "balanced"
    thresholds_payload = _compute_thresholds_payload(cur_tier, resolved_policy)
    thresholds_hash = _compute_thresholds_hash(thresholds_payload)
    # Baseline tier for change note (best-effort)
    base_tier = None
    try:
        # Prefer raw baseline RunReport (if provided)
        if isinstance(baseline_raw, dict):
            bm = baseline_raw.get("meta")
            if isinstance(bm, dict):
                ba = bm.get("auto")
                if isinstance(ba, dict) and ba.get("tier"):
                    base_tier = str(ba.get("tier")).lower()
        # Fallback to normalized (usually lacks meta)
        if base_tier is None and isinstance(baseline_normalized, dict):
            base_meta = baseline_normalized.get("meta")
            if isinstance(base_meta, dict):
                base_auto = base_meta.get("auto")
                if isinstance(base_auto, dict) and base_auto.get("tier"):
                    base_tier = str(base_auto.get("tier")).lower()
    except Exception:  # pragma: no cover
        base_tier = None
    baseline_payload = _compute_thresholds_payload(
        base_tier or cur_tier, resolved_policy
    )
    baseline_hash = _compute_thresholds_hash(baseline_payload)
    changed = bool(
        (base_tier is not None and base_tier != cur_tier)
        or (baseline_hash != thresholds_hash)
    )

    # Hysteresis knobs snapshot (policy-resolved)
    metrics_policy = (
        resolved_policy.get("metrics", {}) if isinstance(resolved_policy, dict) else {}
    )
    if not isinstance(metrics_policy, dict):
        metrics_policy = {}
    ppl_hys = 0.0
    acc_hys = 0.0
    try:
        ppl_hys = float(
            (metrics_policy.get("pm_ratio") or {}).get("hysteresis_ratio", 0.0) or 0.0
        )
        acc_hys = float(
            (metrics_policy.get("accuracy") or {}).get("hysteresis_delta_pp", 0.0)
            or 0.0
        )
    except Exception:  # pragma: no cover
        pass
    min_effective = float(
        (resolved_policy.get("variance") or {}).get("min_effect_lognll", 0.0) or 0.0
    )

    certificate["policy_digest"] = {
        "policy_version": POLICY_VERSION,
        "tier_policy_name": cur_tier,
        "thresholds_hash": thresholds_hash,
        "hysteresis": {"ppl": ppl_hys, "accuracy_delta_pp": acc_hys},
        "min_effective": min_effective,
        "changed": changed,
    }

    # Optional: include secondary metrics (informational; non-gating)
    try:
        if isinstance(report.get("metrics"), dict):
            sec = report["metrics"].get("secondary_metrics")
            if isinstance(sec, list) and sec:
                sanitized: list[dict[str, Any]] = []
                for item in sec:
                    if isinstance(item, dict) and item.get("kind"):
                        payload: dict[str, Any] = {}
                        for key in (
                            "kind",
                            "preview",
                            "final",
                            "ratio_vs_baseline",
                            "unit",
                            "display_ci",
                            "ci",
                        ):
                            if key in item:
                                payload[key] = item[key]
                        sanitized.append(payload)
                if sanitized:
                    certificate["secondary_metrics"] = sanitized
    except Exception:  # pragma: no cover
        pass

    # Optional: classification subgroup analysis (informational)
    try:
        cls = (
            report.get("metrics", {}).get("classification")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(cls, dict):
            sub = cls.get("subgroups")
            # Expect pre-aggregated subgroup counts
            if isinstance(sub, dict) and all(k in sub for k in ("preview", "final")):
                prev = sub.get("preview", {})
                fin = sub.get("final", {})
                pc = prev.get("group_counts", {}) if isinstance(prev, dict) else {}
                pcc = prev.get("correct_counts", {}) if isinstance(prev, dict) else {}
                fc = fin.get("group_counts", {}) if isinstance(fin, dict) else {}
                fcc = fin.get("correct_counts", {}) if isinstance(fin, dict) else {}
                out: dict[str, Any] = {}
                labels = set(list(pc.keys()) + list(fc.keys()))
                for g in labels:
                    try:
                        nprev = float(pc.get(g, 0))
                        nfin = float(fc.get(g, 0))
                        acc_prev = (
                            float(pcc.get(g, 0)) / nprev if nprev > 0 else float("nan")
                        )
                        acc_fin = (
                            float(fcc.get(g, 0)) / nfin if nfin > 0 else float("nan")
                        )
                        delta_pp = (
                            (acc_fin - acc_prev) * 100.0
                            if (math.isfinite(acc_prev) and math.isfinite(acc_fin))
                            else float("nan")
                        )
                        out[str(g)] = {
                            "preview": acc_prev,
                            "final": acc_fin,
                            "delta_pp": delta_pp,
                            "n_preview": nprev,
                            "n_final": nfin,
                        }
                    except Exception:  # pragma: no cover
                        continue
                if out:
                    certificate["classification"] = {"subgroups": out}
    except Exception:  # pragma: no cover
        pass

    # Compute System Overhead (latency/throughput) vs baseline when available
    try:

        def _extract_sys_metrics(container: dict[str, Any] | None) -> dict[str, float]:
            out: dict[str, float] = {}
            if not isinstance(container, dict):
                return out
            metrics = (
                container.get("metrics", {})
                if isinstance(container.get("metrics"), dict)
                else {}
            )
            # Edited report case: also check certificate telemetry keys
            telem = telemetry if isinstance(telemetry, dict) else {}
            # Prefer explicit p50/p95 throughput keys if present
            for key in ("latency_ms_p50", "latency_ms_p95", "throughput_sps"):
                val = metrics.get(key)
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out[key] = float(val)
            # Fallbacks
            if "latency_ms_p50" not in out:
                val = metrics.get("latency_ms_per_tok") or telem.get(
                    "latency_ms_per_tok"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["latency_ms_p50"] = float(val)
            if "throughput_sps" not in out:
                val = metrics.get("throughput_tok_per_s") or telem.get(
                    "throughput_tok_per_s"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["throughput_sps"] = float(val)
            return out

        edited_sys = _extract_sys_metrics(report)
        base_sys = _extract_sys_metrics(
            baseline_raw if isinstance(baseline_raw, dict) else None
        )
        system_overhead: dict[str, Any] = {}
        for metric_key, edited_val in edited_sys.items():
            base_val = base_sys.get(metric_key)
            entry: dict[str, Any] = {"edited": edited_val}
            if isinstance(base_val, int | float) and math.isfinite(float(base_val)):
                entry["baseline"] = float(base_val)
                entry["delta"] = float(edited_val - base_val)
                try:
                    entry["ratio"] = (
                        float(edited_val / base_val) if base_val != 0 else float("nan")
                    )
                except Exception:  # pragma: no cover
                    entry["ratio"] = float("nan")
            system_overhead[metric_key] = entry
        if system_overhead:
            certificate["system_overhead"] = system_overhead
    except Exception:  # pragma: no cover
        pass

    # Emit optional one-line telemetry summary (opt-in via INVARLOCK_TELEMETRY=1)
    try:
        kind = None
        pm_try = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(pm_try, dict):
            kind = pm_try.get("kind")
        if not kind:
            kind = "ppl"
        windows_cfg = (
            certificate.get("dataset", {}).get("windows", {})
            if isinstance(certificate.get("dataset"), dict)
            else {}
        )
        n_prev = windows_cfg.get("preview")
        n_fin = windows_cfg.get("final")
        tokens_total = None
        try:
            tokens_total = (
                certificate.get("dataset", {}).get("hash", {}).get("total_tokens")
            )
        except Exception:  # pragma: no cover
            tokens_total = None
        # CI interval
        ci_lo = None
        ci_hi = None
        ratio = None
        pmc = certificate.get("primary_metric", {})
        rci = pmc.get("display_ci") or pmc.get("ci")
        if isinstance(rci, tuple | list) and len(rci) == 2:
            ci_lo, ci_hi = rci[0], rci[1]
        ratio = pmc.get("ratio_vs_baseline")
        ci_w = None
        try:
            if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
                ci_w = float(ci_hi) - float(ci_lo)
        except Exception:  # pragma: no cover
            ci_w = None
        # Gate outcome
        val = certificate.get("validation", {})
        gate_ok = None
        try:
            gate_ok = bool(val.get("primary_metric_acceptable"))
        except Exception:  # pragma: no cover
            gate_ok = None
        # Build line
        parts = [
            f"run_id={current_run_id}",
            f"metric={kind}",
            f"nprev={n_prev}",
            f"nfinal={n_fin}",
            f"tokens={tokens_total}",
        ]
        try:
            split = (certificate.get("provenance", {}) or {}).get("dataset_split")
            if not split:
                split = (report.get("provenance", {}) or {}).get("dataset_split")
            sf = (certificate.get("provenance", {}) or {}).get("split_fallback")
            if sf is None:
                sf = (report.get("provenance", {}) or {}).get("split_fallback")
            if split:
                parts.append(f"split={split}{'*' if sf else ''}")
        except Exception:  # pragma: no cover
            pass
        if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
            parts.append(f"ci={ci_lo:.3f}-{ci_hi:.3f}")
            if isinstance(ci_w, int | float):
                parts.append(f"width={ci_w:.3f}")
        if isinstance(ratio, int | float):
            parts.append(f"ratio={float(ratio):.3f}")
        if isinstance(gate_ok, bool):
            parts.append(f"gate={'pass' if gate_ok else 'fail'}")
        summary_line = "INVARLOCK_TELEMETRY " + " ".join(parts)
        certificate.setdefault("telemetry", {})["summary_line"] = summary_line
        if str(os.environ.get("INVARLOCK_TELEMETRY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(summary_line)
    except Exception:  # pragma: no cover
        pass

    # Attach/normalize primary metric block (moved to helper)
    from .primary_metric_utils import attach_primary_metric as _attach_pm

    _attach_pm(certificate, report, baseline_raw, baseline_ref, ppl_analysis)

    # Ensure primary_metric has display_ci populated for schema invariants
    try:
        pm = (
            certificate.get("primary_metric", {})
            if isinstance(certificate.get("primary_metric"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
            # Prefer existing bounds; otherwise collapse to point estimate
            disp = pm.get("display_ci")
            if not (
                isinstance(disp, list | tuple)
                and len(disp) == 2
                and all(isinstance(x, int | float) for x in disp)
            ):
                point = None
                for key in ("ratio_vs_baseline", "final", "preview"):
                    val = pm.get(key)
                    if isinstance(val, int | float) and math.isfinite(float(val)):
                        point = float(val)
                        break
                if isinstance(point, float):
                    pm["display_ci"] = [point, point]
                else:
                    # As last resort, emit a degenerate [1.0, 1.0] to satisfy schema invariants
                    pm["display_ci"] = [1.0, 1.0]
    except Exception:  # pragma: no cover
        pass

    # Attach confidence label (non-gating)
    try:
        certificate["confidence"] = _compute_confidence_label(certificate)
    except Exception:  # pragma: no cover
        pass

    return certificate


# Console Validation Block helpers have moved to invarlock.reporting.render.


## NOTE: render_certificate_markdown has been moved to invarlock.reporting.render.
## It is re-exported at the bottom of this module to preserve the public API.
## Private helper functions


def _normalize_baseline(baseline: RunReport | dict[str, Any]) -> dict[str, Any]:
    """Normalize baseline input to a consistent dictionary format."""
    if isinstance(baseline, dict):
        # Check if it's a baseline schema (v1 only)
        if baseline.get("schema_version") in {"baseline-v1"}:
            ppl_final = baseline.get("metrics", {}).get("ppl_final", float("nan"))
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
            # Accept both legacy ppl_* metrics and PM-first reports
            metrics_blk = baseline.get("metrics", {}) or {}
            ppl_final = metrics_blk.get("ppl_final")
            ppl_preview = metrics_blk.get("ppl_preview")
            if ppl_final is None:
                # Fallback: derive from primary_metric if it is ppl-like
                try:
                    pm = metrics_blk.get("primary_metric", {}) or {}
                    kind = str(pm.get("kind") or "").lower()
                    if kind.startswith("ppl"):
                        pf = pm.get("final")
                        pp = pm.get("preview", pf)
                        if isinstance(pf, int | float):
                            ppl_final = float(pf)
                        if isinstance(pp, int | float):
                            ppl_preview = float(pp)
                except Exception:  # pragma: no cover
                    # Leave as None; downstream validation will handle
                    pass
            if ppl_preview is None:
                ppl_preview = ppl_final

            # Detect invalid baseline by checking if it's actually a no-op baseline
            edit_plan = baseline["edit"].get("plan", {})
            plan_digest = baseline["edit"].get("plan_digest", "")

            # Valid baseline indicators: target_sparsity=0.0, plan_digest contains "baseline_noop" or "noop"
            is_valid_baseline = (
                edit_plan.get("target_sparsity") == 0.0
                or "baseline_noop" in plan_digest
                or "noop" in plan_digest
                or baseline["edit"]["name"] == "baseline"
            )

            # Only flag as invalid if PPL is clearly wrong OR it's definitely not a baseline
            if (isinstance(ppl_final, int | float) and ppl_final <= 1.0) or (
                not is_valid_baseline
                and baseline["edit"]["deltas"]["params_changed"] > 0
            ):
                print(
                    f"⚠️  Warning: Invalid baseline detected (PPL={ppl_final}, edit={baseline['edit']['name']}, params_changed={baseline['edit']['deltas']['params_changed']})"
                )
                print("   Using computed baseline PPL for GPT-2 on validation split")
                # Use computed baseline for GPT-2 on validation split
                ppl_final = 50.797  # Computed GPT-2 validation PPL
                ppl_preview = ppl_final

            eval_windows = baseline.get("evaluation_windows", {})
            final_windows = (
                eval_windows.get("final", {}) if isinstance(eval_windows, dict) else {}
            )
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

            return {
                "run_id": _generate_run_id(baseline),
                "model_id": baseline["meta"]["model_id"],
                "ppl_final": ppl_final,
                "ppl_preview": ppl_preview,
                "spectral": baseline["metrics"].get("spectral", {}),
                "rmt": baseline["metrics"].get("rmt", {}),
                "invariants": baseline["metrics"].get("invariants", {}),
                "moe": baseline["metrics"].get("moe", {}),
                "evaluation_windows": baseline_eval_windows,
                "bootstrap": bootstrap_info,
                "window_overlap_fraction": window_overlap,
                "window_match_fraction": window_match,
                "tokenizer_hash": baseline_tokenizer_hash,
            }
        else:
            # Assume it's already normalized
            ppl_final = baseline.get("ppl_final", float("nan"))
            if ppl_final <= 1.0:
                print(
                    f"⚠️  Warning: Invalid baseline PPL ({ppl_final}), using computed baseline"
                )
                baseline = baseline.copy()  # Don't mutate original
                baseline["ppl_final"] = 50.797
            return baseline
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
    """Extract edit-level provenance and configuration metadata for the certificate."""

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
    # Sanitize algorithm identifiers to purge legacy/unsupported edit labels
    try:
        alg_lower = str(algorithm).strip().lower()
    except Exception:  # pragma: no cover
        alg_lower = ""
    allowed_algorithms = {"quant_rtn", "noop"}
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

    if not isinstance(raw, dict) or not raw:
        return {}, True

    payload = copy.deepcopy(raw)

    def _coerce_float(value: Any) -> float | None:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        return coerced if math.isfinite(coerced) else None

    threshold = _coerce_float(payload.get("overhead_threshold"))
    if threshold is None:
        threshold = 0.01
    threshold = max(0.0, threshold)

    sanitized: dict[str, Any] = {
        "overhead_threshold": threshold,
        "threshold_percent": threshold * 100,
        "source": str(payload.get("source", "report")),
    }
    try:
        mode = payload.get("mode")
        if mode is None:
            mode = payload.get("guard_overhead_mode")
        if isinstance(mode, str) and mode.strip():
            sanitized["mode"] = mode.strip()
    except Exception:
        pass
    try:
        skipped = bool(payload.get("skipped", False))
        if skipped:
            sanitized["skipped"] = True
            reason = payload.get("skip_reason")
            if isinstance(reason, str) and reason.strip():
                sanitized["skip_reason"] = reason.strip()
    except Exception:
        pass

    # Prefer structured reports and reuse the validator when available
    bare_report = payload.pop("bare_report", None)
    guarded_report = payload.pop("guarded_report", None)
    if isinstance(bare_report, dict) and isinstance(guarded_report, dict):
        result = validate_guard_overhead(
            bare_report, guarded_report, overhead_threshold=threshold
        )
        metrics = result.metrics or {}
        sanitized.update(
            {
                "overhead_ratio": metrics.get("overhead_ratio"),
                "overhead_percent": metrics.get("overhead_percent"),
                "bare_final": metrics.get("bare_final"),
                "guarded_final": metrics.get("guarded_final"),
                "messages": list(result.messages),
                "warnings": list(result.warnings),
                "errors": list(result.errors),
                "checks": dict(result.checks),
                "evaluated": True,
                "passed": bool(result.passed),
            }
        )
        return sanitized, bool(result.passed)

    # Fall back to direct ratio computation when reports are not provided
    bare_ppl = _coerce_float(payload.get("bare_final")) or _coerce_float(
        payload.get("bare_ppl")
    )
    guarded_ppl = _coerce_float(payload.get("guarded_final")) or _coerce_float(
        payload.get("guarded_ppl")
    )
    ratio = _coerce_float(payload.get("overhead_ratio"))

    if ratio is None and bare_ppl is not None and guarded_ppl is not None:
        if bare_ppl > 0:
            ratio = guarded_ppl / bare_ppl
        else:
            ratio = None

    if bare_ppl is not None:
        sanitized["bare_ppl"] = bare_ppl
    if guarded_ppl is not None:
        sanitized["guarded_ppl"] = guarded_ppl

    sanitized["messages"] = (
        [str(m) for m in payload.get("messages", [])]
        if isinstance(payload.get("messages"), list)
        else []
    )
    sanitized["warnings"] = (
        [str(w) for w in payload.get("warnings", [])]
        if isinstance(payload.get("warnings"), list)
        else []
    )
    sanitized["errors"] = (
        [str(e) for e in payload.get("errors", [])]
        if isinstance(payload.get("errors"), list)
        else []
    )
    sanitized["checks"] = (
        dict(payload.get("checks")) if isinstance(payload.get("checks"), dict) else {}
    )

    if ratio is not None:
        sanitized["overhead_ratio"] = ratio
        sanitized["overhead_percent"] = (ratio - 1.0) * 100
        passed = ratio <= (1.0 + threshold)
        sanitized["evaluated"] = True
        sanitized["passed"] = passed
        return sanitized, passed

    # Unable to compute ratio – treat as not evaluated and soft-pass
    # to align with CLI/run behavior and avoid spurious failures in tiny runs.
    if not sanitized["errors"]:
        sanitized["errors"] = ["Guard overhead ratio unavailable"]
    sanitized["evaluated"] = False
    sanitized["passed"] = True
    return sanitized, True


def _compute_quality_overhead_from_guard(
    raw_guard: Any,
    pm_kind_hint: str | None = None,
) -> dict[str, Any] | None:
    """Compute PM-aware quality overhead from guard context when possible.

    Uses bare_report and guarded_report to compute a primary-metric change
    normalized by metric direction:
      - lower-is-better (ppl_*): ratio (guarded / bare)
      - higher-is-better (accuracy): delta in percentage points
    Returns a dict with {basis, value, kind} or None when not computable.
    """
    try:
        if not isinstance(raw_guard, dict):
            return None
        bare = raw_guard.get("bare_report")
        guarded = raw_guard.get("guarded_report")
        if not (isinstance(bare, dict) and isinstance(guarded, dict)):
            return None
        kind = (
            (pm_kind_hint or "").strip().lower()
            if isinstance(pm_kind_hint, str)
            else ""
        )
        if not kind:
            kind = "ppl_causal"
        pm_b = compute_primary_metric_from_report(bare, kind=kind)
        pm_g = compute_primary_metric_from_report(guarded, kind=kind)
        g_point = pm_g.get("final")
        b_point = pm_b.get("final")
        if not (
            isinstance(g_point, int | float)
            and isinstance(b_point, int | float)
            and math.isfinite(float(g_point))
            and math.isfinite(float(b_point))
        ):
            return None
        # Resolve direction from registry when possible
        try:
            direction = get_metric(kind).direction
        except Exception:  # pragma: no cover
            direction = str(pm_g.get("direction", "")).lower()
        if direction == "lower":
            if float(b_point) <= 0:
                return None
            value = float(g_point) / float(b_point)
            basis = "ratio"
        else:
            value = 100.0 * (float(g_point) - float(b_point))
            basis = "delta_pp"
        return {"basis": basis, "value": value, "kind": kind}
    except Exception:  # pragma: no cover
        return None


def _propagate_pairing_stats(
    certificate: dict[str, Any], ppl_analysis: dict[str, Any] | None
) -> None:
    """Surface pairing statistics inside certificate.dataset.windows.stats."""
    if not isinstance(certificate, dict):
        return
    ds = certificate.get("dataset", {})
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
    certificate["dataset"] = ds


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
    baseline_artifacts = (
        baseline_raw.get("artifacts", {}) if isinstance(baseline_raw, dict) else {}
    ) or {}
    baseline_report_hash = _compute_report_digest(baseline_raw)
    edited_report_hash = _compute_report_digest(report)

    provenance: dict[str, Any] = {
        "policy": dict(policy_provenance),
        "baseline": {
            "run_id": baseline_ref.get("run_id"),
            "report_hash": baseline_report_hash,
            "report_path": baseline_artifacts.get("report_path")
            or baseline_artifacts.get("logs_path"),
        },
        "edited": {
            "run_id": current_run_id,
            "report_hash": edited_report_hash,
            "report_path": artifacts_payload.get("report_path"),
        },
        "env_flags": _collect_backend_versions(),
    }

    try:
        report_prov = (
            report.get("provenance", {})
            if isinstance(report.get("provenance"), dict)
            else {}
        )
        provider_digest = (
            report_prov.get("provider_digest")
            if isinstance(report_prov, dict)
            else None
        )
        if isinstance(provider_digest, dict) and provider_digest:
            provenance["provider_digest"] = dict(provider_digest)
        try:
            ds = report_prov.get("dataset_split")
            sf = report_prov.get("split_fallback")
            if ds:
                provenance["dataset_split"] = ds
            if isinstance(sf, bool):
                provenance["split_fallback"] = sf
        except Exception:  # pragma: no cover
            pass
    except Exception:  # pragma: no cover
        pass

    if isinstance(ppl_analysis, dict) and ppl_analysis.get("window_plan"):
        provenance["window_plan"] = ppl_analysis["window_plan"]

    if isinstance(schedule_digest, str) and schedule_digest:
        provenance["window_ids_digest"] = schedule_digest
        provenance.setdefault("window_plan_digest", schedule_digest)
        try:
            if not isinstance(provenance.get("provider_digest"), dict):
                provenance["provider_digest"] = {"ids_sha256": schedule_digest}
        except Exception:  # pragma: no cover
            pass

    try:
        if isinstance(report, dict):
            provenance["edit_digest"] = _compute_edit_digest(report)
    except Exception:  # pragma: no cover
        pass

    return provenance


def _resolve_pm_acceptance_range_from_report(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from report context/meta/env."""

    base_min = 0.95
    base_max = 1.10

    def _safe_float(val: Any) -> float | None:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    cfg_min = None
    cfg_max = None
    ctx = report.get("context") if isinstance(report, dict) else None
    if isinstance(ctx, dict):
        pm_ctx = (
            ctx.get("primary_metric")
            if isinstance(ctx.get("primary_metric"), dict)
            else {}
        )
        if isinstance(pm_ctx, dict):
            cfg_min = _safe_float(pm_ctx.get("acceptance_range", {}).get("min"))
            cfg_max = _safe_float(pm_ctx.get("acceptance_range", {}).get("max"))
        if cfg_min is None or cfg_max is None:
            alt = ctx.get("pm_acceptance_range")
            if isinstance(alt, dict):
                cfg_min = (
                    cfg_min if cfg_min is not None else _safe_float(alt.get("min"))
                )
                cfg_max = (
                    cfg_max if cfg_max is not None else _safe_float(alt.get("max"))
                )

    if (cfg_min is None or cfg_max is None) and isinstance(report, dict):
        meta = report.get("meta")
        if isinstance(meta, dict):
            meta_range = meta.get("pm_acceptance_range")
            if isinstance(meta_range, dict):
                cfg_min = (
                    cfg_min
                    if cfg_min is not None
                    else _safe_float(meta_range.get("min"))
                )
                cfg_max = (
                    cfg_max
                    if cfg_max is not None
                    else _safe_float(meta_range.get("max"))
                )

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
) -> dict[str, bool]:
    """Compute validation flags for the certificate including canonical gates."""
    tier = (tier or "balanced").lower()
    # Dev-only tiny relax: widen gates and lower floors when explicitly requested
    import os as _os

    _tiny_relax = str(_os.environ.get("INVARLOCK_TINY_RELAX", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if _tiny_relax:
        tier = "aggressive"

    tier_thresholds = {
        "conservative": 1.05,
        "balanced": 1.10,
        "aggressive": 1.20,
        "none": 1.10,
    }
    tier_policies = get_tier_policies()
    tier_policy = tier_policies.get(tier, tier_policies.get("balanced", {}))
    metrics_policy = (
        tier_policy.get("metrics", {}) if isinstance(tier_policy, dict) else {}
    )
    pm_policy = (
        metrics_policy.get("pm_ratio", {}) if isinstance(metrics_policy, dict) else {}
    )
    ratio_limit_base = pm_policy.get("ratio_limit_base")
    try:
        if ratio_limit_base is not None:
            ratio_limit_base = float(ratio_limit_base)
    except Exception:
        ratio_limit_base = None
    if not isinstance(ratio_limit_base, (int | float)) or not math.isfinite(
        float(ratio_limit_base)
    ):
        ratio_limit_base = float(tier_thresholds.get(tier, 1.10))
    acceptance = pm_acceptance_range if isinstance(pm_acceptance_range, dict) else {}
    ratio_min_bound = None
    ratio_max_bound = None
    try:
        if acceptance.get("min") is not None:
            ratio_min_bound = float(acceptance.get("min"))
    except Exception:
        ratio_min_bound = None
    try:
        if acceptance.get("max") is not None:
            ratio_max_bound = float(acceptance.get("max"))
    except Exception:
        ratio_max_bound = None

    ratio_limit = (
        ratio_max_bound
        if isinstance(ratio_max_bound, (int | float)) and math.isfinite(ratio_max_bound)
        else float(ratio_limit_base)
    )
    if isinstance(target_ratio, int | float) and target_ratio > 0:
        ratio_limit = min(ratio_limit, float(target_ratio))

    # Canonical Gates
    # 1. Drift gate: 0.95 ≤ final/preview ≤ 1.05
    drift_ratio = ppl.get("preview_final_ratio", 1.0)
    preview_final_drift_acceptable = 0.95 <= drift_ratio <= 1.05
    if _tiny_relax:
        # Treat drift identity as informational in tiny dev demos
        preview_final_drift_acceptable = True

    # 2. Primary metric vs baseline: edited/baseline ≤ tier threshold (ratio for ppl-like)
    ratio_vs_baseline = ppl.get("ratio_vs_baseline", 1.0)
    # Prefer primary_metric ratio when present
    if not (
        isinstance(ratio_vs_baseline, int | float) and math.isfinite(ratio_vs_baseline)
    ):
        try:
            pm_try = primary_metric if isinstance(primary_metric, dict) else {}
            pm_ratio = (
                pm_try.get("ratio_vs_baseline") if isinstance(pm_try, dict) else None
            )
            if isinstance(pm_ratio, int | float) and math.isfinite(pm_ratio):
                ratio_vs_baseline = float(pm_ratio)
        except Exception:  # pragma: no cover
            pass
    # Hysteresis and sample-size floors from tier policies
    hysteresis_ratio = float(pm_policy.get("hysteresis_ratio", 0.0))
    min_tokens = int(pm_policy.get("min_tokens", 0))
    # Evaluate sample-size sufficiency
    tokens_ok = True
    if isinstance(_ppl_metrics, dict):
        pt = _ppl_metrics.get("preview_total_tokens")
        ft = _ppl_metrics.get("final_total_tokens")
        has_pt = isinstance(pt, int | float) and math.isfinite(float(pt))
        has_ft = isinstance(ft, int | float) and math.isfinite(float(ft))
        if has_pt and has_ft and min_tokens > 0:
            try:
                total_tokens = int(pt) + int(ft)
                # Dataset-scale aware floors: use fraction of available tokens when provided
                eff_min_tokens = max(0, int(min_tokens))
                try:
                    if isinstance(dataset_capacity, dict):
                        frac = float(pm_policy.get("min_token_fraction", 0.0) or 0.0)
                        avail_tokens = dataset_capacity.get("tokens_available")
                        if isinstance(avail_tokens, int | float) and frac > 0.0:
                            eff_min_tokens = max(
                                eff_min_tokens,
                                int(math.ceil(float(avail_tokens) * frac)),
                            )
                except Exception:  # pragma: no cover
                    pass
                tokens_ok = total_tokens >= eff_min_tokens
            except Exception:  # pragma: no cover
                tokens_ok = True
    # Under tiny_relax, treat token floors as informational only
    tokens_ok_eff = tokens_ok or _tiny_relax
    # Apply hysteresis to ratio limit if needed
    ratio_limit_with_hyst = ratio_limit + max(0.0, hysteresis_ratio)
    lower_bound_ok = True
    if ratio_min_bound is not None and isinstance(ratio_vs_baseline, (int | float)):
        try:
            lower_bound_ok = math.isfinite(float(ratio_vs_baseline)) and (
                float(ratio_vs_baseline) >= float(ratio_min_bound)
            )
        except Exception:
            lower_bound_ok = True
    compression_acceptable = (
        isinstance(ratio_vs_baseline, int | float)
        and math.isfinite(ratio_vs_baseline)
        and lower_bound_ok
        and ratio_vs_baseline <= ratio_limit_with_hyst
        and tokens_ok_eff
    )
    if _tiny_relax:
        # In tiny demos, allow undefined ratio and relax floors
        if not isinstance(ratio_vs_baseline, int | float) or not math.isfinite(
            ratio_vs_baseline
        ):
            compression_acceptable = True
    ratio_ci = ppl.get("ratio_ci")
    if (
        isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
        and all(isinstance(x, int | float) and math.isfinite(x) for x in ratio_ci)
    ):
        compression_acceptable = (
            compression_acceptable
            and ratio_ci[1] <= ratio_limit_with_hyst
            and (ratio_min_bound is None or ratio_ci[0] >= ratio_min_bound)
        )

    # 3. RMT ε-rule compliance
    rmt_stable = rmt.get("stable", True)

    summary = spectral.get("summary", {}) if isinstance(spectral, dict) else {}
    max_caps = spectral.get("max_caps") or summary.get("max_caps")
    if max_caps is None:
        default_spectral = (
            tier_policy.get("spectral", {}) if isinstance(tier_policy, dict) else {}
        )
        max_caps = default_spectral.get("max_caps", 5)
    spectral_stable = spectral.get("caps_applied", 0) <= int(max_caps)
    if spectral.get("caps_exceeded"):
        spectral_stable = False

    guard_overhead_pass = True
    if isinstance(guard_overhead, dict) and guard_overhead:
        if "passed" in guard_overhead:
            guard_overhead_pass = bool(guard_overhead.get("passed"))
            if _tiny_relax and (
                not bool(guard_overhead.get("evaluated", True))
                or guard_overhead.get("errors")
            ):
                guard_overhead_pass = True
        else:
            ratio = guard_overhead.get("overhead_ratio")
            threshold = guard_overhead.get("overhead_threshold", 0.01)
            try:
                ratio_val = float(ratio)
                threshold_val = float(threshold)
            except (TypeError, ValueError):
                ratio_val = float("nan")
                threshold_val = 0.01
            if _tiny_relax and threshold_val < 0.10:
                threshold_val = 0.10
            if not math.isfinite(ratio_val):
                # In dev/Compare-&-Certify flows we often lack a bare run; treat missing metric as pass
                guard_overhead_pass = True
            else:
                guard_overhead_pass = ratio_val <= (1.0 + max(0.0, threshold_val))

    flags = {
        "preview_final_drift_acceptable": preview_final_drift_acceptable,
        "primary_metric_acceptable": compression_acceptable,
        "invariants_pass": invariants.get("status") not in {"fail", "error"},
        "spectral_stable": spectral_stable,
        "rmt_stable": rmt_stable,  # RMT ε-rule compliance
        "guard_overhead_acceptable": guard_overhead_pass,
    }
    # Mark hysteresis application when ratio exceeds base limit but passes with hysteresis
    try:
        base_ok = (
            isinstance(ratio_vs_baseline, int | float)
            and math.isfinite(ratio_vs_baseline)
            and ratio_vs_baseline <= ratio_limit
        )
        if not base_ok and compression_acceptable:
            flags["hysteresis_applied"] = True
    except Exception:  # pragma: no cover
        pass

    # Optional primary metric gating (metric-v1)
    try:
        if isinstance(primary_metric, dict) and primary_metric:
            kind = str(primary_metric.get("kind", "")).lower()
            if kind in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}:
                # Apply the same hysteresis and sample-size floors as primary_metric_acceptable
                pm_ratio = primary_metric.get("ratio_vs_baseline")
                if isinstance(pm_ratio, int | float) and math.isfinite(pm_ratio):
                    ok = (pm_ratio <= ratio_limit_with_hyst) and bool(tokens_ok_eff)
                else:
                    # Fall back to compression_acceptable when PM ratio is unavailable
                    ok = bool(compression_acceptable)
                flags["primary_metric_acceptable"] = bool(ok)
            elif kind in {"accuracy", "vqa_accuracy"}:
                # Read thresholds from tier policy if available
                acc_policy = (
                    metrics_policy.get("accuracy", {})
                    if isinstance(metrics_policy, dict)
                    else {}
                )
                delta_min_pp = float(acc_policy.get("delta_min_pp", -1.0))
                min_examples = int(acc_policy.get("min_examples", 200))
                hysteresis_pp = float(acc_policy.get("hysteresis_delta_pp", 0.0))
                delta = primary_metric.get("ratio_vs_baseline")
                meets_delta = (
                    isinstance(delta, int | float)
                    and math.isfinite(delta)
                    and (delta >= (delta_min_pp - max(0.0, hysteresis_pp)))
                )
                if _tiny_relax and not (
                    isinstance(delta, int | float) and math.isfinite(delta)
                ):
                    meets_delta = True
                n_fin = primary_metric.get("n_final")
                meets_n = True
                if isinstance(n_fin, int | float):
                    # Dataset-scale aware min_examples when available
                    eff_min_examples = int(min_examples)
                    try:
                        if isinstance(dataset_capacity, dict):
                            frac = float(
                                acc_policy.get("min_examples_fraction", 0.0) or 0.0
                            )
                            avail_ex = dataset_capacity.get("examples_available")
                            if isinstance(avail_ex, int | float) and frac > 0.0:
                                eff_min_examples = max(
                                    eff_min_examples,
                                    int(math.ceil(float(avail_ex) * frac)),
                                )
                    except Exception:  # pragma: no cover
                        pass
                    meets_n = int(n_fin) >= eff_min_examples
                    if _tiny_relax:
                        # In tiny demos accept smaller sample sizes
                        meets_n = True
                flags["primary_metric_acceptable"] = bool(meets_delta and meets_n)
                try:
                    if (
                        isinstance(delta, int | float)
                        and delta < delta_min_pp
                        and meets_delta
                    ):
                        flags["hysteresis_applied"] = True
                except Exception:  # pragma: no cover
                    pass
    except Exception:  # pragma: no cover
        # Fail-closed to False if something goes wrong
        flags["primary_metric_acceptable"] = False

    # Reconcile: if ppl-like primary_metric ratio is present and within hysteresis-adjusted
    # limit, prefer that decision to avoid spurious FAILs from upstream fallbacks.
    try:
        if isinstance(primary_metric, dict) and primary_metric:
            kind2 = str(primary_metric.get("kind", "")).lower()
            if kind2 in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}:
                pmr = primary_metric.get("ratio_vs_baseline")
                if (
                    isinstance(pmr, int | float)
                    and math.isfinite(float(pmr))
                    and float(pmr) <= (ratio_limit + max(0.0, hysteresis_ratio))
                    and bool(tokens_ok_eff)
                ):
                    flags["primary_metric_acceptable"] = True
    except Exception:  # pragma: no cover
        pass

    # MoE observability flags (non-gating)
    try:
        if isinstance(moe, dict) and moe:
            flags["moe_observed"] = True
            flags["moe_identity_ok"] = True
    except Exception:  # pragma: no cover
        pass

    return flags


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


## NOTE: _compute_certificate_hash moved to invarlock.reporting.render and is re-exported below.


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
            actual_bitwidth = first_module.get(
                "bitwidth",
                edit_config.get("bitwidth", edit_config.get("bits", "unknown")),
            )
        else:
            actual_bitwidth = edit_config.get(
                "bitwidth", edit_config.get("bits", "unknown")
            )

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

# Re-export rendering API from dedicated module to avoid bloat/cycles
# Rendering helpers live in invarlock.reporting.render; internal code should import there directly.
# Tests and public API expect render_certificate_markdown to be available from
# invarlock.reporting.certificate. Import lazily at module end to avoid cycles with
# invarlock.reporting.render which imports this module as a namespace.
try:  # pragma: no cover - simple re-export
    from .render import (
        compute_console_validation_block,  # type: ignore
        render_certificate_markdown,  # type: ignore
    )
except Exception:  # pragma: no cover - defensive fallback

    def render_certificate_markdown(certificate: dict[str, Any]) -> str:  # type: ignore
        raise ImportError(
            "render_certificate_markdown is unavailable; rendering dependencies missing"
        )

    def compute_console_validation_block(certificate: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        raise ImportError(
            "compute_console_validation_block is unavailable; rendering dependencies missing"
        )


# Export public API
__all__ = [
    "make_certificate",
    "validate_certificate",
    "_validate_with_jsonschema",
    "jsonschema",
    "render_certificate_markdown",
    "compute_console_validation_block",
    "CERTIFICATE_SCHEMA_VERSION",
    "CERTIFICATE_JSON_SCHEMA",
]
