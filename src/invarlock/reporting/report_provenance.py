"""Provenance assembly helpers for evaluation report generation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from typing import Any

from invarlock.utils.digest import hash_json

POLICY_VERSION = "policy-v1"
_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


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
    except _NON_FATAL_EXCEPTIONS:
        edits = {}

    family = "cert_only"
    impl_hash = hash_json({"family": family})
    try:
        if str(edits.get("name", "")) == "quant_rtn":
            family = "quantization"
            config = edits.get("config", {})
            if not isinstance(config, dict):
                config = {}
            impl_hash = hash_json({"name": "quant_rtn", "config": config})
    except _NON_FATAL_EXCEPTIONS:
        pass
    return {"family": family, "impl_hash": impl_hash, "version": 1}


def collect_backend_versions() -> dict[str, Any]:
    """Collect backend/library versions for provenance.env_flags."""
    info: dict[str, Any] = {}
    try:
        info["python"] = platform.python_version()
        info["platform"] = platform.platform()
        info["machine"] = platform.machine()
    except _NON_FATAL_EXCEPTIONS:
        pass

    torch: Any | None
    try:  # pragma: no cover - depends on torch availability
        import torch
    except ImportError:  # pragma: no cover - torch not available
        torch = None
    if torch is not None:
        info["torch"] = getattr(torch, "__version__", None)
        torch_version = getattr(torch, "version", None)
        if torch_version is not None:
            info["torch_cuda"] = getattr(torch_version, "cuda", None)
            info["torch_cudnn"] = getattr(torch_version, "cudnn", None)
            info["torch_git"] = getattr(torch_version, "git_version", None)
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["device_name"] = getattr(props, "name", None)
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None and minor is not None:
                    info["sm_capability"] = f"{int(major)}.{int(minor)}"
        except _NON_FATAL_EXCEPTIONS:
            pass
        try:
            if hasattr(torch.backends, "cudnn") and hasattr(
                torch.backends.cudnn, "version"
            ):
                version = torch.backends.cudnn.version()
                info["cudnn_runtime"] = int(version) if version is not None else None
        except _NON_FATAL_EXCEPTIONS:
            pass
        try:
            nccl = getattr(torch.cuda, "nccl", None)
            if nccl is not None and hasattr(nccl, "version"):
                info["nccl"] = str(nccl.version())
        except _NON_FATAL_EXCEPTIONS:
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
        except _NON_FATAL_EXCEPTIONS:
            pass

    try:
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    except _NON_FATAL_EXCEPTIONS:
        cublas_workspace = None
    if cublas_workspace:
        info["cublas_workspace_config"] = cublas_workspace

    return {key: value for key, value in info.items() if value is not None}


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


def build_provenance_block(
    report: Any,
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any],
    artifacts_payload: dict[str, Any],
    policy_provenance: dict[str, Any],
    schedule_digest: str | None,
    ppl_analysis: dict[str, Any],
    current_run_id: str,
    *,
    compute_report_digest_fn: Any,
    collect_backend_versions_fn: Any,
    compute_edit_digest_fn: Any,
) -> dict[str, Any]:
    """Assemble report provenance with policy/baseline/edit/run context."""

    baseline_artifacts = (
        baseline_raw.get("artifacts", {}) if isinstance(baseline_raw, dict) else {}
    ) or {}
    baseline_report_hash = compute_report_digest_fn(baseline_raw)
    edited_report_hash = compute_report_digest_fn(report)

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
        "env_flags": collect_backend_versions_fn(),
    }

    report_map = report if isinstance(report, dict) else {}
    report_provenance = (
        report_map.get("provenance", {})
        if isinstance(report_map.get("provenance"), dict)
        else {}
    )
    provider_digest = (
        report_provenance.get("provider_digest")
        if isinstance(report_provenance, dict)
        else None
    )
    if isinstance(provider_digest, dict) and provider_digest:
        provenance["provider_digest"] = dict(provider_digest)
    dataset_split = report_provenance.get("dataset_split")
    split_fallback = report_provenance.get("split_fallback")
    if dataset_split:
        provenance["dataset_split"] = dataset_split
    if isinstance(split_fallback, bool):
        provenance["split_fallback"] = split_fallback

    if isinstance(ppl_analysis, dict) and ppl_analysis.get("window_plan"):
        provenance["window_plan"] = ppl_analysis["window_plan"]

    if isinstance(schedule_digest, str) and schedule_digest:
        provenance["window_ids_digest"] = schedule_digest
        provenance.setdefault("window_plan_digest", schedule_digest)
        if not isinstance(provenance.get("provider_digest"), dict):
            provenance["provider_digest"] = {"ids_sha256": schedule_digest}

    if isinstance(report_map, dict) and report_map:
        provenance["edit_digest"] = compute_edit_digest_fn(report_map)

    return provenance
