"""Spectral correction execution and evidence-ledger construction."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from .spectral_control import apply_spectral_control


def _selected_finding_records(
    selected_violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, finding in enumerate(selected_violations, start=1):
        item = dict(finding)
        item["finding_id"] = (
            f"finding-{index:04d}:{str(item.get('type') or 'spectral')}"
            f":{str(item.get('module') or 'unknown')}"
        )
        records.append(item)
    return records


def _selected_weight_digests(
    guard: Any, model: Any, selected_modules: set[str]
) -> dict[str, str]:
    digests: dict[str, str] = {}
    scoped_modules = getattr(guard, "_get_scoped_modules", None)
    if not callable(scoped_modules):
        return digests
    live_modules = {str(name): module for name, module in scoped_modules(model)}
    for name in sorted(selected_modules):
        module = live_modules.get(name)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        try:
            dense = weight.detach().contiguous().cpu()
            payload = dense.view(torch.uint8).numpy().tobytes()
        except (RuntimeError, TypeError, ValueError):
            continue
        digest = hashlib.sha256()
        digest.update(str(dense.dtype).encode("utf-8"))
        digest.update(str(tuple(dense.shape)).encode("utf-8"))
        digest.update(payload)
        digests[name] = digest.hexdigest()
    return digests


def run_correction_lifecycle(
    guard: Any,
    model: Any,
    *,
    phase: str,
    pre_correction_metrics: dict[str, float],
    selected_violations: list[dict[str, Any]],
    multiple_testing_selection: dict[str, Any],
    apply_spectral_control_fn: Any = apply_spectral_control,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply selected corrections and bind findings to replayable evidence."""
    selected_findings = _selected_finding_records(selected_violations)
    selected_by_module: dict[str, list[str]] = {}
    for finding in selected_findings:
        module = str(finding.get("module") or "")
        if module:
            selected_by_module.setdefault(module, []).append(finding["finding_id"])

    correction_enabled = bool(getattr(guard, "correction_enabled", False))
    cap_ratio = float(getattr(guard, "correction_cap_ratio", 2.0))
    pre_z_scores = dict(getattr(guard, "latest_z_scores", {}) or {})
    pre_degeneracy = {
        name: dict(values)
        for name, values in (getattr(guard, "latest_degeneracy", {}) or {}).items()
    }
    selected_module_names = set(selected_by_module)
    pre_weight_digests = _selected_weight_digests(guard, model, selected_module_names)
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "correction_enabled": correction_enabled,
        "correction_cap_ratio": cap_ratio,
        "pre_correction_metrics": dict(pre_correction_metrics),
        "pre_correction_z_scores": pre_z_scores,
        "pre_correction_degeneracy": pre_degeneracy,
        "multiple_testing_selection": dict(multiple_testing_selection),
        "selected_findings": selected_findings,
        "corrections": [],
    }

    if not selected_findings:
        ledger["policy_result"] = "no_selected_findings"
        ledger["post_correction_metrics"] = dict(pre_correction_metrics)
        return dict(pre_correction_metrics), ledger

    if not correction_enabled:
        correction_records: list[dict[str, Any]] = []
        evidence_incomplete = False
        for index, module in enumerate(sorted(selected_by_module), start=1):
            pre_sigma = pre_correction_metrics.get(module)
            baseline_sigma = getattr(guard, "baseline_sigmas", {}).get(module)
            weight_digest = pre_weight_digests.get(module)
            if weight_digest is None:
                evidence_incomplete = True
            correction_records.append(
                {
                    "correction_id": f"correction-{index:04d}:{module}",
                    "finding_ids": list(selected_by_module[module]),
                    "module": module,
                    "operation": "none",
                    "attempted": False,
                    "mutation_applied": False,
                    "outcome": (
                        "evidence_missing"
                        if weight_digest is None
                        else "not_attempted_policy_disabled"
                    ),
                    "pre_sigma": pre_sigma,
                    "baseline_sigma": baseline_sigma,
                    "post_sigma": pre_sigma,
                    "scale_factor": 1.0,
                    "pre_weight_digest": weight_digest,
                    "post_weight_digest": weight_digest,
                }
            )
        ledger["corrections"] = correction_records
        ledger["policy_result"] = (
            "evidence_incomplete" if evidence_incomplete else "correction_disabled"
        )
        ledger["post_correction_metrics"] = dict(pre_correction_metrics)
        return dict(pre_correction_metrics), ledger

    control_result = apply_spectral_control_fn(
        model,
        policy={
            "sigma_quantile": guard.sigma_quantile,
            "scope": guard.scope,
            "baseline_sigmas": guard.baseline_sigmas,
            "target_sigma": guard.target_sigma,
            "cap_ratio": cap_ratio,
            "selected_modules": sorted(selected_by_module),
        },
    )
    post_correction_metrics = guard._capture_sigmas(
        model, phase=f"{phase}_post_correction"
    )
    post_weight_digests = _selected_weight_digests(guard, model, selected_module_names)
    corrections = {
        str(item.get("module")): item
        for item in control_result.get("corrections", [])
        if isinstance(item, dict) and item.get("module")
    }
    cap_result = control_result.get("cap_result")
    raw_failures = (
        cap_result.get("failed_modules", []) if isinstance(cap_result, dict) else []
    )
    failures = {
        str(item[0]): str(item[1])
        for item in raw_failures
        if isinstance(item, (list, tuple)) and len(item) >= 2
    }
    correction_records = []
    correction_failed = False
    mutation_count = 0
    for index, module in enumerate(sorted(selected_by_module), start=1):
        pre_sigma = pre_correction_metrics.get(module)
        post_sigma = post_correction_metrics.get(module)
        baseline_sigma = guard.baseline_sigmas.get(module)
        target_sigma = (
            float(baseline_sigma) * cap_ratio
            if isinstance(baseline_sigma, int | float)
            else None
        )
        correction = corrections.get(module)
        mutation_applied = correction is not None
        if correction is not None:
            mutation_count += 1
            outcome = "applied_and_remeasured"
            scale_factor = float(correction.get("scale_factor", 1.0))
        elif module in failures:
            outcome = "correction_failed"
            scale_factor = 1.0
            correction_failed = True
        elif (
            isinstance(pre_sigma, int | float)
            and isinstance(target_sigma, int | float)
            and float(pre_sigma) <= float(target_sigma)
        ):
            outcome = "no_mutation_required"
            scale_factor = 1.0
        else:
            outcome = "correction_failed"
            scale_factor = 1.0
            correction_failed = True
        if post_sigma is None:
            outcome = "post_measurement_missing"
            correction_failed = True
        if (
            pre_weight_digests.get(module) is None
            or post_weight_digests.get(module) is None
        ):
            outcome = "weight_digest_missing"
            correction_failed = True
        correction_records.append(
            {
                "correction_id": f"correction-{index:04d}:{module}",
                "finding_ids": list(selected_by_module[module]),
                "module": module,
                "operation": "relative_spectral_cap",
                "attempted": True,
                "mutation_applied": mutation_applied,
                "outcome": outcome,
                "pre_sigma": pre_sigma,
                "baseline_sigma": baseline_sigma,
                "target_sigma": target_sigma,
                "post_sigma": post_sigma,
                "scale_factor": scale_factor,
                "failure": failures.get(module),
                "pre_weight_digest": pre_weight_digests.get(module),
                "post_weight_digest": post_weight_digests.get(module),
            }
        )
    ledger["corrections"] = correction_records
    ledger["post_correction_metrics"] = dict(post_correction_metrics)
    ledger["policy_result"] = (
        "correction_failed"
        if correction_failed
        else ("corrections_applied" if mutation_count else "corrections_not_required")
    )
    return post_correction_metrics, ledger


def attach_correction_metrics(
    metrics: dict[str, Any], correction_ledger: dict[str, Any]
) -> None:
    """Add correction counts and policy outcome to spectral result metrics."""
    corrections = correction_ledger.get("corrections")
    corrections = corrections if isinstance(corrections, list) else []
    metrics["selected_budgeted_findings"] = int(metrics.get("caps_applied", 0))
    metrics["cap_budget_exceeded"] = bool(metrics.get("caps_exceeded", False))
    metrics["corrections_attempted"] = sum(
        1
        for correction in corrections
        if isinstance(correction, dict) and correction.get("attempted") is True
    )
    metrics["corrections_applied"] = sum(
        1
        for correction in corrections
        if isinstance(correction, dict) and correction.get("mutation_applied") is True
    )
    metrics["correction_policy_result"] = correction_ledger.get("policy_result")


__all__ = ["attach_correction_metrics", "run_correction_lifecycle"]
