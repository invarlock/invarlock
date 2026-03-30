from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .doctor_findings import (
    DATASET_SPLIT_FALLBACK_WARNING,
    DoctorFinding,
    build_bootstrap_replicates_findings,
    build_capacity_findings,
    build_provider_kind_findings,
    build_provider_schema_findings,
    build_split_fallback_findings,
    load_explicit_report_input,
)

DETERMINISM_SHARDS_WARNING = "Provider workers > 0 without deterministic_shards=True; enable deterministic_shards or set workers=0 for determinism."
_DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class DoctorConfigPreflightResult:
    lines: tuple[str, ...]
    findings: tuple[DoctorFinding, ...]
    had_error: bool
    metric_kind: str | None
    policy_meta: dict[str, Any] | None


def _mapping_get(value: object, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    try:
        return getattr(value, key)
    except AttributeError:
        return None


def _error_finding(
    code: str,
    message: str,
    *,
    field: str,
    exc: Exception,
) -> DoctorFinding:
    return DoctorFinding(
        code=code,
        severity="error",
        message=f"{message}: {exc}",
        extra={"field": field, "error": str(exc)},
    )


def _append_findings(
    findings: list[DoctorFinding],
    incoming: list[DoctorFinding],
    *,
    had_error: bool,
    incoming_error: bool = False,
) -> bool:
    findings.extend(incoming)
    return (
        had_error
        or incoming_error
        or any(finding.severity == "error" for finding in incoming)
    )


def run_doctor_config_preflight(
    *,
    config_path: str,
    profile: str | None = None,
    tier: str | None = None,
    baseline: str | None = None,
) -> DoctorConfigPreflightResult:
    import invarlock.core.config_runtime as config_runtime
    import invarlock.core.metric_provider_resolution as metric_provider_resolution

    findings: list[DoctorFinding] = []
    lines: list[str] = []
    had_error = False
    metric_kind: str | None = None
    policy_meta: dict[str, Any] | None = None

    cfg = config_runtime.load_config(config_path)
    if profile:
        cfg = config_runtime.apply_profile(cfg, profile)

    dataset_cfg = getattr(cfg, "dataset", None)
    provider_cfg = getattr(dataset_cfg, "provider", None)

    provider_findings, provider_error = build_provider_kind_findings(provider_cfg)
    had_error = _append_findings(
        findings,
        provider_findings,
        had_error=had_error,
        incoming_error=provider_error,
    )

    schema_findings, schema_error = build_provider_schema_findings(provider_cfg)
    had_error = _append_findings(
        findings,
        schema_findings,
        had_error=had_error,
        incoming_error=schema_error,
    )

    try:
        workers = int(_mapping_get(provider_cfg, "workers") or 0)
    except (TypeError, ValueError) as exc:
        workers = 0
        findings.append(
            _error_finding(
                "D015",
                "Invalid dataset.provider.workers value",
                field="dataset.provider.workers",
                exc=exc,
            )
        )
        had_error = True
    deterministic_shards = bool(_mapping_get(provider_cfg, "deterministic_shards"))
    if workers > 0 and not deterministic_shards:
        findings.append(
            DoctorFinding(
                code="D002",
                severity="warning",
                message=DETERMINISM_SHARDS_WARNING,
                extra={"field": "dataset.provider.deterministic_shards"},
            )
        )
        lines.append(DETERMINISM_SHARDS_WARNING)

    section_fn = getattr(cfg, "section", None)
    if callable(section_fn):
        try:
            eval_section = section_fn("eval")
        except _DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS as exc:
            eval_section = None
            findings.append(
                _error_finding(
                    "D019",
                    "Failed to resolve eval config section",
                    field="eval",
                    exc=exc,
                )
            )
            had_error = True
    else:
        try:
            eval_section = cfg.eval
        except _DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS as exc:
            eval_section = None
            findings.append(
                _error_finding(
                    "D019",
                    "Failed to access eval config section",
                    field="eval",
                    exc=exc,
                )
            )
            had_error = True
    replicates = _mapping_get(_mapping_get(eval_section, "bootstrap"), "replicates")
    findings.extend(build_bootstrap_replicates_findings(replicates))

    if baseline:
        _, baseline_payload, baseline_findings, invalid_baseline = (
            load_explicit_report_input(
                baseline,
                label="Baseline",
                field="baseline",
            )
        )
        had_error = _append_findings(
            findings,
            baseline_findings,
            had_error=had_error,
            incoming_error=invalid_baseline,
        )
        split_findings = build_split_fallback_findings(baseline_payload)
        if split_findings:
            lines.append(DATASET_SPLIT_FALLBACK_WARNING)
            findings.extend(split_findings)

    model_profile = None
    try:
        from invarlock import model_profile as model_profile_mod

        model_cfg = getattr(cfg, "model", None)
        model_profile = model_profile_mod.detect_model_profile(
            model_id=getattr(model_cfg, "id", "") or "",
            adapter=getattr(model_cfg, "adapter", None),
        )
    except _DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS as exc:
        findings.append(
            _error_finding(
                "D016",
                "Model profile detection failed",
                field="model",
                exc=exc,
            )
        )
        had_error = True

    if model_profile is not None:
        try:
            metric_kind, provider_kind, _metric_opts = (
                metric_provider_resolution.resolve_metric_and_provider(
                    cfg,
                    model_profile,
                    resolved_loss_type=getattr(model_profile, "default_loss", None),
                )
            )
        except _DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS as exc:
            metric_kind = None
            provider_kind = None
            findings.append(
                _error_finding(
                    "D017",
                    "Metric/provider resolution failed",
                    field="eval.primary_metric",
                    exc=exc,
                )
            )
            had_error = True
        else:
            lines.append(f"  Metric: {metric_kind} · Provider: {provider_kind}")
            if provider_kind:
                try:
                    from invarlock import model_profile as model_profile_mod
                    from invarlock.eval.data import get_provider

                    provider = get_provider(provider_kind)
                    tokenizer, tokenizer_hash = model_profile_mod.resolve_tokenizer(
                        model_profile
                    )
                    lines.append(
                        f"  Tokenizer: {tokenizer.__class__.__name__} · hash={tokenizer_hash}"
                    )
                    estimate_capacity = getattr(provider, "estimate_capacity", None)
                    if callable(estimate_capacity):
                        cap = estimate_capacity(
                            tokenizer=tokenizer,
                            seq_len=_mapping_get(dataset_cfg, "seq_len"),
                            stride=_mapping_get(dataset_cfg, "stride"),
                            split=_mapping_get(dataset_cfg, "split") or "validation",
                            target_total=int(
                                (_mapping_get(dataset_cfg, "preview_n") or 0)
                                + (_mapping_get(dataset_cfg, "final_n") or 0)
                            ),
                            fast_mode=True,
                        )
                        capacity_findings, insufficient, policy_meta = (
                            build_capacity_findings(
                                cap=cap if isinstance(cap, dict) else {},
                                tier=(tier or "balanced"),
                            )
                        )
                        had_error = _append_findings(
                            findings,
                            capacity_findings,
                            had_error=had_error,
                            incoming_error=insufficient,
                        )
                    else:
                        lines.append(
                            "  [dim]Provider does not expose estimate_capacity()[/dim]"
                        )
                except _DOCTOR_PREFLIGHT_NON_FATAL_EXCEPTIONS as exc:
                    findings.append(
                        _error_finding(
                            "D018",
                            "Provider capacity inspection failed",
                            field="dataset.provider",
                            exc=exc,
                        )
                    )
                    had_error = True

    return DoctorConfigPreflightResult(
        lines=tuple(lines),
        findings=tuple(findings),
        had_error=had_error,
        metric_kind=metric_kind,
        policy_meta=policy_meta,
    )


__all__ = [
    "DETERMINISM_SHARDS_WARNING",
    "DoctorConfigPreflightResult",
    "run_doctor_config_preflight",
]
