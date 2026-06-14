from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigError, MetricsError, ValidationError
from .report_inputs import (
    ReportInputError,
    load_report_input_json,
    resolve_report_input_path,
)

_TEXT_NORMALIZATION_ERRORS = (RuntimeError, TypeError, ValueError)
_FINITE_NUMBER_ERRORS = (OverflowError, RuntimeError, TypeError, ValueError)
_OPTIONAL_BASELINE_REPORT_DATA_KEYS = {"seed"}


def require_run_report_artifact(run_result: str | Path | None, *, stage: str) -> Path:
    """Require the runner to return a concrete report artifact path."""

    if run_result is None:
        raise ConfigError(
            code="E221",
            message=(
                f"{stage} run did not return a report path; the runner contract "
                "requires an explicit report artifact."
            ),
            details={"stage": stage, "reason": "missing_report_path"},
        )

    try:
        return resolve_report_input_path(
            run_result,
            allow_canonical_directory=False,
        )
    except ReportInputError as exc:
        if exc.reason == "not_found":
            message = f"{stage} run returned a missing report path: {exc.path}"
        else:
            message = f"{stage} run returned a non-file report path: {exc.path}"
        raise ConfigError(
            code="E221",
            message=message,
            details={
                "stage": stage,
                "reason": exc.reason,
                "path": str(exc.path),
            },
        ) from exc


def load_validated_baseline_report(
    report_path: str | Path,
    *,
    expected_model_id: str,
    expected_profile: str,
    expected_tier: str,
    expected_adapter: str,
    expected_assurance_mode: str,
    expected_dataset: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Load and validate a reusable baseline report for `evaluate`."""

    try:
        resolved_report, payload = load_report_input_json(
            report_path,
            allow_canonical_directory=False,
        )
    except ReportInputError as exc:
        raise ValidationError(
            code="E222",
            message=_baseline_input_error_message(exc),
            details={"path": str(exc.path), "reason": exc.reason},
        ) from exc

    _validate_baseline_report_payload(
        payload,
        resolved_report=resolved_report,
        expected_model_id=expected_model_id,
        expected_profile=expected_profile,
        expected_tier=expected_tier,
        expected_adapter=expected_adapter,
        expected_assurance_mode=expected_assurance_mode,
        expected_dataset=expected_dataset,
    )
    return resolved_report, payload


def _baseline_input_error_message(exc: ReportInputError) -> str:
    if exc.reason == "not_found":
        return f"Baseline report not found: {exc.path}"
    if exc.reason == "directory_forbidden":
        return (
            "Baseline report must be an explicit report.json file path, not a "
            f"directory: {exc.path}"
        )
    if exc.reason == "unreadable":
        return f"Baseline report is not readable: {exc.path} ({exc.detail})"
    if exc.reason == "invalid_json":
        return f"Baseline report is not valid JSON: {exc.path} ({exc.detail})"
    if exc.reason == "non_object":
        return f"Baseline report must be a JSON object: {exc.path}"
    return f"Baseline report not found: {exc.path}"


def _validate_baseline_report_payload(
    payload: dict[str, Any],
    *,
    resolved_report: Path,
    expected_model_id: str,
    expected_profile: str,
    expected_tier: str,
    expected_adapter: str,
    expected_assurance_mode: str,
    expected_dataset: dict[str, Any] | None = None,
) -> None:
    edit_block = payload.get("edit")
    edit_name = edit_block.get("name") if isinstance(edit_block, dict) else None
    if edit_name != "noop":
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report must be a no-op run (edit.name == 'noop'). "
                f"Got edit.name={edit_name!r} in {resolved_report}"
            ),
            details={"path": str(resolved_report), "field": "edit.name"},
        )

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValidationError(
            code="E222",
            message=f"Baseline report missing meta object in {resolved_report}",
            details={"path": str(resolved_report), "field": "meta"},
        )
    baseline_model_id = meta.get("model_id")
    if not isinstance(baseline_model_id, str) or not _model_ids_equivalent(
        baseline_model_id,
        expected_model_id,
    ):
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report model mismatch. "
                f"Expected {expected_model_id!r}, got {baseline_model_id!r} in "
                f"{resolved_report}"
            ),
            details={"path": str(resolved_report), "field": "meta.model_id"},
        )
    baseline_adapter = meta.get("adapter")
    if not isinstance(baseline_adapter, str) or baseline_adapter != expected_adapter:
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report adapter mismatch. "
                f"Expected {expected_adapter!r}, got {baseline_adapter!r} in "
                f"{resolved_report}"
            ),
            details={"path": str(resolved_report), "field": "meta.adapter"},
        )

    context = payload.get("context")
    if not isinstance(context, dict):
        raise ValidationError(
            code="E222",
            message=f"Baseline report missing context object in {resolved_report}",
            details={"path": str(resolved_report), "field": "context"},
        )
    baseline_profile = context.get("profile")
    if (
        not isinstance(baseline_profile, str)
        or baseline_profile.strip().lower() != expected_profile.strip().lower()
    ):
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report profile mismatch. "
                f"Expected {expected_profile!r}, got {baseline_profile!r} in "
                f"{resolved_report}"
            ),
            details={"path": str(resolved_report), "field": "context.profile"},
        )
    baseline_tier = _baseline_report_tier(payload, context)
    if not isinstance(baseline_tier, str) or baseline_tier != expected_tier:
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report tier mismatch. "
                f"Expected {expected_tier!r}, got {baseline_tier!r} in "
                f"{resolved_report}"
            ),
            details={
                "path": str(resolved_report),
                "field": "context.auto.tier",
            },
        )
    baseline_assurance = _baseline_report_assurance_mode(payload, context)
    if (
        not isinstance(baseline_assurance, str)
        or baseline_assurance.strip().lower() != expected_assurance_mode.strip().lower()
    ):
        raise ValidationError(
            code="E222",
            message=(
                "Baseline report assurance mode mismatch. "
                f"Expected {expected_assurance_mode!r}, got {baseline_assurance!r} "
                f"in {resolved_report}"
            ),
            details={"path": str(resolved_report), "field": "context.assurance.mode"},
        )
    _validate_baseline_dataset(
        payload,
        resolved_report=resolved_report,
        expected_dataset=expected_dataset or {},
    )

    eval_windows = payload.get("evaluation_windows")
    if not isinstance(eval_windows, dict):
        raise ValidationError(
            code="E222",
            message=("Baseline report missing evaluation window payloads."),
            details={"path": str(resolved_report), "field": "evaluation_windows"},
        )

    for phase_name in ("preview", "final"):
        _validate_evaluation_window_phase(
            eval_windows,
            phase_name=phase_name,
            resolved_report=resolved_report,
        )


def _validate_evaluation_window_phase(
    eval_windows: dict[str, Any],
    *,
    phase_name: str,
    resolved_report: Path,
) -> None:
    phase = eval_windows.get(phase_name)
    if not isinstance(phase, dict):
        raise ValidationError(
            code="E222",
            message=(
                f"Baseline report missing evaluation_windows.{phase_name} payloads."
            ),
            details={
                "path": str(resolved_report),
                "field": f"evaluation_windows.{phase_name}",
            },
        )

    window_ids = phase.get("window_ids")
    input_ids = phase.get("input_ids")
    if isinstance(window_ids, list) and window_ids:
        if not isinstance(input_ids, list) or not input_ids:
            raise ValidationError(
                code="E222",
                message=(
                    f"Baseline report missing evaluation_windows.{phase_name}.input_ids."
                ),
                details={
                    "path": str(resolved_report),
                    "field": f"evaluation_windows.{phase_name}.input_ids",
                },
            )
        if len(input_ids) != len(window_ids):
            raise ValidationError(
                code="E222",
                message=(
                    "Baseline report has inconsistent evaluation window payloads for "
                    f"{phase_name}: input_ids={len(input_ids)} "
                    f"window_ids={len(window_ids)}."
                ),
                details={
                    "path": str(resolved_report),
                    "field": f"evaluation_windows.{phase_name}",
                },
            )
        return

    example_ids = phase.get("example_ids")
    records = phase.get("records")
    if isinstance(example_ids, list) and example_ids:
        if not isinstance(records, list) or not records:
            raise ValidationError(
                code="E222",
                message=(
                    f"Baseline report missing evaluation_windows.{phase_name}.records."
                ),
                details={
                    "path": str(resolved_report),
                    "field": f"evaluation_windows.{phase_name}.records",
                },
            )
        if len(records) != len(example_ids):
            raise ValidationError(
                code="E222",
                message=(
                    "Baseline report has inconsistent multimodal evaluation payloads for "
                    f"{phase_name}: records={len(records)} "
                    f"example_ids={len(example_ids)}."
                ),
                details={
                    "path": str(resolved_report),
                    "field": f"evaluation_windows.{phase_name}",
                },
            )
        return

    raise ValidationError(
        code="E222",
        message=(
            "Baseline report missing evaluation evidence for "
            f"evaluation_windows.{phase_name}."
        ),
        details={
            "path": str(resolved_report),
            "field": f"evaluation_windows.{phase_name}",
        },
    )


def _model_ids_equivalent(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    actual_path = Path(actual).expanduser()
    expected_path = Path(expected).expanduser()
    if actual_path.exists() or expected_path.exists():
        try:
            return actual_path.resolve() == expected_path.resolve()
        except OSError:
            return False
    return False


def _baseline_report_tier(
    payload: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    auto_ctx = context.get("auto")
    tier = auto_ctx.get("tier") if isinstance(auto_ctx, dict) else None
    if isinstance(tier, str):
        return tier
    tier = context.get("tier")
    if isinstance(tier, str):
        return tier
    auto_top = payload.get("auto")
    tier = auto_top.get("tier") if isinstance(auto_top, dict) else None
    if isinstance(tier, str):
        return tier
    meta = payload.get("meta")
    auto_meta = meta.get("auto") if isinstance(meta, dict) else None
    tier = auto_meta.get("tier") if isinstance(auto_meta, dict) else None
    if isinstance(tier, str):
        return tier
    return None


def _baseline_report_assurance_mode(
    payload: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    context_assurance = context.get("assurance")
    mode = (
        context_assurance.get("mode") if isinstance(context_assurance, dict) else None
    )
    if isinstance(mode, str):
        return mode
    report_assurance = payload.get("assurance")
    mode = report_assurance.get("mode") if isinstance(report_assurance, dict) else None
    if isinstance(mode, str):
        return mode
    return None


def _validate_baseline_dataset(
    payload: dict[str, Any],
    *,
    resolved_report: Path,
    expected_dataset: dict[str, Any],
) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValidationError(
            code="E222",
            message=f"Baseline report missing data object in {resolved_report}",
            details={"path": str(resolved_report), "field": "data"},
        )
    aliases = {
        "provider": ("provider", "dataset"),
        "split": ("split", "dataset_split"),
        "seq_len": ("seq_len", "sequence_length"),
        "stride": ("stride",),
        "preview_n": ("preview_n",),
        "final_n": ("final_n",),
        "seed": ("seed",),
    }
    for expected_key, report_keys in aliases.items():
        if expected_key not in expected_dataset:
            continue
        expected_value = expected_dataset.get(expected_key)
        actual_value = next(
            (data.get(report_key) for report_key in report_keys if report_key in data),
            None,
        )
        if actual_value is None and expected_key in _OPTIONAL_BASELINE_REPORT_DATA_KEYS:
            continue
        if not _baseline_dataset_value_matches(
            expected_key,
            expected_value,
            actual_value,
        ):
            raise ValidationError(
                code="E222",
                message=(
                    "Baseline report dataset/window-plan mismatch. "
                    f"Expected {expected_key}={expected_value!r}, "
                    f"got {actual_value!r} in {resolved_report}"
                ),
                details={
                    "path": str(resolved_report),
                    "field": f"data.{expected_key}",
                },
            )


def _baseline_dataset_value_matches(
    expected_key: str,
    expected_value: Any,
    actual_value: Any,
) -> bool:
    if actual_value == expected_value:
        return True
    if expected_key == "provider":
        return _provider_values_equivalent(expected_value, actual_value)
    return False


def _provider_kind(value: Any) -> str | None:
    if isinstance(value, str):
        kind = value.strip()
        return kind or None
    if isinstance(value, dict):
        for key in ("kind", "provider", "dataset"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _provider_values_equivalent(expected_value: Any, actual_value: Any) -> bool:
    expected_kind = _provider_kind(expected_value)
    actual_kind = _provider_kind(actual_value)
    return bool(expected_kind and actual_kind and expected_kind == actual_kind)


@dataclass(frozen=True)
class PrimaryMetricPolicyDiagnostic:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"


@dataclass(frozen=True)
class PrimaryMetricPolicyOutcome:
    payload: dict[str, Any]
    error: MetricsError | None
    diagnostic: PrimaryMetricPolicyDiagnostic | None = None


def apply_edited_primary_metric_policy(
    edited_payload: dict[str, Any],
    *,
    profile: str | None,
) -> PrimaryMetricPolicyOutcome:
    """Normalize degraded edited metrics and compute fail-closed outcome."""

    try:
        prof = str(profile or "").strip().lower()
    except _TEXT_NORMALIZATION_ERRORS:
        prof = ""
    if prof not in {"ci", "ci_cpu", "release"}:
        return PrimaryMetricPolicyOutcome(
            payload=edited_payload,
            error=None,
            diagnostic=None,
        )

    normalized_payload = (
        dict(edited_payload) if isinstance(edited_payload, dict) else {}
    )
    metrics = normalized_payload.get("metrics", {})
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    pm = metrics.get("primary_metric", {})
    pm = dict(pm) if isinstance(pm, dict) else {}
    has_metric_block = isinstance(pm, dict) and bool(pm)
    if not has_metric_block:
        return PrimaryMetricPolicyOutcome(
            payload=normalized_payload or edited_payload,
            error=None,
            diagnostic=None,
        )

    pm_prev = pm.get("preview")
    pm_final = pm.get("final")
    pm.get("ratio_vs_baseline")
    degraded = bool(pm.get("invalid") or pm.get("degraded"))
    if not degraded and _finite_number(pm_final):
        return PrimaryMetricPolicyOutcome(
            payload=normalized_payload or edited_payload,
            error=None,
            diagnostic=None,
        )

    degraded_reason = pm.get("degraded_reason") or (
        "non_finite_pm"
        if (not _finite_number(pm_prev) or not _finite_number(pm_final))
        else "primary_metric_degraded"
    )

    meta = edited_payload.get("meta", {}) if isinstance(edited_payload, dict) else {}
    device = meta.get("device") if isinstance(meta, dict) else None
    adapter_name = meta.get("adapter") if isinstance(meta, dict) else None
    edit_name = (
        (edited_payload.get("edit", {}) or {}).get("name")
        if isinstance(edited_payload, dict)
        else None
    )

    pm["degraded"] = True
    pm["invalid"] = pm.get("invalid") or True
    pm["degraded_reason"] = degraded_reason
    metrics["primary_metric"] = pm
    normalized_payload["metrics"] = metrics

    err = MetricsError(
        code="E111",
        message=f"Primary metric degraded or non-finite ({degraded_reason}).",
        details={
            "reason": degraded_reason,
            "adapter": adapter_name,
            "device": device,
            "edit": edit_name,
        },
    )
    return PrimaryMetricPolicyOutcome(
        payload=normalized_payload,
        error=err,
        diagnostic=PrimaryMetricPolicyDiagnostic(
            code="evaluate.primary_metric_degraded",
            message=(
                "Primary metric degraded or non-finite; emitting evaluation report "
                "and marking task degraded. Primary metric computation failed."
            ),
            details={
                "reason": degraded_reason,
                "adapter": adapter_name,
                "device": device,
                "edit": edit_name,
            },
        ),
    )


def _finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    except _FINITE_NUMBER_ERRORS:
        return False


__all__ = [
    "PrimaryMetricPolicyDiagnostic",
    "PrimaryMetricPolicyOutcome",
    "apply_edited_primary_metric_policy",
    "load_validated_baseline_report",
    "require_run_report_artifact",
]
