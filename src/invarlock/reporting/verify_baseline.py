"""Fail-closed baseline contracts for strict assurance verification.

Strict verification accepts only a complete canonical evaluation report as its
independent baseline.  This module deliberately does not normalize legacy
baseline fragments: normalization is useful while producing reports, but it
would let a independently supplied assertion masquerade as measured evidence.
"""

from __future__ import annotations

import math
from typing import Any

from invarlock.core.checkpoint_identity import (
    LEGACY_MODEL_IDENTITY_FIELDS,
    validated_model_identity,
)
from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    normalize_metric_kind,
)

from .report_provenance import compute_report_digest
from .verify_dataset_identity import append_strict_dataset_identity_errors


def _mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _resolve(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _baseline_path(payload: dict[str, Any], logical_path: str) -> str:
    return {
        "dataset.provider": "data.dataset",
        "dataset.split": "data.split",
        "dataset.seq_len": "data.seq_len",
        "dataset.hash.dataset": "data.dataset_hash",
    }.get(logical_path, logical_path)


def _canonical_run_report(payload: dict[str, Any]) -> bool:
    """Recognize the complete report.json emitted by an InvarLock baseline run."""

    required_mappings = (
        "meta",
        "context",
        "edit",
        "data",
        "metrics",
        "evaluation_windows",
        "provenance",
        "artifacts",
    )
    if any(not isinstance(payload.get(key), dict) for key in required_mappings):
        return False
    edit = _mapping(payload.get("edit")) or {}
    if _nonempty_text(edit.get("name")) != "noop":
        return False
    primary_metric = _mapping(_resolve(payload, "metrics.primary_metric"))
    preview = _mapping(_resolve(payload, "evaluation_windows.preview"))
    final = _mapping(_resolve(payload, "evaluation_windows.final"))
    provider_digest = _mapping(_resolve(payload, "provenance.provider_digest"))
    return all(
        value is not None for value in (primary_metric, preview, final, provider_digest)
    )


def _append_required_text_parity(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    path: str,
    label: str,
) -> None:
    subject_value = _nonempty_text(_resolve(subject, path))
    baseline_resolved_path = _baseline_path(baseline, path)
    baseline_value = _nonempty_text(_resolve(baseline, baseline_resolved_path))
    if subject_value is None:
        errors.append(f"Strict baseline provenance requires non-empty report.{path}.")
    if baseline_value is None:
        errors.append(
            "Strict baseline provenance requires non-empty "
            f"supplied_baseline.{baseline_resolved_path}."
        )
    if (
        subject_value is not None
        and baseline_value is not None
        and subject_value != baseline_value
    ):
        errors.append(
            f"Strict baseline {label} mismatch: report.{path}={subject_value!r} "
            f"supplied_baseline.{baseline_resolved_path}={baseline_value!r}."
        )


def _append_required_integer_parity(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    path: str,
    label: str,
) -> None:
    subject_value = _resolve(subject, path)
    baseline_resolved_path = _baseline_path(baseline, path)
    baseline_value = _resolve(baseline, baseline_resolved_path)
    subject_valid = (
        isinstance(subject_value, int)
        and not isinstance(subject_value, bool)
        and subject_value > 0
    )
    baseline_valid = (
        isinstance(baseline_value, int)
        and not isinstance(baseline_value, bool)
        and baseline_value > 0
    )
    if not subject_valid:
        errors.append(f"Strict baseline provenance requires positive report.{path}.")
    if not baseline_valid:
        errors.append(
            "Strict baseline provenance requires positive "
            f"supplied_baseline.{baseline_resolved_path}."
        )
    if subject_valid and baseline_valid and subject_value != baseline_value:
        errors.append(
            f"Strict baseline {label} mismatch: report.{path}={subject_value!r} "
            f"supplied_baseline.{baseline_resolved_path}={baseline_value!r}."
        )


def _tokenizer_hashes(
    errors: list[str], *, payload: dict[str, Any], source: str
) -> tuple[str | None, str | None]:
    provider_hash = _nonempty_text(
        _resolve(payload, "provenance.provider_digest.tokenizer_sha256")
    )
    meta_hash = _nonempty_text(_resolve(payload, "meta.tokenizer_hash"))
    dataset_hash = _nonempty_text(_resolve(payload, "dataset.tokenizer.hash"))
    if dataset_hash is None:
        dataset_hash = _nonempty_text(_resolve(payload, "data.tokenizer_hash"))
    if provider_hash is None:
        errors.append(
            "Strict baseline provenance requires non-empty "
            f"{source}.provenance.provider_digest.tokenizer_sha256."
        )
    if meta_hash is None and dataset_hash is None:
        errors.append(
            "Strict baseline provenance requires a non-empty tokenizer hash at "
            f"{source}.meta.tokenizer_hash or {source}.dataset.tokenizer.hash."
        )
        surface_hash = None
    elif (
        meta_hash is not None and dataset_hash is not None and meta_hash != dataset_hash
    ):
        errors.append(
            f"Strict baseline tokenizer hash fork inside {source}: "
            "meta.tokenizer_hash differs from dataset.tokenizer.hash."
        )
        surface_hash = meta_hash
    else:
        surface_hash = meta_hash or dataset_hash
    if (
        provider_hash is not None
        and surface_hash is not None
        and provider_hash != surface_hash
    ):
        errors.append(
            f"Strict baseline tokenizer digest/hash fork inside {source}: "
            "provider_digest.tokenizer_sha256 differs from the report tokenizer hash."
        )
    return provider_hash, surface_hash


def _append_provenance_parity_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    for path, label in (
        ("provenance.provider_digest.ids_sha256", "provider IDs digest"),
        ("meta.adapter", "adapter identity"),
        ("dataset.provider", "dataset provider"),
        ("dataset.split", "dataset split"),
        ("dataset.hash.dataset", "dataset identity digest"),
    ):
        _append_required_text_parity(
            errors,
            subject=subject,
            baseline=baseline,
            path=path,
            label=label,
        )
    for payload, source in (
        (subject, "report"),
        (baseline, "supplied_baseline"),
    ):
        if _nonempty_text(_resolve(payload, "meta.model_id")) is None:
            errors.append(
                f"Strict baseline provenance requires non-empty {source}.meta.model_id."
            )
    _append_required_integer_parity(
        errors,
        subject=subject,
        baseline=baseline,
        path="dataset.seq_len",
        label="sequence length",
    )

    subject_provider_hash, subject_surface_hash = _tokenizer_hashes(
        errors, payload=subject, source="report"
    )
    baseline_provider_hash, baseline_surface_hash = _tokenizer_hashes(
        errors, payload=baseline, source="supplied_baseline"
    )
    if (
        subject_provider_hash is not None
        and baseline_provider_hash is not None
        and subject_provider_hash != baseline_provider_hash
    ):
        errors.append(
            "Strict baseline tokenizer digest mismatch: subject and supplied "
            "baseline provider digests differ."
        )
    if (
        subject_surface_hash is not None
        and baseline_surface_hash is not None
        and subject_surface_hash != baseline_surface_hash
    ):
        errors.append(
            "Strict baseline tokenizer hash mismatch: subject and supplied baseline "
            "report surfaces differ."
        )

    baseline_ref_hash = _nonempty_text(_resolve(subject, "baseline_ref.tokenizer_hash"))
    if baseline_ref_hash is None:
        errors.append(
            "Strict baseline provenance requires non-empty "
            "report.baseline_ref.tokenizer_hash."
        )
    elif (
        baseline_surface_hash is not None and baseline_ref_hash != baseline_surface_hash
    ):
        errors.append(
            "Strict baseline tokenizer hash mismatch: report.baseline_ref does not "
            "match the supplied baseline."
        )

    append_strict_dataset_identity_errors(
        errors,
        subject=subject,
        baseline=baseline,
    )


def _normalized_text_at(payload: dict[str, Any], *paths: str) -> str | None:
    for path in paths:
        value = _nonempty_text(_resolve(payload, path))
        if value is not None:
            return value.lower()
    return None


def _append_context_binding_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Bind the baseline run's strict execution context to the subject report."""

    expected_profile = _normalized_text_at(subject, "assurance.profile")
    baseline_profile = _normalized_text_at(baseline, "context.profile")
    if baseline_profile is None:
        errors.append(
            "Strict baseline provenance requires non-empty "
            "supplied_baseline.context.profile."
        )
    elif expected_profile is None or baseline_profile != expected_profile:
        errors.append(
            "Strict baseline profile mismatch: supplied baseline execution profile "
            "does not match the subject assurance profile."
        )

    expected_tier = _normalized_text_at(subject, "assurance.tier")
    baseline_tier = _normalized_text_at(baseline, "context.auto.tier")
    if baseline_tier is None:
        errors.append(
            "Strict baseline provenance requires non-empty "
            "supplied_baseline.context.auto.tier."
        )
    elif expected_tier is None or baseline_tier != expected_tier:
        errors.append(
            "Strict baseline tier mismatch: supplied baseline execution tier does "
            "not match the subject assurance tier."
        )

    baseline_assurance = _normalized_text_at(baseline, "context.assurance.mode")
    if baseline_assurance != "strict":
        errors.append(
            "Strict verification requires supplied_baseline.context.assurance.mode="
            "'strict'."
        )


def _append_raw_arm_completeness_errors(
    errors: list[str],
    *,
    baseline: dict[str, Any],
    metric_kind: str,
) -> None:
    """Reject baseline objects with only the final decision fragment populated."""

    primary_metric = _mapping(_resolve(baseline, "metrics.primary_metric")) or {}
    for arm in ("preview", "final"):
        source = f"supplied_baseline.evaluation_windows.{arm}"
        section = _mapping(_resolve(baseline, f"evaluation_windows.{arm}"))
        if section is None:
            errors.append(f"Strict baseline requires complete raw {source} evidence.")
            continue
        id_field = "example_ids" if metric_kind == "accuracy" else "window_ids"
        ids = section.get(id_field)
        if not isinstance(ids, list) or not ids:
            errors.append(f"Strict baseline requires non-empty {source}.{id_field}.")
            continue
        declared_count = _resolve(baseline, f"data.{arm}_n")
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count <= 0
            or declared_count != len(ids)
        ):
            errors.append(
                f"Strict baseline requires data.{arm}_n to equal the raw {arm} "
                f"schedule count ({len(ids)})."
            )
        if metric_kind == "accuracy":
            raw = section.get("example_correct")
            if raw is None:
                raw = section.get("records")
            if not isinstance(raw, list) or len(raw) != len(ids):
                errors.append(
                    f"Strict accuracy baseline requires one raw correctness record "
                    f"per {source}.{id_field}."
                )
            continue
        logloss = section.get("logloss")
        token_counts = section.get("token_counts")
        if not (
            isinstance(logloss, list)
            and isinstance(token_counts, list)
            and len(logloss) == len(token_counts) == len(ids)
        ):
            errors.append(
                "Strict PPL baseline requires equal-length raw "
                f"{source}.window_ids/logloss/token_counts."
            )
        if _finite_number(primary_metric.get(arm)) is None:
            errors.append(
                f"Strict baseline requires finite metrics.primary_metric.{arm}."
            )


def _correctness_vector(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    normalized: list[int] = []
    for entry in value:
        if isinstance(entry, bool):
            normalized.append(int(entry))
        elif isinstance(entry, int | float) and not isinstance(entry, bool):
            if float(entry) not in {0.0, 1.0}:
                return None
            normalized.append(int(float(entry)))
        else:
            return None
    return tuple(normalized)


def _record_correctness(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    values: list[Any] = []
    for record in value:
        if not isinstance(record, dict) or "correct" not in record:
            return None
        values.append(record.get("correct"))
    return _correctness_vector(values)


def _canonical_sample_ids(
    errors: list[str], *, value: Any, source: str
) -> tuple[tuple[str, int | str], ...] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"Strict accuracy baseline requires non-empty {source}.")
        return None
    normalized: list[tuple[str, int | str]] = []
    for index, sample_id in enumerate(value):
        if isinstance(sample_id, bool) or not isinstance(sample_id, int | str):
            errors.append(
                f"{source}[{index}] must be a JSON integer or non-empty string."
            )
            return None
        if isinstance(sample_id, str) and not sample_id:
            errors.append(
                f"{source}[{index}] must be a JSON integer or non-empty string."
            )
            return None
        normalized.append(
            ("integer", sample_id)
            if isinstance(sample_id, int)
            else ("text", sample_id)
        )
    if len(normalized) != len(set(normalized)):
        errors.append(f"{source} contains duplicates.")
        return None
    return tuple(normalized)


def _append_accuracy_baseline_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float,
) -> None:
    baseline_pm = _mapping(baseline.get("primary_metric"))
    if baseline_pm is None:
        baseline_pm = _mapping(_resolve(baseline, "metrics.primary_metric"))
    classification = _mapping(_resolve(baseline, "metrics.classification"))
    final_counts = _mapping(
        classification.get("final") if classification is not None else None
    )
    n_correct = _nonnegative_int(
        classification.get("n_correct") if classification is not None else None
    )
    n_total = _nonnegative_int(
        classification.get("n_total") if classification is not None else None
    )
    final_correct = _nonnegative_int(
        final_counts.get("correct_total") if final_counts is not None else None
    )
    final_total = _nonnegative_int(
        final_counts.get("total") if final_counts is not None else None
    )
    if (
        classification is None
        or n_correct is None
        or n_total is None
        or n_total <= 0
        or n_correct > n_total
    ):
        errors.append(
            "Strict accuracy baseline requires valid measured "
            "supplied_baseline.metrics.classification.n_correct/n_total."
        )
        return
    if (
        classification.get("counts_source") != "measured"
        or classification.get("estimated") is not False
    ):
        errors.append(
            "Strict accuracy baseline requires measured, non-estimated "
            "classification counts."
        )
    if (final_correct, final_total) != (n_correct, n_total):
        errors.append(
            "Strict accuracy baseline final counts differ from canonical "
            "classification totals."
        )

    final_metric = (
        _finite_number(baseline_pm.get("final")) if baseline_pm is not None else None
    )
    expected_final = n_correct / n_total
    if final_metric is None or not math.isclose(
        final_metric, expected_final, rel_tol=tolerance, abs_tol=tolerance
    ):
        errors.append(
            "Supplied baseline accuracy metric/count mismatch: "
            f"recorded={final_metric!r} recomputed={expected_final:.12f}."
        )
    n_final = _nonnegative_int(
        baseline_pm.get("n_final") if baseline_pm is not None else None
    )
    if n_final != n_total:
        errors.append(
            "Strict accuracy baseline requires primary_metric.n_final to equal "
            "classification n_total."
        )

    baseline_final_windows = _mapping(_resolve(baseline, "evaluation_windows.final"))
    subject_final_windows = _mapping(_resolve(subject, "evaluation_windows.final"))
    if baseline_final_windows is None:
        errors.append(
            "Strict accuracy baseline requires supplied_baseline.evaluation_windows.final."
        )
        return
    baseline_ids = _canonical_sample_ids(
        errors,
        value=baseline_final_windows.get("example_ids"),
        source="supplied_baseline.evaluation_windows.final.example_ids",
    )
    subject_ids = _canonical_sample_ids(
        errors,
        value=(
            subject_final_windows.get("example_ids")
            if subject_final_windows is not None
            else None
        ),
        source="report.evaluation_windows.final.example_ids",
    )
    if (
        baseline_ids is not None
        and subject_ids is not None
        and baseline_ids != subject_ids
    ):
        errors.append(
            "Strict paired accuracy baseline requires identical final example_ids "
            "in the exact subject order."
        )

    raw_candidates: list[tuple[str, tuple[int, ...]]] = []
    for source, raw in (
        (
            "supplied_baseline.metrics.classification.final.example_correct",
            final_counts.get("example_correct") if final_counts is not None else None,
        ),
        (
            "supplied_baseline.evaluation_windows.final.example_correct",
            baseline_final_windows.get("example_correct"),
        ),
    ):
        if raw is None:
            continue
        parsed = _correctness_vector(raw)
        if parsed is None:
            errors.append(f"Strict accuracy baseline requires valid binary {source}.")
        else:
            raw_candidates.append((source, parsed))
    records_raw = baseline_final_windows.get("records")
    if records_raw is not None:
        parsed_records = _record_correctness(records_raw)
        if parsed_records is None:
            errors.append(
                "Strict accuracy baseline requires final records with boolean or "
                "0/1 correct fields."
            )
        else:
            raw_candidates.append(
                ("supplied_baseline.evaluation_windows.final.records", parsed_records)
            )
    if not raw_candidates:
        errors.append(
            "Strict accuracy baseline requires raw per-example correctness evidence."
        )
        return
    first_source, first_vector = raw_candidates[0]
    for source, vector in raw_candidates:
        if len(vector) != n_total or sum(vector) != n_correct:
            errors.append(f"Supplied baseline accuracy raw/count mismatch at {source}.")
        if vector != first_vector:
            errors.append(
                f"Supplied baseline accuracy raw evidence fork between {first_source} "
                f"and {source}."
            )
    if baseline_ids is not None and len(baseline_ids) != n_total:
        errors.append(
            "Supplied baseline accuracy example_ids length differs from n_total."
        )


def _append_baseline_reference_binding_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    baseline_run_id = _nonempty_text(_resolve(baseline, "meta.run_id"))
    if baseline_run_id is None:
        errors.append(
            "Strict baseline binding requires non-empty supplied_baseline.meta.run_id."
        )
    subject_run_id = _nonempty_text(subject.get("run_id"))
    if baseline_run_id is not None and baseline_run_id == subject_run_id:
        errors.append(
            "Strict assurance requires distinct subject and baseline run IDs."
        )

    baseline_ref = _mapping(subject.get("baseline_ref"))
    if baseline_ref is None:
        errors.append("Strict baseline binding requires report.baseline_ref.")
        return
    expected_report_hash = compute_report_digest(baseline)
    expected_reference_fields = {
        "run_id": baseline_run_id,
        "model_id": _nonempty_text(_resolve(baseline, "meta.model_id")),
        "adapter": _nonempty_text(_resolve(baseline, "meta.adapter")),
        "tokenizer_hash": (
            _nonempty_text(_resolve(baseline, "meta.tokenizer_hash"))
            or _nonempty_text(_resolve(baseline, "data.tokenizer_hash"))
        ),
        "report_hash": expected_report_hash,
    }
    for field, expected in expected_reference_fields.items():
        observed = _nonempty_text(baseline_ref.get(field))
        if expected is None or observed is None:
            errors.append(
                f"Strict baseline binding requires non-empty report.baseline_ref.{field}."
            )
        elif observed != expected:
            errors.append(
                f"Strict baseline_ref {field} mismatch: report={observed!r} "
                f"supplied_baseline={expected!r}."
            )

    expected_provider_digest = _mapping(
        _resolve(baseline, "provenance.provider_digest")
    )
    bound_provider_digest = _mapping(baseline_ref.get("provider_digest"))
    if expected_provider_digest is None or bound_provider_digest is None:
        errors.append(
            "Strict baseline binding requires report.baseline_ref.provider_digest."
        )
    elif bound_provider_digest != expected_provider_digest:
        errors.append(
            "Strict baseline_ref provider_digest mismatch: report binding does not "
            "identify the supplied baseline provider evidence."
        )

    subject_provenance = _mapping(subject.get("provenance")) or {}
    baseline_binding = _mapping(subject_provenance.get("baseline"))
    if baseline_binding is None:
        errors.append("Strict baseline binding requires report.provenance.baseline.")
        return
    bound_run_id = _nonempty_text(baseline_binding.get("run_id"))
    if bound_run_id != baseline_run_id:
        errors.append(
            "Strict baseline provenance run_id mismatch: report binding does not "
            "identify the supplied baseline."
        )
    bound_report_hash = _nonempty_text(baseline_binding.get("report_hash"))
    if expected_report_hash is None or bound_report_hash != expected_report_hash:
        errors.append(
            "Strict baseline provenance report_hash mismatch: report binding does "
            "not identify the supplied baseline."
        )


def _append_checkpoint_identity_binding_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Require portable typed model identities and exact reference bindings.

    These fields bind what the producer reports loading. Trusted execution or
    independent artifact retention is still required to establish producer
    honesty.
    """

    subject_meta = _mapping(subject.get("meta")) or {}
    baseline_meta = _mapping(baseline.get("meta")) or {}
    subject_ref = _mapping(subject.get("subject_ref"))
    baseline_ref = _mapping(subject.get("baseline_ref"))

    for side, meta, reference in (
        ("subject", subject_meta, subject_ref),
        ("baseline", baseline_meta, baseline_ref),
    ):
        for legacy_field in LEGACY_MODEL_IDENTITY_FIELDS:
            if legacy_field in meta:
                errors.append(
                    f"Strict {side} meta must not declare legacy {legacy_field}; "
                    "use meta.model_identity."
                )
            if reference is not None and legacy_field in reference:
                errors.append(
                    f"Strict {side}_ref must not declare legacy {legacy_field}; "
                    f"use {side}_ref.model_identity."
                )
        identity = validated_model_identity(meta.get("model_identity"))
        if identity is None:
            errors.append(
                f"Strict {side} model identity must declare one canonical remote "
                "revision or local checkpoint tree digest."
            )
            continue

        if reference is None:
            errors.append(f"Strict {side} model identity requires report.{side}_ref.")
            continue
        bound_identity = validated_model_identity(reference.get("model_identity"))
        if bound_identity != identity:
            errors.append(f"Strict {side}_ref model_identity mismatch.")


def append_strict_baseline_contract_errors(
    errors: list[str],
    *,
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    baseline_supplied: bool,
    tolerance: float,
) -> None:
    """Validate canonical shape, provenance parity, and baseline raw evidence."""

    if not baseline_supplied:
        errors.append(
            "Strict assurance verification requires a independently supplied --baseline "
            "canonical evaluation report for every primary metric."
        )
        return
    if baseline_payload is None:
        errors.append(
            "Strict assurance verification could not load the independently supplied "
            "baseline as a JSON object."
        )
        return
    if not _canonical_run_report(baseline_payload):
        errors.append(
            "Strict assurance --baseline must be the complete canonical noop "
            "baseline run report, not a metric or evaluation-report fragment."
        )

    _append_provenance_parity_errors(
        errors,
        subject=report,
        baseline=baseline_payload,
    )
    _append_context_binding_errors(
        errors,
        subject=report,
        baseline=baseline_payload,
    )
    _append_checkpoint_identity_binding_errors(
        errors,
        subject=report,
        baseline=baseline_payload,
    )
    _append_baseline_reference_binding_errors(
        errors,
        subject=report,
        baseline=baseline_payload,
    )

    subject_pm = _mapping(report.get("primary_metric"))
    baseline_pm = _mapping(_resolve(baseline_payload, "metrics.primary_metric"))
    try:
        subject_kind = normalize_metric_kind(
            subject_pm.get("kind") if subject_pm is not None else None
        )
    except (MetricKindContractError, RuntimeError, TypeError, ValueError):
        subject_kind = None
    try:
        baseline_kind = normalize_metric_kind(
            baseline_pm.get("kind") if baseline_pm is not None else None
        )
    except (MetricKindContractError, RuntimeError, TypeError, ValueError):
        baseline_kind = None
    if subject_kind is None or baseline_kind is None:
        errors.append(
            "Strict assurance baseline requires supported subject and baseline "
            "primary metric kinds."
        )
        return
    if subject_kind != baseline_kind:
        errors.append(
            "Strict assurance baseline primary metric kind differs from the subject."
        )
        return
    _append_raw_arm_completeness_errors(
        errors,
        baseline=baseline_payload,
        metric_kind=subject_kind,
    )
    if subject_kind == "accuracy":
        _append_accuracy_baseline_errors(
            errors,
            subject=report,
            baseline=baseline_payload,
            tolerance=tolerance,
        )


__all__ = ["append_strict_baseline_contract_errors"]
