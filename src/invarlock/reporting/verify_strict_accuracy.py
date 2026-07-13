"""Strict accuracy recomputation and count-reconciliation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from .verify_check_helpers_metrics import _coerce_float
from .verify_strict_schedule import _strict_finite_number


def _strict_nonnegative_int(value: Any) -> int | None:
    """Return a non-negative JSON integer without lossy float conversion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _correct_count_from_values(values: Any) -> tuple[int, int] | None:
    if not isinstance(values, list) or not values:
        return None
    correct = 0
    for value in values:
        if isinstance(value, bool):
            correct += int(value)
            continue
        if isinstance(value, int | float) and float(value) in {0.0, 1.0}:
            correct += int(float(value))
            continue
        return None
    return correct, len(values)


def _correct_count_from_records(records: Any) -> tuple[int, int] | None:
    if not isinstance(records, list) or not records:
        return None
    values: list[Any] = []
    for record in records:
        if not isinstance(record, dict) or "correct" not in record:
            return None
        values.append(record.get("correct"))
    return _correct_count_from_values(values)


def _strict_accuracy_arm_records(
    errors: list[str],
    *,
    arm: str,
    windows: dict[str, Any] | None,
    expected: tuple[int, int],
) -> tuple[tuple[str, ...], tuple[bool, ...]] | None:
    source = f"evaluation_windows.{arm}"
    if windows is None:
        errors.append(f"Strict accuracy evidence requires {source} as an object.")
        return None
    records = windows.get("records")
    example_ids = windows.get("example_ids")
    if not isinstance(records, list) or not records:
        errors.append(
            f"Strict accuracy evidence requires {source}.records as a non-empty list."
        )
        return None
    if not isinstance(example_ids, list) or not example_ids:
        errors.append(
            f"Strict accuracy evidence requires {source}.example_ids as a non-empty list."
        )
        return None
    if len(records) != len(example_ids) or len(records) != expected[1]:
        errors.append(
            f"Strict accuracy evidence requires {source} records/example_ids to "
            f"match the expected count {expected[1]}."
        )
        return None

    normalized_ids: list[str] = []
    correctness: list[bool] = []
    for index, (raw_id, record) in enumerate(zip(example_ids, records, strict=True)):
        example_id = str(raw_id).strip() if not isinstance(raw_id, bool) else ""
        if not example_id:
            errors.append(f"{source}.example_ids[{index}] must be a non-empty ID.")
            continue
        if not isinstance(record, dict):
            errors.append(f"{source}.records[{index}] must be an object.")
            continue
        record_id_raw = record.get("id", record.get("example_id"))
        record_id = (
            str(record_id_raw).strip()
            if record_id_raw is not None and not isinstance(record_id_raw, bool)
            else ""
        )
        if record_id != example_id:
            errors.append(
                f"{source}.records[{index}] ID must match example_ids[{index}]."
            )
        correct = record.get("correct")
        if not isinstance(correct, bool):
            errors.append(f"{source}.records[{index}].correct must be a boolean.")
            continue
        normalized_ids.append(example_id)
        correctness.append(correct)
    if len(normalized_ids) != expected[1] or len(correctness) != expected[1]:
        return None
    if len(set(normalized_ids)) != len(normalized_ids):
        errors.append(f"{source}.example_ids must not contain duplicates.")
    observed = (sum(correctness), len(correctness))
    if observed != expected:
        errors.append(
            f"Accuracy count mismatch: {source}.records={observed[0]}/{observed[1]} "
            f"expected={expected[0]}/{expected[1]}."
        )
    input_records = windows.get("input_records")
    if not isinstance(input_records, list) or len(input_records) != len(normalized_ids):
        errors.append(
            f"Strict accuracy evidence requires {source}.input_records in exact record order."
        )
    else:
        for index, (example_id, input_record) in enumerate(
            zip(normalized_ids, input_records, strict=True)
        ):
            input_id_raw = (
                input_record.get("id", input_record.get("example_id"))
                if isinstance(input_record, dict)
                else None
            )
            input_id = (
                str(input_id_raw).strip()
                if input_id_raw is not None and not isinstance(input_id_raw, bool)
                else ""
            )
            if input_id != example_id:
                errors.append(
                    f"{source}.input_records[{index}] ID must match example_ids[{index}]."
                )
    return tuple(normalized_ids), tuple(correctness)


def _provider_ids_digest(example_ids: tuple[str, ...]) -> str:
    values: list[int] | list[str]
    try:
        values = sorted(int(value) for value in example_ids)
    except (TypeError, ValueError):
        values = sorted(example_ids)
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=32).hexdigest()


def _append_count_pair_mismatch(
    errors: list[str],
    *,
    source: str,
    observed: tuple[int, int] | None,
    expected: tuple[int, int],
) -> None:
    if observed is None:
        return
    if observed != expected:
        errors.append(
            f"Accuracy count mismatch: {source}={observed[0]}/{observed[1]} "
            f"expected={expected[0]}/{expected[1]}."
        )


def _append_optional_count_pair(
    errors: list[str],
    *,
    container: dict[str, Any],
    key: str,
    source: str,
    parser: Callable[[Any], tuple[int, int] | None],
    expected: tuple[int, int],
    expected_shape: str,
) -> None:
    """Validate an optional raw-count surface when the producer includes it."""

    if key not in container:
        return
    observed = parser(container.get(key))
    if observed is None:
        errors.append(
            f"Malformed accuracy evidence: {source} must be {expected_shape}."
        )
        return
    _append_count_pair_mismatch(
        errors,
        source=source,
        observed=observed,
        expected=expected,
    )


def _classification_arm_pair(
    errors: list[str],
    *,
    classification: dict[str, Any],
    arm: str,
) -> tuple[int, int] | None:
    block = classification.get(arm)
    if not isinstance(block, dict):
        errors.append(
            f"Strict accuracy evidence requires metrics.classification.{arm} "
            "as an object."
        )
        return None
    correct = _strict_nonnegative_int(block.get("correct_total"))
    total = _strict_nonnegative_int(block.get("total"))
    if correct is None or total is None or total <= 0:
        errors.append(
            f"Strict accuracy evidence requires integer "
            f"metrics.classification.{arm}.correct_total/total with total > 0."
        )
        return None
    if correct > total:
        errors.append(
            f"Accuracy count mismatch: metrics.classification.{arm}.correct_total "
            "exceeds total."
        )
    return correct, total


def _append_accuracy_arm_evidence(
    errors: list[str],
    *,
    arm: str,
    nested: dict[str, Any] | None,
    windows: dict[str, Any] | None,
    expected: tuple[int, int],
) -> None:
    if nested is not None:
        _append_optional_count_pair(
            errors,
            container=nested,
            key="example_correct",
            source=f"metrics.classification.{arm}.example_correct",
            parser=_correct_count_from_values,
            expected=expected,
            expected_shape="a non-empty list containing only booleans or 0/1 values",
        )
        _append_optional_count_pair(
            errors,
            container=nested,
            key="records",
            source=f"metrics.classification.{arm}.records",
            parser=_correct_count_from_records,
            expected=expected,
            expected_shape=(
                "a non-empty list of objects with boolean or 0/1 'correct' values"
            ),
        )

    if windows is None:
        return
    _append_optional_count_pair(
        errors,
        container=windows,
        key="example_correct",
        source=f"evaluation_windows.{arm}.example_correct",
        parser=_correct_count_from_values,
        expected=expected,
        expected_shape="a non-empty list containing only booleans or 0/1 values",
    )
    _append_optional_count_pair(
        errors,
        container=windows,
        key="records",
        source=f"evaluation_windows.{arm}.records",
        parser=_correct_count_from_records,
        expected=expected,
        expected_shape=(
            "a non-empty list of objects with boolean or 0/1 'correct' values"
        ),
    )
    for key in ("input_records", "example_ids", "input_ids", "labels"):
        if key not in windows:
            continue
        values = windows.get(key)
        if not isinstance(values, list):
            errors.append(
                f"Malformed accuracy evidence: evaluation_windows.{arm}.{key} "
                "must be a list."
            )
        elif len(values) != expected[1]:
            errors.append(
                f"Accuracy count mismatch: evaluation_windows.{arm}.{key} has "
                f"{len(values)} entries; expected {expected[1]}."
            )


def _accuracy_count_context(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    pm: dict[str, Any],
    require_strict: bool,
) -> tuple[
    dict[str, Any],
    int | None,
    int | None,
    int | None,
    int | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]:
    metrics = cert_obj.get("metrics")
    cls_value = metrics.get("classification") if isinstance(metrics, dict) else None
    cls = cls_value if isinstance(cls_value, dict) else {}
    n_correct = _strict_nonnegative_int(cls.get("n_correct"))
    n_total = _strict_nonnegative_int(cls.get("n_total"))
    n_preview = _strict_nonnegative_int(pm.get("n_preview"))
    n_final = _strict_nonnegative_int(pm.get("n_final"))
    if not require_strict:
        return cls, n_correct, n_total, n_preview, n_final, None, None

    preview_pair = _classification_arm_pair(errors, classification=cls, arm="preview")
    final_pair = _classification_arm_pair(errors, classification=cls, arm="final")
    if n_preview is None or n_preview <= 0:
        errors.append(
            "Strict accuracy evidence requires a positive integer "
            "primary_metric.n_preview."
        )
    if n_final is None or n_final <= 0:
        errors.append(
            "Strict accuracy evidence requires a positive integer "
            "primary_metric.n_final."
        )
    if n_correct is None or n_total is None or n_total <= 0:
        errors.append(
            "Strict accuracy evidence requires integer "
            "metrics.classification.n_correct/n_total with n_total > 0."
        )
    if pm.get("counts_source") != "measured" or pm.get("estimated") is not False:
        errors.append(
            "Strict accuracy evidence requires primary_metric counts_source=measured "
            "and estimated=false."
        )
    if cls.get("counts_source") != "measured" or cls.get("estimated") is not False:
        errors.append(
            "Strict accuracy evidence requires metrics.classification "
            "counts_source=measured and estimated=false."
        )
    return cls, n_correct, n_total, n_preview, n_final, preview_pair, final_pair


def _append_accuracy_count_errors(
    errors: list[str],
    *,
    n_correct: int,
    n_total: int,
    n_preview: int | None,
    n_final: int | None,
    preview_pair: tuple[int, int] | None,
    final_pair: tuple[int, int] | None,
    require_strict: bool,
) -> None:
    if n_correct > n_total:
        errors.append(
            "Accuracy count mismatch: metrics.classification.n_correct exceeds n_total."
        )
    if require_strict and n_final is not None and n_final != n_total:
        errors.append(
            "Accuracy count mismatch: primary_metric.n_final="
            f"{n_final} differs from metrics.classification.n_total={n_total}."
        )
    if require_strict and final_pair is not None:
        _append_count_pair_mismatch(
            errors,
            source="metrics.classification.final",
            observed=final_pair,
            expected=(n_correct, n_total),
        )
    if not require_strict or preview_pair is None or n_preview is None:
        return
    if n_preview != preview_pair[1]:
        errors.append(
            "Accuracy count mismatch: primary_metric.n_preview="
            f"{n_preview} differs from metrics.classification.preview.total="
            f"{preview_pair[1]}."
        )
    if preview_pair[1] != n_total:
        errors.append(
            "Strict paired accuracy evidence requires equal preview and final "
            "example counts."
        )


def _append_accuracy_value_errors(
    errors: list[str],
    *,
    pm: dict[str, Any],
    expected: tuple[int, int],
    preview_pair: tuple[int, int] | None,
    tol: float,
    require_strict: bool,
) -> float | None:
    final_value = _coerce_float(pm.get("final"))
    if final_value is None:
        if require_strict:
            errors.append("Accuracy mismatch: primary_metric.final must be finite.")
    else:
        recomputed = float(expected[0]) / float(expected[1])
        if abs(final_value - recomputed) > max(1e-12, tol):
            errors.append(
                f"Accuracy mismatch: final={final_value:.12f} recomputed={recomputed:.12f}"
            )

    if require_strict and preview_pair is not None:
        preview_value = _coerce_float(pm.get("preview"))
        recomputed_preview = float(preview_pair[0]) / float(preview_pair[1])
        if preview_value is None:
            errors.append("Accuracy mismatch: primary_metric.preview must be finite.")
        elif abs(preview_value - recomputed_preview) > max(1e-12, tol):
            errors.append(
                "Accuracy mismatch: "
                f"preview={preview_value:.12f} recomputed={recomputed_preview:.12f}"
            )
    if "ratio_vs_baseline" in pm:
        errors.append(
            "Accuracy evidence forbids primary_metric.ratio_vs_baseline; "
            "use delta_vs_baseline_pp."
        )
    return final_value


def _append_accuracy_baseline_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    pm: dict[str, Any],
    final_value: float,
    tol: float,
) -> None:
    baseline_ref = cert_obj.get("baseline_ref")
    baseline_pm = (
        baseline_ref.get("primary_metric") if isinstance(baseline_ref, dict) else None
    )
    baseline_kind = (
        str(baseline_pm.get("kind") or "").strip().lower()
        if isinstance(baseline_pm, dict)
        else ""
    )
    baseline_final = (
        _strict_finite_number(baseline_pm.get("final"))
        if isinstance(baseline_pm, dict)
        else None
    )
    if (
        baseline_kind != "accuracy"
        or baseline_final is None
        or not 0.0 <= baseline_final <= 1.0
    ):
        errors.append(
            "Strict accuracy evidence requires "
            "baseline_ref.primary_metric.kind=accuracy and finite final in [0,1]."
        )
        return
    expected_delta_pp = 100.0 * (final_value - baseline_final)
    delta_pp = _strict_finite_number(pm.get("delta_vs_baseline_pp"))
    if delta_pp is None:
        errors.append(
            "Strict accuracy evidence requires finite "
            "primary_metric.delta_vs_baseline_pp."
        )
    elif abs(delta_pp - expected_delta_pp) > max(1e-12, tol):
        errors.append(
            "Accuracy baseline delta mismatch: "
            f"recorded={delta_pp:.12f} expected={expected_delta_pp:.12f}"
        )


def _append_dataset_window_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    preview_total: int | None,
    final_total: int,
) -> None:
    dataset = cert_obj.get("dataset")
    dataset_windows = dataset.get("windows") if isinstance(dataset, dict) else None
    if (
        isinstance(dataset, dict)
        and "windows" in dataset
        and not isinstance(dataset_windows, dict)
    ):
        errors.append("Malformed accuracy evidence: dataset.windows must be an object.")
    if isinstance(dataset_windows, dict):
        expected_totals = {
            "preview": preview_total,
            "final": final_total,
        }
        for arm, expected_total in expected_totals.items():
            direct_count = _strict_nonnegative_int(dataset_windows.get(arm))
            if arm not in dataset_windows:
                errors.append(
                    f"Strict accuracy evidence requires dataset.windows.{arm}."
                )
            elif direct_count is None:
                errors.append(
                    f"Malformed accuracy evidence: dataset.windows.{arm} must be a "
                    "non-negative integer."
                )
            elif expected_total is not None and direct_count != expected_total:
                errors.append(
                    f"Accuracy count mismatch: dataset.windows.{arm}={direct_count} "
                    f"differs from expected total={expected_total}."
                )
        stats = dataset_windows.get("stats")
        if "stats" in dataset_windows and not isinstance(stats, dict):
            errors.append(
                "Malformed accuracy evidence: dataset.windows.stats must be an object."
            )
        if isinstance(stats, dict):
            expected_stat_totals = {
                "actual_preview": expected_totals["preview"],
                "actual_final": expected_totals["final"],
                "paired_windows": expected_totals["final"],
            }
            for key, expected_total in expected_stat_totals.items():
                value = _strict_nonnegative_int(stats.get(key))
                if key not in stats:
                    errors.append(
                        f"Strict accuracy evidence requires dataset.windows.stats.{key}."
                    )
                    continue
                if value is None:
                    errors.append(
                        f"Malformed accuracy evidence: dataset.windows.stats.{key} "
                        "must be a non-negative integer."
                    )
                elif expected_total is not None and value != expected_total:
                    errors.append(
                        f"Accuracy count mismatch: dataset.windows.stats.{key}={value} "
                        f"differs from expected total={expected_total}."
                    )
            coverage = stats.get("coverage")
            if "coverage" in stats and not isinstance(coverage, dict):
                errors.append(
                    "Malformed accuracy evidence: dataset.windows.stats.coverage "
                    "must be an object."
                )
            for arm, expected_total in expected_totals.items():
                arm_coverage = coverage.get(arm) if isinstance(coverage, dict) else None
                if not isinstance(arm_coverage, dict):
                    errors.append(
                        f"Strict accuracy evidence requires "
                        f"dataset.windows.stats.coverage.{arm} as an object."
                    )
                    continue
                used = _strict_nonnegative_int(arm_coverage.get("used"))
                if used is None:
                    errors.append(
                        f"Malformed accuracy evidence: dataset.windows.stats.coverage."
                        f"{arm}.used must be a non-negative integer."
                    )
                elif expected_total is not None and used != expected_total:
                    errors.append(
                        f"Accuracy count mismatch: dataset.windows.stats.coverage."
                        f"{arm}.used={used} differs from expected total={expected_total}."
                    )


def _append_strict_accuracy_surface_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    cls: dict[str, Any],
    preview_pair: tuple[int, int] | None,
    expected: tuple[int, int],
) -> None:
    nested_preview = (
        cls.get("preview") if isinstance(cls.get("preview"), dict) else None
    )
    nested_final = cls.get("final") if isinstance(cls.get("final"), dict) else None
    evaluation_windows = cert_obj.get("evaluation_windows")
    preview_windows = (
        evaluation_windows.get("preview")
        if isinstance(evaluation_windows, dict)
        and isinstance(evaluation_windows.get("preview"), dict)
        else None
    )
    final_windows = (
        evaluation_windows.get("final")
        if isinstance(evaluation_windows, dict)
        and isinstance(evaluation_windows.get("final"), dict)
        else None
    )
    if isinstance(evaluation_windows, dict):
        for arm in ("preview", "final"):
            if arm in evaluation_windows and not isinstance(
                evaluation_windows.get(arm), dict
            ):
                errors.append(
                    f"Malformed accuracy evidence: evaluation_windows.{arm} "
                    "must be an object."
                )
    if preview_pair is not None:
        preview_raw = _strict_accuracy_arm_records(
            errors, arm="preview", windows=preview_windows, expected=preview_pair
        )
        _append_accuracy_arm_evidence(
            errors,
            arm="preview",
            nested=nested_preview,
            windows=preview_windows,
            expected=preview_pair,
        )
    else:
        preview_raw = None
    final_raw = _strict_accuracy_arm_records(
        errors, arm="final", windows=final_windows, expected=expected
    )
    _append_accuracy_arm_evidence(
        errors,
        arm="final",
        nested=nested_final,
        windows=final_windows,
        expected=expected,
    )
    if preview_raw is not None and final_raw is not None:
        preview_ids, _preview_correctness = preview_raw
        final_ids, _final_correctness = final_raw
        if set(preview_ids) & set(final_ids):
            errors.append("Strict accuracy preview/final example_ids must be disjoint.")
        provenance = cert_obj.get("provenance")
        provider_digest = (
            provenance.get("provider_digest") if isinstance(provenance, dict) else None
        )
        observed_ids_digest = (
            provider_digest.get("ids_sha256")
            if isinstance(provider_digest, dict)
            else None
        )
        expected_ids_digest = _provider_ids_digest((*preview_ids, *final_ids))
        if observed_ids_digest != expected_ids_digest:
            errors.append(
                "Strict accuracy example IDs disagree with "
                "provenance.provider_digest.ids_sha256."
            )
    _append_dataset_window_errors(
        errors,
        cert_obj=cert_obj,
        preview_total=preview_pair[1] if preview_pair is not None else None,
        final_total=expected[1],
    )


def _append_accuracy_recompute_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    pm: dict[str, Any],
    tol: float,
    require_strict: bool,
) -> bool:
    """Cross-reconcile every available accuracy count/evidence surface.

    Returns ``True`` when the canonical top-level classification counts were
    usable.  Strict assurance requires those counts, ``primary_metric.n_final``,
    and measured (non-estimated) evidence.  Optional nested records, window
    arrays, and coverage counters are reconciled whenever present.
    """

    (
        cls,
        n_correct,
        n_total,
        n_preview,
        n_final,
        preview_pair,
        final_pair,
    ) = _accuracy_count_context(
        errors, cert_obj=cert_obj, pm=pm, require_strict=require_strict
    )
    if n_correct is None or n_total is None or n_total <= 0:
        return False
    _append_accuracy_count_errors(
        errors,
        n_correct=n_correct,
        n_total=n_total,
        n_preview=n_preview,
        n_final=n_final,
        preview_pair=preview_pair,
        final_pair=final_pair,
        require_strict=require_strict,
    )
    expected = (n_correct, n_total)
    final_value = _append_accuracy_value_errors(
        errors,
        pm=pm,
        expected=expected,
        preview_pair=preview_pair,
        tol=tol,
        require_strict=require_strict,
    )
    if require_strict and final_value is not None:
        _append_accuracy_baseline_errors(
            errors, cert_obj=cert_obj, pm=pm, final_value=final_value, tol=tol
        )
    if not require_strict:
        return True
    _append_strict_accuracy_surface_errors(
        errors,
        cert_obj=cert_obj,
        cls=cls,
        preview_pair=preview_pair,
        expected=expected,
    )
    return True
