"""Strict PPL recomputation, schedule, and independent-slice checks."""

from __future__ import annotations

import math
from typing import Any

from invarlock.core.bootstrap import (
    INDEPENDENT_SLICE_BOOTSTRAP_METHOD,
    INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET,
    compute_independent_delta_log_ci,
)

from .verify_strict_schedule import (
    _schedule_digest,
    _schedule_window_id_key,
    _strict_finite_number,
)


def _strict_nonnegative_int(value: Any) -> int | None:
    """Return a non-negative JSON integer without lossy float conversion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _append_ppl_arm_recompute_errors(
    errors: list[str],
    *,
    arm: str,
    section: Any,
    primary_metric: dict[str, Any],
    tolerance: float,
    require_analysis_point: bool,
    require_window_ids: bool = False,
) -> float | None:
    if not isinstance(section, dict):
        errors.append(f"Strict PPL evidence requires evaluation_windows.{arm}.")
        return None
    logloss = section.get("logloss")
    token_counts = section.get("token_counts")
    if not (
        isinstance(logloss, list)
        and isinstance(token_counts, list)
        and logloss
        and len(logloss) == len(token_counts)
    ):
        errors.append(
            f"Strict PPL evidence requires non-empty equal-length "
            f"evaluation_windows.{arm}.logloss/token_counts lists."
        )
        return None

    normalized_logloss: list[float] = []
    normalized_counts: list[int] = []
    for index, (loss_raw, count_raw) in enumerate(
        zip(logloss, token_counts, strict=True)
    ):
        loss = _strict_finite_number(loss_raw)
        count = (
            count_raw
            if isinstance(count_raw, int)
            and not isinstance(count_raw, bool)
            and count_raw > 0
            else None
        )
        if loss is None:
            errors.append(f"evaluation_windows.{arm}.logloss[{index}] must be finite.")
        elif loss < 0.0:
            errors.append(
                f"evaluation_windows.{arm}.logloss[{index}] must be non-negative."
            )
            loss = None
        if count is None:
            errors.append(
                f"evaluation_windows.{arm}.token_counts[{index}] must be a positive "
                "JSON integer."
            )
        if loss is None or count is None:
            continue
        normalized_logloss.append(loss)
        normalized_counts.append(count)
    if len(normalized_logloss) != len(logloss):
        return None

    window_ids = section.get("window_ids")
    if not isinstance(window_ids, list):
        if require_window_ids:
            errors.append(
                f"Strict PPL evidence requires evaluation_windows.{arm}.window_ids "
                "as a non-empty list."
            )
        elif "window_ids" in section:
            errors.append(f"evaluation_windows.{arm}.window_ids must be a list.")
    elif isinstance(window_ids, list):
        if require_window_ids and not window_ids:
            errors.append(
                f"Strict PPL evidence requires evaluation_windows.{arm}.window_ids "
                "as a non-empty list."
            )
        for index, window_id in enumerate(window_ids):
            if isinstance(window_id, bool) or not isinstance(window_id, int | str):
                errors.append(
                    f"evaluation_windows.{arm}.window_ids[{index}] must be a "
                    "JSON integer or non-empty string."
                )
            elif isinstance(window_id, str) and not window_id:
                errors.append(
                    f"evaluation_windows.{arm}.window_ids[{index}] must be a "
                    "JSON integer or non-empty string."
                )
        if len(window_ids) != len(logloss):
            errors.append(
                f"evaluation_windows.{arm}.window_ids length differs from "
                "logloss/token_counts."
            )
        if len(window_ids) != len(
            {_schedule_window_id_key(item) for item in window_ids}
        ):
            errors.append(f"evaluation_windows.{arm}.window_ids contains duplicates.")

    denominator = sum(normalized_counts)
    if denominator <= 0:
        errors.append(f"evaluation_windows.{arm} has no positive token weight.")
        return None
    mean_logloss = math.fsum(
        loss * (count / denominator)
        for loss, count in zip(normalized_logloss, normalized_counts, strict=True)
    )
    if not math.isfinite(mean_logloss):
        errors.append(
            f"evaluation_windows.{arm} recomputed mean log-loss is non-finite."
        )
        return None

    analysis_key = f"analysis_point_{arm}"
    analysis_point = _strict_finite_number(primary_metric.get(analysis_key))
    if require_analysis_point and analysis_point is None:
        errors.append(
            f"Strict PPL evidence requires finite primary_metric.{analysis_key}."
        )
    elif analysis_point is not None and abs(analysis_point - mean_logloss) > tolerance:
        errors.append(
            f"Basis mismatch: {analysis_key}={analysis_point:.12f} "
            f"recomputed={mean_logloss:.12f}"
        )

    display_point = _strict_finite_number(primary_metric.get(arm))
    try:
        expected_display = math.exp(mean_logloss)
    except OverflowError:
        errors.append(
            f"evaluation_windows.{arm} recomputed perplexity overflows finite range."
        )
        return None
    if not math.isfinite(expected_display) or expected_display <= 0.0:
        errors.append(
            f"evaluation_windows.{arm} recomputed perplexity is outside the finite "
            "positive range."
        )
        return None
    if display_point is None or display_point <= 0.0:
        errors.append(f"primary_metric.{arm} must be finite and > 0 for PPL.")
    elif not math.isclose(
        display_point,
        expected_display,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        errors.append(
            f"Display mismatch: {arm}={display_point:.12f} "
            f"exp(basis)={expected_display:.12f}"
        )
    return mean_logloss


def _append_declared_count_mismatch(
    errors: list[str],
    *,
    container: Any,
    key: str,
    source: str,
    expected: int,
) -> None:
    if not isinstance(container, dict) or key not in container:
        errors.append(f"Strict PPL evidence requires {source}.")
        return
    observed = _strict_nonnegative_int(container.get(key))
    if observed is None:
        errors.append(f"{source} must be a non-negative JSON integer.")
    elif observed != expected:
        errors.append(f"PPL count mismatch: {source}={observed} expected={expected}.")


def _append_strict_ppl_schedule_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
) -> None:
    evaluation_windows = cert_obj.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        return

    arm_lengths: dict[str, int] = {}
    arm_ids: dict[str, list[Any]] = {}
    for arm in ("preview", "final"):
        section = evaluation_windows.get(arm)
        if not isinstance(section, dict):
            continue
        logloss = section.get("logloss")
        if isinstance(logloss, list):
            arm_lengths[arm] = len(logloss)
        window_ids = section.get("window_ids")
        if isinstance(window_ids, list):
            arm_ids[arm] = window_ids

    dataset = cert_obj.get("dataset")
    dataset_windows = dataset.get("windows") if isinstance(dataset, dict) else None
    if not isinstance(dataset_windows, dict):
        errors.append("Strict PPL evidence requires dataset.windows as an object.")
        return
    stats = dataset_windows.get("stats")
    if not isinstance(stats, dict):
        errors.append(
            "Strict PPL evidence requires dataset.windows.stats as an object."
        )
        return
    coverage = stats.get("coverage")
    if not isinstance(coverage, dict):
        errors.append(
            "Strict PPL evidence requires dataset.windows.stats.coverage as an object."
        )
        coverage = {}

    for arm, expected in arm_lengths.items():
        _append_declared_count_mismatch(
            errors,
            container=dataset_windows,
            key=arm,
            source=f"dataset.windows.{arm}",
            expected=expected,
        )
        _append_declared_count_mismatch(
            errors,
            container=stats,
            key=f"actual_{arm}",
            source=f"dataset.windows.stats.actual_{arm}",
            expected=expected,
        )
        arm_coverage = coverage.get(arm)
        _append_declared_count_mismatch(
            errors,
            container=arm_coverage,
            key="used",
            source=f"dataset.windows.stats.coverage.{arm}.used",
            expected=expected,
        )

    preview_length = arm_lengths.get("preview")
    final_length = arm_lengths.get("final")
    if preview_length is not None and final_length is not None:
        if preview_length != final_length:
            errors.append(
                "Strict PPL schedule policy requires equal preview and final raw "
                "window counts; these slices remain statistically independent."
            )
        _append_declared_count_mismatch(
            errors,
            container=stats,
            key="paired_windows",
            source="dataset.windows.stats.paired_windows",
            expected=final_length,
        )

    bootstrap = stats.get("bootstrap")
    bootstrap_coverage = (
        bootstrap.get("coverage") if isinstance(bootstrap, dict) else None
    )
    if isinstance(bootstrap_coverage, dict):
        for arm, expected in arm_lengths.items():
            _append_declared_count_mismatch(
                errors,
                container=bootstrap_coverage.get(arm),
                key="used",
                source=f"dataset.windows.stats.bootstrap.coverage.{arm}.used",
                expected=expected,
            )

    preview_ids = arm_ids.get("preview")
    final_ids = arm_ids.get("final")
    if isinstance(preview_ids, list) and isinstance(final_ids, list):
        preview_keys = {_schedule_window_id_key(value) for value in preview_ids}
        final_keys = {_schedule_window_id_key(value) for value in final_ids}
        if preview_keys.intersection(final_keys):
            errors.append("Strict PPL preview/final window_ids must be disjoint.")

    if not isinstance(final_ids, list) or not final_ids:
        return
    expected_digest = _schedule_digest(final_ids)
    provenance = cert_obj.get("provenance")
    guard_metric_impact = cert_obj.get("guard_metric_impact")
    digest_fields = (
        (provenance, "window_ids_digest", "provenance.window_ids_digest"),
        (provenance, "window_plan_digest", "provenance.window_plan_digest"),
        (guard_metric_impact, "schedule_digest", "guard_metric_impact.schedule_digest"),
    )
    for container, key, source in digest_fields:
        value = container.get(key) if isinstance(container, dict) else None
        if not isinstance(value, str) or not value:
            errors.append(f"Strict PPL evidence requires {source}.")
        elif value != expected_digest:
            errors.append(
                f"PPL schedule digest differs: {source}={value} "
                f"expected={expected_digest}."
            )


def _append_strict_preview_final_slice_summary_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    preview_mean: float,
    final_mean: float,
    tolerance: float,
) -> None:
    dataset = cert_obj.get("dataset")
    windows = dataset.get("windows") if isinstance(dataset, dict) else None
    stats = windows.get("stats") if isinstance(windows, dict) else None
    if not isinstance(stats, dict):
        return
    if "paired_delta_summary" in stats:
        errors.append(
            "Strict PPL evidence rejects legacy paired_delta_summary for disjoint "
            "preview/final slices."
        )
    summary = stats.get("preview_final_slice_delta_summary")
    if not isinstance(summary, dict):
        errors.append(
            "Strict PPL evidence requires dataset.windows.stats."
            "preview_final_slice_delta_summary."
        )
        return

    if summary.get("basis") != "independent_disjoint_slices":
        errors.append(
            "preview_final_slice_delta_summary.basis must be "
            "independent_disjoint_slices."
        )
    if summary.get("paired") is not False:
        errors.append("preview_final_slice_delta_summary.paired must be false.")
    if summary.get("ci_method") != INDEPENDENT_SLICE_BOOTSTRAP_METHOD:
        errors.append(
            "preview_final_slice_delta_summary.ci_method must be "
            f"{INDEPENDENT_SLICE_BOOTSTRAP_METHOD}."
        )
    if summary.get("ci_reason") is not None:
        errors.append(
            "Strict PPL preview/final slice CI must not record a fallback reason."
        )

    expected_mean = final_mean - preview_mean
    reported_mean = _strict_finite_number(summary.get("mean"))
    if reported_mean is None or not math.isclose(
        reported_mean,
        expected_mean,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        errors.append(
            "preview_final_slice_delta_summary.mean does not match the recomputed "
            "independent-slice log-loss difference."
        )

    evaluation_windows = cert_obj.get("evaluation_windows")
    preview = (
        evaluation_windows.get("preview")
        if isinstance(evaluation_windows, dict)
        else None
    )
    final = (
        evaluation_windows.get("final")
        if isinstance(evaluation_windows, dict)
        else None
    )
    preview_losses = preview.get("logloss") if isinstance(preview, dict) else None
    final_losses = final.get("logloss") if isinstance(final, dict) else None
    preview_weights = preview.get("token_counts") if isinstance(preview, dict) else None
    final_weights = final.get("token_counts") if isinstance(final, dict) else None
    if not (
        isinstance(preview_losses, list)
        and preview_losses
        and isinstance(final_losses, list)
        and final_losses
        and isinstance(preview_weights, list)
        and len(preview_weights) == len(preview_losses)
        and isinstance(final_weights, list)
        and len(final_weights) == len(final_losses)
    ):
        return

    for key, expected in (
        ("preview_windows", len(preview_losses)),
        ("final_windows", len(final_losses)),
    ):
        observed_count = _strict_nonnegative_int(summary.get(key))
        if observed_count != expected:
            errors.append(
                f"preview_final_slice_delta_summary.{key} must equal {expected}."
            )

    bootstrap = stats.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return
    replicates = _strict_nonnegative_int(bootstrap.get("replicates"))
    alpha = _strict_finite_number(bootstrap.get("alpha"))
    seed = bootstrap.get("seed")
    if (
        replicates is None
        or replicates <= 0
        or alpha is None
        or not 0.0 < alpha < 1.0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        return
    expected_seed = seed + INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET
    if bootstrap.get("preview_final_delta_basis") != "independent_disjoint_slices":
        errors.append(
            "dataset.windows.stats.bootstrap.preview_final_delta_basis must be "
            "independent_disjoint_slices."
        )
    if bootstrap.get("preview_final_delta_method") != (
        INDEPENDENT_SLICE_BOOTSTRAP_METHOD
    ):
        errors.append(
            "dataset.windows.stats.bootstrap.preview_final_delta_method is invalid."
        )
    if bootstrap.get("preview_final_delta_seed") != expected_seed:
        errors.append(
            "dataset.windows.stats.bootstrap.preview_final_delta_seed does not "
            "match the producer seed derivation."
        )

    ci = summary.get("ci")
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        errors.append("preview_final_slice_delta_summary.ci must contain two bounds.")
        return
    observed_ci = tuple(_strict_finite_number(value) for value in ci)
    if None in observed_ci:
        errors.append(
            "preview_final_slice_delta_summary.ci bounds must be finite numbers."
        )
        return
    try:
        expected_ci = compute_independent_delta_log_ci(
            final_losses,
            preview_losses,
            final_weights=final_weights,
            preview_weights=preview_weights,
            method="percentile",
            replicates=replicates,
            alpha=alpha,
            seed=expected_seed,
        )
    except ValueError as exc:
        errors.append(f"Independent preview/final slice CI replay failed: {exc}")
        return
    expected_degenerate = math.isclose(
        expected_ci[0],
        expected_ci[1],
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
    if summary.get("degenerate") is not expected_degenerate:
        errors.append(
            "preview_final_slice_delta_summary.degenerate does not match the "
            "replayed independent bootstrap distribution."
        )
    expected_degenerate_reason = (
        "constant_bootstrap_distribution" if expected_degenerate else None
    )
    if summary.get("degenerate_reason") != expected_degenerate_reason:
        errors.append(
            "preview_final_slice_delta_summary.degenerate_reason does not match "
            "the replayed independent bootstrap distribution."
        )
    for observed_bound, expected_bound in zip(observed_ci, expected_ci, strict=True):
        assert observed_bound is not None
        if not math.isclose(
            observed_bound,
            expected_bound,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "preview_final_slice_delta_summary.ci does not match independent "
                "two-slice bootstrap replay."
            )
            break


def _append_strict_ppl_coherence_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    primary_metric: dict[str, Any],
    preview_mean: float | None,
    final_mean: float | None,
    tolerance: float,
) -> None:
    if primary_metric.get("analysis_basis") != "mean_logloss":
        errors.append("Strict PPL evidence requires analysis_basis=mean_logloss.")

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
        not baseline_kind.startswith("ppl")
        or baseline_final is None
        or baseline_final < 1.0
    ):
        errors.append(
            "Strict PPL evidence requires a same-family baseline perplexity >= 1 "
            "in baseline_ref.primary_metric.final."
        )
        expected_ratio = None
    elif final_mean is None:
        expected_ratio = None
    else:
        expected_ratio = math.exp(final_mean) / baseline_final
        recorded_ratio = _strict_finite_number(primary_metric.get("ratio_vs_baseline"))
        if recorded_ratio is None or recorded_ratio <= 0.0:
            errors.append("primary_metric.ratio_vs_baseline must be finite and > 0.")
        elif not math.isclose(
            recorded_ratio,
            expected_ratio,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "Primary metric ratio mismatch against recomputed final and baseline: "
                f"recorded={recorded_ratio:.12f} expected={expected_ratio:.12f}"
            )

    ci = primary_metric.get("ci")
    display_ci = primary_metric.get("display_ci")
    if not (
        isinstance(ci, list | tuple)
        and len(ci) == 2
        and isinstance(display_ci, list | tuple)
        and len(display_ci) == 2
    ):
        errors.append(
            "Strict PPL evidence requires two-bound ci and display_ci arrays."
        )
    else:
        ci_values = tuple(_strict_finite_number(value) for value in ci)
        display_values = tuple(_strict_finite_number(value) for value in display_ci)
        if None in ci_values or None in display_values:
            errors.append("Strict PPL ci/display_ci bounds must be finite numbers.")
        else:
            ci_lower, ci_upper = ci_values
            display_lower, display_upper = display_values
            assert ci_lower is not None and ci_upper is not None
            assert display_lower is not None and display_upper is not None
            if ci_lower > ci_upper or display_lower > display_upper:
                errors.append("Strict PPL ci/display_ci bounds must be ordered.")
            for observed, bound in zip(display_values, ci_values, strict=True):
                assert observed is not None and bound is not None
                try:
                    expected = math.exp(bound)
                except OverflowError:
                    errors.append(
                        "Strict PPL ci exponentiation overflows finite range."
                    )
                    break
                if not math.isfinite(expected) or expected <= 0.0 or observed <= 0.0:
                    errors.append(
                        "Strict PPL ci/display_ci transforms must remain finite and "
                        "positive."
                    )
                    break
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    errors.append(
                        "primary_metric.display_ci mismatch: bounds do not match exp(ci)."
                    )
                    break
            if expected_ratio is not None:
                point = math.log(expected_ratio)
                if point < ci_lower - tolerance or point > ci_upper + tolerance:
                    errors.append(
                        "Strict PPL ci must contain the recomputed baseline log-ratio point."
                    )

    if preview_mean is not None and final_mean is not None:
        _append_strict_preview_final_slice_summary_errors(
            errors,
            cert_obj=cert_obj,
            preview_mean=preview_mean,
            final_mean=final_mean,
            tolerance=tolerance,
        )
