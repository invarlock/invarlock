"""Independent strict-assurance replay of paired PPL confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from invarlock.core.bootstrap import (
    PAIRED_BASELINE_BOOTSTRAP_METHOD,
    PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET,
)
from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)
from invarlock.core.runner_runtime.pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS

from .verify_bootstrap_math import (
    replay_paired_delta_log_ci as compute_paired_delta_log_ci,
)

MAX_STRICT_BOOTSTRAP_REPLICATES = 100_000
MAX_STRICT_BOOTSTRAP_WORK_ITEMS = 10_000_000
STRICT_BOOTSTRAP_ALPHA = 0.05
_STRICT_PROFILE_COVERAGE_FLOORS = {
    "ci": {"preview": 0, "final": 0, "replicates": 1200},
    "release": {"preview": 200, "final": 200, "replicates": 3200},
}
_MAX_EXACT_JSON_INTEGER = (2**53) - 1
_MAX_BOOTSTRAP_SEED = (2**63) - 1 - PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET
_SIGNED_64_MIN = -(2**63)
_SIGNED_64_MAX = (2**63) - 1


@dataclass(frozen=True)
class _RawFinalWindows:
    window_ids: tuple[int, ...]
    logloss: tuple[float, ...]
    token_counts: tuple[int, ...]


@dataclass(frozen=True)
class _BootstrapProvenance:
    replicates: int
    alpha: float
    seed: int


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_ppl_report(report: dict[str, Any]) -> bool:
    primary_metric = report.get("primary_metric")
    kind = primary_metric.get("kind") if isinstance(primary_metric, dict) else None
    try:
        normalized = normalize_metric_kind(kind)
    except (MetricKindContractError, RuntimeError, TypeError, ValueError):
        return False
    return normalized is not None and is_ppl_metric_kind(normalized)


def _raw_window_section(payload: dict[str, Any], arm: str) -> Any:
    evaluation_windows = payload.get("evaluation_windows")
    return evaluation_windows.get(arm) if isinstance(evaluation_windows, dict) else None


def _parse_raw_final_windows(
    errors: list[str],
    *,
    payload: dict[str, Any],
    source: str,
    arm: str = "final",
) -> _RawFinalWindows | None:
    section = _raw_window_section(payload, arm)
    if not isinstance(section, dict):
        errors.append(f"Strict paired PPL bootstrap replay requires {source}.")
        return None

    raw_ids = section.get("window_ids")
    raw_logloss = section.get("logloss")
    raw_counts = section.get("token_counts")
    if not all(
        isinstance(value, list) and value
        for value in (raw_ids, raw_logloss, raw_counts)
    ):
        errors.append(
            "Strict paired PPL bootstrap replay requires non-empty raw "
            f"{source}.window_ids/logloss/token_counts lists."
        )
        return None
    assert isinstance(raw_ids, list)
    assert isinstance(raw_logloss, list)
    assert isinstance(raw_counts, list)
    if not (len(raw_ids) == len(raw_logloss) == len(raw_counts)):
        errors.append(
            "Strict paired PPL bootstrap replay requires equal-length raw "
            f"{source}.window_ids/logloss/token_counts lists."
        )
        return None
    if len(raw_ids) < 2:
        errors.append(
            "Strict paired PPL bootstrap replay requires at least two paired final "
            f"windows in {source}."
        )
        return None

    ids: list[int] = []
    losses: list[float] = []
    counts: list[int] = []
    valid = True
    for index, raw_id in enumerate(raw_ids):
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or not _SIGNED_64_MIN <= raw_id <= _SIGNED_64_MAX
        ):
            errors.append(
                f"{source}.window_ids[{index}] must be a signed 64-bit JSON integer "
                "for canonical paired replay."
            )
            valid = False
        else:
            ids.append(raw_id)

    if len(ids) == len(raw_ids) and len(ids) != len(set(ids)):
        errors.append(f"{source}.window_ids contains duplicates.")
        valid = False

    for index, raw_loss in enumerate(raw_logloss):
        loss = _finite_number(raw_loss)
        if loss is None or loss < 0.0:
            errors.append(
                f"{source}.logloss[{index}] must be a finite non-negative number."
            )
            valid = False
        else:
            losses.append(loss)

    for index, raw_count in enumerate(raw_counts):
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or not 0 < raw_count <= _MAX_EXACT_JSON_INTEGER
        ):
            errors.append(
                f"{source}.token_counts[{index}] must be a positive JSON integer "
                f"no greater than {_MAX_EXACT_JSON_INTEGER}."
            )
            valid = False
        else:
            counts.append(raw_count)

    if not valid:
        return None
    return _RawFinalWindows(
        window_ids=tuple(ids),
        logloss=tuple(losses),
        token_counts=tuple(counts),
    )


def _parse_bootstrap_provenance(
    errors: list[str], report: dict[str, Any]
) -> _BootstrapProvenance | None:
    dataset = report.get("dataset")
    windows = dataset.get("windows") if isinstance(dataset, dict) else None
    stats = windows.get("stats") if isinstance(windows, dict) else None
    bootstrap = stats.get("bootstrap") if isinstance(stats, dict) else None
    if not isinstance(bootstrap, dict):
        errors.append(
            "Strict paired PPL bootstrap replay requires canonical "
            "dataset.windows.stats.bootstrap provenance."
        )
        return None

    valid = True
    if bootstrap.get("enabled") is not True:
        errors.append(
            "Strict paired PPL bootstrap replay requires "
            "dataset.windows.stats.bootstrap.enabled=true."
        )
        valid = False

    method = bootstrap.get("method")
    if method != PAIRED_BASELINE_BOOTSTRAP_METHOD:
        errors.append(
            "Strict paired PPL bootstrap replay requires canonical method "
            f"{PAIRED_BASELINE_BOOTSTRAP_METHOD!r}; got {method!r}."
        )
        valid = False

    replicates = bootstrap.get("replicates")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 0 < replicates <= MAX_STRICT_BOOTSTRAP_REPLICATES
    ):
        errors.append(
            "Strict paired PPL bootstrap replay requires a positive JSON integer "
            "dataset.windows.stats.bootstrap.replicates no greater than "
            f"{MAX_STRICT_BOOTSTRAP_REPLICATES}."
        )
        valid = False

    alpha = _finite_number(bootstrap.get("alpha"))
    if alpha is None or not 0.0 < alpha < 1.0:
        errors.append(
            "Strict paired PPL bootstrap replay requires finite "
            "dataset.windows.stats.bootstrap.alpha in (0,1)."
        )
        valid = False

    seed = bootstrap.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _MAX_BOOTSTRAP_SEED
    ):
        errors.append(
            "Strict paired PPL bootstrap replay requires a non-negative JSON integer "
            "dataset.windows.stats.bootstrap.seed in the supported range."
        )
        valid = False

    if not valid:
        return None
    assert isinstance(replicates, int)
    assert alpha is not None
    assert isinstance(seed, int)
    return _BootstrapProvenance(replicates=replicates, alpha=alpha, seed=seed)


def _strict_profile_and_tier(report: dict[str, Any]) -> tuple[str, str]:
    assurance = report.get("assurance")
    profile = assurance.get("profile") if isinstance(assurance, dict) else None
    tier = assurance.get("tier") if isinstance(assurance, dict) else None
    if not isinstance(profile, str):
        context = report.get("context")
        profile = context.get("profile") if isinstance(context, dict) else ""
    if not isinstance(tier, str):
        auto = report.get("auto")
        tier = auto.get("tier") if isinstance(auto, dict) else ""
    return str(profile or "").strip().lower(), str(tier or "").strip().lower()


def _strict_coverage_floors(report: dict[str, Any]) -> dict[str, int]:
    profile, tier = _strict_profile_and_tier(report)
    tier_floors = BOOTSTRAP_COVERAGE_REQUIREMENTS.get(
        tier, BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]
    )
    profile_floors = _STRICT_PROFILE_COVERAGE_FLOORS.get(profile, {})
    return {
        key: max(int(tier_floors.get(key, 0)), int(profile_floors.get(key, 0)))
        for key in ("preview", "final", "replicates")
    }


def _json_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _validate_coverage_mirror(
    errors: list[str],
    *,
    coverage: Any,
    source: str,
    tier: str,
    floors: dict[str, int],
    used: dict[str, int],
    required: bool,
) -> None:
    if not isinstance(coverage, dict):
        if required:
            errors.append(f"Strict assurance requires {source} coverage evidence.")
        return
    coverage_tier = coverage.get("tier")
    if coverage_tier is not None and coverage_tier != tier:
        errors.append(f"{source}.tier must match the strict assurance tier.")
    for key in ("preview", "final", "replicates"):
        entry = coverage.get(key)
        if not isinstance(entry, dict):
            errors.append(f"Strict assurance requires {source}.{key} evidence.")
            continue
        observed_used = _json_nonnegative_int(entry.get("used"))
        observed_required = _json_nonnegative_int(entry.get("required"))
        observed_ok = entry.get("ok")
        if observed_used is None:
            errors.append(f"{source}.{key}.used must be a non-negative integer.")
        elif observed_used != used[key]:
            errors.append(
                f"{source}.{key}.used disagrees with raw evidence "
                f"({observed_used} != {used[key]})."
            )
        if observed_required is None:
            errors.append(f"{source}.{key}.required must be a non-negative integer.")
        elif observed_required != floors[key]:
            errors.append(
                f"{source}.{key}.required must equal the independently derived "
                f"strict floor ({observed_required} != {floors[key]})."
            )
        expected_ok = used[key] >= floors[key]
        if not isinstance(observed_ok, bool):
            errors.append(f"{source}.{key}.ok must be a boolean.")
        elif observed_ok is not expected_ok:
            errors.append(
                f"{source}.{key}.ok disagrees with independently derived coverage."
            )
        if not expected_ok:
            unit = "bootstrap replicate" if key == "replicates" else "window"
            errors.append(
                f"{key} evidence is below the canonical strict {unit} floor "
                f"({used[key]} < {floors[key]})."
            )


def _append_strict_evidence_volume_errors(
    errors: list[str],
    *,
    report: dict[str, Any],
    subject_preview: _RawFinalWindows,
    subject_final: _RawFinalWindows,
    baseline_final: _RawFinalWindows,
    provenance: _BootstrapProvenance,
) -> None:
    profile, tier = _strict_profile_and_tier(report)
    floors = _strict_coverage_floors(report)
    used = {
        "preview": len(subject_preview.window_ids),
        "final": len(subject_final.window_ids),
        "replicates": provenance.replicates,
    }
    if len(baseline_final.window_ids) < floors["final"]:
        errors.append(
            "supplied baseline final evidence is below the canonical strict window "
            f"floor ({len(baseline_final.window_ids)} < {floors['final']})."
        )
    if len(subject_preview.window_ids) != len(subject_final.window_ids):
        errors.append(
            "strict assurance requires equal raw preview and final window counts."
        )
    if set(subject_preview.window_ids) & set(subject_final.window_ids):
        errors.append("strict assurance raw preview/final window IDs must be disjoint.")
    if not math.isclose(
        provenance.alpha,
        STRICT_BOOTSTRAP_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            f"strict bootstrap alpha must equal {STRICT_BOOTSTRAP_ALPHA:.2f}; "
            f"found {provenance.alpha}."
        )

    dataset = report.get("dataset")
    windows = dataset.get("windows") if isinstance(dataset, dict) else None
    stats = windows.get("stats") if isinstance(windows, dict) else None
    if not isinstance(windows, dict) or not isinstance(stats, dict):
        errors.append("strict assurance requires dataset.windows.stats evidence.")
        return
    count_mirrors = {
        "dataset.windows.preview": (windows.get("preview"), used["preview"]),
        "dataset.windows.final": (windows.get("final"), used["final"]),
        "dataset.windows.stats.actual_preview": (
            stats.get("actual_preview"),
            used["preview"],
        ),
        "dataset.windows.stats.actual_final": (
            stats.get("actual_final"),
            used["final"],
        ),
        "dataset.windows.stats.paired_windows": (
            stats.get("paired_windows"),
            used["final"],
        ),
    }
    for path, (raw, expected) in count_mirrors.items():
        observed = _json_nonnegative_int(raw)
        if observed is None:
            errors.append(f"{path} must be a non-negative integer.")
        elif observed != expected:
            errors.append(
                f"{path} disagrees with raw evidence ({observed} != {expected})."
            )

    _validate_coverage_mirror(
        errors,
        coverage=stats.get("coverage"),
        source="dataset.windows.stats.coverage",
        tier=tier,
        floors=floors,
        used=used,
        required=True,
    )
    bootstrap = stats.get("bootstrap")
    nested_coverage = bootstrap.get("coverage") if isinstance(bootstrap, dict) else None
    if nested_coverage is not None:
        _validate_coverage_mirror(
            errors,
            coverage=nested_coverage,
            source="dataset.windows.stats.bootstrap.coverage",
            tier=tier,
            floors=floors,
            used=used,
            required=False,
        )
    if profile not in _STRICT_PROFILE_COVERAGE_FLOORS:
        errors.append("strict evidence volume requires profile ci or release.")


def _metric_candidates(payload: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    candidates: list[tuple[str, Any]] = []
    if "primary_metric" in payload:
        candidates.append(
            ("supplied_baseline.primary_metric", payload["primary_metric"])
        )
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and "primary_metric" in metrics:
        candidates.append(
            ("supplied_baseline.metrics.primary_metric", metrics["primary_metric"])
        )
    return tuple(candidates)


def _append_baseline_metric_recompute_errors(
    errors: list[str],
    *,
    report: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_windows: _RawFinalWindows,
    tolerance: float,
) -> None:
    total = sum(baseline_windows.token_counts)
    mean_logloss = math.fsum(
        loss * (count / total)
        for loss, count in zip(
            baseline_windows.logloss,
            baseline_windows.token_counts,
            strict=True,
        )
    )
    try:
        expected_final = math.exp(mean_logloss)
    except OverflowError:
        errors.append(
            "Supplied baseline raw final log-loss overflows finite perplexity."
        )
        return
    if not math.isfinite(expected_final) or expected_final < 1.0:
        errors.append(
            "Supplied baseline raw final log-loss does not produce finite PPL >= 1."
        )
        return

    for source, metric in _metric_candidates(baseline_payload):
        final = metric.get("final") if isinstance(metric, dict) else None
        observed = _finite_number(final)
        if observed is not None and not math.isclose(
            observed,
            expected_final,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "Supplied baseline metric/raw-window mismatch: "
                f"{source}.final={observed:.12f} recomputed={expected_final:.12f}."
            )

    baseline_ref = report.get("baseline_ref")
    embedded_metric = (
        baseline_ref.get("primary_metric") if isinstance(baseline_ref, dict) else None
    )
    embedded_final = (
        _finite_number(embedded_metric.get("final"))
        if isinstance(embedded_metric, dict)
        else None
    )
    if embedded_final is not None and not math.isclose(
        embedded_final,
        expected_final,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        errors.append(
            "Report baseline_ref/raw-window mismatch: "
            f"baseline_ref.primary_metric.final={embedded_final:.12f} "
            f"recomputed={expected_final:.12f}."
        )


def _reported_ci(
    errors: list[str], report: dict[str, Any]
) -> tuple[float, float] | None:
    primary_metric = report.get("primary_metric")
    ci = primary_metric.get("ci") if isinstance(primary_metric, dict) else None
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        errors.append(
            "Strict paired PPL bootstrap replay requires two finite "
            "primary_metric.ci bounds."
        )
        return None
    lower = _finite_number(ci[0])
    upper = _finite_number(ci[1])
    if lower is None or upper is None or lower > upper:
        errors.append(
            "Strict paired PPL bootstrap replay requires ordered finite "
            "primary_metric.ci bounds."
        )
        return None
    return lower, upper


def append_strict_ppl_bootstrap_replay_errors(
    errors: list[str],
    *,
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    baseline_supplied: bool,
    tolerance: float,
) -> None:
    """Replay strict paired PPL CI from independently supplied raw baseline evidence."""

    if not _is_ppl_report(report):
        return
    if not baseline_supplied:
        errors.append(
            "Strict paired PPL verification requires independently supplied --baseline "
            "raw final window IDs, logloss, and token_counts."
        )
        return
    if baseline_payload is None:
        errors.append(
            "Strict paired PPL bootstrap replay could not load the "
            "independently supplied baseline."
        )
        return

    subject_preview_windows = _parse_raw_final_windows(
        errors,
        payload=report,
        source="report.evaluation_windows.preview",
        arm="preview",
    )
    subject_windows = _parse_raw_final_windows(
        errors,
        payload=report,
        source="report.evaluation_windows.final",
    )
    baseline_windows = _parse_raw_final_windows(
        errors,
        payload=baseline_payload,
        source="supplied_baseline.evaluation_windows.final",
    )
    provenance = _parse_bootstrap_provenance(errors, report)
    ci = _reported_ci(errors, report)
    if (
        subject_preview_windows is None
        or subject_windows is None
        or baseline_windows is None
    ):
        return

    if subject_windows.window_ids != baseline_windows.window_ids:
        errors.append(
            "Strict paired PPL bootstrap replay requires the supplied baseline raw "
            "final window IDs in the exact subject order."
        )
        return
    if subject_windows.token_counts != baseline_windows.token_counts:
        errors.append(
            "Strict paired PPL bootstrap replay requires identical subject/baseline "
            "token_counts for every paired final window."
        )
        return

    _append_baseline_metric_recompute_errors(
        errors,
        report=report,
        baseline_payload=baseline_payload,
        baseline_windows=baseline_windows,
        tolerance=tolerance,
    )
    if provenance is None or ci is None:
        return
    _append_strict_evidence_volume_errors(
        errors,
        report=report,
        subject_preview=subject_preview_windows,
        subject_final=subject_windows,
        baseline_final=baseline_windows,
        provenance=provenance,
    )
    replay_work_items = len(subject_windows.window_ids) * provenance.replicates
    if replay_work_items > MAX_STRICT_BOOTSTRAP_WORK_ITEMS:
        errors.append(
            "Strict paired PPL bootstrap replay exceeds the verifier work limit: "
            f"windows*replicates={replay_work_items} maximum="
            f"{MAX_STRICT_BOOTSTRAP_WORK_ITEMS}."
        )
        return

    try:
        replayed = compute_paired_delta_log_ci(
            subject_windows.logloss,
            baseline_windows.logloss,
            weights=subject_windows.token_counts,
            method="bca",
            replicates=provenance.replicates,
            alpha=provenance.alpha,
            seed=provenance.seed + PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET,
            strict_lengths=True,
        )
    except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
        errors.append(f"Strict paired PPL bootstrap replay failed: {exc}.")
        return

    for name, observed, expected in zip(("lower", "upper"), ci, replayed, strict=True):
        if not math.isclose(
            observed,
            expected,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "Strict paired PPL bootstrap CI mismatch: "
                f"primary_metric.ci.{name}={observed:.12f} "
                f"replayed={expected:.12f}."
            )


__all__ = [
    "MAX_STRICT_BOOTSTRAP_REPLICATES",
    "MAX_STRICT_BOOTSTRAP_WORK_ITEMS",
    "STRICT_BOOTSTRAP_ALPHA",
    "append_strict_ppl_bootstrap_replay_errors",
]
