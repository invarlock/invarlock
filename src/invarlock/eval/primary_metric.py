"""
Primary metric abstraction and minimal ppl_causal implementation (Phase 1).

This module introduces a light-weight, task-agnostic metric interface and a
registry so the runner and evaluation report builder can evolve beyond causal-LM perplexity.

Phase 1 goal: provide a ppl_causal metric and a helper that can compute point
estimates directly from evaluation window aggregates already present in run
reports. This is the canonical path; no env flag toggles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.core.exceptions import ValidationError
from invarlock.eval.primary_metric_accuracy import (
    compute_accuracy_counts as compute_accuracy_counts,
)
from invarlock.eval.primary_metric_accuracy import (
    infer_binary_label_from_ids as infer_binary_label_from_ids,
)
from invarlock.utils import (
    bootstrap_mean_statistics,
    percentile_interval_from_statistics,
)


@dataclass
class MetricDefaults:
    reps: int = 2000
    ci_level: float = 0.95


@dataclass
class MetricContribution:
    """Per-example contribution to a metric.

    For ppl_* metrics, `value` is per-token mean log-loss for the example and
    `weight` is the number of target tokens. For accuracy, `value` is 0/1 and
    `weight` is ignored.
    """

    value: float
    weight: float = 1.0
    id: str | None = None


class PrimaryMetric(Protocol):
    """Protocol for task-agnostic primary metrics.

    Implementations should describe their semantics and provide helpers to
    compute point estimates (and optionally CIs) from per-example aggregates.
    """

    kind: str
    unit: str
    direction: str  # "lower" | "higher"
    guard_degradation_basis: str  # "relative_increase" | "absolute_drop"
    aggregation_scope: str  # "token" | "sequence" | "example"
    paired: bool
    gating_basis: str  # "point" | "upper" | "lower"
    supports_bootstrap: bool
    defaults: MetricDefaults

    def display_transform(self, x: float) -> float:
        """Map native comparison space to display space.

        ppl_*: exp(x) maps Δlog-loss → ratio
        accuracy: x*100 maps proportion Δ → percentage points
        """

    def point_from_windows(self, *, windows: dict[str, Any]) -> float:
        """Compute a single point estimate from evaluation windows.

        For token-aggregated loss metrics (like ppl), this expects:
        windows = {"logloss": [...], "token_counts": [...]} with matching lengths.
        """

    def accumulate(self, contrib: MetricContribution) -> None:
        """Accumulate a per-example contribution for finalize()."""

    def finalize(self) -> float:
        """Return a point estimate from accumulated contributions (display space)."""

    def paired_compare(
        self,
        subject: Iterable[dict[str, Any] | MetricContribution],
        baseline: Iterable[dict[str, Any] | MetricContribution],
        *,
        reps: int | None = None,
        seed: int | None = None,
        ci_level: float | None = None,
    ) -> dict[str, Any]:
        """Paired compare subject vs baseline with bootstrap CI.

        Returns a dict with native-space delta and display-space values:
        {
          'delta': float,            # native compare space (Δlog-loss for ppl, Δ proportion for accuracy)
          'ci': (lo, hi),            # native-space CI
          'display': float,          # display space (ratio for ppl, pp for accuracy)
          'display_ci': (lo, hi),
          'subject_point': float,    # point estimate in display space
          'baseline_point': float,   # point estimate in display space
          'reps': int, 'ci_level': float, 'paired': True, meta...
        }
        """


@dataclass
class MetricInfo:
    kind: str
    unit: str
    direction: str
    aggregation_scope: str
    paired: bool
    gating_basis: str
    supports_bootstrap: bool


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_NUMERIC_COERCION_ERRORS = (TypeError, ValueError, OverflowError)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except _NUMERIC_COERCION_ERRORS:
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    try:
        return int(value)
    except _NUMERIC_COERCION_ERRORS:
        return None


class _PPLCausal(PrimaryMetric):
    """Token-aggregated perplexity for causal LMs.

    point_from_windows computes ppl = exp(weighted_mean(logloss)).
    """

    kind = "ppl_causal"
    unit = "ppl"
    direction = "lower"
    guard_degradation_basis = "relative_increase"
    aggregation_scope = "token"
    paired = True
    gating_basis = "upper"  # typical gate on ratio upper-bound
    supports_bootstrap = True
    defaults = MetricDefaults()

    def __init__(self) -> None:
        self._values: list[float] = []
        self._weights: list[float] = []

    def display_transform(self, x: float) -> float:
        try:
            return float(math.exp(x))
        except OverflowError:
            return float("inf")

    def point_from_windows(self, *, windows: dict[str, Any]) -> float:
        logloss = list(windows.get("logloss", []) or [])
        token_counts = list(windows.get("token_counts", []) or [])
        if not logloss or not token_counts or len(logloss) != len(token_counts):
            return float("nan")
        sum_w = 0.0
        sum_wx = 0.0
        for ll, w in zip(logloss, token_counts, strict=False):
            llv = _coerce_float(ll)
            wv = _coerce_float(w)
            if llv is None or wv is None:
                continue
            if not math.isfinite(llv) or not math.isfinite(wv) or wv <= 0:
                continue
            sum_w += wv
            sum_wx += wv * llv
        if sum_w <= 0.0:
            return float("nan")
        mean_ll = sum_wx / sum_w
        try:
            return float(math.exp(mean_ll))
        except OverflowError:
            return float("inf")

    def accumulate(self, contrib: MetricContribution) -> None:
        v = _coerce_float(contrib.value)
        w = _coerce_float(contrib.weight)
        if v is None or w is None:
            return
        if not math.isfinite(v) or not math.isfinite(w) or w <= 0:
            return
        self._values.append(v)
        self._weights.append(w)

    def finalize(self) -> float:
        if (
            not self._values
            or not self._weights
            or len(self._values) != len(self._weights)
        ):
            return float("nan")
        sw = 0.0
        swx = 0.0
        for v, w in zip(self._values, self._weights, strict=False):
            sw += w
            swx += w * v
        if sw <= 0.0:
            return float("nan")
        return self.display_transform(swx / sw)

    def _coerce_contrib_array(
        self, items: Iterable[dict[str, Any] | MetricContribution]
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for it in items:
            if isinstance(it, MetricContribution):
                value = _coerce_float(it.value)
                weight = _coerce_float(it.weight)
                if value is not None and weight is not None:
                    out.append((value, weight))
            elif isinstance(it, dict) and "value" in it:
                value = _coerce_float(it.get("value"))
                weight = _coerce_float(it.get("weight", 1.0))
                if value is not None and weight is not None:
                    out.append((value, weight))
        return out

    def paired_compare(
        self,
        subject: Iterable[dict[str, Any] | MetricContribution],
        baseline: Iterable[dict[str, Any] | MetricContribution],
        *,
        reps: int | None = None,
        seed: int | None = None,
        ci_level: float | None = None,
    ) -> dict[str, Any]:
        subj = self._coerce_contrib_array(subject)
        base = self._coerce_contrib_array(baseline)
        # Compute per-example arrays in log space; use weights for paired bootstrap
        subj_vals = [v for (v, _w) in subj]
        base_vals = [v for (v, _w) in base]
        pair_weights = []
        for (_sv, sw), (_bv, bw) in zip(subj, base, strict=False):
            weight = bw if math.isfinite(bw) and bw > 0 else sw
            if not math.isfinite(weight) or weight <= 0:
                weight = 1.0
            pair_weights.append(float(weight))

        # Points in display space
        def _point(
            vals: Sequence[float], weights: Sequence[float] | None = None
        ) -> float:
            if not vals:
                return float("nan")
            assert weights is not None and len(weights) == len(vals)
            sw = 0.0
            swx = 0.0
            for v, w in zip(vals, weights, strict=False):
                sw += w
                swx += w * v
            if sw <= 0:
                return float("nan")
            return self.display_transform(swx / sw)

        subj_point = _point([v for v, _ in subj], [w for _, w in subj])
        base_point = _point([v for v, _ in base], [w for _, w in base])

        # Bootstrap Δlog-loss → CI, then display-transform → ratio CI
        reps_eff = int(reps) if (reps is not None and reps > 0) else self.defaults.reps
        seed_eff = int(seed) if (seed is not None) else 0
        ci_level_eff = (
            float(ci_level) if (ci_level is not None) else self.defaults.ci_level
        )
        alpha = 1.0 - ci_level_eff
        dlog_lo, dlog_hi = compute_paired_delta_log_ci(
            subj_vals,
            base_vals,
            weights=pair_weights,
            method="bca",
            replicates=reps_eff,
            alpha=alpha,
            seed=seed_eff,
            strict_lengths=False,
        )
        if pair_weights and len(pair_weights) >= min(len(subj_vals), len(base_vals)):
            sw = 0.0
            swx = 0.0
            for s, b, w in zip(subj_vals, base_vals, pair_weights, strict=False):
                sw += w
                swx += w * (s - b)
            delta_log = float(swx / sw) if sw > 0 else float("nan")
        else:
            delta_log = float(
                sum((s - b) for s, b in zip(subj_vals, base_vals, strict=False))
                / max(1, min(len(subj_vals), len(base_vals)))
            )
        ratio = self.display_transform(delta_log)
        return {
            "kind": self.kind,
            "unit": self.unit,
            "direction": self.direction,
            "aggregation_scope": self.aggregation_scope,
            "paired": True,
            "gating_basis": self.gating_basis,
            "supports_bootstrap": self.supports_bootstrap,
            "reps": reps_eff,
            "ci_level": ci_level_eff,
            "subject_point": subj_point,
            "baseline_point": base_point,
            "delta": delta_log,
            "ci": [dlog_lo, dlog_hi],
            "display": ratio,
            "display_ci": [
                self.display_transform(dlog_lo),
                self.display_transform(dlog_hi),
            ],
        }


# ── Simple registry ───────────────────────────────────────────────────────
class _PPLMLM(_PPLCausal):
    """Masked LM perplexity.

    Uses masked_token_counts when available; falls back to token_counts.
    """

    kind = "ppl_mlm"

    def point_from_windows(self, *, windows: dict[str, Any]) -> float:
        # Prefer masked_token_counts for MLM
        masked = list(windows.get("masked_token_counts", []) or [])
        if masked:
            win = {"logloss": windows.get("logloss", []), "token_counts": masked}
            return super().point_from_windows(windows=win)
        return super().point_from_windows(windows=windows)


class _PPLSeq2Seq(_PPLCausal):
    """Seq2Seq perplexity (token-aggregated over decoder labels)."""

    kind = "ppl_seq2seq"


class _Accuracy:
    """Example-aggregated accuracy (0..1).

    Accepts either per-example flags (example_correct) or aggregate counts
    (correct_total/total or correct_counts/total_counts).
    """

    kind = "accuracy"
    unit = "accuracy"
    direction = "higher"
    guard_degradation_basis = "absolute_drop"
    aggregation_scope = "example"
    paired = True
    gating_basis = "lower"
    supports_bootstrap = True
    defaults = MetricDefaults()

    def __init__(self) -> None:
        self._values: list[float] = []

    def display_transform(self, x: float) -> float:  # proportion → percentage points
        return float(x * 100.0)

    def point_from_windows(self, *, windows: dict[str, Any]) -> float:
        # Per-example path
        ex = list(windows.get("example_correct", []) or [])
        if ex:
            s = 0.0
            n = 0.0
            for v in ex:
                coerced = _coerce_float(v)
                if coerced is None:
                    continue
                s += 1.0 if coerced > 0.5 else 0.0
                n += 1.0
            if n > 0:
                return s / n
            return float("nan")
        # Aggregate counts path
        for c_key, t_key in (
            ("correct_total", "total"),
            ("correct_counts", "total_counts"),
        ):
            c = windows.get(c_key)
            t = windows.get(t_key)
            c_value = _coerce_float(c)
            t_value = _coerce_float(t)
            if c_value is not None and t_value is not None and t_value > 0:
                total = t_value
                # Optional abstain/tie handling with documented policy
                policy = (
                    windows.get("policy", {})
                    if isinstance(windows.get("policy"), dict)
                    else {}
                )
                abstain = windows.get("abstain_total")
                ties = windows.get("ties_total")
                exclude_abstain = bool(policy.get("exclude_abstain", True))
                count_ties_as_correct = bool(policy.get("ties_count_as_correct", False))
                count_ties_as_incorrect = bool(
                    policy.get("ties_count_as_incorrect", False)
                )
                # Apply abstain exclusion from denominator if requested
                abstain_value = _coerce_float(abstain)
                if exclude_abstain and abstain_value is not None and abstain_value > 0:
                    total = max(1.0, total - abstain_value)
                # Apply tie policy
                ties_value = _coerce_float(ties)
                if ties_value is not None and ties_value > 0:
                    if count_ties_as_correct:
                        c_value += ties_value
                    elif count_ties_as_incorrect:
                        pass
                    else:
                        if exclude_abstain:
                            total = max(1.0, total - ties_value)
                return c_value / total
        return float("nan")

    def accumulate(self, contrib: MetricContribution) -> None:
        v = _coerce_float(contrib.value)
        if v is None:
            return
        if not math.isfinite(v):
            return
        # Clamp to [0,1]
        v = 1.0 if v >= 0.5 else 0.0
        self._values.append(v)

    def finalize(self) -> float:
        if not self._values:
            return float("nan")
        return float(sum(self._values) / float(len(self._values)))

    def _coerce_vals(
        self, items: Iterable[dict[str, Any] | MetricContribution]
    ) -> list[float]:
        out: list[float] = []
        for it in items:
            if isinstance(it, MetricContribution):
                value = _coerce_float(it.value)
                if value is not None:
                    out.append(1.0 if value >= 0.5 else 0.0)
            elif isinstance(it, dict) and "value" in it:
                value = _coerce_float(it.get("value"))
                if value is not None:
                    out.append(1.0 if value >= 0.5 else 0.0)
        return out

    def paired_compare(
        self,
        subject: Iterable[dict[str, Any] | MetricContribution],
        baseline: Iterable[dict[str, Any] | MetricContribution],
        *,
        reps: int | None = None,
        seed: int | None = None,
        ci_level: float | None = None,
    ) -> dict[str, Any]:
        subj = self._coerce_vals(subject)
        base = self._coerce_vals(baseline)
        m = min(len(subj), len(base))
        subj = subj[:m]
        base = base[:m]
        if m == 0:
            return {
                "kind": self.kind,
                "unit": self.unit,
                "direction": self.direction,
                "aggregation_scope": self.aggregation_scope,
                "paired": True,
                "gating_basis": self.gating_basis,
                "supports_bootstrap": self.supports_bootstrap,
                "reps": 0,
                "ci_level": ci_level or self.defaults.ci_level,
                "subject_point": float("nan"),
                "baseline_point": float("nan"),
                "delta": float("nan"),
                "ci": [float("nan"), float("nan")],
                "display": float("nan"),
                "display_ci": [float("nan"), float("nan")],
            }
        # Points in display space for subject/baseline (proportions, no transform)
        subj_point = float(sum(subj) / float(m))
        base_point = float(sum(base) / float(m))
        # Δ in native (proportion) space
        diffs = [float(s - b) for s, b in zip(subj, base, strict=False)]
        delta = float(sum(diffs) / float(m))
        reps_eff = int(reps) if (reps is not None and reps > 0) else self.defaults.reps
        seed_eff = int(seed) if (seed is not None) else 0
        ci_level_eff = (
            float(ci_level) if (ci_level is not None) else self.defaults.ci_level
        )
        alpha = 1.0 - ci_level_eff
        rng = cast(Any, np.random.default_rng(seed_eff))
        stats = bootstrap_mean_statistics(
            np.asarray(diffs, dtype=float),
            n_bootstrap=reps_eff,
            random_state=rng,
        )
        lo, hi = percentile_interval_from_statistics(stats, alpha=alpha)
        return {
            "kind": self.kind,
            "unit": self.unit,
            "direction": self.direction,
            "aggregation_scope": self.aggregation_scope,
            "paired": True,
            "gating_basis": self.gating_basis,
            "supports_bootstrap": self.supports_bootstrap,
            "reps": reps_eff,
            "ci_level": ci_level_eff,
            "subject_point": subj_point,
            "baseline_point": base_point,
            "delta": delta,
            "ci": [lo, hi],
            "display": self.display_transform(delta),
            "display_ci": [self.display_transform(lo), self.display_transform(hi)],
        }


_REGISTRY: dict[str, PrimaryMetric] = {
    _PPLCausal.kind: _PPLCausal(),
    _PPLMLM.kind: _PPLMLM(),
    _PPLSeq2Seq.kind: _PPLSeq2Seq(),
    _Accuracy.kind: _Accuracy(),
}


def get_metric(kind: str) -> PrimaryMetric:
    key = str(kind).lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    raise KeyError(f"Unknown metric kind: {kind}")


def compute_primary_metric_from_report(
    report: dict[str, Any],
    *,
    kind: str = "ppl_causal",
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a primary metric snapshot from a run report (Phase 1 helper).

    Returns a dict that can be attached to report["metrics"]["primary_metric"].
    Includes preview/final points and a kind-specific baseline comparison when
    a compatible baseline is present: ``ratio_vs_baseline`` for perplexity and
    ``delta_vs_baseline_pp`` for accuracy.
    """
    metric = get_metric(kind)
    windows = report.get("evaluation_windows") if isinstance(report, dict) else None

    # Choose window sections
    preview_win: dict[str, Any] = {}
    final_win: dict[str, Any] = {}

    counts_source_tag: str | None = None
    n_prev: int | None = None
    n_fin: int | None = None
    if kind == "accuracy":
        # Prefer classification aggregates if provided (may not have evaluation_windows)
        metrics = (
            report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        )
        clf = metrics.get("classification") if isinstance(metrics, dict) else None
        if isinstance(clf, dict) and clf:
            prev = (
                clf.get("preview", {}) if isinstance(clf.get("preview"), dict) else {}
            )
            fin = clf.get("final", {}) if isinstance(clf.get("final"), dict) else {}
            preview_win = prev
            final_win = fin
            # Attach counts into a small context to help gating
            n_prev = _coerce_int(prev.get("total"))
            prev_examples = prev.get("example_correct")
            if n_prev is None and isinstance(prev_examples, list):
                n_prev = len(prev_examples)
            n_fin = _coerce_int(fin.get("total"))
            fin_examples = fin.get("example_correct")
            if n_fin is None and isinstance(fin_examples, list):
                n_fin = len(fin_examples)
            # Propagate counts source tagging when present
            counts_source = clf.get("counts_source")
            if isinstance(counts_source, str) and counts_source:
                counts_source_tag = counts_source

    if not preview_win and not final_win and isinstance(windows, dict):
        preview_win = (
            windows.get("preview", {})
            if isinstance(windows.get("preview"), dict)
            else {}
        )
        final_win = (
            windows.get("final", {}) if isinstance(windows.get("final"), dict) else {}
        )

    if not preview_win and not final_win:
        # Nothing to compute from
        payload = {
            "kind": metric.kind,
            "unit": metric.unit,
            "direction": metric.direction,
            "aggregation_scope": metric.aggregation_scope,
            "paired": metric.paired,
            "gating_basis": metric.gating_basis,
            "supports_bootstrap": metric.supports_bootstrap,
            "preview": float("nan"),
            "final": float("nan"),
            "invalid": True,
            "degraded": True,
            "degraded_reason": "non_finite_pm",
        }
        return payload
    preview_point = metric.point_from_windows(windows=preview_win)
    final_point = metric.point_from_windows(windows=final_win)

    ratio_vs_baseline = float("nan")
    delta_vs_baseline_pp = float("nan")

    def _is_finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    if isinstance(baseline, dict):
        base_metrics = (
            baseline.get("metrics", {})
            if isinstance(baseline.get("metrics"), dict)
            else {}
        )
        pm_base = base_metrics.get("primary_metric")
        base_kind = (
            str(pm_base.get("kind", "")).lower() if isinstance(pm_base, dict) else ""
        )
        kind_l = str(kind).lower()
        ppl_kinds = {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}
        acc_kinds = {"accuracy"}
        same_family = (kind_l in ppl_kinds and base_kind in ppl_kinds) or (
            kind_l in acc_kinds and base_kind in acc_kinds
        )
        if isinstance(pm_base, dict) and (base_kind == kind_l or same_family):
            base_ref = pm_base.get("final")
            base_value = _coerce_float(base_ref)
            if base_value is not None:
                is_ppl_like = str(kind).lower().startswith("ppl")
                if is_ppl_like and base_value > 0:
                    ratio_vs_baseline = float(final_point) / base_value
                elif str(kind).lower() == "accuracy" and 0 <= base_value <= 1:
                    delta_vs_baseline_pp = 100.0 * (float(final_point) - base_value)

    invalid = True
    invalid = not (_is_finite(preview_point) and _is_finite(final_point))
    degraded_reason = None
    if invalid:
        degraded_reason = "non_finite_pm"

    degraded = bool(degraded_reason or invalid)

    payload = {
        "kind": metric.kind,
        "unit": metric.unit,
        "direction": metric.direction,
        "aggregation_scope": metric.aggregation_scope,
        "paired": metric.paired,
        "gating_basis": metric.gating_basis,
        "supports_bootstrap": metric.supports_bootstrap,
        "preview": preview_point,
        "final": final_point,
        "invalid": invalid,
        "degraded": degraded,
    }
    if kind == "accuracy" and _is_finite(delta_vs_baseline_pp):
        payload["delta_vs_baseline_pp"] = delta_vs_baseline_pp
    elif kind != "accuracy" and _is_finite(ratio_vs_baseline):
        payload["ratio_vs_baseline"] = ratio_vs_baseline
    if degraded and degraded_reason:
        payload["degraded_reason"] = degraded_reason
    # Carry counts for accuracy to aid gating
    if kind == "accuracy":
        if n_prev is not None:
            payload["n_preview"] = int(n_prev)
        if n_fin is not None:
            payload["n_final"] = int(n_fin)
        # Carry counts_source/estimated tag when available
        if isinstance(counts_source_tag, str) and counts_source_tag:
            payload["counts_source"] = counts_source_tag
            payload["estimated"] = counts_source_tag != "measured"
    return payload


def validate_primary_metric_block(block: dict[str, Any]) -> dict[str, Any]:
    """Validate that a primary_metric block has finite preview/final values.

    Raises ValidationError(E402) when preview or final are non-finite.

    Returns the input block on success to enable fluent usage.
    """
    prev_raw = block.get("preview")
    fin_raw = block.get("final")
    if isinstance(prev_raw, bool) or isinstance(fin_raw, bool):
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "preview/final must be numeric, not bool"},
        )
    prev = _coerce_float(prev_raw)
    fin = _coerce_float(fin_raw)
    if prev is None or fin is None:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "missing preview/final"},
        )
    if not math.isfinite(prev) or not math.isfinite(fin):
        details = {
            "reason": "non-finite primary_metric values",
            "preview": prev,
            "final": fin,
        }
        raise ValidationError(
            code="E402", message="METRICS-VALIDATION-FAILED", details=details
        )
    return block
