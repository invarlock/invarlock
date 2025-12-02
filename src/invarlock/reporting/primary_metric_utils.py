from __future__ import annotations

import copy
import math
from typing import Any

from .utils import _coerce_interval, _weighted_mean


def attach_primary_metric(
    certificate: dict[str, Any],
    report: dict[str, Any],
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any] | None,
    ppl_analysis: dict[str, Any] | None,
) -> None:
    """Attach/normalize the primary_metric block on the certificate.

    Behavior mirrors historical logic in certificate.py and preserves structure:
    - Prefer explicit metrics.primary_metric if present
    - Compute missing ratio_vs_baseline, degenerate display_ci
    - ppl window-based analysis info (mean logloss) added when available
    - Fallbacks for classification metrics and eval-window-derived ppl
    - Ensure display_ci always present for schema invariants
    Mutates the certificate in-place.
    """
    # Attach primary metric snapshot when provided in report
    try:
        m = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        pm = m.get("primary_metric") if isinstance(m, dict) else None
        if isinstance(pm, dict) and pm:
            pm_copy = copy.deepcopy(pm)
            # Propagate instability hint from ppl_analysis
            try:
                if isinstance(ppl_analysis, dict) and bool(
                    ppl_analysis.get("unstable")
                ):
                    pm_copy.setdefault("unstable", True)
            except Exception:
                pass
            # Attach analysis-basis numbers for ppl from evaluation windows
            try:
                kind = str(pm_copy.get("kind", "")).lower()
            except Exception:
                kind = ""
            if kind.startswith("ppl"):
                try:
                    eval_windows = (
                        report.get("evaluation_windows", {})
                        if isinstance(report.get("evaluation_windows"), dict)
                        else {}
                    )
                    prev_sec = (
                        eval_windows.get("preview")
                        if isinstance(eval_windows, dict)
                        else None
                    )
                    fin_sec = (
                        eval_windows.get("final")
                        if isinstance(eval_windows, dict)
                        else None
                    )
                    if isinstance(prev_sec, dict) and isinstance(fin_sec, dict):
                        prev_ll = list(prev_sec.get("logloss", []) or [])
                        prev_wc = list(prev_sec.get("token_counts", []) or [])
                        fin_ll = list(fin_sec.get("logloss", []) or [])
                        fin_wc = list(fin_sec.get("token_counts", []) or [])
                        mean_prev = _weighted_mean(prev_ll, prev_wc)
                        mean_fin = _weighted_mean(fin_ll, fin_wc)
                        if math.isfinite(mean_prev):
                            pm_copy["analysis_basis"] = "mean_logloss"
                            pm_copy["analysis_point_preview"] = float(mean_prev)
                        if math.isfinite(mean_fin):
                            pm_copy["analysis_basis"] = "mean_logloss"
                            pm_copy["analysis_point_final"] = float(mean_fin)
                    # Attach analysis-basis CIs for preview/final in log space from report metrics
                    try:
                        dlci = (
                            _coerce_interval(m.get("logloss_delta_ci"))
                            if isinstance(m, dict)
                            else (math.nan, math.nan)
                        )
                        if isinstance(dlci, tuple | list) and len(dlci) == 2:
                            lo, hi = float(dlci[0]), float(dlci[1])
                            if math.isfinite(lo) and math.isfinite(hi):
                                pm_copy.setdefault("ci", (lo, hi))
                    except Exception:
                        pass
                except Exception:
                    pass
            # Ensure ratio_vs_baseline present and consistent
            try:
                base_final = (
                    baseline_ref.get("primary_metric", {}).get("final")
                    if isinstance(baseline_ref, dict)
                    else None
                )
                fin = pm_copy.get("final")
                if (
                    isinstance(fin, int | float)
                    and isinstance(base_final, int | float)
                    and float(base_final) > 0
                ):
                    pm_copy["ratio_vs_baseline"] = float(fin) / float(base_final)
                # Provide a degenerate display CI if missing
                if not isinstance(
                    pm_copy.get("display_ci"), list | tuple
                ) and isinstance(pm_copy.get("final"), int | float):
                    pm_copy["display_ci"] = [
                        float(pm_copy["final"]),
                        float(pm_copy["final"]),
                    ]
            except Exception:
                pass
            certificate["primary_metric"] = pm_copy
    except Exception:
        pass

    def _attach_from_windows() -> None:
        if isinstance(certificate.get("primary_metric"), dict):
            return
        try:
            m = (
                report.get("metrics", {})
                if isinstance(report.get("metrics"), dict)
                else {}
            )
            loss_type = (
                (m.get("loss_type") or "").lower() if isinstance(m, dict) else ""
            )
            if loss_type == "mlm":
                kind_hint = "ppl_mlm"
            elif loss_type in {"seq2seq", "s2s", "t5"}:
                kind_hint = "ppl_seq2seq"
            else:
                kind_hint = "ppl_causal"
            from invarlock.eval.primary_metric import (
                compute_primary_metric_from_report as _pm,
            )

            pm_block = _pm(
                report,
                kind=kind_hint,
                baseline=baseline_raw if isinstance(baseline_raw, dict) else None,
            )
            if isinstance(pm_block, dict) and pm_block:
                certificate["primary_metric"] = pm_block
        except Exception:
            pass

    # First attempt to synthesize PM from evaluation windows before falling back
    _attach_from_windows()

    # Minimal fallback for classification-only reports without explicit primary_metric
    if not isinstance(certificate.get("primary_metric"), dict):
        try:
            metrics_map = report.get("metrics", {}) if isinstance(report, dict) else {}
            clf = (
                metrics_map.get("classification")
                if isinstance(metrics_map, dict)
                else None
            )
            if isinstance(clf, dict) and clf:
                model_id = str(
                    (report.get("meta", {}) or {}).get("model_id", "")
                ).lower()
                pm_kind = "vqa_accuracy" if "vqa" in model_id else "accuracy"
                pm_point = None
                try:
                    val = clf.get("final")
                    if isinstance(val, int | float):
                        pm_point = float(val)
                    elif isinstance(val, dict):
                        num = val.get("correct_total")
                        den = val.get("total")
                        if (
                            isinstance(num, int | float)
                            and isinstance(den, int | float)
                            and float(den) > 0
                        ):
                            pm_point = float(num) / float(den)
                except Exception:
                    pm_point = None
                acc_pm: dict[str, Any] = {
                    "kind": pm_kind,
                    "unit": "accuracy",
                    "direction": "higher",
                    "aggregation_scope": "example",
                    "paired": True,
                    "gating_basis": "point",
                }
                if isinstance(pm_point, float):
                    acc_pm["final"] = pm_point
                    acc_pm.setdefault("display_ci", [pm_point, pm_point])
                try:
                    base_cls = None
                    if isinstance(baseline_raw, dict):
                        bm = (
                            baseline_raw.get("metrics")
                            if isinstance(baseline_raw.get("metrics"), dict)
                            else None
                        )
                        if isinstance(bm, dict):
                            base_cls = bm.get("classification")
                    if base_cls is None and isinstance(baseline_ref, dict):
                        bm = (
                            baseline_ref.get("metrics")
                            if isinstance(baseline_ref.get("metrics"), dict)
                            else None
                        )
                        if isinstance(bm, dict):
                            base_cls = bm.get("classification")
                    acc_run = pm_point
                    acc_base = None
                    if isinstance(base_cls, dict):
                        valb = base_cls.get("final")
                        if isinstance(valb, int | float):
                            acc_base = float(valb)
                        elif isinstance(valb, dict):
                            nb = valb.get("correct_total")
                            db = valb.get("total")
                            if (
                                isinstance(nb, int | float)
                                and isinstance(db, int | float)
                                and float(db) > 0
                            ):
                                acc_base = float(nb) / float(db)
                    if isinstance(acc_run, float) and isinstance(acc_base, float):
                        delta_pp = (acc_run - acc_base) * 100.0
                        acc_pm["ratio_vs_baseline"] = delta_pp
                except Exception:
                    pass
                certificate["primary_metric"] = acc_pm
        except Exception:
            pass

    # Retry attaching from windows if classification fallback did not populate PM
    _attach_from_windows()

    # Ensure primary_metric has display_ci populated for schema invariants
    try:
        pm = (
            certificate.get("primary_metric", {})
            if isinstance(certificate.get("primary_metric"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
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
                    pm["display_ci"] = [1.0, 1.0]
    except Exception:
        pass
