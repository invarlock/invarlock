from __future__ import annotations

from collections.abc import Sequence


def pm(cert: dict) -> dict:
    """Return primary_metric with safe defaults for tests.

    Ensures display_ci is present and 2-length for comparisons.
    """
    assert "primary_metric" in cert, "missing primary_metric"
    m = dict(cert["primary_metric"])  # shallow copy
    if "display_ci" not in m and isinstance(m.get("final"), int | float):
        m["display_ci"] = [m["final"], m["final"]]
    return m


def ppl_report(
    *,
    kind: str = "ppl_causal",
    preview: float | None = None,
    final: object = 1.8,
    ratio_vs_baseline: object | None = None,
    display_ci: list[object] | None = None,
    metrics_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    primary_metric: dict[str, object] = {"kind": kind, "final": final}
    if preview is not None:
        primary_metric["preview"] = preview
    if ratio_vs_baseline is not None:
        primary_metric["ratio_vs_baseline"] = ratio_vs_baseline
    if display_ci is not None:
        primary_metric["display_ci"] = display_ci
    metrics = {"primary_metric": primary_metric}
    if metrics_extra:
        metrics.update(metrics_extra)
    return {"metrics": metrics}


def baseline_ref(final: object) -> dict[str, object]:
    return {"primary_metric": {"final": final}}


def classification_report(
    final: object,
    *,
    model_id: str = "awesome-vqa",
) -> dict[str, object]:
    return {
        "metrics": {"classification": {"final": final}},
        "meta": {"model_id": model_id},
    }


def classification_baseline_raw(final: object) -> dict[str, object]:
    return {"metrics": {"classification": {"final": final}}}


def window_report(
    *,
    preview_logloss: Sequence[float] = (1.0, 2.0),
    final_logloss: Sequence[float] = (3.0,),
    preview_token_counts: Sequence[int] = (1, 1),
    final_token_counts: Sequence[int] = (2,),
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "metrics": metrics or {},
        "evaluation_windows": {
            "preview": {
                "logloss": list(preview_logloss),
                "token_counts": list(preview_token_counts),
            },
            "final": {
                "logloss": list(final_logloss),
                "token_counts": list(final_token_counts),
            },
        },
    }
