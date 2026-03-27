from __future__ import annotations

from typing import Any


def resolve_metric_and_provider(
    cfg: Any,
    model_profile: Any,
    *,
    resolved_loss_type: str | None = None,
    metric_kind_override: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    """Resolve metric kind, provider kind, and metric options from config."""

    provider_val = None
    try:
        provider_val = cfg.dataset.provider
    except Exception:
        provider_val = None

    provider_kind = None
    if isinstance(provider_val, str) and provider_val:
        provider_kind = provider_val
    else:
        try:
            provider_kind = provider_val.kind
        except Exception:
            try:
                provider_kind = provider_val.get("kind")  # type: ignore[attr-defined]
            except Exception:
                provider_kind = None

    if not provider_kind and hasattr(model_profile, "default_provider"):
        provider_kind = model_profile.default_provider
    if not provider_kind:
        provider_kind = "wikitext2"

    metric_cfg = None
    try:
        eval_section = cfg.eval
        metric_cfg = getattr(eval_section, "metric", None)
    except Exception:
        metric_cfg = None

    metric_kind = None
    if isinstance(metric_kind_override, str) and metric_kind_override.strip():
        metric_override = metric_kind_override.strip().lower()
        if metric_override != "auto":
            metric_kind = metric_override

    reps = None
    ci_level = None
    if metric_kind is None and metric_cfg is not None:
        try:
            metric_kind = (
                metric_cfg.get("kind")
                if isinstance(metric_cfg, dict)
                else metric_cfg.kind
            )
        except Exception:
            metric_kind = None
        try:
            reps = (
                metric_cfg.get("reps")
                if isinstance(metric_cfg, dict)
                else metric_cfg.reps
            )
        except Exception:
            reps = None
        try:
            ci_level = (
                metric_cfg.get("ci_level")
                if isinstance(metric_cfg, dict)
                else metric_cfg.ci_level
            )
        except Exception:
            ci_level = None

    if isinstance(metric_kind, str) and metric_kind:
        normalized_kind = metric_kind.strip().lower()
        if normalized_kind == "auto":
            normalized_kind = None
    else:
        normalized_kind = None

    if not normalized_kind:
        default_metric = getattr(model_profile, "default_metric", None)
        if isinstance(default_metric, str) and default_metric.strip():
            normalized_kind = default_metric.strip().lower()

    if not normalized_kind:
        loss_type = (resolved_loss_type or "").strip().lower()
        if loss_type == "classification":
            normalized_kind = "accuracy"
        elif loss_type == "seq2seq":
            normalized_kind = "ppl_seq2seq"
        elif loss_type == "mlm":
            normalized_kind = "ppl_mlm"
        else:
            normalized_kind = "ppl_causal"

    metric_opts: dict[str, float] = {}
    try:
        if reps is not None:
            metric_opts["reps"] = float(int(reps))
    except Exception:
        pass
    try:
        if ci_level is not None:
            metric_opts["ci_level"] = float(ci_level)
    except Exception:
        pass

    return str(normalized_kind), str(provider_kind), metric_opts
