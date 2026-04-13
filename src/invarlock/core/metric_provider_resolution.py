from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .metric_kind_contract import normalize_metric_kind
from .provider_config import resolve_provider_kind_and_kwargs

_METRIC_LOOKUP_ERRORS = (AttributeError, KeyError, TypeError, ValueError)
_METRIC_COERCION_ERRORS = (OverflowError, TypeError, ValueError)


def resolve_metric_and_provider(
    cfg: Any,
    model_profile: Any,
    *,
    resolved_loss_type: str | None = None,
    metric_kind_override: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    """Resolve metric kind, provider kind, and metric options from config."""

    def _metric_value(key: str) -> Any:
        if isinstance(metric_cfg, Mapping):
            return metric_cfg.get(key)
        get_value = getattr(metric_cfg, "get", None)
        if callable(get_value):
            try:
                return get_value(key)
            except _METRIC_LOOKUP_ERRORS:
                pass
        try:
            return getattr(metric_cfg, key)
        except _METRIC_LOOKUP_ERRORS:
            return None

    provider_val = None
    try:
        provider_val = cfg.dataset.provider
    except _METRIC_LOOKUP_ERRORS:
        provider_val = None

    provider_kind, _provider_kwargs = resolve_provider_kind_and_kwargs(provider_val)

    if not provider_kind and hasattr(model_profile, "default_provider"):
        provider_kind = model_profile.default_provider
    if not provider_kind:
        provider_kind = "wikitext2"

    metric_cfg = None
    try:
        section_fn = getattr(cfg, "section", None)
        if callable(section_fn):
            eval_section = section_fn("eval") or {}
            if isinstance(eval_section, Mapping):
                metric_cfg = eval_section.get("metric")
        else:
            try:
                metric_cfg = cfg.eval.metric
            except _METRIC_LOOKUP_ERRORS:
                metric_cfg = None
    except _METRIC_LOOKUP_ERRORS:
        metric_cfg = None

    metric_kind = None
    if isinstance(metric_kind_override, str) and metric_kind_override.strip():
        metric_kind = normalize_metric_kind(metric_kind_override, allow_auto=True)

    reps = None
    ci_level = None
    if metric_kind is None and metric_cfg is not None:
        metric_kind = _metric_value("kind")
        reps = _metric_value("reps")
        ci_level = _metric_value("ci_level")

    normalized_kind: str | None
    normalized_kind = normalize_metric_kind(metric_kind, allow_auto=True)

    if not normalized_kind:
        default_metric = getattr(model_profile, "default_metric", None)
        normalized_kind = normalize_metric_kind(default_metric, allow_auto=True)

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
    except _METRIC_COERCION_ERRORS:
        pass
    try:
        if ci_level is not None:
            metric_opts["ci_level"] = float(ci_level)
    except _METRIC_COERCION_ERRORS:
        pass

    return str(normalized_kind), str(provider_kind), metric_opts
