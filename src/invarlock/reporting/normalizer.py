from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from .report_types import (
    Artifacts,
    DataConfig,
    EditDeltas,
    EditInfo,
    EvalMetrics,
    Flags,
    GuardReport,
    MetaData,
    RunReport,
)


def _str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
        return s if s is not None else default
    except Exception:
        return default


def _as_mapping(x: Any) -> Mapping[str, Any]:
    return x if isinstance(x, Mapping) else {}


def normalize_run_report(report: Mapping[str, Any] | RunReport) -> RunReport:
    """Coerce an arbitrary report-like mapping into a canonical RunReport.

    This is the single entry point for converting pre-canonical or loosely-typed
    data into the strict PM-only RunReport shape used by certificate/report.
    """
    src = _as_mapping(report)

    # ---- meta ----
    meta_in = _as_mapping(src.get("meta"))
    ts = _str(meta_in.get("ts") or datetime.now().isoformat())
    try:
        seed_value = int(meta_in.get("seed", 42))
    except Exception:
        seed_value = 42
    meta_dict: dict[str, Any] = {
        "model_id": _str(meta_in.get("model_id")),
        "adapter": _str(meta_in.get("adapter")),
        "commit": _str(meta_in.get("commit")),
        "seed": seed_value,
        "device": _str(meta_in.get("device", "cpu")),
        "ts": ts,
        "auto": meta_in.get("auto") if isinstance(meta_in.get("auto"), dict) else None,
    }
    # Preserve additional provenance knobs used by certificate/digests.
    for key in (
        "policy_overrides",
        "overrides",
        "plugins",
        "config",
        "seeds",
        "determinism",
        "env_flags",
        "cuda_flags",
        "tokenizer_hash",
        "model_profile",
    ):
        if key in meta_in:
            meta_dict[key] = meta_in.get(key)
    meta = cast(MetaData, meta_dict)

    # ---- data ----
    data_in = _as_mapping(src.get("data"))
    data_dict: dict[str, Any] = {
        "dataset": _str(data_in.get("dataset")),
        "split": _str(data_in.get("split", "validation")),
        "seq_len": int(data_in.get("seq_len", 0) or 0),
        "stride": int(data_in.get("stride", 0) or 0),
        "preview_n": int(data_in.get("preview_n", 0) or 0),
        "final_n": int(data_in.get("final_n", 0) or 0),
    }
    for k in (
        "tokenizer_name",
        "tokenizer_hash",
        "vocab_size",
        "bos_token",
        "eos_token",
        "pad_token",
        "add_prefix_space",
        "dataset_hash",
        "preview_hash",
        "final_hash",
        "preview_total_tokens",
        "final_total_tokens",
        "masked_tokens_total",
        "masked_tokens_preview",
        "masked_tokens_final",
        "loss_type",
    ):
        if k in data_in:
            data_dict[k] = data_in.get(k)
    data = cast(DataConfig, data_dict)

    # ---- edit ----
    edit_in = _as_mapping(src.get("edit"))
    deltas_in = _as_mapping(edit_in.get("deltas"))
    spars_val = deltas_in.get("sparsity")
    deltas = EditDeltas(
        params_changed=int(deltas_in.get("params_changed", 0) or 0),
        sparsity=(float(spars_val) if isinstance(spars_val, int | float) else None),
        bitwidth_map=(
            deltas_in.get("bitwidth_map")
            if isinstance(deltas_in.get("bitwidth_map"), dict)
            else None
        ),
        layers_modified=int(deltas_in.get("layers_modified", 0) or 0),
    )
    edit = EditInfo(
        name=_str(edit_in.get("name")),
        plan_digest=_str(edit_in.get("plan_digest")),
        deltas=deltas,
    )

    # ---- guards ----
    guards_in = src.get("guards")
    guards: list[dict[str, Any]] = []
    if isinstance(guards_in, list):
        guards = [g for g in guards_in if isinstance(g, dict)]

    # ---- metrics ----
    m_in = _as_mapping(src.get("metrics"))
    pm_in = _as_mapping(m_in.get("primary_metric"))
    metrics_out: dict[str, Any] = {"primary_metric": dict(pm_in)}

    # accuracy-style fallback (from classification aggregates)
    if not metrics_out["primary_metric"]:
        cls = (
            m_in.get("classification")
            if isinstance(m_in.get("classification"), dict)
            else None
        )
        if isinstance(cls, dict):
            point = None
            fin = cls.get("final")
            if isinstance(fin, int | float):
                point = float(fin)
            elif isinstance(fin, dict):
                num = fin.get("correct_total")
                den = fin.get("total")
                if (
                    isinstance(num, int | float)
                    and isinstance(den, int | float)
                    and float(den) > 0
                ):
                    point = float(num) / float(den)
            if isinstance(point, float):
                # infer kind from model_id hint when available
                model_id = _str(meta.get("model_id", "")).lower()
                kind = "vqa_accuracy" if "vqa" in model_id else "accuracy"
                pm_acc: dict[str, Any] = {
                    "kind": kind,
                    "unit": "accuracy",
                    "direction": "higher",
                    "aggregation_scope": "example",
                    "paired": True,
                    "gating_basis": "point",
                    "final": point,
                }
                # include n_final when available
                if isinstance(fin, dict) and isinstance(fin.get("total"), int | float):
                    # safe: pm_acc is a plain dict
                    pm_acc["n_final"] = int(fin["total"])
                metrics_out["primary_metric"] = pm_acc

    # carry through selected non-PM fields when present
    for k in (
        "latency_ms_per_tok",
        "latency_ms_p50",
        "latency_ms_p95",
        "memory_mb_peak",
        "throughput_sps",
        "spectral",
        "rmt",
        "invariants",
        "logloss_delta_ci",
        "bootstrap",
        "reduction",
        "moe",
        "window_overlap_fraction",
        "window_match_fraction",
        "paired_windows",
        "paired_delta_summary",
        "window_pairing_reason",
        "window_pairing_preview",
        "window_pairing_final",
        "total_tokens",
        "preview_total_tokens",
        "final_total_tokens",
    ):
        if k in m_in:
            metrics_out[k] = m_in.get(k)
    metrics = cast(EvalMetrics, metrics_out)

    # ---- artifacts ----
    a_in = _as_mapping(src.get("artifacts"))
    artifacts_dict: dict[str, Any] = {
        "events_path": _str(a_in.get("events_path")),
        "logs_path": _str(a_in.get("logs_path")),
        "checkpoint_path": a_in.get("checkpoint_path")
        if a_in.get("checkpoint_path") is None
        or isinstance(a_in.get("checkpoint_path"), str)
        else None,
    }
    artifacts = cast(Artifacts, artifacts_dict)

    # ---- flags ----
    f_in = _as_mapping(src.get("flags"))
    flags = cast(
        Flags,
        {
            "guard_recovered": bool(f_in.get("guard_recovered", False)),
            "rollback_reason": f_in.get("rollback_reason"),
        },
    )

    out: RunReport = RunReport(
        meta=meta,
        data=data,
        edit=edit,
        guards=cast(list[GuardReport], guards),
        metrics=metrics,
        artifacts=artifacts,
        flags=flags,
    )

    # keep evaluation_windows if provided (for deeper pairing-based features)
    ew = src.get("evaluation_windows")
    if isinstance(ew, dict):
        out["evaluation_windows"] = ew

    # keep guard_overhead if provided (for quality_overhead derivation downstream)
    go = src.get("guard_overhead")
    if isinstance(go, Mapping):
        out["guard_overhead"] = dict(go)

    # keep provenance when present (dataset_split, provider_digest, etc.)
    prov = src.get("provenance")
    if isinstance(prov, Mapping):
        out["provenance"] = dict(prov)

    return out


__all__ = ["normalize_run_report"]
