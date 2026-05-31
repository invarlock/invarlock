from __future__ import annotations

from copy import deepcopy

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.reporting import (
    dataset_hashing,
    guards_invariants,
    guards_spectral,
    policy_utils,
    primary_metric_utils,
    report_edit_summary,
    report_normalization,
)
from invarlock.reporting import (
    report_make as cert,
)

__all__ = [
    "_base_baseline",
    "_base_report",
    "_patch_common",
    "_stub_evaluation_report_extractors",
]


def _base_report() -> dict:
    return {
        "run_id": "run-1",
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
            "windows": {"preview": 2, "final": 2},
        },
        "artifacts": {"events_path": "", "logs_path": "", "generated_at": ""},
        "guards": [],
        "guard_overhead": {},
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [10.0, 10.0],
            },
            "paired_delta_summary": {"mean": 0.0, "degenerate": False},
            "logloss_delta_ci": (0.0, 0.0),
            "bootstrap": {"replicates": 400, "coverage": {"preview": {"used": 0}}},
            "window_plan": {"profile": "dev"},
            "spectral": {"caps_applied": 0, "max_caps": 5, "summary": {}},
            "rmt": {"stable": True},
            "variance": {"enabled": False},
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [0.1, 0.2],
                "token_counts": [10, 12],
            },
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.15, 0.25],
                "token_counts": [10, 12],
            },
        },
    }


def _base_baseline() -> dict:
    base = _base_report()
    base["run_id"] = "base-1"
    base["metrics"]["primary_metric"]["final"] = 10.0
    return base


def _patch_common(monkeypatch, report, baseline):
    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda _r: report,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda _b: baseline, raising=False
    )
    monkeypatch.setattr(
        primary_metric_utils,
        "attach_primary_metric",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        primary_metric_mod,
        "compute_primary_metric_from_report",
        lambda *args, **kwargs: {},
    )


def _stub_evaluation_report_extractors(
    monkeypatch,
    *,
    dataset_info=None,
    invariants=None,
    spectral=None,
    rmt=None,
    variance=None,
    structure=None,
    policies_payload=None,
    resolved_policy=None,
):
    dataset_info = dataset_info or {"hash": {}, "windows": {}}
    invariants = invariants or {"status": "ok"}
    spectral = spectral or {"caps_applied": 0}
    rmt = rmt or {"stable": True}
    variance = variance or {"enabled": False}
    structure = structure or {
        "compression_diagnostics": {"execution_status": "successful"}
    }
    policies_payload = policies_payload or {}
    resolved_policy = resolved_policy or {"spectral": {}, "variance": {}}

    monkeypatch.setattr(
        dataset_hashing, "_extract_dataset_info", lambda *_: deepcopy(dataset_info)
    )
    monkeypatch.setattr(
        guards_invariants,
        "_extract_invariants",
        lambda *args, **kwargs: invariants,
        raising=False,
    )
    monkeypatch.setattr(
        guards_spectral,
        "_extract_spectral_analysis",
        lambda *_: spectral,
        raising=False,
    )
    monkeypatch.setattr(cert, "_extract_rmt_analysis", lambda *_: rmt, raising=False)
    monkeypatch.setattr(
        cert, "_extract_variance_analysis", lambda *_: variance, raising=False
    )
    monkeypatch.setattr(
        report_edit_summary,
        "extract_structural_deltas",
        lambda *_: deepcopy(structure),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_extract_effective_policies",
        lambda *_: deepcopy(policies_payload),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils, "_extract_policy_overrides", lambda *_: ["manual"], raising=False
    )
    monkeypatch.setattr(
        policy_utils,
        "_build_resolved_policies",
        lambda *args, **kwargs: deepcopy(resolved_policy),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_compute_policy_digest",
        lambda *_: "resolved-digest",
        raising=False,
    )
