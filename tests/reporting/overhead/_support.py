from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def overhead_run_report(
    *,
    metrics: Mapping[str, Any],
    edit_name: str,
    tier: str,
    profile: str,
) -> dict[str, Any]:
    """Build one explicit canonical run fixture for overhead-only tests."""

    return canonical_run_report(
        {
            "meta": {
                "model_id": "overhead-test-model",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 1,
                "auto": {"tier": tier},
            },
            "context": {"profile": profile},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {
                "name": edit_name,
                "plan_digest": "overhead-test-plan",
                "deltas": {"params_changed": 0, "layers_modified": 0},
            },
            "guards": [],
            "metrics": dict(metrics),
            "evaluation_windows": {
                "preview": {
                    "window_ids": [1],
                    "logloss": [2.302585093],
                    "token_counts": [1],
                },
                "final": {
                    "window_ids": [2],
                    "logloss": [2.302585093],
                    "token_counts": [1],
                },
            },
            "artifacts": {
                "events_path": "",
                "logs_path": "",
                "checkpoint_path": None,
            },
            "flags": {"guard_recovered": False, "rollback_reason": None},
        }
    )


def overhead_baseline_report(
    *,
    metrics: Mapping[str, Any],
    tier: str,
    profile: str,
) -> dict[str, Any]:
    """Build one explicit canonical no-op baseline for overhead-only tests."""

    payload = overhead_run_report(
        metrics=metrics,
        edit_name="noop",
        tier=tier,
        profile=profile,
    )
    return canonical_baseline(payload)


__all__ = ["overhead_baseline_report", "overhead_run_report"]
