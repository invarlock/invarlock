import pytest

import invarlock.reporting.report_normalization as report_normalization
from invarlock.reporting.policy_utils import _resolve_policy_tier
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.utils import _infer_scope_from_modules
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_infer_scope_from_modules_variations():
    assert _infer_scope_from_modules([]) == "unknown"
    mix = ["layer.attn.q_proj", "embeddings.wte"]
    assert _infer_scope_from_modules(mix) in {"attn+embed", "embed+attn"}


def test_resolve_policy_tier_rejects_context_auto():
    report = {"context": {"auto": {"tier": "Conservative"}}}
    with pytest.raises(ValueError, match="meta.auto.tier"):
        _resolve_policy_tier(report)


def test_make_evaluation_report_with_explicit_preview_and_final(monkeypatch):
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 10.0,
            }
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
    }
    baseline = create_empty_report()
    baseline["meta"].update(
        {
            "model_id": "m",
            "adapter": "hf_causal",
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        }
    )
    baseline["context"] = {"profile": "dev"}
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.1,
        "final": 10.2,
    }
    # Bypass schema rigor to focus on branch
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(canonical_run_report(report), canonical_baseline(baseline))
    assert isinstance(cert, dict)
