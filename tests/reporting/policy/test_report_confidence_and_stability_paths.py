from __future__ import annotations

import math

from invarlock.guards.authority import DEFAULT_GUARD_AUTHORITY
from invarlock.reporting.report_primary_metric_policy import is_ppl_kind as _is_ppl_kind
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.report_types import AutoConfig, RunReport, create_empty_report
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting._support_primary_metric import independent_slice_summary


def _mk_report(
    *, replicates: int = 50, bootstrap_method: str = "percentile"
) -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = AutoConfig(
        enabled=False,
        tier="balanced",
        probes_used=0,
        target_pm_ratio=None,
    )
    r["context"] = {"profile": "dev"}
    r["edit"]["name"] = "structured"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["data"]["preview_n"] = 2
    r["data"]["final_n"] = 2
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    r["metrics"]["bootstrap"] = {
        "method": bootstrap_method,
        "replicates": replicates,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
    }
    r["metrics"]["preview_final_slice_delta_summary"] = independent_slice_summary(
        0.0,
        preview_windows=2,
        final_windows=2,
    )
    r["metrics"]["preview_total_tokens"] = 50
    r["metrics"]["final_total_tokens"] = 50
    r["metrics"]["logloss_delta"] = 0.0
    r["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)
    r["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2],
            "logloss": [math.log(10.0), math.log(10.0)],
            "token_counts": [100, 100],
        }
    }
    return r


def _mk_baseline() -> dict:
    return {
        "run_id": "base",
        "model_id": "m",
        "meta": {
            "seed": 0,
            "model_id": "m",
            "adapter": "hf",
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [math.log(10.0), math.log(10.0)],
                "token_counts": [100, 100],
            }
        },
        "data": {
            "seq_len": 8,
            "preview_n": 2,
            "final_n": 2,
            "dataset": "unit",
            "split": "validation",
            "stride": 8,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_is_ppl_kind_accepts_only_canonical_catalog_entries_and_handles_bad_str() -> (
    None
):
    assert _is_ppl_kind("ppl_mlm")
    assert not _is_ppl_kind("perplexity")
    assert not _is_ppl_kind("ppl")
    assert not _is_ppl_kind("accuracy")

    class _Bad:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert not _is_ppl_kind(_Bad())


def test_make_evaluation_report_replicates_zero_keeps_independent_pairing_and_fills_counts(
    monkeypatch,
) -> None:
    report = _mk_report(replicates=0)
    baseline = _mk_baseline()

    # Force _as_count float/negative branches; should fall back to data.preview_n/final_n.
    report["metrics"]["stats"] = {"requested_preview": 1.2, "requested_final": -1}

    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        lambda *_a, **_k: (_a, _k),  # should not be called when replicates=0
    )
    cert = make_report(report, baseline)
    assert validate_report(cert)
    stats = cert["dataset"]["windows"]["stats"]
    assert stats["pairing"] == "independent_preview_final"
    assert stats["requested_preview"] == 2
    assert stats["requested_final"] == 2


def test_make_evaluation_report_uses_bca_when_method_explicit(monkeypatch) -> None:
    report = _mk_report(replicates=10, bootstrap_method="bca")
    baseline = _mk_baseline()

    seen: dict[str, object] = {}

    def _fake_ci(*_a, method: str, **_k):  # noqa: ANN001
        seen["method"] = method
        return (-0.01, 0.01)

    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci", _fake_ci
    )
    cert = make_report(report, baseline)
    assert validate_report(cert)
    assert seen.get("method") == "bca"


def test_make_evaluation_report_percentile_method_used_when_windows_small(
    monkeypatch,
) -> None:
    report = _mk_report(replicates=10, bootstrap_method="percentile")
    baseline = _mk_baseline()

    seen: dict[str, object] = {}

    def _fake_ci(*_a, method: str, **_k):  # noqa: ANN001
        seen["method"] = method
        return (-0.01, 0.01)

    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci", _fake_ci
    )
    cert = make_report(report, baseline)
    assert validate_report(cert)
    assert seen.get("method") == "percentile"


def test_make_evaluation_report_marks_unstable_when_token_floor_violated(
    monkeypatch,
) -> None:
    report = _mk_report(replicates=200)
    baseline = _mk_baseline()
    report["metrics"]["preview_total_tokens"] = 10
    report["metrics"]["final_total_tokens"] = 10

    tier_policies = {
        "balanced": {
            "guard_authority": dict(DEFAULT_GUARD_AUTHORITY),
            "metrics": {"pm_ratio": {"min_tokens": 100}},
        }
    }
    monkeypatch.setattr(
        "invarlock.core.auto_tuning.get_tier_policies",
        lambda *_args, **_kwargs: tier_policies,
    )
    monkeypatch.setattr(
        "invarlock.reporting.report_primary_metric_analysis.get_tier_policies",
        lambda *_args, **_kwargs: tier_policies,
    )
    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )
    cert = make_report(report, baseline)
    assert validate_report(cert)
    assert bool(cert["primary_metric"]["unstable"]) is True
