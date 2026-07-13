from __future__ import annotations

import copy

import pytest

from tests.reporting._support_canonical_reports import (
    canonical_run_report,
    refresh_runtime_policy_receipt,
)


def _canonical_source() -> dict:
    return {
        "meta": {
            "model_id": "fixture-model",
            "adapter": "hf_causal",
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "fixture-data",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {"name": "structured"},
        "guards": [{"name": "spectral", "passed": False}],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.1,
            }
        },
    }


def test_canonical_fixture_preserves_explicit_failed_guard_decision() -> None:
    report = canonical_run_report(_canonical_source())

    assert report["guards"][0]["passed"] is False
    assert report["meta"]["model_id"] == "fixture-model"
    auto = report["meta"]["auto"]
    assert auto is not None
    assert auto["tier"] == "balanced"
    assert report["context"]["profile"] == "dev"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source.update(dataset={"provider": "legacy"}), "data block"),
        (
            lambda source: source["metrics"].update(ppl_final=10.1),
            "metrics.primary_metric",
        ),
        (lambda source: source["meta"].pop("model_id"), "meta.model_id"),
        (lambda source: source["meta"].pop("auto"), "meta.auto.tier"),
        (lambda source: source.pop("context"), "context.profile"),
        (lambda source: source["guards"][0].pop("passed"), "guards\\[0\\].passed"),
        (
            lambda source: source["metrics"]["primary_metric"].pop("preview"),
            "primary_metric.preview",
        ),
    ],
)
def test_canonical_fixture_rejects_implicit_or_legacy_trust_inputs(
    mutation, message: str
) -> None:
    source = copy.deepcopy(_canonical_source())
    mutation(source)

    with pytest.raises(ValueError, match=message):
        canonical_run_report(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.pop("policy_resolution"), "policy_resolution"),
        (lambda report: report["meta"].pop("auto"), "meta.auto.tier"),
        (lambda report: report.pop("context"), "context.profile"),
        (lambda report: report["edit"].pop("name"), "edit.name"),
        (
            lambda report: report["policy_resolution"].update(tier="aggressive"),
            "policy_resolution.tier",
        ),
        (lambda report: report.pop("resolved_policy"), "resolved_policy"),
    ],
)
def test_receipt_refresh_rejects_missing_or_mismatched_context(
    mutation, message: str
) -> None:
    report = canonical_run_report(_canonical_source())
    mutation(report)

    with pytest.raises(ValueError, match=message):
        refresh_runtime_policy_receipt(report)
