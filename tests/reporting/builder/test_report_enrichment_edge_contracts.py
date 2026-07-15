from __future__ import annotations

import math

import pytest

from invarlock.reporting import report_enrichment


class _ExplodingMapping(dict):
    def get(self, *_args, **_kwargs):
        raise ValueError("malformed mapping")


def test_summary_value_sanitization_removes_control_characters() -> None:
    assert report_enrichment._sanitize_summary_value(None) is None
    assert report_enrichment._sanitize_summary_value(" a\n\tb ") == "a b"
    assert report_enrichment._sanitize_summary_value("\n") is None


def test_guard_metric_impact_rejects_decorative_partial_payloads() -> None:
    evaluation: dict = {}
    report_enrichment.attach_guard_metric_impact(
        evaluation,
        {},
        {"metrics": {"primary_metric": {"kind": "accuracy"}}},
        lambda _ctx, kind: {"value": 1.0, "kind": kind},
    )
    assert evaluation == {}

    untouched: dict = {}
    report_enrichment.attach_guard_metric_impact(
        untouched, {}, {}, lambda *_args: {"value": math.inf}
    )
    assert untouched == {}


def test_guard_metric_impact_enrichment_never_overwrites_canonical_evidence() -> None:
    canonical = {"metric_kind": "accuracy", "degradation": 0.0}
    evaluation = {"guard_metric_impact": canonical.copy()}

    report_enrichment.attach_guard_metric_impact(
        evaluation,
        {},
        {"metrics": {"primary_metric": {"kind": "accuracy"}}},
        lambda *_args: {"degradation": 1.0},
    )

    assert evaluation["guard_metric_impact"] == canonical


def test_guard_metric_impact_enrichment_attaches_complete_canonical_evidence() -> None:
    payload = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 10.0,
        "guarded_value": 10.1,
        "bare_facts": {"weighted_logloss_sum": math.log(10.0), "token_count": 1},
        "guarded_facts": {
            "weighted_logloss_sum": math.log(10.1),
            "token_count": 1,
        },
        "degradation": 0.01,
        "degradation_limit": 0.02,
        "display_value": 1.0,
        "display_unit": "percent",
        "evaluated": True,
        "passed": True,
        "checks": {"measurements_valid": True},
        "diagnostics": [],
        "source": "paired_control",
        "schedule_digest": "0" * 32,
    }
    evaluation: dict = {}

    report_enrichment.attach_guard_metric_impact(
        evaluation,
        {},
        {"metrics": {"primary_metric": {"kind": "ppl_causal"}}},
        lambda *_args: payload,
    )

    assert evaluation["guard_metric_impact"] is payload


def test_policy_digest_requires_subject_and_baseline_tiers() -> None:
    with pytest.raises(ValueError, match="subject report requires"):
        report_enrichment.attach_policy_digest(
            {}, {}, {}, {}, {}, lambda *_: {}, lambda _: "hash", "v1"
        )
    with pytest.raises(ValueError, match="baseline requires"):
        report_enrichment.attach_policy_digest(
            {},
            {"tier": "balanced"},
            {},
            {},
            {},
            lambda *_: {},
            lambda _: "hash",
            "v1",
        )


def test_policy_digest_handles_non_mapping_and_malformed_hysteresis() -> None:
    evaluation: dict = {}
    report_enrichment.attach_policy_digest(
        evaluation,
        {"tier": "balanced"},
        {"metrics": "bad", "variance": {"min_effect_lognll": 0.1}},
        {"meta": {"auto": {"tier": "conservative"}}},
        {},
        lambda tier, _policy: {"tier": tier},
        lambda payload: payload["tier"],
        "v1",
    )
    assert evaluation["policy_digest"]["changed"] is True
    assert evaluation["policy_digest"]["hysteresis"] == {
        "ppl": 0.0,
        "accuracy_delta_pp": 0.0,
    }

    malformed: dict = {}
    report_enrichment.attach_policy_digest(
        malformed,
        {"tier": "balanced"},
        {
            "metrics": {"pm_ratio": 1, "accuracy": 1},
            "variance": {"min_effect_lognll": 0.1},
        },
        {"meta": {"auto": {"tier": "balanced"}}},
        {},
        lambda tier, _policy: {"tier": tier},
        lambda payload: payload["tier"],
        "v1",
    )
    assert malformed["policy_digest"]["hysteresis"]["ppl"] == 0.0


def test_secondary_metric_and_classification_sanitization() -> None:
    evaluation: dict = {}
    report_enrichment.attach_secondary_metrics(
        evaluation,
        {
            "metrics": {
                "secondary_metrics": [
                    "noise",
                    {},
                    {"kind": "accuracy", "final": 0.5, "secret": "drop"},
                ]
            }
        },
    )
    assert evaluation["secondary_metrics"] == [{"kind": "accuracy", "final": 0.5}]

    report_enrichment.attach_classification(
        evaluation,
        {
            "metrics": {
                "classification": {
                    "final": {"correct_total": 1, "total": 2},
                    "counts_source": "pseudo_config",
                    "subgroups": {
                        "preview": {
                            "group_counts": {"a": 2, "bad": object()},
                            "correct_counts": {"a": 1},
                        },
                        "final": {
                            "group_counts": {"a": 2, "bad": 1},
                            "correct_counts": {"a": 2},
                        },
                    },
                }
            }
        },
    )
    assert evaluation["metrics"]["classification"]["n_correct"] == 1
    assert evaluation["metrics"]["classification"]["estimated"] is True
    assert evaluation["classification"]["subgroups"]["a"]["delta_pp"] == 50.0


def test_system_overhead_uses_fallback_telemetry_and_zero_baseline() -> None:
    evaluation: dict = {}
    report_enrichment.attach_system_overhead(
        evaluation,
        {"metrics": {"latency_ms_per_tok": 2.0}},
        {"metrics": {"latency_ms_p50": 0.0, "throughput_tok_per_s": 4.0}},
        {"throughput_tok_per_s": 5.0},
    )
    latency = evaluation["system_overhead"]["latency_ms_p50"]
    assert latency == {"edited": 2.0, "baseline": 0.0, "delta": 2.0}
    assert evaluation["system_overhead"]["throughput_sps"]["edited"] == 5.0


def test_display_ci_repair_distinguishes_point_and_estimated_defaults() -> None:
    report = {"primary_metric": {"final": 2.0}}
    report_enrichment.ensure_primary_metric_display_ci(report)
    assert report["primary_metric"]["display_ci"] == [2.0, 2.0]
    assert report["report_build"]["synthesized_fields"]

    estimated = {"primary_metric": {"display_ci": "bad"}}
    report_enrichment.ensure_primary_metric_display_ci(estimated)
    assert estimated["primary_metric"]["display_ci"] == [1.0, 1.0]
    assert estimated["primary_metric"]["estimated"] is True
    assert estimated["report_build"]["repaired_fields"]


def test_telemetry_summary_sanitizes_ids_and_records_split_fallback() -> None:
    evaluation = {
        "dataset": {"windows": {"preview": 1, "final": 2}, "hash": {}},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": [0.1, 0.2],
            "delta_vs_baseline_pp": 1.0,
        },
        "validation": {"primary_metric_acceptable": False},
        "provenance": {"dataset_split": "validation\n", "split_fallback": True},
    }
    report_enrichment.attach_telemetry_summary_line(
        evaluation,
        {"metrics": {"primary_metric": {"kind": "accuracy\n"}}},
        "run\n1",
    )
    summary = evaluation["telemetry"]["summary_line"]
    assert "run_id=run 1" in summary
    assert "split=validation*" in summary
    assert "delta_pp=1.000" in summary
    assert "gate=fail" in summary


def test_confidence_label_uses_resolved_accuracy_thresholds() -> None:
    high = report_enrichment.compute_confidence_label(
        {
            "validation": {"primary_metric_acceptable": True},
            "primary_metric": {"kind": "accuracy", "display_ci": [0.0, 0.5]},
            "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.5}},
        }
    )
    assert high["label"] == "High"
    assert high["basis"] == "accuracy"

    medium = report_enrichment.compute_confidence_label(
        {
            "validation": {"primary_metric_acceptable": True},
            "primary_metric": {
                "kind": "ppl_causal",
                "display_ci": [0.9, 1.0],
                "unstable": True,
            },
        }
    )
    assert medium["label"] == "Medium"


def test_enrichment_helpers_fail_safe_on_malformed_mappings() -> None:
    exploding = _ExplodingMapping()
    evaluation: dict = {}
    report_enrichment.attach_guard_metric_impact(
        evaluation,
        {},
        exploding,
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid impact")),
    )
    report_enrichment.attach_secondary_metrics(evaluation, exploding)
    report_enrichment.attach_classification(evaluation, exploding)
    report_enrichment.attach_system_overhead(evaluation, exploding, None, {})
    report_enrichment.ensure_primary_metric_display_ci(exploding)
    report_enrichment.attach_telemetry_summary_line(evaluation, exploding, "run")
    report_enrichment.attach_confidence_label(
        evaluation, lambda _report: (_ for _ in ()).throw(ValueError("bad confidence"))
    )
    assert "confidence" not in evaluation

    confidence = report_enrichment.compute_confidence_label(
        {
            "validation": {},
            "primary_metric": _ExplodingMapping(),
            "resolved_policy": _ExplodingMapping(),
        }
    )
    assert confidence["label"] == "Low"
