from __future__ import annotations

import copy
from typing import Any, cast

import jsonschema
import pytest

from invarlock.public_contracts import load_verify_output_schema
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_normalization import normalize_baseline
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.verify_check_helpers_metrics import (
    _validate_primary_metric,
    _validate_primary_metric_policy,
)
from invarlock.reporting.verify_system_overhead import validate_system_overhead
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def _ppl_verifier_report() -> dict[str, object]:
    return {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 2.0,
            "final": 2.0,
            "ratio_vs_baseline": 1.0,
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 2.0}},
    }


def _accuracy_verifier_report() -> dict[str, object]:
    return {
        "primary_metric": {
            "kind": "accuracy",
            "preview": 0.7,
            "final": 0.8,
            "delta_vs_baseline_pp": 10.0,
        },
        "baseline_ref": {"primary_metric": {"kind": "accuracy", "final": 0.7}},
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report: report.pop("baseline_ref"),
            "same-kind baseline primary metric",
        ),
        (
            lambda report: report["primary_metric"].pop("ratio_vs_baseline"),
            "missing a finite positive primary_metric.ratio_vs_baseline",
        ),
        (
            lambda report: report["baseline_ref"].update(
                {"primary_metric": {"kind": "accuracy", "final": 0.5}}
            ),
            "same-kind baseline primary metric",
        ),
        (
            lambda report: report["primary_metric"].update({"final": 0.99}),
            "final must be finite and at least 1.0",
        ),
        (
            lambda report: report["baseline_ref"]["primary_metric"].update(
                {"final": 0.99}
            ),
            "baseline final must be at least 1.0",
        ),
    ],
)
def test_common_verifier_rejects_untrustworthy_ppl_comparisons(
    mutation, expected: str
) -> None:
    report = _ppl_verifier_report()
    mutation(report)

    assert expected in "\n".join(_validate_primary_metric(report))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report: report.pop("baseline_ref"),
            "accuracy baseline primary metric",
        ),
        (
            lambda report: report["primary_metric"].pop("delta_vs_baseline_pp"),
            "delta_vs_baseline_pp must be finite",
        ),
        (
            lambda report: report["baseline_ref"].update(
                {"primary_metric": {"kind": "ppl_causal", "final": 2.0}}
            ),
            "accuracy baseline primary metric",
        ),
        (
            lambda report: report["primary_metric"].update({"preview": -0.01}),
            "preview must be finite in [0, 1]",
        ),
        (
            lambda report: report["primary_metric"].update({"final": 1.01}),
            "final must be finite in [0, 1]",
        ),
        (
            lambda report: report["baseline_ref"]["primary_metric"].update(
                {"final": 1.01}
            ),
            "baseline final in [0, 1]",
        ),
    ],
)
def test_common_verifier_rejects_untrustworthy_accuracy_comparisons(
    mutation, expected: str
) -> None:
    report = _accuracy_verifier_report()
    mutation(report)

    assert expected in "\n".join(_validate_primary_metric(report))


def test_release_policy_does_not_default_missing_metric_flag_to_green() -> None:
    errors = _validate_primary_metric_policy(
        _ppl_verifier_report(),
        profile="release",
        recompute_validation_flags_fn=lambda _report: {},
    )

    assert errors == ["Primary metric policy gate failed (tier=balanced)."]


@pytest.mark.parametrize(
    "primary_metric",
    [
        {
            "kind": "ppl_causal",
            "final": 2.0,
            "delta_vs_baseline_pp": 0.0,
        },
        {"kind": "ppl_causal", "final": 0.99},
        {"kind": "accuracy", "final": 0.5, "ratio_vs_baseline": 1.0},
        {"kind": "accuracy", "final": -0.01},
        {"kind": "accuracy", "final": 1.01},
    ],
)
def test_canonical_comparison_baseline_rejects_wrong_family_or_domain(
    primary_metric: dict[str, object],
) -> None:
    baseline = {
        "run_id": "baseline-run",
        "model_id": "baseline-model",
        "primary_metric": primary_metric,
    }

    with pytest.raises(ValueError):
        normalize_baseline(baseline)


def _canonical_ppl_certificate() -> dict[str, Any]:
    subject = canonical_run_report(
        {
            "meta": {
                "model_id": "metric-trust-subject",
                "adapter": "hf_causal",
                "seed": 1,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "noop"},
            "guards": [],
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 2.0,
                    "final": 2.0,
                    "ratio_vs_baseline": 1.0,
                }
            },
            "evaluation_windows": {
                "preview": {
                    "window_ids": ["preview"],
                    "logloss": [0.6931471805599453],
                    "token_counts": [10],
                },
                "final": {
                    "window_ids": ["final"],
                    "logloss": [0.6931471805599453],
                    "token_counts": [10],
                },
            },
        }
    )
    baseline = canonical_baseline(
        {
            "meta": {
                "model_id": "metric-trust-subject",
                "adapter": "hf_causal",
                "seed": 1,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "noop"},
            "guards": [],
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 2.0,
                    "final": 2.0,
                }
            },
            "evaluation_windows": copy.deepcopy(subject["evaluation_windows"]),
        }
    )
    return cast(dict[str, Any], make_report(subject, baseline))


def test_report_schema_requires_family_specific_comparison() -> None:
    certificate = _canonical_ppl_certificate()
    assert validate_report(certificate)

    certificate["primary_metric"].pop("ratio_vs_baseline")

    assert not validate_report(certificate)


def test_report_schema_rejects_retired_paired_delta_summary() -> None:
    certificate = _canonical_ppl_certificate()
    assert validate_report(certificate)

    certificate.setdefault("metrics", {})["paired_delta_summary"] = {"mean": 0.0}

    assert not validate_report(certificate)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (
            {"baseline": 10.0, "edited": 12.0, "delta": 1.0, "ratio": 1.2},
            "delta does not match edited-baseline",
        ),
        (
            {"baseline": 10.0, "edited": 12.0, "delta": 2.0, "ratio": 1.1},
            "ratio does not match edited/baseline",
        ),
        (
            {"edited": 12.0, "delta": 2.0},
            "cannot declare delta or ratio without baseline",
        ),
    ],
)
def test_system_overhead_rejects_unbound_arithmetic(
    entry: dict[str, float], expected: str
) -> None:
    errors = validate_system_overhead({"system_overhead": {"latency_ms_p50": entry}})

    assert expected in "\n".join(errors)


def _verify_output_payload() -> dict[str, Any]:
    return {
        "format_version": "verify-v1",
        "summary": {"ok": True, "reason": "ok"},
        "results": [
            {
                "id": "report.json",
                "schema_version": "v1",
                "kind": "ppl_causal",
                "ok": True,
                "reason": "ok",
                "ratio_vs_baseline": 1.0,
                "ci": None,
                "recompute": {
                    "family": "ppl",
                    "performed": True,
                    "ok": True,
                    "reason": None,
                },
                "guard_warnings_present": False,
                "warning_count": 0,
            }
        ],
    }


def test_verify_output_schema_rejects_null_family_comparison() -> None:
    payload = _verify_output_payload()
    payload["results"][0]["ratio_vs_baseline"] = None

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=load_verify_output_schema())


def test_verify_output_schema_rejects_performed_recompute_with_unknown_result() -> None:
    payload = _verify_output_payload()
    payload["results"][0]["recompute"]["ok"] = None

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=load_verify_output_schema())
