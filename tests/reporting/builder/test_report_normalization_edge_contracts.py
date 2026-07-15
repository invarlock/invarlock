from __future__ import annotations

import math

import pytest

from invarlock.reporting import report_normalization


class _Unstringable:
    def __str__(self) -> str:
        raise ValueError("cannot stringify")


def _baseline(kind: str = "ppl_causal") -> dict:
    return {
        "run_id": " run ",
        "model_id": " model ",
        "primary_metric": {"kind": kind, "preview": 2.0, "final": 2.0},
    }


def test_normalization_scalar_helpers_are_fail_closed() -> None:
    assert report_normalization._finite_float_or_none("1") is None
    assert report_normalization._finite_float_or_none(math.inf) is None
    assert report_normalization._generate_run_id({"meta": "bad"})
    with pytest.raises(ValueError, match="finite numeric"):
        report_normalization._baseline_coerce_valid_ppl("bad", label="ppl")
    with pytest.raises(ValueError, match=">= 1.0"):
        report_normalization._baseline_coerce_valid_ppl(0.5, label="ppl")
    assert report_normalization._baseline_normalize_kind(_Unstringable()) == ""


def test_logloss_baseline_derivation_uses_positive_token_weights() -> None:
    assert report_normalization._baseline_derive_ppl_from_logloss_block([]) is None
    assert report_normalization._baseline_derive_ppl_from_logloss_block({}) is None
    assert (
        report_normalization._baseline_derive_ppl_from_logloss_block(
            {"logloss": ["bad"]}
        )
        is None
    )
    weighted = report_normalization._baseline_derive_ppl_from_logloss_block(
        {"logloss": [0.0, math.log(4)], "token_counts": [1, 3]}
    )
    assert weighted == pytest.approx(4**0.75)
    fallback = report_normalization._baseline_derive_ppl_from_logloss_block(
        {"logloss": [0.0, math.log(4)], "token_counts": [0, "bad"]}
    )
    assert fallback == pytest.approx(2.0)
    assert (
        report_normalization._baseline_derive_ppl_from_logloss_block(
            {"logloss": [math.inf]}
        )
        is None
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"run_id": ""}, "run_id must be non-empty"),
        ({"model_id": ""}, "model_id must be non-empty"),
        ({"primary_metric": {}}, "primary_metric must be a non-empty"),
        ({"primary_metric": {"kind": "custom"}}, "unsupported metric kind"),
        (
            {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "final": 2,
                    "delta_vs_baseline_pp": 1,
                }
            },
            "PPL baselines cannot contain",
        ),
        (
            {
                "primary_metric": {
                    "kind": "accuracy",
                    "final": 0.5,
                    "ratio_vs_baseline": 1,
                }
            },
            "Accuracy baselines cannot contain",
        ),
        (
            {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "final": 2,
                    "ratio_vs_baseline": 0,
                }
            },
            "ratio must be finite and positive",
        ),
        (
            {"primary_metric": {"kind": "accuracy", "final": 2}},
            "final must be finite in",
        ),
        (
            {
                "primary_metric": {
                    "kind": "accuracy",
                    "final": 0.5,
                    "preview": 2,
                }
            },
            "preview must be finite in",
        ),
        (
            {
                "primary_metric": {
                    "kind": "accuracy",
                    "final": 0.5,
                    "delta_vs_baseline_pp": math.inf,
                }
            },
            "delta must be finite",
        ),
    ],
)
def test_canonical_baseline_rejects_invalid_metric_contracts(
    mutation: dict, message: str
) -> None:
    baseline = _baseline()
    baseline.update(mutation)
    with pytest.raises(ValueError, match=message):
        report_normalization._canonical_baseline_output(baseline)


def test_canonical_baseline_normalizes_accuracy_and_ppl() -> None:
    accuracy = _baseline("accuracy")
    accuracy["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.5,
        "final": 0.6,
    }
    accuracy["ppl_final"] = 9
    normalized = report_normalization.normalize_baseline(accuracy)
    assert normalized["run_id"] == "run"
    assert "ppl_final" not in normalized

    ppl = _baseline()
    ppl.pop("ppl_final", None)
    normalized = report_normalization.normalize_baseline(ppl)
    assert normalized["ppl_final"] == 2.0
    assert normalized["ppl_preview"] == 2.0


def test_normalize_baseline_rejects_legacy_and_missing_ppl() -> None:
    with pytest.raises(ValueError, match="legacy"):
        report_normalization.normalize_baseline({"schema_version": "v1"})
    baseline = _baseline()
    baseline["primary_metric"].pop("final")
    with pytest.raises(ValueError, match="finite ppl_final"):
        report_normalization.normalize_baseline(baseline)


def test_run_report_baseline_contract_rejects_missing_and_unknown_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = {
        "meta": {},
        "data": {},
        "edit": {},
        "guards": [],
        "metrics": {},
        "artifacts": {},
        "flags": {},
    }
    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda report: report,
    )
    with pytest.raises(ValueError, match="requires primary_metric"):
        report_normalization.normalize_baseline(marker)
    marker["metrics"] = {"primary_metric": {"kind": "custom"}}
    with pytest.raises(ValueError, match="unsupported metric kind"):
        report_normalization.normalize_baseline(marker)
