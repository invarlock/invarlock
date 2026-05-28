import math

import pytest

from invarlock.reporting import (
    report_primary_metric_policy as primary_metric_policy_mod,
)
from invarlock.reporting import report_provenance as provenance_mod
from invarlock.reporting import report_schema as schema_mod
from invarlock.reporting import report_validation_allowlist as allowlist_mod


def test_load_validation_allowlist_prefers_contract_file(monkeypatch):
    monkeypatch.setattr(
        allowlist_mod, "load_json_contract", lambda _filename: ["a", "b"]
    )
    keys = allowlist_mod.load_validation_allowlist()
    assert "a" in keys and "b" in keys

    monkeypatch.setattr(
        allowlist_mod, "load_json_contract", lambda _filename: {"bad": True}
    )
    with pytest.raises(
        allowlist_mod.ValidationAllowlistContractError,
        match="non-empty JSON array of strings",
    ):
        allowlist_mod.load_validation_allowlist()


def test_load_validation_allowlist_strict_prefers_contract_file(monkeypatch):
    monkeypatch.setattr(
        allowlist_mod, "load_json_contract", lambda _filename: ["a", "b"]
    )

    assert allowlist_mod.load_validation_allowlist_strict() == {"a", "b"}


def test_apply_validation_allowlist_schema_fails_closed() -> None:
    original = schema_mod.REPORT_JSON_SCHEMA.get("properties")
    try:
        schema_mod.REPORT_JSON_SCHEMA["properties"] = None
        with pytest.raises(RuntimeError, match="properties must be a mapping"):
            allowlist_mod.apply_validation_allowlist_schema(
                schema_mod.REPORT_JSON_SCHEMA, {"primary_metric_acceptable"}
            )
    finally:
        schema_mod.REPORT_JSON_SCHEMA["properties"] = original


def test_compute_edit_digest_paths():
    quant_digest = provenance_mod.compute_edit_digest(
        {"edit": {"name": "quant_rtn", "plan": {}}}
    )
    assert quant_digest["family"] == "quantization"

    noop_digest = provenance_mod.compute_edit_digest(
        {"edit": {"name": "noop", "plan": {}}}
    )
    assert noop_digest["family"] == "report_only"


def test_is_ppl_kind_variants():
    assert primary_metric_policy_mod.is_ppl_kind("ppl_causal")
    assert primary_metric_policy_mod.is_ppl_kind("ppl_seq2seq")
    assert not primary_metric_policy_mod.is_ppl_kind("perplexity")
    assert not primary_metric_policy_mod.is_ppl_kind("accuracy")


def test_fallback_paired_windows():
    cov = {"preview": {"used": 7}}
    assert primary_metric_policy_mod.fallback_paired_windows(0, cov) == 7
    assert primary_metric_policy_mod.fallback_paired_windows(5, cov) == 5
    assert primary_metric_policy_mod.fallback_paired_windows(0, {}) == 0


def test_enforce_drift_ratio_identity_and_alignment():
    # Matching ratio should return computed ratio
    ratio = primary_metric_policy_mod.enforce_drift_ratio_identity(
        paired_windows=4,
        delta_mean=math.log(1.1),
        drift_ratio=1.1,
        window_plan_profile="ci",
    )
    assert pytest.approx(ratio, rel=1e-3) == 1.1

    # Mismatch in CI profile should raise
    with pytest.raises(ValueError):
        primary_metric_policy_mod.enforce_drift_ratio_identity(
            paired_windows=4,
            delta_mean=0.5,
            drift_ratio=1.1,
            window_plan_profile="ci",
        )

    # Ratio CI alignment: paired baseline enforces exp(logloss_delta_ci)
    with pytest.raises(ValueError):
        primary_metric_policy_mod.enforce_ratio_ci_alignment(
            "paired_baseline", (1.0, 1.1), (-0.2, -0.1)
        )

    # Matching ratios should pass quietly
    primary_metric_policy_mod.enforce_ratio_ci_alignment(
        "paired_baseline",
        (math.exp(-0.2), math.exp(-0.1)),
        (-0.2, -0.1),
    )
