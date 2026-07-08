from __future__ import annotations

import pytest

from invarlock.reporting import report_normalization as normalization_mod
from invarlock.reporting import report_primary_metric_policy as pm_policy
from invarlock.reporting import report_schema as allowlist_mod
from invarlock.reporting import report_schema as schema_mod


def test_normalize_baseline_handles_schema_v1():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"commit_sha": "abcdef1234567890", "model_id": "demo"},
        "metrics": {"ppl_final": 42.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "ok"},
    }
    normalized = normalization_mod.normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["ppl_final"] == 42.0
    assert normalized["spectral"] == {"caps": 1}


def test_normalize_baseline_infers_primary_metric():
    baseline = {
        "meta": {"model_id": "demo", "adapter": "hf"},
        "edit": {"name": "baseline", "deltas": {"params_changed": 0}, "plan": {}},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.0,
                "final": 10.0,
            },
            "bootstrap": {"coverage": {"used": 1}},
            "spectral": {},
            "rmt": {},
            "invariants": {},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.2]}},
    }
    normalized = normalization_mod.normalize_baseline(baseline)
    assert normalized["ppl_final"] == 10.0
    assert normalized["ppl_preview"] == 9.0
    assert normalized["evaluation_windows"]["final"]["logloss"] == [0.2]


def test_normalize_baseline_raises_on_invalid():
    baseline = {
        "meta": {"model_id": "demo"},
        "edit": {"name": "quantize", "plan": {}, "deltas": {"params_changed": 5}},
        "metrics": {
            "ppl_final": 0.0,
            "ppl_preview": 0.0,
            "spectral": {},
            "rmt": {},
            "invariants": {},
        },
    }
    with pytest.raises(ValueError, match="Invalid baseline"):
        normalization_mod.normalize_baseline(baseline)


def test_validate_evaluation_report_uses_jsonschema(monkeypatch):
    class DummySchema:
        def __init__(self):
            self.calls = 0

        def validate(self, instance, schema):
            self.calls += 1

    dummy = DummySchema()
    monkeypatch.setattr(schema_mod, "jsonschema", dummy, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-1",
        "primary_metric": {"kind": "ppl_causal"},
        "validation": {"primary_metric_acceptable": True},
    }
    assert schema_mod.validate_report(evaluation_report) is True
    assert dummy.calls == 1


def _minimal_schema_valid_report() -> dict:
    return {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-tail",
        "artifacts": {},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {"preview": 1, "final": 1, "stats": {}},
        },
        "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }


def test_report_schema_validates_primary_metric_tail_shape() -> None:
    if schema_mod.jsonschema is None:
        pytest.skip("jsonschema not installed")
    evaluation_report = _minimal_schema_valid_report()
    evaluation_report["primary_metric_tail"] = {
        "evaluated": True,
        "passed": True,
        "warned": False,
        "mode": "warn",
        "policy": {"quantile": 0.95},
        "stats": {"q95": 0.01},
    }

    assert schema_mod.validate_report(evaluation_report) is True


def test_report_schema_rejects_malformed_primary_metric_tail() -> None:
    if schema_mod.jsonschema is None:
        pytest.skip("jsonschema not installed")
    evaluation_report = _minimal_schema_valid_report()
    evaluation_report["primary_metric_tail"] = {
        "evaluated": "yes",
        "policy": "not-a-policy",
        "stats": [],
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_payload_when_jsonschema_fails(monkeypatch):
    class FailingSchema:
        def validate(self, instance, schema):
            raise ValueError("boom")

    monkeypatch.setattr(schema_mod, "jsonschema", FailingSchema(), raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-2",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }
    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_invalid_flags(monkeypatch):
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-3",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": "yes"},
    }
    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_blank_run_id_without_jsonschema(
    monkeypatch,
):
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "   ",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_accepts_short_nonblank_run_id_without_jsonschema(
    monkeypatch,
) -> None:
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r1 ",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_blank_metric_kind_without_jsonschema(
    monkeypatch,
) -> None:
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-4",
        "primary_metric": {"kind": "   "},
        "validation": {"primary_metric_acceptable": True},
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_boolean_metric_final_without_jsonschema(
    monkeypatch,
) -> None:
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-5",
        "primary_metric": {"final": True},
        "validation": {"primary_metric_acceptable": True},
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_rejects_non_mapping_validation_without_jsonschema(
    monkeypatch,
) -> None:
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    evaluation_report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-6",
        "primary_metric": {"final": 1.0},
        "validation": [],
    }

    assert schema_mod.validate_report(evaluation_report) is False


def test_load_validation_allowlist_prefers_contracts_file(monkeypatch):
    keys = ["primary_metric_acceptable", "guard_overhead_acceptable", "custom_flag"]
    monkeypatch.setattr(allowlist_mod, "load_json_contract", lambda _filename: keys)
    loaded = allowlist_mod.load_validation_allowlist()
    assert loaded == {str(k) for k in keys}


def test_validate_evaluation_report_handles_mapping_errors() -> None:
    class ExplodingMapping(dict):
        def get(self, *_args, **_kwargs):
            raise ValueError("boom")

    evaluation_report = ExplodingMapping()
    assert schema_mod.validate_report(evaluation_report) is False


def test_validate_evaluation_report_normalizes_missing_validation_from_mapping(
    monkeypatch,
) -> None:
    class NonDictValidationMapping(dict):
        def get(self, key, default=None):
            if key == "validation":
                return []
            return super().get(key, default)

        def __contains__(self, key):
            if key == "validation":
                return False
            return super().__contains__(key)

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    evaluation_report = NonDictValidationMapping(
        {
            "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
            "run_id": "run-7",
            "primary_metric": {"kind": "ppl_causal", "final": 1.0},
        }
    )

    assert schema_mod.validate_report(evaluation_report) is True


def test_propagate_pairing_stats_adds_missing_fields():
    evaluation_report = {"dataset": {"windows": {}}}
    ppl_analysis = {
        "stats": {
            "pairing": "paired_baseline",
            "paired_windows": 4,
            "coverage": {"preview": {"used": 3}},
            "window_match_fraction": 0.9,
            "window_overlap_fraction": 0.4,
            "window_pairing_reason": "id_match",
            "requested_preview": 2,
            "requested_final": 2,
            "actual_preview": 2,
            "actual_final": 2,
            "coverage_ok": True,
        }
    }
    pm_policy.propagate_pairing_stats(evaluation_report, ppl_analysis)
    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert stats["pairing"] == "paired_baseline"
    assert stats["paired_windows"] == 4
    assert stats["coverage"]["preview"]["used"] == 3
    assert stats["window_match_fraction"] == pytest.approx(0.9)
    assert stats["window_pairing_reason"] == "id_match"


def test_propagate_pairing_stats_ignores_missing_dataset():
    report: dict[str, object] = {}
    pm_policy.propagate_pairing_stats(report, {"stats": {}})
    assert report == {"dataset": {"windows": {"stats": {}}}}


def test_normalize_baseline_handles_v1_schema_structure():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"commit_sha": "abcdef1234567890", "model_id": "gpt2"},
        "metrics": {"ppl_final": 25.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "ok"},
    }
    normalized = normalization_mod.normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["model_id"] == "gpt2"
    assert normalized["ppl_final"] == 25.0
    assert normalized["spectral"] == {"caps": 1}


def test_normalize_baseline_raises_for_invalid_ppl():
    baseline = {
        "meta": {"model_id": "demo"},
        "edit": {"name": "baseline", "plan": {}, "deltas": {"params_changed": 0}},
        "metrics": {"ppl_final": 0.0, "spectral": {}, "rmt": {}, "invariants": {}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with pytest.raises(ValueError, match="Invalid baseline"):
        normalization_mod.normalize_baseline(baseline)


def test_normalize_baseline_extracts_runreport_payload():
    baseline = {
        "meta": {"model_id": "demo", "auto": {"tier": "balanced"}},
        "edit": {
            "name": "baseline",
            "plan": {"target_sparsity": 0.0},
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 12.0, "preview": 11.0},
            "spectral": {"caps_applied": 0},
            "rmt": {"stable": True},
            "invariants": {"status": "ok"},
            "bootstrap": {},
            "window_overlap_fraction": 0.4,
            "window_match_fraction": 1.0,
        },
        "evaluation_windows": {
            "final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]},
        },
    }
    normalized = normalization_mod.normalize_baseline(baseline)
    assert normalized["run_id"] is not None
    assert normalized["spectral"]["caps_applied"] == 0
    assert normalized["evaluation_windows"]["final"]["logloss"] == [0.1, 0.2]
