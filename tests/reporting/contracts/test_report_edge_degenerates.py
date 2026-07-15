import pytest

import invarlock.reporting.policy_utils as policy_utils
import invarlock.reporting.report_normalization as report_normalization
from invarlock.reporting.guards_invariants import _extract_invariants
from invarlock.reporting.policy_utils import _build_resolved_policies
from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_make import make_report
from invarlock.reporting.utils import _infer_scope_from_modules, _pair_logloss_windows
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_infer_scope_from_modules_no_family_match():
    # None of the family tokens present -> returns "all"
    assert _infer_scope_from_modules(["foo.bar", "baz.qux"]) == "all"


def test_pair_logloss_windows_non_numeric_filtered():
    # Non-numeric entries should be ignored; insufficient pairing -> None
    run = {"window_ids": [1, "x", 3], "logloss": [0.1, "bad", 0.3]}
    base = {"window_ids": [1, 2, 3], "logloss": [0.11, 0.2, "bad"]}
    assert _pair_logloss_windows(run, base) is None


def test_build_resolved_policies_with_empty_tier_defaults(monkeypatch):
    # Simulate missing tier presets; function should fall back to internal defaults
    monkeypatch.setattr(
        policy_utils, "get_tier_policies", lambda *_a, **_k: {}, raising=False
    )
    resolved = _build_resolved_policies(
        "balanced",
        spectral={"multiple_testing": {"method": "bh", "alpha": 0.07, "m": 3}},
        rmt={"epsilon_by_family": {"ffn": 0.1}},
        variance={"predictive_gate": {"sided": "one_sided"}},
    )
    # Defaults applied sanely without crashing
    assert resolved["spectral"]["sigma_quantile"] == pytest.approx(0.95)
    assert resolved["spectral"]["deadband"] == pytest.approx(0.1)
    assert isinstance(resolved["spectral"].get("max_caps"), int)
    # RMT map normalized
    assert resolved["rmt"]["epsilon_by_family"]["ffn"] == pytest.approx(0.1)
    # Variance sided flag resolved to boolean
    assert resolved["variance"]["predictive_one_sided"] is True


def test_extract_invariants_ignores_non_dict_violations_and_non_dict_values():
    # invariants dict contains non-dict boolean -> treated as boolean
    report = {
        "metrics": {
            "invariants": {
                "bool_check": False,  # becomes a failure entry
                "dict_check": {
                    "passed": False,
                    "violations": ["not-a-dict", {"type": "warn"}],
                },
            }
        },
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 1,
                    "violations_found": 1,
                    "fatal_violations": 0,
                    "warning_violations": 1,
                },
                # Also include non-dict items here which must be ignored
                "violations": [
                    "bad",
                    {"check": "x", "type": "mismatch", "severity": "warning"},
                ],
            }
        ],
    }
    out = _extract_invariants(report)
    # We should have at least one failure recorded from the boolean and one from dict violation
    assert out["status"] in {"warn", "fail"}
    assert any(
        isinstance(v, dict) for v in out["failures"]
    )  # only dict violations kept


def test_render_report_markdown_guard_metric_impact_na(monkeypatch):
    # Minimal valid report/baseline to build a evaluation_report
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
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
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = {**report, "edit": {"name": "noop"}}
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(canonical_run_report(report), canonical_baseline(baseline))
    # Inject a guard_metric_impact section with None/NaN values; renderer should not crash
    cert["guard_metric_impact"] = {
        "display_value": None,
        "degradation": float("nan"),
        "display_limit": 1.0,
    }
    md = render_report_markdown(cert)
    assert isinstance(md, str) and "Guard Observability" in md
