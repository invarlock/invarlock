import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.guards_analysis import _extract_invariants
from invarlock.reporting.policy_utils import _build_resolved_policies
from invarlock.reporting.render import render_certificate_markdown
from invarlock.reporting.utils import _infer_scope_from_modules, _pair_logloss_windows


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
        "invarlock.reporting.certificate.TIER_POLICIES", {}, raising=False
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


def test_render_certificate_markdown_guard_overhead_na(monkeypatch):
    # Minimal valid report/baseline to build a certificate
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
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
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    monkeypatch.setattr(
        "invarlock.reporting.certificate.validate_report", lambda _: True
    )
    cert = make_certificate(report, baseline)
    # Inject a guard_overhead section with None/NaN values; renderer should not crash
    cert["guard_overhead"] = {
        "overhead_percent": None,
        "overhead_ratio": float("nan"),
        "threshold_percent": 1.0,
    }
    md = render_certificate_markdown(cert)
    assert isinstance(md, str) and "Guard Observability" in md
