from __future__ import annotations

import importlib
from pathlib import Path

from tests.cli._support_transformers import install_transformers_tokenizer_stub


def _import_verify_module():
    install_transformers_tokenizer_stub()
    return importlib.import_module("invarlock.reporting.verify_contract")


def test_recompute_validation_flags_and_policy_gate_paths(monkeypatch) -> None:
    verify_mod = _import_verify_module()
    captured: dict[str, object] = {}

    def _fake_compute_validation_flags(**kwargs):
        captured.update(kwargs)
        return {"primary_metric_acceptable": False}

    monkeypatch.setattr(
        verify_mod, "compute_validation_flags", _fake_compute_validation_flags
    )
    monkeypatch.setattr(
        verify_mod,
        "resolve_tiny_relax_from_report",
        lambda report: bool(report.get("tiny")),
    )

    report_bad = {
        "primary_metric": "bad",
        "telemetry": "bad",
        "auto": "bad",
        "resolved_policy": {"metrics": "bad"},
        "spectral": "bad",
        "rmt": "bad",
        "invariants": "bad",
        "guard_overhead": "bad",
        "primary_metric_tail": "bad",
    }
    flags = verify_mod._recompute_validation_flags(report_bad)
    assert flags["primary_metric_acceptable"] is False
    assert captured["tier"] == "balanced"
    assert captured["_ppl_metrics"] == {}
    assert captured["ppl"] == {}
    assert captured["get_tier_policies_fn"] is None

    captured.clear()
    report_good = {
        "primary_metric": {"ratio_vs_baseline": "1.05", "preview": 10.0, "final": 10.2},
        "telemetry": {"preview_total_tokens": "10", "final_total_tokens": 20},
        "dataset": {
            "windows": {
                "stats": {
                    "coverage": {
                        "preview": {"used": 10, "required": 8, "ok": True},
                        "final": {"used": 20, "required": 8, "ok": True},
                    }
                }
            }
        },
        "auto": {"tier": "Conservative", "target_pm_ratio": "1.1"},
        "context": {
            "primary_metric": {
                "acceptance_range": {"min": 0.95, "max": 1.15},
                "drift_band": {"min": 0.9, "max": 1.3},
            }
        },
        "resolved_policy": {"metrics": {"pm_ratio": {"min_tokens": 1}}},
        "spectral": {},
        "rmt": {},
        "invariants": {},
        "guard_overhead": {},
        "primary_metric_tail": {},
        "tiny": True,
    }
    verify_mod._recompute_validation_flags(report_good)
    assert captured["tier"] == "conservative"
    assert captured["target_ratio"] == 1.1
    assert captured["tiny_relax"] is True
    assert captured["_ppl_metrics"] == {
        "preview_total_tokens": 10,
        "final_total_tokens": 20,
        "bootstrap": {
            "coverage": {
                "preview": {"used": 10, "required": 8, "ok": True},
                "final": {"used": 20, "required": 8, "ok": True},
            }
        },
    }
    assert captured["pm_acceptance_range"] == {"min": 0.95, "max": 1.15}
    assert captured["pm_drift_band"] == {"min": 0.9, "max": 1.3}
    assert callable(captured["get_tier_policies_fn"])
    tier_cfg = captured["get_tier_policies_fn"]()
    assert tier_cfg["conservative"]["metrics"]["pm_ratio"]["min_tokens"] == 1

    assert verify_mod._validate_primary_metric_policy({}, profile="dev") == []
    monkeypatch.setattr(
        verify_mod,
        "_recompute_validation_flags",
        lambda _report: {"primary_metric_acceptable": True},
    )
    assert verify_mod._validate_primary_metric_policy({}, profile="ci") == []
    monkeypatch.setattr(
        verify_mod,
        "_recompute_validation_flags",
        lambda _report: {"primary_metric_acceptable": False},
    )
    errs = verify_mod._validate_primary_metric_policy(
        {"telemetry": {"preview_total_tokens": 10}, "auto": {}}, profile="release"
    )
    assert errs == ["Primary metric policy gate failed (tier=balanced)."]


def test_primary_metric_policy_uses_serialized_acceptance_range() -> None:
    verify_mod = _import_verify_module()

    relaxed = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 11.2,
            "ratio_vs_baseline": 1.12,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
        "context": {"primary_metric": {"acceptance_range": {"min": 0.95, "max": 1.15}}},
    }
    assert verify_mod._validate_primary_metric_policy(relaxed, profile="release") == []

    strict = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.8,
            "ratio_vs_baseline": 1.08,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
        "context": {"primary_metric": {"acceptance_range": {"min": 0.95, "max": 1.05}}},
    }
    errs = verify_mod._validate_primary_metric_policy(strict, profile="release")
    assert errs == ["Primary metric policy gate failed (tier=balanced)."]


def test_warn_adapter_family_mismatch(tmp_path: Path) -> None:
    verify_mod = _import_verify_module()
    # Create a baseline report with adapter provenance
    baseline = tmp_path / "baseline_report.json"
    baseline.write_text(
        """
        {
          "meta": {
            "plugins": {
              "adapter": {
                "provenance": {"family": "hf", "library": "transformers", "version": "0.0"}
              }
            }
          }
        }
        """.strip()
    )

    cert = {
        "plugins": {
            "adapter": {
                "provenance": {"family": "ggml", "library": "ggml", "version": "0.0"}
            }
        },
        "provenance": {"baseline": {"report_path": str(baseline)}},
    }
    # Should not raise; may emit a soft warning to console
    verify_mod._warn_adapter_family_mismatch(
        baseline,
        cert,
        trusted_baseline_path=baseline,
    )
