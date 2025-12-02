from __future__ import annotations

from copy import deepcopy

import pytest

from invarlock.reporting import certificate as cert


def _optional_sections_report() -> tuple[dict, dict]:
    window_ids = [11, 12, 13]
    token_counts = [20, 18, 22]
    base_report = {
        "run_id": "optional-run",
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "device": "cpu",
            "seed": 5,
            "auto": {"tier": "balanced", "probes_used": ["spectral"]},
        },
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 16,
            "stride": 8,
            "preview_n": 3,
            "final_n": 3,
            "windows": {"preview": 3, "final": 3},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": window_ids,
                "logloss": [0.1, 0.12, 0.14],
                "token_counts": token_counts,
            },
            "final": {
                "window_ids": window_ids,
                "logloss": [0.11, 0.13, 0.15],
                "token_counts": token_counts,
            },
        },
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "plan-digest",
            "deltas": {"params_changed": 8, "layers_modified": 2},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.4,
                "ratio_vs_baseline": 1.04,
                "display_ci": [10.2, 10.6],
            },
            "paired_delta_summary": {"mean": 0.0, "degenerate": False},
            "bootstrap": {"replicates": 300, "coverage": {"preview": {"used": 3}}},
            "window_plan": {"profile": "dev"},
            "window_overlap_fraction": 0.5,
            "window_match_fraction": 0.9,
            "secondary_metrics": [
                {
                    "kind": "accuracy_aux",
                    "final": 0.91,
                    "ratio_vs_baseline": 1.02,
                    "unit": "%",
                    "display_ci": [0.9, 0.93],
                },
                {"final": 1.0, "unit": "%", "display_ci": [0.95, 1.05]},
            ],
            "classification": {
                "subgroups": {
                    "preview": {
                        "group_counts": {"A": 10, "B": 20},
                        "correct_counts": {"A": 9, "B": 16},
                    },
                    "final": {
                        "group_counts": {"A": 12, "B": 18, "C": 6},
                        "correct_counts": {"A": 10, "B": 17, "C": 5},
                    },
                }
            },
            "latency_ms_p50": 18.0,
            "latency_ms_p95": 24.0,
            "throughput_sps": 420.0,
        },
        "artifacts": {"events_path": "", "logs_path": "", "generated_at": ""},
    }
    baseline = deepcopy(base_report)
    baseline["run_id"] = "baseline-run"
    baseline["metrics"]["primary_metric"]["final"] = 10.0
    baseline["metrics"]["primary_metric"]["ratio_vs_baseline"] = 1.0
    baseline["metrics"]["secondary_metrics"][0]["final"] = 0.89
    baseline["metrics"]["secondary_metrics"][0]["ratio_vs_baseline"] = 1.0
    baseline["metrics"]["secondary_metrics"].pop()  # Drop invalid entry to keep diff
    baseline["metrics"].pop("classification", None)
    baseline["metrics"]["latency_ms_p95"] = None
    baseline["metrics"]["throughput_sps"] = 450.0
    baseline["metrics"]["latency_ms_p50"] = 16.0
    return base_report, baseline


@pytest.mark.usefixtures("_clear_env_vars")
def test_make_certificate_populates_optional_sections(monkeypatch):
    report, baseline = _optional_sections_report()

    # ensure normalization helpers do not short-circuit rich payload
    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)

    certificate = cert.make_certificate(report, baseline)

    sec = certificate.get("secondary_metrics")
    assert isinstance(sec, list) and len(sec) == 1
    assert sec[0]["kind"] == "accuracy_aux"
    cls = certificate.get("classification", {}).get("subgroups", {})
    assert "A" in cls and "C" in cls
    system = certificate.get("system_overhead", {})
    assert "latency_ms_p50" in system and "delta" in system["latency_ms_p50"]
    # Latency p95 lacks baseline; ensure fallback only reports edited value
    assert "baseline" not in system["latency_ms_p95"]


def test_make_certificate_policy_digest_marks_changed(monkeypatch):
    report, baseline = _optional_sections_report()
    report = deepcopy(report)
    report["guards"] = [
        {"name": "spectral", "policy": {"max_caps": 5}, "metrics": {"caps_applied": 1}}
    ]
    baseline = deepcopy(baseline)
    baseline["meta"]["auto"]["tier"] = "conservative"

    monkeypatch.setattr(
        cert, "_normalize_and_validate_report", lambda value: value, raising=False
    )
    monkeypatch.setattr(cert, "_normalize_baseline", lambda value: value, raising=False)

    certificate = cert.make_certificate(report, baseline)
    digest = certificate["policy_digest"]
    assert digest["tier_policy_name"] == "balanced"
    assert digest["changed"] is True


@pytest.fixture
def _clear_env_vars(monkeypatch):
    # guard against INVARLOCK_TELEMETRY residue altering telemetry block
    monkeypatch.delenv("INVARLOCK_TELEMETRY", raising=False)
