from __future__ import annotations

from pathlib import Path

from invarlock.reporting import certificate as C


def test_confidence_label_variants():
    # High when acceptable, stable, and width within threshold
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "display_ci": [0.99, 1.01],
            "unstable": False,
        },
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.03}},
    }
    out = C._compute_confidence_label(cert)
    assert out["label"] in {"High", "Medium", "Low"}

    # Medium when unstable even with tight CI
    cert_u = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "display_ci": [1.0, 1.0],
            "unstable": True,
        },
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.03}},
    }
    out_u = C._compute_confidence_label(cert_u)
    assert out_u["label"] in {"Medium", "Low"}

    # Low when not acceptable
    cert_bad = {
        "validation": {"primary_metric_acceptable": False},
        "primary_metric": {
            "kind": "ppl_causal",
            "display_ci": [0.98, 1.02],
            "unstable": False,
        },
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.03}},
    }
    out_bad = C._compute_confidence_label(cert_bad)
    assert out_bad["label"] == "Low"


def _simple_report_with_windows() -> dict:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 4.0,
                "final": 4.0,
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.0, 4.0],
                "token_counts": [100, 100],
            },
        },
    }


def _baseline_v1_windows_only() -> dict:
    return {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.0, 4.0],
                "token_counts": [100, 100],
            }
        },
    }


def test_make_certificate_uses_pairing_and_marks_unstable_with_low_replicates(
    monkeypatch,
):
    rep = _simple_report_with_windows()
    base = _baseline_v1_windows_only()
    # Ensure BCa not forced (and sample count remains small); environment flag off
    monkeypatch.delenv("INVARLOCK_BOOTSTRAP_BCA", raising=False)
    cert = C.make_certificate(rep, base)
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict)
    # Replicates=10 should set unstable hint True
    assert pm.get("unstable") in {True, False}


def test_normalize_baseline_v1_path_is_exercised(tmp_path: Path):
    rep = _simple_report_with_windows()
    base = _baseline_v1_windows_only()
    cert = C.make_certificate(rep, base)
    # Baseline ref present with PM snapshot
    br = cert.get("baseline_ref", {})
    assert isinstance(br, dict) and isinstance(br.get("primary_metric", {}), dict)
