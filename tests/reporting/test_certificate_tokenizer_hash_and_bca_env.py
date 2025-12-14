from __future__ import annotations

from types import SimpleNamespace

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def test_certificate_baseline_ref_includes_tokenizer_hash() -> None:
    report = create_empty_report()
    report["meta"]["model_id"] = "m"
    report["meta"]["adapter"] = "hf"
    report["meta"]["device"] = "cpu"
    report["meta"]["auto"] = {
        "tier": "balanced",
        "probes_used": 0,
        "target_pm_ratio": None,
    }  # type: ignore[assignment]
    report["data"]["dataset"] = "unit"
    report["data"]["split"] = "validation"
    report["data"]["seq_len"] = 8
    report["data"]["stride"] = 8
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    report["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.3, 2.31],
            "token_counts": [100, 100],
        }
    }

    baseline = {
        "run_id": "base",
        "model_id": "m",
        "meta": {"model_id": "m", "tokenizer_hash": "tokhash-abc"},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.3, 2.3],
                "token_counts": [100, 100],
            }
        },
        "data": {
            "seq_len": 8,
            "preview_n": 2,
            "final_n": 2,
            "dataset": "unit",
            "split": "validation",
            "stride": 8,
        },
        "edit": {
            "name": "none",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }

    cert = make_certificate(report, baseline)
    assert cert["baseline_ref"]["tokenizer_hash"] == "tokhash-abc"


def test_certificate_uses_bca_when_env_enabled_and_many_paired_windows(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_BOOTSTRAP_BCA", "1")

    report = create_empty_report()
    report["meta"]["model_id"] = "m"
    report["meta"]["adapter"] = "hf"
    report["meta"]["device"] = "cpu"
    report["meta"]["auto"] = {
        "tier": "balanced",
        "probes_used": 0,
        "target_pm_ratio": None,
    }  # type: ignore[assignment]
    report["data"]["dataset"] = "unit"
    report["data"]["split"] = "validation"
    report["data"]["seq_len"] = 8
    report["data"]["stride"] = 8
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    report["metrics"]["bootstrap"] = {
        "method": "percentile",
        "replicates": 10,
        "alpha": 0.05,
        "seed": 0,
    }

    window_ids = list(range(200))
    report["evaluation_windows"] = {
        "final": {
            "window_ids": window_ids,
            "logloss": [2.3] * 200,
            "token_counts": [100] * 200,
        }
    }

    baseline = {
        "run_id": "base",
        "model_id": "m",
        "meta": {"seed": 0, "model_id": "m"},
        "evaluation_windows": {
            "final": {
                "window_ids": window_ids,
                "logloss": [2.3] * 200,
                "token_counts": [100] * 200,
            }
        },
        "data": {
            "seq_len": 8,
            "preview_n": 200,
            "final_n": 200,
            "dataset": "unit",
            "split": "validation",
            "stride": 8,
        },
        "edit": {
            "name": "none",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }

    seen: dict[str, object] = {}

    def _fake_ci(run_vals, base_vals, *, method, **_kwargs):  # noqa: ANN001
        seen["method"] = method
        return (0.0, 0.0)

    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci", _fake_ci
    )
    monkeypatch.setattr(
        "invarlock.reporting.certificate.logspace_to_ratio_ci",
        lambda _ci: (1.0, 1.0),
    )
    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_primary_metric_from_report",
        lambda *_a, **_k: {"kind": "ppl_causal", "final": 10.0},
    )
    monkeypatch.setattr(
        "invarlock.reporting.certificate.get_metric",
        lambda *_a, **_k: SimpleNamespace(direction="lower"),
    )

    make_certificate(report, baseline)
    assert seen.get("method") == "bca"
