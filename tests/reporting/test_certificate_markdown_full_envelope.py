from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_base_report() -> dict:
    # Minimal viable RunReport for make_certificate
    return {
        "meta": {"model_id": "gpt2", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
            "latency_ms_per_tok": 1.2,
            "memory_mb_peak": 100.0,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_render_certificate_markdown_full_envelope() -> None:
    rep = _mk_base_report()
    base = _mk_base_report()
    cert = make_certificate(rep, base)

    # Enrich certificate with many optional sections to stimulate rendering branches
    cert.setdefault("meta", {}).update(
        {
            "invarlock_version": "0.2.0",
            "env_flags": {"FOO": "1", "BAR": "0"},
            "cuda_flags": {
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
            },
        }
    )
    cert.setdefault("policy_provenance", {}).update(
        {
            "tier": "balanced",
            "overrides": ["metrics.pm_ratio.hysteresis_ratio=0.002"],
            "policy_digest": cert.get("policy_digest", {}).get("thresholds_hash"),
            "resolved_at": cert.get("artifacts", {}).get("generated_at"),
        }
    )
    cert.setdefault("resolved_policy", {})["metrics"] = {
        "pm_ratio": {"min_tokens": 50000}
    }

    # Dataset extras
    ds = cert.get("dataset", {})
    ds.setdefault("hash", {})
    ds["hash"].update(
        {
            "dataset": "deadbeef",
            "preview_tokens": 10,
            "final_tokens": 10,
            "total_tokens": 20,
        }
    )
    ds["tokenizer"] = {
        "name": "tok",
        "hash": "tokhash",
        "vocab_size": 50257,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": None,
        "add_prefix_space": False,
    }

    # Secondary metrics and system overhead
    cert["secondary_metrics"] = [
        {
            "kind": "latency_ms_p50",
            "preview": 1.2,
            "final": 1.1,
            "ratio_vs_baseline": 0.92,
            "display_ci": [0.9, 1.1],
        }
    ]
    cert["system_overhead"] = {
        "latency_ms_p50": {
            "baseline": 10.0,
            "edited": 12.0,
            "delta": 2.0,
            "ratio": 1.2,
        },
        "latency_ms_p95": {
            "baseline": 20.0,
            "edited": 20.0,
            "delta": 0.0,
            "ratio": 1.0,
        },
        "throughput_sps": {"baseline": 0.0, "edited": 0.0},  # triggers N/A formatting
    }

    # Classification subgroups
    cert["classification"] = {
        "subgroups": {
            "A": {
                "n_preview": 5,
                "n_final": 6,
                "preview": 0.8,
                "final": 0.83,
                "delta_pp": +3.0,
            }
        }
    }

    # Structural changes (non-quant path)
    cert["structure"] = {
        "params_changed": 10,
        "layers_modified": 2,
        "bitwidths": [8, 8],
        "ranks": [64],
        "compression_diagnostics": {
            "execution_status": "successful",
            "target_analysis": {
                "modules_found": 10,
                "modules_eligible": 4,
                "modules_modified": 2,
                "scope": "ffn",
            },
            "parameter_analysis": {"bits": {"value": 8, "effectiveness": "high"}},
            "algorithm_details": {"name": "RTN"},
            "warnings": ["none"],
        },
    }

    # MoE observability (non-gating)
    cert["moe"] = {
        "top_k": 2,
        "capacity_factor": 1.2,
        "expert_drop_rate": 0.0,
        "utilization_count": 10,
        "utilization_mean": 0.5,
    }

    # Render
    md = render_certificate_markdown(cert)
    # Sanity check headings
    assert "InvarLock Evaluation Certificate" in md
    assert "Executive Summary" in md
    assert "Primary Metric" in md
    assert "System Overhead" in md
    assert "Dataset and Provenance" in md
    assert "Policy Configuration" in md
