from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_report_payload_policy import (
    build_artifacts_payload,
    build_edit_payload,
    build_flags_payload,
    build_guard_entries,
    build_metrics_payload,
    build_run_report_context,
    build_run_report_data,
    build_run_report_meta,
    build_snapshot_provenance,
    merge_core_timing_metrics,
)


def test_build_run_report_context_splits_known_sections() -> None:
    payload = build_run_report_context(
        profile_normalized="ci",
        auto_config={"enabled": True},
        run_context={
            "run": {"seed": 43},
            "eval": {"loss": {"resolved_type": "causal"}},
            "assurance": {"policy": "strict"},
            "ignored": "x",
        },
    )

    assert payload == {
        "profile": "ci",
        "auto": {"enabled": True},
        "assurance": {"policy": "strict"},
        "run": {"seed": 43},
        "eval": {"loss": {"resolved_type": "causal"}},
    }


def test_build_run_report_meta_collects_profile_and_optional_fields() -> None:
    model_profile = SimpleNamespace(
        family="gpt",
        default_loss="causal",
        module_selectors={"decoder": ["x"]},
        invariants=("weights",),
        cert_lints=({"code": "L001"},),
    )

    payload = build_run_report_meta(
        model_id="gpt2",
        adapter="hf_causal",
        resolved_device="cpu",
        commit_value="abc123",
        seed_bundle={"python": 43, "numpy": 43},
        auto_config={"tier": "balanced"},
        guard_overhead_threshold=0.01,
        model_profile=model_profile,
        timestamp="2026-03-27T12:00:00",
        invarlock_version="0.5.0",
        env_flags={"mps_available": False},
        determinism_meta={"warn_only": False},
        pm_acceptance_range=(0.9, 1.1),
        pm_drift_band=(0.95, 1.05),
    )

    assert payload["model_id"] == "gpt2"
    assert payload["adapter"] == "hf_causal"
    assert payload["guard_overhead_threshold"] == 0.01
    assert payload["model_profile"]["family"] == "gpt"
    assert payload["model_profile"]["cert_lints"] == [{"code": "L001"}]
    assert payload["pm_acceptance_range"] == (0.9, 1.1)
    assert payload["pm_drift_band"] == (0.95, 1.05)


def test_build_run_report_data_merges_dataset_meta_and_tokenizer_fallback() -> None:
    payload, tokenizer_hash = build_run_report_data(
        canonical_dataset_id="wikitext2",
        resolved_split="validation",
        seq_len=512,
        stride=256,
        preview_count=64,
        final_count=64,
        dataset_meta_context={"tokenizer_hash": "tok123", "dataset_hash": "ds456"},
        tokenizer_hash=None,
    )

    assert payload["dataset"] == "wikitext2"
    assert payload["dataset_hash"] == "ds456"
    assert tokenizer_hash == "tok123"


def test_build_edit_payload_applies_core_deltas_and_label_override() -> None:
    edit_payload, context_edit = build_edit_payload(
        core_edit={
            "plan_digest": "plan123",
            "deltas": {"params_changed": 4, "layers_modified": 2, "sparsity": 0.5},
            "mask_digest": {"preview": "abc"},
        },
        edit_name="quant_rtn",
        edit_label="demo",
    )

    assert edit_payload["name"] == "demo"
    assert edit_payload["algorithm"] == "demo"
    assert edit_payload["plan_digest"] == "plan123"
    assert edit_payload["deltas"]["layers_modified"] == 2
    assert edit_payload["mask_digest"] == {"preview": "abc"}
    assert context_edit == {"name": "demo", "params_changed": 4, "layers_modified": 2}


def test_merge_core_timing_metrics_coerces_numeric_and_preserves_bad_values() -> None:
    merged = merge_core_timing_metrics(
        {"prepare": 0.1},
        {"timings": {"prepare": "1.5", "edit": object(), "eval": 2}},
    )

    assert merged["prepare"] == 1.5
    assert merged["eval"] == 2.0
    assert "edit" in merged


def test_build_metrics_payload_merges_optional_and_dataset_fallbacks() -> None:
    payload = build_metrics_payload(
        core_metrics={
            "latency_ms_per_tok": 1.2,
            "memory_mb_peak": 256.0,
            "primary_metric_tail": {"status": "ok"},
            "timings": {"prepare": 1.0},
        },
        window_plan_context={
            "requested_preview": 64,
            "requested_final": 64,
            "actual_preview": 60,
            "actual_final": 61,
            "coverage_ok": True,
            "capacity": {"max_windows": 200},
        },
        dataset_meta_context={"masked_tokens_total": 33, "loss_type": "mlm"},
        resolved_loss_type=None,
    )

    assert payload["latency_ms_per_tok"] == 1.2
    assert payload["window_capacity"] == {"max_windows": 200}
    assert payload["stats"]["requested_preview"] == 64
    assert payload["masked_tokens_total"] == 33
    assert payload["loss_type"] == "mlm"


def test_build_guard_entries_flags_artifacts_and_snapshot_provenance() -> None:
    guards = {
        "spectral": {
            "passed": False,
            "action": "rollback",
            "policy": {"threshold": 1.0},
            "metrics": {"delta": 2.0},
            "final_z_scores": [1.2],
        },
        "skip-me": "not-a-dict",
    }

    entries = build_guard_entries(guards)
    assert entries == [
        {
            "name": "spectral",
            "passed": False,
            "action": "rollback",
            "policy": {"threshold": 1.0},
            "metrics": {"delta": 2.0},
            "actions": [],
            "violations": [],
            "warnings": [],
            "errors": [],
            "details": {},
            "final_z_scores": [1.2],
        }
    ]
    assert build_flags_payload(guards) == {
        "guard_recovered": True,
        "rollback_reason": None,
    }
    assert build_artifacts_payload(
        event_path="/tmp/events.jsonl", mask_artifact_path=None
    ) == {
        "events_path": "/tmp/events.jsonl",
        "logs_path": "",
        "checkpoint_path": None,
    }
    assert build_snapshot_provenance({"restore_failed": 1}) == {
        "restore_failed": True,
        "reload_path_used": False,
    }
