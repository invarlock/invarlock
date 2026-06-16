from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import invarlock.reporting.report_files as report_files
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.run_report_contract import (
    assemble_run_report,
    build_run_report_context,
    persist_run_report_outputs,
)


def test_assemble_run_report_builds_report_and_metrics(tmp_path: Path) -> None:
    core_report = SimpleNamespace(
        context={
            "dataset_meta": {"source": "wikitext2"},
            "window_plan": {"kind": "fixed"},
        },
        edit={},
        guards={"invariants": {"passed": True}},
        metrics={"latency_s": 1.5},
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(id="gpt2", adapter="hf_causal"),
        dataset=SimpleNamespace(provider="wikitext2", seq_len=128, stride=64),
        meta=SimpleNamespace(commit="abc123"),
    )

    result = assemble_run_report(
        core_report=core_report,
        cfg=cfg,
        run_context={"profile": "dev"},
        profile_normalized="dev",
        auto_config={"enabled": True},
        resolved_device="cpu",
        seed_bundle={"python": 43},
        guard_overhead_threshold=0.01,
        model_profile=SimpleNamespace(name="causal"),
        determinism_meta={"deterministic": True},
        pm_acceptance_range=(0.9, 1.1),
        pm_drift_band=(0.95, 1.05),
        tokenizer_hash="tokhash",
        resolved_split="validation",
        preview_count=8,
        final_count=8,
        snapshot_provenance={"reload_path_used": True},
        edit_op=SimpleNamespace(name="noop"),
        edit_label=None,
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        resolved_loss_type="causal",
        timings={"load_model": 1.0},
        guard_overhead_payload={"passed": True, "evaluated": False},
        baseline=None,
        preview_records=[],
        final_records=[],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        profile="dev",
        used_fallback_split=False,
        baseline_report_data=None,
        effective_preview=8,
        effective_final=8,
        metric_kind="ppl_causal",
        window_plan={"kind": "fixed"},
        debug_metric_diffs_enabled=False,
        create_empty_report_fn=lambda: {
            "meta": {},
            "data": {},
            "edit": {},
            "artifacts": {},
            "metrics": {},
            "guards": [],
            "flags": {},
            "provenance": {},
        },
        build_run_report_context_fn=lambda **kwargs: {
            "profile": kwargs["profile_normalized"]
        },
        build_run_report_meta_fn=lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "device": kwargs["resolved_device"],
            "seed": kwargs["seed_bundle"]["python"],
        },
        canonical_dataset_id_fn=lambda provider: provider,
        safe_int_fn=int,
        build_run_report_data_fn=lambda **kwargs: (
            {"preview_n": kwargs["preview_count"], "final_n": kwargs["final_count"]},
            kwargs["tokenizer_hash"],
        ),
        build_snapshot_provenance_fn=lambda payload: payload,
        build_edit_payload_fn=lambda **kwargs: (
            {"name": kwargs["edit_name"]},
            {"label": "noop"},
        ),
        persist_ref_masks_fn=lambda core_report, run_dir: run_dir / "ref_masks.json",
        build_artifacts_payload_fn=lambda **kwargs: {
            "event_path": str(kwargs["event_path"]),
            "mask_artifact_path": str(kwargs["mask_artifact_path"]),
        },
        merge_core_timing_metrics_fn=lambda timings, metrics: {
            **timings,
            "latency_s": metrics["latency_s"],
        },
        build_metrics_payload_fn=lambda **kwargs: {
            "primary_metric": {"kind": "ppl_causal", "final": 1.0}
        },
        prepare_guard_overhead_report_fn=lambda payload, **kwargs: payload,
        finalize_run_provenance_fn=lambda **kwargs: SimpleNamespace(
            missing_evaluation_windows_for_baseline=False
        ),
        build_guard_entries_fn=lambda guards: [{"name": "invariants"}],
        build_flags_payload_fn=lambda guards: {"all_passed": True},
        enrich_run_report_metrics_fn=lambda **kwargs: SimpleNamespace(
            pairing_violations=(), debug_diffs_line="", report=kwargs["report"]
        ),
        optional_torch_fn=lambda: None,
        environ={},
    )

    assert result.report["meta"]["model_id"] == "gpt2"
    assert result.report["data"]["preview_n"] == 8
    assert result.report["edit"]["name"] == "noop"
    assert result.report["artifacts"]["mask_artifact_path"].endswith("ref_masks.json")
    assert result.report["flags"]["all_passed"] is True
    assert result.timings["latency_s"] == 1.5


def test_persist_run_report_outputs_adds_telemetry_and_saved_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _save_telemetry(report, run_dir, filename):
        out = run_dir / filename
        out.write_text("{}", encoding="utf-8")
        return out

    def _save_report(report, out_dir, formats=None, filename_prefix="report"):
        out = out_dir / "report.json"
        out.write_text("{}", encoding="utf-8")
        return {"json": out}

    monkeypatch.setattr(report_files, "save_report", _save_report)

    report = create_empty_report()
    result = persist_run_report_outputs(
        report=report,
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        telemetry=True,
        save_telemetry_report_fn=_save_telemetry,
    )

    assert (tmp_path / "report.json").is_file()
    assert result.report_path_out == str(tmp_path / "report.json")
    assert result.telemetry_saved_path == str(tmp_path / "telemetry.json")
    assert report["artifacts"]["telemetry_path"].endswith("telemetry.json")


def test_persist_run_report_outputs_preserves_existing_backend_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _save_report(report, out_dir, formats=None, filename_prefix="report"):
        out = out_dir / "report.json"
        out.write_text("{}", encoding="utf-8")
        return {"json": out}

    monkeypatch.setattr(report_files, "save_report", _save_report)
    existing_inventory = {
        "schema": "invarlock/backend-inventory-v1",
        "adapter": "hf_bnb",
        "backend": "bitsandbytes",
        "backend_version": "0.49.0",
        "transformers_version": "4.57.0",
        "quantization_config": {"load_in_8bit": True},
        "quantized_module_count": 2,
        "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
        "device_map": "auto",
        "memory_footprint": {"reported_bytes": 10, "method": "test"},
        "load_smoke": True,
        "inference_smoke": True,
    }
    (tmp_path / "backend_inventory.json").write_text(
        json.dumps(existing_inventory),
        encoding="utf-8",
    )

    report = create_empty_report()
    report["meta"]["adapter"] = "hf_bnb"
    result = persist_run_report_outputs(
        report=report,
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        telemetry=False,
        save_telemetry_report_fn=lambda *args, **kwargs: None,
    )

    assert result.saved_files["backend_inventory"].endswith("backend_inventory.json")
    payload = json.loads((tmp_path / "backend_inventory.json").read_text("utf-8"))
    assert payload["quantized_module_count"] == 2


def test_persist_run_report_outputs_ignores_corrupt_backend_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _save_report(report, out_dir, formats=None, filename_prefix="report"):
        out = out_dir / "report.json"
        out.write_text("{}", encoding="utf-8")
        return {"json": out}

    monkeypatch.setattr(report_files, "save_report", _save_report)
    (tmp_path / "backend_inventory.json").write_text("[", encoding="utf-8")

    report = create_empty_report()
    result = persist_run_report_outputs(
        report=report,
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        telemetry=False,
        save_telemetry_report_fn=lambda *args, **kwargs: None,
    )

    assert "backend_inventory" not in result.saved_files


def test_persist_run_report_outputs_marks_context_inventory_inference_smoke(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _save_report(report, out_dir, formats=None, filename_prefix="report"):
        out = out_dir / "report.json"
        out.write_text("{}", encoding="utf-8")
        return {"json": out}

    monkeypatch.setattr(report_files, "save_report", _save_report)
    inventory = {
        "schema": "invarlock/backend-inventory-v1",
        "adapter": "hf_bnb",
        "backend": "bitsandbytes",
        "backend_version": "0.49.0",
        "transformers_version": "4.57.0",
        "quantization_config": {"load_in_8bit": True},
        "quantized_module_count": 2,
        "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
        "device_map": "auto",
        "memory_footprint": {"reported_bytes": 10, "method": "test"},
        "load_smoke": True,
        "inference_smoke": False,
    }

    report = create_empty_report()
    report["meta"]["adapter"] = "hf_bnb"
    result = persist_run_report_outputs(
        report=report,
        run_dir=tmp_path,
        run_config=SimpleNamespace(
            event_path=tmp_path / "events.jsonl",
            context={"_backend_inventory": inventory},
        ),
        telemetry=False,
        save_telemetry_report_fn=lambda *args, **kwargs: None,
    )

    assert result.saved_files["backend_inventory"].endswith("backend_inventory.json")
    payload = json.loads((tmp_path / "backend_inventory.json").read_text("utf-8"))
    assert payload["load_smoke"] is True
    assert payload["inference_smoke"] is True


def test_assemble_run_report_preserves_primary_metric_context() -> None:
    core_report = SimpleNamespace(
        context={"dataset_meta": {}, "window_plan": {}},
        edit={},
        guards={},
        metrics={},
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(id="gpt2", adapter="hf_causal"),
        dataset=SimpleNamespace(provider="wikitext2", seq_len=128, stride=64),
        meta=SimpleNamespace(commit="abc123"),
    )

    result = assemble_run_report(
        core_report=core_report,
        cfg=cfg,
        run_context={
            "primary_metric": {
                "drift_band": {"min": 0.9, "max": 1.2},
                "acceptance_range": {"min": 0.95, "max": 1.1},
            }
        },
        profile_normalized="ci",
        auto_config={"enabled": True},
        resolved_device="cpu",
        seed_bundle={"python": 43},
        guard_overhead_threshold=0.01,
        model_profile=SimpleNamespace(name="causal"),
        determinism_meta={},
        pm_acceptance_range=(0.95, 1.1),
        pm_drift_band=(0.9, 1.2),
        tokenizer_hash=None,
        resolved_split="validation",
        preview_count=8,
        final_count=8,
        snapshot_provenance={},
        edit_op=SimpleNamespace(name="noop"),
        edit_label=None,
        run_dir=Path.cwd(),
        run_config=SimpleNamespace(event_path=Path("events.jsonl")),
        resolved_loss_type="causal",
        timings={},
        guard_overhead_payload=None,
        baseline=None,
        preview_records=[],
        final_records=[],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        profile="ci",
        used_fallback_split=False,
        baseline_report_data=None,
        effective_preview=8,
        effective_final=8,
        metric_kind="ppl_causal",
        window_plan=None,
        debug_metric_diffs_enabled=False,
        create_empty_report_fn=create_empty_report,
        build_run_report_context_fn=build_run_report_context,
        build_run_report_meta_fn=lambda **kwargs: {},
        canonical_dataset_id_fn=lambda provider: provider,
        safe_int_fn=int,
        build_run_report_data_fn=lambda **kwargs: ({}, kwargs["tokenizer_hash"]),
        build_snapshot_provenance_fn=lambda payload: payload,
        build_edit_payload_fn=lambda **kwargs: ({}, None),
        persist_ref_masks_fn=lambda core_report, run_dir: None,
        build_artifacts_payload_fn=lambda **kwargs: {},
        merge_core_timing_metrics_fn=lambda timings, metrics: timings,
        build_metrics_payload_fn=lambda **kwargs: {},
        prepare_guard_overhead_report_fn=lambda payload, **kwargs: payload,
        finalize_run_provenance_fn=lambda **kwargs: SimpleNamespace(
            missing_evaluation_windows_for_baseline=False
        ),
        build_guard_entries_fn=lambda guards: [],
        build_flags_payload_fn=lambda guards: {},
        enrich_run_report_metrics_fn=lambda **kwargs: SimpleNamespace(
            pairing_violations=(), debug_diffs_line="", report=kwargs["report"]
        ),
        optional_torch_fn=lambda: None,
        environ={},
    )

    assert result.report["context"]["primary_metric"] == {
        "drift_band": {"min": 0.9, "max": 1.2},
        "acceptance_range": {"min": 0.95, "max": 1.1},
    }
