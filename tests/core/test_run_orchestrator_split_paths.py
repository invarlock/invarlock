from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from invarlock.core.run_orchestrator import (
    RunExecutionRequest,
    RunExecutionServices,
    execute_run_request,
)


class _Config:
    def __init__(self) -> None:
        self.model = SimpleNamespace(adapter="stub", id="stub-model")
        self.dataset = SimpleNamespace(
            provider="synthetic",
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=4,
            seed=42,
            split="validation",
        )
        self.edit = SimpleNamespace(name="noop")
        self.guards = {"order": []}
        self.eval = {"loss": {"type": "auto"}}
        self.output = SimpleNamespace(dir="runs")
        self.context = {}

    def model_dump(self) -> dict[str, object]:
        return {"edit": {"name": "noop"}}

    def section(self, _name: str) -> object:
        raise TypeError("section dispatch disabled for test")


def test_execute_run_request_emits_typed_diagnostics_and_nonfatal_warnings(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._build_run_context_payload_impl",
        lambda **_kwargs: {"dataset": {}, "eval": {"loss": {"resolved_type": "ce"}}},
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._build_run_execution_config_payloads_impl",
        lambda **_kwargs: SimpleNamespace(auto_config={}, edit_config={}),
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._resolve_pm_acceptance_range_impl",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._resolve_pm_drift_band_impl",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._resolve_guard_overhead_threshold_impl",
        lambda _cfg: 0.01,
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._should_measure_overhead_impl",
        lambda _profile, _cfg: (False, False, None),
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator._resolve_retry_validation_transition_impl",
        lambda *_args, **_kwargs: SimpleNamespace(
            action="passed",
            failed_gates=(),
            updated_edit_config={},
            head_adjustment=None,
            diagnostics=(),
            next_attempt=None,
            error_message=None,
        ),
    )
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: SimpleNamespace(
            get_adapter=lambda _name: SimpleNamespace(name="stub"),
            get_edit=lambda _name: SimpleNamespace(name="noop"),
            get_guard=lambda _name: (_ for _ in ()).throw(KeyError(_name)),
            get_plugin_metadata=lambda name, plugin_type: {
                "name": name,
                "module": f"{plugin_type}.{name}",
                "version": "test",
            },
        ),
    )
    monkeypatch.setattr("invarlock.core.runner.CoreRunner", lambda: object())

    config = _Config()
    request = RunExecutionRequest(
        config=str(tmp_path / "config.yaml"),
        device="cpu",
        profile="dev",
        baseline=str(tmp_path / "baseline.json"),
    )
    observed = []

    services = RunExecutionServices(
        SnapshotRestoreFailed=RuntimeError,
        adjust_edit_params=lambda *args, **kwargs: SimpleNamespace(
            params={}, diagnostics=()
        ),
        assemble_run_report=lambda **_kwargs: SimpleNamespace(
            report={
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    }
                },
                "artifacts": {},
            },
            timings={},
            provenance_result=SimpleNamespace(
                missing_evaluation_windows_for_baseline=False
            ),
            metrics_enrichment=SimpleNamespace(
                pairing_violations=(),
                debug_diffs_line=None,
            ),
        ),
        build_snapshot_execution_plan=lambda **_kwargs: SimpleNamespace(
            model=object(),
            restore_fn=None,
            skip_model_load=False,
            snapshot_tmpdir=None,
            snapshot_provenance={"restore_failed": False, "reload_path_used": False},
            emitted_skip_overhead_warning=False,
            snapshot_enabled=False,
            diagnostics=(),
        ),
        build_provider_dataset_plan=lambda **_kwargs: None,
        execute_guarded_run=lambda **_kwargs: (
            SimpleNamespace(
                edit={},
                metrics={},
                guards={},
                context={},
                evaluation_windows={},
                status="success",
            ),
            object(),
        ),
        load_baseline_pairing_evidence=lambda **_kwargs: SimpleNamespace(
            status="fallback",
            message="baseline schedule unavailable",
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=None,
        ),
        materialize_run_dataset=lambda **_kwargs: SimpleNamespace(
            diagnostics=(
                SimpleNamespace(
                    code="dataset.code",
                    message="code message",
                    details={"source": "inner"},
                ),
                SimpleNamespace(
                    kind="dataset.kind",
                    message="kind message",
                    severity="warning",
                    metadata={"marker": 1},
                ),
                SimpleNamespace(message="legacy message"),
            ),
            resolved_split="validation",
            used_fallback_split=False,
            tokenizer=None,
            tokenizer_hash=None,
            calibration_data=[],
            dataset_meta={},
            window_plan=None,
            preview_count=1,
            final_count=1,
            effective_preview=1,
            effective_final=1,
            preview_mask_counts=[],
            final_mask_counts=[],
            preview_records=[],
            final_records=[],
        ),
        free_model_memory=lambda _model: None,
        init_retry_controller=lambda **_kwargs: SimpleNamespace(
            attempt_history=[{"attempt": 1}],
            get_attempt_summary=lambda: ["not-a-dict"],
        ),
        load_model_with_cfg=lambda *args, **kwargs: object(),
        persist_run_report_outputs=lambda **_kwargs: SimpleNamespace(
            report_path_out=str(tmp_path / "report.json"),
            telemetry_saved_path=None,
            telemetry_error="telemetry failed",
        ),
        prepare_config_for_run=lambda **_kwargs: config,
        resolve_device_and_output=lambda _cfg, **_kwargs: ("cpu", tmp_path),
        resolve_snapshot_config=lambda _context: {},
        resolve_snapshot_retry_transition=lambda **_kwargs: SimpleNamespace(
            skip_model_load=False,
            emitted_skip_overhead_warning=False,
            diagnostics=(),
        ),
        run_bare_control=lambda **_kwargs: None,
        safe_int=lambda value, default=0: int(value if value is not None else default),
        to_serialisable_dict=lambda value: value,
        validate_retry_evaluation_report=lambda **_kwargs: SimpleNamespace(
            telemetry_summary=None
        ),
        validate_and_harvest_baseline_schedule=lambda **_kwargs: None,
        materialize_baseline_pairing_schedule=lambda **_kwargs: None,
        resolve_tokenizer=lambda **_kwargs: (None, None),
        detect_model_profile=lambda **_kwargs: SimpleNamespace(default_loss="ce"),
        get_psutil=lambda: None,
        get_torch=lambda: SimpleNamespace(
            initial_seed=lambda: 7,
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(benchmark=False, deterministic=False)
            ),
        ),
    )

    outcome = execute_run_request(
        request,
        services=services,
        observer=observed.append,
    )

    assert outcome.ok is True
    assert outcome.result is not None
    assert outcome.result.report_path == str(tmp_path / "report.json")
    event_map = {
        (event.phase, event.code): event.details
        for event in outcome.events
        if event.code in {"dataset.code", "dataset.kind", "legacy_notice"}
    }
    assert event_map[("diagnostic", "dataset.code")]["diagnostic_source"] == "dataset"
    assert event_map[("diagnostic", "dataset.code")]["source"] == "inner"
    assert event_map[("diagnostic", "dataset.kind")]["marker"] == 1
    assert event_map[("diagnostic", "dataset.kind")]["severity"] == "warning"
    assert event_map[("diagnostic", "legacy_notice")]["message"] == "legacy message"
    assert ("diagnostic", "baseline_schedule_fallback") in {
        (event.phase, event.code) for event in outcome.events
    }
    assert ("status", "telemetry_failed") in {
        (event.phase, event.code) for event in outcome.events
    }
    assert ("summary", "retry_summary") not in {
        (event.phase, event.code) for event in outcome.events
    }
