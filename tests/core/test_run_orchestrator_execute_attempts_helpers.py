from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_orchestrator import RunAttemptStartedEvent
from invarlock.core.run_orchestrator_execute_attempts import (
    _emit_attempt_start,
    _emit_primary_metric_summary_from_report,
    _enforce_guard_overhead_budget,
    _execute_attempt_core,
    _process_attempt_result,
    _resolve_export_model_dir,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    _AttemptExecutionState,
    _RunExecutionState,
)


class _TimedStep:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_emit_attempt_start_emits_follow_on_attempt_without_retry_controller() -> None:
    events: list[object] = []

    _emit_attempt_start(
        emit=events.append,
        retry_controller=None,
        attempt=2,
        max_attempts=4,
    )

    assert len(events) == 1
    assert isinstance(events[0], RunAttemptStartedEvent)
    assert events[0].attempt == 2
    assert events[0].max_attempts is None


def test_execute_attempt_core_continues_after_retryable_restore_failure() -> None:
    transitions: list[tuple[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    freed_models: list[object | None] = []

    execution_state = _RunExecutionState(
        runner=object(),
        auto_config={},
        edit_config={"alpha": 1},
        model=object(),
        restore_fn=None,
        snapshot_tmpdir=None,
        snapshot_provenance={},
        skip_model_load=False,
        emitted_skip_overhead_warning=False,
    )

    result = _execute_attempt_core(
        attempt=1,
        max_attempts=3,
        retry_controller=object(),
        seed_bundle={"python": 7},
        seed_value=7,
        edit_op=SimpleNamespace(name="noop"),
        cfg=object(),
        adapter=object(),
        run_config=object(),
        guards=[],
        calibration_data=[],
        preview_count=1,
        final_count=1,
        resolved_device="cpu",
        profile_normalized="dev",
        guard_overhead_threshold=0.01,
        skip_overhead=False,
        skip_overhead_source=None,
        measure_guard_overhead=False,
        resolved_loss_type="ce",
        prefer_local_files_only=False,
        execution_state=execution_state,
        adjust_edit_params_fn=lambda *_args, **_kwargs: SimpleNamespace(
            params={"alpha": 1}, diagnostics=()
        ),
        run_bare_control_fn=lambda **_kwargs: None,
        execute_guarded_run_fn=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("restore failed")
        ),
        snapshot_restore_failed_type=RuntimeError,
        build_restore_failure_attempt_summary_fn=lambda: {
            "passed": False,
            "failures": ["restore_failed"],
        },
        decide_failed_retry_transition_fn=lambda *_args, **_kwargs: SimpleNamespace(
            should_retry=True,
            next_attempt=2,
            diagnostics=(SimpleNamespace(code="retry.failure"),),
        ),
        free_model_memory_fn=lambda model: freed_models.append(model),
        emit=lambda _event: None,
        emit_transition=lambda kind, diagnostic: transitions.append((kind, diagnostic)),
        emit_diagnostic=lambda **kwargs: diagnostics.append(kwargs),
        halt=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("halt should not run")
        ),
        record_timed_step=lambda _label: _TimedStep(),
    )

    assert result.should_continue is True
    assert result.attempt == 2
    assert len(freed_models) == 1
    assert freed_models[0] is not None
    assert execution_state.model is None
    assert diagnostics[0]["code"] == "snapshot_restore_fallback"
    assert transitions[0][0] == "retry_failure"


def test_execute_attempt_core_falls_back_to_reload_without_retry_controller() -> None:
    diagnostics: list[dict[str, object]] = []
    freed_models: list[object | None] = []
    retry_decisions: list[object] = []

    execution_state = _RunExecutionState(
        runner=object(),
        auto_config={},
        edit_config={"alpha": 1},
        model=object(),
        restore_fn=object(),
        snapshot_tmpdir=None,
        snapshot_provenance={"restore_failed": False, "reload_path_used": False},
        skip_model_load=False,
        emitted_skip_overhead_warning=False,
    )

    result = _execute_attempt_core(
        attempt=1,
        max_attempts=1,
        retry_controller=None,
        seed_bundle={"python": 7},
        seed_value=7,
        edit_op=SimpleNamespace(name="noop"),
        cfg=object(),
        adapter=object(),
        run_config=object(),
        guards=[],
        calibration_data=[],
        preview_count=1,
        final_count=1,
        resolved_device="cpu",
        profile_normalized="dev",
        guard_overhead_threshold=0.01,
        skip_overhead=False,
        skip_overhead_source=None,
        measure_guard_overhead=False,
        resolved_loss_type="ce",
        prefer_local_files_only=False,
        execution_state=execution_state,
        adjust_edit_params_fn=lambda *_args, **_kwargs: SimpleNamespace(
            params={"alpha": 1}, diagnostics=()
        ),
        run_bare_control_fn=lambda **_kwargs: None,
        execute_guarded_run_fn=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("restore failed")
        ),
        snapshot_restore_failed_type=RuntimeError,
        build_restore_failure_attempt_summary_fn=lambda: {},
        decide_failed_retry_transition_fn=lambda *_args, **_kwargs: (
            retry_decisions.append(object())
        ),
        free_model_memory_fn=lambda model: freed_models.append(model),
        emit=lambda _event: None,
        emit_transition=lambda *_args, **_kwargs: None,
        emit_diagnostic=lambda **kwargs: diagnostics.append(kwargs),
        halt=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("halt should not run")
        ),
        record_timed_step=lambda _label: _TimedStep(),
    )

    assert result.should_continue is True
    assert result.attempt == 1
    assert execution_state.model is None
    assert execution_state.restore_fn is None
    assert execution_state.snapshot_provenance["restore_failed"] is True
    assert len(freed_models) == 1
    assert diagnostics[0]["code"] == "snapshot_restore_fallback"
    assert retry_decisions == []


def test_execute_attempt_core_halts_after_repeated_reload_restore_failure() -> None:
    execution_state = _RunExecutionState(
        runner=object(),
        auto_config={},
        edit_config={},
        model=object(),
        restore_fn=object(),
        snapshot_tmpdir=None,
        snapshot_provenance={"restore_failed": True, "reload_path_used": False},
        skip_model_load=False,
        emitted_skip_overhead_warning=False,
    )
    halted: list[tuple[str, dict[str, object]]] = []

    with pytest.raises(RuntimeError, match="halted"):
        _execute_attempt_core(
            attempt=1,
            max_attempts=1,
            retry_controller=None,
            seed_bundle={"python": 7},
            seed_value=7,
            edit_op=SimpleNamespace(name="noop"),
            cfg=object(),
            adapter=object(),
            run_config=object(),
            guards=[],
            calibration_data=[],
            preview_count=1,
            final_count=1,
            resolved_device="cpu",
            profile_normalized="dev",
            guard_overhead_threshold=0.01,
            skip_overhead=False,
            skip_overhead_source=None,
            measure_guard_overhead=False,
            resolved_loss_type="ce",
            prefer_local_files_only=False,
            execution_state=execution_state,
            adjust_edit_params_fn=lambda *_args, **_kwargs: SimpleNamespace(
                params={}, diagnostics=()
            ),
            run_bare_control_fn=lambda **_kwargs: None,
            execute_guarded_run_fn=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("restore failed again")
            ),
            snapshot_restore_failed_type=RuntimeError,
            build_restore_failure_attempt_summary_fn=lambda: {},
            decide_failed_retry_transition_fn=lambda *_args, **_kwargs: SimpleNamespace(
                should_retry=False,
                next_attempt=1,
                diagnostics=(),
            ),
            free_model_memory_fn=lambda _model: None,
            emit=lambda _event: None,
            emit_transition=lambda *_args, **_kwargs: None,
            emit_diagnostic=lambda **_kwargs: None,
            halt=lambda code, **kwargs: (
                halted.append((code, kwargs)),
                (_ for _ in ()).throw(RuntimeError("halted")),
            )[1],
            record_timed_step=lambda _label: _TimedStep(),
        )

    assert halted[0][0] == "snapshot_restore_failed"


def test_execute_attempt_core_halts_with_default_failed_status_message() -> None:
    execution_state = _RunExecutionState(
        runner=object(),
        auto_config={},
        edit_config={},
        model=object(),
        restore_fn=None,
        snapshot_tmpdir=None,
        snapshot_provenance={},
        skip_model_load=False,
        emitted_skip_overhead_warning=False,
    )
    halted: list[tuple[str, dict[str, object]]] = []

    with pytest.raises(RuntimeError, match="halted"):
        _execute_attempt_core(
            attempt=1,
            max_attempts=1,
            retry_controller=None,
            seed_bundle={"python": 7},
            seed_value=7,
            edit_op=SimpleNamespace(name="noop"),
            cfg=object(),
            adapter=object(),
            run_config=object(),
            guards=[],
            calibration_data=[],
            preview_count=1,
            final_count=1,
            resolved_device="cpu",
            profile_normalized="dev",
            guard_overhead_threshold=0.01,
            skip_overhead=False,
            skip_overhead_source=None,
            measure_guard_overhead=False,
            resolved_loss_type="ce",
            prefer_local_files_only=False,
            execution_state=execution_state,
            adjust_edit_params_fn=lambda *_args, **_kwargs: SimpleNamespace(
                params={}, diagnostics=()
            ),
            run_bare_control_fn=lambda **_kwargs: None,
            execute_guarded_run_fn=lambda **_kwargs: (
                SimpleNamespace(status="failed", error=""),
                object(),
            ),
            snapshot_restore_failed_type=RuntimeError,
            build_restore_failure_attempt_summary_fn=lambda: {},
            decide_failed_retry_transition_fn=lambda *_args, **_kwargs: SimpleNamespace(
                should_retry=False,
                next_attempt=1,
                diagnostics=(),
            ),
            free_model_memory_fn=lambda _model: None,
            emit=lambda _event: None,
            emit_transition=lambda *_args, **_kwargs: None,
            emit_diagnostic=lambda **_kwargs: None,
            halt=lambda code, **kwargs: (
                halted.append((code, kwargs)),
                (_ for _ in ()).throw(RuntimeError("halted")),
            )[1],
            record_timed_step=lambda _label: _TimedStep(),
        )

    assert halted[0][0] == "pipeline_failed"
    assert (
        halted[0][1]["summary"]
        == "Guarded run failed before report assembly (status: failed)."
    )


def test_resolve_export_model_dir_covers_dict_configuration_paths(
    tmp_path: Path,
) -> None:
    from_dict = _resolve_export_model_dir(
        output_cfg={"model_path": "artifacts/model"},
        run_dir=tmp_path,
        export_dir_override=None,
        optional_runtime_exceptions=(RuntimeError, TypeError, ValueError),
    )
    from_subdir = _resolve_export_model_dir(
        output_cfg={"model_subdir": "hf-export"},
        run_dir=tmp_path,
        export_dir_override=None,
        optional_runtime_exceptions=(RuntimeError, TypeError, ValueError),
    )

    assert from_dict == tmp_path / "artifacts/model"
    assert from_subdir == tmp_path / "hf-export"


def test_resolve_export_model_dir_uses_override_when_output_is_none(
    tmp_path: Path,
) -> None:
    resolved = _resolve_export_model_dir(
        output_cfg=None,
        run_dir=tmp_path,
        export_dir_override="override-dir",
        optional_runtime_exceptions=(RuntimeError, TypeError, ValueError),
    )

    assert resolved == tmp_path / "override-dir"


def test_emit_primary_metric_summary_from_report_swallows_emit_failures() -> None:
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 2.0,
                "ratio_vs_baseline": 1.5,
            }
        }
    }

    _emit_primary_metric_summary_from_report(
        report=report,
        emit=lambda _event: (_ for _ in ()).throw(TypeError("emit boom")),
    )
    _emit_primary_metric_summary_from_report(
        report={"metrics": {"primary_metric": {"preview": "bad", "final": 2.0}}},
        emit=lambda _event: (_ for _ in ()).throw(AssertionError("should not emit")),
    )


def test_enforce_guard_overhead_budget_handles_context_lookup_errors() -> None:
    summaries: list[dict[str, object]] = []
    halts: list[tuple[str, dict[str, object]]] = []

    class _BadContext:
        def get(self, *_args, **_kwargs):
            raise TypeError("bad context")

    _enforce_guard_overhead_budget(
        report={
            "guard_overhead": {
                "passed": False,
                "evaluated": True,
                "overhead_threshold": 0.05,
            }
        },
        run_config=SimpleNamespace(context=_BadContext()),
        measure_guard_overhead=True,
        guard_overhead_threshold=0.02,
        emit_guard_overhead_summary=lambda info, default_threshold: summaries.append(
            {
                "info": info,
                "default_threshold": default_threshold,
            }
        ),
        halt=lambda code, **kwargs: halts.append((code, kwargs)),
    )

    assert summaries[0]["default_threshold"] == 0.02
    assert halts == [
        (
            "guard_overhead_budget_exceeded",
            {"threshold_fraction": 0.05},
        )
    ]


def test_process_attempt_result_maps_halt_invarlock_errors() -> None:
    halts: list[tuple[str, dict[str, object]]] = []

    def _halt(code: str, **kwargs):
        halts.append((code, kwargs))
        if code == "baseline_windows_missing":
            raise InvarlockError(code="E001", message="missing baseline windows")
        raise RuntimeError("stop-after-invarlock-error")

    with pytest.raises(RuntimeError, match="stop-after-invarlock-error"):
        _process_attempt_result(
            attempt_state=_AttemptExecutionState(
                attempt=1,
                edit_config={},
                guard_overhead_payload=None,
                core_report=SimpleNamespace(status="success"),
                model=None,
                should_continue=False,
            ),
            timings={},
            report_path_out=None,
            cfg=object(),
            profile_normalized="dev",
            profile="dev",
            baseline="baseline.json",
            edit_label=None,
            metric_kind=None,
            export_model_requested=False,
            export_dir_override=None,
            telemetry=False,
            resolved_loss_type="ce",
            tokenizer=None,
            tokenizer_hash=None,
            resolved_split="validation",
            preview_count=1,
            final_count=1,
            effective_preview=1,
            effective_final=1,
            preview_records=[],
            final_records=[],
            preview_mask_counts=[],
            final_mask_counts=[],
            use_mlm=False,
            used_fallback_split=False,
            baseline_report_data=None,
            window_plan=None,
            model_profile=SimpleNamespace(),
            determinism_meta=None,
            guard_overhead_threshold=0.02,
            pm_acceptance_range=None,
            pm_drift_band=None,
            seed_bundle={},
            run_dir=Path("."),
            run_config=SimpleNamespace(context={}),
            auto_config={},
            resolved_device="cpu",
            snapshot_provenance={},
            edit_op=SimpleNamespace(name="noop"),
            adapter=object(),
            model=None,
            measure_guard_overhead=False,
            retry_controller=None,
            validate_retry_evaluation_report_fn=lambda **_kwargs: None,
            resolve_retry_validation_transition_fn=lambda *_args, **_kwargs: None,
            record_retry_attempt_fn=lambda **_kwargs: None,
            persist_run_report_outputs_fn=lambda **_kwargs: SimpleNamespace(
                report_path_out=None,
                telemetry_saved_path=None,
                telemetry_error=None,
            ),
            assemble_run_report_fn=lambda **_kwargs: SimpleNamespace(
                report={"metrics": {}, "artifacts": {}},
                timings={},
                provenance_result=SimpleNamespace(
                    missing_evaluation_windows_for_baseline=True,
                    missing_evaluation_windows_message="baseline windows missing",
                ),
                metrics_enrichment=SimpleNamespace(
                    pairing_violations=(),
                    debug_diffs_line=None,
                ),
            ),
            cfg_value=lambda _cfg, _name: {},
            emit=lambda _event: None,
            emit_diagnostic=lambda **_kwargs: None,
            emit_guard_overhead_summary=lambda *_args, **_kwargs: None,
            emit_transition=lambda *_args, **_kwargs: None,
            halt=_halt,
            fail_run=lambda *_args, **_kwargs: None,
            optional_runtime_exceptions=(RuntimeError, TypeError, ValueError),
        )

    assert halts[0][0] == "baseline_windows_missing"
    assert halts[1][0] == "invarlock_error"
