from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.exceptions import InvarlockError
from invarlock.core.orchestration.attempts import (
    _build_skipped_guard_metric_impact_payload,
    _emit_attempt_start,
    _emit_primary_metric_summary_from_report,
    _enforce_guard_metric_impact_budget,
    _execute_attempt_core,
    _process_attempt_result,
    _resolve_export_model_dir,
)
from invarlock.core.orchestration.helpers import (
    _AttemptExecutionState,
    _RunExecutionState,
)
from invarlock.core.run_orchestrator import (
    RunAttemptStartedEvent,
    RunPrimaryMetricSummaryEvent,
)


class _TimedStep:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_skipped_guard_metric_impact_payload_is_not_a_pass() -> None:
    payload = _build_skipped_guard_metric_impact_payload(
        guard_metric_degradation_limit=0.01,
        skip_guard_metric_impact_source="config:context.run.skip_guard_metric_impact_check",
    )

    assert payload["skipped"] is True
    assert payload["evaluated"] is False
    assert payload["passed"] is False


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
        emitted_skip_guard_metric_impact_warning=False,
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
        guard_metric_degradation_limit=0.01,
        skip_guard_metric_impact=False,
        skip_guard_metric_impact_source=None,
        measure_guard_metric_impact=False,
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


@pytest.mark.parametrize(
    ("attempt", "expected_restore_before_run"),
    [(1, False), (2, True)],
)
def test_execute_attempt_core_only_skips_pristine_first_bare_restore(
    attempt: int,
    expected_restore_before_run: bool,
) -> None:
    bare_calls: list[dict[str, object]] = []
    guarded_calls: list[dict[str, object]] = []
    execution_state = _RunExecutionState(
        runner=object(),
        auto_config={},
        edit_config={"alpha": 1},
        model=object(),
        restore_fn=lambda: None,
        snapshot_tmpdir=None,
        snapshot_provenance={"restore_failed": False},
        skip_model_load=False,
        emitted_skip_guard_metric_impact_warning=False,
    )

    result = _execute_attempt_core(
        attempt=attempt,
        max_attempts=3,
        retry_controller=object() if attempt > 1 else None,
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
        profile_normalized="ci",
        guard_metric_degradation_limit=0.01,
        skip_guard_metric_impact=False,
        skip_guard_metric_impact_source=None,
        measure_guard_metric_impact=True,
        resolved_loss_type="ce",
        prefer_local_files_only=False,
        execution_state=execution_state,
        adjust_edit_params_fn=lambda *_args, **_kwargs: SimpleNamespace(
            params={"alpha": 2}, diagnostics=()
        ),
        run_bare_control_fn=lambda **kwargs: bare_calls.append(kwargs) or {},
        execute_guarded_run_fn=lambda **kwargs: (
            guarded_calls.append(kwargs)
            or (SimpleNamespace(status="success"), execution_state.model)
        ),
        snapshot_restore_failed_type=RuntimeError,
        build_restore_failure_attempt_summary_fn=lambda: {},
        decide_failed_retry_transition_fn=lambda *_args, **_kwargs: None,
        free_model_memory_fn=lambda _model: None,
        emit=lambda _event: None,
        emit_transition=lambda *_args, **_kwargs: None,
        emit_diagnostic=lambda **_kwargs: None,
        halt=lambda *_args, **_kwargs: None,
        record_timed_step=lambda _label: _TimedStep(),
    )

    assert result.should_continue is False
    assert bare_calls[0]["restore_before_run"] is expected_restore_before_run
    assert guarded_calls[0]["restore_fn"] is execution_state.restore_fn


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
        emitted_skip_guard_metric_impact_warning=False,
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
        guard_metric_degradation_limit=0.01,
        skip_guard_metric_impact=False,
        skip_guard_metric_impact_source=None,
        measure_guard_metric_impact=False,
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
        emitted_skip_guard_metric_impact_warning=False,
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
            guard_metric_degradation_limit=0.01,
            skip_guard_metric_impact=False,
            skip_guard_metric_impact_source=None,
            measure_guard_metric_impact=False,
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
        emitted_skip_guard_metric_impact_warning=False,
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
            guard_metric_degradation_limit=0.01,
            skip_guard_metric_impact=False,
            skip_guard_metric_impact_source=None,
            measure_guard_metric_impact=False,
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
    events: list[object] = []

    def _emit(event: object) -> None:
        events.append(event)
        raise TypeError("emit boom")

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

    result = _emit_primary_metric_summary_from_report(
        report=report,
        emit=_emit,
    )
    assert result is None
    assert len(events) == 1
    assert isinstance(events[0], RunPrimaryMetricSummaryEvent)
    assert events[0].metric_kind == "ppl_causal"
    assert events[0].preview == 1.0
    assert events[0].final == 2.0
    assert events[0].ratio_vs_baseline == 1.5
    _emit_primary_metric_summary_from_report(
        report={"metrics": {"primary_metric": {"preview": "bad", "final": 2.0}}},
        emit=lambda _event: (_ for _ in ()).throw(AssertionError("should not emit")),
    )


def test_enforce_guard_metric_impact_budget_halts_failed_measurement() -> None:
    summaries: list[dict[str, object]] = []
    halts: list[tuple[str, dict[str, object]]] = []

    _enforce_guard_metric_impact_budget(
        report={
            "guard_metric_impact": {
                "passed": False,
                "evaluated": True,
                "degradation": 1.1,
                "degradation_limit": 0.05,
                "degradation_basis": "absolute_drop",
            }
        },
        measure_guard_metric_impact=True,
        guard_metric_degradation_limit=0.02,
        emit_guard_metric_impact_summary=lambda info, default_limit: summaries.append(
            {
                "info": info,
                "default_limit": default_limit,
            }
        ),
        halt=lambda code, **kwargs: halts.append((code, kwargs)),
    )

    assert summaries[0]["default_limit"] == 0.02
    assert halts == [
        (
            "guard_metric_impact_budget_exceeded",
            {
                "degradation_limit": 0.05,
                "degradation_basis": "absolute_drop",
            },
        )
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {},
        {"passed": True, "evaluated": True, "degradation_limit": 0.01},
        {
            "passed": True,
            "evaluated": True,
            "degradation": None,
            "degradation_limit": 0.01,
        },
        {
            "passed": True,
            "evaluated": False,
            "degradation": 1.0,
            "degradation_limit": 0.01,
        },
        {
            "passed": True,
            "evaluated": True,
            "degradation": float("nan"),
            "degradation_limit": 0.01,
        },
        {
            "passed": True,
            "evaluated": True,
            "degradation": 1.0,
            "degradation_limit": -0.01,
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": 1.0,
            "degradation_limit": "not-a-number",
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": 1.0,
            "degradation_limit": True,
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": 1.0,
            "degradation_limit": -0.01,
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": "not-a-number",
            "degradation_limit": 0.01,
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": True,
            "degradation_limit": 0.01,
        },
        {
            "passed": False,
            "evaluated": True,
            "degradation": float("nan"),
            "degradation_limit": 0.01,
        },
        {
            "passed": False,
            "evaluated": False,
            "degradation": 1.0,
            "degradation_limit": 0.01,
        },
    ],
)
def test_enforce_guard_metric_impact_budget_fails_when_evidence_unavailable(
    evidence,
) -> None:
    halts: list[tuple[str, dict[str, object]]] = []

    _enforce_guard_metric_impact_budget(
        report={"guard_metric_impact": evidence} if evidence is not None else {},
        measure_guard_metric_impact=True,
        guard_metric_degradation_limit=0.01,
        emit_guard_metric_impact_summary=lambda *_args, **_kwargs: None,
        halt=lambda code, **kwargs: halts.append((code, kwargs)),
    )

    assert halts
    assert halts[0][0] == "guard_metric_impact_unavailable"


def test_enforce_guard_metric_impact_budget_rejects_malformed_recorded_pass() -> None:
    halts: list[tuple[str, dict[str, object]]] = []
    _enforce_guard_metric_impact_budget(
        report={
            "guard_metric_impact": {
                "passed": True,
                "evaluated": True,
                "degradation": 1.2,
                "degradation_limit": 0.01,
            }
        },
        measure_guard_metric_impact=True,
        guard_metric_degradation_limit=0.01,
        emit_guard_metric_impact_summary=lambda *_args, **_kwargs: None,
        halt=lambda code, **kwargs: halts.append((code, kwargs)),
    )

    assert halts == [
        ("guard_metric_impact_unavailable", {"reason": "canonical contract invalid"})
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
                guard_metric_impact_payload=None,
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
            guard_metric_degradation_limit=0.02,
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
            measure_guard_metric_impact=False,
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
            emit_guard_metric_impact_summary=lambda *_args, **_kwargs: None,
            emit_transition=lambda *_args, **_kwargs: None,
            halt=_halt,
            fail_run=lambda *_args, **_kwargs: None,
            optional_runtime_exceptions=(RuntimeError, TypeError, ValueError),
        )

    assert halts[0][0] == "baseline_windows_missing"
    assert halts[1][0] == "invarlock_error"
