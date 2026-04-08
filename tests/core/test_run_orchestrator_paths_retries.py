from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from invarlock.core.retry import RetryController
from invarlock.core.run_orchestrator import (
    RunDiagnosticEvent,
    RunExecutionRequest,
    RunExecutionServices,
    RunFailureEvent,
    RunRetryAttemptStartedEvent,
    RunRetrySummaryEvent,
    execute_run_request,
)
from tests.core.test_run_orchestrator_paths import (
    _Config,
    _install_common_monkeypatches,
    _make_services,
)


def test_execute_run_request_surfaces_persistence_failures(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()

    _install_common_monkeypatches(monkeypatch)
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "persist_run_report_outputs": lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("persist boom")
            ),
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    failure_codes = [
        event.failure.code
        for event in outcome.events
        if isinstance(event, RunFailureEvent)
    ]
    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "pipeline_failed"
    assert failure_codes == ["pipeline_failed"]


def test_execute_run_request_retries_after_snapshot_restore_failure(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()

    _install_common_monkeypatches(monkeypatch)
    controller = RetryController(max_attempts=2)
    call_models: list[object | None] = []
    call_restore_fns: list[object | None] = []
    base_services = _make_services(tmp_path, config)

    def _execute_guarded_run(**kwargs):
        call_models.append(kwargs["model"])
        call_restore_fns.append(kwargs["restore_fn"])
        if len(call_models) == 1:
            raise RuntimeError("restore boom")
        return (
            SimpleNamespace(
                edit={},
                metrics={},
                guards={},
                context={},
                evaluation_windows={},
                status="success",
            ),
            object(),
        )

    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "init_retry_controller": lambda **_kwargs: controller,
            "execute_guarded_run": _execute_guarded_run,
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    diagnostic_codes = {
        event.code for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    }
    retry_events = [
        event
        for event in outcome.events
        if isinstance(event, RunRetryAttemptStartedEvent)
    ]
    retry_summaries = [
        event for event in outcome.events if isinstance(event, RunRetrySummaryEvent)
    ]

    assert outcome.ok is True
    assert call_models[0] is not None
    assert call_models[1] is None
    assert call_restore_fns == [None, None]
    assert "snapshot_restore_fallback" in diagnostic_codes
    assert [event.attempt for event in retry_events] == [2]
    assert len(retry_summaries) == 1
    assert retry_summaries[0].summary["total_attempts"] == 2
    assert retry_summaries[0].summary["attempts"][0]["failures"] == ["restore_failed"]
    assert retry_summaries[0].summary["attempts"][1]["report_passed"] is True


def test_execute_run_request_fails_when_snapshot_restore_retry_is_exhausted(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()

    _install_common_monkeypatches(monkeypatch)
    controller = RetryController(max_attempts=1)
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "init_retry_controller": lambda **_kwargs: controller,
            "execute_guarded_run": lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("restore boom")
            ),
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    diagnostic_codes = {
        event.code for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    }
    failure_codes = [
        event.failure.code
        for event in outcome.events
        if isinstance(event, RunFailureEvent)
    ]

    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "snapshot_restore_failed"
    assert failure_codes == ["snapshot_restore_failed"]
    assert "snapshot_restore_fallback" in diagnostic_codes
    assert not any(isinstance(event, RunRetrySummaryEvent) for event in outcome.events)
