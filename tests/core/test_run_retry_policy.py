from __future__ import annotations

from invarlock.core.retry import RetryDiagnostic
from invarlock.core.run_retry_policy import (
    apply_mask_only_head_autotune,
    build_restore_failure_attempt_summary,
    build_retry_result_summary,
    decide_failed_retry_transition,
    record_retry_attempt,
    resolve_retry_validation_decision,
    resolve_retry_validation_transition,
)


def test_build_retry_result_summary_collects_failed_gates() -> None:
    out = build_retry_result_summary(
        {"primary_metric_acceptable": False, "drift_ok": True}
    )

    assert out == {
        "passed": False,
        "failures": ["primary_metric_acceptable"],
        "validation": {
            "primary_metric_acceptable": False,
            "drift_ok": True,
        },
    }


def test_apply_mask_only_head_autotune_updates_search_state() -> None:
    updated, adjustment = apply_mask_only_head_autotune(
        {
            "heads": {
                "mask_only": True,
                "_auto_search": {
                    "keep_low": 0,
                    "keep_high": 8,
                    "keep_current": 4,
                },
            }
        },
        {"primary_metric_acceptable": False, "drift_ok": True},
    )

    assert adjustment == {
        "global_k": 6,
        "keep_low": 4,
        "keep_high": 8,
        "failed_gate_count": 1,
    }
    assert updated["heads"]["global_k"] == 6
    assert updated["heads"]["_auto_search"] == {
        "keep_low": 4,
        "keep_high": 8,
        "keep_current": 6,
    }


def test_apply_mask_only_head_autotune_noops_without_supported_section() -> None:
    original = {"heads": {"mask_only": False}}
    updated, adjustment = apply_mask_only_head_autotune(
        original,
        {"primary_metric_acceptable": False},
    )

    assert updated == original
    assert adjustment is None


def test_apply_mask_only_head_autotune_fails_closed_on_bad_values() -> None:
    updated, adjustment = apply_mask_only_head_autotune(
        {
            "head_budget": {
                "mask_only": True,
                "_auto_search": {"keep_low": "bad", "keep_high": 8},
            }
        },
        {"primary_metric_acceptable": False},
    )

    assert updated == {
        "head_budget": {
            "mask_only": True,
            "_auto_search": {"keep_low": "bad", "keep_high": 8},
        }
    }
    assert adjustment is None


class _FakeRetryController:
    def __init__(
        self, *, should_retry: bool, diagnostics: tuple[RetryDiagnostic, ...] = ()
    ) -> None:
        self._should_retry = should_retry
        self._diagnostics = list(diagnostics)
        self.recorded: list[tuple[int, dict[str, object], dict[str, object]]] = []
        self.last_passed: bool | None = None

    def record_attempt(
        self,
        attempt: int,
        result_summary: dict[str, object],
        edit_config: dict[str, object],
    ) -> None:
        self.recorded.append((attempt, dict(result_summary), dict(edit_config)))

    def should_retry(self, passed: bool) -> bool:
        self.last_passed = passed
        return self._should_retry

    def drain_diagnostics(self) -> tuple[RetryDiagnostic, ...]:
        diagnostics = tuple(self._diagnostics)
        self._diagnostics.clear()
        return diagnostics


def test_record_retry_attempt_noops_without_controller() -> None:
    record_retry_attempt(
        None,
        attempt=2,
        attempt_summary={"passed": False},
        edit_config={"energy_keep": 0.99},
    )


def test_record_retry_attempt_normalizes_payloads() -> None:
    controller = _FakeRetryController(should_retry=False)

    record_retry_attempt(
        controller,
        attempt=2,
        attempt_summary={"passed": False, "failures": ["gate"]},
        edit_config={"energy_keep": 0.99},
    )

    assert controller.recorded == [
        (
            2,
            {"passed": False, "failures": ["gate"]},
            {"energy_keep": 0.99},
        )
    ]


def test_decide_failed_retry_transition_advances_attempt_and_drains_diagnostics() -> (
    None
):
    controller = _FakeRetryController(
        should_retry=True,
        diagnostics=(
            RetryDiagnostic(
                code="retry.budget_available",
                message="Retry budget available",
            ),
        ),
    )

    transition = decide_failed_retry_transition(
        controller,
        attempt=2,
        attempt_summary={"passed": False, "failures": ["primary_metric_acceptable"]},
        edit_config={"energy_keep": 0.99},
    )

    assert controller.last_passed is False
    assert transition.should_retry is True
    assert transition.next_attempt == 3
    assert transition.diagnostics == (
        RetryDiagnostic(
            code="retry.budget_available",
            message="Retry budget available",
        ),
    )
    assert controller.recorded == [
        (
            2,
            {
                "passed": False,
                "failures": ["primary_metric_acceptable"],
            },
            {"energy_keep": 0.99},
        )
    ]


def test_decide_failed_retry_transition_holds_attempt_when_stopping() -> None:
    controller = _FakeRetryController(should_retry=False)

    transition = decide_failed_retry_transition(
        controller,
        attempt=3,
        attempt_summary={"passed": False, "failures": ["restore_failed"]},
        edit_config={"clamp_ratio": 0.01},
    )

    assert transition.should_retry is False
    assert transition.next_attempt == 3
    assert transition.diagnostics == ()


def test_build_restore_failure_attempt_summary_is_stable() -> None:
    assert build_restore_failure_attempt_summary() == {
        "passed": False,
        "failures": ["restore_failed"],
        "validation": {},
    }


def test_resolve_retry_validation_decision_marks_retry_and_applies_autotune() -> None:
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "failed",
            "failed_gates": ("primary_metric_acceptable",),
            "validation": {"primary_metric_acceptable": False},
            "error_message": None,
        },
    )()

    decision = resolve_retry_validation_decision(
        edit_config={
            "heads": {
                "mask_only": True,
                "_auto_search": {
                    "keep_low": 0,
                    "keep_high": 8,
                    "keep_current": 4,
                },
            }
        },
        validation_result=validation_result,
        should_retry=True,
    )

    assert decision.action == "retry"
    assert decision.failed_gates == ("primary_metric_acceptable",)
    assert decision.head_adjustment == {
        "global_k": 6,
        "keep_low": 4,
        "keep_high": 8,
        "failed_gate_count": 1,
    }
    assert decision.updated_edit_config["heads"]["global_k"] == 6


def test_resolve_retry_validation_decision_marks_error_fail_closed() -> None:
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "error",
            "failed_gates": (),
            "validation": {},
            "error_message": "boom",
        },
    )()

    decision = resolve_retry_validation_decision(
        edit_config={"energy_keep": 0.99},
        validation_result=validation_result,
        should_retry=False,
    )

    assert decision.action == "error"
    assert decision.failed_gates == ("report_error",)
    assert decision.error_message == "boom"


def test_resolve_retry_validation_transition_records_pass_without_retry() -> None:
    controller = _FakeRetryController(
        should_retry=True,
        diagnostics=(RetryDiagnostic(code="retry.unused", message="unused"),),
    )
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "passed",
            "attempt_summary": {"passed": True, "failures": [], "validation": {}},
            "failed_gates": (),
            "validation": {},
            "error_message": None,
        },
    )()

    transition = resolve_retry_validation_transition(
        controller,
        attempt=2,
        validation_result=validation_result,
        edit_config={"energy_keep": 0.99},
    )

    assert transition.action == "passed"
    assert transition.next_attempt == 2
    assert controller.recorded == [
        (2, {"passed": True, "failures": [], "validation": {}}, {"energy_keep": 0.99})
    ]


def test_resolve_retry_validation_transition_retries_with_diagnostics() -> None:
    controller = _FakeRetryController(
        should_retry=True,
        diagnostics=(RetryDiagnostic(code="retry.retry", message="retry"),),
    )
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "failed",
            "attempt_summary": {
                "passed": False,
                "failures": ["primary_metric_acceptable"],
                "validation": {"primary_metric_acceptable": False},
            },
            "failed_gates": ("primary_metric_acceptable",),
            "validation": {"primary_metric_acceptable": False},
            "error_message": None,
        },
    )()

    transition = resolve_retry_validation_transition(
        controller,
        attempt=2,
        validation_result=validation_result,
        edit_config={
            "heads": {
                "mask_only": True,
                "_auto_search": {
                    "keep_low": 0,
                    "keep_high": 8,
                    "keep_current": 4,
                },
            }
        },
    )

    assert transition.action == "retry"
    assert transition.diagnostics == (
        RetryDiagnostic(code="retry.retry", message="retry"),
    )
    assert transition.next_attempt == 3
    assert transition.head_adjustment == {
        "global_k": 6,
        "keep_low": 4,
        "keep_high": 8,
        "failed_gate_count": 1,
    }


def test_resolve_retry_validation_transition_stops_on_error_without_retry_probe() -> (
    None
):
    controller = _FakeRetryController(should_retry=True)
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "error",
            "attempt_summary": {
                "passed": False,
                "failures": ["report_error"],
                "validation": {},
            },
            "failed_gates": (),
            "validation": {},
            "error_message": "boom",
        },
    )()

    transition = resolve_retry_validation_transition(
        controller,
        attempt=2,
        validation_result=validation_result,
        edit_config={"energy_keep": 0.99},
    )

    assert transition.action == "error"
    assert transition.error_message == "boom"
    assert controller.last_passed is None


def test_resolve_retry_validation_decision_marks_passed_without_mutation() -> None:
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "passed",
            "failed_gates": (),
            "validation": {},
            "error_message": None,
        },
    )()

    decision = resolve_retry_validation_decision(
        edit_config={"energy_keep": 0.99},
        validation_result=validation_result,
        should_retry=False,
    )

    assert decision.action == "passed"
    assert decision.failed_gates == ()
    assert decision.updated_edit_config == {"energy_keep": 0.99}
    assert decision.head_adjustment is None


def test_resolve_retry_validation_decision_marks_exhausted_when_budget_done() -> None:
    validation_result = type(
        "ValidationResult",
        (),
        {
            "status": "failed",
            "failed_gates": ("pm_ratio",),
            "validation": {"pm_ratio": False},
            "error_message": None,
        },
    )()

    decision = resolve_retry_validation_decision(
        edit_config={"energy_keep": 0.99},
        validation_result=validation_result,
        should_retry=False,
    )

    assert decision.action == "exhausted"
    assert decision.failed_gates == ("pm_ratio",)
    assert decision.updated_edit_config == {"energy_keep": 0.99}
