from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from invarlock.core.retry import RetryDiagnostic


def build_retry_result_summary(
    validation: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Build a stable retry summary from evaluation-report validation flags."""
    validation_map = dict(validation or {})
    failures = [str(key) for key, value in validation_map.items() if not value]
    return {
        "passed": not failures,
        "failures": failures,
        "validation": validation_map,
    }


@dataclass(frozen=True)
class RetryFailureTransition:
    should_retry: bool
    next_attempt: int
    diagnostics: tuple[RetryDiagnostic, ...]


@dataclass(frozen=True)
class RetryValidationDecision:
    status: str
    updated_edit_config: dict[str, Any]
    validation_gates: tuple[str, ...] = ()
    diagnostics: tuple[RetryDiagnostic, ...] = ()
    head_adjustment: dict[str, int] | None = None
    error: RetryDiagnostic | None = None


@dataclass(frozen=True)
class RetryValidationTransition:
    status: str
    updated_edit_config: dict[str, Any]
    validation_gates: tuple[str, ...] = ()
    diagnostics: tuple[RetryDiagnostic, ...] = ()
    next_attempt: int | None = None
    head_adjustment: dict[str, int] | None = None
    error: RetryDiagnostic | None = None


def record_retry_attempt(
    retry_controller: Any,
    *,
    attempt: int,
    attempt_summary: Mapping[str, Any] | None,
    edit_config: Mapping[str, Any] | None,
) -> None:
    """Persist retry attempt data through the controller when enabled."""
    if retry_controller is None:
        return
    retry_controller.record_attempt(
        attempt,
        dict(attempt_summary or {}),
        dict(edit_config or {}),
    )


def decide_failed_retry_transition(
    retry_controller: Any,
    *,
    attempt: int,
    attempt_summary: Mapping[str, Any] | None,
    edit_config: Mapping[str, Any] | None,
    passed: bool = False,
) -> RetryFailureTransition:
    """Record a failed attempt and decide whether the loop should continue."""
    if retry_controller is None:
        return RetryFailureTransition(
            should_retry=False,
            next_attempt=attempt,
            diagnostics=(),
        )

    record_retry_attempt(
        retry_controller,
        attempt=attempt,
        attempt_summary=attempt_summary,
        edit_config=edit_config,
    )
    should_retry = bool(retry_controller.should_retry(bool(passed)))
    drain_diagnostics = getattr(retry_controller, "drain_diagnostics", None)
    if callable(drain_diagnostics):
        diagnostics = tuple(drain_diagnostics())
    else:
        diagnostics = ()
    next_attempt = attempt + 1 if should_retry else attempt
    return RetryFailureTransition(
        should_retry=should_retry,
        next_attempt=next_attempt,
        diagnostics=diagnostics,
    )


def build_restore_failure_attempt_summary() -> dict[str, Any]:
    return {
        "passed": False,
        "failures": ["restore_failed"],
        "validation": {},
    }


def apply_mask_only_head_autotune(
    edit_config: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Advance mask-only head-search state after a failed validation attempt."""
    updated = dict(edit_config or {})
    validation_map = dict(validation or {})

    for section_key in ("heads", "head_budget", "head_budgets"):
        head_section = updated.get(section_key)
        if not isinstance(head_section, Mapping):
            continue
        search = head_section.get("_auto_search")
        if not (isinstance(search, Mapping) and head_section.get("mask_only")):
            continue
        try:
            keep_low = int(search.get("keep_low", 0))
            keep_high = int(search.get("keep_high", search.get("total_heads", 0)))
            keep_current = int(search.get("keep_current", keep_high))
        except (TypeError, ValueError):
            return updated, None

        # A failed gate always reduces pruning aggressiveness for the next attempt.
        keep_low = max(keep_low, keep_current)
        next_keep = int((keep_low + keep_high + 1) // 2)

        next_search = dict(search)
        next_search.update(
            {
                "keep_low": keep_low,
                "keep_high": keep_high,
                "keep_current": next_keep,
            }
        )
        next_head_section = dict(head_section)
        next_head_section["_auto_search"] = next_search
        next_head_section["global_k"] = next_keep
        updated[section_key] = next_head_section

        return updated, {
            "global_k": next_keep,
            "keep_low": keep_low,
            "keep_high": keep_high,
            "failed_gate_count": len(
                [key for key, value in validation_map.items() if not value]
            ),
        }

    return updated, None


def _validation_gates_from_result(validation_result: Any) -> tuple[str, ...]:
    gates = getattr(validation_result, "validation_gates", ())
    return tuple(str(gate) for gate in (gates or ()))


def _validation_error_diagnostic(
    validation_result: Any,
    *,
    validation_gates: tuple[str, ...],
) -> RetryDiagnostic:
    diagnostic = getattr(validation_result, "diagnostic", None)
    if isinstance(diagnostic, RetryDiagnostic):
        return diagnostic

    return RetryDiagnostic(
        code="retry.validation_error",
        message="Retry validation failed",
        severity="error",
        details={"validation_gates": validation_gates},
    )


def resolve_retry_validation_decision(
    *,
    edit_config: Mapping[str, Any] | None,
    validation_result: Any,
    should_retry: bool,
) -> RetryValidationDecision:
    updated_edit_config = dict(edit_config or {})
    status = str(getattr(validation_result, "status", "error") or "error")
    validation_gates = _validation_gates_from_result(validation_result)

    if status == "passed":
        return RetryValidationDecision(
            status="passed",
            updated_edit_config=updated_edit_config,
            validation_gates=validation_gates,
        )

    if status == "failed":
        next_edit_config, head_adjustment = apply_mask_only_head_autotune(
            updated_edit_config,
            getattr(validation_result, "validation", None),
        )
        return RetryValidationDecision(
            status="retry" if should_retry else "exhausted",
            updated_edit_config=next_edit_config,
            validation_gates=validation_gates,
            head_adjustment=head_adjustment,
        )

    error = _validation_error_diagnostic(
        validation_result, validation_gates=validation_gates or ("report_error",)
    )
    return RetryValidationDecision(
        status="error",
        updated_edit_config=updated_edit_config,
        validation_gates=validation_gates or ("report_error",),
        diagnostics=(error,),
        error=error,
    )


def resolve_retry_validation_transition(
    retry_controller: Any,
    *,
    attempt: int,
    validation_result: Any,
    edit_config: Mapping[str, Any] | None,
) -> RetryValidationTransition:
    status = str(getattr(validation_result, "status", "error") or "error")
    attempt_summary = getattr(validation_result, "attempt_summary", None)

    if status == "passed":
        record_retry_attempt(
            retry_controller,
            attempt=attempt,
            attempt_summary=attempt_summary,
            edit_config=edit_config,
        )
        decision = resolve_retry_validation_decision(
            edit_config=edit_config,
            validation_result=validation_result,
            should_retry=False,
        )
        return RetryValidationTransition(
            status=decision.status,
            updated_edit_config=decision.updated_edit_config,
            validation_gates=decision.validation_gates,
            next_attempt=attempt,
            head_adjustment=decision.head_adjustment,
            error=decision.error,
        )

    if status == "failed":
        transition = decide_failed_retry_transition(
            retry_controller,
            attempt=attempt,
            attempt_summary=attempt_summary,
            edit_config=edit_config,
            passed=False,
        )
        decision = resolve_retry_validation_decision(
            edit_config=edit_config,
            validation_result=validation_result,
            should_retry=transition.should_retry,
        )
        return RetryValidationTransition(
            status=decision.status,
            updated_edit_config=decision.updated_edit_config,
            validation_gates=decision.validation_gates,
            diagnostics=transition.diagnostics + decision.diagnostics,
            next_attempt=transition.next_attempt,
            head_adjustment=decision.head_adjustment,
            error=decision.error,
        )

    record_retry_attempt(
        retry_controller,
        attempt=attempt,
        attempt_summary=attempt_summary,
        edit_config=edit_config,
    )
    decision = resolve_retry_validation_decision(
        edit_config=edit_config,
        validation_result=validation_result,
        should_retry=False,
    )
    return RetryValidationTransition(
        status=decision.status,
        updated_edit_config=decision.updated_edit_config,
        validation_gates=decision.validation_gates,
        diagnostics=decision.diagnostics,
        next_attempt=attempt,
        head_adjustment=decision.head_adjustment,
        error=decision.error,
    )
