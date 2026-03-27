from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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
    notices: tuple[str, ...]


@dataclass(frozen=True)
class RetryValidationDecision:
    action: str
    updated_edit_config: dict[str, Any]
    failed_gates: tuple[str, ...]
    head_adjustment: dict[str, int] | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RetryValidationTransition:
    action: str
    updated_edit_config: dict[str, Any]
    failed_gates: tuple[str, ...]
    notices: tuple[str, ...] = ()
    next_attempt: int | None = None
    head_adjustment: dict[str, int] | None = None
    error_message: str | None = None


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
            notices=(),
        )

    record_retry_attempt(
        retry_controller,
        attempt=attempt,
        attempt_summary=attempt_summary,
        edit_config=edit_config,
    )
    should_retry = bool(retry_controller.should_retry(bool(passed)))
    drain_notices = getattr(retry_controller, "drain_notices", None)
    notices = tuple(drain_notices() if callable(drain_notices) else ())
    next_attempt = attempt + 1 if should_retry else attempt
    return RetryFailureTransition(
        should_retry=should_retry,
        next_attempt=next_attempt,
        notices=notices,
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


def resolve_retry_validation_decision(
    *,
    edit_config: Mapping[str, Any] | None,
    validation_result: Any,
    should_retry: bool,
) -> RetryValidationDecision:
    updated_edit_config = dict(edit_config or {})
    status = str(getattr(validation_result, "status", "error") or "error")
    failed_gates = tuple(
        str(gate) for gate in (getattr(validation_result, "failed_gates", ()) or ())
    )

    if status == "passed":
        return RetryValidationDecision(
            action="passed",
            updated_edit_config=updated_edit_config,
            failed_gates=failed_gates,
        )

    if status == "failed":
        next_edit_config, head_adjustment = apply_mask_only_head_autotune(
            updated_edit_config,
            getattr(validation_result, "validation", None),
        )
        return RetryValidationDecision(
            action="retry" if should_retry else "exhausted",
            updated_edit_config=next_edit_config,
            failed_gates=failed_gates,
            head_adjustment=head_adjustment,
        )

    error_message = getattr(validation_result, "error_message", None)
    return RetryValidationDecision(
        action="error",
        updated_edit_config=updated_edit_config,
        failed_gates=failed_gates or ("report_error",),
        error_message=str(error_message or "Retry validation failed"),
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
            action=decision.action,
            updated_edit_config=decision.updated_edit_config,
            failed_gates=decision.failed_gates,
            next_attempt=attempt,
            head_adjustment=decision.head_adjustment,
            error_message=decision.error_message,
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
            action=decision.action,
            updated_edit_config=decision.updated_edit_config,
            failed_gates=decision.failed_gates,
            notices=transition.notices,
            next_attempt=transition.next_attempt,
            head_adjustment=decision.head_adjustment,
            error_message=decision.error_message,
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
        action=decision.action,
        updated_edit_config=decision.updated_edit_config,
        failed_gates=decision.failed_gates,
        next_attempt=attempt,
        head_adjustment=decision.head_adjustment,
        error_message=decision.error_message,
    )
