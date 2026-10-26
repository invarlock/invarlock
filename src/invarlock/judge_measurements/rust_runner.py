"""Explicit native collection coordination with the audited Inspect SDK boundary.

Rust owns slots, requests, admission budgets, pacing, checkpoint transitions and
retained-event validation. Python constructs the optional SDK's config objects,
invokes its existing audited call projection and transfers completed callbacks.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from invarlock.judge_measurements import runner as sdk
from invarlock.judge_measurements.collector import INSPECT_VERSION, InspectJudgeError
from invarlock.judge_measurements.contracts import canonical_payload


class _NativePacer:
    def __init__(self, coordinator: Any, trial_id: str) -> None:
        self.coordinator = coordinator
        self.trial_id = trial_id

    async def wait(self) -> None:
        # The native pacing mutex is separate from checkpoint state, so a queued
        # start cannot prevent retention of another completed provider callback.
        await asyncio.to_thread(self.coordinator.wait_start, self.trial_id)


async def collect(
    *,
    plan: Any,
    options: Any,
    runner: Any,
    model: Any,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    on_stop: Any = None,
    status: dict[str, Any] | None = None,
) -> Any:
    """Use native state transitions; never redispatch a resumed admission."""
    if status is not None:
        status.clear()
    if not isinstance(runner.checkpoint_directory, Path):
        raise InspectJudgeError("checkpoint_directory must be a Path")
    sdk._require_qualified_live_provider_model(options.grader)
    if importlib.metadata.version("inspect-ai") != INSPECT_VERSION:
        raise InspectJudgeError("unsupported installed Inspect version")
    if str(model) != options.grader:
        raise InspectJudgeError(
            "Inspect model identity differs from the approved grader"
        )
    sdk._require_provider_retries_disabled(model)
    sdk._require_clean_model_configuration(model)
    try:
        native = importlib.import_module("invarlock_native")
    except ImportError as exc:
        raise InspectJudgeError(
            "the explicit Rust execution backend requires invarlock_native"
        ) from exc
    coordinator = None
    try:
        coordinator = native.CollectionCoordinator(
            str(runner.checkpoint_directory.absolute()),
            canonical_payload(plan),
            canonical_payload(asdict(options)),
            canonical_payload(baseline_run),
            canonical_payload(subject_run),
            runner.scorer_id,
            runner.invocation_timeout_seconds,
            stop_after_batches=runner.stop_after_batches,
        )
        module = importlib.import_module("inspect_ai.model")
        config = module.GenerateConfig(**json.loads(coordinator.sdk_config_json()))

        async def invoke(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            event = await sdk._call_one(
                model,
                request=call["request"],
                config=config,
                options=options,
                pacer=_NativePacer(coordinator, call["trial_id"]),
                check_directory=coordinator.verify,
            )
            return call["trial_id"], event

        def retain(outcome: tuple[str, dict[str, Any]]) -> None:
            trial_id, event = outcome
            coordinator.complete(trial_id, canonical_payload(event))

        async def drain(tasks: list[asyncio.Task[Any]]) -> None:
            coordinator.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            for outcome in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(outcome, tuple):
                    # Rust treats observing the same completed callback twice
                    # as idempotent retention and rejects a changed duplicate.
                    retain(outcome)

        while calls := json.loads(coordinator.begin_batch_json()):
            tasks = [asyncio.create_task(invoke(call)) for call in calls]
            try:
                async with asyncio.timeout(coordinator.remaining_seconds()):
                    for task in asyncio.as_completed(tasks):
                        retain(await task)
            except TimeoutError:
                await drain(tasks)
                coordinator.stop("deadline")
                break
            except BaseException as exc:
                await drain(tasks)
                coordinator.stop(
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                raise
        final_status = json.loads(coordinator.status_json())
        reason = final_status["stop_reason"]
        if status is not None:
            status["stop_reason"] = reason
            if reason == "capacity_exhausted":
                status["capacity"] = final_status["capacity"]
        if on_stop is not None:
            on_stop(reason)
        return json.loads(coordinator.measurements_json())
    except ValueError as exc:
        raise InspectJudgeError(str(exc)) from exc
    finally:
        if coordinator is not None:
            coordinator.close()
