"""Bounded live collection through one caller-supplied Inspect model."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
)

from .collector import (
    EXPORT_FORMAT,
    EXPORT_PROFILE,
    INSPECT_VERSION,
    CollectionOptions,
    InspectJudgeError,
    _check_options,
    _render_request,
    import_export,
    prepare_collection,
    prepare_inspect_config,
)

_HEADER = "collection.json"
_ATTEMPT_PREFIX = "attempt-"


@dataclass(frozen=True)
class RunnerOptions:
    """Execution-only limits and durable checkpoint identity."""

    checkpoint_directory: Path
    scorer_id: str
    max_elapsed_seconds: int

    def validate(self) -> None:
        if not isinstance(self.checkpoint_directory, Path):
            raise InspectJudgeError("checkpoint_directory must be a Path")
        if (
            not isinstance(self.scorer_id, str)
            or not self.scorer_id
            or len(self.scorer_id) > 128
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in self.scorer_id
            )
        ):
            raise InspectJudgeError("scorer_id must be a bounded identifier")
        if (
            type(self.max_elapsed_seconds) is not int
            or not 1 <= self.max_elapsed_seconds <= 7 * 24 * 60 * 60
        ):
            raise InspectJudgeError("max_elapsed_seconds must be between 1 and 604800")


class _EventSink:
    def __init__(self) -> None:
        self.pending: Any | None = None
        self.complete: Any | None = None

    def on_pending(self, event: Any) -> None:
        if self.pending is not None:
            raise InspectJudgeError("one judge call emitted more than one model event")
        self.pending = event

    def on_complete(self, event: Any) -> None:
        if self.complete is not None:
            raise InspectJudgeError(
                "one judge call completed more than one model event"
            )
        self.complete = event


class _Pacer:
    def __init__(self, spacing: float) -> None:
        self._spacing = spacing
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self._spacing - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


def _header(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> dict[str, Any]:
    return {
        "format": EXPORT_FORMAT,
        "profile": EXPORT_PROFILE,
        "inspect_version": INSPECT_VERSION,
        "plan_sha256": measurement_plan_digest(plan),
        "collection": asdict(options),
        "scorer_id": runner.scorer_id,
    }


def _initialize_checkpoint(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> None:
    runner.validate()
    directory = runner.checkpoint_directory
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    expected = canonical_payload(_header(plan, options, runner))
    path = directory / _HEADER
    if path.exists():
        observed = read_regular_file_bytes(
            path, label="Inspect judge checkpoint header", max_bytes=65536
        )
        if observed != expected:
            raise InspectJudgeError("checkpoint identity differs from this collection")
    else:
        write_file_no_replace(path, expected)


def _empty_export(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> dict[str, Any]:
    digest = measurement_plan_digest(plan)
    samples = []
    for binding in sorted(plan["answer_bindings"], key=lambda row: row["case_id"]):
        binding_raw = cast(dict[str, Any], binding)
        for side in ("baseline", "subject"):
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                samples.append(
                    {
                        "id": expected_trial_id(
                            digest, binding["case_id"], side, repetition
                        ),
                        "epoch": 1,
                        "metadata": {
                            "case_id": binding["case_id"],
                            "side": side,
                            "repetition": repetition,
                            "plan_sha256": digest,
                            "answer_sha256": binding_raw[f"{side}_answer_sha256"],
                            "scorer_id": runner.scorer_id,
                        },
                        "events": [],
                    }
                )
    return {
        "format": EXPORT_FORMAT,
        "profile": EXPORT_PROFILE,
        "inspect_version": INSPECT_VERSION,
        "collection": asdict(options),
        "samples": samples,
    }


def _checkpoint_export(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> dict[str, Any]:
    exported = _empty_export(plan, options, runner)
    by_id = {sample["id"]: sample for sample in exported["samples"]}
    for path in sorted(
        runner.checkpoint_directory.iterdir(), key=lambda item: item.name
    ):
        if path.name == _HEADER:
            continue
        if not path.name.startswith(_ATTEMPT_PREFIX) or path.suffix != ".json":
            raise InspectJudgeError("checkpoint directory contains an unknown entry")
        payload = read_regular_file_bytes(
            path, label="Inspect judge checkpoint attempt", max_bytes=2 * 1024 * 1024
        )
        record = parse_json_bytes(payload, label="Inspect judge checkpoint attempt")
        if not isinstance(record, dict) or set(record) != {
            "trial_id",
            "attempt",
            "event",
        }:
            raise InspectJudgeError("checkpoint attempt has an unsupported shape")
        trial_id = record["trial_id"]
        attempt = record["attempt"]
        if trial_id not in by_id or type(attempt) is not int:
            raise InspectJudgeError("checkpoint attempt has an unknown identity")
        events = by_id[trial_id]["events"]
        if attempt != len(events) + 1:
            raise InspectJudgeError(
                "checkpoint attempts are missing, duplicated, or reordered"
            )
        events.append(record["event"])
    return exported


def _request_id(response: Any, output: Any) -> str | None:
    for value in (
        response.get("id") if isinstance(response, dict) else None,
        response.get("request_id") if isinstance(response, dict) else None,
        getattr(output, "metadata", None),
    ):
        if isinstance(value, str) and 0 < len(value) <= 256:
            return value
        if isinstance(value, dict):
            candidate = value.get("request_id")
            if isinstance(candidate, str) and 0 < len(candidate) <= 256:
                return candidate
    return None


def _project_event(
    event: Any,
    *,
    expected_request: dict[str, Any],
    options: CollectionOptions,
    failure_status: str | None,
    failure_message: str | None,
) -> dict[str, Any]:
    def public_message(message: Any) -> dict[str, str]:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise InspectJudgeError("Inspect model input must use plain text messages")
        return {"role": role, "content": content}

    event_input = getattr(event, "input", None)
    if not isinstance(event_input, list):
        raise InspectJudgeError("Inspect model event is missing its input")
    projected_input = [public_message(message) for message in event_input]
    if projected_input != expected_request["messages"]:
        raise InspectJudgeError("Inspect model input differs from the approved request")
    event_model = getattr(event, "model", None)
    if event_model != options.grader:
        raise InspectJudgeError("Inspect model event differs from the approved grader")
    if getattr(event, "tools", None) != []:
        raise InspectJudgeError("Inspect model event used tools")
    tool_choice = getattr(event, "tool_choice", None)
    if tool_choice != "none":
        raise InspectJudgeError("Inspect model event used an unsupported tool choice")
    retries = getattr(event, "retries", 0)
    if retries not in (None, 0):
        raise InspectJudgeError("Inspect model event contains SDK retries")
    if getattr(event, "cache", None) is not None:
        raise InspectJudgeError("Inspect model event used a cache")
    output = getattr(event, "output", None)
    call = getattr(event, "call", None)
    call_request = getattr(call, "request", None)
    call_response = getattr(call, "response", None)
    call_error = getattr(call, "error", None)
    if not isinstance(call_request, dict):
        call_request = expected_request
    if call_response is not None and not isinstance(call_response, dict):
        raise InspectJudgeError("Inspect provider response must be a JSON object")
    usage = getattr(output, "usage", None)
    projected_usage = None
    if usage is not None:
        input_tokens = int(getattr(usage, "input_tokens", 0))
        input_tokens += int(getattr(usage, "input_tokens_cache_read", 0) or 0)
        input_tokens += int(getattr(usage, "input_tokens_cache_write", 0) or 0)
        projected_usage = {
            "input_tokens": input_tokens,
            "output_tokens": int(getattr(usage, "output_tokens", 0)),
        }
        total_cost = getattr(usage, "total_cost", None)
        if (
            total_cost is not None
            and round(float(total_cost) * 1_000_000) > options.cost_microusd_per_call
        ):
            raise InspectJudgeError(
                "provider cost exceeded the reserved per-call amount"
            )
    resolved_model = getattr(output, "model", None)
    completion = getattr(output, "completion", "")
    stop_reason = None
    choices = getattr(output, "choices", None)
    if choices:
        stop_reason = getattr(choices[0], "stop_reason", None)
    event_error = getattr(event, "error", None)
    if failure_status is None and (
        event_error is not None or getattr(output, "error", None)
    ):
        failure_status = (
            "refusal" if getattr(output, "error", None) else "transport_error"
        )
        failure_message = str(getattr(output, "error", None) or event_error)
    error = None
    if failure_status is not None:
        error = {
            "status": failure_status,
            "code": "inspect-call-failed",
            "message": (failure_message or "Inspect judge call did not complete")[
                :4096
            ],
        }
        resolved_model = None
        projected_usage = None
        stop_reason = None
    config = expected_request["config"]
    event_config = getattr(event, "config", None)
    projected_config = {
        "temperature": getattr(event_config, "temperature", None),
        "top_p": getattr(event_config, "top_p", None),
        "max_tokens": getattr(event_config, "max_tokens", None),
        "seed": getattr(event_config, "seed", None),
        "max_retries": getattr(event_config, "max_retries", None),
    }
    expected_config = {
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "max_tokens": config["max_output_tokens"],
        "seed": config["seed"],
        "max_retries": 0,
    }
    if projected_config != expected_config:
        raise InspectJudgeError("Inspect model event has changed generation settings")
    return {
        "event": "model",
        "uuid": str(getattr(event, "uuid", None) or "missing-event-id"),
        "role": "grader",
        "model": event_model,
        "input": projected_input,
        "tools": [],
        "tool_choice": "none",
        "config": projected_config,
        "retries": 0,
        "cache": getattr(event, "cache", None),
        "call": {
            "request": call_request,
            "response": call_response,
            "error": True if error is not None else call_error,
        },
        "output": {
            "model": resolved_model,
            "request_id": _request_id(call_response, output),
            "finish_reason": stop_reason,
            "usage": projected_usage,
            "completion": completion
            if isinstance(completion, str)
            else str(completion),
        },
        "error": error,
    }


def _frozen_rows(
    baseline_run: dict[str, Any], subject_run: dict[str, Any]
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for side, run in (("baseline", baseline_run), ("subject", subject_run)):
        for record in run.get("records", []):
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise InspectJudgeError("frozen run contains an invalid record")
            row = rows.setdefault(record["id"], {})
            if "input" in row and row["input"] != record.get("input"):
                raise InspectJudgeError("frozen paired inputs differ")
            if not isinstance(record.get("input"), str) or not isinstance(
                record.get("output"), str
            ):
                raise InspectJudgeError("frozen runs require text inputs and outputs")
            row["input"] = record["input"]
            row[side] = record["output"]
    return rows


async def _call_one(
    model: Any,
    *,
    request: dict[str, Any],
    config: Any,
    options: CollectionOptions,
    pacer: _Pacer,
) -> dict[str, Any]:
    await pacer.wait()
    model_module = importlib.import_module("inspect_ai.model")
    sink_module = importlib.import_module("inspect_ai.model._model")
    sink = _EventSink()
    messages = []
    for message in request["messages"]:
        cls = (
            model_module.ChatMessageSystem
            if message["role"] == "system"
            else model_module.ChatMessageUser
        )
        messages.append(cls(content=message["content"]))
    failure_status = None
    failure_message = None
    try:
        with sink_module.use_model_event_sink(sink):
            async with asyncio.timeout(options.request_timeout_seconds):
                await model.generate(
                    input=messages,
                    tools=[],
                    tool_choice="none",
                    config=config,
                    cache=False,
                )
    except TimeoutError as exc:
        failure_status = "timeout_ambiguous"
        failure_message = str(exc) or "judge request deadline elapsed"
    except asyncio.CancelledError:
        failure_status = "cancelled"
        failure_message = "judge collection was cancelled"
    except Exception as exc:
        failure_status = "transport_error"
        failure_message = f"{type(exc).__name__}: {exc}"
    event = sink.complete or sink.pending
    if event is None:
        raise InspectJudgeError("Inspect did not expose a model event for the call")
    return _project_event(
        event,
        expected_request=request,
        options=options,
        failure_status=failure_status,
        failure_message=failure_message,
    )


async def collect(
    *,
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    model: Any,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> JudgeMeasurements:
    """Collect or resume fixed-answer judgments with durable attempt shards.

    The caller creates the Inspect model and therefore owns provider credentials.
    This function never accepts or writes credentials, provider URLs, or headers.
    """
    _check_options(plan, options)
    runner.validate()
    if importlib.metadata.version("inspect-ai") != INSPECT_VERSION:
        raise InspectJudgeError("unsupported installed Inspect version")
    if str(model) != options.grader:
        raise InspectJudgeError(
            "Inspect model identity differs from the approved grader"
        )
    _initialize_checkpoint(plan, options, runner)
    config = prepare_inspect_config(plan, options)
    rows = _frozen_rows(baseline_run, subject_run)
    pacer = _Pacer(60 / options.requests_per_minute)
    started = time.monotonic()

    while True:
        exported = _checkpoint_export(plan, options, runner)
        checkpoint = import_export(
            canonical_payload(exported),
            plan=plan,
            options=options,
            baseline_run=baseline_run,
            subject_run=subject_run,
        )
        batch = prepare_collection(
            plan,
            options,
            checkpoint=checkpoint,
            baseline_run=baseline_run,
            subject_run=subject_run,
        )
        if not batch["next_batch"] or batch["budget_exhausted"]:
            return checkpoint
        remaining = runner.max_elapsed_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return checkpoint

        async def run_item(
            item: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            row = rows[item["case_id"]]
            request = _render_request(
                plan, input_text=row["input"], answer=row[item["side"]]
            )
            event = await _call_one(
                model,
                request=request,
                config=config,
                options=options,
                pacer=pacer,
            )
            return item, event

        tasks: list[asyncio.Task[tuple[dict[str, Any], dict[str, Any]]]] = []
        for item in batch["next_batch"]:
            tasks.append(asyncio.create_task(run_item(item)))
        try:
            async with asyncio.timeout(remaining):
                for completed_task in asyncio.as_completed(tasks):
                    item, event = await completed_task
                    record = {
                        "trial_id": item["trial_id"],
                        "attempt": item["attempt"],
                        "event": event,
                    }
                    candidate = _checkpoint_export(plan, options, runner)
                    for sample in candidate["samples"]:
                        if sample["id"] == item["trial_id"]:
                            sample["events"].append(event)
                            break
                    import_export(
                        canonical_payload(candidate),
                        plan=plan,
                        options=options,
                        baseline_run=baseline_run,
                        subject_run=subject_run,
                    )
                    path = runner.checkpoint_directory / (
                        f"{_ATTEMPT_PREFIX}{item['trial_id']}-{item['attempt']:02d}.json"
                    )
                    write_file_no_replace(path, canonical_payload(record))
        except TimeoutError:
            for pending_task in tasks:
                pending_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return import_export(
                canonical_payload(_checkpoint_export(plan, options, runner)),
                plan=plan,
                options=options,
                baseline_run=baseline_run,
                subject_run=subject_run,
            )
