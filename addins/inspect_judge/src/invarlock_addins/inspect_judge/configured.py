"""Installed, explicitly configured collection against the qualified endpoint."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
from collections.abc import Callable, Mapping
from typing import Any

from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements

from .collector import (
    INSPECT_VERSION,
    CollectionOptions,
    InspectJudgeError,
    _check_options,
)
from .runner import RunnerOptions, collect

_BASE_URL = "https://api.openai.com/v1"
_SDK_VERSIONS = {
    "inspect-ai": INSPECT_VERSION,
    "openai": "3.13.0",
    "httpx": "0.28.1",
}


def validate_collection_environment(
    options: CollectionOptions,
    environment: Mapping[str, str] | None = None,
    require_credentials: bool = True,
) -> dict[str, Any]:
    """Check live dependencies and credentials without constructing a model.

    An explicit environment supplies credentials without changing process state.
    Endpoint overrides in either it or the process environment are unsupported.
    Returned metadata contains no credential values.
    """
    options.validate()
    if options.inspect_version != INSPECT_VERSION:
        raise InspectJudgeError("live collection requires current Inspect version")
    if not options.grader.startswith("openai/"):
        raise InspectJudgeError("configured collection requires an OpenAI grader")
    if type(require_credentials) is not bool:
        raise InspectJudgeError("require_credentials must be a boolean")
    current = os.environ if environment is None else environment
    for source in (current, os.environ):
        if any(name in source for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE")):
            raise InspectJudgeError(
                "OpenAI endpoint environment overrides are unsupported"
            )
        if "OPENAI_SAFETY_IDENTIFIER" in source:
            raise InspectJudgeError(
                "OpenAI safety identifier environment overrides are unsupported"
            )
    key = current.get("OPENAI_API_KEY")
    credential_available = isinstance(key, str) and bool(key.strip())
    if require_credentials and not credential_available:
        raise InspectJudgeError(
            "OPENAI_API_KEY must be set in the collector environment"
        )
    for distribution, expected in _SDK_VERSIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise InspectJudgeError(
                'install "invarlock[judge]" for live collection'
            ) from None
        if version != expected:
            raise InspectJudgeError(
                f"live collection requires {distribution}=={expected}"
            )
    try:
        for module in ("inspect_ai.model", "openai", "httpx"):
            importlib.import_module(module)
    except Exception:
        raise InspectJudgeError(
            "the pinned Inspect collection dependencies could not be imported"
        ) from None
    return {
        "provider": "openai",
        "base_url": _BASE_URL,
        "service_tier": "default",
        "sdk_versions": dict(_SDK_VERSIONS),
        "credential_available": credential_available,
    }


async def _close_client(client: Any) -> None:
    # Finish releasing pooled connections even if another cancellation arrives
    # during cleanup. Preserve cancellation after the close task has settled.
    task = asyncio.create_task(client.close())
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    if cancellation is not None:
        # Retrieve a cleanup failure as well, so it cannot become an unhandled
        # task exception while the caller's cancellation takes precedence.
        if not task.cancelled():
            task.exception()
        raise cancellation
    task.result()


async def collect_configured(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    environment: Mapping[str, str] | None = None,
    *,
    on_stop: Callable[[str], None] | None = None,
) -> JudgeMeasurements:
    """Collect frozen judgments with the pinned SDK and environment-only key."""
    _check_options(plan, options)
    runner.validate()
    if plan["schedule"]["max_attempts"] != 1:
        raise InspectJudgeError(
            "live Inspect collection currently requires max_attempts=1"
        )
    current = dict(os.environ if environment is None else environment)
    validate_collection_environment(options, current)
    module = importlib.import_module("inspect_ai.model")
    try:
        model = module.get_model(
            options.grader,
            config=module.GenerateConfig(),
            base_url=_BASE_URL,
            api_key=current["OPENAI_API_KEY"],
            responses_api=False,
            service_tier="default",
            max_retries=0,
            memoize=False,
        )
    except Exception:
        raise InspectJudgeError(
            "could not initialize the configured Inspect grader"
        ) from None
    client = getattr(getattr(model, "api", None), "client", None)
    succeeded = False
    try:
        if str(getattr(client, "base_url", "")) not in {_BASE_URL, _BASE_URL + "/"}:
            raise InspectJudgeError(
                "Inspect provider must use the official OpenAI endpoint"
            )
        if getattr(model.api, "service_tier", None) != "default":
            raise InspectJudgeError(
                "Inspect provider must use the standard service tier"
            )
        result = await collect(
            plan=plan,
            options=options,
            runner=runner,
            model=model,
            baseline_run=baseline_run,
            subject_run=subject_run,
            on_stop=on_stop,
        )
        succeeded = True
        return result
    finally:
        if client is not None:
            try:
                await _close_client(client)
            except Exception:
                if succeeded:
                    raise InspectJudgeError(
                        "could not close the Inspect provider client"
                    ) from None
