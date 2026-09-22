"""Installed, explicitly configured collection against the qualified endpoint."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.security import network_policy_allows, temporarily_allow_network

from .collector import (
    INSPECT_VERSION,
    CollectionOptions,
    InspectJudgeError,
    _check_options,
)
from .runner import RunnerOptions, collect

_BASE_SDK_VERSIONS = {
    "inspect-ai": INSPECT_VERSION,
    "httpx": "0.28.1",
    "httpx2": "2.12.0",
}


@dataclass(frozen=True)
class _ProviderConfig:
    credential_names: tuple[str, ...]
    distribution: str
    version: str
    module: str
    base_url: str | None
    endpoint_variables: tuple[str, ...]
    forbidden_variables: tuple[str, ...] = ()


_PROVIDERS = {
    "openai": _ProviderConfig(
        credential_names=("OPENAI_API_KEY",),
        distribution="openai",
        version="3.13.0",
        module="openai",
        base_url="https://api.openai.com/v1",
        endpoint_variables=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        forbidden_variables=(
            "OPENAI_SAFETY_IDENTIFIER",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
        ),
    ),
    "anthropic": _ProviderConfig(
        credential_names=("ANTHROPIC_API_KEY",),
        distribution="anthropic",
        version="1.6.0",
        module="anthropic",
        base_url="https://api.anthropic.com",
        endpoint_variables=("ANTHROPIC_BASE_URL",),
        forbidden_variables=("ANTHROPIC_AUTH_TOKEN",),
    ),
    "google": _ProviderConfig(
        credential_names=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        distribution="google-genai",
        version="2.24.0",
        module="google.genai",
        base_url="https://generativelanguage.googleapis.com",
        endpoint_variables=(
            "GOOGLE_BASE_URL",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_VERTEX_BASE_URL",
            "VERTEX_BASE_URL",
        ),
        forbidden_variables=(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_USE_ADC",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "VERTEX_API_KEY",
            "GOOGLE_GENAI_CLIENT_MODE",
            "GOOGLE_GENAI_REPLAYS_DIRECTORY",
            "GOOGLE_GENAI_REPLAY_ID",
        ),
    ),
    "openrouter": _ProviderConfig(
        credential_names=("OPENROUTER_API_KEY",),
        distribution="openai",
        version="3.13.0",
        module="openai",
        base_url="https://openrouter.ai/api/v1",
        endpoint_variables=("OPENROUTER_BASE_URL",),
        forbidden_variables=("OPENAI_ORG_ID", "OPENAI_PROJECT_ID"),
    ),
}
_SDK_VERSIONS = {
    **_BASE_SDK_VERSIONS,
    **{config.distribution: config.version for config in _PROVIDERS.values()},
}


def _provider_config(grader: str) -> tuple[str, _ProviderConfig]:
    provider, separator, model_name = grader.partition("/")
    if separator != "/" or provider not in _PROVIDERS or not model_name:
        supported = ", ".join(sorted(_PROVIDERS))
        raise InspectJudgeError(
            f"configured collection requires a supported provider ({supported})"
        )
    if provider != "openrouter" and "/" in model_name:
        raise InspectJudgeError(
            f"configured {provider} collection requires the direct API model name"
        )
    return provider, _PROVIDERS[provider]


def _provider_sdk_versions(config: _ProviderConfig) -> dict[str, str]:
    return {
        **_BASE_SDK_VERSIONS,
        config.distribution: config.version,
    }


def _credential(
    config: _ProviderConfig, environment: Mapping[str, str]
) -> tuple[str, str | None]:
    for name in config.credential_names:
        value = environment.get(name)
        if isinstance(value, str) and value.strip():
            return name, value
    return config.credential_names[0], None


def validate_collection_environment(
    options: CollectionOptions,
    environment: Mapping[str, str] | None = None,
    require_credentials: bool = True,
) -> dict[str, Any]:
    """Check live dependencies and credentials without constructing a model.

    An explicit environment supplies credentials without changing process state.
    Endpoint and alternate-auth overrides in either it or the process
    environment are unsupported. Returned metadata contains no credential
    values.
    """
    options.validate()
    if options.inspect_version != INSPECT_VERSION:
        raise InspectJudgeError("live collection requires current Inspect version")
    provider, config = _provider_config(options.grader)
    if type(require_credentials) is not bool:
        raise InspectJudgeError("require_credentials must be a boolean")
    current = os.environ if environment is None else environment
    for source in (current, os.environ):
        if any(name in source for name in config.endpoint_variables):
            raise InspectJudgeError(
                f"{provider} endpoint environment overrides are unsupported"
            )
        if any(name in source for name in config.forbidden_variables):
            raise InspectJudgeError(
                f"{provider} alternate authentication or request controls are unsupported"
            )
    credential_name, key = _credential(config, current)
    credential_available = key is not None
    if require_credentials and not credential_available:
        raise InspectJudgeError(
            f"{credential_name} must be set in the collector environment"
        )
    sdk_versions = _provider_sdk_versions(config)
    for distribution, expected in sdk_versions.items():
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
        for module in ("inspect_ai.model", config.module, "httpx", "httpx2"):
            importlib.import_module(module)
    except Exception:
        raise InspectJudgeError(
            "the pinned Inspect collection dependencies could not be imported"
        ) from None
    return {
        "provider": provider,
        "base_url": config.base_url or "provider-default",
        "service_tier": "default" if provider == "openai" else None,
        "credential_variable": credential_name,
        "sdk_versions": sdk_versions,
        "credential_available": credential_available,
    }


async def _close_model(model: Any) -> None:
    # Finish releasing pooled connections even if another cancellation arrives
    # during cleanup. Preserve cancellation after the close task has settled.
    close = getattr(getattr(model, "api", None), "aclose", None)
    if not callable(close):
        raise InspectJudgeError("Inspect provider does not expose async cleanup")
    task = asyncio.create_task(close())
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


def _bound_google_client(model: Any) -> None:
    """Bound the pinned provider's SDK and malformed-function retry loops."""
    api = model.api
    original_client = api.model_client
    types = importlib.import_module("google.genai.types")

    def model_client(http_options: Any = None) -> Any:
        if http_options is None:
            http_options = api._http_options()
        http_options.retry_options = types.HttpRetryOptions(attempts=1)
        client = original_client(http_options)
        generate = client.aio.models.generate_content
        called = False

        async def generate_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal called
            if called:
                raise InspectJudgeError(
                    "Google provider attempted an unreserved internal retry"
                )
            called = True
            kwargs["config"] = kwargs["config"].model_copy(
                update={
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                }
            )
            try:
                return await generate(*args, **kwargs)
            finally:
                client.close()

        client.aio.models.generate_content = generate_once
        return client

    api.model_client = model_client
    api.streaming = False
    api._invarlock_single_attempt = True


def _configure_anthropic_sampling(model: Any, plan: JudgeMeasurementPlan) -> None:
    """Use temperature alone; recent Claude APIs reject both sampling knobs."""
    api = model.api
    original_generate = api.generate
    original_create = api.client.messages.create
    calls: ContextVar[int] = ContextVar("anthropic_admitted_requests", default=0)
    config = importlib.import_module("inspect_ai.model").GenerateConfig(
        reasoning_effort=plan["judge"]["config"]["reasoning_effort"]
    )
    ignores_temperature = api.is_claude_4_7_or_later() or api.is_using_thinking(config)
    if ignores_temperature and Decimal(plan["judge"]["config"]["temperature"]) != 1:
        raise InspectJudgeError("this Anthropic model requires approved temperature=1")

    async def create_once(*args: Any, **kwargs: Any) -> Any:
        if calls.get() != 0:
            raise InspectJudgeError(
                "Anthropic provider attempted an unreserved continuation"
            )
        calls.set(1)
        return await original_create(*args, **kwargs)

    async def generate(input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:
        token = calls.set(0)
        try:
            return await original_generate(
                input, tools, tool_choice, config.model_copy(update={"top_p": None})
            )
        finally:
            calls.reset(token)

    api.client.messages.create = create_once
    api.streaming = False
    api._invarlock_single_attempt = True
    api.generate = generate


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
    """Collect frozen judgments with pinned SDKs and an environment-only key.

    ``runner.stop_after_batches`` requests a graceful invocation stop after
    durable results, without changing the plan or checkpoint budget identity.
    """
    allow_judge_network = os.environ.get(
        "INVARLOCK_ALLOW_JUDGE_NETWORK", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not allow_judge_network and not network_policy_allows():
        raise InspectJudgeError(
            "live judge collection requires an allowed network policy; "
            "invoke only the collection command with INVARLOCK_ALLOW_JUDGE_NETWORK=1. "
            "Preflight and retained-measurement import remain available offline"
        )
    with temporarily_allow_network() if allow_judge_network else nullcontext():
        return await _collect_configured(
            plan,
            options,
            runner,
            baseline_run,
            subject_run,
            environment,
            on_stop=on_stop,
        )


async def _collect_configured(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    environment: Mapping[str, str] | None,
    *,
    on_stop: Callable[[str], None] | None,
) -> JudgeMeasurements:
    """Keep model initialization, collection and cleanup inside the judge scope."""
    _check_options(plan, options)
    runner.validate()
    if plan["schedule"]["max_attempts"] != 1:
        raise InspectJudgeError(
            "live Inspect collection currently requires max_attempts=1"
        )
    current = dict(os.environ if environment is None else environment)
    validate_collection_environment(options, current)
    provider, config = _provider_config(options.grader)
    _, key = _credential(config, current)
    assert key is not None  # validated above
    module = importlib.import_module("inspect_ai.model")
    model_args: dict[str, Any] = {
        "config": module.GenerateConfig(),
        "api_key": key,
        "memoize": False,
    }
    if config.base_url is not None:
        model_args["base_url"] = config.base_url
    if provider in {"openai", "openrouter"}:
        model_args["responses_api"] = False
    if provider == "openai":
        model_args["service_tier"] = "default"
    if provider != "google":
        model_args["max_retries"] = 0
    try:
        model = module.get_model(options.grader, **model_args)
    except Exception:
        raise InspectJudgeError(
            "could not initialize the configured Inspect grader"
        ) from None
    succeeded = False
    try:
        api = getattr(model, "api", None)
        if provider == "google":
            _bound_google_client(model)
        elif provider == "anthropic":
            _configure_anthropic_sampling(model, plan)
        client = getattr(api, "client", None)
        observed_base_url = getattr(client, "base_url", None)
        if observed_base_url is None:
            observed_base_url = getattr(api, "base_url", None)
        if config.base_url is not None and str(observed_base_url).rstrip("/") != str(
            config.base_url
        ).rstrip("/"):
            raise InspectJudgeError(
                f"Inspect provider must use the official {provider} endpoint"
            )
        if provider == "openai" and getattr(api, "service_tier", None) != "default":
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
        try:
            await _close_model(model)
        except Exception:
            if succeeded:
                raise InspectJudgeError(
                    "could not close the Inspect provider client"
                ) from None
