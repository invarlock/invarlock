"""Installed commands for runtime-provider side production and paired replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import typer

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)

RUNTIME_BEHAVIOR_RUN_SIDE_CLI_FORMAT = "runtime-behavior-run-side-cli-v1"
RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT = "runtime-behavior-verify-pair-cli-v1"
RUNTIME_BEHAVIOR_BUILD_POLICY_CLI_FORMAT = "runtime-behavior-build-policy-cli-v1"
RUNTIME_BEHAVIOR_BUILD_SCHEDULE_CLI_FORMAT = "runtime-behavior-build-schedule-cli-v1"
RUNTIME_BEHAVIOR_PREPARE_BINDING_CLI_FORMAT = "runtime-behavior-prepare-binding-cli-v1"
MAX_RUNTIME_PROVIDER_SETTINGS_BYTES = 1024 * 1024
MAX_RUNTIME_BEHAVIOR_INPUT_BYTES = 16 * 1024 * 1024
_BEHAVIORAL_BINDING_FIELDS = frozenset(
    {
        "provider_name",
        "artifact_format",
        "artifact_identity_sha256",
        "outer_image_digest",
        "execution_settings_sha256",
    }
)


class SideRole(StrEnum):
    BASELINE = "baseline"
    SUBJECT = "subject"


runtime_behavior_app = typer.Typer(
    help=(
        "Build a schedule and directed authorization, prepare native provider "
        "bindings, produce native provider sides, then verify the directed pair."
    ),
    no_args_is_help=True,
)


def _provider(name: str) -> Any:
    from invarlock.core.registry import get_registry

    return get_registry().get_runtime_provider(name)


def _run_side_api(**kwargs: object) -> Any:
    from invarlock.runtime_behavior import run_side

    return cast(Any, run_side)(**kwargs)


def _verify_pair_api(**kwargs: object) -> Any:
    from invarlock.runtime_behavior import verify_pair

    return cast(Any, verify_pair)(**kwargs)


def _load_scalar_object(
    path: Path, *, label: str
) -> dict[str, str | int | float | bool | None]:
    try:
        payload = read_regular_file_bytes(
            path,
            label=label,
            max_bytes=MAX_RUNTIME_PROVIDER_SETTINGS_BYTES,
        )
        decoded = parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    settings: dict[str, str | int | float | bool | None] = {}
    for key, value in decoded.items():
        if (
            not isinstance(key, str)
            or value is not None
            and not isinstance(value, str | int | float | bool)
        ):
            raise ValueError(f"{label} must contain JSON scalar values")
        settings[key] = value
    return settings


def _load_json_value(path: Path, *, label: str, max_bytes: int) -> object:
    try:
        payload = read_regular_file_bytes(path, label=label, max_bytes=max_bytes)
        return parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        raise ValueError(str(exc)) from exc


def _load_settings(path: Path) -> dict[str, str | int | float | bool | None]:
    return _load_scalar_object(path, label="runtime provider settings")


def _load_behavioral_binding(path: Path, *, role: str) -> dict[str, object]:
    binding = _load_scalar_object(path, label=f"{role} behavioral binding")
    if set(binding) != _BEHAVIORAL_BINDING_FIELDS:
        missing = sorted(_BEHAVIORAL_BINDING_FIELDS - set(binding))
        unknown = sorted(set(binding) - _BEHAVIORAL_BINDING_FIELDS)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"{role} behavioral binding must contain exactly the five directed "
            f"fields ({'; '.join(details)})"
        )
    return cast(dict[str, object], binding)


def _required_path(value: str | None, *, option: str) -> Path:
    if value is None or not value.strip():
        raise ValueError(f"{option} is required for the selected provider")
    return Path(value)


def _native_bindings(
    *,
    provider_name: str,
    artifact: str,
    backend_executable: str,
    backend_source: str | None,
    tokenizer_contract: str | None,
) -> object:
    if provider_name == "llama_cpp":
        from invarlock.runtime_providers.llama_cpp_session import (
            LlamaCppRuntimeBindings,
        )

        return LlamaCppRuntimeBindings(
            gguf_path=Path(artifact),
            executable_path=Path(backend_executable),
            source_archive_path=_required_path(
                backend_source, option="--backend-source"
            ),
        )
    if provider_name == "tensorrt_llm":
        from invarlock.runtime_providers.tensorrt_llm_session import (
            TensorRTLLMRuntimeBindings,
        )

        return TensorRTLLMRuntimeBindings(
            engine_bundle_path=Path(artifact),
            tokenizer_contract_path=_required_path(
                tokenizer_contract, option="--tokenizer-contract"
            ),
            runner_executable_path=Path(backend_executable),
        )
    if provider_name == "hf_transformers":
        raise ValueError(
            "the standalone command cannot construct an in-process Hugging Face "
            "model, adapter, and scorer; use the Python run_side API with prebound "
            "objects"
        )
    raise ValueError(
        f"the installed command has no ephemeral binding adapter for {provider_name!r}"
    )


def _provider_device_kind(provider_name: str) -> str:
    if provider_name == "llama_cpp":
        return "cpu"
    if provider_name == "tensorrt_llm":
        return "cuda"
    raise ValueError(
        f"the installed command cannot infer a device kind for {provider_name!r}"
    )


def _native_provider_inputs(
    *,
    provider_name: str,
    model_id: str,
    settings_path: str,
    artifact: str,
    backend_executable: str,
    backend_source: str | None,
    tokenizer_contract: str | None,
    container_image_digest: str,
) -> tuple[Any, Any, Any, Any]:
    from invarlock.core.runtime_provider import (
        ModelRuntimeSpec,
        RuntimeExecutionContext,
        artifact_identity_sha256,
    )

    provider = _provider(provider_name)
    spec = ModelRuntimeSpec(
        provider_name=provider_name,
        model_id=model_id,
        settings=_load_settings(Path(settings_path)),
    )
    bindings = _native_bindings(
        provider_name=provider_name,
        artifact=artifact,
        backend_executable=backend_executable,
        backend_source=backend_source,
        tokenizer_contract=tokenizer_contract,
    )
    provider.validate_config(spec)
    identity = provider.identify_artifact(spec)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=container_image_digest,
        device_kind=_provider_device_kind(provider_name),
        artifact_identity_sha256=artifact_identity_sha256(identity),
        native_model=bindings,
    )
    return provider, spec, context, identity


def _emit(payload: dict[str, object], *, json_out: bool, success: str) -> None:
    if json_out:
        typer.echo(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        typer.echo(success)


def _fail(
    *,
    format_version: str,
    json_out: bool,
    error: Exception,
) -> None:
    message = " ".join(str(error).split()) or "runtime behavior command failed"
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "errors": [message],
                    "format_version": format_version,
                    "ok": False,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        typer.echo(f"Runtime behavior command failed: {message}", err=True)
    raise typer.Exit(2)


@runtime_behavior_app.command(
    "build-schedule",
    help="Build a canonical schedule from closed records and dataset identity JSON.",
)
def build_schedule_command(
    records: str = typer.Option(..., "--records"),
    dataset_identity: str = typer.Option(..., "--dataset-identity"),
    out: str = typer.Option(..., "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Derive record input digests and write one no-clobber schedule."""

    try:
        from invarlock.core.runtime_provider import (
            build_runtime_behavioral_schedule_from_material,
            canonical_runtime_behavioral_schedule_json,
        )
        from invarlock.runtime_behavior.io import atomic_write_new

        raw_records = _load_json_value(
            Path(records),
            label="runtime behavioral records",
            max_bytes=MAX_RUNTIME_BEHAVIOR_INPUT_BYTES,
        )
        raw_dataset_identity = _load_json_value(
            Path(dataset_identity),
            label="runtime behavioral dataset identity",
            max_bytes=MAX_RUNTIME_PROVIDER_SETTINGS_BYTES,
        )
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ):
            raise ValueError("runtime behavioral records must be a JSON array")
        if not isinstance(raw_dataset_identity, Mapping):
            raise ValueError(
                "runtime behavioral dataset identity must be a JSON object"
            )
        schedule = build_runtime_behavioral_schedule_from_material(
            dataset_identity=raw_dataset_identity,
            records=raw_records,
        )
        output_path = Path(out)
        atomic_write_new(
            output_path,
            canonical_runtime_behavioral_schedule_json(schedule),
        )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail(
            format_version=RUNTIME_BEHAVIOR_BUILD_SCHEDULE_CLI_FORMAT,
            json_out=json_out,
            error=exc,
        )
        return

    _emit(
        {
            "format_version": RUNTIME_BEHAVIOR_BUILD_SCHEDULE_CLI_FORMAT,
            "ok": True,
            "output": str(output_path),
            "record_count": len(schedule.records),
            "schedule_sha256": schedule.schedule_sha256,
        },
        json_out=json_out,
        success=f"Runtime behavioral schedule written: {output_path}",
    )


@runtime_behavior_app.command(
    "build-policy",
    help="Build a directed policy-pack-v3 for one baseline/subject pair.",
)
def build_policy_command(
    schedule: str = typer.Option(..., "--schedule"),
    baseline_binding: str = typer.Option(..., "--baseline-binding"),
    subject_binding: str = typer.Option(..., "--subject-binding"),
    tier: str = typer.Option("balanced", "--tier"),
    minimum_subject_score: float = typer.Option(..., "--minimum-subject-score"),
    maximum_regression: float = typer.Option(..., "--maximum-regression"),
    evidence_surface: list[str] = typer.Option([], "--evidence-surface"),
    out: str = typer.Option(..., "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Bind exact role inputs to one no-clobber behavioral authorization."""

    try:
        from invarlock.core.runtime_provider import load_runtime_behavioral_schedule
        from invarlock.policy_pack import build_behavioral_policy_pack
        from invarlock.runtime_behavior.io import atomic_write_new, canonical_json_bytes

        loaded_schedule = load_runtime_behavioral_schedule(Path(schedule))
        pack = build_behavioral_policy_pack(
            tier=tier,
            schedule_sha256=loaded_schedule.schedule_sha256,
            baseline=_load_behavioral_binding(Path(baseline_binding), role="baseline"),
            subject=_load_behavioral_binding(Path(subject_binding), role="subject"),
            metric_kind="exact_match",
            minimum_subject_score=minimum_subject_score,
            maximum_regression=maximum_regression,
            dataset_identity=loaded_schedule.dataset_identity.to_payload(),
            required_evidence_surfaces=evidence_surface or None,
        )
        output_path = Path(out)
        atomic_write_new(output_path, canonical_json_bytes(pack))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail(
            format_version=RUNTIME_BEHAVIOR_BUILD_POLICY_CLI_FORMAT,
            json_out=json_out,
            error=exc,
        )
        return

    _emit(
        {
            "format_version": RUNTIME_BEHAVIOR_BUILD_POLICY_CLI_FORMAT,
            "ok": True,
            "output": str(output_path),
            "policy_digest": pack["policy_digest"],
            "schedule_sha256": loaded_schedule.schedule_sha256,
        },
        json_out=json_out,
        success=f"Runtime behavioral policy written: {output_path}",
    )


@runtime_behavior_app.command(
    "prepare-binding",
    help=(
        "Validate native runtime inputs inside the same strict container boundary "
        "used by side production and emit one directed policy binding."
    ),
)
def prepare_binding_command(
    provider_name: str = typer.Option(
        ...,
        "--provider",
        help="Native provider: llama_cpp or tensorrt_llm.",
    ),
    model_id: str = typer.Option(
        ...,
        "--model-id",
        help="Provider-required privacy-safe artifact identity name.",
    ),
    settings: str = typer.Option(
        ...,
        "--settings",
        help="Strict JSON object containing the selected provider's public settings.",
    ),
    artifact: str = typer.Option(
        ...,
        "--artifact",
        help="Mounted GGUF file or TensorRT-LLM engine directory.",
    ),
    backend_executable: str = typer.Option(
        ...,
        "--backend-executable",
        help="Mounted llama-completion or TensorRT-LLM runner executable.",
    ),
    backend_source: str | None = typer.Option(
        None,
        "--backend-source",
        help="Required for llama_cpp: mounted authenticated source archive.",
    ),
    tokenizer_contract: str | None = typer.Option(
        None,
        "--tokenizer-contract",
        help="Required for tensorrt_llm: mounted external tokenizer contract.",
    ),
    container_image_digest: str = typer.Option(
        ...,
        "--container-image-digest",
        help=(
            "Reviewed sha256 digest that must match INVARLOCK_RUNTIME_IMAGE_DIGEST "
            "and the digest embedded in INVARLOCK_RUNTIME_IMAGE; the launch must "
            "also set INVARLOCK_CONTAINER_EXECUTION=1."
        ),
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="New path for the canonical five-field binding; never replaced.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one versioned machine-readable result object.",
    ),
) -> None:
    """Open and close the exact native provider inputs before authorizing them."""

    session: Any | None = None
    try:
        from invarlock.core.runtime_provider import (
            artifact_identity_sha256,
            runtime_execution_settings_from_mapping,
        )
        from invarlock.reporting.validation.runtime_behavioral_claim import (
            runtime_execution_settings_sha256,
        )
        from invarlock.runtime_behavior.io import atomic_write_new, canonical_json_bytes

        provider, spec, context, identity = _native_provider_inputs(
            provider_name=provider_name,
            model_id=model_id,
            settings_path=settings,
            artifact=artifact,
            backend_executable=backend_executable,
            backend_source=backend_source,
            tokenizer_contract=tokenizer_contract,
            container_image_digest=container_image_digest,
        )
        session = provider.open(spec, context)
        execution_settings = runtime_execution_settings_from_mapping(
            spec.settings,
            allow_network=False,
        )
        binding = {
            "provider_name": provider.name,
            "artifact_format": identity.artifact_format,
            "artifact_identity_sha256": artifact_identity_sha256(identity),
            "outer_image_digest": container_image_digest,
            "execution_settings_sha256": runtime_execution_settings_sha256(
                execution_settings
            ),
        }
        session.close()
        session = None
        output_path = Path(out)
        atomic_write_new(output_path, canonical_json_bytes(binding))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if session is not None:
            try:
                session.close()
            except (OSError, RuntimeError, ValueError):
                pass
        _fail(
            format_version=RUNTIME_BEHAVIOR_PREPARE_BINDING_CLI_FORMAT,
            json_out=json_out,
            error=exc,
        )
        return

    _emit(
        {
            "artifact_identity_sha256": binding["artifact_identity_sha256"],
            "execution_settings_sha256": binding["execution_settings_sha256"],
            "format_version": RUNTIME_BEHAVIOR_PREPARE_BINDING_CLI_FORMAT,
            "ok": True,
            "output": str(output_path),
            "provider_name": provider.name,
        },
        json_out=json_out,
        success=f"Runtime provider binding written: {output_path}",
    )


@runtime_behavior_app.command(
    "run-side",
    help="Run and strictly verify one baseline or subject provider side.",
)
def run_side_command(
    role: SideRole = typer.Option(..., "--role", case_sensitive=True),
    provider_name: str = typer.Option(..., "--provider"),
    model_id: str = typer.Option(..., "--model-id"),
    settings: str = typer.Option(
        ...,
        "--settings",
        help="Strict JSON object containing the selected provider's public settings.",
    ),
    artifact: str = typer.Option(
        ...,
        "--artifact",
        help="Mounted GGUF file or TensorRT-LLM engine directory.",
    ),
    backend_executable: str = typer.Option(
        ...,
        "--backend-executable",
        help="Mounted llama-completion or TensorRT-LLM runner executable.",
    ),
    backend_source: str | None = typer.Option(
        None,
        "--backend-source",
        help="Mounted authenticated llama.cpp source archive.",
    ),
    tokenizer_contract: str | None = typer.Option(
        None,
        "--tokenizer-contract",
        help="Mounted TensorRT-LLM external tokenizer contract.",
    ),
    container_image_digest: str = typer.Option(
        ...,
        "--container-image-digest",
        help=(
            "Reviewed sha256 digest that must match the required container "
            "runtime-boundary environment bindings."
        ),
    ),
    schedule: str = typer.Option(..., "--schedule"),
    policy_pack: str = typer.Option(..., "--policy-pack"),
    out: str = typer.Option(..., "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Construct only ephemeral bindings; portable output remains path-free."""

    try:
        provider, spec, context, _identity = _native_provider_inputs(
            provider_name=provider_name,
            model_id=model_id,
            settings_path=settings,
            artifact=artifact,
            backend_executable=backend_executable,
            backend_source=backend_source,
            tokenizer_contract=tokenizer_contract,
            container_image_digest=container_image_digest,
        )
        result = _run_side_api(
            role=role.value,
            provider=provider,
            spec=spec,
            context=context,
            schedule_path=Path(schedule),
            policy_pack_path=Path(policy_pack),
            output_directory=Path(out),
        )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail(
            format_version=RUNTIME_BEHAVIOR_RUN_SIDE_CLI_FORMAT,
            json_out=json_out,
            error=exc,
        )
        return

    payload: dict[str, object] = {
        "format_version": RUNTIME_BEHAVIOR_RUN_SIDE_CLI_FORMAT,
        "manifest": str(result.manifest_path),
        "ok": True,
        "provider_name": provider_name,
        "role": role.value,
        "side_directory": str(result.directory),
    }
    _emit(
        payload,
        json_out=json_out,
        success=f"Runtime {role.value} side verified: {result.directory}",
    )


@runtime_behavior_app.command(
    "verify-pair",
    help="Replay two directed side bundles and publish a positive receipt.",
)
def verify_pair_command(
    baseline: str = typer.Option(..., "--baseline"),
    subject: str = typer.Option(..., "--subject"),
    schedule: str = typer.Option(..., "--schedule"),
    policy_pack: str = typer.Option(..., "--policy-pack"),
    receipt: str = typer.Option(..., "--receipt"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    try:
        result = _verify_pair_api(
            baseline_directory=Path(baseline),
            subject_directory=Path(subject),
            schedule_path=Path(schedule),
            policy_pack_path=Path(policy_pack),
            receipt_path=Path(receipt),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail(
            format_version=RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT,
            json_out=json_out,
            error=exc,
        )
        return

    verification = result.verification
    payload = {
        "baseline_score": cast(Any, verification).baseline_score,
        "format_version": RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT,
        "ok": True,
        "receipt": str(result.receipt_path),
        "regression": cast(Any, verification).regression,
        "subject_score": cast(Any, verification).subject_score,
    }
    _emit(
        payload,
        json_out=json_out,
        success=f"Runtime provider pair verified: {result.receipt_path}",
    )


__all__ = [
    "RUNTIME_BEHAVIOR_BUILD_SCHEDULE_CLI_FORMAT",
    "RUNTIME_BEHAVIOR_BUILD_POLICY_CLI_FORMAT",
    "RUNTIME_BEHAVIOR_PREPARE_BINDING_CLI_FORMAT",
    "RUNTIME_BEHAVIOR_RUN_SIDE_CLI_FORMAT",
    "RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT",
    "runtime_behavior_app",
]
