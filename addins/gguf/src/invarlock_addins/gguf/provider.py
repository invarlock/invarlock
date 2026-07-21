"""First-party GGUF runtime provider backed by a pinned llama.cpp CLI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from invarlock_addins.gguf import __version__ as ADDIN_VERSION
from invarlock_addins.gguf.session import (
    LlamaCppRuntimeBindings,
    LlamaCppSession,
    LlamaCppSessionConfig,
    authenticate_llama_cpp_backend_files,
    inspect_llama_cpp_backend,
    validate_llama_cpp_backend_version,
)

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    GGUFArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBehavioralSchedule,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    artifact_identity_sha256,
    runtime_execution_settings_from_mapping,
)
from invarlock.core.runtime_provider.request_bindings import (
    LLAMA_CPP_REQUEST_SETTINGS as _ALLOWED_SETTINGS,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
from invarlock.runtime_security_helpers import (
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    strict_container_boundary_present,
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_IPV4_ROUTE_PATH = Path("/proc/net/route")
_IPV6_ROUTE_PATH = Path("/proc/net/ipv6_route")
_CPU_INFO_PATH = Path("/proc/cpuinfo")
_MAX_CPU_INFO_BYTES = 1024 * 1024
_CPU_IDENTITY_FIELDS = frozenset(
    {
        "address sizes",
        "bugs",
        "cpu architecture",
        "cpu cores",
        "cpu family",
        "cpu implementer",
        "cpu model",
        "cpu part",
        "cpu revision",
        "cpu variant",
        "features",
        "flags",
        "hardware",
        "isa",
        "microcode",
        "model",
        "model name",
        "revision",
        "siblings",
        "stepping",
        "uarch",
        "vendor_id",
    }
)
_REQUIRED_SETTINGS = _ALLOWED_SETTINGS
_POSITIVE_INTEGER_SETTINGS = frozenset(
    {
        "artifact_byte_length",
        "batch_size",
        "context_length",
        "max_output_tokens",
        "timeout_seconds",
    }
)
_DIGEST_SETTINGS = frozenset(
    {
        "artifact_sha256",
        "backend_binary_sha256",
        "backend_source_sha256",
        "gguf_metadata_sha256",
        "tensor_inventory_sha256",
        "tokenizer_metadata_sha256",
    }
)


def _required_text(settings: Mapping[str, JSONScalar], name: str) -> str:
    value = settings.get(name)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty trimmed printable string")
    return value


def _required_digest(settings: Mapping[str, JSONScalar], name: str) -> str:
    value = _required_text(settings, name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


def _observe_linux_cpu() -> RuntimeDeviceFacts:
    if platform.system() != "Linux":
        raise ValueError("llama_cpp execution requires Linux")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_CPU_INFO_PATH, flags)
    except OSError as exc:
        raise ValueError("llama_cpp cannot observe the Linux CPU identity") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("llama_cpp Linux CPU identity is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, _MAX_CPU_INFO_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_CPU_INFO_BYTES:
                raise ValueError("llama_cpp Linux CPU identity exceeds the byte limit")
        payload = b"".join(chunks)
        if not payload:
            raise ValueError("llama_cpp Linux CPU identity is empty")
    except OSError as exc:
        raise ValueError("llama_cpp cannot read the Linux CPU identity") from exc
    finally:
        os.close(descriptor)
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("llama_cpp Linux CPU identity is not UTF-8") from exc
    model_name: str | None = None
    for line in decoded.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip().lower() not in {
            "model name",
            "hardware",
            "cpu model",
            "processor",
        }:
            continue
        normalized = " ".join(value.split())
        if normalized and not normalized.isdecimal():
            model_name = normalized
            break
    machine = " ".join(os.uname().machine.split())
    if not machine or any(ord(character) < 32 for character in machine):
        raise ValueError("llama_cpp Linux machine identity is invalid")
    observed_name = model_name or "CPU"
    identity_fields: dict[str, set[str]] = {}
    for line in decoded.splitlines():
        key, separator, value = line.partition(":")
        canonical_key = " ".join(key.strip().lower().split())
        normalized_value = " ".join(value.split())
        if separator and canonical_key in _CPU_IDENTITY_FIELDS and normalized_value:
            identity_fields.setdefault(canonical_key, set()).add(normalized_value)
    canonical_identity = {
        "fields": {
            key: sorted(values) for key, values in sorted(identity_fields.items())
        },
        "machine": machine,
    }
    cpu_identity_sha256 = hashlib.sha256(
        json.dumps(
            canonical_identity,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return RuntimeDeviceFacts(
        device_kind="cpu",
        device_name=(
            f"{observed_name} [{machine}; cpu_identity_sha256={cpu_identity_sha256}]"
        ),
    )


def _required_integer(
    settings: Mapping[str, JSONScalar], name: str, *, positive: bool
) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    return value


def _require_isolated_network_namespace(
    *,
    ipv4_route_path: Path = _IPV4_ROUTE_PATH,
    ipv6_route_path: Path = _IPV6_ROUTE_PATH,
) -> None:
    """Fail closed unless every kernel route is bound to loopback."""

    try:
        ipv4_lines = ipv4_route_path.read_text(
            encoding="ascii", errors="strict"
        ).splitlines()
        ipv6_lines = ipv6_route_path.read_text(
            encoding="ascii", errors="strict"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("llama_cpp cannot verify the network namespace") from exc

    if (
        not ipv4_lines
        or not ipv4_lines[0].split()
        or ipv4_lines[0].split()[0] != "Iface"
    ):
        raise ValueError("llama_cpp cannot verify the IPv4 route table")
    ipv4_interfaces: set[str] = set()
    for line in ipv4_lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 11:
            raise ValueError("llama_cpp cannot verify the IPv4 route table")
        ipv4_interfaces.add(fields[0])

    ipv6_interfaces: set[str] = set()
    for line in ipv6_lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10:
            raise ValueError("llama_cpp cannot verify the IPv6 route table")
        ipv6_interfaces.add(fields[-1])

    if (ipv4_interfaces | ipv6_interfaces) - {"lo"}:
        raise ValueError("llama_cpp requires a network-disabled container")


def _require_strict_container_boundary(context: RuntimeExecutionContext) -> None:
    if context.container_image_digest is None:
        raise ValueError(
            "strict llama_cpp execution requires a pinned outer container image"
        )
    if not strict_container_boundary_present():
        raise ValueError(
            "strict llama_cpp execution requires the authenticated container boundary"
        )
    runtime_image_digest = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    if _IMAGE_DIGEST.fullmatch(runtime_image_digest) is None:
        raise ValueError(
            f"strict llama_cpp execution requires canonical {RUNTIME_IMAGE_DIGEST_ENV}"
        )
    if runtime_image_digest != context.container_image_digest:
        raise ValueError(
            "strict llama_cpp runtime image digest does not match the container context"
        )
    runtime_image = os.environ.get(RUNTIME_IMAGE_ENV, "")
    repository, separator, embedded_digest = runtime_image.rpartition("@")
    if runtime_image != runtime_image_digest and not (
        repository and separator and embedded_digest == runtime_image_digest
    ):
        raise ValueError(
            f"strict llama_cpp execution requires {RUNTIME_IMAGE_ENV} to embed "
            "the exact runtime image digest"
        )
    _require_isolated_network_namespace()


def _require_inspection_container_boundary() -> None:
    """Authenticate the current immutable runtime before probing native code."""

    if not strict_container_boundary_present():
        raise ValueError(
            "llama_cpp inspection requires the authenticated container boundary"
        )
    runtime_image_digest = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    if _IMAGE_DIGEST.fullmatch(runtime_image_digest) is None:
        raise ValueError(
            f"llama_cpp inspection requires canonical {RUNTIME_IMAGE_DIGEST_ENV}"
        )
    runtime_image = os.environ.get(RUNTIME_IMAGE_ENV, "")
    repository, separator, embedded_digest = runtime_image.rpartition("@")
    if runtime_image != runtime_image_digest and not (
        repository and separator and embedded_digest == runtime_image_digest
    ):
        raise ValueError(
            f"llama_cpp inspection requires {RUNTIME_IMAGE_ENV} to embed the "
            "exact runtime image digest"
        )
    _require_isolated_network_namespace()


class LlamaCppProvider:
    """Authenticate one GGUF artifact through pinned raw llama-completion."""

    name = "llama_cpp"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError(
                f"provider_name must be {self.name!r}, got {spec.provider_name!r}"
            )
        unknown = set(spec.settings) - _ALLOWED_SETTINGS
        if unknown:
            rendered = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported llama_cpp setting(s): {rendered}")
        missing = _REQUIRED_SETTINGS - set(spec.settings)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ValueError(f"missing llama_cpp setting(s): {rendered}")
        for name in _DIGEST_SETTINGS:
            _required_digest(spec.settings, name)
        for name in _POSITIVE_INTEGER_SETTINGS:
            _required_integer(spec.settings, name, positive=True)
        _required_integer(spec.settings, "seed", positive=False)
        validate_llama_cpp_backend_version(
            _required_text(spec.settings, "backend_version")
        )
        artifact_sha256 = _required_digest(spec.settings, "artifact_sha256")
        expected_model_id = f"gguf-sha256-{artifact_sha256}.gguf"
        if spec.model_id != expected_model_id:
            raise ValueError(
                "llama_cpp model_id must be the privacy-safe full GGUF digest name"
            )

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=("gguf",),
            tasks=("text_causal",),
            metrics=("exact_match",),
            execution_modes=("container", "local_process"),
            required_extra=None,
            required_image=None,
        )

    def inspect_runtime_spec(
        self,
        bindings: LlamaCppRuntimeBindings,
        *,
        seed: int,
        context_length: int,
        batch_size: int,
        max_output_tokens: int,
        timeout_seconds: int,
    ) -> ModelRuntimeSpec:
        """Derive one complete spec from authenticated local runtime inputs."""

        if not isinstance(bindings, LlamaCppRuntimeBindings):
            raise ValueError("llama_cpp inspection requires native runtime bindings")
        _require_inspection_container_boundary()
        identity = read_gguf_artifact_identity(bindings.gguf_path)
        backend = inspect_llama_cpp_backend(bindings)
        if read_gguf_artifact_identity(bindings.gguf_path) != identity:
            raise ValueError("GGUF artifact changed during runtime inspection")
        spec = ModelRuntimeSpec(
            provider_name=self.name,
            model_id=identity.artifact_name,
            settings={
                "artifact_byte_length": identity.byte_length,
                "artifact_sha256": identity.sha256,
                "backend_binary_sha256": backend.binary_sha256,
                "backend_source_sha256": backend.source_sha256,
                "backend_version": backend.version,
                "batch_size": batch_size,
                "context_length": context_length,
                "gguf_metadata_sha256": identity.gguf_metadata_sha256,
                "max_output_tokens": max_output_tokens,
                "seed": seed,
                "tensor_inventory_sha256": identity.tensor_inventory_sha256,
                "timeout_seconds": timeout_seconds,
                "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
            },
        )
        self.validate_config(spec)
        if self.identify_artifact(spec) != identity:
            raise ValueError("derived GGUF settings do not reproduce artifact identity")
        return spec

    def identify_artifact(self, spec: ModelRuntimeSpec) -> GGUFArtifactIdentity:
        self.validate_config(spec)
        return GGUFArtifactIdentity(
            artifact_name=spec.model_id,
            sha256=_required_digest(spec.settings, "artifact_sha256"),
            byte_length=_required_integer(
                spec.settings, "artifact_byte_length", positive=True
            ),
            gguf_metadata_sha256=_required_digest(
                spec.settings, "gguf_metadata_sha256"
            ),
            tensor_inventory_sha256=_required_digest(
                spec.settings, "tensor_inventory_sha256"
            ),
            tokenizer_metadata_sha256=_required_digest(
                spec.settings, "tokenizer_metadata_sha256"
            ),
        )

    def authenticate_artifact(
        self, spec: ModelRuntimeSpec, artifact_path: Path
    ) -> GGUFArtifactIdentity:
        """Authenticate one local GGUF file without starting llama.cpp."""

        expected = self.identify_artifact(spec)
        observed = read_gguf_artifact_identity(artifact_path)
        if observed != expected:
            raise ValueError("primary GGUF artifact identity does not match spec")
        return observed

    def validate_evaluation_inputs(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
        schedule: RuntimeBehavioralSchedule,
    ) -> None:
        """Authenticate static GGUF runtime inputs without starting llama.cpp."""

        self.validate_config(spec)
        if schedule.task not in self.capabilities().tasks:
            raise ValueError(
                f"llama_cpp does not support schedule task {schedule.task!r}"
            )
        resources.require_support_names(
            frozenset({"backend_executable", "backend_source"})
        )
        if resources.device_kind != "cpu":
            raise ValueError("llama_cpp input preflight requires a CPU device")
        bindings = LlamaCppRuntimeBindings(
            gguf_path=resources.primary_path(),
            executable_path=resources.support_path("backend_executable"),
            source_archive_path=resources.support_path("backend_source"),
        )
        self.authenticate_artifact(spec, bindings.gguf_path)
        authenticate_llama_cpp_backend_files(
            bindings,
            expected_binary_sha256=_required_digest(
                spec.settings, "backend_binary_sha256"
            ),
            expected_source_sha256=_required_digest(
                spec.settings, "backend_source_sha256"
            ),
        )

    def prepare_execution(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
    ) -> RuntimeExecutionContext:
        """Bind one root-confined GGUF artifact to its authenticated backend."""

        self.validate_config(spec)
        resources.require_support_names(
            frozenset({"backend_executable", "backend_source"})
        )
        if resources.device_kind != "cpu":
            raise ValueError("llama_cpp preparation requires a CPU device")
        bindings = LlamaCppRuntimeBindings(
            gguf_path=resources.primary_path(),
            executable_path=resources.support_path("backend_executable"),
            source_archive_path=resources.support_path("backend_source"),
        )
        identity = self.authenticate_artifact(spec, bindings.gguf_path)
        return RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=resources.container_image_digest,
            device_kind="cpu",
            artifact_identity_sha256=artifact_identity_sha256(identity),
            provider_state=bindings,
        )

    def open(
        self,
        spec: ModelRuntimeSpec,
        context: RuntimeExecutionContext,
    ) -> LlamaCppSession:
        self.validate_config(spec)
        if context.allow_network:
            raise ValueError("llama_cpp execution must disable network access")
        if context.device_kind != "cpu":
            raise ValueError("llama_cpp execution requires a CPU device")
        if context.strict:
            _require_strict_container_boundary(context)
        if context.strict and context.artifact_identity_sha256 is None:
            raise ValueError(
                "strict llama_cpp execution requires artifact identity binding"
            )
        if not isinstance(context.provider_state, LlamaCppRuntimeBindings):
            raise ValueError("llama_cpp requires ephemeral runtime bindings in context")
        if context.scorer is not None:
            raise ValueError("llama_cpp does not accept in-process scorer bindings")

        identity = self.identify_artifact(spec)
        expected_identity_sha256 = artifact_identity_sha256(identity)
        if (
            context.artifact_identity_sha256 is not None
            and context.artifact_identity_sha256 != expected_identity_sha256
        ):
            raise ValueError(
                "runtime context artifact identity does not match llama_cpp spec"
            )
        observed_identity = read_gguf_artifact_identity(
            context.provider_state.gguf_path
        )
        if observed_identity != identity:
            raise ValueError(
                "bound GGUF artifact identity does not match llama_cpp spec"
            )

        execution_settings = runtime_execution_settings_from_mapping(
            spec.settings,
            allow_network=False,
        )
        return LlamaCppSession(
            LlamaCppSessionConfig(
                artifact_identity=identity,
                backend_binary_sha256=_required_digest(
                    spec.settings, "backend_binary_sha256"
                ),
                backend_source_sha256=_required_digest(
                    spec.settings, "backend_source_sha256"
                ),
                backend_version=_required_text(spec.settings, "backend_version"),
                execution_settings=execution_settings,
                capabilities=self.capabilities(),
                plugin=RuntimeProviderPluginIdentity(
                    name=self.name,
                    distribution="invarlock-runtime-gguf",
                    distribution_version=ADDIN_VERSION,
                ),
                device=_observe_linux_cpu(),
                outer_image_digest=context.container_image_digest,
                bindings=context.provider_state,
            )
        )


__all__ = [
    "LlamaCppProvider",
    "LlamaCppRuntimeBindings",
]
