"""Caller-owned runtime resources for evaluation transactions.

Evaluation requests describe portable comparison intent.  They cannot grant host
permissions, choose executable support files, or assert the image in which they
run.  This module keeps those bindings in a separate caller-owned object and
turns them into the closed provider ABI resource contract.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from invarlock._optional_runtime_profiles import OPTIONAL_RUNTIME_PROVIDER_PROFILES
from invarlock.core.evaluation_request import ComparisonSideRequest
from invarlock.core.runtime_provider import RuntimeArtifactResources, RuntimeProvider
from invarlock.runtime_security_helpers import resolve_runtime_image_digest

type RuntimeSideRole = Literal["baseline", "subject"]


class RuntimeResourceResolutionError(ValueError):
    """Raised when trusted execution resources are absent or escape their root."""


class RuntimeResourceResolver(Protocol):
    """Resolve resources without consulting untrusted request fields for support."""

    def resolve(
        self,
        *,
        request_root: Path,
        role: RuntimeSideRole,
        side: ComparisonSideRequest,
        provider: RuntimeProvider,
    ) -> RuntimeArtifactResources: ...


@dataclass(frozen=True)
class ProviderResourceBinding:
    """Trusted root and support files for one optional provider add-in."""

    root: Path
    support_resources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise RuntimeResourceResolutionError(
                "provider resource root must be absolute"
            )
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self,
            "support_resources",
            MappingProxyType(dict(self.support_resources)),
        )


@dataclass(frozen=True)
class CallerRuntimeResources:
    """Closed runtime bindings supplied by the caller."""

    container_image_digest: str
    default_device: str = "cpu"
    side_devices: Mapping[RuntimeSideRole, str] = field(default_factory=dict)
    provider_bindings: Mapping[str, ProviderResourceBinding] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        side_devices = dict(self.side_devices)
        if set(side_devices) - {"baseline", "subject"}:
            raise RuntimeResourceResolutionError("runtime side device key is invalid")
        provider_bindings = dict(self.provider_bindings)
        object.__setattr__(self, "side_devices", MappingProxyType(side_devices))
        object.__setattr__(
            self, "provider_bindings", MappingProxyType(provider_bindings)
        )

    def resolve(
        self,
        *,
        request_root: Path,
        role: RuntimeSideRole,
        side: ComparisonSideRequest,
        provider: RuntimeProvider,
    ) -> RuntimeArtifactResources:
        """Bind one request artifact beneath a caller-selected resource root."""

        if provider.name != side.runtime.provider:
            raise RuntimeResourceResolutionError(
                f"{role} runtime provider identity does not match the request"
            )
        artifact = side.artifact.path
        if artifact is None:
            raise RuntimeResourceResolutionError(
                f"{role} run mode requires a local artifact"
            )
        if provider.name == "hf_transformers":
            root = Path(request_root)
            support: Mapping[str, str] = {}
        else:
            binding = self.provider_bindings.get(provider.name)
            if binding is None:
                raise RuntimeResourceResolutionError(
                    f"{role} provider {provider.name!r} requires caller-owned "
                    "resource configuration"
                )
            root = binding.root
            support = binding.support_resources
        try:
            primary = artifact.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeResourceResolutionError(
                f"{role} artifact is outside the caller-owned provider resource root"
            ) from exc
        if not primary or primary == ".":
            raise RuntimeResourceResolutionError(
                f"{role} artifact must name a resource beneath its root"
            )
        try:
            return RuntimeArtifactResources(
                root=root,
                primary_artifact=primary,
                support_resources=support,
                device_kind=self.side_devices.get(role, self.default_device),
                container_image_digest=self.container_image_digest,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeResourceResolutionError(
                f"{role} runtime resources are invalid: {exc}"
            ) from exc


def _optional_provider_binding(
    *,
    root_variable: str,
    support_variables: Mapping[str, str],
) -> ProviderResourceBinding | None:
    root_value = os.environ.get(root_variable)
    observed_support = {
        name: value
        for name, variable in support_variables.items()
        if (value := os.environ.get(variable)) is not None
    }
    if root_value is None:
        if observed_support:
            raise RuntimeResourceResolutionError(
                f"{root_variable} is required when provider support is configured"
            )
        return None
    missing = sorted(set(support_variables) - set(observed_support))
    if missing:
        variable = support_variables[missing[0]]
        raise RuntimeResourceResolutionError(
            f"{variable} is required for the configured provider"
        )
    return ProviderResourceBinding(
        root=Path(root_value), support_resources=observed_support
    )


def caller_runtime_resources_from_environment() -> CallerRuntimeResources:
    """Load the small trusted runtime context exposed by the public CLI."""

    image_digest = resolve_runtime_image_digest()
    if image_digest is None:
        raise RuntimeResourceResolutionError(
            "INVARLOCK_RUNTIME_IMAGE_DIGEST must bind the executing image"
        )
    default_device = os.environ.get("INVARLOCK_RUNTIME_DEVICE", "cpu")
    side_devices: dict[RuntimeSideRole, str] = {}
    device_variables: tuple[tuple[RuntimeSideRole, str], ...] = (
        ("baseline", "INVARLOCK_BASELINE_RUNTIME_DEVICE"),
        ("subject", "INVARLOCK_SUBJECT_RUNTIME_DEVICE"),
    )
    for role, variable in device_variables:
        value = os.environ.get(variable)
        if value is not None:
            side_devices[role] = value
    provider_bindings: dict[str, ProviderResourceBinding] = {}
    for profile in OPTIONAL_RUNTIME_PROVIDER_PROFILES.values():
        binding = _optional_provider_binding(
            root_variable=profile.resource_root_environment,
            support_variables=dict(profile.support_resource_environment),
        )
        if binding is not None:
            provider_bindings[profile.provider_name] = binding
    return CallerRuntimeResources(
        container_image_digest=image_digest,
        default_device=default_device,
        side_devices=side_devices,
        provider_bindings=provider_bindings,
    )


__all__ = [
    "CallerRuntimeResources",
    "ProviderResourceBinding",
    "RuntimeResourceResolutionError",
    "RuntimeResourceResolver",
    "RuntimeSideRole",
    "caller_runtime_resources_from_environment",
]
