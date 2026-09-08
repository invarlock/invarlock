"""Explicit, closed resource configuration for the existing OCI launch resolver."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)

MAX_RUNTIME_PROFILE_BYTES = 16 * 1024
_SIDE = {"image", "image_digest", "device", "entrypoint"}
_COMMON = _SIDE | {"engine", "cpus", "memory_mib", "user"}


class RuntimeProfileError(StrictJsonError):
    """The caller-selected resource profile is ambiguous or incompatible."""


@dataclass(frozen=True)
class RuntimeProfile:
    digest: str
    runtime: dict[str, str]
    baseline: dict[str, str]
    subject: dict[str, str]


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    arguments: dict[str, str]
    sources: dict[str, str]


def load_runtime_profile(path: Path) -> RuntimeProfile:
    raw = read_regular_file_bytes(
        path, label="runtime profile", max_bytes=MAX_RUNTIME_PROFILE_BYTES
    )
    value = parse_json_bytes(raw, label="runtime profile")
    if (
        not isinstance(value, dict)
        or set(value) - {"format", "runtime", "baseline", "subject"}
        or value.get("format") != "invarlock/runtime-profile-v1"
        or "runtime" not in value
    ):
        raise RuntimeProfileError(
            "runtime profile requires format invarlock/runtime-profile-v1 and a closed runtime object"
        )
    groups: dict[str, dict[str, str]] = {}
    for group in ("runtime", "baseline", "subject"):
        fields = value.get(group, {})
        allowed = _COMMON if group == "runtime" else _SIDE
        if not isinstance(fields, dict) or set(fields) - allowed:
            raise RuntimeProfileError(
                f"runtime profile {group} has unsupported fields or is not an object"
            )
        groups[group] = {}
        for name, item in fields.items():
            if name == "memory_mib":
                if type(item) is not int or item <= 0:
                    raise RuntimeProfileError(
                        "runtime profile memory_mib must be a positive integer"
                    )
                groups[group][name] = str(item)
            else:
                if (
                    type(item) is not str
                    or not item.strip()
                    or len(item) > 2048
                    or any(ord(c) < 32 for c in item)
                ):
                    raise RuntimeProfileError(
                        f"runtime profile {group}.{name} must be a nonempty string without control characters"
                    )
                groups[group][name] = item
    return RuntimeProfile(
        "sha256:" + hashlib.sha256(raw).hexdigest(),
        groups["runtime"],
        groups["baseline"],
        groups["subject"],
    )


def resolve_runtime_profile(
    profile: RuntimeProfile,
    *,
    explicit: Mapping[str, str],
    environment: Mapping[str, str],
) -> ResolvedRuntimeProfile:
    """Resolve profile mode; callers without a profile keep their old resolver path."""
    from invarlock.evaluation_oci import OciWorkerLimits

    limits = OciWorkerLimits()
    defaults = {
        "engine": "docker",
        "device": "cpu",
        "entrypoint": "auto",
        "cpus": limits.cpus,
        "memory_mib": str(limits.memory_mib),
        "user": limits.user,
        "image": "",
        "image_digest": "",
    }
    # Every side setting is materialized below. Do not let the legacy resolver
    # validate an unused common environment fallback ahead of those settings.
    arguments: dict[str, str] = {
        "default_device": "cpu",
        "runtime_entrypoint": "auto",
    }
    sources: dict[str, str] = {}

    def select(field: str, side: str | None = None) -> str:
        common_name = "container_engine" if field == "engine" else "runtime_" + field
        env_name = (
            "INVARLOCK_CONTAINER_ENGINE"
            if field == "engine"
            else "INVARLOCK_RUNTIME_" + field.upper()
        )
        candidates: list[tuple[str | None, str]] = []
        if side:
            name = side + "_runtime_" + field
            candidates.append((explicit.get(name), "--" + name.replace("_", "-")))
        candidates.append(
            (explicit.get(common_name), "--" + common_name.replace("_", "-"))
        )
        if side:
            candidates.append(
                (getattr(profile, side).get(field), f"profile.{side}.{field}")
            )
        candidates.append((profile.runtime.get(field), f"profile.runtime.{field}"))
        if side:
            key = "INVARLOCK_" + side.upper() + "_RUNTIME_" + field.upper()
            candidates.append((environment.get(key), key))
        candidates.extend(
            [(environment.get(env_name), env_name), (defaults[field], "default")]
        )
        for value, origin in candidates:
            if value is not None and (value or origin == "default"):
                sources[f"{side or 'runtime'}.{field}"] = origin
                return value
        raise AssertionError("every runtime field has a default")

    for field, argument in [
        ("engine", "engine"),
        ("cpus", "runtime_cpus"),
        ("memory_mib", "runtime_memory_mib"),
        ("user", "runtime_user"),
    ]:
        arguments[argument] = select(field)
    for side in ("baseline", "subject"):
        for field, suffix in [
            ("image", "image_ref"),
            ("image_digest", "image_digest"),
            ("device", "device"),
            ("entrypoint", "entrypoint"),
        ]:
            arguments[side + "_" + suffix] = select(field, side)
        image = arguments[side + "_image_ref"]
        digest = arguments[side + "_image_digest"]
        embedded = re.search(r"@(sha256:[0-9a-f]{64})$", image)
        if embedded:
            if digest and digest != embedded.group(1):
                raise RuntimeProfileError(
                    f"{side} runtime image and digest conflict: update the matching --{side}-runtime-image-digest or the profile digest; no override was repaired"
                )
            if not digest:
                arguments[side + "_image_digest"] = embedded.group(1)
                sources[side + ".image_digest"] = (
                    sources[side + ".image"] + " (embedded digest)"
                )
    return ResolvedRuntimeProfile(arguments, sources)
