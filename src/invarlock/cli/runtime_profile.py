"""Explicit, closed resource configuration for the existing OCI launch resolver."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invarlock.evaluation_runtime import ResolvedRuntimeConfig, ResolvedRuntimeSide
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
    profile: RuntimeProfile | None,
    *,
    explicit: Mapping[str, str],
    environment: Mapping[str, str],
) -> ResolvedRuntimeConfig:
    """Resolve profile and no-profile inputs into one immutable runtime selection."""
    from invarlock.evaluation_oci import OciWorkerLimits, _memory_limit_mib

    explicit = dict(explicit)
    environment = dict(environment)
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
    sources: dict[str, str] = {}
    profile_runtime = dict(profile.runtime) if profile is not None else {}
    profile_sides = {
        "baseline": dict(profile.baseline) if profile is not None else {},
        "subject": dict(profile.subject) if profile is not None else {},
    }

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
                (
                    profile_sides[side].get(field),
                    f"profile.{side}.{field}",
                )
            )
        candidates.append((profile_runtime.get(field), f"profile.runtime.{field}"))
        if side:
            key = "INVARLOCK_" + side.upper() + "_RUNTIME_" + field.upper()
            candidates.append((environment.get(key), key))
        candidates.extend(
            [(environment.get(env_name), env_name), (defaults[field], "default")]
        )
        for value, origin in candidates:
            if value is not None:
                sources[f"{side or 'runtime'}.{field}"] = origin
                return value
        raise AssertionError("every runtime field has a default")

    def resolve_side(side: str) -> ResolvedRuntimeSide:
        image = select("image", side)
        digest = select("image_digest", side)
        embedded = re.search(r"(?:^|@)(sha256:[0-9a-f]{64})$", image)
        if embedded:
            if digest and digest != embedded.group(1):
                raise RuntimeProfileError(
                    f"{side} runtime image and digest conflict: update the matching --{side}-runtime-image-digest or the profile digest; no override was repaired"
                )
            if not digest and sources[side + ".image_digest"] == "default":
                digest = embedded.group(1)
                sources[side + ".image_digest"] = (
                    sources[side + ".image"] + " (embedded digest)"
                )
        return ResolvedRuntimeSide(
            image_ref=image,
            image_digest=digest,
            device=select("device", side),
            entrypoint=select("entrypoint", side),
        )

    engine = select("engine")
    executable = shutil.which(engine, path=environment.get("PATH", os.defpath))
    engine_path = str(Path(executable).resolve()) if executable is not None else None
    sources["runtime.engine_path"] = "PATH" if "PATH" in environment else "default"
    return ResolvedRuntimeConfig(
        engine=engine,
        engine_path=engine_path,
        cpus=select("cpus"),
        memory_mib=_memory_limit_mib(select("memory_mib")),
        user=select("user"),
        baseline=resolve_side("baseline"),
        subject=resolve_side("subject"),
        sources=sources,
    )
