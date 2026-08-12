#!/usr/bin/env python3
"""Run one repository-maintained runtime evidence qualification transaction.

This maintainer tool deliberately composes the public ``evaluate``, ``verify``,
and ``report`` commands instead of adding another public CLI workflow.  It is
stdlib-only so a missing dependency in the selected qualification environment
is reported as a structured stage failure rather than preventing this driver
from starting.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

try:
    from scripts.qualification_source import authenticate_bundle
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve(strict=True).parent))
    from qualification_source import (  # type: ignore[import-not-found, no-redef]
        authenticate_bundle,
    )

FORMAT_VERSION = "invarlock/runtime-qualification-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_MAX_DIAGNOSTIC_CHARS = 16 * 1024
_MAX_CHILD_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_CANDIDATE_MANIFEST_BYTES = 1024 * 1024
_MAX_CANDIDATE_WHEEL_BYTES = 512 * 1024 * 1024
_MAX_CANDIDATE_WHEEL_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_CANDIDATE_WHEEL_MEMBERS = 50_000
_MAX_CANDIDATE_WHEELS = 8
_MAX_SOURCE_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_SOURCE_MEMBERS = 50_000
_CHILD_ENVIRONMENT_KEYS = (
    "CONTAINER_HOST",
    "CUDA_VISIBLE_DEVICES",
    "DOCKER_CONFIG",
    "DOCKER_HOST",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_CACHE",
    "TZ",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
)
_PROVIDER_ENVIRONMENT_KEYS = (
    "INVARLOCK_GGUF_BACKEND_EXECUTABLE",
    "INVARLOCK_GGUF_BACKEND_SOURCE",
    "INVARLOCK_GGUF_RESOURCE_ROOT",
    "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE",
    "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT",
    "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT",
    "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT",
)
_LIVE_QUALIFICATION_SOURCES = (
    "scripts/qualification_source.py",
    "scripts/runtime_qualification.py",
)
_CANDIDATE_MANIFEST_FORMAT = "invarlock/qualification-candidate-wheels-v1"
_CANDIDATE_DISTRIBUTION_SOURCES = {
    "invarlock": ("src/invarlock", "invarlock"),
    "invarlock-diagnostics": (
        "addins/diagnostics/src/invarlock_addins/diagnostics",
        "invarlock_addins/diagnostics",
    ),
    "invarlock-runtime-gguf": (
        "addins/gguf/src/invarlock_addins/gguf",
        "invarlock_addins/gguf",
    ),
    "invarlock-runtime-hf-vision-text": (
        "addins/multimodal/src/invarlock_addins/multimodal",
        "invarlock_addins/multimodal",
    ),
    "invarlock-runtime-tensorrt-llm": (
        "addins/tensorrt_llm/src/invarlock_addins/tensorrt_llm",
        "invarlock_addins/tensorrt_llm",
    ),
}
_CANDIDATE_PROBE = r"""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

site = Path(
    os.environ["INVARLOCK_QUALIFICATION_CANDIDATE_SITE"]
).resolve(strict=True)
expected = json.loads(os.environ["INVARLOCK_CANDIDATE_DISTRIBUTIONS"])
observed = {}
providers = []
for distribution in importlib.metadata.distributions(path=[str(site)]):
    name = distribution.metadata.get("Name")
    version = distribution.metadata.get("Version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise SystemExit("candidate distribution metadata is incomplete")
    normalized = name.lower().replace("_", "-").replace(".", "-")
    if normalized in observed:
        raise SystemExit("candidate distribution identity is duplicated")
    observed[normalized] = version
    for entry_point in distribution.entry_points:
        if entry_point.group != "invarlock.runtime_providers":
            continue
        provider = entry_point.load()
        module = importlib.import_module(provider.__module__)
        module_file = Path(module.__file__).resolve(strict=True)
        if not module_file.is_relative_to(site):
            raise SystemExit("candidate provider entry point escaped candidate site")
        providers.append(entry_point.name)
if observed != expected:
    raise SystemExit("candidate distribution discovery does not match manifest")
import invarlock
core_file = Path(invarlock.__file__).resolve(strict=True)
if not core_file.is_relative_to(site):
    raise SystemExit("candidate core import escaped candidate site")
if invarlock.__version__ != expected.get("invarlock"):
    raise SystemExit("candidate core version does not match wheel metadata")
print(json.dumps({
    "distributions": observed,
    "format_version": "invarlock/qualification-candidate-probe-v1",
    "ok": True,
    "providers": sorted(providers),
}, separators=(",", ":"), sort_keys=True))
"""
_PYTHON_BOOTSTRAP = r"""
import os
import runpy
import sys
import sysconfig
from pathlib import Path

candidate = Path(
    os.environ["INVARLOCK_QUALIFICATION_CANDIDATE_SITE"]
).resolve(strict=True)
roots = [candidate]
venv = Path(os.path.abspath(sys.executable)).parent.parent
if venv.joinpath("pyvenv.cfg").is_file():
    if os.name == "nt":
        roots.append(venv / "Lib" / "site-packages")
    else:
        roots.append(
            venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
paths = sysconfig.get_paths()
roots.extend(Path(paths[name]) for name in ("purelib", "platlib"))
sys.path[:0] = [
    str(root)
    for index, root in enumerate(roots)
    if root.is_dir() and root not in roots[:index]
]
mode, target, *arguments = sys.argv[1:]
sys.argv = [target, *arguments]
if mode == "module":
    runpy.run_module(target, run_name="__main__", alter_sys=True)
elif mode == "path":
    runpy.run_path(target, run_name="__main__")
elif mode == "code":
    exec(compile(target, "<qualification-probe>", "exec"), {"__name__": "__main__"})
else:
    raise SystemExit("unsupported qualification bootstrap mode")
"""


class QualificationError(RuntimeError):
    """One fail-closed qualification-stage failure."""

    def __init__(self, stage: str, message: str, *, diagnostic: object = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class PythonIdentity:
    """Stable identity of the interpreter selected for qualification children."""

    path: str
    resolved_path: str
    sha256: str
    stat_identity: tuple[int, int, int, int]

    def summary(self) -> dict[str, str]:
        return {
            "path": self.path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CandidateWheelSpec:
    """Caller-authorized candidate wheel and independently expected digest."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class CandidateWheelIdentity:
    """Captured candidate distribution identity used by child commands."""

    distribution: str
    version: str
    filename: str
    sha256: str

    def summary(self) -> dict[str, str]:
        return {
            "distribution": self.distribution,
            "filename": self.filename,
            "sha256": self.sha256,
            "version": self.version,
        }


@dataclass(frozen=True)
class QualificationInputs:
    """Closed caller-owned inputs shared by readiness and execution."""

    mode: str
    python: str
    request: Path
    request_root: Path
    signing_key: Path
    runtime_image: str
    runtime_image_digest: str
    evidence: Path
    trust_profile: Path
    receipt: Path
    canary_evidence: Path | None
    canary_receipt: Path | None
    canary_trust_profile: Path | None
    source_commit: str
    source_bundle: Path
    source_bundle_sha256: str
    source_execution_sha256: str
    candidate_wheel_manifest: Path
    container_engine: str
    container_engine_path: str
    container_engine_sha256: str
    runtime_device: str
    runtime_cpus: str | None
    runtime_memory_mib: int | None
    runtime_user: str | None
    report: Path | None
    summary: Path | None


@dataclass(frozen=True)
class ExecutionContext:
    """Authenticated helper snapshot and empty child working directory."""

    source_root: Path
    working_directory: Path
    child_path: str
    candidate_site: Path
    candidate_manifest_sha256: str
    candidate_wheels: tuple[CandidateWheelIdentity, ...]
    python_identity: PythonIdentity


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_regular_file(path: Path, *, label: str, stage: str) -> str:
    """Hash one opened regular file and reject final-component substitution."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(stage, f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError(stage, f"{label} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise QualificationError(stage, f"{label} changed while it was hashed")
        return f"sha256:{digest.hexdigest()}"
    finally:
        os.close(descriptor)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def no_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite number: {value}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise QualificationError(
            "configuration", f"{label} is not strict JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise QualificationError("configuration", f"{label} must be a JSON object")
    return payload


def _candidate_wheel_specs(
    manifest_path: Path,
) -> tuple[str, tuple[CandidateWheelSpec, ...]]:
    lexical = Path(os.path.abspath(os.fspath(manifest_path)))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "configuration", "candidate-wheel manifest is unavailable"
        ) from exc
    if resolved != lexical:
        raise QualificationError(
            "configuration", "candidate-wheel manifest must not traverse links"
        )
    raw = _read_regular_bytes(
        resolved,
        label="candidate-wheel manifest",
        max_bytes=_MAX_CANDIDATE_MANIFEST_BYTES,
    )
    payload = _strict_json_object(raw, label="candidate-wheel manifest")
    if (
        set(payload) != {"format_version", "wheels"}
        or payload.get("format_version") != _CANDIDATE_MANIFEST_FORMAT
    ):
        raise QualificationError(
            "configuration", "candidate-wheel manifest contract is invalid"
        )
    wheels = payload.get("wheels")
    if (
        not isinstance(wheels, list)
        or not wheels
        or len(wheels) > _MAX_CANDIDATE_WHEELS
    ):
        raise QualificationError(
            "configuration", "candidate-wheel manifest inventory is invalid"
        )
    specs: list[CandidateWheelSpec] = []
    observed_paths: set[Path] = set()
    for item in wheels:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise QualificationError(
                "configuration", "candidate-wheel manifest entry is invalid"
            )
        value = item.get("path")
        if not isinstance(value, str) or not value or "\x00" in value:
            raise QualificationError(
                "configuration", "candidate-wheel manifest path is invalid"
            )
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = resolved.parent / candidate
        candidate = Path(os.path.abspath(os.fspath(candidate)))
        try:
            candidate_resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(
                "configuration", "candidate wheel is unavailable"
            ) from exc
        if (
            candidate_resolved != candidate
            or candidate.suffix != ".whl"
            or candidate in observed_paths
        ):
            raise QualificationError(
                "configuration", "candidate wheel path is invalid or repeated"
            )
        observed_paths.add(candidate)
        specs.append(
            CandidateWheelSpec(
                path=candidate,
                sha256=_digest(
                    item.get("sha256"),
                    label="candidate-wheel digest",
                    stage="configuration",
                ),
            )
        )
    return _sha256_bytes(raw), tuple(specs)


def _file_identity(path: Path, *, label: str) -> tuple[str, tuple[int, int, int, int]]:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError("configuration", f"{label} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError("configuration", f"{label} is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise QualificationError("configuration", f"{label} changed while it was read")
    return f"sha256:{digest.hexdigest()}", identity


def _python_identity(value: str) -> PythonIdentity:
    path = Path(qualification_python(value))
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "configuration", "qualification Python executable is unavailable"
        ) from exc
    if not os.access(resolved, os.X_OK):
        raise QualificationError(
            "configuration", "qualification Python must be executable"
        )
    digest, identity = _file_identity(resolved, label="qualification Python")
    return PythonIdentity(
        path=str(path),
        resolved_path=str(resolved),
        sha256=digest,
        stat_identity=identity,
    )


def _assert_python_identity(identity: PythonIdentity, *, stage: str) -> None:
    try:
        resolved = Path(identity.path).resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            stage, "qualification Python became unavailable"
        ) from exc
    if str(resolved) != identity.resolved_path:
        raise QualificationError(
            stage, "qualification Python path changed after binding"
        )
    try:
        digest, observed = _file_identity(resolved, label="qualification Python")
    except QualificationError as exc:
        raise QualificationError(stage, exc.message) from exc
    if digest != identity.sha256 or observed != identity.stat_identity:
        raise QualificationError(stage, "qualification Python changed after binding")


def _safe_wheel_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _wheel_distribution(
    archive: zipfile.ZipFile,
) -> tuple[str, str, str]:
    metadata_members = [
        member
        for member in archive.infolist()
        if member.filename.endswith(".dist-info/METADATA") and not member.is_dir()
    ]
    if len(metadata_members) != 1:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel metadata inventory is invalid"
        )
    member = metadata_members[0]
    if member.file_size > _MAX_CANDIDATE_WHEEL_MEMBER_BYTES:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel metadata is too large"
        )
    message = BytesParser(policy=policy.default).parsebytes(archive.read(member))
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if message.defects or len(names) != 1 or len(versions) != 1:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel metadata identity is invalid"
        )
    distribution = _normalized_distribution(str(names[0]))
    version = str(versions[0])
    if distribution not in _CANDIDATE_DISTRIBUTION_SOURCES or not version:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel distribution is not maintained"
        )
    return distribution, version, member.filename.rsplit("/", 1)[0]


def _write_candidate_member(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=stat.S_IRWXU)
    descriptor = -1
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IRUSR,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel extraction failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _capture_candidate_wheel(
    spec: CandidateWheelSpec,
    *,
    archived: dict[str, bytes],
    candidate_site: Path,
) -> CandidateWheelIdentity:
    descriptor = -1
    try:
        descriptor = os.open(
            spec.path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > _MAX_CANDIDATE_WHEEL_BYTES
        ):
            raise QualificationError(
                "candidate_bootstrap", "candidate wheel is not one bounded regular file"
            )
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            observed_digest = f"sha256:{digest.hexdigest()}"
            if observed_digest != spec.sha256:
                raise QualificationError(
                    "candidate_bootstrap",
                    "candidate wheel digest does not match manifest",
                )
            handle.seek(0)
            try:
                archive = zipfile.ZipFile(handle)
            except zipfile.BadZipFile as exc:
                raise QualificationError(
                    "candidate_bootstrap", "candidate wheel is not a readable archive"
                ) from exc
            with archive:
                members = archive.infolist()
                if not members or len(members) > _MAX_CANDIDATE_WHEEL_MEMBERS:
                    raise QualificationError(
                        "candidate_bootstrap",
                        "candidate wheel member inventory is invalid",
                    )
                names = [member.filename for member in members]
                if len(names) != len(set(names)):
                    raise QualificationError(
                        "candidate_bootstrap",
                        "candidate wheel repeats an archive member",
                    )
                distribution, version, dist_info = _wheel_distribution(archive)
                source_prefix, package_prefix = _CANDIDATE_DISTRIBUTION_SOURCES[
                    distribution
                ]
                expected_sources = {
                    relative.removeprefix(source_prefix + "/"): payload
                    for relative, payload in archived.items()
                    if relative.startswith(source_prefix + "/")
                }
                observed_sources: dict[str, bytes] = {}
                package_archive_prefix = package_prefix + "/"
                total_size = 0
                for member in members:
                    if (
                        not _safe_wheel_member(member.filename)
                        or stat.S_ISLNK(member.external_attr >> 16)
                        or member.flag_bits & 0x1
                    ):
                        raise QualificationError(
                            "candidate_bootstrap",
                            "candidate wheel contains an unsafe member",
                        )
                    if member.file_size > _MAX_CANDIDATE_WHEEL_MEMBER_BYTES:
                        raise QualificationError(
                            "candidate_bootstrap", "candidate wheel member is too large"
                        )
                    total_size += member.file_size
                    if total_size > _MAX_CANDIDATE_WHEEL_BYTES:
                        raise QualificationError(
                            "candidate_bootstrap",
                            "candidate wheel expands beyond its limit",
                        )
                    if member.is_dir():
                        continue
                    if member.filename.endswith(".pth"):
                        raise QualificationError(
                            "candidate_bootstrap",
                            "candidate wheel contains an import hook",
                        )
                    if not (
                        member.filename.startswith(package_archive_prefix)
                        or member.filename.startswith(dist_info + "/")
                    ):
                        raise QualificationError(
                            "candidate_bootstrap",
                            "candidate wheel contains an unbound payload",
                        )
                    payload = archive.read(member)
                    if member.filename.startswith(package_archive_prefix):
                        relative = member.filename.removeprefix(package_archive_prefix)
                        observed_sources[relative] = payload
                    _write_candidate_member(
                        candidate_site.joinpath(*PurePosixPath(member.filename).parts),
                        payload,
                    )
                if observed_sources != expected_sources:
                    raise QualificationError(
                        "candidate_bootstrap",
                        "candidate wheel sources do not match authenticated source",
                    )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise QualificationError(
            "candidate_bootstrap", "candidate wheel changed while captured"
        )
    return CandidateWheelIdentity(
        distribution=distribution,
        version=version,
        filename=spec.path.name,
        sha256=spec.sha256,
    )


def _is_execution_source(relative: str) -> bool:
    if relative in {
        "scripts/qualification_precheck.py",
        "scripts/qualification_receipt_check.py",
        "scripts/qualification_source.py",
        "scripts/runtime_qualification.py",
    }:
        return True
    parts = Path(relative).parts
    if len(parts) >= 3 and parts[:2] == ("src", "invarlock"):
        return True
    return (
        len(parts) >= 5
        and parts[0] == "addins"
        and parts[2:4] == ("src", "invarlock_addins")
    )


def _source_archive_files(payload: bytes, *, source_commit: str) -> dict[str, bytes]:
    """Load the bounded execution-source inventory from one captured archive."""

    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        raise QualificationError(
            "configuration", "source bundle must be a readable Git tar archive"
        ) from exc
    with archive:
        if (archive.pax_headers or {}).get("comment") != source_commit:
            raise QualificationError(
                "configuration", "source bundle does not identify the declared commit"
            )
        members = archive.getmembers()
        if len(members) > _MAX_SOURCE_MEMBERS:
            raise QualificationError(
                "configuration", "source bundle contains too many entries"
            )
        selected: dict[str, bytes] = {}
        for member in members:
            if not _is_execution_source(member.name):
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise QualificationError(
                    "configuration", "source bundle execution-source path is unsafe"
                )
            if member.name in selected:
                raise QualificationError(
                    "configuration", "source bundle repeats an execution-source path"
                )
            if member.isdir():
                continue
            if not member.isfile() or member.size > _MAX_SOURCE_MEMBER_BYTES:
                raise QualificationError(
                    "configuration", "source bundle execution-source entry is invalid"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise QualificationError(
                    "configuration",
                    "source bundle execution-source entry is unreadable",
                )
            value = extracted.read(_MAX_SOURCE_MEMBER_BYTES + 1)
            if len(value) > _MAX_SOURCE_MEMBER_BYTES:
                raise QualificationError(
                    "configuration", "source bundle execution-source entry is too large"
                )
            selected[member.name] = value
    if not selected:
        raise QualificationError(
            "configuration", "source bundle execution inventory is empty"
        )
    return selected


def _execution_inventory_digest(files: dict[str, bytes]) -> str:
    inventory = hashlib.sha256()
    for relative in sorted(files):
        inventory.update(relative.encode("utf-8"))
        inventory.update(b"\0")
        inventory.update(hashlib.sha256(files[relative]).digest())
    return f"sha256:{inventory.hexdigest()}"


def _authenticated_execution_sources(
    path: Path,
    *,
    declared_digest: str,
    source_commit: str,
    root: Path,
) -> dict[str, bytes]:
    """Authenticate the exact Git bundle, then retain only executable source."""

    try:
        _identity, payload = authenticate_bundle(
            repository=root,
            commit=source_commit,
            bundle=path,
            bundle_sha256=declared_digest,
        )
    except SystemExit as exc:
        raise QualificationError("configuration", str(exc)) from exc
    return _source_archive_files(payload, source_commit=source_commit)


def _authenticate_source_bundle(
    path: Path,
    *,
    declared_digest: str,
    source_commit: str,
    root: Path,
) -> str:
    archived = _authenticated_execution_sources(
        path,
        declared_digest=declared_digest,
        source_commit=source_commit,
        root=root,
    )
    _bind_live_qualification_sources(archived, root=root)
    return _execution_inventory_digest(archived)


def _bind_live_qualification_sources(archived: dict[str, bytes], *, root: Path) -> None:
    """Require the live bootstrap code to be part of the authenticated source."""

    for relative in _LIVE_QUALIFICATION_SOURCES:
        expected = archived.get(relative)
        if expected is None:
            raise QualificationError(
                "configuration",
                f"authenticated source is missing live qualification file {relative}",
            )
        observed = _read_regular_bytes(
            root / relative,
            label=f"live qualification file {relative}",
        )
        if observed != expected:
            raise QualificationError(
                "configuration",
                f"live qualification file {relative} does not match authenticated source",
            )


@contextmanager
def _frozen_execution_context(
    inputs: QualificationInputs,
) -> Iterator[ExecutionContext]:
    """Stage authenticated helpers and candidate wheels in a closed import root."""

    root = _repository_root()
    archived = _authenticated_execution_sources(
        inputs.source_bundle,
        declared_digest=inputs.source_bundle_sha256,
        source_commit=inputs.source_commit,
        root=root,
    )
    _bind_live_qualification_sources(archived, root=root)
    if _execution_inventory_digest(archived) != inputs.source_execution_sha256:
        raise QualificationError(
            "configuration", "source bundle execution inventory changed before use"
        )
    manifest_sha256, wheel_specs = _candidate_wheel_specs(
        inputs.candidate_wheel_manifest
    )
    python_identity = _python_identity(inputs.python)
    with tempfile.TemporaryDirectory(prefix="invarlock-qualification-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        source_root = private_root / "source"
        working_directory = private_root / "work"
        candidate_site = private_root / "candidate-site"
        source_root.mkdir(mode=0o700)
        working_directory.mkdir(mode=0o700)
        candidate_site.mkdir(mode=0o700)
        engine_bin = private_root / "engine-bin"
        engine_bin.mkdir(mode=0o700)
        engine_entry = engine_bin / inputs.container_engine
        try:
            os.symlink(inputs.container_engine_path, engine_entry)
            if engine_entry.resolve(strict=True) != Path(inputs.container_engine_path):
                raise OSError("private container-engine entry resolved elsewhere")
        except OSError as exc:
            raise QualificationError(
                "configuration",
                "selected container engine could not be isolated",
            ) from exc
        for relative, payload in archived.items():
            destination = source_root.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        candidate_wheels = tuple(
            _capture_candidate_wheel(
                spec,
                archived=archived,
                candidate_site=candidate_site,
            )
            for spec in wheel_specs
        )
        distributions = [item.distribution for item in candidate_wheels]
        versions = {item.version for item in candidate_wheels}
        if "invarlock" not in distributions:
            raise QualificationError(
                "candidate_bootstrap", "candidate wheel set must include invarlock"
            )
        if len(distributions) != len(set(distributions)) or len(versions) != 1:
            raise QualificationError(
                "candidate_bootstrap",
                "candidate distributions must be unique and version-aligned",
            )
        child_path = str(engine_bin)
        context = ExecutionContext(
            source_root=source_root,
            working_directory=working_directory,
            child_path=child_path,
            candidate_site=candidate_site,
            candidate_manifest_sha256=manifest_sha256,
            candidate_wheels=tuple(
                sorted(candidate_wheels, key=lambda item: item.distribution)
            ),
            python_identity=python_identity,
        )
        probed = _successful_json(
            _run(
                [inputs.python, "-c", _CANDIDATE_PROBE],
                context=context,
                stage="candidate_probe",
            ),
            stage="candidate_probe",
            expected_format="invarlock/qualification-candidate-probe-v1",
        )
        expected_distributions = {
            item.distribution: item.version for item in context.candidate_wheels
        }
        if probed.get("distributions") != expected_distributions:
            raise QualificationError(
                "candidate_probe",
                "candidate probe did not discover the exact candidate wheel set",
            )
        yield context


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    stage: str = "configuration",
    max_bytes: int | None = None,
) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise QualificationError(stage, f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError(stage, f"{label} must be a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise QualificationError(stage, f"{label} exceeds its size limit")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            if max_bytes is not None and len(payload) > max_bytes:
                raise QualificationError(stage, f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise QualificationError(stage, f"{label} changed while it was read")
        return bytes(payload)
    except OSError as exc:
        raise QualificationError(stage, f"{label} could not be read") from exc
    finally:
        os.close(descriptor)


@contextmanager
def _captured_request(
    inputs: QualificationInputs,
) -> Iterator[tuple[QualificationInputs, bytes]]:
    """Keep preflight and execution on one private request-byte snapshot."""

    request_bytes = _read_regular_bytes(
        inputs.request,
        label="request",
        max_bytes=_MAX_REQUEST_BYTES,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="invarlock-qualification-request-"
        ) as temporary:
            snapshot = Path(temporary) / f"request{inputs.request.suffix}"
            descriptor = os.open(
                snapshot,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(request_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.chmod(snapshot, stat.S_IRUSR)
            yield replace(inputs, request=snapshot), request_bytes
    except OSError as exc:
        raise QualificationError(
            "configuration", "private request snapshot could not be created"
        ) from exc


def _fresh_destination(path: Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.name in {"", ".", ".."}:
        raise QualificationError("configuration", f"{label} must name a file")
    candidate = Path(os.path.abspath(os.fspath(supplied)))
    current = Path(candidate.anchor)
    for part in candidate.parent.parts[1:]:
        current /= part
        try:
            facts = current.lstat()
        except OSError as exc:
            raise QualificationError(
                "configuration", f"{label} parent must be an existing directory"
            ) from exc
        if stat.S_ISLNK(facts.st_mode) or not stat.S_ISDIR(facts.st_mode):
            raise QualificationError(
                "configuration", f"{label} parent must be a real directory"
            )
    if candidate.exists() or candidate.is_symlink():
        raise QualificationError("configuration", f"{label} already exists")
    return candidate


@contextmanager
def _opened_real_directory(path: Path, *, label: str) -> Iterator[int]:
    """Open an absolute directory through no-follow descriptor traversal."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.anchor, directory_flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise QualificationError(
            "configuration", f"{label} parent must remain a real directory"
        ) from exc
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def qualification_python(value: str | os.PathLike[str]) -> str:
    """Return an absolute executable path without dereferencing a venv symlink."""

    rendered = os.path.abspath(os.fspath(value))
    try:
        facts = os.stat(rendered)
    except OSError as exc:
        raise QualificationError(
            "configuration", "qualification Python executable is unavailable"
        ) from exc
    if not stat.S_ISREG(facts.st_mode) or not os.access(rendered, os.X_OK):
        raise QualificationError(
            "configuration", "qualification Python must be an executable file"
        )
    return rendered


def _digest(value: object, *, label: str, stage: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise QualificationError(stage, f"{label} must be a lowercase sha256 digest")
    return value


def _diagnostic(completed: subprocess.CompletedProcess[str]) -> object:
    def decode(value: str) -> object:
        bounded = value.strip()[:_MAX_DIAGNOSTIC_CHARS]
        if not bounded:
            return ""
        try:
            return json.loads(bounded)
        except json.JSONDecodeError:
            return bounded

    stdout = decode(completed.stdout)
    stderr = decode(completed.stderr)
    return {
        "returncode": completed.returncode,
        "output": stdout or stderr,
        "stdout": stdout,
        "stderr": stderr,
    }


def _repository_root() -> Path:
    root = Path(__file__).resolve(strict=True).parents[1]
    if not root.joinpath("src", "invarlock").is_dir():
        raise QualificationError(
            "configuration", "qualification driver is outside an InvarLock source tree"
        )
    return root


def _child_environment(context: ExecutionContext) -> dict[str, str]:
    """Build the closed environment consumed by the isolated Python bootstrap."""

    environment = {
        key: os.environ[key]
        for key in (*_CHILD_ENVIRONMENT_KEYS, *_PROVIDER_ENVIRONMENT_KEYS)
        if key in os.environ
    }
    environment.update(
        {
            "INVARLOCK_ALLOW_NETWORK": "0",
            "INVARLOCK_ALLOW_REMOTE_CODE": "0",
            "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PATH": context.child_path,
            "INVARLOCK_QUALIFICATION_CANDIDATE_SITE": str(context.candidate_site),
            "INVARLOCK_CANDIDATE_DISTRIBUTIONS": json.dumps(
                {item.distribution: item.version for item in context.candidate_wheels},
                separators=(",", ":"),
                sort_keys=True,
            ),
            "TOKENIZERS_PARALLELISM": environment.get(
                "TOKENIZERS_PARALLELISM", "false"
            ),
        }
    )
    return environment


def _isolated_python_command(
    command: list[str],
    *,
    identity: PythonIdentity,
    stage: str,
) -> list[str]:
    if not command:
        raise QualificationError(stage, "qualification command is empty")
    try:
        selected = Path(command[0]).resolve(strict=True)
    except OSError:
        return command
    if str(selected) != identity.resolved_path:
        return command
    arguments = command[1:]
    if len(arguments) >= 2 and arguments[0] == "-m":
        mode, target, remaining = "module", arguments[1], arguments[2:]
    elif len(arguments) >= 2 and arguments[0] == "-c":
        mode, target, remaining = "code", arguments[1], arguments[2:]
    elif arguments and not arguments[0].startswith("-"):
        mode, target, remaining = "path", arguments[0], arguments[1:]
    else:
        raise QualificationError(
            stage, "qualification Python invocation is not supported"
        )
    return [
        identity.path,
        "-I",
        "-S",
        "-c",
        _PYTHON_BOOTSTRAP,
        mode,
        target,
        *remaining,
    ]


def _run(
    command: list[str],
    *,
    context: ExecutionContext,
    stage: str,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    _assert_python_identity(context.python_identity, stage=stage)
    executed_command = _isolated_python_command(
        command,
        identity=context.python_identity,
        stage=stage,
    )
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            executed = subprocess.run(
                executed_command,
                cwd=context.working_directory,
                env=_child_environment(context),
                check=False,
                stdout=stdout,
                stderr=stderr,
                text=True,
                input=stdin,
            )
            _assert_python_identity(context.python_identity, stage=stage)
            stdout_size = stdout.tell()
            stderr_size = stderr.tell()
            if max(stdout_size, stderr_size) > _MAX_CHILD_OUTPUT_BYTES:
                raise QualificationError(
                    stage, f"{stage} command exceeded its output limit"
                )
            stdout.seek(0)
            stderr.seek(0)
            completed = subprocess.CompletedProcess(
                executed_command,
                executed.returncode,
                stdout.read().decode("utf-8", errors="replace"),
                stderr.read().decode("utf-8", errors="replace"),
            )
    except OSError as exc:
        raise QualificationError(
            stage,
            f"{stage} command could not be started",
            diagnostic=str(exc),
        ) from exc
    if completed.returncode != 0:
        raise QualificationError(
            stage,
            f"{stage} command failed",
            diagnostic=_diagnostic(completed),
        )
    return completed


def _successful_json(
    completed: subprocess.CompletedProcess[str],
    *,
    stage: str,
    expected_format: str,
) -> dict[str, object]:
    def object_without_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise QualificationError(
            stage,
            f"{stage} did not return JSON",
            diagnostic=completed.stdout[:_MAX_DIAGNOSTIC_CHARS],
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("ok") is not True
        or value.get("format_version") != expected_format
    ):
        raise QualificationError(
            stage,
            f"{stage} did not return the expected successful result format",
            diagnostic=value,
        )
    return value


def _evaluation_command(inputs: QualificationInputs) -> list[str]:
    command = [
        inputs.python,
        "-m",
        "invarlock",
        "evaluate",
        str(inputs.request),
        "--request-root",
        str(inputs.request_root),
        "--signing-key",
        str(inputs.signing_key),
        "--runtime-image",
        inputs.runtime_image,
        "--runtime-image-digest",
        inputs.runtime_image_digest,
        "--container-engine",
        inputs.container_engine,
        "--runtime-device",
        inputs.runtime_device,
    ]
    if inputs.runtime_cpus is not None:
        command.extend(("--runtime-cpus", inputs.runtime_cpus))
    if inputs.runtime_memory_mib is not None:
        command.extend(("--runtime-memory-mib", str(inputs.runtime_memory_mib)))
    if inputs.runtime_user is not None:
        command.extend(("--runtime-user", inputs.runtime_user))
    return command


def _planned_evidence(preflight: dict[str, object], *, request_root: Path) -> Path:
    output = preflight.get("output")
    if not isinstance(output, str) or not output:
        raise QualificationError(
            "preflight_binding", "preflight did not resolve an evidence destination"
        )
    path = Path(output)
    if not path.is_absolute():
        path = request_root / path
    return Path(os.path.abspath(os.fspath(path)))


def _qualification_precheck(
    inputs: QualificationInputs,
    *,
    preflight: dict[str, object],
    context: ExecutionContext,
) -> dict[str, object]:
    helper = context.source_root / "scripts" / "qualification_precheck.py"
    completed = _run(
        [
            inputs.python,
            str(helper),
            "--trust-profile",
            str(inputs.trust_profile),
            "--receipt",
            str(inputs.receipt),
        ],
        context=context,
        stage="trust_precheck",
        stdin=json.dumps(preflight, separators=(",", ":"), sort_keys=True),
    )
    return _successful_json(
        completed,
        stage="trust_precheck",
        expected_format="invarlock/qualification-precheck-v1",
    )


def _role_digests(value: object, *, label: str, stage: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"baseline", "subject"}:
        raise QualificationError(stage, f"{label} must bind baseline and subject")
    return {
        role: _digest(value.get(role), label=f"{label} {role}", stage=stage)
        for role in ("baseline", "subject")
    }


def _precheck_bindings(
    precheck: dict[str, object], *, receipt: Path
) -> dict[str, object]:
    stage = "trust_precheck"
    verifier_identity = precheck.get("verifier_identity")
    if not isinstance(verifier_identity, str) or not verifier_identity.strip():
        raise QualificationError(stage, "qualification verifier identity is missing")
    prechecked_receipt = precheck.get("receipt")
    if prechecked_receipt != str(receipt):
        raise QualificationError(
            stage, "qualification precheck changed the receipt destination"
        )
    return {
        "artifact_digests": _role_digests(
            precheck.get("artifact_digests"), label="artifact digests", stage=stage
        ),
        "evidence_signer_fingerprint": _digest(
            precheck.get("evidence_signer_fingerprint"),
            label="evidence signer fingerprint",
            stage=stage,
        ),
        "policy_digest": _digest(
            precheck.get("policy_digest"), label="policy digest", stage=stage
        ),
        "request_digest": _digest(
            precheck.get("request_digest"),
            label="normalized request digest",
            stage=stage,
        ),
        "runtime_digests": _role_digests(
            precheck.get("runtime_digests"), label="runtime digests", stage=stage
        ),
        "schedule_digest": _digest(
            precheck.get("schedule_digest"), label="schedule digest", stage=stage
        ),
        "trust_profile_digest": _digest(
            precheck.get("trust_profile_digest"),
            label="qualification trust-profile digest",
            stage=stage,
        ),
        "verifier_fingerprint": _digest(
            precheck.get("verifier_fingerprint"),
            label="verifier fingerprint",
            stage=stage,
        ),
        "verifier_identity": verifier_identity,
        "receipt": prechecked_receipt,
    }


def _preflight(
    inputs: QualificationInputs, *, context: ExecutionContext
) -> tuple[dict[str, object], dict[str, object]]:
    _runtime_source_binding(inputs, context=context)
    completed = _run(
        [*_evaluation_command(inputs), "--preflight", "--json"],
        context=context,
        stage="preflight",
    )
    _assert_container_engine_unchanged(inputs, stage="preflight")
    preflight = _successful_json(
        completed,
        stage="preflight",
        expected_format="invarlock/evaluation-preflight-v2",
    )
    if (
        _planned_evidence(preflight, request_root=inputs.request_root)
        != inputs.evidence
    ):
        raise QualificationError(
            "preflight_binding",
            "preflight evidence destination does not match --evidence",
        )
    precheck = _qualification_precheck(inputs, preflight=preflight, context=context)
    return preflight, _precheck_bindings(precheck, receipt=inputs.receipt)


def _assert_container_engine_unchanged(
    inputs: QualificationInputs, *, stage: str
) -> None:
    observed_engine = _sha256_regular_file(
        Path(inputs.container_engine_path),
        label="container engine",
        stage=stage,
    )
    if observed_engine != inputs.container_engine_sha256:
        raise QualificationError(stage, "container engine changed after configuration")


def _runtime_source_binding(
    inputs: QualificationInputs, *, context: ExecutionContext
) -> None:
    _assert_container_engine_unchanged(inputs, stage="runtime_source")
    completed = _run(
        [inputs.container_engine_path, "image", "inspect", inputs.runtime_image],
        context=context,
        stage="runtime_source",
    )
    _assert_container_engine_unchanged(inputs, stage="runtime_source")
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError(
            "runtime_source", "runtime image inspection did not return JSON"
        ) from exc
    if isinstance(decoded, list) and len(decoded) == 1 and isinstance(decoded[0], dict):
        image = decoded[0]
    elif isinstance(decoded, dict):
        image = decoded
    else:
        raise QualificationError(
            "runtime_source", "runtime image inspection returned an ambiguous result"
        )
    config = image.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        raise QualificationError(
            "runtime_source", "runtime image source labels are missing"
        )
    expected = {
        "dev.invarlock.source-bundle-sha256": inputs.source_bundle_sha256,
        "org.opencontainers.image.revision": inputs.source_commit,
    }
    for label, value in expected.items():
        if labels.get(label) != value:
            raise QualificationError(
                "runtime_source", f"runtime image {label} does not match frozen source"
            )


def _receipt_check(
    inputs: QualificationInputs,
    *,
    context: ExecutionContext,
    expected_pack_digest: str,
    expected_verifier_fingerprint: object,
) -> str:
    helper = context.source_root / "scripts" / "qualification_receipt_check.py"
    checked = _successful_json(
        _run(
            [
                inputs.python,
                str(helper),
                "--receipt",
                str(inputs.receipt),
                "--evidence",
                str(inputs.evidence),
                "--trust-profile",
                str(inputs.trust_profile),
            ],
            context=context,
            stage="receipt_verification",
        ),
        stage="receipt_verification",
        expected_format="invarlock/qualification-receipt-check-v1",
    )
    if checked.get("pack_manifest_digest") != expected_pack_digest:
        raise QualificationError(
            "receipt_verification",
            "signed receipt does not bind the verified evidence pack",
        )
    if checked.get("verifier_fingerprint") != expected_verifier_fingerprint:
        raise QualificationError(
            "receipt_verification",
            "signed receipt does not bind the prechecked verifier",
        )
    return _digest(
        checked.get("receipt_sha256"),
        label="verified receipt digest",
        stage="receipt_verification",
    )


def _canary_prerequisite(
    inputs: QualificationInputs, *, context: ExecutionContext
) -> dict[str, object]:
    """Reverify one signed canary pack against this exact runtime image."""

    if (
        inputs.canary_evidence is None
        or inputs.canary_receipt is None
        or inputs.canary_trust_profile is None
    ):
        raise QualificationError(
            "canary_prerequisite", "signed canary inputs are incomplete"
        )
    helper = context.source_root / "scripts" / "qualification_receipt_check.py"
    checked = _successful_json(
        _run(
            [
                inputs.python,
                str(helper),
                "--receipt",
                str(inputs.canary_receipt),
                "--evidence",
                str(inputs.canary_evidence),
                "--trust-profile",
                str(inputs.canary_trust_profile),
                "--expected-runtime-image-digest",
                inputs.runtime_image_digest,
                "--expected-request",
                str(inputs.request),
                "--expected-request-root",
                str(inputs.request_root),
                "--expected-runtime-device",
                inputs.runtime_device,
            ],
            context=context,
            stage="canary_prerequisite",
        ),
        stage="canary_prerequisite",
        expected_format="invarlock/qualification-receipt-check-v1",
    )
    if checked.get("runtime_image_digest") != inputs.runtime_image_digest:
        raise QualificationError(
            "canary_prerequisite",
            "signed canary does not bind the exact qualification image",
        )
    compatibility = checked.get("compatibility")
    if not isinstance(compatibility, dict):
        raise QualificationError(
            "canary_prerequisite",
            "signed canary does not bind target provider/task compatibility",
        )
    return {
        "pack_manifest_digest": _digest(
            checked.get("pack_manifest_digest"),
            label="canary pack-manifest digest",
            stage="canary_prerequisite",
        ),
        "receipt_sha256": _digest(
            checked.get("receipt_sha256"),
            label="canary receipt digest",
            stage="canary_prerequisite",
        ),
        "runtime_image_digest": inputs.runtime_image_digest,
        "compatibility": compatibility,
    }


def _base_result(
    inputs: QualificationInputs,
    *,
    context: ExecutionContext,
    request_sha256: str,
    verification_inputs: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "ok": True,
        "mode": inputs.mode,
        "source": {
            "commit": inputs.source_commit,
            "bundle_sha256": inputs.source_bundle_sha256,
            "execution_tree_sha256": inputs.source_execution_sha256,
        },
        "runtime": {
            "image": inputs.runtime_image,
            "image_digest": inputs.runtime_image_digest,
        },
        "container_engine": {
            "name": inputs.container_engine,
            "sha256": inputs.container_engine_sha256,
        },
        "host_runtime": {
            "candidate_manifest_sha256": context.candidate_manifest_sha256,
            "candidate_wheels": [item.summary() for item in context.candidate_wheels],
            "python": context.python_identity.summary(),
        },
        "request_sha256": request_sha256,
        "trust_profile_digest": verification_inputs["trust_profile_digest"],
        "verification_inputs": verification_inputs,
    }


def _readiness_captured(
    inputs: QualificationInputs,
    *,
    request_bytes: bytes,
    context: ExecutionContext,
) -> dict[str, object]:
    if inputs.report is not None:
        _fresh_destination(inputs.report, label="qualification report")
    canary = _canary_prerequisite(inputs, context=context)
    preflight, verification_inputs = _preflight(inputs, context=context)
    result = _base_result(
        inputs,
        context=context,
        request_sha256=_sha256_bytes(request_bytes),
        verification_inputs=verification_inputs,
    )
    result.update(
        {
            "stage": "ready",
            "evidence": str(inputs.evidence),
            "schedule_digest": preflight.get("schedule_digest"),
            "policy_digest": preflight.get("policy_digest"),
            "runtime_image_digests": preflight.get("runtime_image_digests"),
            "canary": canary,
        }
    )
    return result


def readiness(inputs: QualificationInputs) -> dict[str, object]:
    """Perform execution-free qualification and bind the resolved identities."""

    with (
        _captured_request(inputs) as (captured, request_bytes),
        _frozen_execution_context(captured) as context,
    ):
        return _readiness_captured(
            captured,
            request_bytes=request_bytes,
            context=context,
        )


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    """Publish complete private bytes atomically without replacing any entry."""

    destination = _fresh_destination(path, label="qualification summary")
    with _opened_real_directory(
        destination.parent, label="qualification summary"
    ) as parent_descriptor:
        temporary_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    temporary_name,
                    destination.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise QualificationError(
                        "summary", "qualification summary already exists"
                    ) from exc
                raise QualificationError(
                    "summary", "qualification summary could not be published"
                ) from exc
            try:
                os.fsync(parent_descriptor)
            except OSError:
                pass
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def _verify_binding_unit(
    verified: dict[str, object], *, expected: dict[str, object], receipt: Path
) -> None:
    strict_success = {
        "assurance_status": "verified",
        "authenticity": "pinned",
        "errors": [],
        "integrity_ok": True,
        "policy_verdict": "pass",
        "reports_verified": True,
        "verification_scope": "paired_comparison",
        "warnings": [],
    }
    for field, required in strict_success.items():
        if verified.get(field) != required:
            raise QualificationError(
                "verification_binding",
                f"verification did not return strict {field}",
            )
    anchors = verified.get("anchors")
    if not isinstance(anchors, dict):
        raise QualificationError(
            "verification_binding", "verification result did not return trust anchors"
        )
    comparisons = {
        "artifact digests": (
            anchors.get("artifact_digests"),
            expected["artifact_digests"],
        ),
        "policy digest": (anchors.get("policy_digest"), expected["policy_digest"]),
        "runtime digests": (
            anchors.get("runtime_digests"),
            expected["runtime_digests"],
        ),
        "schedule digest": (
            anchors.get("schedule_digest"),
            expected["schedule_digest"],
        ),
        "evidence signer fingerprint": (
            anchors.get("signer_fingerprint"),
            expected["evidence_signer_fingerprint"],
        ),
        "normalized request digest": (
            verified.get("request_digest"),
            expected["request_digest"],
        ),
        "observed evidence signer fingerprint": (
            verified.get("signer_fingerprint"),
            expected["evidence_signer_fingerprint"],
        ),
        "trust-profile digest": (
            verified.get("trust_profile_digest"),
            expected["trust_profile_digest"],
        ),
        "verifier fingerprint": (
            verified.get("verifier_fingerprint"),
            expected["verifier_fingerprint"],
        ),
        "verifier identity": (
            verified.get("verifier_identity"),
            expected["verifier_identity"],
        ),
        "receipt destination": (verified.get("signed_receipt"), receipt.name),
    }
    for label, (observed, required) in comparisons.items():
        if observed != required:
            raise QualificationError(
                "verification_binding",
                f"verification did not bind the prechecked {label}",
            )


def _run_captured(
    inputs: QualificationInputs,
    *,
    request_bytes: bytes,
    context: ExecutionContext,
) -> dict[str, object]:
    assert inputs.summary is not None
    _fresh_destination(inputs.summary, label="qualification summary")
    if inputs.report is not None:
        _fresh_destination(inputs.report, label="qualification report")
    request_sha256 = _sha256_bytes(request_bytes)
    canary = (
        None
        if inputs.mode == "canary"
        else _canary_prerequisite(inputs, context=context)
    )
    preflight_result, verification_inputs = _preflight(inputs, context=context)
    _runtime_source_binding(inputs, context=context)

    completed_evaluation = _run(
        [*_evaluation_command(inputs), "--json"],
        context=context,
        stage="evaluation",
    )
    _assert_container_engine_unchanged(inputs, stage="evaluation")
    evaluated = _successful_json(
        completed_evaluation,
        stage="evaluation",
        expected_format="invarlock/evaluation-result-v1",
    )
    published = evaluated.get("evidence")
    published_path = Path(published) if isinstance(published, str) else None
    if published_path is not None and not published_path.is_absolute():
        published_path = inputs.request_root / published_path
    if (
        published_path is None
        or Path(os.path.abspath(os.fspath(published_path))) != inputs.evidence
    ):
        raise QualificationError(
            "evaluation_binding",
            "evaluation did not publish the exact requested evidence destination",
        )
    pack_digest = _digest(
        evaluated.get("pack_manifest_digest"),
        label="published pack-manifest digest",
        stage="evaluation_binding",
    )
    if _sha256_bytes(
        _read_regular_bytes(
            inputs.request,
            label="captured request",
            stage="evaluation_binding",
            max_bytes=_MAX_REQUEST_BYTES,
        )
    ) != _sha256_bytes(request_bytes):
        raise QualificationError(
            "evaluation_binding", "captured request changed during qualification"
        )

    verified = _successful_json(
        _run(
            [
                inputs.python,
                "-m",
                "invarlock",
                "verify",
                str(inputs.evidence),
                "--trust-profile",
                str(inputs.trust_profile),
                "--receipt",
                str(inputs.receipt),
                "--json",
            ],
            context=context,
            stage="verification",
        ),
        stage="verification",
        expected_format="invarlock/evidence-pack-verify-v1",
    )
    verified_pack = _digest(
        verified.get("pack_manifest_digest"),
        label="verified pack-manifest digest",
        stage="verification_binding",
    )
    if verified_pack != pack_digest:
        raise QualificationError(
            "verification_binding",
            "verification did not bind the pack published by evaluation",
        )
    _verify_binding_unit(verified, expected=verification_inputs, receipt=inputs.receipt)
    receipt_digest = _receipt_check(
        inputs,
        context=context,
        expected_pack_digest=pack_digest,
        expected_verifier_fingerprint=verification_inputs["verifier_fingerprint"],
    )

    report_identity: dict[str, object] | None = None
    if inputs.report is not None:
        rendered = _successful_json(
            _run(
                [
                    inputs.python,
                    "-m",
                    "invarlock",
                    "report",
                    str(inputs.evidence),
                    "--html",
                    str(inputs.report),
                    "--explain",
                    "--json",
                ],
                context=context,
                stage="report",
            ),
            stage="report",
            expected_format="invarlock/evidence-report-v1",
        )
        if rendered.get("pack_manifest_digest") != pack_digest:
            raise QualificationError(
                "report_binding",
                "rendered report does not bind the verified evidence pack",
            )
        if rendered.get("html") != str(inputs.report):
            raise QualificationError(
                "report_binding", "renderer changed the report destination"
            )
        report_identity = {
            "pack_manifest_digest": pack_digest,
            "sha256": _sha256_bytes(
                _read_regular_bytes(
                    inputs.report, label="qualification report", stage="report"
                )
            ),
        }
        repeated_receipt_digest = _receipt_check(
            inputs,
            context=context,
            expected_pack_digest=pack_digest,
            expected_verifier_fingerprint=verification_inputs["verifier_fingerprint"],
        )
        if repeated_receipt_digest != receipt_digest:
            raise QualificationError(
                "report_binding", "receipt changed while the report was rendered"
            )

    result = _base_result(
        inputs,
        context=context,
        request_sha256=request_sha256,
        verification_inputs=verification_inputs,
    )
    result.update(
        {
            "stage": "complete",
            "evidence": {"pack_manifest_digest": pack_digest},
            "preflight": {
                "artifact_digests": preflight_result.get("artifact_digests"),
                "evidence_signer_fingerprint": preflight_result.get(
                    "evidence_signer_fingerprint"
                ),
                "policy_digest": preflight_result.get("policy_digest"),
                "runtime_image_digests": preflight_result.get("runtime_image_digests"),
                "schedule_digest": preflight_result.get("schedule_digest"),
            },
            "receipt": {"sha256": receipt_digest},
        }
    )
    if report_identity is not None:
        result["report"] = report_identity
    if canary is not None:
        result["canary"] = canary
    _atomic_no_clobber(inputs.summary, _json_bytes(result))
    return result


def run(inputs: QualificationInputs) -> dict[str, object]:
    """Run evaluate, independently verify, optionally report, and summarize."""

    with (
        _captured_request(inputs) as (captured, request_bytes),
        _frozen_execution_context(captured) as context,
    ):
        return _run_captured(
            captured,
            request_bytes=request_bytes,
            context=context,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("readiness", "run", "canary"):
        command = subparsers.add_parser(mode)
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--signing-key", type=Path, required=True)
        command.add_argument("--runtime-image", required=True)
        command.add_argument("--runtime-image-digest", required=True)
        command.add_argument("--evidence", type=Path, required=True)
        command.add_argument("--trust-profile", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if mode != "canary":
            command.add_argument("--canary-evidence", type=Path, required=True)
            command.add_argument("--canary-receipt", type=Path, required=True)
            command.add_argument("--canary-trust-profile", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--source-bundle", type=Path, required=True)
        command.add_argument("--source-bundle-sha256", required=True)
        command.add_argument("--candidate-wheel-manifest", type=Path, required=True)
        command.add_argument(
            "--container-engine", choices=("docker", "podman"), default="docker"
        )
        command.add_argument("--runtime-device", default="cuda")
        command.add_argument("--runtime-cpus")
        command.add_argument("--runtime-memory-mib", type=int)
        command.add_argument("--runtime-user")
        command.add_argument("--report", type=Path)
        if mode in {"run", "canary"}:
            command.add_argument("--summary", type=Path, required=True)
    return parser


def _inputs(arguments: argparse.Namespace) -> QualificationInputs:
    if _SOURCE_COMMIT.fullmatch(arguments.source_commit) is None:
        raise QualificationError(
            "configuration",
            "source commit must be 40-64 lowercase hexadecimal characters",
        )
    source_bundle = _digest(
        arguments.source_bundle_sha256,
        label="source-bundle digest",
        stage="configuration",
    )
    source_bundle_path = Path(os.path.abspath(os.fspath(arguments.source_bundle)))
    source_execution = _authenticate_source_bundle(
        source_bundle_path,
        declared_digest=source_bundle,
        source_commit=arguments.source_commit,
        root=_repository_root(),
    )
    runtime_digest = _digest(
        arguments.runtime_image_digest,
        label="runtime-image digest",
        stage="configuration",
    )
    if not (
        arguments.runtime_image == runtime_digest
        or arguments.runtime_image.endswith("@" + runtime_digest)
    ):
        raise QualificationError(
            "configuration",
            "runtime image must equal or embed its immutable qualification digest",
        )
    request = Path(os.path.abspath(os.fspath(arguments.request)))
    try:
        request_root = request.parent.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "configuration", "request parent must be an existing directory"
        ) from exc
    evidence_candidate = Path(os.path.abspath(os.fspath(arguments.evidence)))
    trust_profile = Path(os.path.abspath(os.fspath(arguments.trust_profile)))
    canary_evidence_argument = getattr(arguments, "canary_evidence", None)
    canary_receipt_argument = getattr(arguments, "canary_receipt", None)
    canary_trust_profile_argument = getattr(arguments, "canary_trust_profile", None)
    canary_evidence = (
        Path(os.path.abspath(os.fspath(canary_evidence_argument)))
        if canary_evidence_argument is not None
        else None
    )
    canary_receipt = (
        Path(os.path.abspath(os.fspath(canary_receipt_argument)))
        if canary_receipt_argument is not None
        else None
    )
    canary_trust_profile = (
        Path(os.path.abspath(os.fspath(canary_trust_profile_argument)))
        if canary_trust_profile_argument is not None
        else None
    )
    receipt_candidate = Path(os.path.abspath(os.fspath(arguments.receipt)))
    report_candidate = (
        Path(os.path.abspath(os.fspath(arguments.report)))
        if arguments.report is not None
        else None
    )
    summary_argument = getattr(arguments, "summary", None)
    summary_candidate = (
        Path(os.path.abspath(os.fspath(summary_argument)))
        if summary_argument is not None
        else None
    )
    output_candidates = [
        receipt_candidate,
        *(path for path in (report_candidate, summary_candidate) if path is not None),
    ]
    if len(set(output_candidates)) != len(output_candidates):
        raise QualificationError(
            "configuration", "receipt, report, and summary must be distinct outputs"
        )
    for output in output_candidates:
        try:
            output.relative_to(evidence_candidate)
        except ValueError:
            continue
        raise QualificationError(
            "configuration",
            "qualification outputs must remain outside the immutable evidence pack",
        )
    evidence = _fresh_destination(
        evidence_candidate,
        label="evidence destination",
    )
    receipt = _fresh_destination(
        receipt_candidate,
        label="verification receipt",
    )
    report = (
        _fresh_destination(report_candidate, label="qualification report")
        if report_candidate is not None
        else None
    )
    summary = (
        _fresh_destination(summary_candidate, label="qualification summary")
        if summary_candidate is not None
        else None
    )
    engine_path = shutil.which(arguments.container_engine)
    if engine_path is None:
        raise QualificationError(
            "configuration", "selected container engine is unavailable"
        )
    container_engine_path = str(Path(engine_path).resolve(strict=True))
    container_engine_sha256 = _sha256_regular_file(
        Path(container_engine_path),
        label="container engine",
        stage="configuration",
    )
    return QualificationInputs(
        mode=arguments.mode,
        python=qualification_python(arguments.python),
        request=request,
        request_root=request_root,
        signing_key=Path(os.path.abspath(os.fspath(arguments.signing_key))),
        runtime_image=arguments.runtime_image,
        runtime_image_digest=runtime_digest,
        evidence=evidence,
        trust_profile=trust_profile,
        receipt=receipt,
        canary_evidence=canary_evidence,
        canary_receipt=canary_receipt,
        canary_trust_profile=canary_trust_profile,
        source_commit=arguments.source_commit,
        source_bundle=source_bundle_path,
        source_bundle_sha256=source_bundle,
        source_execution_sha256=source_execution,
        candidate_wheel_manifest=Path(
            os.path.abspath(os.fspath(arguments.candidate_wheel_manifest))
        ),
        container_engine=arguments.container_engine,
        container_engine_path=container_engine_path,
        container_engine_sha256=container_engine_sha256,
        runtime_device=arguments.runtime_device,
        runtime_cpus=arguments.runtime_cpus,
        runtime_memory_mib=arguments.runtime_memory_mib,
        runtime_user=arguments.runtime_user,
        report=report,
        summary=summary,
    )


def _fail(error: QualificationError, *, mode: str | None) -> NoReturn:
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "ok": False,
        "mode": mode,
        "stage": error.stage,
        "errors": [error.message],
    }
    if error.diagnostic is not None:
        payload["diagnostic"] = error.diagnostic
    print(_json_bytes(payload).decode("utf-8"), end="", file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        inputs = _inputs(arguments)
        result = readiness(inputs) if inputs.mode == "readiness" else run(inputs)
    except QualificationError as exc:
        _fail(exc, mode=getattr(arguments, "mode", None))
    print(_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
