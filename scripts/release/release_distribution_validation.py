"""Distribution archive validation for release preflight."""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import os
import re
import stat
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

PACKAGE_NAME = "invarlock"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_METADATA_BYTES = 1_048_576
MAX_ARCHIVE_MEMBERS = 50_000
RUNTIME_PACKAGE_SUFFIXES = frozenset({".json", ".py", ".pyi", ".yaml", ".yml"})
RUNTIME_PACKAGE_FILENAMES = frozenset({"py.typed"})
IGNORED_RUNTIME_PACKAGE_FILENAMES = frozenset({".DS_Store"})
IMPORT_AFFECTING_SUFFIXES = frozenset({".pth"})
EXECUTABLE_PAYLOAD_SUFFIXES = frozenset(
    {".dll", ".dylib", ".exe", ".pyd", ".py", ".pyc", ".pyo", ".so"}
)
RequirementIdentity = tuple[str, tuple[str, ...], str, str, str]


class ReleasePreflightError(RuntimeError):
    """Raised when a candidate fails a release preflight invariant."""


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


@dataclass(frozen=True)
class ReleasePreflightConfig:
    repo_root: Path
    release_sha: str
    expected_version: str
    dist_dir: Path
    hash_manifest: Path


@dataclass(frozen=True)
class DistributionArtifacts:
    wheel: Path
    sdist: Path
    hashes: dict[str, str]


@dataclass(frozen=True)
class DistributionValidationSpec:
    """Checkout and archive identity for one first-party distribution."""

    project_root: Path
    distribution_name: str
    version: str
    package_path: str

    @property
    def normalized_name(self) -> str:
        return re.sub(r"[-_.]+", "_", self.distribution_name).lower()

    @property
    def dist_info_root(self) -> str:
        return f"{self.normalized_name}-{self.version}.dist-info"

    @property
    def sdist_root(self) -> str:
        return f"{self.normalized_name}-{self.version}"

    @property
    def egg_info_root(self) -> str:
        return f"src/{self.normalized_name}.egg-info"


@dataclass(frozen=True)
class CheckoutSource:
    size: int
    sha256: str


@dataclass(frozen=True)
class ExpectedPackageMetadata:
    """Install-relevant metadata derived from one exact pyproject.toml."""

    name: str
    version: str
    requires_python: str | None
    requires_dist: tuple[RequirementIdentity, ...]
    provides_extra: tuple[str, ...]


@dataclass(frozen=True)
class InstalledWheelImport:
    module_file: Path
    module_version: str
    distribution_name: str
    distribution_version: str
    distribution_root: Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReleasePreflightError(f"{label} must be a regular file")


def _require_executable_file(path: Path, label: str) -> None:
    if (
        not path.exists()
        or not path.resolve().is_file()
        or not os.access(path, os.X_OK)
    ):
        raise ReleasePreflightError(f"{label} must be an executable file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_archive_member_name(name: str) -> bool:
    member = PurePosixPath(name)
    return (
        bool(name)
        and not member.is_absolute()
        and "\\" not in name
        and all(part not in {"", ".", ".."} for part in member.parts)
    )


def _parse_package_metadata(
    raw: bytes,
    *,
    label: str,
    expected: ExpectedPackageMetadata,
) -> None:
    if len(raw) > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} metadata is too large")
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if message.defects:
        raise ReleasePreflightError(f"{label} metadata is malformed")
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if (
        len(names) != 1
        or re.sub(r"[-_.]+", "-", str(names[0])).casefold()
        != re.sub(r"[-_.]+", "-", expected.name).casefold()
    ):
        raise ReleasePreflightError(
            f"{label} metadata package name does not match {expected.name}"
        )
    if len(versions) != 1 or str(versions[0]) != expected.version:
        raise ReleasePreflightError(
            f"{label} metadata version does not match expected version"
        )
    requires_python = message.get_all("Requires-Python", [])
    if expected.requires_python is None:
        if requires_python:
            raise ReleasePreflightError(
                f"{label} metadata Requires-Python does not match checkout"
            )
    elif (
        len(requires_python) != 1
        or _canonical_specifier(
            str(requires_python[0]), label=f"{label} Requires-Python"
        )
        != expected.requires_python
    ):
        raise ReleasePreflightError(
            f"{label} metadata Requires-Python does not match checkout"
        )
    observed_requirements = tuple(
        sorted(
            _requirement_identity(str(value), label=f"{label} Requires-Dist")
            for value in message.get_all("Requires-Dist", [])
        )
    )
    if observed_requirements != expected.requires_dist:
        raise ReleasePreflightError(
            f"{label} metadata Requires-Dist does not match checkout"
        )
    observed_extras = tuple(
        sorted(
            _canonical_extra(str(value), label=f"{label} Provides-Extra")
            for value in message.get_all("Provides-Extra", [])
        )
    )
    if observed_extras != expected.provides_extra:
        raise ReleasePreflightError(
            f"{label} metadata Provides-Extra does not match checkout"
        )


def _canonical_specifier(value: str, *, label: str) -> str:
    try:
        return str(SpecifierSet(value))
    except InvalidSpecifier as exc:
        raise ReleasePreflightError(f"{label} is invalid") from exc


def _canonical_extra(value: str, *, label: str) -> str:
    if not value or value != value.strip():
        raise ReleasePreflightError(f"{label} is invalid")
    normalized = str(canonicalize_name(value))
    if not normalized:
        raise ReleasePreflightError(f"{label} is invalid")
    return normalized


def _requirement_identity(
    value: str, *, label: str, extra: str | None = None
) -> RequirementIdentity:
    try:
        requirement = Requirement(value)
        marker = requirement.marker
        if extra is not None:
            extra_marker = f'extra == "{extra}"'
            marker = Marker(
                extra_marker if marker is None else f"({marker}) and {extra_marker}"
            )
    except (InvalidMarker, InvalidRequirement) as exc:
        raise ReleasePreflightError(f"{label} is invalid") from exc
    return (
        str(canonicalize_name(requirement.name)),
        tuple(sorted(str(canonicalize_name(item)) for item in requirement.extras)),
        str(requirement.specifier),
        requirement.url or "",
        str(marker) if marker is not None else "",
    )


def _project_table(project_root: Path) -> dict[str, object]:
    pyproject = project_root / "pyproject.toml"
    _require_regular_file(pyproject, "checkout pyproject.toml")
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = payload["project"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ReleasePreflightError("checkout project metadata is unreadable") from exc
    if not isinstance(project, dict):
        raise ReleasePreflightError("checkout project metadata is unreadable")
    return project


def _expected_package_metadata(project_root: Path) -> ExpectedPackageMetadata:
    project = _project_table(project_root)
    name = project.get("name")
    version = project.get("version")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
    ):
        raise ReleasePreflightError("checkout project identity is invalid")
    requires_python_value = project.get("requires-python")
    if requires_python_value is not None and not isinstance(requires_python_value, str):
        raise ReleasePreflightError("checkout requires-python is invalid")
    requires_python = (
        _canonical_specifier(
            requires_python_value, label="checkout project requires-python"
        )
        if requires_python_value is not None
        else None
    )
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(value, str) for value in dependencies
    ):
        raise ReleasePreflightError("checkout project dependencies are invalid")
    requirements = [
        _requirement_identity(value, label="checkout project dependency")
        for value in dependencies
    ]
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ReleasePreflightError(
            "checkout project optional dependencies are invalid"
        )
    extras: list[str] = []
    for extra_name, values in optional.items():
        if not isinstance(extra_name, str):
            raise ReleasePreflightError(
                "checkout project optional dependency name is invalid"
            )
        extra = _canonical_extra(
            extra_name, label="checkout project optional dependency name"
        )
        if extra in extras:
            raise ReleasePreflightError(
                "checkout project optional dependency names are ambiguous"
            )
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ReleasePreflightError(
                "checkout project optional dependencies are invalid"
            )
        extras.append(extra)
        requirements.extend(
            _requirement_identity(
                value,
                label="checkout project optional dependency",
                extra=extra,
            )
            for value in values
        )
    return ExpectedPackageMetadata(
        name=name,
        version=version,
        requires_python=requires_python,
        requires_dist=tuple(sorted(requirements)),
        provides_extra=tuple(sorted(extras)),
    )


def _entry_point_group(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ReleasePreflightError(f"{label} entry points must be a table")
    entries: dict[str, str] = {}
    for name, target in value.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(target, str)
            or not target.strip()
        ):
            raise ReleasePreflightError(f"{label} entry point is invalid")
        entries[name] = target.strip()
    return entries


def read_distribution_project(project_root: Path) -> tuple[str, str]:
    """Read one first-party distribution name and version from its checkout."""

    metadata = _expected_package_metadata(project_root)
    return metadata.name, metadata.version


def _expected_entry_points(repo_root: Path) -> dict[str, dict[str, str]]:
    pyproject = repo_root / "pyproject.toml"
    _require_regular_file(pyproject, "checkout pyproject.toml")
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleasePreflightError("checkout pyproject.toml is unreadable") from exc
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ReleasePreflightError("checkout pyproject.toml has no project table")
    expected: dict[str, dict[str, str]] = {}
    for project_key, group_name in (
        ("scripts", "console_scripts"),
        ("gui-scripts", "gui_scripts"),
    ):
        if project_key in project:
            group = _entry_point_group(
                project[project_key], label=f"project.{project_key}"
            )
            if group:
                expected[group_name] = group
    if "entry-points" in project:
        custom_groups = project["entry-points"]
        if not isinstance(custom_groups, dict):
            raise ReleasePreflightError("project.entry-points must be a table")
        for group_name, entries in custom_groups.items():
            if not isinstance(group_name, str) or not group_name.strip():
                raise ReleasePreflightError("project entry-point group is invalid")
            group = _entry_point_group(
                entries, label=f"project.entry-points.{group_name}"
            )
            if group:
                expected[group_name] = group
    return expected


def _parse_entry_points(raw: bytes, *, label: str) -> dict[str, dict[str, str]]:
    if len(raw) > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} entry points are too large")
    parser = _CaseSensitiveConfigParser(interpolation=None, strict=True)
    try:
        parser.read_string(raw.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise ReleasePreflightError(f"{label} entry points are unreadable") from exc
    if parser.defaults():
        raise ReleasePreflightError(f"{label} entry points must not use defaults")
    entries: dict[str, dict[str, str]] = {}
    for group_name in parser.sections():
        group: dict[str, str] = {}
        for name, target in parser.items(group_name, raw=True):
            if not name.strip() or not target.strip() or name in group:
                raise ReleasePreflightError(f"{label} entry point is invalid")
            group[name] = target.strip()
        entries[group_name] = group
    return entries


def _validate_entry_points(
    raw: bytes | None,
    *,
    expected: dict[str, dict[str, str]],
    label: str,
) -> None:
    observed = _parse_entry_points(raw, label=label) if raw is not None else {}
    if observed != expected:
        raise ReleasePreflightError(f"{label} entry points do not match checkout")


def _zip_member_sha256(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).decode("ascii").rstrip("=")


def _validate_wheel_record(
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    record_name: str,
) -> None:
    record = archive.getinfo(record_name)
    if record.file_size == 0 or record.file_size > MAX_METADATA_BYTES:
        raise ReleasePreflightError("wheel RECORD is missing or invalid")
    try:
        rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReleasePreflightError("wheel RECORD is unreadable") from exc
    archive_files = {
        member.filename: member for member in members if not member.is_dir()
    }
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not _safe_archive_member_name(row[0]) or row[0] in recorded:
            raise ReleasePreflightError("wheel RECORD has invalid or duplicate entries")
        recorded[row[0]] = (row[1], row[2])
    if set(recorded) != set(archive_files):
        raise ReleasePreflightError("wheel RECORD does not cover the wheel contents")
    for name, member in archive_files.items():
        digest, size = recorded[name]
        if name == record_name:
            if digest or size:
                raise ReleasePreflightError("wheel RECORD entry must not self-hash")
            continue
        if not digest.startswith("sha256=") or not size.isdigit():
            raise ReleasePreflightError("wheel RECORD entry is missing sha256 or size")
        if size != str(member.file_size) or digest.removeprefix(
            "sha256="
        ) != _zip_member_sha256(archive, member):
            raise ReleasePreflightError("wheel RECORD does not match wheel contents")


def _is_import_affecting_path(name: str) -> bool:
    return Path(name).suffix.lower() in IMPORT_AFFECTING_SUFFIXES


def _tar_member_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReleasePreflightError("sdist source member is unreadable")
    digest = hashlib.sha256()
    for block in iter(lambda: extracted.read(1_048_576), b""):
        digest.update(block)
    return digest.hexdigest()


def _directory_is_needed(directory: str, files: dict[str, CheckoutSource]) -> bool:
    normalized = directory.rstrip("/")
    return not normalized or any(
        path == normalized or path.startswith(f"{normalized}/") for path in files
    )


def _validate_egg_info_member(member: tarfile.TarInfo, relative: str) -> None:
    if member.isdir():
        return
    if member.size > MAX_METADATA_BYTES:
        raise ReleasePreflightError("sdist egg-info member is too large")
    suffix = Path(relative).suffix.lower()
    if suffix in IMPORT_AFFECTING_SUFFIXES or suffix in EXECUTABLE_PAYLOAD_SUFFIXES:
        raise ReleasePreflightError(
            "sdist egg-info must not contain executable or import payloads"
        )


def _validate_generated_sdist_setup_cfg(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> None:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReleasePreflightError("generated sdist setup.cfg is unreadable")
    raw = extracted.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ReleasePreflightError("generated sdist setup.cfg is too large")
    parser = _CaseSensitiveConfigParser(interpolation=None, strict=True)
    try:
        parser.read_string(raw.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise ReleasePreflightError("generated sdist setup.cfg is invalid") from exc
    expected = {"egg_info": {"tag_build": "", "tag_date": "0"}}
    observed = {
        section: dict(parser.items(section, raw=True)) for section in parser.sections()
    }
    if parser.defaults() or observed != expected:
        raise ReleasePreflightError(
            "generated sdist setup.cfg contains unsupported build configuration"
        )


def _package_ancestor(path: str, package_path: str) -> bool:
    normalized = path.rstrip("/")
    return bool(normalized) and package_path.startswith(f"{normalized}/")


def _checkout_package_files(
    spec: DistributionValidationSpec,
) -> dict[str, CheckoutSource]:
    if not _safe_archive_member_name(spec.package_path):
        raise ReleasePreflightError("distribution package path is invalid")
    source_root = spec.project_root.joinpath(
        "src", *PurePosixPath(spec.package_path).parts
    )
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReleasePreflightError("checkout runtime package directory is missing")
    sources: dict[str, CheckoutSource] = {}
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts or path.name in (
            IGNORED_RUNTIME_PACKAGE_FILENAMES
        ):
            continue
        if path.is_symlink():
            raise ReleasePreflightError(
                "checkout runtime package must not contain links"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleasePreflightError(
                "checkout runtime package contains a non-regular file"
            )
        if (
            path.suffix not in RUNTIME_PACKAGE_SUFFIXES
            and path.name not in RUNTIME_PACKAGE_FILENAMES
        ):
            raise ReleasePreflightError(
                "checkout runtime package contains an unexpected file"
            )
        _require_regular_file(path, "checkout runtime package file")
        sources[relative.as_posix()] = CheckoutSource(
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
    if not sources:
        raise ReleasePreflightError("checkout runtime package has no source files")
    return sources


def _wheel_member_matches(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    expected: CheckoutSource,
) -> bool:
    expected_digest = (
        base64.urlsafe_b64encode(bytes.fromhex(expected.sha256))
        .decode("ascii")
        .rstrip("=")
    )
    return (
        member.file_size == expected.size
        and _zip_member_sha256(archive, member) == expected_digest
    )


def _read_zip_metadata(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, label: str
) -> bytes:
    if member.file_size > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} is too large")
    return archive.read(member)


def _read_tar_metadata(
    archive: tarfile.TarFile, member: tarfile.TarInfo, *, label: str
) -> bytes:
    if member.size > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} is too large")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReleasePreflightError(f"{label} is unreadable")
    raw = extracted.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} is too large")
    return raw


def _validate_wheel_distribution(
    spec: DistributionValidationSpec,
    wheel: Path,
    sources: dict[str, CheckoutSource],
    *,
    expected_metadata: ExpectedPackageMetadata,
    expected_entry_points: dict[str, dict[str, str]],
) -> None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ReleasePreflightError("wheel contains too many archive members")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ReleasePreflightError("wheel contains duplicate archive members")
            metadata_members: list[zipfile.ZipInfo] = []
            package_prefix = f"{spec.package_path}/"
            namespace_root = PurePosixPath(spec.package_path).parts[0]
            allowed_roots = {namespace_root, spec.dist_info_root}
            for member in members:
                if not _safe_archive_member_name(member.filename):
                    raise ReleasePreflightError(
                        "wheel has an unsafe archive member name"
                    )
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ReleasePreflightError("wheel contains a symbolic link")
                parts = PurePosixPath(member.filename).parts
                top_level = parts[0]
                if _is_import_affecting_path(member.filename):
                    raise ReleasePreflightError(
                        "wheel must not contain .pth import payloads"
                    )
                if top_level.endswith(".data"):
                    raise ReleasePreflightError("wheel must not contain .data payloads")
                if top_level not in allowed_roots:
                    raise ReleasePreflightError(
                        "wheel contains an unexpected top-level payload"
                    )
                if len(parts) == 1 and not member.is_dir():
                    raise ReleasePreflightError(
                        "wheel top-level payload must be a directory"
                    )
                if top_level == namespace_root and not (
                    member.filename.startswith(package_prefix)
                    or (
                        member.is_dir()
                        and _package_ancestor(member.filename, spec.package_path)
                    )
                ):
                    raise ReleasePreflightError(
                        "wheel contains a source-unbound namespace payload"
                    )
                if top_level == spec.dist_info_root and not member.is_dir():
                    if member.file_size > MAX_METADATA_BYTES:
                        raise ReleasePreflightError(
                            "wheel dist-info member is too large"
                        )
                    if (
                        Path(member.filename).suffix.lower()
                        in EXECUTABLE_PAYLOAD_SUFFIXES
                    ):
                        raise ReleasePreflightError(
                            "wheel dist-info must not contain executable payloads"
                        )
                if member.filename.endswith(".dist-info/METADATA"):
                    metadata_members.append(member)
            if len(metadata_members) != 1:
                raise ReleasePreflightError(
                    "wheel must contain exactly one dist-info METADATA file"
                )
            metadata = metadata_members[0]
            if metadata.filename.rsplit("/", 1)[0] != spec.dist_info_root:
                raise ReleasePreflightError(
                    "wheel dist-info root does not match the expected package version"
                )
            required_members = {
                f"{spec.dist_info_root}/WHEEL",
                f"{spec.dist_info_root}/RECORD",
                f"{spec.package_path}/__init__.py",
            }
            if not required_members.issubset(names):
                raise ReleasePreflightError(
                    "wheel is missing required package or wheel metadata members"
                )
            _validate_wheel_record(archive, members, f"{spec.dist_info_root}/RECORD")
            _parse_package_metadata(
                _read_zip_metadata(archive, metadata, label="wheel metadata"),
                label="wheel",
                expected=expected_metadata,
            )
            entry_name = f"{spec.dist_info_root}/entry_points.txt"
            entry_member = next(
                (member for member in members if member.filename == entry_name), None
            )
            entry_raw = (
                _read_zip_metadata(archive, entry_member, label="wheel entry points")
                if entry_member is not None
                else None
            )
            _validate_entry_points(
                entry_raw, expected=expected_entry_points, label="wheel"
            )
            observed = {
                member.filename.removeprefix(package_prefix): member
                for member in members
                if member.filename.startswith(package_prefix) and not member.is_dir()
            }
            if set(observed) != set(sources):
                raise ReleasePreflightError(
                    "wheel runtime package files do not match the exact checkout"
                )
            for member in members:
                if not member.is_dir() or not member.filename.startswith(
                    package_prefix
                ):
                    continue
                relative = member.filename.removeprefix(package_prefix)
                if not _directory_is_needed(relative, sources):
                    raise ReleasePreflightError(
                        "wheel contains an unexpected runtime package directory"
                    )
            for relative, expected in sources.items():
                if not _wheel_member_matches(archive, observed[relative], expected):
                    raise ReleasePreflightError(
                        "wheel runtime sources do not match the exact checkout"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleasePreflightError("wheel is not a readable wheel archive") from exc


def _validate_checkout_bound_sdist_member(
    spec: DistributionValidationSpec,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    relative: PurePosixPath,
) -> None:
    checkout_path = spec.project_root.joinpath(*relative.parts)
    _require_regular_file(checkout_path, "sdist supplemental source")
    if member.size != checkout_path.stat().st_size or _tar_member_sha256(
        archive, member
    ) != _sha256(checkout_path):
        raise ReleasePreflightError(
            "sdist supplemental source does not match the exact checkout"
        )


def _validate_sdist_distribution(
    spec: DistributionValidationSpec,
    sdist: Path,
    sources: dict[str, CheckoutSource],
    *,
    expected_metadata: ExpectedPackageMetadata,
    expected_entry_points: dict[str, dict[str, str]],
) -> None:
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ReleasePreflightError("sdist contains too many archive members")
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ReleasePreflightError("sdist contains duplicate archive members")
            source_prefix = f"{spec.sdist_root}/"
            package_root = f"src/{spec.package_path}"
            package_prefix = f"{package_root}/"
            egg_info_prefix = f"{spec.egg_info_root}/"
            for member in members:
                if not _safe_archive_member_name(member.name):
                    raise ReleasePreflightError(
                        "sdist has an unsafe archive member name"
                    )
                if not (member.isdir() or member.isreg()):
                    raise ReleasePreflightError(
                        "sdist contains a non-regular archive member"
                    )
                if member.name != spec.sdist_root and not member.name.startswith(
                    source_prefix
                ):
                    raise ReleasePreflightError(
                        "sdist contains an unexpected top-level payload"
                    )
                if member.name == spec.sdist_root:
                    if not member.isdir():
                        raise ReleasePreflightError(
                            "sdist source root must be a directory"
                        )
                    continue
                relative_text = member.name.removeprefix(source_prefix)
                relative = PurePosixPath(relative_text)
                if _is_import_affecting_path(relative_text):
                    raise ReleasePreflightError(
                        "sdist must not contain .pth import payloads"
                    )
                if relative_text == package_root or _package_ancestor(
                    relative_text, package_root
                ):
                    if not member.isdir():
                        raise ReleasePreflightError(
                            "sdist runtime package root must be a directory"
                        )
                    continue
                if relative_text.startswith(package_prefix):
                    package_relative = relative_text.removeprefix(package_prefix)
                    if member.isdir() and not _directory_is_needed(
                        package_relative, sources
                    ):
                        raise ReleasePreflightError(
                            "sdist contains an unexpected runtime package directory"
                        )
                    continue
                if relative_text == spec.egg_info_root:
                    if not member.isdir():
                        raise ReleasePreflightError(
                            "sdist egg-info root must be a directory"
                        )
                    continue
                if relative_text.startswith(egg_info_prefix):
                    _validate_egg_info_member(
                        member, relative_text.removeprefix(egg_info_prefix)
                    )
                    continue
                if relative.parts and relative.parts[0] == "src":
                    if len(relative.parts) == 1 and member.isdir():
                        continue
                    raise ReleasePreflightError(
                        "sdist contains an unexpected source package"
                    )
                if relative_text == "PKG-INFO":
                    continue
                if (
                    relative_text == "setup.cfg"
                    and not (spec.project_root / "setup.cfg").exists()
                ):
                    _validate_generated_sdist_setup_cfg(archive, member)
                    continue
                if member.isdir():
                    checkout_directory = spec.project_root.joinpath(*relative.parts)
                    if (
                        checkout_directory.is_symlink()
                        or not checkout_directory.is_dir()
                    ):
                        raise ReleasePreflightError(
                            "sdist contains an unexpected supplemental directory"
                        )
                    continue
                _validate_checkout_bound_sdist_member(spec, archive, member, relative)
            metadata = next(
                (
                    member
                    for member in members
                    if member.name == f"{spec.sdist_root}/PKG-INFO"
                ),
                None,
            )
            if (
                metadata is None
                or not metadata.isreg()
                or f"{spec.sdist_root}/pyproject.toml" not in names
            ):
                raise ReleasePreflightError(
                    "sdist does not have the expected source-root metadata"
                )
            _parse_package_metadata(
                _read_tar_metadata(archive, metadata, label="sdist metadata"),
                label="sdist",
                expected=expected_metadata,
            )
            entry_name = f"{spec.sdist_root}/{spec.egg_info_root}/entry_points.txt"
            entry_member = next(
                (member for member in members if member.name == entry_name), None
            )
            entry_raw = (
                _read_tar_metadata(archive, entry_member, label="sdist entry points")
                if entry_member is not None
                else None
            )
            _validate_entry_points(
                entry_raw, expected=expected_entry_points, label="sdist"
            )
            observed = {
                member.name.removeprefix(f"{spec.sdist_root}/{package_prefix}"): member
                for member in members
                if member.name.startswith(f"{spec.sdist_root}/{package_prefix}")
                and member.isreg()
            }
            if set(observed) != set(sources):
                raise ReleasePreflightError(
                    "sdist runtime package files do not match the exact checkout"
                )
            for source_name, expected in sources.items():
                member = observed[source_name]
                if (
                    member.size != expected.size
                    or _tar_member_sha256(archive, member) != expected.sha256
                ):
                    raise ReleasePreflightError(
                        "sdist runtime sources do not match the exact checkout"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ReleasePreflightError("sdist is not a readable source archive") from exc


def validate_distribution_pair(
    spec: DistributionValidationSpec, *, wheel: Path, sdist: Path
) -> None:
    """Validate one wheel/sdist pair against one exact first-party checkout."""

    _require_regular_file(wheel, "wheel artifact")
    _require_regular_file(sdist, "sdist artifact")
    _require_regular_file(
        spec.project_root / "pyproject.toml", "checkout pyproject.toml"
    )
    expected_metadata = _expected_package_metadata(spec.project_root)
    if (
        expected_metadata.name != spec.distribution_name
        or expected_metadata.version != spec.version
    ):
        raise ReleasePreflightError(
            "distribution validation identity does not match checkout metadata"
        )
    expected_entry_points = _expected_entry_points(spec.project_root)
    sources = _checkout_package_files(spec)
    _validate_wheel_distribution(
        spec,
        wheel,
        sources,
        expected_metadata=expected_metadata,
        expected_entry_points=expected_entry_points,
    )
    _validate_sdist_distribution(
        spec,
        sdist,
        sources,
        expected_metadata=expected_metadata,
        expected_entry_points=expected_entry_points,
    )


def _find_distribution_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    if dist_dir.is_symlink() or not dist_dir.is_dir():
        raise ReleasePreflightError("distribution directory must be a real directory")
    entries = sorted(dist_dir.iterdir(), key=lambda path: path.name)
    if any(entry.is_symlink() for entry in entries):
        raise ReleasePreflightError(
            "distribution directory must not contain symbolic links"
        )
    wheels = [entry for entry in entries if entry.name.endswith(".whl")]
    sdists = [entry for entry in entries if entry.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleasePreflightError(
            "distribution directory must contain exactly one wheel and one sdist"
        )
    wheel, sdist = wheels[0], sdists[0]
    _require_regular_file(wheel, "wheel artifact")
    _require_regular_file(sdist, "sdist artifact")
    return wheel, sdist


def _load_hash_manifest(path: Path, expected_names: set[str]) -> dict[str, str]:
    _require_regular_file(path, "wheel/sdist hash manifest")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ReleasePreflightError("wheel/sdist hash manifest is too large")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or not SHA256_RE.fullmatch(fields[0]):
            raise ReleasePreflightError(
                f"wheel/sdist hash manifest line {line_number} is not sha256sum format"
            )
        name = fields[1].removeprefix("*")
        if (
            not name
            or Path(name).name != name
            or not _safe_archive_member_name(name)
            or name in entries
        ):
            raise ReleasePreflightError(
                "wheel/sdist hash manifest has an unsafe or duplicate filename"
            )
        entries[name] = fields[0]
    if set(entries) != expected_names:
        raise ReleasePreflightError(
            "wheel/sdist hash manifest must list exactly the built distributions"
        )
    return entries


def validate_distributions(config: ReleasePreflightConfig) -> DistributionArtifacts:
    wheel, sdist = _find_distribution_artifacts(config.dist_dir)
    expected_hashes = _load_hash_manifest(
        config.hash_manifest, {wheel.name, sdist.name}
    )
    observed_hashes = {wheel.name: _sha256(wheel), sdist.name: _sha256(sdist)}
    for name, digest in observed_hashes.items():
        if expected_hashes[name] != digest:
            raise ReleasePreflightError(
                "wheel/sdist hash manifest does not match built distributions"
            )
    validate_distribution_pair(
        DistributionValidationSpec(
            project_root=config.repo_root,
            distribution_name=PACKAGE_NAME,
            version=config.expected_version,
            package_path=PACKAGE_NAME,
        ),
        wheel=wheel,
        sdist=sdist,
    )
    return DistributionArtifacts(wheel=wheel, sdist=sdist, hashes=observed_hashes)
