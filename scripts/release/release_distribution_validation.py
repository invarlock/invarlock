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

PACKAGE_NAME = "invarlock"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_METADATA_BYTES = 1_048_576
RUNTIME_PACKAGE_SUFFIXES = frozenset({".json", ".py", ".pyi", ".yaml", ".yml"})
RUNTIME_PACKAGE_FILENAMES = frozenset({"py.typed"})
IGNORED_RUNTIME_PACKAGE_FILENAMES = frozenset({".DS_Store"})
IMPORT_AFFECTING_SUFFIXES = frozenset({".pth"})
EXECUTABLE_PAYLOAD_SUFFIXES = frozenset(
    {".dll", ".dylib", ".exe", ".pyd", ".py", ".pyc", ".pyo", ".so"}
)


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


def _resolve_from_repo(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


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


def _parse_package_metadata(raw: bytes, *, label: str, expected_version: str) -> None:
    if len(raw) > MAX_METADATA_BYTES:
        raise ReleasePreflightError(f"{label} metadata is too large")
    message = BytesParser(policy=policy.default).parsebytes(raw)
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if len(names) != 1 or str(names[0]).casefold() != PACKAGE_NAME:
        raise ReleasePreflightError(
            f"{label} metadata package name is not {PACKAGE_NAME}"
        )
    if len(versions) != 1 or str(versions[0]) != expected_version:
        raise ReleasePreflightError(
            f"{label} metadata version does not match expected version"
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


def _validate_wheel_surface(
    members: list[zipfile.ZipInfo], *, dist_info_root: str
) -> None:
    allowed_roots = {PACKAGE_NAME, dist_info_root}
    for member in members:
        parts = PurePosixPath(member.filename).parts
        top_level = parts[0]
        if _is_import_affecting_path(member.filename):
            raise ReleasePreflightError("wheel must not contain .pth import payloads")
        if top_level.endswith(".data"):
            raise ReleasePreflightError("wheel must not contain .data payloads")
        if top_level not in allowed_roots:
            raise ReleasePreflightError(
                "wheel contains an unexpected top-level payload"
            )
        if len(parts) == 1 and not member.is_dir():
            raise ReleasePreflightError("wheel top-level payload must be a directory")
        if (
            top_level == dist_info_root
            and not member.is_dir()
            and Path(member.filename).suffix.lower() in EXECUTABLE_PAYLOAD_SUFFIXES
        ):
            raise ReleasePreflightError(
                "wheel dist-info must not contain executable payloads"
            )


def _validate_wheel_metadata(wheel: Path, expected_version: str) -> str:
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ReleasePreflightError("wheel contains duplicate archive members")
            candidates = []
            for member in members:
                if not _safe_archive_member_name(member.filename):
                    raise ReleasePreflightError(
                        "wheel has an unsafe archive member name"
                    )
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ReleasePreflightError("wheel contains a symbolic link")
                if member.filename.endswith(".dist-info/METADATA"):
                    candidates.append(member)
            if len(candidates) != 1:
                raise ReleasePreflightError(
                    "wheel must contain exactly one dist-info METADATA file"
                )
            metadata = candidates[0]
            dist_info_root = metadata.filename.rsplit("/", 1)[0]
            if dist_info_root != f"{PACKAGE_NAME}-{expected_version}.dist-info":
                raise ReleasePreflightError(
                    "wheel dist-info root does not match the expected package version"
                )
            required_members = {
                f"{dist_info_root}/WHEEL",
                f"{dist_info_root}/RECORD",
                "invarlock/__init__.py",
            }
            if not required_members.issubset(names):
                raise ReleasePreflightError(
                    "wheel is missing required package or wheel metadata members"
                )
            _validate_wheel_surface(members, dist_info_root=dist_info_root)
            _validate_wheel_record(
                archive,
                members,
                f"{dist_info_root}/RECORD",
            )
            if metadata.file_size > MAX_METADATA_BYTES:
                raise ReleasePreflightError("wheel metadata is too large")
            _parse_package_metadata(
                archive.read(metadata),
                label="wheel",
                expected_version=expected_version,
            )
            return dist_info_root
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleasePreflightError("wheel is not a readable wheel archive") from exc


def _validate_wheel_entry_points(
    wheel: Path,
    *,
    dist_info_root: str,
    expected: dict[str, dict[str, str]],
) -> None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            name = f"{dist_info_root}/entry_points.txt"
            raw = archive.read(name) if name in archive.namelist() else None
            _validate_entry_points(raw, expected=expected, label="wheel")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleasePreflightError("wheel is not a readable wheel archive") from exc


def _validate_sdist_metadata(sdist: Path, expected_version: str) -> None:
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ReleasePreflightError("sdist contains duplicate archive members")
            for member in members:
                if not _safe_archive_member_name(member.name):
                    raise ReleasePreflightError(
                        "sdist has an unsafe archive member name"
                    )
                if (
                    member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isreg())
                ):
                    raise ReleasePreflightError(
                        "sdist contains a non-regular archive member"
                    )
            source_root = f"{PACKAGE_NAME}-{expected_version}"
            metadata_name = f"{source_root}/PKG-INFO"
            metadata = next(
                (member for member in members if member.name == metadata_name), None
            )
            if (
                metadata is None
                or not metadata.isreg()
                or f"{source_root}/pyproject.toml" not in names
            ):
                raise ReleasePreflightError(
                    "sdist does not have the expected source-root metadata"
                )
            if metadata.size > MAX_METADATA_BYTES:
                raise ReleasePreflightError("sdist metadata is too large")
            extracted = archive.extractfile(metadata)
            if extracted is None:
                raise ReleasePreflightError("sdist PKG-INFO is unreadable")
            _parse_package_metadata(
                extracted.read(MAX_METADATA_BYTES + 1),
                label="sdist",
                expected_version=expected_version,
            )
    except (OSError, tarfile.TarError) as exc:
        raise ReleasePreflightError("sdist is not a readable source archive") from exc


def _checkout_runtime_files(repo_root: Path) -> dict[str, str]:
    source_root = repo_root / "src" / PACKAGE_NAME
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReleasePreflightError("checkout runtime package directory is missing")
    sources: dict[str, str] = {}
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
        sources[relative.as_posix()] = _sha256(path)
    if not sources:
        raise ReleasePreflightError("checkout runtime package has no source files")
    return sources


def _tar_member_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReleasePreflightError("sdist source member is unreadable")
    digest = hashlib.sha256()
    for block in iter(lambda: extracted.read(1_048_576), b""):
        digest.update(block)
    return digest.hexdigest()


def _directory_is_needed(directory: str, files: dict[str, str]) -> bool:
    normalized = directory.rstrip("/")
    return not normalized or any(
        path == normalized or path.startswith(f"{normalized}/") for path in files
    )


def _validate_wheel_package_directories(
    members: list[zipfile.ZipInfo], source_hashes: dict[str, str]
) -> None:
    for member in members:
        if not member.filename.startswith(f"{PACKAGE_NAME}/"):
            continue
        relative = member.filename.removeprefix(f"{PACKAGE_NAME}/")
        if member.is_dir() and not _directory_is_needed(relative, source_hashes):
            raise ReleasePreflightError(
                "wheel contains an unexpected runtime package directory"
            )


def _validate_egg_info_member(member: tarfile.TarInfo, relative: str) -> None:
    if member.isdir():
        return
    suffix = Path(relative).suffix.lower()
    if suffix in IMPORT_AFFECTING_SUFFIXES or suffix in EXECUTABLE_PAYLOAD_SUFFIXES:
        raise ReleasePreflightError(
            "sdist egg-info must not contain executable or import payloads"
        )


def _validate_checkout_bound_sdist_member(
    config: ReleasePreflightConfig,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    relative: PurePosixPath,
) -> None:
    checkout_path = config.repo_root.joinpath(*relative.parts)
    _require_regular_file(checkout_path, "sdist supplemental source")
    if _tar_member_sha256(archive, member) != _sha256(checkout_path):
        raise ReleasePreflightError(
            "sdist supplemental source does not match the exact checkout"
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


def _validate_sdist_surface(
    config: ReleasePreflightConfig,
    archive: tarfile.TarFile,
    source_hashes: dict[str, str],
) -> None:
    source_root = f"{PACKAGE_NAME}-{config.expected_version}"
    source_prefix = f"{source_root}/"
    package_root = f"src/{PACKAGE_NAME}"
    package_prefix = f"{package_root}/"
    egg_info_root = f"src/{PACKAGE_NAME}.egg-info"
    egg_info_prefix = f"{egg_info_root}/"
    for member in archive.getmembers():
        if member.name != source_root and not member.name.startswith(source_prefix):
            raise ReleasePreflightError(
                "sdist contains an unexpected top-level payload"
            )
        if member.name == source_root:
            if not member.isdir():
                raise ReleasePreflightError("sdist source root must be a directory")
            continue
        relative_text = member.name.removeprefix(source_prefix)
        relative = PurePosixPath(relative_text)
        if relative_text and _is_import_affecting_path(relative_text):
            raise ReleasePreflightError("sdist must not contain .pth import payloads")
        if not relative_text:
            continue
        if relative_text == package_root:
            if not member.isdir():
                raise ReleasePreflightError(
                    "sdist runtime package root must be a directory"
                )
            continue
        if relative_text.startswith(package_prefix):
            package_relative = relative_text.removeprefix(package_prefix)
            if member.isdir() and not _directory_is_needed(
                package_relative, source_hashes
            ):
                raise ReleasePreflightError(
                    "sdist contains an unexpected runtime package directory"
                )
            continue
        if relative_text == egg_info_root:
            if not member.isdir():
                raise ReleasePreflightError("sdist egg-info root must be a directory")
            continue
        if relative_text.startswith(egg_info_prefix):
            _validate_egg_info_member(
                member, relative_text.removeprefix(egg_info_prefix)
            )
            continue
        if relative.parts and relative.parts[0] == "src":
            if len(relative.parts) == 1 and member.isdir():
                continue
            raise ReleasePreflightError("sdist contains an unexpected source package")
        if relative_text == "PKG-INFO":
            continue
        if (
            relative_text == "setup.cfg"
            and not (config.repo_root / "setup.cfg").exists()
        ):
            _validate_generated_sdist_setup_cfg(archive, member)
            continue
        if member.isdir():
            checkout_directory = config.repo_root.joinpath(*relative.parts)
            if checkout_directory.is_symlink() or not checkout_directory.is_dir():
                raise ReleasePreflightError(
                    "sdist contains an unexpected supplemental directory"
                )
            continue
        _validate_checkout_bound_sdist_member(config, archive, member, relative)


def _validate_sdist_entry_points(
    config: ReleasePreflightConfig,
    archive: tarfile.TarFile,
    *,
    expected: dict[str, dict[str, str]],
) -> None:
    source_root = f"{PACKAGE_NAME}-{config.expected_version}"
    name = f"{source_root}/src/{PACKAGE_NAME}.egg-info/entry_points.txt"
    member = next((item for item in archive.getmembers() if item.name == name), None)
    raw: bytes | None = None
    if member is not None:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ReleasePreflightError("sdist entry points are unreadable")
        raw = extracted.read(MAX_METADATA_BYTES + 1)
    _validate_entry_points(raw, expected=expected, label="sdist")


def _validate_distribution_checkout_binding(
    config: ReleasePreflightConfig,
    wheel: Path,
    sdist: Path,
    *,
    expected_entry_points: dict[str, dict[str, str]],
) -> None:
    source_hashes = _checkout_runtime_files(config.repo_root)
    try:
        with zipfile.ZipFile(wheel) as archive:
            wheel_member_list = archive.infolist()
            _validate_wheel_package_directories(wheel_member_list, source_hashes)
            wheel_members = {member.filename: member for member in wheel_member_list}
            wheel_package_members = {
                name.removeprefix(f"{PACKAGE_NAME}/")
                for name, member in wheel_members.items()
                if name.startswith(f"{PACKAGE_NAME}/") and not member.is_dir()
            }
            if wheel_package_members != set(source_hashes):
                raise ReleasePreflightError(
                    "wheel runtime package files do not match the exact checkout"
                )
            for relative, expected_digest in source_hashes.items():
                wheel_member = wheel_members.get(f"{PACKAGE_NAME}/{relative}")
                if wheel_member is None or _zip_member_sha256(
                    archive, wheel_member
                ) != base64.urlsafe_b64encode(bytes.fromhex(expected_digest)).decode(
                    "ascii"
                ).rstrip("="):
                    raise ReleasePreflightError(
                        "wheel runtime sources do not match the exact checkout"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleasePreflightError("wheel is not a readable wheel archive") from exc
    source_root = f"{PACKAGE_NAME}-{config.expected_version}"
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            _validate_sdist_surface(config, archive, source_hashes)
            _validate_sdist_entry_points(
                config, archive, expected=expected_entry_points
            )
            members = {member.name: member for member in archive.getmembers()}
            pyproject = members.get(f"{source_root}/pyproject.toml")
            if pyproject is None or _tar_member_sha256(archive, pyproject) != _sha256(
                config.repo_root / "pyproject.toml"
            ):
                raise ReleasePreflightError(
                    "sdist build metadata does not match the exact checkout"
                )
            package_prefix = f"{source_root}/src/{PACKAGE_NAME}/"
            sdist_package_members = {
                name.removeprefix(package_prefix)
                for name, member in members.items()
                if name.startswith(package_prefix) and member.isreg()
            }
            if sdist_package_members != set(source_hashes):
                raise ReleasePreflightError(
                    "sdist runtime package files do not match the exact checkout"
                )
            for relative, expected_digest in source_hashes.items():
                sdist_member = members.get(
                    f"{source_root}/src/{PACKAGE_NAME}/{relative}"
                )
                if (
                    sdist_member is None
                    or _tar_member_sha256(archive, sdist_member) != expected_digest
                ):
                    raise ReleasePreflightError(
                        "sdist runtime sources do not match the exact checkout"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ReleasePreflightError("sdist is not a readable source archive") from exc


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
    expected_entry_points = _expected_entry_points(config.repo_root)
    dist_info_root = _validate_wheel_metadata(wheel, config.expected_version)
    _validate_wheel_entry_points(
        wheel,
        dist_info_root=dist_info_root,
        expected=expected_entry_points,
    )
    _validate_sdist_metadata(sdist, config.expected_version)
    _validate_distribution_checkout_binding(
        config,
        wheel,
        sdist,
        expected_entry_points=expected_entry_points,
    )
    return DistributionArtifacts(wheel=wheel, sdist=sdist, hashes=observed_hashes)
