"""Bind a narrowly excepted installed component to approved lock and wheel bytes."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

_LIMIT = 128 * 1024 * 1024
_EXTRAS = frozenset({"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"})


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_path(path: Path) -> Path:
    path = path.absolute()
    _require(".." not in path.parts, "parent traversal in filesystem path")
    for parent in reversed((path, *path.parents)):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            _require(not stat.S_ISLNK(info.st_mode), f"symlink path: {parent}")
            if parent != path:
                _require(stat.S_ISDIR(info.st_mode), f"non-directory parent: {parent}")
    return path


def _read(path: Path) -> bytes:
    path = _safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"non-regular file: {path}")
        _require(before.st_size <= _LIMIT, f"file exceeds audit bound: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(_LIMIT + 1)
        after = os.fstat(fd)
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )

        def stable(info):
            return tuple(getattr(info, field) for field in fields)

        _require(
            stable(before) == stable(after) and len(data) == before.st_size,
            f"file changed during read: {path}",
        )
        _safe_path(path)
        _require(
            stable(path.stat()) == stable(after), f"file replaced during read: {path}"
        )
        return data
    finally:
        os.close(fd)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def _json(data: bytes | str) -> object:
    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    _require(
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and value == path.as_posix()
        and value != ".",
        "lock must use its exact repository-relative path",
    )
    return value


def _lock(value: str, pin: str) -> tuple[bytes, dict[str, tuple[str, set[str]]]]:
    _relative(value)
    data = _read(Path(value))
    _require(
        re.fullmatch(r"[0-9a-f]{64}", pin) and _sha(data) == pin, "lock SHA256 mismatch"
    )
    pins: dict[str, tuple[str, set[str]]] = {}
    pending = ""
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("\\")
        pending += " " + (line[:-1].strip() if continued else line)
        if continued:
            continue
        match = re.fullmatch(
            r"\s*([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;\\]+)((?:\s+--hash=sha256:[0-9a-f]{64})+)\s*",
            pending,
        )
        _require(
            match is not None, "lock requires exact marker-free hashed package pins"
        )
        assert match is not None
        package = _name(match[1])
        _require(package not in pins, "duplicate package in lock")
        pins[package] = (match[2], set(re.findall(r"sha256:([0-9a-f]{64})", match[3])))
        pending = ""
    _require(not pending and bool(pins), "incomplete or empty lock")
    return data, pins


def _metadata(data: bytes) -> tuple[str, str]:
    metadata = BytesParser().parsebytes(data)
    names, versions = metadata.get_all("Name", []), metadata.get_all("Version", [])
    _require(len(names) == len(versions) == 1, "metadata requires one Name and Version")
    name, version = names[0], versions[0]
    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version),
        "invalid metadata identity",
    )
    return _name(name), version


def _record(data: bytes) -> dict[str, tuple[str, str]]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8")), strict=True))
    _require(all(len(row) == 3 and row[0] for row in rows), "invalid RECORD row")
    result = {row[0]: (row[1], row[2]) for row in rows}
    _require(len(result) == len(rows), "duplicate RECORD row")
    return result


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()
    ).decode().rstrip("=")


def _wheel(path: Path) -> tuple[bytes, dict[str, bytes], str, str, str]:
    data = _read(path)
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        infos = wheel.infolist()
        _require(
            len(infos) <= 20000 and sum(i.file_size for i in infos) <= _LIMIT,
            "wheel exceeds audit bound",
        )
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            canonical = PurePosixPath(name.rstrip("/"))
            _require(
                name not in seen
                and "\\" not in name
                and not canonical.is_absolute()
                and ".." not in canonical.parts
                and name == canonical.as_posix() + ("/" if info.is_dir() else ""),
                "unsafe or duplicate wheel member",
            )
            seen.add(name)
            mode = stat.S_IFMT(info.external_attr >> 16)
            _require(
                mode in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG),
                "non-regular wheel member",
            )
            _require(
                not any(part.endswith(".data") for part in canonical.parts)
                and not name.endswith((".pth", ".pyc")),
                "unsupported wheel relocation or executable extra",
            )
            if not info.is_dir():
                files[name] = wheel.read(info)
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    _require(len(records) == 1, "wheel requires one dist-info RECORD")
    record_name = records[0]
    dist = record_name.split("/")[0]
    _require(record_name == dist + "/RECORD", "nested dist-info")
    record = _record(files[record_name])
    _require(set(record) == set(files), "wheel RECORD inventory mismatch")
    for name, payload in files.items():
        expected = (
            ("", "")
            if name == record_name
            else (_record_hash(payload), str(len(payload)))
        )
        _require(record[name] == expected, f"wheel RECORD digest mismatch: {name}")
    _require(
        dist + "/METADATA" in files and dist + "/WHEEL" in files,
        "wheel metadata missing",
    )
    name, version = _metadata(files[dist + "/METADATA"])
    _require(
        dist == f"{name.replace('-', '_')}-{version}.dist-info",
        "wheel dist-info identity mismatch",
    )
    _require(
        all(
            not part.endswith(".dist-info") or part == dist
            for file in files
            for part in PurePosixPath(file).parts
        ),
        "additional wheel dist-info",
    )
    return data, files, dist, name, version


def _inventory(root: Path) -> tuple[dict[str, str], dict[str, Path]]:
    _safe_path(root)
    _require(root.is_dir(), "installed path must be a directory")
    versions: dict[str, str] = {}
    locations: dict[str, Path] = {}
    for item in sorted(root.iterdir()):
        _safe_path(item)
        _require(
            not item.name.endswith(".egg-info"),
            "unsupported or ambiguous egg-info installation",
        )
        if not item.name.endswith(".dist-info"):
            continue
        _require(item.is_dir(), "dist-info must be a directory")
        name, version = _metadata(_read(item / "METADATA"))
        _require(name not in versions, "duplicate installed distribution")
        _require(
            item.name == f"{name.replace('-', '_')}-{version}.dist-info",
            "installed dist-info identity mismatch",
        )
        versions[name], locations[name] = version, item
    return versions, locations


def _payload(root: Path, directory: str) -> dict[str, bytes]:
    result: dict[str, bytes] = {}

    def unavailable(error: OSError) -> None:
        raise error

    for parent, directories, filenames in os.walk(
        root / directory, followlinks=False, onerror=unavailable
    ):
        for name in directories:
            _safe_path(Path(parent) / name)
        for name in filenames:
            path = Path(parent) / name
            result[path.relative_to(root).as_posix()] = _read(path)
    return result


def _bind_component(
    root: Path, files: dict[str, bytes], dist: str, package: str, wheel_data: bytes
) -> dict:
    """Authenticate one component; the caller separately binds the full inventory."""
    top_data = files.get(dist + "/top_level.txt", b"").decode("utf-8")
    top = top_data.strip()
    _require(
        top.isidentifier() and top == package.replace("-", "_"),
        "wheel requires one canonical package namespace",
    )
    _require(
        top + "/__init__.py" in files
        and {name.split("/")[0] for name in files} == {top, dist},
        "unsupported wheel package inventory",
    )
    _require(not (root / (top + ".py")).exists(), "ambiguous top-level package module")
    _require(
        not any(item.name.startswith(top + ".") for item in root.iterdir()),
        "ambiguous alternate package module",
    )
    _safe_path(root / top)
    _require((root / top).is_dir(), "installed package directory missing")
    actual = _payload(root, top) | _payload(root, dist)
    permitted = {dist + "/" + extra for extra in _EXTRAS}
    _require(
        set(files) <= set(actual) and set(actual) <= set(files) | permitted,
        "installed payload inventory differs from wheel",
    )
    for name, payload in files.items():
        if name != dist + "/RECORD":
            _require(
                actual[name] == payload, f"installed payload differs from wheel: {name}"
            )
    if dist + "/INSTALLER" in actual:
        _require(
            actual[dist + "/INSTALLER"] == b"pip\n", "unexpected installed INSTALLER"
        )
    if dist + "/REQUESTED" in actual:
        _require(actual[dist + "/REQUESTED"] == b"", "unexpected installed REQUESTED")
    if dist + "/direct_url.json" in actual:
        direct = _json(actual[dist + "/direct_url.json"])
        _require(
            isinstance(direct, dict)
            and set(direct) == {"url", "archive_info"}
            and isinstance(direct["url"], str)
            and isinstance(direct["archive_info"], dict),
            "unexpected direct_url metadata",
        )
        archive = direct["archive_info"]
        _require(
            set(archive) <= {"hash", "hashes"}
            and archive.get("hashes") == {"sha256": _sha(wheel_data)}
            and (
                "hash" not in archive or archive["hash"] == "sha256=" + _sha(wheel_data)
            ),
            "direct_url wheel hash mismatch",
        )
    installed_record = _record(actual[dist + "/RECORD"])
    for name, payload in actual.items():
        expected_record = (
            ("", "")
            if name == dist + "/RECORD"
            else (_record_hash(payload), str(len(payload)))
        )
        _require(
            installed_record.get(name) == expected_record,
            f"installed RECORD differs from payload: {name}",
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(files.get(dist + "/entry_points.txt", b"").decode("utf-8"))
    scripts = (
        set(parser["console_scripts"])
        if parser.has_section("console_scripts")
        else set()
    )
    for name in set(installed_record) - set(actual):
        parts = PurePosixPath(name).parts
        prefix = parts[:-2]
        _require(
            "\\" not in name
            and len(parts) >= 2
            and parts[-2] == "bin"
            and parts[-1] in scripts
            and len(prefix) <= 3
            and all(part == ".." for part in prefix),
            "unexpected installed RECORD entry",
        )
    return {
        "installed_component_files": {
            name: _sha(data) for name, data in sorted(actual.items())
        },
        "generated_console_scripts_not_byte_bound": sorted(
            set(installed_record) - set(actual)
        ),
    }


def bind_installation(args: argparse.Namespace, entries: list) -> dict:
    lock_data, lock = _lock(args.installed_lock, args.installed_lock_sha256)
    bootstrap_data, bootstrap = _lock(
        args.installed_bootstrap_lock, args.installed_bootstrap_lock_sha256
    )
    _require(
        args.installed_bootstrap_lock == "requirements/workflows/pip-bootstrap.txt",
        "unapproved bootstrap lock source",
    )
    wheel_data, files, dist, package, version = _wheel(Path(args.installed_wheel))
    approvals = [
        entry
        for entry in entries
        if package in entry.packages
        and version in entry.versions
        and args.installed_lock in entry.allowed_sources
    ]
    _require(bool(approvals), "component or lock source has no approved exception")
    _require(
        package in lock
        and lock[package][0] == version
        and _sha(wheel_data) in lock[package][1],
        "wheel is not the exact locked artifact",
    )
    project_data, project_files, project_dist, project, project_version = _wheel(
        Path(args.installed_project_wheel)
    )
    _require(
        project == "invarlock" and project != package,
        "project wheel must identify invarlock",
    )
    expected = {name: pin[0] for name, pin in lock.items()}
    for name, (pin, _hashes) in bootstrap.items():
        _require(
            name not in expected or expected[name] == pin,
            "conflicting bootstrap identity",
        )
        expected[name] = pin
    _require(
        project not in expected or expected[project] == project_version,
        "conflicting project identity",
    )
    expected[project] = project_version
    root = Path(args.path[0]).absolute()
    installed, locations = _inventory(root)
    _require(
        installed == expected,
        f"installed distribution inventory differs from approved inputs: missing={sorted(set(expected) - set(installed))}, extra={sorted(set(installed) - set(expected))}, mismatched={sorted(name for name in installed.keys() & expected.keys() if installed[name] != expected[name])}",
    )
    _require(
        _read(locations[project] / "METADATA")
        == project_files[project_dist + "/METADATA"],
        "installed project metadata differs from project wheel",
    )
    component = _bind_component(root, files, dist, package, wheel_data)
    return {
        "lock": args.installed_lock,
        "lock_sha256": _sha(lock_data),
        "bootstrap_lock": args.installed_bootstrap_lock,
        "bootstrap_lock_sha256": _sha(bootstrap_data),
        "wheel_sha256": _sha(wheel_data),
        "project_wheel_sha256": _sha(project_data),
        "project_identity": {"name": project, "version": project_version},
        "package": package,
        "version": version,
        "installed_path": str(root),
        "installed_distribution_inventory": installed,
        **component,
        "approved_advisories": sorted({entry.advisory for entry in approvals}),
        "scope": "Exact installed distribution names and versions; byte authentication of the excepted component only. Other dependencies remain fully audited without exceptions. Trusted workflow filesystem ownership is required throughout the audit.",
    }


def _classify(raw: object, binding: dict, returncode: int) -> tuple[list, list]:
    _require(
        isinstance(raw, dict)
        and set(raw) <= {"dependencies", "fixes"}
        and raw.get("fixes", []) == []
        and isinstance(raw.get("dependencies"), list),
        "invalid scanner JSON envelope",
    )
    observed: dict[str, str] = {}
    accepted, blocking = [], []
    for dependency in raw["dependencies"]:
        _require(
            isinstance(dependency, dict)
            and set(dependency) == {"name", "version", "vulns"}
            and isinstance(dependency["name"], str)
            and isinstance(dependency["version"], str)
            and isinstance(dependency["vulns"], list),
            "incomplete or skipped scanner dependency",
        )
        name, version = _name(dependency["name"]), dependency["version"]
        _require(name not in observed, "duplicate scanner dependency")
        observed[name] = version
        for finding in dependency["vulns"]:
            _require(
                isinstance(finding, dict)
                and isinstance(finding.get("id"), str)
                and bool(finding["id"])
                and isinstance(finding.get("aliases", []), list)
                and all(
                    isinstance(alias, str) and alias
                    for alias in finding.get("aliases", [])
                )
                and isinstance(finding.get("fix_versions"), list)
                and all(isinstance(v, str) for v in finding["fix_versions"]),
                "invalid scanner vulnerability",
            )
            record = {"name": name, "version": version, "finding": finding}
            # Only registered canonical IDs authorize acceptance. Scanner aliases
            # remain visible but cannot turn an unrelated ID into an exception.
            applies = (
                name == binding["package"]
                and version == binding["version"]
                and finding["id"] in binding["approved_advisories"]
            )
            (accepted if applies else blocking).append(record)
    _require(
        observed == binding["installed_distribution_inventory"],
        "scanner inventory differs from authenticated installed inventory",
    )
    _require(
        returncode in (0, 1) and (returncode == 1) == bool(accepted or blocking),
        "scanner status does not match complete findings",
    )
    return accepted, blocking


def run_bound_audit(args: argparse.Namespace, load_allowlist) -> int:
    report: dict = {
        "status": "blocked",
        "accepted_findings": [],
        "blocking_findings": [],
        "raw_stdout": "",
        "raw_stderr": "",
        "scope": "Temporary component exception; not a clean vulnerability scan.",
    }
    try:
        required = [
            args.installed_lock,
            args.installed_lock_sha256,
            args.installed_wheel,
            args.installed_bootstrap_lock,
            args.installed_bootstrap_lock_sha256,
            args.installed_project_wheel,
            args.report,
        ]
        _require(
            all(required) and len(args.path) == 1 and not args.requirement,
            "bound installed mode requires one --path and all installed binding/report options, without --requirement",
        )
        policy_data = _read(Path(args.allowlist))
        _json(policy_data)
        _owner, entries = load_allowlist(Path(args.allowlist))
        report["allowlist_sha256"] = _sha(policy_data)
        binding = bind_installation(args, entries)
        report["binding"] = binding
        command = ["pip-audit", "--path", args.path[0], "--format", "json"]
        report["scanner_command"] = command
        completed = subprocess.run(
            command, check=False, capture_output=True, timeout=300
        )
        report.update(
            {
                "scanner_returncode": completed.returncode,
                "raw_stdout": completed.stdout.decode("utf-8", errors="replace"),
                "raw_stderr": completed.stderr.decode("utf-8", errors="replace"),
                "raw_stdout_base64": base64.b64encode(completed.stdout).decode(),
                "raw_stderr_base64": base64.b64encode(completed.stderr).decode(),
            }
        )
        raw = _json(completed.stdout)
        report["raw_findings"] = raw
        accepted, blocking = _classify(raw, binding, completed.returncode)
        report["accepted_findings"], report["blocking_findings"] = accepted, blocking
        _require(
            _read(Path(args.allowlist)) == policy_data, "allowlist changed during audit"
        )
        _owner, current_entries = load_allowlist(Path(args.allowlist))
        _require(
            bind_installation(args, current_entries) == binding,
            "installed binding changed during audit",
        )
        report["status"] = (
            "blocked" if blocking else "accepted_exception" if accepted else "clean"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        UnicodeError,
        zipfile.BadZipFile,
        csv.Error,
        configparser.Error,
        SystemExit,
        subprocess.SubprocessError,
    ) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.TimeoutExpired):
            report["raw_stdout_base64"] = base64.b64encode(exc.stdout or b"").decode()
            report["raw_stderr_base64"] = base64.b64encode(exc.stderr or b"").decode()
    if not args.report:
        print(json.dumps(report, indent=2))
        return 1
    destination = _safe_path(Path(args.report))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(destination)
    fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        _require(
            stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "report must be regular"
        )
        os.ftruncate(stream.fileno(), 0)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Installed audit: {report['status']}; report: {args.report}")
    return 0 if report["status"] in {"clean", "accepted_exception"} else 1
