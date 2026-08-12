from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import release_distribution_validation as validation


def _metadata(
    *,
    requires_python: str | None = None,
    extras: tuple[str, ...] = (),
) -> validation.ExpectedPackageMetadata:
    return validation.ExpectedPackageMetadata(
        name="example",
        version="1.0",
        requires_python=requires_python,
        requires_dist=(),
        provides_extra=extras,
    )


def _spec(project_root: Path) -> validation.DistributionValidationSpec:
    return validation.DistributionValidationSpec(
        project_root=project_root,
        distribution_name="example",
        version="1.0",
        package_path="example",
    )


def _record(files: dict[str, bytes]) -> bytes:
    record_name = "example-1.0.dist-info/RECORD"
    rows: list[list[str]] = []
    for name, payload in sorted(files.items()):
        if name == record_name:
            rows.append([name, "", ""])
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
        rows.append([name, f"sha256={digest.rstrip('=')}", str(len(payload))])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


def _write_wheel(
    path: Path,
    *,
    extra_files: dict[str, bytes] | None = None,
    directories: tuple[str, ...] = (),
) -> dict[str, bytes]:
    files = {
        "example/__init__.py": b"VALUE = 1\n",
        "example-1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: example\nVersion: 1.0\n\n"
        ),
        "example-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "example-1.0.dist-info/RECORD": b"",
    }
    files.update(extra_files or {})
    files["example-1.0.dist-info/RECORD"] = _record(files)
    with zipfile.ZipFile(path, "w") as archive:
        for directory in directories:
            archive.writestr(directory.rstrip("/") + "/", b"")
        for name, payload in files.items():
            archive.writestr(name, payload)
    return files


def _validate_wheel(path: Path, project_root: Path) -> None:
    validation._validate_wheel_distribution(
        _spec(project_root),
        path,
        {
            "__init__.py": validation.CheckoutSource(
                size=len(b"VALUE = 1\n"),
                sha256=hashlib.sha256(b"VALUE = 1\n").hexdigest(),
            )
        },
        expected_metadata=_metadata(),
        expected_entry_points={},
    )


def _tar_info(
    name: str, payload: bytes | None = None
) -> tuple[tarfile.TarInfo, bytes | None]:
    member = tarfile.TarInfo(name)
    if payload is None:
        member.type = tarfile.DIRTYPE
        member.mode = 0o755
    else:
        member.size = len(payload)
        member.mode = 0o644
    return member, payload


def _symlink_tar_info(name: str) -> tuple[tarfile.TarInfo, None]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.SYMTYPE
    member.linkname = "target"
    return member, None


def _write_sdist(
    path: Path, entries: list[tuple[tarfile.TarInfo, bytes | None]]
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in entries:
            archive.addfile(
                member, io.BytesIO(payload) if payload is not None else None
            )


def _minimal_sdist_entries(
    *, package_payload: bytes | None = None
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    root = "example-1.0"
    entries = [
        _tar_info(root),
        _tar_info(f"{root}/src"),
        _tar_info(f"{root}/src/example"),
        _tar_info(f"{root}/src/example.egg-info"),
        _tar_info(
            f"{root}/PKG-INFO",
            b"Metadata-Version: 2.1\nName: example\nVersion: 1.0\n\n",
        ),
        _tar_info(
            f"{root}/pyproject.toml",
            b'[project]\nname = "example"\nversion = "1.0"\n',
        ),
        _tar_info(
            f"{root}/setup.cfg",
            b"[egg_info]\ntag_build =\ntag_date = 0\n",
        ),
    ]
    if package_payload is not None:
        entries.append(_tar_info(f"{root}/src/example/__init__.py", package_payload))
    return entries


@pytest.mark.parametrize("candidate", [Path("missing"), Path("directory")])
def test_executable_file_must_be_a_real_executable(
    tmp_path: Path, candidate: Path
) -> None:
    path = tmp_path / candidate
    if candidate.name == "directory":
        path.mkdir()
    with pytest.raises(validation.ReleasePreflightError, match="executable file"):
        validation._require_executable_file(path, "release Python")


def test_executable_file_rejects_non_executable_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "python"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(validation.ReleasePreflightError, match="executable file"):
        validation._require_executable_file(path, "release Python")


@pytest.mark.parametrize(
    ("raw", "expected", "message"),
    [
        (b"x" * (validation.MAX_METADATA_BYTES + 1), _metadata(), "too large"),
        (b"Name: wrong\nVersion: 1.0\n\n", _metadata(), "package name"),
        (b"Name: example\nVersion: 2.0\n\n", _metadata(), "version"),
        (
            b"Name: example\nVersion: 1.0\nRequires-Python: >=3.12\n\n",
            _metadata(),
            "Requires-Python",
        ),
        (
            b"Name: example\nVersion: 1.0\nProvides-Extra: other\n\n",
            _metadata(extras=("wanted",)),
            "Provides-Extra",
        ),
    ],
)
def test_package_metadata_rejects_tampering(
    raw: bytes,
    expected: validation.ExpectedPackageMetadata,
    message: str,
) -> None:
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._parse_package_metadata(raw, label="wheel", expected=expected)


def test_package_metadata_rejects_parser_defects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = SimpleNamespace(defects=[ValueError("bad header")])
    parser = SimpleNamespace(parsebytes=lambda _raw: malformed)
    monkeypatch.setattr(validation, "BytesParser", lambda **_kwargs: parser)
    with pytest.raises(validation.ReleasePreflightError, match="malformed"):
        validation._parse_package_metadata(
            b"ignored", label="wheel", expected=_metadata()
        )


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (validation._canonical_specifier, "not-a-specifier"),
        (validation._canonical_extra, " spaced "),
        (validation._requirement_identity, "not a valid requirement !!!"),
    ],
)
def test_invalid_checkout_metadata_tokens_are_rejected(
    function: object, value: str
) -> None:
    with pytest.raises(validation.ReleasePreflightError, match="invalid"):
        function(value, label="checkout field")  # type: ignore[operator]


def test_empty_canonical_extra_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "canonicalize_name", lambda _value: "")
    with pytest.raises(validation.ReleasePreflightError, match="invalid"):
        validation._canonical_extra("feature", label="checkout extra")


def test_project_table_rejects_invalid_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="unreadable"):
        validation._project_table(tmp_path)


def test_optional_dependency_name_must_be_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        validation,
        "_project_table",
        lambda _root: {
            "name": "example",
            "version": "1.0",
            "optional-dependencies": {1: []},
        },
    )
    with pytest.raises(validation.ReleasePreflightError, match="name is invalid"):
        validation._expected_package_metadata(tmp_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('[project]\nname="example"\nversion="1.0"\nscripts=[]\n', "must be a table"),
        (
            '[project]\nname="example"\nversion="1.0"\n[project.scripts]\n" "="x:y"\n',
            "entry point is invalid",
        ),
        (
            '[project]\nname="example"\nversion="1.0"\nentry-points=[]\n',
            "must be a table",
        ),
        (
            '[project]\nname="example"\nversion="1.0"\n[project.entry-points.""]\nx="x:y"\n',
            "group is invalid",
        ),
    ],
)
def test_checkout_entry_points_reject_invalid_tables(
    tmp_path: Path, payload: str, message: str
) -> None:
    (tmp_path / "pyproject.toml").write_text(payload, encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._expected_entry_points(tmp_path)


def test_checkout_entry_points_reject_invalid_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="unreadable"):
        validation._expected_entry_points(tmp_path)


def test_checkout_entry_points_require_project_and_ignore_empty_groups(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('name = "not-a-project"\n', encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="no project table"):
        validation._expected_entry_points(tmp_path)

    pyproject.write_text(
        '[project]\nname="example"\nversion="1.0"\n'
        "[project.scripts]\n"
        "[project.entry-points.empty]\n",
        encoding="utf-8",
    )
    assert validation._expected_entry_points(tmp_path) == {}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"x" * (validation.MAX_METADATA_BYTES + 1), "too large"),
        (b"\xff", "unreadable"),
        (b"[DEFAULT]\nvalue = x\n", "must not use defaults"),
        (b"[group]\nname =\n", "entry point is invalid"),
    ],
)
def test_entry_point_metadata_rejects_invalid_content(raw: bytes, message: str) -> None:
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._parse_entry_points(raw, label="wheel")


def test_wheel_record_rejects_malformed_rows(tmp_path: Path) -> None:
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg.py", b"pass\n")
        archive.writestr("RECORD", b"pkg.py,sha256=bad\n")
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(
            validation.ReleasePreflightError, match="invalid or duplicate"
        ):
            validation._validate_wheel_record(archive, archive.infolist(), "RECORD")


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (b"pkg.py,,,extra\nRECORD,,\n", "invalid or duplicate"),
        (b"RECORD,,\n", "does not cover"),
        (b"pkg.py,,5\nRECORD,,\n", "missing sha256"),
        (b"pkg.py,sha256=bad,5\nRECORD,,\n", "does not match"),
    ],
)
def test_wheel_record_rejects_inconsistent_integrity_rows(
    tmp_path: Path, record: bytes, message: str
) -> None:
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg.py", b"pass\n")
        archive.writestr("RECORD", record)
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(validation.ReleasePreflightError, match=message):
            validation._validate_wheel_record(archive, archive.infolist(), "RECORD")


def test_wheel_record_must_not_hash_itself(tmp_path: Path) -> None:
    payload = b"pass\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
    record = (
        f"pkg.py,sha256={digest.rstrip('=')},{len(payload)}\nRECORD,sha256=bad,1\n"
    ).encode()
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg.py", payload)
        archive.writestr("RECORD", record)
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(
            validation.ReleasePreflightError, match="must not self-hash"
        ):
            validation._validate_wheel_record(archive, archive.infolist(), "RECORD")


def test_wheel_record_rejects_unreadable_text(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("RECORD", b"\xff")
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(validation.ReleasePreflightError, match="unreadable"):
            validation._validate_wheel_record(archive, archive.infolist(), "RECORD")


def test_wheel_record_rejects_an_empty_record(tmp_path: Path) -> None:
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("RECORD", b"")
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(
            validation.ReleasePreflightError, match="missing or invalid"
        ):
            validation._validate_wheel_record(
                archive,
                archive.infolist(),
                "RECORD",
            )


def test_sdist_hash_requires_extractable_member() -> None:
    archive = SimpleNamespace(extractfile=lambda _member: None)
    with pytest.raises(validation.ReleasePreflightError, match="unreadable"):
        validation._tar_member_sha256(archive, tarfile.TarInfo("source.py"))  # type: ignore[arg-type]


def test_egg_info_rejects_oversized_and_executable_payloads() -> None:
    directory = tarfile.TarInfo("egg-info")
    directory.type = tarfile.DIRTYPE
    validation._validate_egg_info_member(directory, "")

    oversized = tarfile.TarInfo("PKG-INFO")
    oversized.size = validation.MAX_METADATA_BYTES + 1
    with pytest.raises(validation.ReleasePreflightError, match="too large"):
        validation._validate_egg_info_member(oversized, "PKG-INFO")

    executable = tarfile.TarInfo("payload.py")
    with pytest.raises(validation.ReleasePreflightError, match="executable"):
        validation._validate_egg_info_member(executable, "payload.py")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "unreadable"),
        (b"x" * (validation.MAX_METADATA_BYTES + 1), "too large"),
        (b"\xff", "invalid"),
        (b"[install]\nopt = value\n", "unsupported"),
    ],
)
def test_generated_setup_cfg_rejects_unsafe_content(
    payload: bytes | None, message: str
) -> None:
    archive = SimpleNamespace(
        extractfile=lambda _member: None if payload is None else io.BytesIO(payload)
    )
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._validate_generated_sdist_setup_cfg(  # type: ignore[arg-type]
            archive, tarfile.TarInfo("setup.cfg")
        )


@pytest.mark.parametrize(
    ("package_path", "setup", "message"),
    [
        ("../unsafe", "directory", "package path is invalid"),
        ("example", "missing", "directory is missing"),
        ("example", "empty", "no source files"),
        ("example", "unexpected", "unexpected file"),
    ],
)
def test_checkout_package_rejects_invalid_source_trees(
    tmp_path: Path, package_path: str, setup: str, message: str
) -> None:
    source = tmp_path / "src/example"
    if setup != "missing":
        source.mkdir(parents=True)
    if setup == "unexpected":
        (source / "native.bin").write_bytes(b"native")
    spec = validation.DistributionValidationSpec(
        project_root=tmp_path,
        distribution_name="example",
        version="1.0",
        package_path=package_path,
    )
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._checkout_package_files(spec)


def test_checkout_package_rejects_source_links(tmp_path: Path) -> None:
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    (source / "linked.py").symlink_to(target)
    with pytest.raises(validation.ReleasePreflightError, match="contain links"):
        validation._checkout_package_files(_spec(tmp_path))


def test_checkout_package_rejects_a_nonregular_source(tmp_path: Path) -> None:
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    os.mkfifo(source / "generated.py")

    with pytest.raises(validation.ReleasePreflightError, match="non-regular file"):
        validation._checkout_package_files(_spec(tmp_path))


def test_archive_metadata_readers_enforce_bounds() -> None:
    zip_member = SimpleNamespace(file_size=validation.MAX_METADATA_BYTES + 1)
    with pytest.raises(validation.ReleasePreflightError, match="too large"):
        validation._read_zip_metadata(SimpleNamespace(), zip_member, label="wheel")  # type: ignore[arg-type]

    tar_member = tarfile.TarInfo("metadata")
    tar_member.size = validation.MAX_METADATA_BYTES + 1
    with pytest.raises(validation.ReleasePreflightError, match="too large"):
        validation._read_tar_metadata(SimpleNamespace(), tar_member, label="sdist")  # type: ignore[arg-type]

    tar_member.size = 0
    archive = SimpleNamespace(extractfile=lambda _member: None)
    with pytest.raises(validation.ReleasePreflightError, match="unreadable"):
        validation._read_tar_metadata(archive, tar_member, label="sdist")  # type: ignore[arg-type]

    archive = SimpleNamespace(
        extractfile=lambda _member: io.BytesIO(
            b"x" * (validation.MAX_METADATA_BYTES + 1)
        )
    )
    with pytest.raises(validation.ReleasePreflightError, match="too large"):
        validation._read_tar_metadata(archive, tar_member, label="sdist")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extra_files", "directories", "message"),
    [
        ({"../unsafe.py": b"bad"}, (), "unsafe archive member"),
        ({"example": b"bad"}, (), "top-level payload must be a directory"),
        ({"unsafe.pth": b"import bad\n"}, (), "pth import"),
        ({"payload.data/value": b"bad"}, (), "data payload"),
        ({"other/payload.txt": b"bad"}, (), "unexpected top-level"),
        ({"example-1.0.dist-info/payload.py": b"bad"}, (), "executable payload"),
        ({}, ("example/unused",), "unexpected runtime package directory"),
    ],
)
def test_wheel_validation_rejects_structural_payloads(
    tmp_path: Path,
    extra_files: dict[str, bytes],
    directories: tuple[str, ...],
    message: str,
) -> None:
    wheel = tmp_path / "candidate.whl"
    _write_wheel(wheel, extra_files=extra_files, directories=directories)
    with pytest.raises(validation.ReleasePreflightError, match=message):
        _validate_wheel(wheel, tmp_path)


def test_wheel_validation_rejects_symlink_member(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    files = _write_wheel(wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        link = zipfile.ZipInfo("example/link.py")
        link.create_system = 3
        link.external_attr = (0o120777 << 16) | 0xA000
        archive.writestr(link, "__init__.py")
        for name, payload in files.items():
            archive.writestr(name, payload)
    with pytest.raises(validation.ReleasePreflightError, match="symbolic link"):
        _validate_wheel(wheel, tmp_path)


def test_wheel_validation_requires_canonical_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    files = _write_wheel(wheel)
    files.pop("example-1.0.dist-info/METADATA")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)

    with pytest.raises(
        validation.ReleasePreflightError,
        match="exactly one dist-info METADATA",
    ):
        _validate_wheel(wheel, tmp_path)


def test_wheel_validation_rejects_corrupt_archive(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"not a wheel")
    with pytest.raises(validation.ReleasePreflightError, match="readable wheel"):
        _validate_wheel(wheel, tmp_path)


def test_wheel_validation_rejects_member_count_and_duplicate_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "candidate.whl"
    _write_wheel(wheel)
    monkeypatch.setattr(validation, "MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(validation.ReleasePreflightError, match="too many"):
        _validate_wheel(wheel, tmp_path)

    monkeypatch.setattr(validation, "MAX_ARCHIVE_MEMBERS", 100)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("example/__init__.py", b"changed\n")
    with pytest.raises(validation.ReleasePreflightError, match="duplicate"):
        _validate_wheel(wheel, tmp_path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (_symlink_tar_info("example-1.0/link"), "non-regular archive member"),
        (_tar_info("../unsafe", b"bad"), "unsafe archive"),
        (_tar_info("outside/file", b"bad"), "unexpected top-level"),
        (_tar_info("example-1.0/src/other.py", b"bad"), "unexpected source package"),
        (_tar_info("example-1.0/docs"), "unexpected supplemental directory"),
    ],
)
def test_sdist_validation_rejects_structural_payloads(
    tmp_path: Path,
    extra: tuple[tarfile.TarInfo, bytes | None],
    message: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "1.0"\n', encoding="utf-8"
    )
    sdist = tmp_path / "candidate.tar.gz"
    _write_sdist(sdist, [extra])
    with pytest.raises(validation.ReleasePreflightError, match=message):
        validation._validate_sdist_distribution(
            _spec(tmp_path),
            sdist,
            {},
            expected_metadata=_metadata(),
            expected_entry_points={},
        )


def test_sdist_validation_rejects_missing_and_substituted_runtime_sources(
    tmp_path: Path,
) -> None:
    pyproject = b'[project]\nname = "example"\nversion = "1.0"\n'
    (tmp_path / "pyproject.toml").write_bytes(pyproject)
    sources = {
        "__init__.py": validation.CheckoutSource(
            size=4, sha256=hashlib.sha256(b"good").hexdigest()
        )
    }

    missing = tmp_path / "missing.tar.gz"
    _write_sdist(missing, _minimal_sdist_entries())
    with pytest.raises(validation.ReleasePreflightError, match="files do not match"):
        validation._validate_sdist_distribution(
            _spec(tmp_path),
            missing,
            sources,
            expected_metadata=_metadata(),
            expected_entry_points={},
        )

    substituted = tmp_path / "substituted.tar.gz"
    _write_sdist(substituted, _minimal_sdist_entries(package_payload=b"evil"))
    with pytest.raises(validation.ReleasePreflightError, match="sources do not match"):
        validation._validate_sdist_distribution(
            _spec(tmp_path),
            substituted,
            sources,
            expected_metadata=_metadata(),
            expected_entry_points={},
        )


def test_sdist_validation_rejects_corrupt_archive(tmp_path: Path) -> None:
    sdist = tmp_path / "candidate.tar.gz"
    sdist.write_bytes(b"not an sdist")
    with pytest.raises(
        validation.ReleasePreflightError, match="readable source archive"
    ):
        validation._validate_sdist_distribution(
            _spec(tmp_path),
            sdist,
            {},
            expected_metadata=_metadata(),
            expected_entry_points={},
        )


def test_distribution_pair_rejects_checkout_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "candidate.whl"
    sdist = tmp_path / "candidate.tar.gz"
    wheel.touch()
    sdist.touch()
    monkeypatch.setattr(
        validation, "_expected_package_metadata", lambda _root: _metadata()
    )
    spec = validation.DistributionValidationSpec(
        project_root=tmp_path,
        distribution_name="other",
        version="1.0",
        package_path="example",
    )
    (tmp_path / "pyproject.toml").touch()
    with pytest.raises(validation.ReleasePreflightError, match="identity"):
        validation.validate_distribution_pair(spec, wheel=wheel, sdist=sdist)


def test_distribution_discovery_and_manifest_reject_unsafe_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(validation.ReleasePreflightError, match="real directory"):
        validation._find_distribution_artifacts(tmp_path / "missing")

    dist = tmp_path / "dist"
    dist.mkdir()
    with pytest.raises(validation.ReleasePreflightError, match="exactly one"):
        validation._find_distribution_artifacts(dist)

    manifest = tmp_path / "hashes.txt"
    manifest.write_text("not-a-digest candidate.whl\n", encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="sha256sum"):
        validation._load_hash_manifest(manifest, {"candidate.whl"})

    digest = "0" * 64
    manifest.write_text(f"{digest} ../candidate.whl\n", encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="unsafe or duplicate"):
        validation._load_hash_manifest(manifest, {"candidate.whl"})

    manifest.write_text(f"{digest} other.whl\n", encoding="utf-8")
    with pytest.raises(validation.ReleasePreflightError, match="exactly"):
        validation._load_hash_manifest(manifest, {"candidate.whl"})


def test_hash_manifest_rejects_oversized_file(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.txt"
    manifest.write_bytes(b"x" * (validation.MAX_METADATA_BYTES + 1))
    with pytest.raises(validation.ReleasePreflightError, match="too large"):
        validation._load_hash_manifest(manifest, set())


def test_validate_distributions_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "candidate.whl"
    sdist = tmp_path / "candidate.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    manifest = tmp_path / "hashes.txt"
    manifest.write_text(
        f"{'0' * 64} {wheel.name}\n{'0' * 64} {sdist.name}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        validation, "_find_distribution_artifacts", lambda _root: (wheel, sdist)
    )
    config = validation.ReleasePreflightConfig(
        repo_root=tmp_path,
        release_sha="a" * 40,
        expected_version="1.0",
        dist_dir=tmp_path,
        hash_manifest=manifest,
    )
    with pytest.raises(validation.ReleasePreflightError, match="does not match"):
        validation.validate_distributions(config)
