from __future__ import annotations

import base64
import hashlib
import io
import stat
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import release_distribution_validation as dist


def test_path_and_file_boundaries_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    regular = root / "file.py"
    regular.write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 2\n", encoding="utf-8")
    assert dist._is_within(regular, root)
    assert not dist._is_within(outside, root)
    assert dist._resolve_from_repo(root, Path("file.py")) == regular
    assert dist._resolve_from_repo(root, outside) == outside
    dist._require_regular_file(regular, "source")
    with pytest.raises(dist.ReleasePreflightError, match="regular file"):
        dist._require_regular_file(root, "source")
    executable = root / "tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    dist._require_executable_file(executable, "tool")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(dist.ReleasePreflightError, match="executable file"):
        dist._require_executable_file(executable, "tool")
    assert len(dist._sha256(regular)) == 64


@pytest.mark.parametrize(
    ("name", "safe"),
    [
        ("invarlock/module.py", True),
        ("", False),
        ("/absolute.py", False),
        ("../escape.py", False),
        ("folder\\module.py", False),
        ("folder//module.py", True),
    ],
)
def test_archive_member_name_validation(name: str, safe: bool) -> None:
    assert dist._safe_archive_member_name(name) is safe


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"x" * (dist.MAX_METADATA_BYTES + 1), "too large"),
        (b"Name: other\nVersion: 1.0\n", "package name"),
        (b"Name: invarlock\nName: invarlock\nVersion: 1.0\n", "package name"),
        (b"Name: invarlock\nVersion: 2.0\n", "version"),
    ],
)
def test_package_metadata_rejects_oversize_or_ambiguous_identity(
    raw: bytes, expected: str
) -> None:
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._parse_package_metadata(raw, label="wheel", expected_version="1.0")


def test_package_metadata_accepts_one_exact_identity() -> None:
    assert (
        dist._parse_package_metadata(
            b"Metadata-Version: 2.3\nName: InvarLock\nVersion: 1.0\n",
            label="wheel",
            expected_version="1.0",
        )
        is None
    )


@pytest.mark.parametrize(
    "value",
    [[], {"": "pkg:main"}, {"name": ""}, {1: "pkg:main"}, {"name": 1}],
)
def test_entry_point_group_requires_nonempty_string_mappings(value: object) -> None:
    with pytest.raises(dist.ReleasePreflightError, match="entry point"):
        dist._entry_point_group(value, label="project.scripts")


def test_expected_entry_points_supports_all_declared_groups(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "invarlock"
version = "1.0"
[project.scripts]
invarlock = "invarlock.cli:main"
[project.gui-scripts]
invarlock-gui = "invarlock.gui:main"
[project.entry-points."invarlock.guards"]
spectral = "invarlock.guards:spectral"
""",
        encoding="utf-8",
    )
    assert dist._expected_entry_points(tmp_path) == {
        "console_scripts": {"invarlock": "invarlock.cli:main"},
        "gui_scripts": {"invarlock-gui": "invarlock.gui:main"},
        "invarlock.guards": {"spectral": "invarlock.guards:spectral"},
    }


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("not = toml =", "unreadable"),
        ("[tool.demo]\nvalue=1\n", "no project table"),
        ("[project]\nentry-points=[]\n", "entry-points must be a table"),
        ('[project.entry-points]\n"" = {}\n', "group is invalid"),
    ],
)
def test_expected_entry_points_rejects_malformed_checkout_metadata(
    tmp_path: Path, contents: str, expected: str
) -> None:
    (tmp_path / "pyproject.toml").write_text(contents, encoding="utf-8")
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._expected_entry_points(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"x" * (dist.MAX_METADATA_BYTES + 1), "too large"),
        (b"\xff", "unreadable"),
        (b"[DEFAULT]\nname=pkg:main\n", "must not use defaults"),
        (b"[console_scripts]\n = pkg:main\n", "unreadable"),
        (b"[console_scripts]\nname = \n", "entry point is invalid"),
    ],
)
def test_entry_point_payload_rejects_oversize_invalid_or_defaulted_data(
    raw: bytes, expected: str
) -> None:
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._parse_entry_points(raw, label="wheel")


def test_entry_point_validation_requires_exact_checkout_mapping() -> None:
    raw = b"[console_scripts]\nName = pkg:main\n"
    expected = {"console_scripts": {"Name": "pkg:main"}}
    assert dist._parse_entry_points(raw, label="wheel") == expected
    dist._validate_entry_points(raw, expected=expected, label="wheel")
    with pytest.raises(dist.ReleasePreflightError, match="do not match"):
        dist._validate_entry_points(None, expected=expected, label="wheel")


def _write_zip(path: Path, entries: list[tuple[str, bytes, int | None]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            archive.writestr(info, payload)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([("../METADATA", b"", None)], "unsafe"),
        ([("invarlock-1.0.dist-info/METADATA", b"", stat.S_IFLNK)], "symbolic link"),
        ([], "exactly one"),
        ([("invarlock-other.dist-info/METADATA", b"", None)], "dist-info root"),
    ],
)
def test_wheel_metadata_rejects_unsafe_linked_or_forked_layouts(
    tmp_path: Path,
    entries: list[tuple[str, bytes, int | None]],
    expected: str,
) -> None:
    wheel = tmp_path / "candidate.whl"
    _write_zip(wheel, entries)
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._validate_wheel_metadata(wheel, "1.0")


def test_wheel_metadata_rejects_duplicate_archive_members(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(
            wheel,
            [
                ("invarlock-1.0.dist-info/METADATA", b"Name: invarlock\n", None),
                ("invarlock-1.0.dist-info/METADATA", b"Name: invarlock\n", None),
            ],
        )
    with pytest.raises(dist.ReleasePreflightError, match="duplicate"):
        dist._validate_wheel_metadata(wheel, "1.0")


def _record_row(name: str, payload: bytes) -> str:
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    )
    return f"{name},sha256={digest},{len(payload)}"


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (b"", "missing or invalid"),
        (b"\xff", "unreadable"),
        (b"bad,row\n", "invalid or duplicate"),
        (b"RECORD,,\n", "does not cover"),
        (b"module.py,sha256=bad,1\nRECORD,sha256=bad,1\n", "must not self-hash"),
        (b"module.py,,1\nRECORD,,\n", "missing sha256 or size"),
        (b"module.py,sha256=bad,1\nRECORD,,\n", "does not match"),
    ],
)
def test_wheel_record_rejects_missing_forked_or_unbound_rows(
    tmp_path: Path, record: bytes, expected: str
) -> None:
    wheel = tmp_path / "record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("RECORD", record)
        archive.writestr("module.py", b"x")
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(dist.ReleasePreflightError, match=expected):
            dist._validate_wheel_record(archive, archive.infolist(), "RECORD")


def test_wheel_record_accepts_exact_digest_size_and_unhashed_self_row(
    tmp_path: Path,
) -> None:
    payload = b"x"
    wheel = tmp_path / "record.whl"
    record = (_record_row("module.py", payload) + "\nRECORD,,\n").encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("module.py", payload)
        archive.writestr("RECORD", record)
    with zipfile.ZipFile(wheel) as archive:
        assert (
            dist._validate_wheel_record(archive, archive.infolist(), "RECORD") is None
        )


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in members:
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload) if member.isreg() else None)


def _tar_member(name: str, *, kind: bytes = tarfile.REGTYPE) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = kind
    return member


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (
            [
                (_tar_member("invarlock-1.0/PKG-INFO"), b"x"),
                (_tar_member("invarlock-1.0/PKG-INFO"), b"x"),
            ],
            "duplicate",
        ),
        ([(_tar_member("../PKG-INFO"), b"x")], "unsafe"),
        (
            [(_tar_member("invarlock-1.0/link", kind=tarfile.SYMTYPE), b"")],
            "non-regular",
        ),
        ([(_tar_member("invarlock-1.0/pyproject.toml"), b"x")], "source-root metadata"),
        (
            [
                (_tar_member("invarlock-1.0/pyproject.toml"), b"x"),
                (
                    _tar_member("invarlock-1.0/PKG-INFO"),
                    b"x" * (dist.MAX_METADATA_BYTES + 1),
                ),
            ],
            "metadata is too large",
        ),
    ],
)
def test_sdist_metadata_rejects_duplicate_unsafe_special_or_incomplete_archives(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    expected: str,
) -> None:
    sdist = tmp_path / "candidate.tar.gz"
    _write_tar(sdist, members)
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._validate_sdist_metadata(sdist, "1.0")


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (_tar_member("foreign/file.txt"), "unexpected top-level"),
        (_tar_member("invarlock-1.0"), "source root must be a directory"),
        (_tar_member("invarlock-1.0/src/invarlock"), "runtime package root"),
        (
            _tar_member("invarlock-1.0/src/invarlock/extra/", kind=tarfile.DIRTYPE),
            "unexpected runtime package directory",
        ),
        (_tar_member("invarlock-1.0/src/invarlock.egg-info"), "egg-info root"),
        (_tar_member("invarlock-1.0/src/other.py"), "unexpected source package"),
        (
            _tar_member("invarlock-1.0/docs/", kind=tarfile.DIRTYPE),
            "unexpected supplemental directory",
        ),
    ],
)
def test_sdist_surface_rejects_unexpected_roots_packages_and_directories(
    tmp_path: Path, member: tarfile.TarInfo, expected: str
) -> None:
    config = SimpleNamespace(repo_root=tmp_path, expected_version="1.0")
    archive = SimpleNamespace(getmembers=lambda: [member])
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._validate_sdist_surface(config, archive, {"__init__.py": "digest"})


def test_sdist_surface_accepts_only_needed_and_checkout_bound_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    members = [
        _tar_member("invarlock-1.0", kind=tarfile.DIRTYPE),
        _tar_member("invarlock-1.0/", kind=tarfile.DIRTYPE),
        _tar_member("invarlock-1.0/src/invarlock", kind=tarfile.DIRTYPE),
        _tar_member("invarlock-1.0/src/invarlock.egg-info", kind=tarfile.DIRTYPE),
        _tar_member("invarlock-1.0/src", kind=tarfile.DIRTYPE),
        _tar_member("invarlock-1.0/docs", kind=tarfile.DIRTYPE),
    ]
    archive = SimpleNamespace(getmembers=lambda: members)
    config = SimpleNamespace(repo_root=tmp_path, expected_version="1.0")
    assert (
        dist._validate_sdist_surface(config, archive, {"__init__.py": "digest"}) is None
    )


def test_unreadable_distribution_archives_are_normalized(tmp_path: Path) -> None:
    wheel = tmp_path / "bad.whl"
    sdist = tmp_path / "bad.tar.gz"
    wheel.write_bytes(b"not zip")
    sdist.write_bytes(b"not tar")
    with pytest.raises(dist.ReleasePreflightError, match="readable wheel"):
        dist._validate_wheel_metadata(wheel, "1.0")
    with pytest.raises(dist.ReleasePreflightError, match="readable source"):
        dist._validate_sdist_metadata(sdist, "1.0")


def test_checkout_runtime_surface_rejects_missing_links_and_unknown_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(dist.ReleasePreflightError, match="directory is missing"):
        dist._checkout_runtime_files(tmp_path)
    package = tmp_path / "src" / "invarlock"
    package.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("x=1\n", encoding="utf-8")
    link = package / "linked.py"
    link.symlink_to(outside)
    with pytest.raises(dist.ReleasePreflightError, match="must not contain links"):
        dist._checkout_runtime_files(tmp_path)
    link.unlink()
    (package / "binary.bin").write_bytes(b"binary")
    with pytest.raises(dist.ReleasePreflightError, match="unexpected file"):
        dist._checkout_runtime_files(tmp_path)


def test_checkout_runtime_hashes_supported_files_and_ignores_build_noise(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "invarlock"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("x=1\n", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    (package / ".DS_Store").write_bytes(b"ignored")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored")
    assert set(dist._checkout_runtime_files(tmp_path)) == {"__init__.py", "py.typed"}
    assert dist._directory_is_needed("", {"module.py": "digest"})
    assert dist._directory_is_needed("nested", {"nested/module.py": "digest"})
    assert not dist._directory_is_needed("extra", {"module.py": "digest"})


def test_empty_checkout_runtime_package_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "src" / "invarlock").mkdir(parents=True)
    with pytest.raises(dist.ReleasePreflightError, match="has no source files"):
        dist._checkout_runtime_files(tmp_path)


def test_wheel_runtime_surface_rejects_file_root_and_unneeded_directory() -> None:
    root_file = zipfile.ZipInfo("invarlock")
    with pytest.raises(dist.ReleasePreflightError, match="top-level payload"):
        dist._validate_wheel_surface(
            [root_file], dist_info_root="invarlock-1.0.dist-info"
        )
    extra_directory = zipfile.ZipInfo("invarlock/extra/")
    with pytest.raises(dist.ReleasePreflightError, match="unexpected runtime package"):
        dist._validate_wheel_package_directories(
            [extra_directory], {"__init__.py": "digest"}
        )


def test_egg_info_directories_are_inert() -> None:
    member = _tar_member("invarlock-1.0/src/invarlock.egg-info", kind=tarfile.DIRTYPE)
    assert dist._validate_egg_info_member(member, "nested") is None


def test_distribution_directory_requires_exact_real_pair(tmp_path: Path) -> None:
    with pytest.raises(dist.ReleasePreflightError, match="real directory"):
        dist._find_distribution_artifacts(tmp_path / "missing")
    distribution = tmp_path / "dist"
    distribution.mkdir()
    with pytest.raises(dist.ReleasePreflightError, match="exactly one"):
        dist._find_distribution_artifacts(distribution)
    (distribution / "pkg.whl").write_bytes(b"wheel")
    (distribution / "pkg.tar.gz").write_bytes(b"sdist")
    wheel, sdist = dist._find_distribution_artifacts(distribution)
    assert wheel.name == "pkg.whl"
    assert sdist.name == "pkg.tar.gz"


def test_distribution_directory_rejects_symlinked_entries(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    target = tmp_path / "candidate.whl"
    target.write_bytes(b"wheel")
    (distribution / "candidate.whl").symlink_to(target)
    with pytest.raises(dist.ReleasePreflightError, match="symbolic links"):
        dist._find_distribution_artifacts(distribution)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("not-a-hash file.whl\n", "not sha256sum"),
        (("a" * 64) + " ../escape.whl\n", "unsafe or duplicate"),
        (("a" * 64) + " other.whl\n", "exactly the built"),
    ],
)
def test_hash_manifest_rejects_invalid_unsafe_or_wrong_entries(
    tmp_path: Path, contents: str, expected: str
) -> None:
    manifest = tmp_path / "hashes.txt"
    manifest.write_text(contents, encoding="utf-8")
    with pytest.raises(dist.ReleasePreflightError, match=expected):
        dist._load_hash_manifest(manifest, {"candidate.whl"})


def test_hash_manifest_accepts_comments_and_star_filename(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.txt"
    manifest.write_text(
        "# generated\n\n" + ("a" * 64) + " *candidate.whl\n", encoding="utf-8"
    )
    assert dist._load_hash_manifest(manifest, {"candidate.whl"}) == {
        "candidate.whl": "a" * 64
    }


def test_hash_manifest_rejects_oversized_input(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.txt"
    manifest.write_bytes(b"x" * (dist.MAX_METADATA_BYTES + 1))
    with pytest.raises(dist.ReleasePreflightError, match="too large"):
        dist._load_hash_manifest(manifest, set())
