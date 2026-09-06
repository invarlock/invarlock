"""Authenticate whole-release Expat inputs and reject ambiguous identities."""

from __future__ import annotations

import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from examples.qualification import k2_runtime_expat as expat


def signature(signing=expat.SIGNING_KEY, primary=expat.PRIMARY_KEY):
    return f"[GNUPG:] VALIDSIG {signing} 2026-08-31 1788182984 0 4 0 1 8 00 {primary}\n".encode()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    data = {name: name.encode() for name in expat.INPUT_HASHES}
    for name, payload in data.items():
        (tmp_path / name).write_bytes(payload)
    monkeypatch.setattr(
        expat,
        "INPUT_HASHES",
        {name: hashlib.sha256(payload).hexdigest() for name, payload in data.items()},
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=signature(), stderr=b"")

    monkeypatch.setattr(expat.subprocess, "run", run)
    return tmp_path, data, calls


def test_exact_release_and_isolated_signature(bundle):
    root, data, calls = bundle
    assert expat.authenticate(root) == data
    assert len(calls) == 2
    for command, options in calls:
        assert "--no-autostart" in command and "--no-auto-key-retrieve" in command
        assert "--no-options" in command and "--homedir" in command
        assert options["timeout"] == 30 and options["check"] is True
    assert "--import" in calls[0][0]
    assert "--verify" in calls[1][0]


@pytest.mark.parametrize("name", list(expat.INPUT_HASHES))
def test_changed_release_component_rejected_before_gpg(bundle, name):
    root, _, calls = bundle
    (root / name).write_bytes(b"substituted")
    with pytest.raises(ValueError, match="identity differs"):
        expat.authenticate(root)
    assert calls == []


@pytest.mark.parametrize(
    "status",
    [
        b"",
        signature() * 2,
        signature(signing="0" * 40),
        signature(primary="0" * 40),
        signature() + b"[GNUPG:] REVKEYSIG bad\n",
        signature().replace(b" 8 00 ", b" 2 00 "),
        b"[GNUPG:] VALIDSIG incomplete\n",
    ],
)
def test_signature_identity_and_status_fail_closed(status):
    with pytest.raises(ValueError):
        expat.valid_signer(status)


def test_failed_gpg_does_not_authenticate(bundle, monkeypatch):
    root, *_ = bundle

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(expat.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        expat.authenticate(root)


def test_signature_output_is_bounded(bundle, monkeypatch):
    root, *_ = bundle
    monkeypatch.setattr(
        expat.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"x" * 65537, stderr=b""),
    )
    with pytest.raises(ValueError, match="output exceeds"):
        expat.authenticate(root)


def test_input_rejects_symlink_directory_and_size(tmp_path):
    (tmp_path / "file").write_bytes(b"1234")
    with pytest.raises(ValueError, match="bounded regular"):
        expat.read(tmp_path / "file", 3)
    with pytest.raises(ValueError, match="bounded regular"):
        expat.read(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "file")
    with pytest.raises(OSError):
        expat.read(tmp_path / "link")


def test_prepared_manifest_binds_verified_release(bundle):
    import json

    root, data, _ = bundle
    prepared = expat.prepared_inputs(root)
    assert {name: prepared["expat/" + name] for name in data} == data
    report = json.loads(prepared["expat/source-authentication.json"])
    assert report["input_sha256"] == expat.INPUT_HASHES
    assert report["package_version"] == expat.PACKAGE_VERSION


def test_source_archive_extraction_and_escape_rejection(tmp_path):
    import io
    import tarfile

    def archive(name, *, symlink=False):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:xz") as output:
            directory = tarfile.TarInfo("expat-2.8.4/")
            directory.type = tarfile.DIRTYPE
            output.addfile(directory)
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                output.addfile(member)
            else:
                member.size = 4
                output.addfile(member, io.BytesIO(b"data"))
        return stream.getvalue()

    root = expat.unpack_source(archive("expat-2.8.4/lib/source.c"), tmp_path / "good")
    assert (root / "lib/source.c").read_bytes() == b"data"
    for index, name in enumerate(("expat-2.8.4/../escape", "other/file", "/absolute")):
        with pytest.raises(ValueError, match="unsupported member"):
            expat.unpack_source(archive(name), tmp_path / str(index))
    with pytest.raises(ValueError, match="unsupported member"):
        expat.unpack_source(
            archive("expat-2.8.4/link", symlink=True), tmp_path / "link"
        )


def test_package_inventory_records_actual_bytes_and_bounded_links(tmp_path):
    (tmp_path / "DEBIAN").mkdir()
    (tmp_path / "DEBIAN/control").write_text("metadata")
    (tmp_path / "library").write_bytes(b"actual library")
    (tmp_path / "link").symlink_to("library")
    result = expat.file_inventory(tmp_path)
    assert result == {
        "library": {"sha256": hashlib.sha256(b"actual library").hexdigest()},
        "link": {"symlink": "library"},
    }
    (tmp_path / "escape").symlink_to("../outside")
    with pytest.raises(ValueError, match="library directory"):
        expat.file_inventory(tmp_path)


def metadata(name):
    return {
        "Package": name,
        "Version": expat.PACKAGE_VERSION,
        "Architecture": "amd64",
        "Source": f"expat ({expat.PACKAGE_VERSION})",
        "Depends": f"libexpat1 (= {expat.PACKAGE_VERSION})"
        if name.endswith("-dev")
        else "libc6 (>= 2.38)",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("Version", "2.6.1"),
        ("Architecture", "arm64"),
        ("Source", "ubuntu"),
        ("Depends", "libexpat1"),
    ],
)
def test_package_metadata_cannot_relabel_or_mismatch_dev(key, value):
    fields = metadata("libexpat1-dev")
    expat.validate_package_identity(fields, "libexpat1-dev")
    fields[key] = value
    with pytest.raises(ValueError):
        expat.validate_package_identity(fields, "libexpat1-dev")


@pytest.fixture
def package_build(tmp_path, monkeypatch):
    import platform
    from pathlib import Path

    source = tmp_path / "source"
    (source / "lib").mkdir(parents=True)
    (source / "COPYING").write_text("upstream license")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        expat, "authenticate", lambda *_: {"expat-2.8.4.tar.xz": b"archive"}
    )
    monkeypatch.setattr(expat, "unpack_source", lambda *_: source)
    monkeypatch.setattr(
        expat, "verify_installed", lambda output: {"verified": str(output)}
    )
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "cmake" and "-B" in command:
            build = Path(command[command.index("-B") + 1])
            build.mkdir()
            (build / "expat_config.h").write_text("generated config")
        elif command[:2] == ["cmake", "--install"]:
            build = Path(command[2])
            stage = Path(kwargs["env"]["DESTDIR"])
            lib = stage / expat.LIBDIR
            lib.mkdir(parents=True)
            stem = "libexpatw" if "libexpatw" in build.name else "libexpat"
            if build.name.endswith("-shared"):
                (lib / (stem + ".so.1.12.4")).write_bytes(b"shared")
                (lib / (stem + ".so.1")).symlink_to(stem + ".so.1.12.4")
                (lib / (stem + ".so")).symlink_to(stem + ".so.1")
            else:
                (lib / (stem + ".a")).write_bytes(b"static")
            (stage / "usr/include").mkdir(parents=True)
            (stage / "usr/include/expat.h").write_text("public header")
            (stage / "usr/share/doc/expat").mkdir(parents=True)
            (stage / "usr/share/doc/expat/AUTHORS").write_text("upstream authors")
        elif command[0] == "readelf":
            return (
                "[libexpatw.so.1]" if "libexpatw" in command[-1] else "[libexpat.so.1]"
            ).encode()
        elif command[0] == "nm":
            return b"XML_ExpatVersion T 1 2\n"
        elif command[0] == "dpkg-shlibdeps":
            return b"shlibs:Depends=libc6 (>= 2.38)\n"
        elif command[:2] == ["dpkg-deb", "--root-owner-group"]:
            Path(command[-1]).write_bytes(Path(command[-2]).name.encode())
        elif command[:2] == ["dpkg-deb", "--field"]:
            return (
                "\n".join(
                    k + ": " + v
                    for k, v in metadata(Path(command[-1]).name.split("_")[0]).items()
                )
                + "\n"
            ).encode()
        return b"build log\n"

    monkeypatch.setattr(expat, "execute", execute)
    return tmp_path, calls, execute


def test_build_packages_keep_both_widths_static_dev_and_distinct_identity(
    package_build,
):
    import json

    root, calls, _ = package_build
    output = root / "result"
    assert expat.build_install(root, output) == {"verified": str(output)}
    report = json.loads((output / "build-report.json").read_text())
    files = report["installed_files"]
    for stem in ("libexpat", "libexpatw"):
        for suffix in (".a", ".so", ".so.1", ".so.1.12.4"):
            assert str(expat.LIBDIR / (stem + suffix)) in files
    assert len(report["package_sha256"]) == 2
    assert {name for name in files if name.startswith("usr/share/doc/")} == {
        "usr/share/doc/libexpat1/copyright",
        "usr/share/doc/libexpat1-dev/copyright",
    }
    assert len([c for c in calls if c[0] == "ctest"]) == 2
    assert len([c for c in calls if c[0] == "cc"]) == 4
    assert report["native_parser_checks"] == [
        "narrow-shared",
        "narrow-static",
        "wide-shared",
        "wide-static",
    ]
    assert any(c[0] == "dpkg" and c[1] == "--install" for c in calls)
    assert calls[-1] == ["ldconfig"]


@pytest.mark.parametrize("failure", ["abi", "dependencies", "metadata"])
def test_package_build_fails_before_install_on_bad_abi_or_metadata(
    package_build, monkeypatch, failure
):
    root, calls, execute = package_build

    def changed(command, **kwargs):
        if failure == "abi" and command[0] == "readelf":
            return b"[libexpat.so.2]"
        if failure == "dependencies" and command[0] == "dpkg-shlibdeps":
            return b"unknown metadata"
        if failure == "metadata" and command[:2] == ["dpkg-deb", "--field"]:
            return b"Package: wrong\n"
        return execute(command, **kwargs)

    monkeypatch.setattr(expat, "execute", changed)
    with pytest.raises(ValueError):
        expat.build_install(root, root / "result")
    assert not any(c[0] == "dpkg" for c in calls)


def test_package_build_rejects_platform_and_existing_output(
    tmp_path, monkeypatch, bundle
):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux"):
        expat.build_install(tmp_path, tmp_path)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(FileExistsError):
        expat.build_install(bundle[0], tmp_path)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    import ctypes
    import json
    import shutil
    from pathlib import Path

    root, output = tmp_path / "system", tmp_path / "evidence"
    output.mkdir()
    trees = {}
    for package in expat.PACKAGES:
        tree = tmp_path / package
        (tree / expat.LIBDIR).mkdir(parents=True)
        trees[package] = tree
    for stem in ("libexpat", "libexpatw"):
        runtime = trees["libexpat1"] / expat.LIBDIR
        development = trees["libexpat1-dev"] / expat.LIBDIR
        (runtime / (stem + ".so.1.12.4")).write_bytes(b"exact shared artifact")
        (runtime / (stem + ".so.1")).symlink_to(stem + ".so.1.12.4")
        (development / (stem + ".a")).write_bytes(b"exact static artifact")
        (development / (stem + ".so")).symlink_to(stem + ".so.1")
    files = {}
    artifacts = {}
    for package, tree in trees.items():
        shutil.copytree(tree, root, dirs_exist_ok=True, symlinks=True)
        files.update(expat.file_inventory(tree))
        name = f"{package}_{expat.PACKAGE_VERSION}_amd64.deb"
        (output / name).write_bytes(package.encode())
        artifacts[name] = hashlib.sha256(package.encode()).hexdigest()
    report = {
        "format": "invarlock/k2-expat-build-v1",
        "source_version": expat.VERSION,
        "package_version": expat.PACKAGE_VERSION,
        "input_sha256": expat.INPUT_HASHES,
        "recipe_sha256": hashlib.sha256(expat.read(Path(expat.__file__))).hexdigest(),
        "package_sha256": artifacts,
        "installed_files": files,
    }
    (output / "build-report.json").write_text(json.dumps(report))
    state = {
        "package_version": expat.PACKAGE_VERSION + " amd64 installed",
        "library_version": b"expat_2.8.4",
        "python_version": b"expat_2.8.4",
        "overlap": False,
    }

    def execute(command, **kwargs):
        if command[:2] == ["dpkg-deb", "--field"]:
            return (
                "\n".join(
                    k + ": " + v
                    for k, v in metadata(Path(command[-1]).name.split("_")[0]).items()
                )
                + "\n"
            ).encode()
        if command[0] == "dpkg-query":
            return state["package_version"].encode()
        if command[:2] == ["dpkg-deb", "--extract"]:
            package = Path(command[-2]).name.split("_")[0]
            shutil.copytree(trees[package], Path(command[-1]), symlinks=True)
            if state["overlap"]:
                (Path(command[-1]) / "duplicate").write_text("same")
            return b""
        return state["python_version"]

    monkeypatch.setattr(expat, "execute", execute)

    def library(_):
        return SimpleNamespace(XML_ExpatVersion=lambda: state["library_version"])

    monkeypatch.setattr(ctypes, "CDLL", library)
    return root, output, report, state


def test_installed_verification_recomputes_deb_payload_and_library_versions(installed):
    root, output, report, _ = installed
    result = expat.verify_installed(output, root)
    assert result["source_version"] == "2.8.4"
    assert result["installed_file_count"] == 8
    assert result["package_sha256"] == report["package_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "report",
        "artifact_set",
        "artifact_bytes",
        "installed_version",
        "overlap",
        "recorded_payload",
        "installed_bytes",
        "link",
        "old_library",
        "loaded_version",
        "python_version",
    ],
)
def test_installed_evidence_cannot_mask_changed_packages_or_loaded_code(
    installed, change
):
    import json

    root, output, report, state = installed
    if change == "report":
        report["source_version"] = "2.6.1"
    elif change == "artifact_set":
        report["package_sha256"].pop(next(iter(report["package_sha256"])))
    elif change == "artifact_bytes":
        (output / next(iter(report["package_sha256"]))).write_bytes(b"changed")
    elif change == "installed_version":
        state["package_version"] = "2.6.1 amd64 installed"
    elif change == "overlap":
        state["overlap"] = True
    elif change == "recorded_payload":
        report["installed_files"].pop(next(iter(report["installed_files"])))
    elif change == "installed_bytes":
        (root / expat.LIBDIR / "libexpat.a").write_bytes(b"changed")
    elif change == "link":
        path = root / expat.LIBDIR / "libexpat.so"
        path.unlink()
        path.symlink_to("libexpat.so.9")
    elif change == "old_library":
        (root / expat.LIBDIR / "libexpat.so.1.9.1").write_bytes(b"old")
    elif change == "loaded_version":
        state["library_version"] = b"expat_2.6.1"
    else:
        state["python_version"] = b"expat_2.6.1"
    (output / "build-report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        expat.verify_installed(output, root)


def test_executor_retains_subprocess_failure():
    import sys

    assert expat.execute([sys.executable, "-c", 'print("ok")']) == b"ok\n"
    with pytest.raises(subprocess.CalledProcessError):
        expat.execute([sys.executable, "-c", "raise SystemExit(3)"])


def test_expat_cli_dispatches_and_returns_actual_result(tmp_path, monkeypatch, capsys):
    import json

    calls = []
    monkeypatch.setattr(
        expat,
        "verify_installed",
        lambda output: calls.append(output) or {"verified": True},
    )
    expat.main(["verify", "--output", str(tmp_path)])
    assert calls == [tmp_path]
    assert json.loads(capsys.readouterr().out) == {"verified": True}
    monkeypatch.setattr(
        expat, "build_install", lambda bundle, output: {"source_version": "2.8.4"}
    )
    expat.main(
        ["build-install", "--bundle", str(tmp_path), "--output", str(tmp_path / "out")]
    )
    assert json.loads(capsys.readouterr().out) == {"source_version": "2.8.4"}


def test_expat_archive_member_bound(tmp_path):
    import io
    import tarfile

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for index in range(1001):
            member = tarfile.TarInfo(f"expat-2.8.4/{index}")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    with pytest.raises(ValueError, match="size or member bound"):
        expat.unpack_source(stream.getvalue(), tmp_path)


def test_build_log_retains_stderr_and_partial_failure_output(tmp_path):
    import sys

    log = tmp_path / "build.log"
    expat.execute([sys.executable, "-c", 'print("configured")'], log=log)
    with pytest.raises(subprocess.CalledProcessError):
        expat.execute(
            [
                sys.executable,
                "-c",
                'import sys; print("partial build"); print("compiler diagnostic",file=sys.stderr); raise SystemExit(2)',
            ],
            log=log,
        )
    data = log.read_text()
    assert (
        "configured\n" in data
        and "partial build\n" in data
        and "compiler diagnostic\n" in data
    )
    assert (
        expat.execute(
            [
                sys.executable,
                "-c",
                'import sys; print("identity"); print("warning",file=sys.stderr)',
            ]
        )
        == b"identity\n"
    )
