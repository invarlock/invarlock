"""Keep image preparation attributable and reject untrusted build inputs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import pytest

from examples.qualification import k2_runtime_build as build


def _wheel_bytes(
    *,
    metadata=b"Name: invarlock\nVersion: 0.15.1\n",
    member="invarlock-0.15.1.dist-info/METADATA",
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, metadata)
    return buffer.getvalue()


def _inputs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    runtime = root / "examples/qualification/k2-horizon/runtime"
    runtime.mkdir(parents=True)
    for name in ("Dockerfile", "os-security-pins.txt"):
        (runtime / name).write_text(name)
    for name in (
        "k2_campaign.py",
        "k2_producer.py",
        "k2_native_probe.py",
        "k2_runtime_source.py",
        "k2_runtime_build.py",
        "k2_runtime_apt.py",
        "k2_runtime_expat.py",
    ):
        (root / "examples/qualification" / name).write_text(name)
    (runtime.parent / "catalog.json").write_text(
        json.dumps({"reviewed_source_files": {"python/sglang/model.py": "a" * 64}})
    )
    lock = root / "requirements.txt"
    pip = tmp_path / build.PIP_WHEEL
    pip.write_bytes(b"pip fixture")
    pip_digest = hashlib.sha256(pip.read_bytes()).hexdigest()
    lock.write_text(f"pip==26.2 --hash=sha256:{pip_digest}\n")
    monkeypatch.setattr(build, "PIP_WHEEL_SHA256", pip_digest)
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "RUNTIME", runtime)
    monkeypatch.setattr(build, "LOCK", lock)
    wheel = tmp_path / "invarlock-0.15.1-py3-none-any.whl"
    wheel.write_bytes(_wheel_bytes())
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()

    def derive(archive, output):
        output.mkdir()
        (output / "source-derivation.json").write_text("{}")
        return {"excluded_operations": ["Outlines grammar backend"]}

    monkeypatch.setattr(build.source, "prepare", derive)
    monkeypatch.setattr(
        build.expat,
        "prepared_inputs",
        lambda *_: {"expat/source-authentication.json": b"{}"},
    )
    monkeypatch.setattr(
        build, "_apt_inputs", lambda *args: {"apt/deb-artifacts.sha256": b"fixture"}
    )
    return wheel, expected


def test_context_binds_exact_wheel_source_and_campaign_inputs(tmp_path, monkeypatch):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "context"
    result = build.prepare(
        tmp_path / "archive",
        wheel,
        expected,
        output,
        expat_bundle=None,
        pip_wheel=tmp_path / build.PIP_WHEEL,
        apt_bundle=None,
        expected_apt_manifest="fixture",
    )
    assert result["status"] == "prepared_not_built"
    assert result["core_wheel_filename"] == wheel.name
    assert result["core_distribution_version"] == "0.15.1"
    assert result["input_sha256"][f"core/{wheel.name}"] == expected
    pip = tmp_path / build.PIP_WHEEL
    assert (
        result["input_sha256"][f"bootstrap/{build.PIP_WHEEL}"]
        == hashlib.sha256(pip.read_bytes()).hexdigest()
    )
    assert (output / "bootstrap" / build.PIP_WHEEL).read_bytes() == pip.read_bytes()
    assert (output / "bootstrap/pip-wheel.sha256").read_text() == (
        f"{build.PIP_WHEEL_SHA256}  /usr/share/invarlock-k2/bootstrap/{build.PIP_WHEEL}\n"
    )
    assert result["reviewed_source_files"] == {"python/sglang/model.py": "a" * 64}
    assert (output / "core" / wheel.name).read_bytes() == wheel.read_bytes()
    assert json.loads((output / "build-inputs.json").read_text()) == result
    with pytest.raises(FileExistsError):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            output,
            expat_bundle=None,
            pip_wheel=tmp_path / build.PIP_WHEEL,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )


@pytest.mark.parametrize("expected", ["not-a-hash", "0" * 64])
def test_wrong_core_identity_cannot_create_build_context(
    tmp_path, monkeypatch, expected
):
    wheel, _ = _inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="wheel"):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            tmp_path / "output",
            expat_bundle=None,
            pip_wheel=tmp_path / build.PIP_WHEEL,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "name",
    [
        "core.whl",
        "other-0.15.1-py3-none-any.whl",
        "invarlock-0.15.1-cp312-cp312-linux_x86_64.whl",
        "invarlock-0.15.1-1-py3-none-any.whl",
        "Invarlock-0.15.1-py3-none-any.whl",
        "invarlock-0.15.1-py2.py3-none-any.whl",
    ],
)
def test_core_wheel_rejects_noncanonical_or_unsupported_filenames(
    tmp_path, monkeypatch, name
):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    wheel = wheel.rename(tmp_path / name)
    with pytest.raises(ValueError, match="filename"):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            tmp_path / "output",
            expat_bundle=None,
            pip_wheel=tmp_path / build.PIP_WHEEL,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "name",
        "version",
        "missing",
        "duplicate_header",
        "duplicate_member",
        "extra_metadata",
        "oversized",
        "invalid_zip",
        "wrong_member",
    ],
)
def test_core_wheel_rejects_false_or_ambiguous_embedded_identity(
    tmp_path, monkeypatch, kind
):
    wheel, _ = _inputs(tmp_path, monkeypatch)
    metadata = b"Name: invarlock\nVersion: 0.15.1\n"
    member = "invarlock-0.15.1.dist-info/METADATA"
    if kind == "name":
        metadata = metadata.replace(b"invarlock", b"another")
    elif kind == "version":
        metadata = metadata.replace(b"0.15.1", b"0.15.0")
    elif kind == "missing":
        metadata = b"Name: invarlock\n"
    elif kind == "duplicate_header":
        metadata += b"Version: 0.15.1\n"
    elif kind == "oversized":
        metadata += b"a" * 65536
    elif kind == "wrong_member":
        member = "invarlock-0.15.0.dist-info/METADATA"
    data = _wheel_bytes(metadata=metadata, member=member)
    if kind in {"duplicate_member", "extra_metadata"}:
        buffer = io.BytesIO(data)
        with zipfile.ZipFile(buffer, "a") as archive:
            if kind == "duplicate_member":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(member, metadata)
            else:
                archive.writestr("another-1.dist-info/METADATA", metadata)
        data = buffer.getvalue()
    elif kind == "invalid_zip":
        data = b"not a wheel archive"
    wheel.write_bytes(data)
    with pytest.raises(ValueError, match="metadata|archive"):
        build.prepare(
            tmp_path / "archive",
            wheel,
            hashlib.sha256(data).hexdigest(),
            tmp_path / "output",
            expat_bundle=None,
            pip_wheel=tmp_path / build.PIP_WHEEL,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )
    assert not (tmp_path / "output").exists()


def test_failed_derivation_removes_only_its_new_context(tmp_path, monkeypatch):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("keep")

    def reject(*args):
        raise ValueError("wrong archive")

    monkeypatch.setattr(build.source, "prepare", reject)
    with pytest.raises(ValueError, match="archive"):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            tmp_path / "output",
            expat_bundle=None,
            pip_wheel=tmp_path / build.PIP_WHEEL,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )
    assert not (tmp_path / "output").exists()
    assert unrelated.read_text() == "keep"


def test_input_reader_refuses_special_symlink_and_oversized_files(tmp_path):
    regular = tmp_path / "regular"
    regular.write_bytes(b"test")
    with pytest.raises(ValueError, match="bounded"):
        build._read(regular, 1)
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(OSError):
        build._read(link, 10)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        build._read(fifo, 10)


@pytest.mark.parametrize(
    "kind", ["missing", "tampered", "symlink", "fifo", "oversized", "renamed", "sdist"]
)
def test_bootstrap_rejects_substituted_or_unbounded_wheels_before_output(
    tmp_path, monkeypatch, kind
):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    pip = tmp_path / build.PIP_WHEEL
    if kind == "missing":
        pip.unlink()
    elif kind == "tampered":
        pip.write_bytes(b"untrusted code")
    elif kind in {"symlink", "fifo"}:
        pip.unlink()
        if kind == "symlink":
            pip.symlink_to(wheel)
        else:
            os.mkfifo(pip)
    elif kind == "oversized":
        with pip.open("wb") as stream:
            stream.truncate(4 * 1024 * 1024 + 1)
    elif kind == "renamed":
        pip = pip.rename(tmp_path / "pip-24.0-py3-none-any.whl")
    else:
        pip.write_bytes(b"sdist fixture")
        build.LOCK.write_text(
            build.LOCK.read_text().rstrip()
            + f" --hash=sha256:{hashlib.sha256(pip.read_bytes()).hexdigest()}\n"
        )
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, OSError)):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            output,
            expat_bundle=None,
            pip_wheel=pip,
            apt_bundle=None,
            expected_apt_manifest="fixture",
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["absent", "version", "hash", "duplicate", "option"])
def test_bootstrap_requires_one_exact_hash_bound_lock_record(
    tmp_path, monkeypatch, kind
):
    _inputs(tmp_path, monkeypatch)
    lock = build.LOCK.read_bytes()
    if kind == "absent":
        lock = b"other==1\n"
    elif kind == "version":
        lock = lock.replace(b"26.2", b"24.0")
    elif kind == "hash":
        lock = lock.replace(build.PIP_WHEEL_SHA256.encode(), b"0" * 64)
    elif kind == "duplicate":
        lock += lock
    else:
        lock = lock.rstrip() + b" --extra-index-url=https://example.invalid\n"
    with pytest.raises(ValueError, match="maintained lock"):
        build._pip_inputs(tmp_path / build.PIP_WHEEL, lock)


def test_bootstrap_accepts_continued_hashes_without_accepting_the_sdist(
    tmp_path, monkeypatch
):
    _inputs(tmp_path, monkeypatch)
    lock = (
        "pip==26.2 \\\n"
        f"    --hash=sha256:{'a' * 64} \\\n"
        f"    --hash=sha256:{build.PIP_WHEEL_SHA256}\nother==1\n"
    ).encode()
    payloads = build._pip_inputs(tmp_path / build.PIP_WHEEL, lock)
    assert payloads[f"bootstrap/{build.PIP_WHEEL}"] == b"pip fixture"


def test_cli_requires_and_passes_the_independent_os_manifest(tmp_path, monkeypatch):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    assert (
        build.main(
            [
                "--archive",
                str(tmp_path / "archive"),
                "--expat-bundle",
                str(tmp_path / "expat"),
                "--pip-wheel",
                str(tmp_path / build.PIP_WHEEL),
                "--core-wheel",
                str(wheel),
                "--expected-core-wheel-sha256",
                expected,
                "--apt-bundle",
                str(tmp_path / "apt"),
                "--expected-apt-manifest-sha256",
                "a" * 64,
                "--output",
                str(tmp_path / "context"),
            ]
        )
        == 0
    )


def _os_bundle(tmp_path, *, manifest=None):
    bundle = tmp_path / "apt"
    (bundle / "debs").mkdir(parents=True)
    (bundle / "debs/package_1_amd64.deb").write_bytes(b"package fixture")
    hashed = hashlib.sha256(b"package fixture").hexdigest()
    data = (
        manifest
        if manifest is not None
        else f"{hashed}  /out/debs/package_1_amd64.deb\n".encode()
    )
    (bundle / "deb-artifacts.sha256").write_bytes(data)
    (bundle / "deb-packages.tsv").write_text("package_1_amd64.deb\tpackage\t1\tamd64\n")
    (bundle / "repository-metadata").mkdir()
    (bundle / "repository-metadata/ubuntu.sources").write_bytes(b"repository fixture")
    (bundle / "package-indexes").mkdir()
    (bundle / "package-indexes/ubuntu-archive-keyring.gpg").write_bytes(
        b"public key fixture"
    )
    return bundle, hashlib.sha256(data).hexdigest()


def test_os_bundle_authenticates_every_selected_package(tmp_path, monkeypatch):
    bundle, expected = _os_bundle(tmp_path)
    monkeypatch.setattr(
        build.apt,
        "verify",
        lambda *args: {
            "indexes": [],
            "keyring_sha256": hashlib.sha256(b"public key fixture").hexdigest(),
            "package_table_sha256": hashlib.sha256(
                (bundle / "deb-packages.tsv").read_bytes()
            ).hexdigest(),
        },
    )
    values = build._apt_inputs(bundle, expected)
    assert values["apt/debs/package_1_amd64.deb"] == b"package fixture"
    (bundle / "debs/package_1_amd64.deb").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity"):
        build._apt_inputs(bundle, expected)
    with pytest.raises(ValueError, match="independent expected"):
        build._apt_inputs(bundle, "0" * 64)


@pytest.mark.parametrize("kind", ["empty", "path", "duplicate"])
def test_os_bundle_rejects_ambiguous_or_empty_manifests(tmp_path, kind):
    hashed = hashlib.sha256(b"package fixture").hexdigest()
    line = f"{hashed}  /out/debs/package_1_amd64.deb\n"
    data = (
        b""
        if kind == "empty"
        else (line * 2).encode()
        if kind == "duplicate"
        else ("0" * 64 + "  /out/debs/../escape.deb\n").encode()
    )
    bundle, expected = _os_bundle(tmp_path, manifest=data)
    with pytest.raises(ValueError, match="empty|path"):
        build._apt_inputs(bundle, expected)


def test_real_build_cli_rejects_wrong_wheel_before_output(tmp_path):
    import subprocess
    import sys

    wheel = tmp_path / "core.whl"
    wheel.write_bytes(b"wrong")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.qualification.k2_runtime_build",
            "--archive",
            str(tmp_path / "source.tar.gz"),
            "--expat-bundle",
            str(tmp_path / "expat"),
            "--pip-wheel",
            str(tmp_path / build.PIP_WHEEL),
            "--core-wheel",
            str(wheel),
            "--expected-core-wheel-sha256",
            "0" * 64,
            "--apt-bundle",
            str(tmp_path / "apt"),
            "--expected-apt-manifest-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "core wheel identity differs" in result.stderr
    assert not (tmp_path / "output").exists()


def test_context_copies_only_replayed_index_bytes_and_rejects_later_changes(
    tmp_path, monkeypatch
):
    bundle, expected = _os_bundle(tmp_path)
    index = bundle / "package-indexes/reviewed_Packages.xz"
    index.write_bytes(b"authenticated index fixture")
    release = bundle / "repository-metadata/reviewed_InRelease"
    release.write_bytes(b"authenticated release fixture")
    report = {
        "indexes": [
            {
                "path": index.name,
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "release": release.name,
                "release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
            }
        ],
        "keyring_sha256": hashlib.sha256(b"public key fixture").hexdigest(),
        "package_table_sha256": hashlib.sha256(
            (bundle / "deb-packages.tsv").read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(build.apt, "verify", lambda *args: report)
    retained = build._apt_inputs(bundle, expected)
    assert retained["apt/package-indexes/reviewed_Packages.xz"] == index.read_bytes()
    assert (
        retained["apt/repository-metadata/reviewed_InRelease"] == release.read_bytes()
    )

    def changed_after_replay(*args):
        index.write_bytes(b"substituted after authentication")
        return report

    monkeypatch.setattr(build.apt, "verify", changed_after_replay)
    with pytest.raises(ValueError, match="metadata changed"):
        build._apt_inputs(bundle, expected)


@pytest.mark.parametrize("reader", [build._read, build.apt.read_input])
def test_build_input_growth_after_stat_still_obeys_the_read_bound(
    tmp_path, monkeypatch, reader
):
    path = tmp_path / "changing"
    path.write_bytes(b"a")
    original = os.fstat

    def observe_then_grow(descriptor):
        observed = original(descriptor)
        with path.open("ab") as stream:
            stream.write(b"bc")
        return observed

    monkeypatch.setattr(os, "fstat", observe_then_grow)
    with pytest.raises(ValueError, match="size bound"):
        reader(path, 2)


@pytest.mark.parametrize("failure", ["signature", "index"])
def test_cli_reports_signature_and_compressed_index_failures_as_input_errors(
    tmp_path, monkeypatch, capsys, failure
):
    import lzma
    import subprocess

    def reject(*args, **kwargs):
        if failure == "signature":
            raise subprocess.CalledProcessError(2, ["gpgv"])
        raise lzma.LZMAError("invalid package index")

    monkeypatch.setattr(build, "prepare", reject)
    with pytest.raises(SystemExit) as caught:
        build.main(
            [
                "--archive",
                "source.tar.gz",
                "--expat-bundle",
                str(tmp_path / "expat"),
                "--pip-wheel",
                str(tmp_path / build.PIP_WHEEL),
                "--core-wheel",
                "core.whl",
                "--expected-core-wheel-sha256",
                "0" * 64,
                "--apt-bundle",
                "apt",
                "--expected-apt-manifest-sha256",
                "0" * 64,
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert caught.value.code == 2
    assert "K2 runtime build:" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
