"""Keep image preparation attributable and reject untrusted build inputs."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from examples.qualification import k2_runtime_build as build


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
    ):
        (root / "examples/qualification" / name).write_text(name)
    (runtime.parent / "catalog.json").write_text(
        json.dumps({"reviewed_source_files": {"python/sglang/model.py": "a" * 64}})
    )
    lock = root / "requirements.txt"
    lock.write_text("package==1\n")
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "RUNTIME", runtime)
    monkeypatch.setattr(build, "LOCK", lock)
    wheel = tmp_path / "core.whl"
    wheel.write_bytes(b"authenticated fixture")
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()

    def derive(archive, output):
        output.mkdir()
        (output / "source-derivation.json").write_text("{}")
        return {"excluded_operations": ["Outlines grammar backend"]}

    monkeypatch.setattr(build.source, "prepare", derive)
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
        apt_bundle=None,
        expected_apt_manifest="fixture",
    )
    assert result["status"] == "prepared_not_built"
    assert result["input_sha256"]["core.whl"] == expected
    assert result["reviewed_source_files"] == {"python/sglang/model.py": "a" * 64}
    assert (output / "core.whl").read_bytes() == wheel.read_bytes()
    assert json.loads((output / "build-inputs.json").read_text()) == result
    with pytest.raises(FileExistsError):
        build.prepare(
            tmp_path / "archive",
            wheel,
            expected,
            output,
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


def test_cli_requires_and_passes_the_independent_os_manifest(tmp_path, monkeypatch):
    wheel, expected = _inputs(tmp_path, monkeypatch)
    assert (
        build.main(
            [
                "--archive",
                str(tmp_path / "archive"),
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
