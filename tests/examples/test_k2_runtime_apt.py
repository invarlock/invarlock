"""Reject broken signed-index and package identity links before image preparation."""

from __future__ import annotations

import hashlib
import lzma
from pathlib import Path

import pytest

from examples.qualification import k2_runtime_apt as apt


def test_signed_hash_section_is_closed_and_unambiguous():
    record = b"SHA256:\n " + b"a" * 64 + b" 3 main/binary-amd64/Packages\nSHA512:\n"
    assert apt.release_entries(record) == {"main/binary-amd64/Packages": ("a" * 64, 3)}
    with pytest.raises(ValueError, match="duplicate"):
        apt.release_entries(record.replace(b"SHA512:", record.split(b"\n")[1]))
    with pytest.raises(ValueError, match="SHA256"):
        apt.release_entries(b"SHA512:\n")


def test_index_decompression_enforces_identity_size_and_single_stream():
    raw = b"Package: example\nVersion: 1\n\n"
    compressed = lzma.compress(raw)
    expected = (hashlib.sha256(raw).hexdigest(), len(raw))
    assert apt.decode_index(compressed, expected) == raw
    assert apt.decode_index(compressed + lzma.compress(b""), expected) == raw
    for changed, claim in [
        (compressed, ("0" * 64, len(raw))),
        (compressed, (expected[0], 1)),
        (compressed + b"extra", expected),
        (b"bad", expected),
    ]:
        with pytest.raises((ValueError, lzma.LZMAError)):
            apt.decode_index(changed, claim)


def test_package_records_reject_duplicate_fields_and_oversized_records():
    assert (
        list(
            apt.package_records(
                b"Package: example\nDescription: text\n continuation\nVersion: 1\n\n"
            )
        )[0]["Version"]
        == "1"
    )
    with pytest.raises(ValueError, match="duplicate"):
        list(apt.package_records(b"Package: first\nPackage: second\n\n"))
    with pytest.raises(ValueError, match="bound"):
        list(apt.package_records(b"Package: " + b"x" * (1024 * 1024)))


def _bundle(tmp_path, monkeypatch, *, package_digest=None, size=None):
    root = tmp_path / "bundle"
    metadata = root / "repository-metadata"
    indexes = root / "package-indexes"
    metadata.mkdir(parents=True)
    indexes.mkdir()
    keyring = b"explicit public key fixture"
    (indexes / "ubuntu-archive-keyring.gpg").write_bytes(keyring)
    monkeypatch.setattr(apt, "KEYRING_SHA256", hashlib.sha256(keyring).hexdigest())
    monkeypatch.setattr(
        apt,
        "RELEASES",
        {"fixture_InRelease": "https://archive.ubuntu.com/ubuntu/dists/noble"},
    )
    monkeypatch.setattr(apt, "COMPONENTS", ("main",))
    monkeypatch.setattr(apt, "verify_signature", lambda data, key: data)
    data = b"selected package fixture"
    hashed = package_digest or hashlib.sha256(data).hexdigest()
    raw = f"Package: example\nVersion: 1\nArchitecture: amd64\nSHA256: {hashed}\nSize: {len(data) if size is None else size}\nFilename: pool/main/e/example.deb\n\n".encode()
    compressed = lzma.compress(raw)
    release = f"Origin: Ubuntu\nLabel: Ubuntu\nSuite: noble\nCodename: noble\nSHA256:\n {hashlib.sha256(raw).hexdigest()} {len(raw)} main/binary-amd64/Packages\n {hashlib.sha256(compressed).hexdigest()} {len(compressed)} main/binary-amd64/Packages.xz\n".encode()
    (metadata / "fixture_InRelease").write_bytes(release)
    (indexes / "fixture_main_Packages.xz").write_bytes(compressed)
    (root / "deb-packages.tsv").write_text("example_1_amd64.deb\texample\t1\tamd64\n")
    return root, {"example_1_amd64.deb": data}, compressed


def test_every_package_has_a_replayed_signed_index_link(tmp_path, monkeypatch):
    root, selected, _ = _bundle(tmp_path, monkeypatch)
    report = apt.verify(root, selected)
    assert report["status"] == "signed_indexes_and_payloads_verified"
    assert (
        report["packages"][0]["sha256"]
        == hashlib.sha256(selected["example_1_amd64.deb"]).hexdigest()
    )
    assert report["indexes"][0]["release"] == "fixture_InRelease"
    assert (
        report["package_table_sha256"]
        == hashlib.sha256((root / "deb-packages.tsv").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mode",
    [
        "compressed",
        "hash",
        "size",
        "bad_size",
        "bad_hash",
        "table_duplicate",
        "table_shape",
        "table_missing",
        "extra",
    ],
)
def test_changed_or_unlinked_artifacts_cannot_receive_a_success_report(
    tmp_path, monkeypatch, mode
):
    root, selected, _ = _bundle(
        tmp_path,
        monkeypatch,
        package_digest="0" * 64
        if mode == "hash"
        else "not-a-hash"
        if mode == "bad_hash"
        else None,
        size=1 if mode == "size" else "unknown" if mode == "bad_size" else None,
    )
    if mode == "compressed":
        (root / "package-indexes/fixture_main_Packages.xz").write_bytes(b"changed")
    if mode == "table_duplicate":
        path = root / "deb-packages.tsv"
        path.write_bytes(path.read_bytes() * 2)
    if mode == "table_shape":
        (root / "deb-packages.tsv").write_text("ambiguous\n")
    if mode == "table_missing":
        (root / "deb-packages.tsv").write_text("")
    if mode == "extra":
        (root / "package-indexes/unselected").write_text("untrusted")
    with pytest.raises(ValueError):
        apt.verify(root, selected)


def test_signature_uses_only_the_pinned_keyring_and_captured_release_bytes(monkeypatch):
    keyring = b"trusted fixture"
    release = b"-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\nSigned fixture\n-----BEGIN PGP SIGNATURE-----\nfixture\n-----END PGP SIGNATURE-----\n"
    monkeypatch.setattr(apt, "KEYRING_SHA256", hashlib.sha256(keyring).hexdigest())
    calls = []

    def run(command, **kwargs):
        assert Path(command[-1]).read_bytes() == release
        assert Path(command[-2]).read_bytes() == keyring
        assert kwargs["check"] and kwargs["timeout"] == 30
        calls.append(command)
        return __import__("types").SimpleNamespace(stdout=b"verified plaintext")

    monkeypatch.setattr(apt.subprocess, "run", run)
    assert apt.verify_signature(release, keyring) == b"verified plaintext"
    assert calls[0][0] == "gpgv"
    with pytest.raises(ValueError, match="keyring"):
        apt.verify_signature(release, keyring + b"changed")


def test_fetch_checks_bytes_before_retention_and_never_overwrites(
    tmp_path, monkeypatch
):
    root, selected, compressed = _bundle(tmp_path, monkeypatch)
    path = root / "package-indexes/fixture_main_Packages.xz"
    with pytest.raises(FileExistsError):
        apt.fetch(root)
    path.unlink()
    monkeypatch.setattr(
        apt.urllib.request,
        "urlopen",
        lambda *args, **kwargs: __import__("io").BytesIO(b"bad"),
    )
    with pytest.raises(ValueError, match="downloaded"):
        apt.fetch(root)
    assert not path.exists()
    monkeypatch.setattr(
        apt.urllib.request,
        "urlopen",
        lambda *args, **kwargs: __import__("io").BytesIO(compressed),
    )
    assert apt.main(["--bundle", str(root)]) == 0
    assert apt.verify(root, selected)["packages"]
    with pytest.raises(SystemExit) as caught:
        apt.main(["--bundle", str(root)])
    assert caught.value.code == 2


def test_decompression_and_signed_size_limits_stop_oversized_inputs(
    tmp_path, monkeypatch
):
    root, _, _ = _bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "INDEX_LIMIT", 1)
    with pytest.raises(ValueError, match="bound"):
        list(apt.signed_indexes(root))
    with pytest.raises(ValueError, match="bound"):
        apt.decode_index(b"", ("0" * 64, 2))
    monkeypatch.setattr(apt, "INDEX_LIMIT", 100)
    with pytest.raises(ValueError, match="streams"):
        apt.decode_index(lzma.compress(b"") * 65, (hashlib.sha256(b"").hexdigest(), 0))


def test_regular_file_reader_rejects_links_fifos_and_large_inputs(tmp_path):
    import os

    regular = tmp_path / "regular"
    regular.write_bytes(b"too large")
    with pytest.raises(ValueError, match="bounded"):
        apt.read_input(regular, 1)
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(OSError):
        apt.read_input(link, 100)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        apt.read_input(fifo, 100)


def test_record_without_final_blank_line_and_malformed_hash_entry():
    assert list(
        apt.package_records(b"\nPackage: example\nVersion: 1\nX-Cargo-Built-Using:")
    ) == [{"Package": "example", "Version": "1", "X-Cargo-Built-Using": ""}]
    with pytest.raises(ValueError, match="SHA256"):
        apt.release_entries(b"SHA256:\n unsupported\n")


def test_real_ubuntu_signature_excludes_unsigned_wrapper_fields():
    import shutil
    import subprocess

    if shutil.which("gpgv") is None:
        pytest.skip("gpgv is required for the real Ubuntu signing-chain test")
    fixture = Path(__file__).parent / "fixtures/ubuntu"
    release = (fixture / "noble-backports.InRelease").read_bytes()
    keyring = (fixture / "ubuntu-archive-keyring.gpg").read_bytes()
    plaintext = apt.verify_signature(release, keyring)
    assert plaintext.startswith(b"Origin: Ubuntu\n")
    assert "main/binary-amd64/Packages.xz" in apt.release_entries(plaintext)
    injection = b"\nSHA256:\n " + b"1" * 64 + b" 12 unsigned-injected-index\n"
    for modified in (
        injection + release,
        release + injection,
        release + release,
        release.replace(
            b"-----BEGIN PGP SIGNATURE-----",
            b"-----BEGIN PGP SIGNATURE-----\n-----BEGIN PGP SIGNATURE-----",
        ),
    ):
        with pytest.raises(ValueError, match="clear-signed"):
            apt.verify_signature(modified, keyring)
    with pytest.raises(subprocess.CalledProcessError):
        apt.verify_signature(
            release.replace(b"Origin: Ubuntu", b"Origin: Forged"), keyring
        )


@pytest.mark.parametrize("mode", ["suite", "duplicate"])
def test_signed_release_identity_cannot_be_relabelled(tmp_path, monkeypatch, mode):
    root, _, _ = _bundle(tmp_path, monkeypatch)
    path = root / "repository-metadata/fixture_InRelease"
    data = path.read_bytes()
    path.write_bytes(
        data.replace(b"Suite: noble", b"Suite: elsewhere")
        if mode == "suite"
        else data.replace(b"Origin: Ubuntu", b"Origin: Ubuntu\nOrigin: Ubuntu")
    )
    with pytest.raises(ValueError, match="identity"):
        list(apt.signed_indexes(root))


def test_verified_plaintext_is_bounded(monkeypatch):
    keyring = b"trusted fixture"
    monkeypatch.setattr(apt, "KEYRING_SHA256", hashlib.sha256(keyring).hexdigest())
    release = b"-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\nFixture\n-----BEGIN PGP SIGNATURE-----\nfixture\n-----END PGP SIGNATURE-----\n"
    monkeypatch.setattr(
        apt.subprocess,
        "run",
        lambda *args, **kwargs: __import__("types").SimpleNamespace(
            stdout=b"x" * (1024 * 1024 + 1)
        ),
    )
    with pytest.raises(ValueError, match="plaintext.*bound"):
        apt.verify_signature(release, keyring)
