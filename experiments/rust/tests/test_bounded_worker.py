"""Identity binding must not retain an entire extension member in memory."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from invarlock_rust_experiment import bounded_worker, worker


def wheels(tmp_path, monkeypatch, *, large=False):
    installed = tmp_path / "installed"
    metadata = {}
    for key, prefix in (
        ("root_wheel", "invarlock"),
        ("native_wheel", "invarlock_native"),
    ):
        name = prefix + "/payload.so"
        destination = installed / name
        destination.parent.mkdir(parents=True)
        wheel_path = tmp_path / (prefix + ".whl")
        block = bytes(range(256)) * 1024
        count = 96 if large and key == "native_wheel" else 1
        with (
            destination.open("wb") as disk,
            zipfile.ZipFile(
                wheel_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive,
        ):
            with archive.open(name, "w") as member:
                for _ in range(count):
                    disk.write(block)
                    member.write(block)
            archive.writestr(
                prefix + "-0.1.dist-info/RECORD", b"mutable installation record"
            )
        metadata[key] = {"path": str(wheel_path), "sha256": worker.sha_file(wheel_path)}

    class Distribution:
        def locate_file(self, name):
            return installed / name

    monkeypatch.setattr(
        worker.importlib.metadata, "distribution", lambda _: Distribution()
    )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    return path, metadata, installed


def test_large_compressed_member_uses_bounded_reads_and_matches_existing_binding(
    tmp_path, monkeypatch
):
    path, _, _ = wheels(tmp_path, monkeypatch, large=True)
    expected = worker.check_wheel_binding(path)
    original = zipfile.ZipExtFile.read
    sizes = []

    def bounded(self, size=-1):
        assert 0 < size <= 256 * 1024
        sizes.append(size)
        return original(self, size)

    def whole_member_forbidden(*args, **kwargs):
        raise AssertionError("whole member allocation")

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded)
    monkeypatch.setattr(zipfile.ZipFile, "read", whole_member_forbidden)
    assert bounded_worker.check_wheel_binding(path) == expected
    assert len(sizes) >= 96


@pytest.mark.parametrize("target", ["installed", "wheel", "crc"])
def test_changed_or_corrupt_bytes_are_still_rejected(tmp_path, monkeypatch, target):
    path, metadata, installed = wheels(tmp_path, monkeypatch)
    if target == "installed":
        (installed / "invarlock_native/payload.so").write_bytes(b"different")
    elif target == "wheel":
        with Path(metadata["native_wheel"]["path"]).open("ab") as stream:
            stream.write(b"changed wheel bytes")
    else:
        wheel = Path(metadata["native_wheel"]["path"])
        # Retain a readable archive and deliberately contradict its member CRC.
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(
                installed / "invarlock_native/payload.so", "invarlock_native/payload.so"
            )
        raw = bytearray(wheel.read_bytes())
        offset = raw.index(b"PK\x01\x02") + 16
        raw[offset] ^= 1
        wheel.write_bytes(raw)
        metadata["native_wheel"]["sha256"] = hashlib.sha256(raw).hexdigest()
        path.write_text(json.dumps(metadata))
    with pytest.raises((RuntimeError, zipfile.BadZipFile)):
        bounded_worker.check_wheel_binding(path)


def test_main_preserves_worker_failure_exit_status(monkeypatch):
    from invarlock_rust_experiment import bounded_worker, full_worker

    monkeypatch.setattr(full_worker, "main", lambda: 1)
    monkeypatch.setattr(full_worker, "check_wheel_binding", lambda _: {})
    assert bounded_worker.main() == 1
