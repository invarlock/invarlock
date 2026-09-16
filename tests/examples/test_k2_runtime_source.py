"""Authenticate the restricted runtime source before extracting or changing it."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

from examples.qualification import k2_runtime_source as runtime


def _archive(tmp_path, monkeypatch, *, extra=None, metadata=None):
    files = {
        "python/pyproject.toml": metadata
        or b'[project]\nname="sglang"\ndependencies=[\n  "outlines==0.1.11",\n  "torch==2.13.0",\n]\n',
        "python/sglang/srt/constrained/outlines_backend.py": b'"""Optional grammar."""\nimport outlines\n',
        "python/sglang/srt/constrained/outlines_jump_forward.py": b'"""Optional cache."""\nfrom outlines.caching import cache\n',
        "python/sglang/srt/models/xllm.py": b'"""Native model fixture; never executed."""\n',
    }
    if extra:
        files.update(extra)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in files.items():
            entry = tarfile.TarInfo(runtime.PREFIX + name)
            entry.size = len(data)
            stream.addfile(entry, io.BytesIO(data))
    monkeypatch.setattr(
        runtime, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        runtime,
        "PATCH_HASHES",
        {
            name: hashlib.sha256(files[name]).hexdigest()
            for name in runtime.PATCH_HASHES
        },
    )
    return archive, files


def test_unauthenticated_source_never_creates_output(tmp_path):
    path = tmp_path / "wrong.tar.gz"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="archive identity"):
        runtime.prepare(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_restricted_source_removes_dependency_and_rejects_optional_imports(
    tmp_path, monkeypatch
):
    archive, files = _archive(tmp_path, monkeypatch)
    output = tmp_path / "output"
    manifest = runtime.prepare(archive, output)
    assert (output / "python/sglang/srt/models/xllm.py").read_bytes() == files[
        "python/sglang/srt/models/xllm.py"
    ]
    assert b'"outlines' not in (output / "python/pyproject.toml").read_bytes()
    assert b'"torch==2.13.0"' in (output / "python/pyproject.toml").read_bytes()
    assert set(manifest["changed_files"]) == set(runtime.PATCH_HASHES)
    for index, name in enumerate(runtime.BLOCKED_MODULES):
        spec = importlib.util.spec_from_file_location(
            f"excluded_{index}", output / name
        )
        with pytest.raises(
            RuntimeError, match="unavailable in the restricted K2 runtime"
        ):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert json.loads((output / "source-derivation.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        runtime.prepare(archive, output)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "python/../../escape"])
def test_even_authenticated_fixture_cannot_escape_destination(
    tmp_path, monkeypatch, name
):
    archive, _ = _archive(tmp_path, monkeypatch, extra={name: b"bad"})
    with pytest.raises(ValueError, match="path"):
        runtime.prepare(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_patch_hash_and_exact_dependency_preconditions_fail_closed(
    tmp_path, monkeypatch
):
    archive, _ = _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime, "PATCH_HASHES", dict.fromkeys(runtime.PATCH_HASHES, "0" * 64)
    )
    with pytest.raises(ValueError, match="reviewed file"):
        runtime.prepare(archive, tmp_path / "output")
    second = tmp_path / "second"
    second.mkdir()
    archive, _ = _archive(second, monkeypatch, metadata=b"[project]\ndependencies=[]\n")
    with pytest.raises(ValueError, match="dependency"):
        runtime.prepare(archive, tmp_path / "output")


def _rewrite(archive, monkeypatch, append):
    with tarfile.open(archive) as source:
        entries = [
            (item, source.extractfile(item).read()) for item in source if item.isfile()
        ]
    with tarfile.open(archive, "w:gz") as target:
        for item, data in entries:
            target.addfile(item, io.BytesIO(data))
        for item, data in append:
            target.addfile(item, None if data is None else io.BytesIO(data))
    monkeypatch.setattr(
        runtime, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )


def test_duplicate_member_and_unknown_link_are_rejected(tmp_path, monkeypatch):
    for index, kind in enumerate(("duplicate", "link", "fifo", "root")):
        folder = tmp_path / str(index)
        folder.mkdir()
        archive, files = _archive(folder, monkeypatch)
        item = tarfile.TarInfo(runtime.PREFIX + "extra")
        data = b""
        if kind == "duplicate":
            item.name = runtime.PREFIX + "python/pyproject.toml"
        elif kind == "link":
            item.type, item.linkname, data = tarfile.SYMTYPE, "../../outside", None
        elif kind == "fifo":
            item.type, data = tarfile.FIFOTYPE, None
        else:
            item.name = "another-root/file"
        _rewrite(archive, monkeypatch, [(item, data)])
        with pytest.raises(ValueError, match="duplicate|unsupported|path"):
            runtime.prepare(archive, folder / "output")
        assert not (folder / "output").exists()


def test_reviewed_symlink_is_reified_and_missing_target_rejected(tmp_path, monkeypatch):
    archive, files = _archive(tmp_path, monkeypatch, extra={"target": b"safe"})
    monkeypatch.setattr(runtime, "LINKS", {"link": "target"})
    link = tarfile.TarInfo(runtime.PREFIX + "link")
    link.type, link.linkname = tarfile.SYMTYPE, "target"
    directory = tarfile.TarInfo(runtime.PREFIX + "folder/")
    directory.type = tarfile.DIRTYPE
    root = tarfile.TarInfo(runtime.PREFIX)
    root.type = tarfile.DIRTYPE
    _rewrite(archive, monkeypatch, [(root, None), (directory, None), (link, None)])
    runtime.prepare(archive, tmp_path / "output")
    assert (tmp_path / "output/link").read_bytes() == b"safe"
    assert not (tmp_path / "output/link").is_symlink()
    monkeypatch.setattr(runtime, "LINKS", {"missing": "absent"})
    link.name, link.linkname = runtime.PREFIX + "missing", "absent"
    other = tmp_path / "other"
    other.mkdir()
    archive, _ = _archive(other, monkeypatch)
    _rewrite(archive, monkeypatch, [(link, None)])
    with pytest.raises(ValueError, match="target"):
        runtime.prepare(archive, other / "output")


def test_expanded_archive_bound_is_checked_before_output(tmp_path, monkeypatch):
    archive, _ = _archive(tmp_path, monkeypatch)

    class TooMany:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getmembers(self):
            return [None] * 20001

    monkeypatch.setattr(runtime.tarfile, "open", lambda **kwargs: TooMany())
    with pytest.raises(ValueError, match="bound"):
        runtime.prepare(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_reports_candidate_source_and_rejects_untrusted_bytes(
    tmp_path, monkeypatch, capsys
):
    archive, _ = _archive(tmp_path, monkeypatch)
    assert (
        runtime.main(["--archive", str(archive), "--output", str(tmp_path / "output")])
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "source_prepared_not_runtime_ready"
    )
    with pytest.raises(SystemExit) as caught:
        runtime.main(
            [
                "--archive",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "another"),
            ]
        )
    assert caught.value.code == 2


def test_executable_cli_rejects_source_without_creating_output(tmp_path):
    import subprocess
    import sys

    archive = tmp_path / "untrusted.tar.gz"
    archive.write_bytes(b"untrusted")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(runtime.__file__).resolve()),
            "--archive",
            str(archive),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "archive identity" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_archive_input_must_be_a_regular_file_without_blocking(tmp_path, kind):
    import os
    import subprocess
    import sys

    archive = tmp_path / "input.tar.gz"
    if kind == "fifo":
        os.mkfifo(archive)
    else:
        target = tmp_path / "target.tar.gz"
        target.write_bytes(b"untrusted")
        archive.symlink_to(target)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(runtime.__file__).resolve()),
            "--archive",
            str(archive),
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert result.returncode == 2
    assert not (tmp_path / "output").exists()
