from __future__ import annotations

import importlib.metadata
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock
from scripts.release import gguf_runtime_blackbox as blackbox
from tests.scripts._gguf_blackbox_support import (
    valid_result,
    write_json,
    write_side_bundle,
)


def _stat(*, mode: int = stat.S_IFREG | 0o600, size: int = 1, mtime: int = 1):
    return SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=mode,
        st_size=size,
        st_mtime_ns=mtime,
        st_ctime_ns=1,
    )


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("nonregular", "regular file"),
        ("short_read", "changed while hashing"),
        ("grew", "changed while hashing"),
        ("metadata", "changed while hashing"),
        ("read_error", "read safely"),
    ],
)
def test_descriptor_hashing_detects_file_substitution_and_races(
    monkeypatch: pytest.MonkeyPatch, failure: str, expected: str
) -> None:
    stats = [_stat()]
    reads: list[bytes | OSError] = [b"x", b""]
    if failure == "nonregular":
        stats = [_stat(mode=stat.S_IFDIR | 0o700)]
    elif failure == "short_read":
        reads = [b""]
    elif failure == "grew":
        reads = [b"x", b"extra"]
    elif failure == "metadata":
        stats = [_stat(), _stat(mtime=2)]
    elif failure == "read_error":
        reads = [OSError("device failure")]

    def read(_descriptor: int, _size: int) -> bytes:
        value = reads.pop(0)
        if isinstance(value, OSError):
            raise value
        return value

    monkeypatch.setattr(blackbox, "FIXTURE_BYTE_LENGTH", 1)
    monkeypatch.setattr(blackbox.os, "open", lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(blackbox.os, "close", lambda _descriptor: None)
    monkeypatch.setattr(blackbox.os, "fstat", lambda _descriptor: stats.pop(0))
    monkeypatch.setattr(blackbox.os, "read", read)

    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._sha256_file(Path("fixture.gguf"))


def test_result_digest_binding_is_independent_of_structural_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    result = valid_result(image_digest=digest)
    monkeypatch.setattr(blackbox, "SCORING_OBSERVATION_SHA256", "0" * 64)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="observation digest"):
        blackbox._validate_result_payload(
            blackbox._canonical_json(result) + b"\n", image_digest=digest
        )


def test_installed_wheel_guard_accepts_distribution_bound_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_file = Path(invarlock.__file__).resolve(strict=True)
    distribution_root = package_file.parents[1]
    distribution = SimpleNamespace(locate_file=lambda _name: distribution_root)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(
        blackbox.importlib.metadata, "distribution", lambda _name: distribution
    )

    assert blackbox._require_installed_wheel() is None


def test_installed_wheel_guard_normalizes_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(
        blackbox.importlib.metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError("invarlock")
        ),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="wheel is unavailable"):
        blackbox._require_installed_wheel()


def test_private_writer_normalizes_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        blackbox.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="could not be written"):
        blackbox._write_canonical_new(tmp_path / "input.json", {"secret": True})


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (0, b"{}", "framing"),
        (0, b"not-json\n", "result is invalid"),
        (0, b'{"format_version": "v1", "ok": true}\n', "not canonical"),
        (0, b'{"format_version":"other","ok":true}\n', "format"),
        (2, b'{"format_version":"v1","ok":true}\n', "outcome"),
    ],
)
def test_installed_cli_rejects_malformed_or_inconsistent_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: bytes,
    expected: str,
) -> None:
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (status, payload, b"private stderr"),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._run_installed_cli(("build-schedule",), expected_format="v1")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'{"key":1,"key":2}', "unique JSON object"),
        (b"not-json", "valid JSON"),
        (b"[]", "unique JSON object"),
        (b'{"key": 1}', "canonical encoding"),
    ],
)
def test_portable_json_rejects_ambiguous_encodings(
    tmp_path: Path, payload: bytes, expected: str
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(payload)
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._portable_json(artifact)


def test_portable_json_rejects_missing_unbounded_and_nonregular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(blackbox.GGUFBlackBoxError, match="could not be read"):
        blackbox._portable_json(tmp_path / "missing.json")
    with pytest.raises(blackbox.GGUFBlackBoxError, match="bounded regular"):
        blackbox._portable_json(tmp_path)

    artifact = tmp_path / "large.json"
    artifact.write_bytes(b"{}")
    monkeypatch.setattr(blackbox, "_MAX_RESULT_BYTES", 1)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="bounded regular"):
        blackbox._portable_json(artifact)


def test_cli_side_rejects_unexpected_or_unreadable_bundle(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "b" * 64
    side = tmp_path / "side"
    write_side_bundle(side, role="baseline", image_digest=digest)
    (side / "extra.json").write_bytes(b"{}")
    with pytest.raises(blackbox.GGUFBlackBoxError, match="unexpected file set"):
        blackbox._validate_cli_side(side, role="baseline", image_digest=digest)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="cannot be inspected"):
        blackbox._validate_cli_side(
            tmp_path / "missing", role="baseline", image_digest=digest
        )


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("observation", "observation does not match"),
        ("digest", "observation digest"),
        ("report", "report did not verify"),
    ],
)
def test_cli_side_rejects_tampered_observation_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: str,
) -> None:
    digest = "sha256:" + "c" * 64
    side = tmp_path / failure
    write_side_bundle(side, role="baseline", image_digest=digest)
    if failure == "observation":
        write_json(side / "runtime-scoring.observation.json", {"status": "wrong"})
    elif failure == "digest":
        monkeypatch.setattr(blackbox, "CLI_SCORING_OBSERVATION_SHA256", "0" * 64)
    else:
        report = json.loads((side / "evaluation.report.json").read_bytes())
        report["score"] = 0.0
        write_json(side / "evaluation.report.json", report)

    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._validate_cli_side(side, role="baseline", image_digest=digest)


def _closed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
    ):
        monkeypatch.setenv(name, "0")


def test_cli_journey_rejects_missing_or_nonexecutable_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _closed_environment(monkeypatch)
    monkeypatch.setattr(blackbox, "_CONTAINER_CLI", str(tmp_path / "missing"))
    with pytest.raises(blackbox.GGUFBlackBoxError, match="CLI is unavailable"):
        blackbox._inside_cli_journey(image_digest="sha256:" + "d" * 64)

    cli = tmp_path / "invarlock"
    cli.write_text("not executable", encoding="utf-8")
    monkeypatch.setattr(blackbox, "_CONTAINER_CLI", str(cli))
    with pytest.raises(blackbox.GGUFBlackBoxError, match="not executable"):
        blackbox._inside_cli_journey(image_digest="sha256:" + "d" * 64)


def test_cli_journey_rejects_preexisting_private_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _closed_environment(monkeypatch)
    cli = tmp_path / "invarlock"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o700)
    work = tmp_path / "already-exists"
    work.mkdir()
    monkeypatch.setattr(blackbox, "_CONTAINER_CLI", str(cli))
    monkeypatch.setattr(blackbox, "_CONTAINER_WORK_ROOT", str(work))
    with pytest.raises(blackbox.GGUFBlackBoxError, match="could not be created"):
        blackbox._inside_cli_journey(image_digest="sha256:" + "e" * 64)
