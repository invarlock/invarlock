from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import runtime_release_asset_handoff as handoff
from tests.scripts._runtime_release_evidence_test_support import (
    RELEASE_COMMIT,
    RELEASE_TAG,
    REPOSITORY,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_COMMIT,
    build_legacy_asset,
    stage_legacy_asset,
)


@pytest.mark.parametrize("digest", ["bad", "A" * 64, "a" * 63])
def test_asset_filename_rejects_noncanonical_digest(digest: str) -> None:
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="digest"):
        handoff._asset_filename(
            release_tag=RELEASE_TAG,
            source_commit=SOURCE_COMMIT,
            asset_sha256=digest,
        )


def test_asset_filename_rejects_abbreviated_source_commit() -> None:
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="full lowercase"):
        handoff._asset_filename(
            release_tag=RELEASE_TAG,
            source_commit=SOURCE_COMMIT[:12],
            asset_sha256="a" * 64,
        )


def test_bounded_reader_rejects_missing_symlink_directory_and_oversize(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="safely readable"
    ):
        handoff._read_regular_file(tmp_path / "missing", label="input", limit=2)

    regular = tmp_path / "regular"
    regular.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="safely readable"
    ):
        handoff._read_regular_file(link, label="input", limit=4)
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="bounded regular"
    ):
        handoff._read_regular_file(tmp_path, label="input", limit=4)
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="bounded regular"
    ):
        handoff._read_regular_file(regular, label="input", limit=2)


def test_bounded_reader_detects_growth_mutation_and_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    real_read = handoff.os.read

    monkeypatch.setattr(handoff.os, "read", lambda _fd, _size: b"xx")
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="byte limit"):
        handoff._read_regular_file(source, label="input", limit=1)

    def fail_read(_fd: int, _size: int) -> bytes:
        raise OSError("read failed")

    monkeypatch.setattr(handoff.os, "read", fail_read)
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="could not be read"
    ):
        handoff._read_regular_file(source, label="input", limit=2)

    monkeypatch.setattr(handoff.os, "read", real_read)
    real_fstat = handoff.os.fstat
    calls = 0

    def changed_fstat(fd: int):
        nonlocal calls
        calls += 1
        value = real_fstat(fd)
        if calls == 1:
            return value
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(handoff.os, "fstat", changed_fstat)
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="changed"):
        handoff._read_regular_file(source, label="input", limit=2)


@pytest.mark.parametrize("failure", ["zero", "write", "fsync"])
def test_exclusive_writer_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output = tmp_path / "output"
    if failure == "zero":
        monkeypatch.setattr(handoff.os, "write", lambda _fd, _view: 0)
    elif failure == "write":

        def fail_write(_fd: int, _view: memoryview) -> int:
            raise OSError("write failed")

        monkeypatch.setattr(handoff.os, "write", fail_write)
    else:

        def fail_fsync(_fd: int) -> None:
            raise OSError("fsync failed")

        monkeypatch.setattr(handoff.os, "fsync", fail_fsync)

    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="could not be written"
    ):
        handoff._write_exclusive(output, b"payload", label="output")
    assert not output.exists()


def test_exclusive_writer_creates_owner_readonly_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    handoff._write_exclusive(output, b"payload", label="output")

    assert output.read_bytes() == b"payload"
    assert stat.S_IMODE(output.stat().st_mode) == handoff._PRIVATE_FILE_MODE == 0o400


def test_verify_rejects_digest_filename_asset_bytes_and_missing_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    common = {
        "asset": asset,
        "release_tag": RELEASE_TAG,
        "expected_source_commit": SOURCE_COMMIT,
        "expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "expected_asset_sha256": str(result["asset_sha256"]),
        "expected_providers": frozenset({"llama_cpp"}),
        "expected_qualifications": frozenset({"llama_cpp"}),
        "require_behavioral_claim": False,
    }

    wrong_digest_name = digest_file.with_name("wrong.sha256")
    wrong_digest_name.write_bytes(digest_file.read_bytes())
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="canonical name"):
        handoff.verify_handoff(digest_file=wrong_digest_name, **common)

    asset.chmod(0o644)
    asset.write_bytes(asset.read_bytes() + b"tampered")
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="digest changed"):
        handoff.verify_handoff(digest_file=digest_file, **common)

    asset.write_bytes(asset.read_bytes()[: -len(b"tampered")])
    monkeypatch.setattr(
        handoff.runtime_release_evidence,
        "validate_asset",
        lambda *_args, **_kwargs: {"qualification_count": "one"},
    )
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="omitted"):
        handoff.verify_handoff(digest_file=digest_file, **common)


def test_stage_rejects_unsafe_directory_and_wrong_source_digest(tmp_path: Path) -> None:
    source, digest = build_legacy_asset(tmp_path)
    missing = tmp_path / "missing"
    link = tmp_path / "output-link"
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    link.symlink_to(real_output, target_is_directory=True)
    common = {
        "source_asset": source,
        "release_tag": RELEASE_TAG,
        "expected_source_commit": SOURCE_COMMIT,
        "expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "expected_asset_sha256": digest,
        "expected_providers": frozenset({"llama_cpp"}),
        "expected_qualifications": frozenset({"llama_cpp"}),
        "require_behavioral_claim": False,
    }
    for output in (missing, link, source):
        with pytest.raises(
            handoff.RuntimeReleaseAssetHandoffError, match="non-symlink directory"
        ):
            handoff.stage_handoff(output_dir=output, **common)

    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="digest does not match"
    ):
        handoff.stage_handoff(
            output_dir=real_output,
            **{**common, "expected_asset_sha256": "f" * 64},
        )


def test_stage_rolls_back_both_outputs_when_final_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, digest = build_legacy_asset(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        handoff,
        "verify_handoff",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("verification failed")),
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        handoff.stage_handoff(
            source_asset=source,
            output_dir=output,
            release_tag=RELEASE_TAG,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=digest,
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )
    assert not list(output.iterdir())


def test_main_dispatches_stage_verify_upload_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [
        "--release-tag",
        RELEASE_TAG,
        "--expected-source-commit",
        SOURCE_COMMIT,
        "--expected-source-archive-sha256",
        SOURCE_ARCHIVE_SHA256,
        "--expected-asset-sha256",
        "a" * 64,
        "--expected-provider",
        "llama_cpp",
        "--expected-qualification",
        "llama_cpp",
    ]
    source = tmp_path / "asset"
    digest = tmp_path / "asset.sha256"
    output = tmp_path / "output"
    output.mkdir()

    monkeypatch.setattr(handoff, "stage_handoff", lambda **_kwargs: {"status": "stage"})
    assert (
        handoff.main(
            ["stage", "--asset", str(source), "--output-dir", str(output), *common]
        )
        == 0
    )
    assert capsys.readouterr().out == '{"status":"stage"}\n'

    monkeypatch.setattr(
        handoff, "verify_handoff", lambda **_kwargs: {"status": "verify"}
    )
    assert (
        handoff.main(
            ["verify", "--asset", str(source), "--digest-file", str(digest), *common]
        )
        == 0
    )
    assert capsys.readouterr().out == '{"status":"verify"}\n'

    monkeypatch.setattr(
        handoff, "upload_handoff", lambda **_kwargs: {"status": "upload"}
    )
    assert (
        handoff.main(
            [
                "upload",
                "--asset",
                str(source),
                "--digest-file",
                str(digest),
                "--repository",
                REPOSITORY,
                "--expected-release-commit",
                RELEASE_COMMIT,
                *common,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == '{"status":"upload"}\n'

    def fail_stage(**_kwargs: object) -> dict[str, object]:
        raise handoff.RuntimeReleaseAssetHandoffError("closed")

    monkeypatch.setattr(handoff, "stage_handoff", fail_stage)
    with pytest.raises(SystemExit) as raised:
        handoff.main(
            ["stage", "--asset", str(source), "--output-dir", str(output), *common]
        )
    assert raised.value.code == 2
    assert "closed" in capsys.readouterr().err

    monkeypatch.setattr(handoff, "stage_handoff", lambda **_kwargs: None)
    with pytest.raises(SystemExit) as missing_result:
        handoff.main(
            ["stage", "--asset", str(source), "--output-dir", str(output), *common]
        )
    assert missing_result.value.code == 2
    assert "without a handoff result" in capsys.readouterr().err
