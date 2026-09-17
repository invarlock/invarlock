from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from scripts.release import verify_hosted_distributions as verifier


def _write_ledger(path: Path, lines: list[str]) -> str:
    payload = ("\n".join(lines) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _expected_files() -> dict[str, str]:
    return {
        "invarlock-1.2.3-py3-none-any.whl": "1" * 64,
        "invarlock-1.2.3.tar.gz": "2" * 64,
    }


def _metadata(expected: dict[str, str]) -> bytes:
    return json.dumps(
        {
            "urls": [
                {
                    "filename": name,
                    "digests": {"sha256": digest},
                    "url": f"https://files.example.invalid/{name}",
                }
                for name, digest in expected.items()
            ]
        }
    ).encode()


def test_open_url_sets_release_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    response = io.BytesIO(b"response")

    def fake_urlopen(request: Any, *, timeout: float) -> io.BytesIO:
        observed["request"] = request
        observed["timeout"] = timeout
        return response

    monkeypatch.setattr(verifier.urllib.request, "urlopen", fake_urlopen)

    assert verifier._open_url("https://example.invalid/file", timeout=7) is response
    request = observed["request"]
    assert request.full_url == "https://example.invalid/file"
    assert request.get_header("User-agent") == "invarlock-release-verifier/1"
    assert observed["timeout"] == 7


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            [
                f"{'1' * 64}  dist/invarlock-1.2.3-py3-none-any.whl",
                f"{'2' * 64}  other/invarlock-1.2.3-py3-none-any.whl",
            ],
            "duplicate distribution filename",
        ),
        ([f"{'1' * 64}  dist/invarlock-1.2.3.tar.gz"], "expected 10"),
    ],
)
def test_build_ledger_rejects_duplicate_or_incomplete_file_sets(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    ledger = tmp_path / "SHA256SUMS"
    digest = _write_ledger(ledger, lines)

    with pytest.raises(verifier.HostedDistributionVerificationError, match=message):
        verifier._parse_build_ledger(ledger, expected_ledger_sha256=digest)


def test_build_ledger_requires_one_wheel_and_sdist_per_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verifier, "PROJECTS", ("demo",))
    lines = [
        f"{'1' * 64}  dist/demo-1.2.3-py3-none-any.whl",
        f"{'2' * 64}  dist/demo-1.2.3-py2-none-any.whl",
    ]
    ledger = tmp_path / "SHA256SUMS"
    digest = _write_ledger(ledger, lines)

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="expected one wheel and one source archive",
    ):
        verifier._parse_build_ledger(ledger, expected_ledger_sha256=digest)


def test_hosted_download_failure_removes_partial_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _expected_files()
    metadata_url = f"{verifier.API_ROOTS['pypi']}/invarlock/1.2.3/json"

    def fake_open_url(url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        if url == metadata_url:
            return io.BytesIO(_metadata(expected))
        raise urllib.error.URLError("download unavailable")

    monkeypatch.setattr(verifier, "_open_url", fake_open_url)
    wheel_staging = tmp_path / "staging"
    wheel_staging.mkdir()

    with pytest.raises(urllib.error.URLError, match="download unavailable"):
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project="invarlock",
            version="1.2.3",
            expected=expected,
            timeout=30,
            wheel_destination=wheel_staging,
        )

    assert list(wheel_staging.iterdir()) == []


def test_hosted_missing_mode_propagates_non_not_found_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError(
        "https://example.invalid/metadata",
        503,
        "unavailable",
        None,
        None,
    )
    monkeypatch.setattr(
        verifier,
        "_open_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(urllib.error.HTTPError) as failure:
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project="invarlock",
            version="1.2.3",
            expected=_expected_files(),
            timeout=30,
            allow_missing=True,
        )
    assert failure.value.code == 503


def test_hosted_download_failure_without_materialization_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected_files()
    metadata_url = f"{verifier.API_ROOTS['pypi']}/invarlock/1.2.3/json"

    def fake_open_url(url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        if url == metadata_url:
            return io.BytesIO(_metadata(expected))
        raise urllib.error.URLError("download unavailable")

    monkeypatch.setattr(verifier, "_open_url", fake_open_url)

    with pytest.raises(urllib.error.URLError, match="download unavailable"):
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project="invarlock",
            version="1.2.3",
            expected=expected,
            timeout=30,
        )


def test_hosted_byte_mismatch_removes_materialized_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _expected_files()
    metadata_url = f"{verifier.API_ROOTS['pypi']}/invarlock/1.2.3/json"

    def fake_open_url(url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        if url == metadata_url:
            return io.BytesIO(_metadata(expected))
        return io.BytesIO(b"substituted bytes")

    monkeypatch.setattr(verifier, "_open_url", fake_open_url)
    wheel_staging = tmp_path / "staging"
    wheel_staging.mkdir()

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="hosted artifact bytes differ",
    ):
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project="invarlock",
            version="1.2.3",
            expected=expected,
            timeout=30,
            wheel_destination=wheel_staging,
        )

    assert list(wheel_staging.iterdir()) == []


def test_wheelhouse_destination_rejects_unavailable_or_non_directory_parent(
    tmp_path: Path,
) -> None:
    missing_parent = tmp_path / "missing" / "wheelhouse"
    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="parent is unavailable",
    ):
        verifier._wheelhouse_destination(missing_parent)

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="parent is not a real directory",
    ):
        verifier._wheelhouse_destination(parent_file / "wheelhouse")


def test_publish_wheelhouse_rejects_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "wheelhouse"
    destination.mkdir()

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="wheel destination already exists",
    ):
        verifier._publish_wheelhouse(source, destination)


def test_publish_wheelhouse_rejects_unexpected_staging_entry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    destination = tmp_path / "wheelhouse"

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="unexpected entry",
    ):
        verifier._publish_wheelhouse(source, destination)

    assert not destination.exists()


def test_publish_wheelhouse_wraps_filesystem_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "demo.whl").write_bytes(b"wheel")
    destination = tmp_path / "wheelhouse"

    def fail_link(*args: Any, **kwargs: Any) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(verifier.os, "link", fail_link)

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="could not publish",
    ):
        verifier._publish_wheelhouse(source, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "unknown"}, "unsupported publish target"),
        ({"version": "../1.2.3"}, "release version is malformed"),
        ({"attempts": 0}, "retry configuration is invalid"),
        ({"retry_delay": -1}, "retry configuration is invalid"),
        ({"timeout": 0}, "retry configuration is invalid"),
    ],
)
def test_front_door_rejects_invalid_configuration(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "ledger_path": tmp_path / "unused",
        "expected_ledger_sha256": "0" * 64,
        "target": "pypi",
        "version": "1.2.3",
    }
    arguments.update(kwargs)

    with pytest.raises(verifier.HostedDistributionVerificationError, match=message):
        verifier.verify_hosted_distributions(**arguments)


def test_cli_translates_verification_error_to_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(**kwargs: Any) -> None:
        raise verifier.HostedDistributionVerificationError("hosted files differ")

    monkeypatch.setattr(verifier, "verify_hosted_distributions", fail)

    with pytest.raises(SystemExit, match="hosted files differ"):
        verifier.main(
            [
                "--ledger",
                str(tmp_path / "SHA256SUMS"),
                "--expected-ledger-sha256",
                "0" * 64,
                "--target",
                "pypi",
                "--version",
                "1.2.3",
            ]
        )
