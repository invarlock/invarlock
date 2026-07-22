from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts.release import verify_hosted_distributions as verifier


def _release_fixture(
    tmp_path: Path, *, target: str
) -> tuple[Path, str, dict[str, bytes], dict[str, bytes]]:
    version = "1.2.3"
    metadata: dict[str, bytes] = {}
    downloads: dict[str, bytes] = {}
    ledger_lines: list[str] = []
    for project in verifier.PROJECTS:
        prefix = re.sub(r"[-_.]+", "_", project).lower()
        filenames = (
            f"{prefix}-{version}-py3-none-any.whl",
            f"{prefix}-{version}.tar.gz",
        )
        entries: list[dict[str, Any]] = []
        for filename in filenames:
            payload = f"published bytes for {filename}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            download_url = f"https://files.example.invalid/{filename}"
            downloads[download_url] = payload
            ledger_lines.append(f"{digest}  dist/{filename}")
            entries.append(
                {
                    "filename": filename,
                    "digests": {"sha256": digest},
                    "url": download_url,
                }
            )
        metadata_url = f"{verifier.API_ROOTS[target]}/{project}/{version}/json"
        metadata[metadata_url] = json.dumps({"urls": entries}).encode()
    ledger_bytes = ("\n".join(sorted(ledger_lines)) + "\n").encode()
    ledger = tmp_path / "SHA256SUMS"
    ledger.write_bytes(ledger_bytes)
    return ledger, hashlib.sha256(ledger_bytes).hexdigest(), metadata, downloads


def _install_fake_network(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: dict[str, bytes],
    downloads: dict[str, bytes],
) -> None:
    def fake_open_url(url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        if url in metadata:
            return io.BytesIO(metadata[url])
        if url in downloads:
            return io.BytesIO(downloads[url])
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(verifier, "_open_url", fake_open_url)


@pytest.mark.parametrize("target", ["pypi", "testpypi"])
def test_verify_hosted_distributions_accepts_exact_build_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )

    verifier.verify_hosted_distributions(
        ledger_path=ledger,
        expected_ledger_sha256=ledger_digest,
        target=target,
        version="v1.2.3",
        attempts=1,
    )


def test_verify_hosted_distributions_accepts_authenticated_project_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = "pypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    selected = verifier.PROJECTS[:-1]
    tensorrt_metadata = (
        f"{verifier.API_ROOTS[target]}/{verifier.PROJECTS[-1]}/1.2.3/json"
    )
    metadata.pop(tensorrt_metadata)
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )

    verifier.verify_hosted_distributions(
        ledger_path=ledger,
        expected_ledger_sha256=ledger_digest,
        target=target,
        version="v1.2.3",
        attempts=1,
        projects=selected,
    )


@pytest.mark.parametrize("projects", [(), ("unknown",), ("invarlock", "invarlock")])
def test_verify_hosted_distributions_rejects_invalid_project_subset(
    tmp_path: Path, projects: tuple[str, ...]
) -> None:
    ledger, ledger_digest, _metadata, _downloads = _release_fixture(
        tmp_path, target="pypi"
    )
    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="project selection is invalid",
    ):
        verifier.verify_hosted_distributions(
            ledger_path=ledger,
            expected_ledger_sha256=ledger_digest,
            target="pypi",
            version="v1.2.3",
            attempts=1,
            projects=projects,
        )


def test_verify_hosted_distributions_materializes_only_ledger_selected_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = "testpypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )
    wheelhouse = tmp_path / "wheelhouse"

    verifier.verify_hosted_distributions(
        ledger_path=ledger,
        expected_ledger_sha256=ledger_digest,
        target=target,
        version="v1.2.3",
        attempts=1,
        wheelhouse=wheelhouse,
    )

    materialized = sorted(wheelhouse.iterdir())
    assert len(materialized) == len(verifier.PROJECTS)
    assert all(path.suffix == ".whl" for path in materialized)
    for path in materialized:
        expected = next(
            payload
            for url, payload in downloads.items()
            if url.endswith("/" + path.name)
        )
        assert path.read_bytes() == expected


def test_verify_hosted_distributions_refuses_existing_wheelhouse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = "testpypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="wheel destination already exists",
    ):
        verifier.verify_hosted_distributions(
            ledger_path=ledger,
            expected_ledger_sha256=ledger_digest,
            target=target,
            version="v1.2.3",
            attempts=1,
            wheelhouse=wheelhouse,
        )


def test_materialization_failure_leaves_no_partial_wheelhouse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = "testpypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )
    monkeypatch.setattr(verifier, "_MAX_DISTRIBUTION_BYTES", 4)
    wheelhouse = tmp_path / "wheelhouse"

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="hosted artifact is too large",
    ):
        verifier.verify_hosted_distributions(
            ledger_path=ledger,
            expected_ledger_sha256=ledger_digest,
            target=target,
            version="v1.2.3",
            attempts=1,
            wheelhouse=wheelhouse,
        )

    assert not wheelhouse.exists()
    assert list(tmp_path.glob(".invarlock-hosted-wheels-*")) == []


def test_hosted_project_rejects_oversized_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "invarlock"
    version = "1.2.3"
    metadata_url = f"{verifier.API_ROOTS['pypi']}/{project}/{version}/json"
    monkeypatch.setattr(verifier, "_MAX_METADATA_BYTES", 8)
    _install_fake_network(
        monkeypatch,
        metadata={metadata_url: b"{" + (b" " * 8) + b"}"},
        downloads={},
    )

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="metadata is too large",
    ):
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project=project,
            version=version,
            expected={
                "invarlock-1.2.3-py3-none-any.whl": "1" * 64,
                "invarlock-1.2.3.tar.gz": "2" * 64,
            },
            timeout=30,
        )


@pytest.mark.parametrize("mutation", ["filename", "declared-digest", "bytes"])
def test_verify_hosted_distributions_rejects_stale_or_substituted_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    target = "testpypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    metadata_url = f"{verifier.API_ROOTS[target]}/invarlock/1.2.3/json"
    payload = json.loads(metadata[metadata_url])
    first_entry = payload["urls"][0]
    if mutation == "filename":
        first_entry["filename"] = f"stale-{first_entry['filename']}"
    elif mutation == "declared-digest":
        first_entry["digests"]["sha256"] = "0" * 64
    else:
        downloads[first_entry["url"]] = b"substituted hosted bytes"
    metadata[metadata_url] = json.dumps(payload).encode()
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )
    sleeps: list[float] = []

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="failed after 3 attempts",
    ):
        verifier.verify_hosted_distributions(
            ledger_path=ledger,
            expected_ledger_sha256=ledger_digest,
            target=target,
            version="1.2.3",
            attempts=3,
            retry_delay=0,
            sleep=sleeps.append,
        )

    assert sleeps == [0, 0]


def test_verify_hosted_distributions_rejects_wrong_build_ledger_digest(
    tmp_path: Path,
) -> None:
    ledger, _ledger_digest, _metadata, _downloads = _release_fixture(
        tmp_path, target="pypi"
    )

    with pytest.raises(
        verifier.HostedDistributionVerificationError,
        match="ledger changed after build",
    ):
        verifier.verify_hosted_distributions(
            ledger_path=ledger,
            expected_ledger_sha256="0" * 64,
            target="pypi",
            version="1.2.3",
            attempts=1,
        )


def test_cli_verifies_hosted_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = "testpypi"
    ledger, ledger_digest, metadata, downloads = _release_fixture(
        tmp_path, target=target
    )
    _install_fake_network(
        monkeypatch,
        metadata=metadata,
        downloads=downloads,
    )

    result = verifier.main(
        [
            "--ledger",
            str(ledger),
            "--expected-ledger-sha256",
            ledger_digest,
            "--target",
            target,
            "--version",
            "v1.2.3",
            "--attempts",
            "1",
            "--project",
            "invarlock",
        ]
    )

    assert result == 0
    assert "testpypi release 1.2.3 matches build ledger" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("malformed-digest", "ledger digest is malformed"),
        ("missing-ledger", "cannot read distribution digest ledger"),
        ("non-utf8", "ledger is not UTF-8"),
        ("traversal", "ledger is malformed"),
    ],
)
def test_build_ledger_rejects_malformed_or_unreadable_inputs(
    tmp_path: Path, case: str, message: str
) -> None:
    ledger = tmp_path / "SHA256SUMS"
    ledger_bytes = b"\xff" if case == "non-utf8" else b"0" * 64 + b"  ../bad.whl\n"
    if case != "missing-ledger":
        ledger.write_bytes(ledger_bytes)
    expected_digest = hashlib.sha256(ledger_bytes).hexdigest()
    if case == "malformed-digest":
        expected_digest = "not-a-digest"

    with pytest.raises(verifier.HostedDistributionVerificationError, match=message):
        verifier._parse_build_ledger(
            ledger,
            expected_ledger_sha256=expected_digest,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("non-object", "metadata is malformed"),
        ("no-url-list", "metadata has no URL list"),
        ("non-object-entry", "metadata has malformed files"),
        ("duplicate-file", "malformed or duplicate files"),
        ("insecure-url", "download URL is invalid"),
    ],
)
def test_hosted_project_rejects_malformed_metadata_and_insecure_urls(
    monkeypatch: pytest.MonkeyPatch, case: str, message: str
) -> None:
    project = "invarlock"
    version = "1.2.3"
    expected = {
        "invarlock-1.2.3-py3-none-any.whl": "1" * 64,
        "invarlock-1.2.3.tar.gz": "2" * 64,
    }
    entries = [
        {
            "filename": filename,
            "digests": {"sha256": digest},
            "url": "http://files.example.invalid/" + filename,
        }
        for filename, digest in expected.items()
    ]
    payload: object
    if case == "non-object":
        payload = []
    elif case == "no-url-list":
        payload = {"urls": {}}
    elif case == "non-object-entry":
        payload = {"urls": [None]}
    elif case == "duplicate-file":
        payload = {"urls": [entries[0], entries[0]]}
    else:
        payload = {"urls": entries}

    metadata_url = f"{verifier.API_ROOTS['pypi']}/{project}/{version}/json"
    _install_fake_network(
        monkeypatch,
        metadata={metadata_url: json.dumps(payload).encode()},
        downloads={},
    )

    with pytest.raises(verifier.HostedDistributionVerificationError, match=message):
        verifier._verify_project(
            api_root=verifier.API_ROOTS["pypi"],
            project=project,
            version=version,
            expected=expected,
            timeout=30,
        )
