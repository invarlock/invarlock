from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.release import runtime_release_asset_handoff as handoff
from scripts.release import runtime_release_evidence as evidence

SOURCE_COMMIT = "a" * 40
RELEASE_COMMIT = "9" * 40
SOURCE_ARCHIVE_SHA256 = "b" * 64
RELEASE_TAG = "v0.13.0"
REPOSITORY = "invarlock/invarlock"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build_asset(tmp_path: Path) -> tuple[Path, str]:
    summary = tmp_path / "gguf.json"
    summary.write_bytes(
        _canonical(
            {
                "evidence_sha256": "d" * 64,
                "fixture_revision": "e" * 40,
                "format_version": evidence.GGUF_FORMAT,
                "image_digest": "sha256:" + "c" * 64,
                "runs": 2,
                "status": "ok",
            }
        )
        + b"\n"
    )
    asset = tmp_path / "runtime-release-evidence.tar.gz"
    evidence.build_asset(
        output=asset,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries={"llama_cpp": summary},
        behavioral_receipts=[],
    )
    return asset, hashlib.sha256(asset.read_bytes()).hexdigest()


def _stage(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    source, digest = _build_asset(tmp_path)
    output = tmp_path / "handoff"
    output.mkdir()
    result = handoff.stage_handoff(
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
    return (
        result,
        output / str(result["asset_filename"]),
        output / str(result["digest_filename"]),
    )


def _completed(arguments: list[str], *, stdout: str = "", code: int = 0):
    return subprocess.CompletedProcess(arguments, code, stdout=stdout, stderr="")


def test_stage_and_verify_use_digest_bound_names_without_exposing_paths(
    tmp_path: Path,
) -> None:
    result, asset, digest_file = _stage(tmp_path)

    assert result["status"] == "ok"
    assert result["format_version"] == handoff.HANDOFF_FORMAT
    assert result["asset_filename"] == (
        f"invarlock-{RELEASE_TAG}-runtime-evidence-source-{SOURCE_COMMIT[:12]}-"
        f"{result['asset_sha256']}.tar.gz"
    )
    assert result["digest_filename"] == f"{result['asset_filename']}.sha256"
    assert result["asset_size"] == asset.stat().st_size
    assert str(tmp_path) not in handoff._canonical_json(result)
    assert digest_file.read_text(encoding="ascii") == (
        f"{result['asset_sha256']}  {result['asset_filename']}\n"
    )

    repeated = handoff.verify_handoff(
        asset=asset,
        digest_file=digest_file,
        release_tag=RELEASE_TAG,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_asset_sha256=str(result["asset_sha256"]),
        expected_providers=frozenset({"llama_cpp"}),
        expected_qualifications=frozenset({"llama_cpp"}),
        require_behavioral_claim=False,
    )
    assert repeated == result


def test_stage_never_replaces_an_existing_immutable_handoff(tmp_path: Path) -> None:
    result, asset, digest_file = _stage(tmp_path)
    original_asset = asset.read_bytes()
    original_digest = digest_file.read_bytes()
    source = tmp_path / "runtime-release-evidence.tar.gz"

    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="already exists"):
        handoff.stage_handoff(
            source_asset=source,
            output_dir=asset.parent,
            release_tag=RELEASE_TAG,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )

    assert asset.read_bytes() == original_asset
    assert digest_file.read_bytes() == original_digest


def test_verify_rejects_noncanonical_name_and_digest_sidecar(tmp_path: Path) -> None:
    result, asset, digest_file = _stage(tmp_path)
    renamed = asset.with_name("runtime-evidence.tar.gz")
    renamed.write_bytes(asset.read_bytes())

    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="immutable canonical name"
    ):
        handoff.verify_handoff(
            asset=renamed,
            digest_file=digest_file,
            release_tag=RELEASE_TAG,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )

    digest_file.chmod(0o644)
    digest_file.write_text("0" * 64 + "  wrong.tar.gz\n", encoding="ascii")
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="not canonical"):
        handoff.verify_handoff(
            asset=asset,
            digest_file=digest_file,
            release_tag=RELEASE_TAG,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )


@pytest.mark.parametrize(
    "expected_qualifications",
    [frozenset(), frozenset({"llama_cpp:substituted"})],
)
def test_verify_rejects_missing_or_substituted_qualification_identity(
    tmp_path: Path, expected_qualifications: frozenset[str]
) -> None:
    result, asset, digest_file = _stage(tmp_path)

    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="set does not match"
    ):
        handoff.verify_handoff(
            asset=asset,
            digest_file=digest_file,
            release_tag=RELEASE_TAG,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=expected_qualifications,
            require_behavioral_claim=False,
        )


def test_upload_requires_matching_remote_tag_and_published_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = _stage(tmp_path)
    calls: list[list[str]] = []
    view_count = 0

    def fake_gh(arguments: list[str]):
        nonlocal view_count
        calls.append(arguments)
        if arguments[0] == "api":
            return _completed(
                arguments,
                stdout=json.dumps(
                    {"object": {"type": "commit", "sha": RELEASE_COMMIT}}
                ),
            )
        if arguments[:2] == ["release", "upload"]:
            return _completed(arguments)
        assert arguments[:2] == ["release", "view"]
        view_count += 1
        assets: list[dict[str, object]] = []
        if view_count == 2:
            assets = [
                {
                    "name": result["asset_filename"],
                    "size": asset.stat().st_size,
                    "digest": f"sha256:{result['asset_sha256']}",
                },
                {
                    "name": result["digest_filename"],
                    "size": digest_file.stat().st_size,
                    "digest": "sha256:"
                    + hashlib.sha256(digest_file.read_bytes()).hexdigest(),
                },
            ]
        return _completed(
            arguments,
            stdout=json.dumps(
                {"tagName": RELEASE_TAG, "isDraft": False, "assets": assets}
            ),
        )

    monkeypatch.setattr(handoff, "_run_gh", fake_gh)

    uploaded = handoff.upload_handoff(
        asset=asset,
        digest_file=digest_file,
        repository=REPOSITORY,
        release_tag=RELEASE_TAG,
        expected_release_commit=RELEASE_COMMIT,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_asset_sha256=str(result["asset_sha256"]),
        expected_providers=frozenset({"llama_cpp"}),
        expected_qualifications=frozenset({"llama_cpp"}),
        require_behavioral_claim=False,
    )

    assert uploaded["status"] == "uploaded"
    assert uploaded["source_commit"] == SOURCE_COMMIT
    assert uploaded["release_commit"] == RELEASE_COMMIT
    upload_call = next(call for call in calls if call[:2] == ["release", "upload"])
    assert "--clobber" not in upload_call
    assert upload_call[2] == RELEASE_TAG


def test_upload_rejects_missing_remote_asset_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = _stage(tmp_path)
    view_count = 0

    def fake_gh(arguments: list[str]):
        nonlocal view_count
        if arguments[0] == "api":
            return _completed(
                arguments,
                stdout=json.dumps(
                    {"object": {"type": "commit", "sha": RELEASE_COMMIT}}
                ),
            )
        if arguments[:2] == ["release", "upload"]:
            return _completed(arguments)
        view_count += 1
        assets: list[dict[str, object]] = []
        if view_count == 2:
            assets = [
                {"name": result["asset_filename"], "size": asset.stat().st_size},
                {
                    "name": result["digest_filename"],
                    "size": digest_file.stat().st_size,
                },
            ]
        return _completed(
            arguments,
            stdout=json.dumps(
                {"tagName": RELEASE_TAG, "isDraft": False, "assets": assets}
            ),
        )

    monkeypatch.setattr(handoff, "_run_gh", fake_gh)

    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="asset digest"):
        handoff.upload_handoff(
            asset=asset,
            digest_file=digest_file,
            repository=REPOSITORY,
            release_tag=RELEASE_TAG,
            expected_release_commit=RELEASE_COMMIT,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )


def test_upload_fails_before_mutation_when_remote_tag_binding_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = _stage(tmp_path)
    calls: list[list[str]] = []

    def fake_gh(arguments: list[str]):
        calls.append(arguments)
        return _completed(
            arguments,
            stdout=json.dumps({"object": {"type": "commit", "sha": "f" * 40}}),
        )

    monkeypatch.setattr(handoff, "_run_gh", fake_gh)

    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="release commit"):
        handoff.upload_handoff(
            asset=asset,
            digest_file=digest_file,
            repository=REPOSITORY,
            release_tag=RELEASE_TAG,
            expected_release_commit=RELEASE_COMMIT,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256=str(result["asset_sha256"]),
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )

    assert all(call[:2] != ["release", "upload"] for call in calls)


@pytest.mark.parametrize(
    "release_tag",
    [
        "0.13.0",
        "v0.13",
        "v01.2.3",
        "v0.13.0/asset",
        "v0.13.0 RC1",
        "v0.13.0rc1",
        "v0.13.0-rc..1",
    ],
)
def test_release_tag_cannot_inject_a_path(release_tag: str) -> None:
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="semantic version"
    ):
        handoff._asset_filename(
            release_tag=release_tag,
            source_commit=SOURCE_COMMIT,
            asset_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "repository", ["../..", "owner/.", "/repository", "owner/repository/extra"]
)
def test_repository_cannot_inject_an_api_path(repository: str) -> None:
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="OWNER/NAME"):
        handoff.upload_handoff(
            asset=Path("unused"),
            digest_file=Path("unused.sha256"),
            repository=repository,
            release_tag=RELEASE_TAG,
            expected_release_commit=RELEASE_COMMIT,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256="a" * 64,
            expected_providers=frozenset({"llama_cpp"}),
            expected_qualifications=frozenset({"llama_cpp"}),
            require_behavioral_claim=False,
        )
