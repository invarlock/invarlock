from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.release import runtime_release_asset_handoff as handoff
from tests.scripts._runtime_release_evidence_test_support import (
    RELEASE_COMMIT,
    RELEASE_TAG,
    REPOSITORY,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_COMMIT,
    completed,
    stage_legacy_asset,
)


def _upload_arguments(
    result: dict[str, object], asset: Path, digest_file: Path
) -> dict[str, object]:
    return {
        "asset": asset,
        "digest_file": digest_file,
        "repository": REPOSITORY,
        "release_tag": RELEASE_TAG,
        "expected_release_commit": RELEASE_COMMIT,
        "expected_source_commit": SOURCE_COMMIT,
        "expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "expected_asset_sha256": str(result["asset_sha256"]),
        "expected_providers": frozenset({"llama_cpp"}),
        "expected_qualifications": frozenset({"llama_cpp"}),
        "require_behavioral_claim": False,
    }


def _published_assets(
    result: dict[str, object], asset: Path, digest_file: Path
) -> list[dict[str, object]]:
    return [
        {
            "name": result["asset_filename"],
            "size": asset.stat().st_size,
            "digest": f"sha256:{result['asset_sha256']}",
        },
        {
            "name": result["digest_filename"],
            "size": digest_file.stat().st_size,
            "digest": "sha256:" + hashlib.sha256(digest_file.read_bytes()).hexdigest(),
        },
    ]


@pytest.mark.parametrize(
    "failure", [OSError("missing"), subprocess.TimeoutExpired("gh", 1)]
)
def test_run_gh_closes_process_launch_and_timeout_failures(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise failure

    monkeypatch.setattr(handoff.subprocess, "run", fail)
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="could not be executed"
    ):
        handoff._run_gh(["release", "view"])


@pytest.mark.parametrize(
    "completed_result",
    [
        completed([], code=1),
        completed([], stdout="not-json"),
        completed([], stdout="[]"),
    ],
)
def test_parse_gh_object_rejects_failed_or_open_metadata(
    completed_result: subprocess.CompletedProcess[str],
) -> None:
    with pytest.raises(
        handoff.RuntimeReleaseAssetHandoffError, match="verified|metadata"
    ):
        handoff._parse_gh_object(completed_result, label="remote")


def test_remote_tag_resolution_peels_annotated_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {"object": {"type": "tag", "sha": "1" * 40}},
            {"object": {"type": "tag", "sha": "2" * 40}},
            {"object": {"type": "commit", "sha": RELEASE_COMMIT}},
        ]
    )
    monkeypatch.setattr(
        handoff,
        "_run_gh",
        lambda arguments: completed(arguments, stdout=json.dumps(next(responses))),
    )
    assert (
        handoff._resolve_remote_tag_commit(
            repository=REPOSITORY, release_tag=RELEASE_TAG
        )
        == RELEASE_COMMIT
    )


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"object": "not-an-object"},
        {"object": {"type": "commit", "sha": "bad"}},
        {"object": {"type": "blob", "sha": "1" * 40}},
    ],
)
def test_remote_tag_resolution_rejects_malformed_targets(
    monkeypatch: pytest.MonkeyPatch, record: dict[str, object]
) -> None:
    monkeypatch.setattr(
        handoff,
        "_run_gh",
        lambda arguments: completed(arguments, stdout=json.dumps(record)),
    )
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="one commit"):
        handoff._resolve_remote_tag_commit(
            repository=REPOSITORY, release_tag=RELEASE_TAG
        )


def test_remote_tag_resolution_rejects_excessive_annotation_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        handoff,
        "_run_gh",
        lambda arguments: completed(
            arguments,
            stdout=json.dumps({"object": {"type": "tag", "sha": "1" * 40}}),
        ),
    )
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="one commit"):
        handoff._resolve_remote_tag_commit(
            repository=REPOSITORY, release_tag=RELEASE_TAG
        )


@pytest.mark.parametrize("assets", [None, {}, ["not-an-object"]])
def test_release_assets_requires_a_closed_object_list(assets: object) -> None:
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="metadata"):
        handoff._release_assets({"assets": assets})


def test_release_record_uses_narrow_json_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return completed(
            arguments,
            stdout=json.dumps({"tagName": RELEASE_TAG, "isDraft": False, "assets": []}),
        )

    monkeypatch.setattr(handoff, "_run_gh", fake)
    record = handoff._release_record(repository=REPOSITORY, release_tag=RELEASE_TAG)
    assert record["tagName"] == RELEASE_TAG
    assert calls == [
        [
            "release",
            "view",
            RELEASE_TAG,
            "--repo",
            REPOSITORY,
            "--json",
            "tagName,isDraft,assets",
        ]
    ]


def test_upload_rejects_invalid_expected_release_commit(
    tmp_path: Path,
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="full lowercase"):
        handoff.upload_handoff(
            **{
                **_upload_arguments(result, asset, digest_file),
                "expected_release_commit": RELEASE_COMMIT[:12],
            }
        )


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"tagName": "v9.9.9", "isDraft": False, "assets": []},
        {"tagName": RELEASE_TAG, "isDraft": True, "assets": []},
        {"tagName": RELEASE_TAG, "isDraft": None, "assets": []},
    ],
)
def test_upload_requires_matching_published_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: object
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    monkeypatch.setattr(
        handoff, "_resolve_remote_tag_commit", lambda **_kwargs: RELEASE_COMMIT
    )
    monkeypatch.setattr(handoff, "_release_record", lambda **_kwargs: record)
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="published"):
        handoff.upload_handoff(**_upload_arguments(result, asset, digest_file))


def test_upload_rejects_existing_asset_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    monkeypatch.setattr(
        handoff, "_resolve_remote_tag_commit", lambda **_kwargs: RELEASE_COMMIT
    )
    monkeypatch.setattr(
        handoff,
        "_release_record",
        lambda **_kwargs: {
            "tagName": RELEASE_TAG,
            "isDraft": False,
            "assets": [{"name": result["asset_filename"]}],
        },
    )
    monkeypatch.setattr(
        handoff,
        "_run_gh",
        lambda _arguments: pytest.fail("upload command must not be invoked"),
    )
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="already exist"):
        handoff.upload_handoff(**_upload_arguments(result, asset, digest_file))


def test_upload_command_failure_never_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    monkeypatch.setattr(
        handoff, "_resolve_remote_tag_commit", lambda **_kwargs: RELEASE_COMMIT
    )
    monkeypatch.setattr(
        handoff,
        "_release_record",
        lambda **_kwargs: {"tagName": RELEASE_TAG, "isDraft": False, "assets": []},
    )
    monkeypatch.setattr(
        handoff, "_run_gh", lambda arguments: completed(arguments, code=1)
    )
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="upload failed"):
        handoff.upload_handoff(**_upload_arguments(result, asset, digest_file))


@pytest.mark.parametrize(
    "defect", ["asset-missing", "digest-missing", "asset-size", "digest-size"]
)
def test_upload_rejects_incomplete_or_size_mismatched_remote_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    assets = _published_assets(result, asset, digest_file)
    if defect == "asset-missing":
        assets.pop(0)
    elif defect == "digest-missing":
        assets.pop(1)
    elif defect == "asset-size":
        assets[0]["size"] = 0
    else:
        assets[1]["size"] = 0
    assets.append({"name": 1, "size": 0})
    records = iter(
        [
            {"tagName": RELEASE_TAG, "isDraft": False, "assets": []},
            {"tagName": RELEASE_TAG, "isDraft": False, "assets": assets},
        ]
    )
    monkeypatch.setattr(
        handoff, "_resolve_remote_tag_commit", lambda **_kwargs: RELEASE_COMMIT
    )
    monkeypatch.setattr(handoff, "_release_record", lambda **_kwargs: next(records))
    monkeypatch.setattr(handoff, "_run_gh", lambda arguments: completed(arguments))
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match="metadata"):
        handoff.upload_handoff(**_upload_arguments(result, asset, digest_file))


@pytest.mark.parametrize("defect", ["asset-digest", "sidecar-digest"])
def test_upload_rejects_remote_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    result, asset, digest_file = stage_legacy_asset(tmp_path)
    assets = _published_assets(result, asset, digest_file)
    assets[0 if defect == "asset-digest" else 1]["digest"] = "sha256:" + "f" * 64
    records = iter(
        [
            {"tagName": RELEASE_TAG, "isDraft": False, "assets": []},
            {"tagName": RELEASE_TAG, "isDraft": False, "assets": assets},
        ]
    )
    monkeypatch.setattr(
        handoff, "_resolve_remote_tag_commit", lambda **_kwargs: RELEASE_COMMIT
    )
    monkeypatch.setattr(handoff, "_release_record", lambda **_kwargs: next(records))
    monkeypatch.setattr(handoff, "_run_gh", lambda arguments: completed(arguments))
    message = "asset digest" if defect == "asset-digest" else "digest file"
    with pytest.raises(handoff.RuntimeReleaseAssetHandoffError, match=message):
        handoff.upload_handoff(**_upload_arguments(result, asset, digest_file))
