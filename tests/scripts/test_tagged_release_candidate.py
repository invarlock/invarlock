from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.release import tagged_release_candidate as candidate

SHA = "a" * 40
TAG = "v1.2.3"
RUN_ID = "12345"
REPOSITORY = "invarlock/invarlock"
API_URL = "https://api.github.com"
TOKEN = "test-token"


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()


def _run_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "conclusion": "success",
        "event": "push",
        "head_branch": TAG,
        "head_sha": SHA,
        "id": int(RUN_ID),
        "path": ".github/workflows/release.yml@refs/tags/v1.2.3",
    }
    payload.update(overrides)
    return payload


def _opener(
    payload: object,
    *,
    captured: dict[str, object] | None = None,
) -> candidate.UrlOpener:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def open_url(request: Request, *, timeout: int) -> FakeResponse:
        if captured is not None:
            captured.update(request=request, timeout=timeout)
        return FakeResponse(raw)

    return open_url


def _dist_tree(root: Path, *, tag: str = TAG) -> str:
    version = tag.removeprefix("v")
    paths = candidate.expected_distribution_paths(version)
    lines: list[str] = []
    for index, relative in enumerate(sorted(paths)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"archive-{index}\n".encode())
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    ledger = root / "SHA256SUMS"
    ledger.write_text("".join(lines), encoding="utf-8")
    return hashlib.sha256(ledger.read_bytes()).hexdigest()


def test_authenticate_tagged_run_accepts_exact_successful_tag_run() -> None:
    captured: dict[str, object] = {}

    run_id = candidate.authenticate_tagged_run(
        release_sha=SHA,
        release_tag=TAG,
        candidate_run_id=RUN_ID,
        repository=REPOSITORY,
        api_url=API_URL,
        token=TOKEN,
        opener=_opener(_run_payload(), captured=captured),
    )

    assert run_id == int(RUN_ID)
    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == (
        "https://api.github.com/repos/invarlock/invarlock/actions/runs/12345"
    )
    assert request.get_header("Authorization") == "Bearer test-token"
    assert captured["timeout"] == 30


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": 999},
        {"event": "workflow_dispatch"},
        {"conclusion": "failure"},
        {"head_sha": "c" * 40},
        {"head_branch": "main"},
        {"path": ".github/workflows/other.yml"},
    ],
)
def test_authenticate_tagged_run_rejects_wrong_workflow_run(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(candidate.CandidateError, match="successful exact-tag"):
        candidate.authenticate_tagged_run(
            release_sha=SHA,
            release_tag=TAG,
            candidate_run_id=RUN_ID,
            repository=REPOSITORY,
            api_url=API_URL,
            token=TOKEN,
            opener=_opener(_run_payload(**overrides)),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("release_sha", "A" * 40, "release SHA"),
        ("release_tag", "latest", "release tag"),
        ("candidate_run_id", "01", "run ID"),
        ("repository", "one-component", "repository"),
        ("api_url", "http://api.github.com", "API URL"),
        ("api_url", "https://user@example.test", "API URL"),
        ("api_url", "https://api.github.com?token=value", "API URL"),
    ],
)
def test_authenticate_tagged_run_rejects_malformed_identity(
    field: str, value: str, message: str
) -> None:
    arguments = {
        "release_sha": SHA,
        "release_tag": TAG,
        "candidate_run_id": RUN_ID,
        "repository": REPOSITORY,
        "api_url": API_URL,
        "token": TOKEN,
        "opener": _opener(_run_payload()),
    }
    arguments[field] = value

    with pytest.raises(candidate.CandidateError, match=message):
        candidate.authenticate_tagged_run(**arguments)  # type: ignore[arg-type]


def test_authenticate_tagged_run_requires_token_before_network() -> None:
    def unexpected_network(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("network must not be called")

    with pytest.raises(candidate.CandidateError, match="authentication is unavailable"):
        candidate.authenticate_tagged_run(
            release_sha=SHA,
            release_tag=TAG,
            candidate_run_id=RUN_ID,
            repository=REPOSITORY,
            api_url=API_URL,
            token="",
            opener=unexpected_network,
        )


def test_authenticate_tagged_run_closes_network_and_json_failures() -> None:
    def failed_network(*_args: object, **_kwargs: object) -> FakeResponse:
        raise OSError("offline")

    common = {
        "release_sha": SHA,
        "release_tag": TAG,
        "candidate_run_id": RUN_ID,
        "repository": REPOSITORY,
        "api_url": API_URL,
        "token": TOKEN,
    }
    with pytest.raises(candidate.CandidateError, match="unable to authenticate"):
        candidate.authenticate_tagged_run(**common, opener=failed_network)
    with pytest.raises(candidate.CandidateError, match="strict JSON"):
        candidate.authenticate_tagged_run(
            **common,
            opener=lambda *_args, **_kwargs: FakeResponse(b'{"id":1,"id":2}\n'),
        )
    with pytest.raises(candidate.CandidateError, match="metadata is too large"):
        candidate.authenticate_tagged_run(
            **common,
            opener=lambda *_args, **_kwargs: FakeResponse(
                b"x" * (candidate.MAX_JSON_BYTES + 1)
            ),
        )


def test_verify_distribution_ledger_accepts_exact_ten_file_candidate(
    tmp_path: Path,
) -> None:
    expected = _dist_tree(tmp_path)

    assert candidate.verify_distribution_ledger(tmp_path, TAG) == expected


@pytest.mark.parametrize(
    ("ledger_text", "message"),
    [
        ("not-a-ledger\n", "malformed"),
        (f"{'a' * 64}  ../escape.whl\n", "unsafe"),
        (f"{'a' * 64}  duplicate.whl\n{'b' * 64}  duplicate.whl\n", "duplicate"),
    ],
)
def test_verify_distribution_ledger_rejects_invalid_contract(
    tmp_path: Path, ledger_text: str, message: str
) -> None:
    (tmp_path / "SHA256SUMS").write_text(ledger_text, encoding="utf-8")

    with pytest.raises(candidate.CandidateError, match=message):
        candidate.verify_distribution_ledger(tmp_path, TAG)


def test_verify_distribution_ledger_rejects_missing_unexpected_and_changed_files(
    tmp_path: Path,
) -> None:
    _dist_tree(tmp_path)
    ledger = tmp_path / "SHA256SUMS"
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    ledger.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(candidate.CandidateError, match="file set"):
        candidate.verify_distribution_ledger(tmp_path, TAG)

    _dist_tree(tmp_path)
    first_relative = sorted(candidate.expected_distribution_paths("1.2.3"))[0]
    (tmp_path / first_relative).write_bytes(b"changed")
    with pytest.raises(candidate.CandidateError, match="digest mismatch"):
        candidate.verify_distribution_ledger(tmp_path, TAG)


def test_verify_distribution_ledger_rejects_missing_nonregular_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(candidate.CandidateError, match="unavailable"):
        candidate.verify_distribution_ledger(tmp_path, TAG)

    ledger = tmp_path / "SHA256SUMS"
    ledger.mkdir()
    with pytest.raises(candidate.CandidateError, match="regular file"):
        candidate.verify_distribution_ledger(tmp_path, TAG)
    ledger.rmdir()

    _dist_tree(tmp_path)
    first_relative = sorted(candidate.expected_distribution_paths("1.2.3"))[0]
    archive = tmp_path / first_relative
    target = tmp_path.parent / f"{tmp_path.name}-target"
    target.write_bytes(archive.read_bytes())
    archive.unlink()
    archive.symlink_to(target)
    with pytest.raises(candidate.CandidateError, match="regular file"):
        candidate.verify_distribution_ledger(tmp_path, TAG)


def test_github_output_helpers_emit_only_validated_scalars(tmp_path: Path) -> None:
    output = tmp_path.parent / f"{tmp_path.name}-github-output"

    candidate.write_run_output(output, int(RUN_ID))
    candidate.write_ledger_output(output, "b" * 64)

    assert output.read_text(encoding="utf-8") == (
        f"artifact_run_id={RUN_ID}\ndist_ledger_sha256={'b' * 64}\n"
    )


def test_main_authenticates_then_verifies_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = _dist_tree(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-github-output"
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(candidate.urllib.request, "urlopen", _opener(_run_payload()))

    assert (
        candidate.main(
            [
                "authenticate",
                "--release-sha",
                SHA,
                "--release-tag",
                TAG,
                "--candidate-run-id",
                RUN_ID,
                "--repository",
                REPOSITORY,
                "--api-url",
                API_URL,
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        candidate.main(
            [
                "verify-ledger",
                "--dist-dir",
                str(tmp_path),
                "--release-tag",
                TAG,
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8") == (
        f"artifact_run_id={RUN_ID}\ndist_ledger_sha256={digest}\n"
    )

    assert (
        candidate.main(
            [
                "authenticate",
                "--release-sha",
                SHA,
                "--release-tag",
                TAG,
                "--candidate-run-id",
                "bad",
                "--repository",
                REPOSITORY,
                "--api-url",
                API_URL,
                "--github-output",
                str(output),
            ]
        )
        == 1
    )
    assert "workflow run ID is malformed" in capsys.readouterr().err
