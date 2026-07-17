from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.release import testpypi_promotion as promotion

SHA = "a" * 40
TAG = "v1.2.3"
DIGEST = "b" * 64
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


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dist_ledger_sha256": DIGEST,
        "format_version": promotion.FORMAT_VERSION,
        "release_sha": SHA,
        "release_tag": TAG,
        "source_run_id": int(RUN_ID),
        "target": "testpypi",
    }
    payload.update(overrides)
    return payload


def _write_manifest(path: Path, **overrides: object) -> Path:
    path.write_text(
        json.dumps(_manifest_payload(**overrides), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _run_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "conclusion": "success",
        "event": "workflow_dispatch",
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
) -> promotion.UrlOpener:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def open_url(request: Request, *, timeout: int) -> FakeResponse:
        if captured is not None:
            captured.update(request=request, timeout=timeout)
        return FakeResponse(raw)

    return open_url


def _authorize(
    manifest: Path,
    *,
    opener: promotion.UrlOpener | None = None,
    token: str = TOKEN,
    repository: str = REPOSITORY,
    api_url: str = API_URL,
) -> promotion.PromotionAuthorization:
    selected_opener = opener if opener is not None else _opener(_run_payload())
    return promotion.authorize_promotion(
        manifest_path=manifest,
        release_sha=SHA,
        release_tag=TAG,
        candidate_run_id=RUN_ID,
        repository=repository,
        api_url=api_url,
        token=token,
        opener=selected_opener,
    )


def test_record_promotion_writes_one_canonical_closed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "promotion.json"

    payload = promotion.record_promotion(
        output=output,
        release_sha=SHA,
        release_tag=TAG,
        dist_ledger_sha256=DIGEST,
        source_run_id=RUN_ID,
    )

    assert payload == _manifest_payload()
    assert output.read_bytes() == (
        json.dumps(
            _manifest_payload(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("release_sha", "A" * 40, "release SHA"),
        ("release_tag", "latest", "release tag"),
        ("dist_ledger_sha256", "b" * 63, "ledger digest"),
        ("source_run_id", "01", "run ID"),
    ],
)
def test_record_promotion_rejects_malformed_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    arguments = {
        "output": tmp_path / "promotion.json",
        "release_sha": SHA,
        "release_tag": TAG,
        "dist_ledger_sha256": DIGEST,
        "source_run_id": RUN_ID,
    }
    arguments[field] = value

    with pytest.raises(promotion.PromotionError, match=message):
        promotion.record_promotion(**arguments)  # type: ignore[arg-type]


def test_record_promotion_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "promotion.json"
    output.write_text("retain me", encoding="utf-8")

    with pytest.raises(promotion.PromotionError, match="created safely"):
        promotion.record_promotion(
            output=output,
            release_sha=SHA,
            release_tag=TAG,
            dist_ledger_sha256=DIGEST,
            source_run_id=RUN_ID,
        )

    assert output.read_text(encoding="utf-8") == "retain me"


@pytest.mark.parametrize(
    "raw",
    [
        b"[]\n",
        b'{"format_version":"one","format_version":"two"}\n',
        b'{"format_version":NaN}\n',
        b"not-json\n",
    ],
)
def test_load_promotion_rejects_non_strict_json(tmp_path: Path, raw: bytes) -> None:
    manifest = tmp_path / "promotion.json"
    manifest.write_bytes(raw)

    with pytest.raises(promotion.PromotionError):
        promotion.load_promotion(manifest)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": "field"}, "contract"),
        ({"format_version": "old"}, "version"),
        ({"release_sha": "c" * 40}, "commit"),
        ({"release_tag": "v1.2.4"}, "tag"),
        ({"source_run_id": 999}, "run identity"),
        ({"target": "pypi"}, "requires a TestPyPI"),
        ({"dist_ledger_sha256": "bad"}, "ledger digest"),
    ],
)
def test_authorize_promotion_rejects_manifest_binding_mismatch(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json", **overrides)

    with pytest.raises(promotion.PromotionError, match=message):
        _authorize(manifest)


def test_authorize_promotion_authenticates_exact_workflow_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")
    captured: dict[str, object] = {}

    result = _authorize(
        manifest,
        opener=_opener(_run_payload(), captured=captured),
    )

    assert result == promotion.PromotionAuthorization(
        artifact_run_id=int(RUN_ID),
        dist_ledger_sha256=DIGEST,
    )
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
        {"event": "push"},
        {"conclusion": "failure"},
        {"head_sha": "c" * 40},
        {"path": ".github/workflows/other.yml"},
    ],
)
def test_authorize_promotion_rejects_wrong_workflow_run(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")

    with pytest.raises(promotion.PromotionError, match="successful exact-commit"):
        _authorize(manifest, opener=_opener(_run_payload(**overrides)))


def test_authorize_promotion_fails_without_token_before_network(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")

    def unexpected_network(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("network must not be called")

    with pytest.raises(promotion.PromotionError, match="authentication is unavailable"):
        _authorize(manifest, opener=unexpected_network, token="")


def test_authorize_promotion_closes_network_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")

    def failed_network(*_args: object, **_kwargs: object) -> FakeResponse:
        raise OSError("offline")

    with pytest.raises(promotion.PromotionError, match="unable to authenticate"):
        _authorize(manifest, opener=failed_network)


@pytest.mark.parametrize(
    ("repository", "api_url"),
    [
        ("one-component", API_URL),
        (REPOSITORY, "http://api.github.com"),
        (REPOSITORY, "https://user@example.test"),
        (REPOSITORY, "https://api.github.com?token=value"),
    ],
)
def test_authorize_promotion_rejects_untrusted_api_identity(
    tmp_path: Path, repository: str, api_url: str
) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")

    with pytest.raises(promotion.PromotionError, match="malformed"):
        _authorize(manifest, repository=repository, api_url=api_url)


def test_authorize_promotion_rejects_malformed_api_json(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "promotion.json")

    def malformed_api(_request: Request, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        return FakeResponse(b'{"id":1,"id":2}\n')

    with pytest.raises(promotion.PromotionError, match="strict JSON"):
        _authorize(manifest, opener=malformed_api)


def test_current_candidate_and_github_output_are_closed(tmp_path: Path) -> None:
    authorization = promotion.current_candidate(
        candidate_run_id=RUN_ID,
        dist_ledger_sha256=DIGEST,
    )
    output = tmp_path / "github-output"

    promotion.write_github_outputs(output, authorization)

    assert output.read_text(encoding="utf-8") == (
        f"artifact_run_id={RUN_ID}\ndist_ledger_sha256={DIGEST}\n"
    )


def test_main_records_manifest_and_reports_invalid_current_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "promotion.json"
    assert (
        promotion.main(
            [
                "record",
                "--output",
                str(manifest),
                "--release-sha",
                SHA,
                "--release-tag",
                TAG,
                "--dist-ledger-sha256",
                DIGEST,
                "--source-run-id",
                RUN_ID,
            ]
        )
        == 0
    )
    assert promotion.load_promotion(manifest) == _manifest_payload()

    github_output = tmp_path / "github-output"
    assert (
        promotion.main(
            [
                "current",
                "--candidate-run-id",
                "bad",
                "--dist-ledger-sha256",
                DIGEST,
                "--github-output",
                str(github_output),
            ]
        )
        == 1
    )
    assert "workflow run ID is malformed" in capsys.readouterr().err
    assert not github_output.exists()
