from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ref(path: Path, rel_path: str) -> dict[str, str]:
    return {
        "path": rel_path,
        "digest": f"sha256:{_sha256_file(path)}",
    }


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_pack(pack_dir: Path, *, cert_rel_path: str) -> Path:
    final_verdict = pack_dir / "results/final_verdict.json"
    source_repo = pack_dir / "metadata/source_repo.json"
    environment = pack_dir / "metadata/environment.json"
    materials = pack_dir / "metadata/model_revisions.json"
    cert = pack_dir / cert_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text("{}", encoding="utf-8")

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
        cert_rel_path,
    ]
    _write_checksums(pack_dir, covered)

    manifest = {
        "format": "proof-pack-v1",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_file(pack_dir / "checksums.sha256"),
        "subject": {
            "name": "final_verdict",
            **_digest_ref(final_verdict, "results/final_verdict.json"),
        },
        "invocation": {
            "config_source": _digest_ref(source_repo, "metadata/source_repo.json")
        },
        "environment": _digest_ref(environment, "metadata/environment.json"),
        "materials": [
            {
                "name": "model_revisions",
                **_digest_ref(materials, "metadata/model_revisions.json"),
            }
        ],
    }
    _write_json(pack_dir / "manifest.json", manifest)
    return pack_dir


def test_proof_pack_help_lists_verify() -> None:
    result = CliRunner().invoke(app, ["proof-pack", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output


def test_proof_pack_verify_json_round_trip(monkeypatch, tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        cert_rel_path="certs/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"

    monkeypatch.setattr(
        "invarlock.proof_pack._run_verify_command",
        lambda reports, profile: (
            0,
            {
                "format_version": "verify-v1",
                "ok": True,
                "reports": [str(path) for path in reports],
                "resolution": {"exit_code": 0},
            },
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "proof-pack",
            "verify",
            str(pack_dir),
            "--json",
            "--json-out",
            str(json_out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "proof-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True


def test_proof_pack_verify_human_success(monkeypatch, tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        cert_rel_path="certs/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.proof_pack._run_verify_command",
        lambda reports, profile: (
            0,
            {
                "format_version": "verify-v1",
                "ok": True,
                "reports": [str(path) for path in reports],
                "resolution": {"exit_code": 0},
            },
        ),
        raising=False,
    )

    result = CliRunner().invoke(app, ["proof-pack", "verify", str(pack_dir)])

    assert result.exit_code == 0, result.output
    assert "WARNING:" in result.output
    assert "Proof pack verified" in result.output


def test_proof_pack_verify_rejects_missing_pack(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["proof-pack", "verify", str(tmp_path / "missing"), "--json"],
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert payload["resolution"]["exit_code"] == 3


def test_proof_pack_verify_human_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["proof-pack", "verify", str(tmp_path / "missing")]
    )

    assert result.exit_code == 3
    assert "ERROR:" in result.output
    assert "Pack directory not found" in result.output
