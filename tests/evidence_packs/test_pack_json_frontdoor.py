from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from invarlock.evidence_pack_edit_common import _load_json_sidecar
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports
from invarlock.runtime_security import RuntimeManifestExecution, write_runtime_manifest
from scripts.evidence_packs.python import verify_pack_checks


def test_duplicate_key_manifest_cannot_pass_frontdoor_validation(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "format": "evidence-pack-v1",
          "format": "evidence-pack-v1",
          "checksums_sha256": "checksums.sha256",
          "checksums_sha256_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
        """,
        encoding="utf-8",
    )

    assert verify_pack_checks.main(["validate-manifest", str(manifest)]) == 1


def test_duplicate_key_manifest_cannot_pass_provenance_verification(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text(
        """
        {
          "format": "evidence-pack-v1",
          "format": "evidence-pack-v1"
        }
        """,
        encoding="utf-8",
    )

    assert verify_pack_checks.main(["manifest-provenance", str(pack_dir)]) == 1


def test_duplicate_key_scenarios_cannot_drive_classification(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        """
        {
          "schema": "evidence_pack_scenarios_v1",
          "schema_version": 1,
          "scenarios": [{"id": "clean", "strictness": "must_pass"}],
          "scenarios": [{"id": "clean", "strictness": "must_fail"}]
        }
        """,
        encoding="utf-8",
    )
    assert verify_pack_checks.main(["scenarios-manifest", str(scenarios)]) == 1
    assert (
        verify_pack_checks.main(["scenario-strictness", str(scenarios), "clean"]) == 1
    )


def test_edit_metadata_sidecar_frontdoor_rejects_duplicate_keys(tmp_path: Path) -> None:
    sidecar = tmp_path / "edit_metadata.json"
    sidecar.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")

    payload, error = _load_json_sidecar(sidecar)

    assert payload is None
    assert error is not None
    assert "duplicate key" in error


@pytest.mark.parametrize(
    "payload",
    (
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e9999}',
        b'{"value":',
        b'{"value":"\xff"}',
        b"[]",
    ),
)
def test_verification_frontdoor_rejects_nonstandard_or_nonobject_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(payload)

    assert verify_pack_checks.main(["json-object", str(path)]) == 1


def test_verification_frontdoor_explains_invalid_json_object_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "model_revisions.json"
    path.write_text("[]", encoding="utf-8")

    assert (
        verify_pack_checks.main(
            ["json-object", str(path), "--label", "model_revisions.json"]
        )
        == 1
    )

    assert "model_revisions.json" in capsys.readouterr().err


def test_report_mode_accepts_a_bound_report_only_fixture(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(
        """
        {
          "schema_version": "v1",
          "run_id": "report-only-fixture",
          "artifacts": {},
          "plugins": {},
          "meta": {},
          "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
              "preview": 1,
              "final": 1,
              "stats": {
                "window_match_fraction": 1.0,
                "window_overlap_fraction": 0.0,
                "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                "paired_windows": 1
              }
            }
          },
          "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 2.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0]
          },
          "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 5.0}},
          "evaluation_windows": {
            "final": {"logloss": [2.302585092994046], "token_counts": [1]}
          }
        }
        """,
        encoding="utf-8",
    )
    write_runtime_manifest(
        report_path,
        execution=RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="invarlock-runtime:fixture",
            image_digest="sha256:" + "a" * 64,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )

    result = run_verify_reports(
        [report_path],
        profile="dev",
        assurance_mode="report",
        json_mode=True,
    )

    assert result.outcome is VerifyOutcome.OK
    assert result.payload["summary"] == {"ok": True, "reason": "ok"}
    runtime = result.payload["results"][0]["verification"]["runtime_provenance"]
    assert runtime["binding_verified"] is True


def test_verification_frontdoor_rejects_symlinked_control_json(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"ok": true}', encoding="utf-8")
    link = tmp_path / "manifest.json"
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform-specific filesystem policy
        pytest.skip(f"symlinks unavailable in test filesystem: {exc}")

    assert verify_pack_checks.main(["json-object", str(link)]) == 1


def test_signature_check_preserves_unsigned_pack_behavior(tmp_path: Path) -> None:
    assert verify_pack_checks.main(["signature", str(tmp_path)]) == 0


def test_successful_verifier_without_strict_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        verify_pack_checks.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["invarlock", "verify"],
            returncode=0,
            stdout=b'{"ok":NaN}',
        ),
    )

    returncode, payload = verify_pack_checks._verify_command(
        [tmp_path / "evaluation.report.json"],
        profile="dev",
        report_assurance="report",
    )

    assert returncode == 1
    assert payload is None
