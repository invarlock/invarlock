from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from scripts.smoke.guard_validation_smoke import (
    _render_markdown,
    build_guard_validation_smoke,
)
from tests.scripts._support_release_evidence_check import (
    release_checker_module as _release_checker_module,
)
from tests.scripts._support_release_evidence_check import repo_root as _repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_guard_smoke(guard_dir: Path) -> dict[str, object]:
    guard_dir.mkdir(exist_ok=True)
    payload = build_guard_validation_smoke(replicates=5, seed=7)
    (guard_dir / "guard-validation-smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (guard_dir / "guard-validation-smoke.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _write_offline_bundle(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    bundle_root = "invarlock-0.9.0-offline-bundle"
    manifest = {
        "schema": "invarlock/release-offline-bundle-v1",
        "distributions": [
            {"path": "dist/invarlock-0.9.0-py3-none-any.whl"},
            {"path": "dist/invarlock-0.9.0.tar.gz"},
        ],
    }
    tarball = output_dir / "invarlock-0.9.0-offline-bundle.tar.gz"
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(manifest_path, arcname=f"{bundle_root}/release_manifest.json")
    manifest_path.unlink()


def _write_valid_release_evidence(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    dist = tmp_path / "dist"
    release = tmp_path / "release"
    sbom = tmp_path / "sbom.json"
    guard_dir = tmp_path / "guard-validation"
    offline_dir = tmp_path / "offline"
    dist.mkdir()
    wheel = dist / "invarlock-0.9.0-py3-none-any.whl"
    sdist = dist / "invarlock-0.9.0.tar.gz"
    wheel.write_text("wheel", encoding="utf-8")
    sdist.write_text("sdist", encoding="utf-8")
    (release / "strict").mkdir(parents=True)
    (release / "wheel-sdist-hashes.txt").write_text(
        f"{_sha256(wheel)}  {wheel.name}\n{_sha256(sdist)}  {sdist.name}\n",
        encoding="utf-8",
    )
    (release / "runtime-image-digest.txt").write_text(
        "sha256:" + "1" * 64 + "\n",
        encoding="utf-8",
    )
    (release / "strict" / "evaluation.report.json").write_text(
        json.dumps(
            {
                "assurance": {
                    "mode": "strict",
                    "verdict": "pending_verifier",
                    "fallback_fields_used": False,
                },
                "report_build": {
                    "synthesized_fields": [],
                    "repaired_fields": [],
                    "fallback_fields": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (release / "strict" / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": True},
                "results": [
                    {
                        "id": str(release / "strict" / "evaluation.report.json"),
                        "verification": {
                            "runtime_provenance": {
                                "status": "expected_image_digest_matched",
                                "verified": True,
                                "binding_verified": True,
                                "expected_digest_matched": True,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sbom.write_text('{"bomFormat":"CycloneDX"}', encoding="utf-8")
    _write_guard_smoke(guard_dir)
    _write_offline_bundle(offline_dir)
    return dist, release, sbom, guard_dir, offline_dir


def _release_check_command(
    repo_root: Path,
    *,
    release: Path,
    dist: Path,
    sbom: Path,
    guard_dir: Path,
    offline_dir: Path,
    json_output: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "release" / "evidence_contracts.py"),
        "release",
        "--root",
        str(release),
        "--dist",
        str(dist),
        "--sbom",
        str(sbom),
        "--guard-validation-json",
        str(guard_dir / "guard-validation-smoke.json"),
        "--guard-validation-md",
        str(guard_dir / "guard-validation-smoke.md"),
        "--offline-bundle-dir",
        str(offline_dir),
    ]
    if json_output:
        command.append("--json")
    return command


def test_release_artifact_shape_check_is_explicitly_non_authoritative(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    dist, release, sbom, guard_dir, offline_dir = _write_valid_release_evidence(
        tmp_path
    )
    module = _release_checker_module(repo_root)
    failures = module.check_release_evidence(
        release_root=release,
        dist_root=dist,
        sbom_path=sbom,
        guard_validation_json=guard_dir / "guard-validation-smoke.json",
        guard_validation_markdown=guard_dir / "guard-validation-smoke.md",
        offline_bundle_dir=offline_dir,
    )
    assert failures == []
    base_args = _release_check_command(
        repo_root,
        release=release,
        dist=dist,
        sbom=sbom,
        guard_dir=guard_dir,
        offline_dir=offline_dir,
    )[2:]
    assert module.main(base_args) == 0
    assert module.main([*base_args, "--json"]) == 0

    proc = subprocess.run(
        _release_check_command(
            repo_root,
            release=release,
            dist=dist,
            sbom=sbom,
            guard_dir=guard_dir,
            offline_dir=offline_dir,
            json_output=True,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["ok"] is True
    assert summary["schema"] == "invarlock/release-artifact-shape-check-v1"
    assert summary["check_scope"] == "local artifact shape only"
    assert summary["authoritative_release_approval"] is False


def test_guard_validation_v1_rejects_forged_and_retired_artifacts(
    tmp_path: Path,
) -> None:
    module = _release_checker_module(_repo_root())
    guard_dir = tmp_path / "guard-validation"
    payload = _write_guard_smoke(guard_dir)
    json_path = guard_dir / "guard-validation-smoke.json"
    markdown_path = guard_dir / "guard-validation-smoke.md"

    failures: list[str] = []
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert failures == []

    forged = json.loads(json.dumps(payload))
    forged["rate_rows"][0]["null_trigger_rate"] = 0.5
    json_path.write_text(json.dumps(forged), encoding="utf-8")
    failures.clear()
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert any("null_trigger_rate does not match outcomes" in item for item in failures)
    assert any("evidence_sha256 does not match" in item for item in failures)

    retired = json.loads(json.dumps(payload))
    retired["schema"] = "invarlock/guard-validation-smoke-v2"
    json_path.write_text(json.dumps(retired), encoding="utf-8")
    failures.clear()
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert "guard-validation JSON schema is not recognized." in failures

    source_forgery = json.loads(json.dumps(payload))
    source_forgery["source_identity"]["producer"]["sha256"] = "sha256:" + "0" * 64
    json_path.write_text(json.dumps(source_forgery), encoding="utf-8")
    failures.clear()
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert any("producer source digest does not match" in item for item in failures)

    json_path.write_text(json.dumps(payload), encoding="utf-8")
    markdown_path.write_text("# forged\n", encoding="utf-8")
    failures.clear()
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert any("markdown bytes do not match" in item for item in failures)
    assert any("markdown does not render" in item for item in failures)


def test_guard_validation_v1_rejects_ambiguous_json_and_symlinks(
    tmp_path: Path,
) -> None:
    module = _release_checker_module(_repo_root())
    guard_dir = tmp_path / "guard-validation"
    _write_guard_smoke(guard_dir)
    json_path = guard_dir / "guard-validation-smoke.json"
    markdown_path = guard_dir / "guard-validation-smoke.md"
    original_json = json_path.read_text(encoding="utf-8")

    json_path.write_text(
        original_json.replace(
            '"schema": "invarlock/guard-validation-smoke-v1"',
            '"schema": "invarlock/guard-validation-smoke-v1", '
            '"schema": "invarlock/guard-validation-smoke-v1"',
        ),
        encoding="utf-8",
    )
    failures: list[str] = []
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert any("duplicate JSON key 'schema'" in item for item in failures)

    json_path.write_text(
        original_json.replace(
            '"null_trigger_rate": 0.0', '"null_trigger_rate": NaN', 1
        ),
        encoding="utf-8",
    )
    failures.clear()
    module._validate_guard_validation(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    )
    assert any("non-finite JSON value 'NaN'" in item for item in failures)

    json_path.write_text(original_json, encoding="utf-8")
    json_link = tmp_path / "guard.json"
    markdown_link = tmp_path / "guard.md"
    json_link.symlink_to(json_path)
    markdown_link.symlink_to(markdown_path)
    failures.clear()
    module._validate_guard_validation(
        json_path=json_link,
        markdown_path=markdown_link,
        failures=failures,
    )
    assert any("JSON must be a readable regular file" in item for item in failures)
    assert any("markdown must be a readable regular file" in item for item in failures)


def test_release_evidence_check_reports_missing_artifacts(tmp_path: Path) -> None:
    repo_root = _repo_root()
    module = _release_checker_module(repo_root)
    exit_code = module.main(
        _release_check_command(
            repo_root,
            release=tmp_path / "release",
            dist=tmp_path / "dist",
            sbom=tmp_path / "sbom.json",
            guard_dir=tmp_path / "guard-validation",
            offline_dir=tmp_path / "offline",
        )[2:]
    )
    assert exit_code == 1

    proc = subprocess.run(
        _release_check_command(
            repo_root,
            release=tmp_path / "release",
            dist=tmp_path / "dist",
            sbom=tmp_path / "sbom.json",
            guard_dir=tmp_path / "guard-validation",
            offline_dir=tmp_path / "offline",
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "wheel artifact missing" in proc.stderr
    assert "strict example report missing" in proc.stderr
    assert "guard-validation JSON must be a readable regular file" in proc.stderr
    assert "offline release bundle missing" in proc.stderr


def test_release_evidence_check_rejects_weak_artifact_contents(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    dist, release, sbom, guard_dir, offline_dir = _write_valid_release_evidence(
        tmp_path
    )
    (release / "wheel-sdist-hashes.txt").write_text(
        "0" * 64
        + "  invarlock-0.9.0-py3-none-any.whl\n"
        + "1" * 64
        + "  invarlock-0.9.0.tar.gz\n",
        encoding="utf-8",
    )
    (release / "runtime-image-digest.txt").write_text(
        "invarlock:latest\n",
        encoding="utf-8",
    )
    (release / "strict" / "evaluation.report.json").write_text(
        json.dumps(
            {
                "assurance": {
                    "mode": "off",
                    "verdict": "pass",
                    "fallback_fields_used": True,
                },
                "report_build": {"fallback_fields": ["primary_metric.display_ci"]},
            }
        ),
        encoding="utf-8",
    )
    (release / "strict" / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": False},
                "results": [
                    {
                        "id": "other.report.json",
                        "verification": {
                            "runtime_provenance": {
                                "status": "manifest_bound",
                                "verified": False,
                                "binding_verified": True,
                                "expected_digest_matched": False,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(
        (guard_dir / "guard-validation-smoke.json").read_text(encoding="utf-8")
    )
    payload["rate_rows"] = payload["rate_rows"][:4]
    (guard_dir / "guard-validation-smoke.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    bad_bundle = offline_dir / "bad.tar.gz"
    with tarfile.open(bad_bundle, "w:gz"):
        pass
    module = _release_checker_module(repo_root)
    failures = module.check_release_evidence(
        release_root=release,
        dist_root=dist,
        sbom_path=sbom,
        guard_validation_json=guard_dir / "guard-validation-smoke.json",
        guard_validation_markdown=guard_dir / "guard-validation-smoke.md",
        offline_bundle_dir=offline_dir,
    )
    assert any("wheel/sdist hash mismatch" in failure for failure in failures)
    assert any("assurance.mode must be strict" in failure for failure in failures)

    proc = subprocess.run(
        _release_check_command(
            repo_root,
            release=release,
            dist=dist,
            sbom=sbom,
            guard_dir=guard_dir,
            offline_dir=offline_dir,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "wheel/sdist hash mismatch" in proc.stderr
    assert "runtime image digest must contain exactly one" in proc.stderr
    assert "assurance.mode must be strict" in proc.stderr
    assert "summary.ok must be true" in proc.stderr
    assert "independently supplied runtime image digest pin" in proc.stderr
    assert "rate_rows must contain exactly 12 rows" in proc.stderr
    assert "offline release bundle manifest missing" in proc.stderr


def test_release_evidence_check_rejects_malformed_hash_and_json_edges(
    tmp_path: Path,
) -> None:
    module = _release_checker_module(_repo_root())
    failures: list[str] = []

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert module._load_json(invalid_json, "invalid payload", failures) is None
    assert any("invalid payload is not valid JSON" in item for item in failures)

    assert module._parse_hash_entries(tmp_path / "missing-hashes.txt", []) == {}
    hash_file = tmp_path / "hashes.txt"
    hash_file.write_text(
        "\n# comment\nnot-a-hash-line\n" + "2" * 64 + "  *dist/file.whl\n",
        encoding="utf-8",
    )
    failures.clear()
    entries = module._parse_hash_entries(hash_file, failures)
    assert entries["dist/file.whl"] == "2" * 64
    assert entries["file.whl"] == "2" * 64
    assert any("must use sha256sum format" in item for item in failures)

    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "invarlock-0.9.0-py3-none-any.whl"
    sdist = dist / "invarlock-0.9.0.tar.gz"
    wheel.write_text("wheel", encoding="utf-8")
    sdist.write_text("sdist", encoding="utf-8")
    hash_file.write_text(f"{_sha256(wheel)}  {wheel.name}\n", encoding="utf-8")
    failures.clear()
    module._validate_dist_hashes(dist_root=dist, hash_path=hash_file, failures=failures)
    assert any("hash missing for artifact" in item for item in failures)

    empty_hashes = tmp_path / "empty-hashes.txt"
    empty_hashes.write_text("# no hashes\n", encoding="utf-8")
    failures.clear()
    module._validate_dist_hashes(
        dist_root=dist,
        hash_path=empty_hashes,
        failures=failures,
    )
    assert any("has no valid entries" in item for item in failures)


def test_release_evidence_contract_owner_paths(tmp_path: Path) -> None:
    module = _release_checker_module(_repo_root())
    failures: list[str] = []
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.whl"
    artifact.write_text("wheel", encoding="utf-8")

    assert module._existing_globs(root, ("*.whl",)) == [artifact]
    assert module._dist_artifacts(root) == [artifact]
    assert module._sha256(artifact) == _sha256(artifact)
    assert module._load_json(tmp_path / "missing.json", "missing", failures) is None
    assert any("missing missing" in item for item in failures)

    failures.clear()
    module._require_file(artifact, "artifact", failures)
    module._require_any(root, ("*.whl",), "wheel", failures)
    assert failures == []

    failures.clear()
    missing_hashes = tmp_path / "missing-hashes.txt"
    module._validate_dist_hashes(
        dist_root=root,
        hash_path=missing_hashes,
        failures=failures,
    )
    assert failures == []

    failures.clear()
    empty_dist = tmp_path / "empty-dist"
    empty_dist.mkdir()
    hashes = tmp_path / "hashes.txt"
    hashes.write_text("", encoding="utf-8")
    module._validate_dist_hashes(
        dist_root=empty_dist,
        hash_path=hashes,
        failures=failures,
    )
    assert failures == []

    failures.clear()
    module._validate_runtime_digest(tmp_path / "missing-digest.txt", failures)
    assert failures == []

    failures.clear()
    bad_guard_json = tmp_path / "guard.json"
    bad_guard_json.write_text("[]", encoding="utf-8")
    empty_md = tmp_path / "guard.md"
    empty_md.write_text("", encoding="utf-8")
    module._validate_guard_validation(
        json_path=bad_guard_json,
        markdown_path=empty_md,
        failures=failures,
    )
    assert any(
        "guard-validation JSON must be a JSON object" in item for item in failures
    )

    failures.clear()
    bad_sbom = tmp_path / "sbom.json"
    bad_sbom.write_text("[]", encoding="utf-8")
    module._validate_sbom(bad_sbom, failures)
    assert failures == ["SBOM must be a JSON object."]

    failures.clear()
    good_sbom = tmp_path / "good-sbom.json"
    good_sbom.write_text('{"bomFormat": "CycloneDX"}', encoding="utf-8")
    module._validate_sbom(good_sbom, failures)
    assert failures == []

    failures.clear()
    DistHashManifest = module.DistHashManifest

    DistHashManifest({}).validate_artifacts(dist_root=root, failures=failures)
    assert failures == ["wheel/sdist hashes file has no valid entries."]

    failures.clear()
    empty_dist_root = tmp_path / "no-artifacts"
    empty_dist_root.mkdir()
    DistHashManifest({}).validate_artifacts(
        dist_root=empty_dist_root,
        failures=failures,
    )
    assert failures == []


def test_release_evidence_contract_edge_paths(tmp_path: Path) -> None:
    module = _release_checker_module(_repo_root())
    ReleaseEvidenceManifest = module.ReleaseEvidenceManifest

    dist = tmp_path / "dist"
    release = tmp_path / "release"
    guard_dir = tmp_path / "guard"
    offline = tmp_path / "offline"
    dist.mkdir()
    (dist / "invarlock-0.9.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (dist / "invarlock-0.9.0.tar.gz").write_text("sdist", encoding="utf-8")
    (release / "strict").mkdir(parents=True)
    (release / "wheel-sdist-hashes.txt").write_text("", encoding="utf-8")
    (release / "runtime-image-digest.txt").write_text(
        "sha256:" + "1" * 64,
        encoding="utf-8",
    )
    (release / "strict" / "evaluation.report.json").write_text(
        json.dumps(
            {
                "assurance": {
                    "mode": "strict",
                    "verdict": "pending_verifier",
                    "fallback_fields_used": False,
                },
                "report_build": {
                    "synthesized_fields": [],
                    "repaired_fields": [],
                    "fallback_fields": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (release / "strict" / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": True},
                "results": [
                    {
                        "id": "evaluation.report.json",
                        "verification": {
                            "runtime_provenance": {
                                "status": "expected_image_digest_matched",
                                "verified": True,
                                "binding_verified": True,
                                "expected_digest_matched": True,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sbom = tmp_path / "sbom.json"
    sbom.write_text("[]", encoding="utf-8")
    _write_guard_smoke(guard_dir)
    _write_offline_bundle(offline)

    failures = ReleaseEvidenceManifest(
        release_root=release,
        dist_root=dist,
        sbom_path=sbom,
        guard_validation_json=guard_dir / "guard-validation-smoke.json",
        guard_validation_markdown=guard_dir / "guard-validation-smoke.md",
        offline_bundle_dir=offline,
    ).validate()

    assert any("hashes file has no valid entries" in item for item in failures)
    assert "SBOM must be a JSON object." in failures


def test_release_evidence_check_rejects_report_and_verify_shape_edges(
    tmp_path: Path,
) -> None:
    module = _release_checker_module(_repo_root())
    failures: list[str] = []
    report = tmp_path / "evaluation.report.json"
    verify = tmp_path / "verify.json"

    report.write_text("[]", encoding="utf-8")
    module._validate_strict_report(report, failures)
    assert any(
        "strict example report must be a JSON object" in item for item in failures
    )

    failures.clear()
    report.write_text("{}", encoding="utf-8")
    module._validate_strict_report(report, failures)
    assert any("missing assurance object" in item for item in failures)

    failures.clear()
    report.write_text(
        json.dumps(
            {
                "assurance": {
                    "mode": "strict",
                    "verdict": "unexpected",
                    "fallback_fields_used": False,
                }
            }
        ),
        encoding="utf-8",
    )
    module._validate_strict_report(report, failures)
    assert any("assurance.verdict must be" in item for item in failures)
    assert any("missing report_build object" in item for item in failures)

    failures.clear()
    verify.write_text("[]", encoding="utf-8")
    module._validate_strict_verify(verify, report, failures)
    assert any(
        "strict verifier output must be a JSON object" in item for item in failures
    )

    failures.clear()
    verify.write_text(json.dumps({"summary": {"ok": True}, "results": []}))
    module._validate_strict_verify(verify, report, failures)
    assert any("must include at least one result" in item for item in failures)

    failures.clear()
    verify.write_text(
        json.dumps(
            {
                "summary": {"ok": True},
                "results": [
                    None,
                    {"id": str(report), "verification": "bad"},
                    {"id": str(report), "verification": {"runtime_provenance": "bad"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    module._validate_strict_verify(verify, report, failures)
    assert any(
        "independently supplied runtime image digest pin" in item for item in failures
    )
