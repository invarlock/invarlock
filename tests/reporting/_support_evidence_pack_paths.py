from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.reporting import verify_contract as verify_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

__all__ = [
    "RUNTIME_MANIFEST_FILENAME",
    "VerifyExecutionResult",
    "VerifyOutcome",
    "_build_pack",
    "_build_report_payload",
    "_digest",
    "_patch_verify_result",
    "_read_manifest",
    "_sign_pack",
    "_successful_verify_payload",
    "_successful_verify_result",
    "_write_build_inputs",
    "_write_json",
    "_write_manifest_and_checksums",
    "_write_pack_with_manifest",
    "_write_pack_scaffold",
    "_write_runtime_manifest",
    "evidence_pack_integrity_mod",
    "evidence_pack_mod",
]

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_manifest(pack_dir: Path) -> dict[str, object]:
    return json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(path: Path) -> str:
    return evidence_pack_mod._sha256_file(path)


def _digest_ref(path: Path, rel_path: str) -> dict[str, str]:
    return {
        "path": rel_path,
        "digest": f"sha256:{_sha256_file(path)}",
    }


def _write_runtime_manifest(report_path: Path) -> None:
    _write_json(
        report_path.parent / RUNTIME_MANIFEST_FILENAME,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-03-21T00:00:00+00:00",
            "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
            "execution_mode": "container",
            "report": {
                "filename": report_path.name,
                "path": report_path.as_posix(),
                "sha256": _sha256_file(report_path),
            },
            "config": {
                "path": None,
                "sha256": None,
                "source": "missing",
            },
            "runtime": {
                "container_execution": True,
                "image_digest": _VALID_TEST_IMAGE_DIGEST,
                "image_ref": "invarlock-runtime:local",
                "allow_network": False,
                "allow_remote_code": False,
                "allow_third_party_plugins": False,
            },
        },
    )


def _successful_verify_payload(reports: list[Path]) -> dict[str, object]:
    return {
        "format_version": "verify-v1",
        "ok": True,
        "reports": [str(path) for path in reports],
    }


def _successful_verify_result(
    reports: list[Path],
) -> VerifyExecutionResult:
    return VerifyExecutionResult(
        outcome=VerifyOutcome.OK,
        payload=_successful_verify_payload(reports),
        diagnostics=(),
    )


def _patch_verify_result(
    monkeypatch,  # noqa: ANN001
    *,
    outcome: VerifyOutcome = VerifyOutcome.OK,
    payload: object | None = None,
) -> None:
    if payload is None:
        payload = {"ok": outcome is VerifyOutcome.OK}

    def _run_verify_command(
        reports,  # noqa: ANN001
        profile=None,  # noqa: ANN001
        report_assurance="report",  # noqa: ANN001
    ) -> VerifyExecutionResult:
        _ = reports, profile, report_assurance
        return VerifyExecutionResult(outcome=outcome, payload=payload, diagnostics=())

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        _run_verify_command,
        raising=True,
    )


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_report_payload() -> dict[str, object]:
    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    return {
        "schema_version": "v1",
        "run_id": "evidence-pack-cli-test",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {
            "run_id": "baseline-run",
            "model_id": "model",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "artifacts_extra": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": True,
        },
        "rmt": {
            "evaluated": True,
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": True,
        },
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "evaluation_windows": {
            "final": {
                "logloss": [math.log(10.0)],
                "token_counts": [1],
            }
        },
    }


def _build_pack(
    pack_dir: Path,
    *,
    report_rel_path: str,
    report_payload: object | None = None,
) -> Path:
    final_verdict = pack_dir / "results/final_verdict.json"
    source_repo = pack_dir / "metadata/source_repo.json"
    environment = pack_dir / "metadata/environment.json"
    materials = pack_dir / "metadata/model_revisions.json"
    report = pack_dir / report_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    if report_payload is None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}", encoding="utf-8")
    else:
        _write_json(report, report_payload)
    _write_runtime_manifest(report)

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
        report_rel_path,
        str((Path(report_rel_path).parent / RUNTIME_MANIFEST_FILENAME).as_posix()),
    ]
    _write_checksums(pack_dir, covered)

    manifest = {
        "format": "evidence-pack-v1",
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


def _write_pack_scaffold(pack_dir: Path) -> tuple[Path, Path, Path]:
    report_path = (
        pack_dir / "reports" / "model" / "clean" / "noop" / "evaluation.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    _write_json(report_path.parent / RUNTIME_MANIFEST_FILENAME, {"ok": True})

    final_verdict = pack_dir / "results" / "final_verdict.json"
    environment = pack_dir / "metadata" / "environment.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(environment, {"platform": "test"})
    return report_path, final_verdict, environment


def _write_pack_with_manifest(
    pack_dir: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    checksum_lines: list[str] | None = None,
    with_error_report: bool = False,
) -> Path:
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
        manifest_overrides=manifest_overrides,
        checksum_lines=checksum_lines,
    )
    if with_error_report:
        error_dir = pack_dir / "reports" / "model" / "errors" / "noop"
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    return pack_dir


def _write_build_inputs(tmp_path: Path, *, readme: bool = False) -> SimpleNamespace:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    source_repo = tmp_path / "source_repo.json"
    environment = tmp_path / "environment.json"
    material = tmp_path / "material.json"
    readme_path = tmp_path / "README.md"

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(material, {"name": "demo"})
    if readme:
        readme_path.write_text("# Evidence Pack\n", encoding="utf-8")

    return SimpleNamespace(
        final_verdict=final_verdict,
        report_path=report_path,
        runtime_manifest=runtime_manifest,
        source_repo=source_repo,
        environment=environment,
        material=material,
        readme=readme_path,
    )


def _write_manifest_and_checksums(
    pack_dir: Path,
    *,
    report_path: Path,
    final_verdict: Path,
    environment: Path,
    manifest_overrides: dict[str, object] | None = None,
    checksum_lines: list[str] | None = None,
) -> None:
    rel_report = str(report_path.relative_to(pack_dir)).replace("\\", "/")
    rel_runtime = str(
        (report_path.parent / RUNTIME_MANIFEST_FILENAME).relative_to(pack_dir)
    ).replace("\\", "/")
    rel_verdict = str(final_verdict.relative_to(pack_dir)).replace("\\", "/")
    rel_environment = str(environment.relative_to(pack_dir)).replace("\\", "/")
    if checksum_lines is None:
        checksum_lines = [
            f"{evidence_pack_mod._sha256_bytes(final_verdict.read_bytes())}  {rel_verdict}",
            f"{evidence_pack_mod._sha256_bytes(environment.read_bytes())}  {rel_environment}",
            f"{evidence_pack_mod._sha256_bytes(report_path.read_bytes())}  {rel_report}",
            f"{evidence_pack_mod._sha256_bytes((report_path.parent / RUNTIME_MANIFEST_FILENAME).read_bytes())}  {rel_runtime}",
        ]
    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    manifest = {
        "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": evidence_pack_mod._sha256_bytes(
            checksums_path.read_bytes()
        ),
        "subject": {
            "name": "final_verdict",
            "path": rel_verdict,
            "digest": evidence_pack_mod._sha256_file(final_verdict),
        },
        "environment": {
            "path": rel_environment,
            "digest": evidence_pack_mod._sha256_file(environment),
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    _write_json(pack_dir / "manifest.json", manifest)


def _sign_pack(
    pack_dir: Path,
    tmp_path: Path,
    *,
    record_manifest_fingerprint: bool = True,
    manifest_fingerprint_override: str | None = None,
) -> str:
    key_root = (
        tmp_path
        / f"evidence-pack-signing-key-{len(list(tmp_path.glob('evidence-pack-signing-key-*.pem'))):02d}.pem"
    )
    private_key = key_root
    public_key = key_root.with_name(f"{key_root.stem}.pub.pem")
    fingerprint = evidence_pack_mod._generate_signing_keypair(
        private_key,
        public_key_path=public_key,
    )
    if record_manifest_fingerprint or manifest_fingerprint_override is not None:
        manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["signing_key_fingerprint"] = (
            fingerprint
            if manifest_fingerprint_override is None
            else manifest_fingerprint_override
        )
        _write_json(pack_dir / "manifest.json", manifest)
    evidence_pack_mod._sign_manifest(
        pack_dir / "manifest.json", signing_key_path=private_key
    )
    return fingerprint
