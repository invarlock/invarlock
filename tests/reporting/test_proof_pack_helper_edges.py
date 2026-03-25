from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.proof_pack as proof_pack_mod
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return proof_pack_mod._sha256_bytes(data)


def _digest(path: Path) -> str:
    return proof_pack_mod._sha256_file(path)


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
            f"{_sha256_bytes(final_verdict.read_bytes())}  {rel_verdict}",
            f"{_sha256_bytes(environment.read_bytes())}  {rel_environment}",
            f"{_sha256_bytes(report_path.read_bytes())}  {rel_report}",
            f"{_sha256_bytes((report_path.parent / RUNTIME_MANIFEST_FILENAME).read_bytes())}  {rel_runtime}",
        ]
    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    manifest = {
        "format": proof_pack_mod.PROOF_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_bytes(checksums_path.read_bytes()),
        "subject": {
            "name": "final_verdict",
            "path": rel_verdict,
            "digest": _digest(final_verdict),
        },
        "environment": {
            "path": rel_environment,
            "digest": _digest(environment),
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    _write_json(pack_dir / "manifest.json", manifest)


def test_manual_validate_manifest_reports_structural_errors() -> None:
    assert proof_pack_mod._manual_validate_manifest("bad") == [
        "manifest must decode to a JSON object"
    ]

    errors = proof_pack_mod._manual_validate_manifest(
        {
            "format": "wrong",
            "checksums_sha256": "other.txt",
            "checksums_sha256_digest": "short",
            "network_mode": "wifi",
            "artifacts": {},
            "builder": {"id": "", "name": ""},
            "subject": {"path": "", "digest": "bad"},
            "invocation": {"config_source": "bad", "parameters": "bad"},
            "environment": {"path": "", "digest": "bad"},
            "materials": [{"name": "", "path": "", "digest": "bad"}],
        }
    )

    assert any("manifest format must be" in error for error in errors)
    assert any("checksums_sha256 must point" in error for error in errors)
    assert any("64-char sha256 hex" in error for error in errors)
    assert any("network_mode must be" in error for error in errors)
    assert any("artifacts must be a list" in error for error in errors)
    assert any("builder.id must be a non-empty string" in error for error in errors)
    assert any("builder.name must be a non-empty string" in error for error in errors)
    assert any("subject.path must be a non-empty string" in error for error in errors)
    assert any(
        "subject.digest must be a sha256:... string" in error for error in errors
    )
    assert any(
        "invocation.config_source must be an object" in error for error in errors
    )
    assert any("invocation.parameters must be an object" in error for error in errors)
    assert any(
        "environment.path must be a non-empty string" in error for error in errors
    )
    assert any(
        "materials[0].name must be a non-empty string" in error for error in errors
    )

    errors = proof_pack_mod._manual_validate_manifest(
        {
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "builder": "bad",
            "invocation": "bad",
            "materials": "bad",
        }
    )
    assert any("manifest missing required field: format" in error for error in errors)
    assert any("manifest builder must be an object" in error for error in errors)
    assert any("manifest invocation must be an object" in error for error in errors)
    assert any("manifest materials must be a list" in error for error in errors)


def test_validate_manifest_and_load_json_object_cover_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")
    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "load_proof_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod.jsonschema,
        "validate",
        lambda instance, schema: (_ for _ in ()).throw(ValueError("schema boom")),
        raising=True,
    )
    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    payload, errors = proof_pack_mod._load_json_object(
        tmp_path / "missing.json", label="demo"
    )
    assert payload is None
    assert errors == [f"demo file not found: {tmp_path / 'missing.json'}"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    payload, errors = proof_pack_mod._load_json_object(bad_json, label="demo")
    assert payload is None
    assert "demo is not valid JSON" in errors[0]

    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2]", encoding="utf-8")
    payload, errors = proof_pack_mod._load_json_object(array_json, label="demo")
    assert payload is None
    assert errors == [f"demo must decode to a JSON object: {array_json}"]


def test_material_and_reference_helpers_cover_invalid_paths(tmp_path: Path) -> None:
    assert proof_pack_mod._material_spec("missing-separator") is None
    assert proof_pack_mod._material_spec(" =demo.json") is None
    assert proof_pack_mod._material_spec("demo= ") is None
    assert proof_pack_mod._material_spec("demo=payload.json") == (
        "demo",
        Path("payload.json"),
    )

    assert proof_pack_mod._validate_material_name("good.name-1") is None
    assert "material names must match" in str(
        proof_pack_mod._validate_material_name("../bad")
    )

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    target = pack_dir / "file.json"
    target.write_text("{}", encoding="utf-8")

    assert (
        proof_pack_mod._validate_reference(
            pack_dir=pack_dir, label="demo", payload="bad"
        )
        == []
    )
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"digest": _digest(target)},
    ) == ["demo must include a non-empty path when digest verification is enabled"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "bad"},
    ) == ["demo digest must be a sha256:... string"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "../escape.json", "digest": _digest(target)},
    ) == ["demo path escapes the pack root: ../escape.json"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={
            "path": "missing.json",
            "digest": "sha256:" + ("a" * 64),
        },
    ) == ["demo path is missing: missing.json"]
    mismatch = proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "sha256:" + ("b" * 64)},
    )
    assert "demo digest mismatch for file.json" in mismatch[0]
    assert (
        proof_pack_mod._validate_reference(
            pack_dir=pack_dir,
            label="demo",
            payload={"path": "file.json", "digest": _digest(target)},
        )
        == []
    )


def test_checksum_and_extra_file_helpers_cover_error_paths(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("checksums_sha256_digest")
    _write_json(pack_dir / "manifest.json", manifest)
    assert proof_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json missing checksums_sha256_digest (pack is not tamper-evident)."
    ]

    manifest["checksums_sha256_digest"] = ""
    _write_json(pack_dir / "manifest.json", manifest)
    assert proof_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json checksums_sha256_digest is empty."
    ]

    manifest["checksums_sha256_digest"] = "a" * 64
    _write_json(pack_dir / "manifest.json", manifest)
    assert (
        "checksums.sha256 digest mismatch"
        in proof_pack_mod._verify_manifest_binds_checksums(pack_dir)[0]
    )

    (pack_dir / "checksums.sha256").write_text(
        "\n".join(
            [
                "not a checksum line",
                f"{'a' * 64}  ../escape.txt",
                f"{'b' * 64}  missing.txt",
                f"{'c' * 64}  {final_verdict.relative_to(pack_dir).as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entries, errors = proof_pack_mod._parse_checksums(pack_dir)
    assert entries[0][1] == "../escape.txt"
    assert "line 1 is not a valid sha256 entry" in errors[0]
    checksum_errors, covered = proof_pack_mod._verify_checksums(pack_dir)
    assert "../escape.txt" in covered
    assert any("escapes the pack root" in error for error in checksum_errors)
    assert any("missing from pack: missing.txt" in error for error in checksum_errors)
    assert any(
        "checksum mismatch for results/final_verdict.json" in error
        for error in checksum_errors
    )

    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    extra_errors, extra_warnings = proof_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered, strict=False
    )
    assert extra_errors == []
    assert "extra files not covered" in extra_warnings[0]
    extra_errors, extra_warnings = proof_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered, strict=True
    )
    assert "extra files not covered" in extra_errors[0]
    assert extra_warnings == []


def test_verify_gpg_covers_missing_binary_signature_failure_and_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert errors == []
    assert warnings == ["manifest.json.asc missing; pack is unsigned."]
    assert fingerprint is None

    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=True)
    assert errors == [
        "manifest.json.asc missing (strict mode requires a signed manifest)."
    ]
    assert warnings == []
    assert fingerprint is None

    (pack_dir / "manifest.json.asc").write_text("sig", encoding="utf-8")
    monkeypatch.setattr(
        proof_pack_mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
        raising=True,
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert errors == []
    assert "gpg not found" in warnings[0]
    assert fingerprint is None

    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=True)
    assert errors == ["gpg not found (strict mode requires signature verification)."]
    assert warnings == []
    assert fingerprint is None

    monkeypatch.setattr(
        proof_pack_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="bad signature"
        ),
        raising=True,
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert "manifest signature verification failed." in errors[0]
    assert warnings == []
    assert fingerprint is None

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["signing_key_fingerprint"] = "EXPECTED"
    _write_json(pack_dir / "manifest.json", manifest)
    monkeypatch.setattr(
        proof_pack_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="[GNUPG:] VALIDSIG ACTUAL 20260101 0 4 0 1 10 00 00\n",
            stderr="",
        ),
        raising=True,
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert "signing_key_fingerprint (EXPECTED) does not match" in errors[0]
    assert warnings == []
    assert fingerprint == "ACTUAL"


def test_verify_reports_and_inspect_cover_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_pack = tmp_path / "empty"
    empty_pack.mkdir()
    errors, payload = proof_pack_mod._verify_reports(
        empty_pack, json_out_path=None, profile="dev"
    )
    assert errors == ["No reports found in pack."]
    assert payload is None

    error_only_pack = tmp_path / "error-only"
    error_report = (
        error_only_pack
        / "reports"
        / "model"
        / "errors"
        / "noop"
        / "evaluation.report.json"
    )
    error_report.parent.mkdir(parents=True, exist_ok=True)
    error_report.write_text("{}", encoding="utf-8")
    errors, payload = proof_pack_mod._verify_reports(
        error_only_pack, json_out_path=None, profile="dev"
    )
    assert errors == [
        "No clean reports found in pack (only error-injection reports present)."
    ]
    assert payload is None

    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    (pack_dir / "reports" / "model" / "errors" / "noop").mkdir(
        parents=True, exist_ok=True
    )
    (
        pack_dir / "reports" / "model" / "errors" / "noop" / "evaluation.report.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )
    json_out = tmp_path / "nested.json"
    verify_calls: list[list[str]] = []

    def _fake_run_verify(reports: list[Path], *, profile: str):
        verify_calls.append([str(path) for path in reports])
        if len(verify_calls) == 1:
            return 1, {"ok": False}
        raise RuntimeError("ignore nested error reports")

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        _fake_run_verify,
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_dir, json_out_path=json_out, profile="release"
    )
    assert errors == ["invarlock verify reported report verification failures."]
    assert payload == {"ok": False}
    assert json.loads(json_out.read_text(encoding="utf-8")) == {"ok": False}
    assert len(verify_calls) == 2

    missing_payload, exit_code = proof_pack_mod.inspect_proof_pack(tmp_path / "missing")
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert missing_payload["ok"] is False

    invalid_pack = tmp_path / "invalid"
    invalid_pack.mkdir()
    (invalid_pack / "manifest.json").write_text("{invalid", encoding="utf-8")
    (invalid_pack / "checksums.sha256").write_text("", encoding="utf-8")
    invalid_payload, exit_code = proof_pack_mod.inspect_proof_pack(invalid_pack)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_FORMAT
    assert invalid_payload["ok"] is False


def test_build_and_verify_proof_pack_cover_usage_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"ok": True})
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    _write_json(runtime_manifest, {"ok": True})
    source_repo = tmp_path / "source_repo.json"
    _write_json(source_repo, {"commit": "abc123"})
    environment = tmp_path / "environment.json"
    _write_json(environment, {"platform": "test"})
    material = tmp_path / "material.json"
    _write_json(material, {"name": "demo"})

    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-none",
        final_verdict_path=final_verdict,
        report_paths=[],
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_USAGE
    assert "at least one --report input" in payload["errors"][0]

    existing_out = tmp_path / "existing"
    existing_out.mkdir()
    payload, exit_code = proof_pack_mod.build_proof_pack(
        existing_out,
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_USAGE
    assert "already exists" in payload["errors"][0]

    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-invalid-material",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        material_specs=[("../bad", material), ("../bad", material)],
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_FORMAT
    assert any("Invalid material name" in error for error in payload["errors"])
    assert any("Duplicate material name" in error for error in payload["errors"])

    runtime_manifest.unlink()
    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-missing-sidecar",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_FORMAT
    assert any("report sidecar file not found" in error for error in payload["errors"])
    _write_json(runtime_manifest, {"ok": True})

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: (2, {"ok": False}),
        raising=True,
    )
    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-verify-fail",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_REPORTS
    assert payload["verify"] == {"ok": False}

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: (0, {"ok": True}),
        raising=True,
    )
    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-ok",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        source_repo_path=source_repo,
        environment_path=environment,
        material_specs=[("demo", material)],
        readme_path=tmp_path / "missing-readme.md",
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_OK
    assert payload["ok"] is True
    assert any("README file not found" in warning for warning in payload["warnings"])

    payload, exit_code = proof_pack_mod.verify_proof_pack(
        tmp_path / "missing-pack", skip_verify=True
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert payload["ok"] is False

    payload, exit_code = proof_pack_mod.verify_proof_pack(
        tmp_path / "out-ok",
        json_out_path=(tmp_path / "out-ok" / "verify.json"),
        skip_verify=True,
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_USAGE
    assert "--json-out must point outside the pack directory." in payload["errors"]


def test_manual_validate_manifest_accepts_valid_optional_sections() -> None:
    payload = {
        "format": proof_pack_mod.PROOF_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": "a" * 64,
        "network_mode": "offline",
        "artifacts": [],
        "builder": {"id": "builder-1", "name": "Builder"},
        "subject": {
            "path": "results/final_verdict.json",
            "digest": "sha256:" + ("b" * 64),
        },
        "invocation": {
            "config_source": {
                "path": "metadata/source_repo.json",
                "digest": "sha256:" + ("c" * 64),
            },
            "parameters": {"profile": "ci"},
        },
        "environment": {
            "path": "metadata/environment.json",
            "digest": "sha256:" + ("d" * 64),
        },
        "materials": [
            {
                "name": "evidence",
                "path": "metadata/evidence.json",
                "digest": "sha256:" + ("e" * 64),
            }
        ],
    }

    assert proof_pack_mod._manual_validate_manifest(payload) == []


def test_validate_manifest_uses_manual_validation_when_schema_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "wrong",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "load_proof_pack_manifest_schema",
        lambda: None,
        raising=True,
    )

    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert any("manifest format must be" in error for error in errors)


def test_validate_reference_allows_empty_path_and_digest_pair(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    assert (
        proof_pack_mod._validate_reference(
            pack_dir=pack_dir,
            label="demo",
            payload={"path": None, "digest": None},
        )
        == []
    )


def test_verify_manifest_attestation_rejects_non_object_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert proof_pack_mod.verify_manifest_attestation(pack_dir) == [
        "manifest must decode to a JSON object"
    ]


def test_verify_manifest_attestation_skips_non_dict_invocation_and_materials(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
        manifest_overrides={
            "subject": None,
            "invocation": "not-a-dict",
            "materials": "not-a-list",
        },
    )

    assert proof_pack_mod.verify_manifest_attestation(pack_dir) == []


def test_parse_checksums_ignores_blank_lines(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "checksums.sha256").write_text(
        f"\n{'a' * 64}  results/final_verdict.json\n\n",
        encoding="utf-8",
    )

    entries, errors = proof_pack_mod._parse_checksums(pack_dir)
    assert errors == []
    assert entries == [("a" * 64, "results/final_verdict.json")]


def test_verify_gpg_success_without_validsig_returns_no_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    (pack_dir / "manifest.json.asc").write_text("sig", encoding="utf-8")

    monkeypatch.setattr(
        proof_pack_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="[GNUPG:] GOODSIG TEST KEY\n",
            stderr="",
        ),
        raising=True,
    )

    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert errors == []
    assert warnings == []
    assert fingerprint is None


def test_verify_gpg_success_with_matching_fingerprint_returns_signer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["signing_key_fingerprint"] = "MATCHED"
    _write_json(pack_dir / "manifest.json", manifest)
    (pack_dir / "manifest.json.asc").write_text("sig", encoding="utf-8")

    monkeypatch.setattr(
        proof_pack_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="[GNUPG:] VALIDSIG MATCHED 20260101 0 4 0 1 10 00 00\n",
            stderr="",
        ),
        raising=True,
    )

    errors, warnings, fingerprint = proof_pack_mod._verify_gpg(pack_dir, strict=False)
    assert errors == []
    assert warnings == []
    assert fingerprint == "MATCHED"


def test_inspect_proof_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    _write_json(missing_manifest / "checksums.sha256", {})

    payload, exit_code = proof_pack_mod.inspect_proof_pack(missing_manifest)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert payload["issues"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    payload, exit_code = proof_pack_mod.inspect_proof_pack(missing_checksums)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert payload["issues"] == ["checksums.sha256 missing in pack."]


def test_inspect_proof_pack_signed_pack_omits_unsigned_warning_and_reports_extras(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    (pack_dir / "manifest.json.asc").write_text("sig", encoding="utf-8")
    (pack_dir / "extra.bin").write_text("extra", encoding="utf-8")

    payload, exit_code = proof_pack_mod.inspect_proof_pack(pack_dir)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_OK
    assert not any("pack is unsigned" in issue for issue in payload["issues"])
    assert any("extra files not covered" in issue for issue in payload["issues"])


def test_build_proof_pack_copies_readme_and_environment_without_optional_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    environment = tmp_path / "environment.json"
    readme = tmp_path / "README.md"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})
    _write_json(environment, {"platform": "test"})
    readme.write_text("# Proof Pack\n", encoding="utf-8")

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: (0, {"ok": True}),
        raising=True,
    )

    payload, exit_code = proof_pack_mod.build_proof_pack(
        tmp_path / "out-readme",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        environment_path=environment,
        readme_path=readme,
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_OK
    assert payload["ok"] is True
    manifest = json.loads(
        (tmp_path / "out-readme" / "manifest.json").read_text(encoding="utf-8")
    )
    assert "invocation" not in manifest
    assert "materials" not in manifest
    assert manifest["environment"]["path"] == "metadata/environment.json"
    assert (tmp_path / "out-readme" / "README.md").is_file()


def test_verify_proof_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    (missing_manifest / "checksums.sha256").write_text("", encoding="utf-8")

    payload, exit_code = proof_pack_mod.verify_proof_pack(
        missing_manifest, skip_verify=True
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert payload["errors"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    payload, exit_code = proof_pack_mod.verify_proof_pack(
        missing_checksums, skip_verify=True
    )
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_MISSING
    assert payload["errors"] == ["checksums.sha256 missing in pack."]


def test_verify_proof_pack_returns_format_for_invalid_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": "wrong",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    (pack_dir / "checksums.sha256").write_text("", encoding="utf-8")

    payload, exit_code = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_FORMAT
    assert payload["errors"]
    assert any(
        "schema validation failed" in error or "manifest format must be" in error
        for error in payload["errors"]
    )


def test_verify_proof_pack_returns_signature_failure_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_gpg",
        lambda pack_dir, strict: (["bad signature"], [], "FPR123"),
        raising=True,
    )

    payload, exit_code = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)
    assert exit_code == proof_pack_mod.PROOF_PACK_VERIFY_SIGNATURE
    assert payload["errors"] == ["bad signature"]
    assert payload["signer_fingerprint"] == "FPR123"


def test_build_verify_result_includes_signer_and_verify_payload(tmp_path: Path) -> None:
    payload = proof_pack_mod._build_verify_result(
        pack_dir=tmp_path / "pack",
        ok=False,
        strict=True,
        skip_verify=False,
        warnings=["warn"],
        errors=["err"],
        signer_fingerprint="ABC123",
        verify_payload={"ok": False},
        exit_code=proof_pack_mod.PROOF_PACK_VERIFY_SIGNATURE,
    )

    assert payload["signer_fingerprint"] == "ABC123"
    assert payload["verify"] == {"ok": False}
