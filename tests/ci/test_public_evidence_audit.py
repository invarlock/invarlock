from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scripts.checks.public_evidence_checks.common import (
    PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
    _check_public_evidence_privacy,
    _sha256_file,
)
from scripts.checks.public_evidence_checks.index import (
    _check_packaged_public_evidence_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "checks" / "check_public_evidence.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("check_public_evidence", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_evidence_audit_accepts_empty_current_state() -> None:
    module = _load_audit_module()
    assert module.check_public_evidence() == []


def test_public_evidence_audit_respects_root_override(tmp_path: Path) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "fixtures" / "demo"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    (artifact_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "contract_fixture",
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )

    assert module.check_public_evidence(evidence_root) == []


def test_public_evidence_release_closure_requires_current_negative_index(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    evidence_root.mkdir()
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")

    errors = module.check_public_evidence(
        evidence_root,
        require_current_negative_evidence=True,
    )

    assert errors == [
        f"{evidence_root}: release closure requires a validated "
        "current negative-evidence index"
    ]


def test_historical_reclassification_requires_explicit_noncurrent_metadata(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "caught_regressions" / "not-current"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    (artifact_dir / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "runtime.manifest.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "historical_archived_fixture",
                "summary": "historical fixture",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )

    errors = module.check_public_evidence(evidence_root)

    assert any(
        "must retain its prior evidence classification" in error for error in errors
    )
    assert any(
        "must state its non-current verifier status" in error for error in errors
    )
    assert any("must expect current-contract rejection" in error for error in errors)


def test_public_evidence_audit_rejects_stale_schema_in_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "fixtures" / "demo"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    (artifact_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "contract_fixture",
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PUBLIC_EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(
        module,
        "_check_packaged_public_evidence_index",
        lambda *_args, **_kwargs: None,
    )

    errors = module.check_public_evidence(evidence_root)

    assert any("not valid under the current schema" in error for error in errors)


def test_public_evidence_audit_rejects_duplicate_root_pack_report(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "fixtures" / "demo"
    pack_report = artifact_dir / "evidence_pack" / "reports" / "report-001"
    pack_report.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    report_payload = json.dumps({"ok": True})
    (artifact_dir / "evaluation.report.json").write_text(
        report_payload,
        encoding="utf-8",
    )
    (pack_report / "evaluation.report.json").write_text(
        report_payload,
        encoding="utf-8",
    )
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "contract_fixture",
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )

    errors = module.check_public_evidence(evidence_root)

    assert any("duplicate root evaluation reports waste" in error for error in errors)
    assert any("duplicate of canonical pack report" in error for error in errors)


def test_public_evidence_audit_validates_packaged_index_local_artifacts(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "published_basis" / "demo"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    report_path = artifact_dir / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")
    index_path = tmp_path / "published_basis_index.json"
    index_path.write_text(
        json.dumps(
            {
                "format_version": PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
                "carrier_policy": {"installed_wheel": "compact_index_only"},
                "published_basis_count": 1,
                "entries": [
                    {
                        "slug": "demo",
                        "path": "public_evidence/published_basis/demo",
                        "artifacts": {
                            "evaluation_report": {
                                "kind": "file",
                                "path": (
                                    "public_evidence/published_basis/demo/"
                                    "evaluation.report.json"
                                ),
                                "size_bytes": report_path.stat().st_size,
                                "sha256": _sha256_file(report_path),
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors: list[str] = []
    _check_packaged_public_evidence_index(
        errors,
        evidence_root.resolve(),
        index_path=index_path,
    )

    assert errors == []


def test_public_evidence_audit_requires_external_asset_for_missing_index_artifact(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "public_evidence"
    evidence_root.mkdir(parents=True)
    index_path = tmp_path / "published_basis_index.json"
    index_path.write_text(
        json.dumps(
            {
                "format_version": PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
                "carrier_policy": {"installed_wheel": "compact_index_only"},
                "published_basis_count": 1,
                "entries": [
                    {
                        "slug": "demo",
                        "path": "public_evidence/published_basis/demo",
                        "artifacts": {
                            "evaluation_report": {
                                "kind": "file",
                                "path": (
                                    "public_evidence/published_basis/demo/"
                                    "evaluation.report.json"
                                ),
                                "size_bytes": 11,
                                "sha256": "sha256:" + "0" * 64,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors: list[str] = []
    _check_packaged_public_evidence_index(
        errors,
        evidence_root.resolve(),
        index_path=index_path,
    )

    assert any(
        "missing local artifact and external_asset reference" in error
        for error in errors
    )


def test_public_evidence_audit_accepts_external_asset_for_missing_index_artifact(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "public_evidence"
    evidence_root.mkdir(parents=True)
    index_path = tmp_path / "published_basis_index.json"
    index_path.write_text(
        json.dumps(
            {
                "format_version": PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
                "carrier_policy": {"installed_wheel": "compact_index_only"},
                "published_basis_count": 1,
                "entries": [
                    {
                        "slug": "demo",
                        "path": "public_evidence/published_basis/demo",
                        "artifacts": {
                            "evaluation_report": {
                                "kind": "file",
                                "path": (
                                    "public_evidence/published_basis/demo/"
                                    "evaluation.report.json"
                                ),
                                "size_bytes": 11,
                                "sha256": "sha256:" + "0" * 64,
                                "external_asset": {
                                    "url": (
                                        "https://github.com/example/repo/"
                                        "releases/download/v1/demo.tar.zst"
                                    ),
                                    "size_bytes": 11,
                                    "sha256": "sha256:" + "0" * 64,
                                },
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors: list[str] = []
    _check_packaged_public_evidence_index(
        errors,
        evidence_root.resolve(),
        index_path=index_path,
    )

    assert errors == []


def test_public_evidence_audit_rejects_private_execution_details(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    artifact_dir = evidence_root / "fixtures" / "demo"
    artifact_dir.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    (artifact_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence_pack_recipe.json").write_text(
        json.dumps(
            {
                "commands": [
                    "runner --host root@203.0.113.10 --out /root/private-run",
                    "evaluate --report-out /private/tmp/invarlock-report",
                    "evaluate --cache /Users/alice/invarlock-cache",
                    "evaluate --tmp /private/var/folders/ab/cd/T/invarlock-run",
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": "contract_fixture",
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )

    errors = module.check_public_evidence(evidence_root)

    assert any("root_ssh_target" in error for error in errors)
    assert any("private_ip_address" in error for error in errors)
    assert any("absolute_root_path" in error for error in errors)
    assert any("private_tmp_path" in error for error in errors)
    assert any("macos_user_home_path" in error for error in errors)
    assert any("private_macos_var_folder_path" in error for error in errors)
    assert not any("203.0.113.10" in error for error in errors)


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("artifact.json", '{"path":"\\u002froot\\u002fprivate-run"}\n'),
        ("artifact.jsonl", '{"path":"\\u002froot\\u002fprivate-run"}\n'),
        ("artifact.yaml", 'path: "\\u002froot\\u002fprivate-run"\n'),
    ],
)
def test_public_evidence_audit_scans_decoded_structured_values(
    tmp_path: Path,
    filename: str,
    payload: str,
) -> None:
    evidence_root = tmp_path / "public_evidence"
    evidence_root.mkdir()
    (evidence_root / filename).write_text(payload, encoding="utf-8")
    errors: list[str] = []

    _check_public_evidence_privacy(errors, evidence_root)

    assert any("decoded value: absolute_root_path" in error for error in errors)


def test_public_evidence_audit_rejects_provider_credentials_and_private_endpoints(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "public_evidence"
    evidence_root.mkdir()
    (evidence_root / "provider.json").write_text(
        json.dumps(
            {
                "provider": {
                    "api_key": "redacted-is-still-not-public-metadata",
                    "endpoint": "https://runner.private.internal/v1",
                }
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []

    _check_public_evidence_privacy(errors, evidence_root)

    assert any("credential_field" in error for error in errors)
    assert any("private_endpoint" in error for error in errors)


def _write_minimal_evidence_dir(
    artifact_dir: Path,
    *,
    module,
    report: dict,
    evidence_class: str = "strict_pass_fixture",
) -> None:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "evaluation.report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (artifact_dir / "runtime.manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evidence.meta.json").write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "evidence_class": evidence_class,
                "summary": "fixture report",
                "artifact_paths": {
                    "evaluation_report": "evaluation.report.json",
                    "runtime_manifest": "runtime.manifest.json",
                },
                "verifier_commands": ["invarlock verify evaluation.report.json"],
            }
        ),
        encoding="utf-8",
    )


def test_public_evidence_audit_rejects_low_quality_published_image_text(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    (evidence_root / "README.md").parent.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    _write_minimal_evidence_dir(
        evidence_root / "published_basis" / "weak_vlm",
        module=module,
        report={
            "dataset": {"provider": "vision_text"},
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.03,
                "n_final": 400,
                "counts_source": "measured",
                "estimated": False,
            },
            "classification": {
                "final": {"correct_total": 12, "total": 400},
            },
        },
    )

    errors = module.check_public_evidence(evidence_root)

    assert any("final accuracy 0.0300 is below 0.10" in error for error in errors)


def test_public_evidence_audit_accepts_adequate_published_image_text_shape_records(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    (evidence_root / "README.md").parent.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    _write_minimal_evidence_dir(
        evidence_root / "published_basis" / "adequate_vlm",
        module=module,
        report={
            "dataset": {"provider": "vision_text"},
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.85,
                "n_final": 400,
                "counts_source": "measured",
                "estimated": False,
            },
            "classification": {
                "final": {"correct_total": 340, "total": 400},
            },
            "eval_windows": {
                "final": {
                    "records": [
                        {"prediction": '{"answer": "red cup"}'},
                        {"prediction": '{"answer": "cat"}'},
                    ]
                }
            },
        },
    )

    assert module.check_public_evidence(evidence_root) == []


def test_public_evidence_audit_rejects_bad_embedded_answer_shape(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    evidence_root = tmp_path / "public_evidence"
    (evidence_root / "README.md").parent.mkdir(parents=True)
    (evidence_root / "README.md").write_text("# public evidence\n", encoding="utf-8")
    _write_minimal_evidence_dir(
        evidence_root / "published_basis" / "verbose_vlm",
        module=module,
        report={
            "dataset": {"provider": "vision_text"},
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.85,
                "n_final": 400,
                "counts_source": "measured",
                "estimated": False,
            },
            "classification": {
                "final": {"correct_total": 340, "total": 400},
            },
            "eval_windows": {
                "final": {
                    "records": [
                        {
                            "prediction": (
                                "The user wants me to inspect the image and explain "
                                "my reasoning before answering red cup."
                            )
                        },
                        {"prediction": '{"answer": "cat"}'},
                    ]
                }
            },
        },
    )

    errors = module.check_public_evidence(evidence_root)

    assert any("answer-shape rate 0.5000 is below 0.95" in error for error in errors)
