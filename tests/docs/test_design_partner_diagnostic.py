from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = REPO_ROOT / "docs" / "user-guide" / "design-partner-diagnostic.md"
CASE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "design_partner_diagnostic"
    / "case.env.example"
)
COMPARE_WRAPPER = (
    REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
)
HANDOFF_WRAPPER = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "public_e2e"
    / "run_public_e2e_release_review.sh"
)
HANDOFF_BINDER = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "design_partner_diagnostic"
    / "bind_subject_handoff.py"
)


def _run_handoff_binder(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HANDOFF_BINDER), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _template_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        names.add(stripped.split("=", 1)[0])
    return names


def test_design_partner_case_template_is_complete_and_checked_by_runbook() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    template = CASE_TEMPLATE.read_text(encoding="utf-8")
    names = _template_names(template)

    assert names == {
        "ALLOW_NETWORK",
        "BASELINE_ADAPTER",
        "BASELINE_MODEL",
        "BASELINE_REPORT",
        "BASELINE_REVISION",
        "CASE_ID",
        "EXPECTED_GUARD_AUTHORITY",
        "EXPECTED_RUNTIME_IMAGE_DIGEST",
        "POLICY_PACK",
        "PROFILE",
        "REPORT_OUT",
        "SUBJECT_ADAPTER",
        "SUBJECT_CHANGE_KIND",
        "SUBJECT_MODEL",
        "SUBJECT_REVISION",
        "SUBJECT_TRANSFORMATION_RECEIPT",
        "TIER",
    }
    for name in names:
        assert name in runbook

    assert 'EXPECTED_GUARD_AUTHORITY="enforce"' in template
    assert 'PROFILE="ci"' in template
    assert 'TIER="balanced"' in template
    assert 'ALLOW_NETWORK="0"' in template
    assert "REPLACE_WITH_" in template


def test_design_partner_commands_follow_supported_wrapper_contracts() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    compare = COMPARE_WRAPPER.read_text(encoding="utf-8")
    handoff = HANDOFF_WRAPPER.read_text(encoding="utf-8")

    compare_flags = {
        "--baseline",
        "--subject",
        "--subject-revision",
        "--baseline-revision",
        "--baseline-adapter",
        "--subject-adapter",
        "--baseline-report",
        "--policy-pack",
        "--expected-runtime-image-digest",
        "--profile",
        "--tier",
        "--lane",
        "--report-out",
        "--allow-network",
    }
    handoff_flags = {
        "--report",
        "--baseline",
        "--policy-pack",
        "--expected-runtime-image-digest",
        "--profile",
        "--assurance",
        "--runtime-provenance",
        "--output-dir",
        "--force",
    }
    for flag in compare_flags:
        assert flag in runbook
        assert flag in compare
    for flag in handoff_flags:
        assert flag in runbook
        assert flag in handoff

    for artifact in (
        "evaluation.report.json",
        "verify.json",
        "evaluation.html",
        "run_summary.txt",
        "run_command.txt",
        "lane_artifact.json",
        "invarlock-verify.json",
        "release-review.md",
        "ci-summary.md",
        "subject-handoff-binding.json",
        "subject-transformation-receipt",
    ):
        assert artifact in runbook


def test_design_partner_handoff_binds_remote_subject_inputs_and_detects_tampering(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    report_path = handoff_dir / "evaluation.report.json"
    report = {
        "meta": {
            "model_id": "partner/transformed-model",
            "model_identity": {
                "kind": "remote_revision",
                "revision": revision,
            },
        },
        "subject_ref": {
            "model_id": "partner/transformed-model",
            "model_identity": {
                "kind": "remote_revision",
                "revision": revision,
            },
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt_path = tmp_path / "vendor-transformation-receipt.json"
    receipt_bytes = b'{"tool":"partner-quantizer","output":"subject"}\n'
    receipt_path.write_bytes(receipt_bytes)

    created = _run_handoff_binder(
        "create",
        "--handoff-dir",
        str(handoff_dir),
        "--subject-model",
        "hf:partner/transformed-model",
        "--subject-revision",
        revision,
        "--subject-change-kind",
        "quantization",
        "--transformation-receipt",
        str(receipt_path),
    )
    assert created.returncode == 0, created.stderr

    copied_receipt = handoff_dir / "subject-transformation-receipt"
    manifest_path = handoff_dir / "subject-handoff-binding.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert copied_receipt.read_bytes() == receipt_bytes
    assert manifest == {
        "schema": "invarlock/design-partner-subject-handoff-v1",
        "subject": {
            "change_kind": "quantization",
            "model_id": "partner/transformed-model",
            "model_identity": {
                "kind": "remote_revision",
                "revision": revision,
            },
        },
        "artifacts": {
            "evaluation_report": {
                "path": "evaluation.report.json",
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "size_bytes": report_path.stat().st_size,
            },
            "transformation_receipt": {
                "path": "subject-transformation-receipt",
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "size_bytes": len(receipt_bytes),
            },
        },
    }
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")

    verified = _run_handoff_binder("verify", "--handoff-dir", str(handoff_dir))
    assert verified.returncode == 0, verified.stderr

    copied_receipt.write_bytes(receipt_bytes[:-1] + b"X")
    rejected = _run_handoff_binder("verify", "--handoff-dir", str(handoff_dir))
    assert rejected.returncode != 0
    assert "transformation receipt digest mismatch" in rejected.stderr


def test_design_partner_handoff_rejects_missing_or_mismatched_remote_revision(
    tmp_path: Path,
) -> None:
    report_revision = "b" * 40
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    (handoff_dir / "evaluation.report.json").write_text(
        json.dumps(
            {
                "meta": {
                    "model_id": "partner/remote-subject",
                    "model_identity": {
                        "kind": "remote_revision",
                        "revision": report_revision,
                    },
                },
                "subject_ref": {
                    "model_id": "partner/remote-subject",
                    "model_identity": {
                        "kind": "remote_revision",
                        "revision": report_revision,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.txt"
    receipt_path.write_text("reviewer receipt\n", encoding="utf-8")
    common = (
        "create",
        "--handoff-dir",
        str(handoff_dir),
        "--subject-model",
        "partner/remote-subject",
        "--subject-change-kind",
        "fine-tuning",
        "--transformation-receipt",
        str(receipt_path),
    )

    missing = _run_handoff_binder(*common)
    assert missing.returncode != 0
    assert "remote subject requires an immutable --subject-revision" in missing.stderr

    mismatched = _run_handoff_binder(
        *common,
        "--subject-revision",
        "c" * 40,
    )
    assert mismatched.returncode != 0
    assert "does not match evaluation report subject identity" in mismatched.stderr


def test_design_partner_handoff_rejects_generated_artifact_symlink_destinations(
    tmp_path: Path,
) -> None:
    revision = "e" * 40
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text('{"producer":"partner"}\n', encoding="utf-8")

    def prepare_handoff(name: str) -> Path:
        handoff_dir = tmp_path / name
        handoff_dir.mkdir()
        (handoff_dir / "evaluation.report.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "model_id": "partner/subject",
                        "model_identity": {
                            "kind": "remote_revision",
                            "revision": revision,
                        },
                    },
                    "subject_ref": {
                        "model_id": "partner/subject",
                        "model_identity": {
                            "kind": "remote_revision",
                            "revision": revision,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return handoff_dir

    def create(handoff_dir: Path) -> subprocess.CompletedProcess[str]:
        return _run_handoff_binder(
            "create",
            "--handoff-dir",
            str(handoff_dir),
            "--subject-model",
            "partner/subject",
            "--subject-revision",
            revision,
            "--subject-change-kind",
            "quantization",
            "--transformation-receipt",
            str(receipt_path),
        )

    outside_receipt = tmp_path / "outside-receipt"
    outside_receipt.write_bytes(b"must remain unchanged\n")
    receipt_handoff = prepare_handoff("receipt-symlink")
    (receipt_handoff / "subject-transformation-receipt").symlink_to(outside_receipt)
    receipt_result = create(receipt_handoff)
    assert receipt_result.returncode != 0
    assert "transformation receipt destination must be absent or a regular file" in (
        receipt_result.stderr
    )
    assert outside_receipt.read_bytes() == b"must remain unchanged\n"
    assert not (receipt_handoff / "subject-handoff-binding.json").exists()

    outside_manifest = tmp_path / "outside-manifest"
    outside_manifest.write_bytes(b"must also remain unchanged\n")
    manifest_handoff = prepare_handoff("manifest-symlink")
    (manifest_handoff / "subject-handoff-binding.json").symlink_to(outside_manifest)
    manifest_result = create(manifest_handoff)
    assert manifest_result.returncode != 0
    assert "subject handoff binding destination must be absent or a regular file" in (
        manifest_result.stderr
    )
    assert outside_manifest.read_bytes() == b"must also remain unchanged\n"
    assert not (manifest_handoff / "subject-transformation-receipt").exists()


def test_design_partner_bash_blocks_are_syntactically_valid() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)

    assert len(blocks) == 5
    for block in blocks:
        subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=True,
        )


def test_design_partner_acceptance_check_accepts_bound_enforce_result_and_rejects_observe(
    tmp_path: Path,
) -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    acceptance_block = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)[-1]
    report_path = tmp_path / "evaluation.report.json"
    verify_path = tmp_path / "verify.json"
    authority = {
        "spectral": "enforce",
        "rmt": "enforce",
        "variance": "enforce",
    }
    report = {
        "meta": {
            "model_id": "partner/subject",
            "model_identity": {
                "kind": "remote_revision",
                "revision": "d" * 40,
            },
        },
        "subject_ref": {
            "model_id": "partner/subject",
            "model_identity": {
                "kind": "remote_revision",
                "revision": "d" * 40,
            },
        },
        "resolved_policy": {"guard_authority": authority},
        "assurance": {"guard_authority": authority},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    verify = {
        "summary": {"ok": True},
        "results": [
            {
                "ok": True,
                "verification": {
                    "runtime_provenance": {"status": "expected_image_digest_matched"},
                    "receipt": {"subject_report_sha256": report_digest},
                },
            }
        ],
    }
    verify_path.write_text(json.dumps(verify), encoding="utf-8")
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    (handoff_dir / "evaluation.report.json").write_bytes(report_path.read_bytes())
    transformation_receipt = tmp_path / "transformation-receipt.json"
    transformation_receipt.write_text('{"ok":true}\n', encoding="utf-8")
    bound = _run_handoff_binder(
        "create",
        "--handoff-dir",
        str(handoff_dir),
        "--subject-model",
        "partner/subject",
        "--subject-revision",
        "d" * 40,
        "--subject-change-kind",
        "quantization",
        "--transformation-receipt",
        str(transformation_receipt),
    )
    assert bound.returncode == 0, bound.stderr
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "REPORT_OUT": str(tmp_path),
    }

    accepted = subprocess.run(
        ["bash"],
        input=acceptance_block,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "strict all-enforce diagnostic accepted" in accepted.stdout

    report["resolved_policy"]["guard_authority"]["rmt"] = "observe"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rejected = subprocess.run(
        ["bash"],
        input=acceptance_block,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert rejected.returncode != 0
    assert "resolved policy is not all-enforce" in rejected.stderr


def test_design_partner_runbook_keeps_acceptance_and_scope_explicit() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "genuinely transformed" in normalized
    assert "all-`enforce`" in text
    assert "`observe`" in text
    assert "does not satisfy this runbook's all-enforce success criterion" in normalized
    assert "verification.receipt" in text
    assert "receipt_bound_untrusted" in normalized
    assert "no general-purpose evidence-pack build command" in normalized
    for unsupported_runtime in ("tensorrt", "onnx", "gguf", "coreml"):
        assert unsupported_runtime in normalized
