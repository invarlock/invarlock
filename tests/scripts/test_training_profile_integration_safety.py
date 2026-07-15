from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evidence_packs.python.editing.training_receipt import (
    with_receipt_digest,
)
from scripts.evidence_packs.python.editing.training_runtime import directory_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "examples" / "integrations" / "_shared" / "training_profiles.sh"
RUNNERS = (
    REPO_ROOT / "examples" / "integrations" / "peft_lora" / "run_tiny_peft_lora.sh",
    REPO_ROOT / "examples" / "integrations" / "fine_tune" / "run_tiny_fine_tune.sh",
)


@pytest.mark.parametrize(
    ("runner", "profile_id"),
    [
        (RUNNERS[0], "tiny_gpt2_lora_v1"),
        (RUNNERS[1], "tiny_gpt2_full_ft_v1"),
    ],
)
def test_training_runners_accept_path_lookup_python_command(
    runner: Path,
    profile_id: str,
    tmp_path: Path,
) -> None:
    """A selected bare command is valid through PATH, not as a local file."""
    fake_python = tmp_path / "python"
    invocation_log = tmp_path / "python-invocations.log"
    fake_python.write_text(
        "#!/bin/sh\n"
        'if [ "${1-}" = "-c" ]; then\n'
        "  exit 0\n"
        "fi\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_PYTHON_LOG"\n'
        "exit 71\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_LOG": str(invocation_log),
    }
    env.pop("PYTHON_BIN", None)

    completed = subprocess.run(
        [
            str(runner),
            "--lane",
            "host",
            "--device",
            "cpu",
            "--training-profile",
            profile_id,
            "--subject-dir",
            str(tmp_path / "subject"),
            "--fixture-dir",
            str(tmp_path / "fixture"),
            "--report-out",
            str(tmp_path / "report"),
            "--materialize-only",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "Missing training dependencies" not in completed.stderr
    assert invocation_log.read_text(encoding="utf-8").startswith("-")


def _call_shell_function(function_call: str) -> subprocess.CompletedProcess[str]:
    script = f"source {shlex.quote(str(HELPER))}\n{function_call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _prepare(path: Path, *, force: bool) -> subprocess.CompletedProcess[str]:
    return _call_shell_function(
        "integration_prepare_training_output "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(path))} {int(force)}"
    )


def _mark(path: Path) -> subprocess.CompletedProcess[str]:
    return _call_shell_function(
        "integration_mark_training_output "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(path))}"
    )


@pytest.mark.parametrize(
    "protected",
    [Path("/"), Path.home(), REPO_ROOT, REPO_ROOT.parent],
)
def test_force_rejects_protected_training_output_paths(protected: Path) -> None:
    result = _prepare(protected, force=True)

    assert result.returncode != 0
    assert "Refusing protected subject output path" in result.stderr


def test_force_rejects_unowned_directory_and_symlink(tmp_path: Path) -> None:
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("keep\n", encoding="utf-8")

    result = _prepare(unowned, force=True)

    assert result.returncode != 0
    assert "Refusing to replace unowned subject output" in result.stderr
    assert (unowned / "keep.txt").is_file()

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    result = _prepare(linked, force=True)

    assert result.returncode != 0
    assert "Refusing symlink subject output path" in result.stderr
    assert linked.is_symlink()
    assert target.is_dir()


def test_owned_training_output_can_be_replaced_but_inode_reuse_cannot(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "model.safetensors").write_bytes(b"model")
    assert _mark(subject).returncode == 0

    replacement = _prepare(subject, force=True)

    assert replacement.returncode == 0
    assert not subject.exists()
    assert not list(tmp_path.glob(".subject.invarlock-training-output.json"))

    subject.mkdir()
    assert _mark(subject).returncode == 0
    subject.rmdir()
    subject.mkdir()
    (subject / "keep.txt").write_text("new inode\n", encoding="utf-8")

    mismatch = _prepare(subject, force=True)

    assert mismatch.returncode != 0
    assert "ownership-marker mismatch" in mismatch.stderr
    assert (subject / "keep.txt").is_file()


def test_fresh_training_output_is_allowed_but_stale_marker_is_preserved(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "subject"
    fresh = _prepare(subject, force=False)

    assert fresh.returncode == 0
    assert not subject.exists()

    marker = tmp_path / ".subject.invarlock-training-output.json"
    marker.write_text("stale\n", encoding="utf-8")

    result = _prepare(subject, force=False)

    assert result.returncode != 0
    assert "unexpected ownership-marker path" in result.stderr
    assert marker.read_text(encoding="utf-8") == "stale\n"
    assert not subject.exists()


def _write_fake_python(path: Path) -> Path:
    log_path = path / "verify-invocations.jsonl"
    executable = path / "python-dispatch"
    executable.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

if len(sys.argv) > 1 and Path(sys.argv[1]).name == "create_edit_model.py":
    with Path({str(log_path)!r}).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sys.argv[2:]) + "\\n")
    print(json.dumps({{"source": "fake-profile-verifier", "status": "verified"}}))
    raise SystemExit(int(os.environ.get("FAKE_VERIFY_STATUS", "0")))
os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _write_bound_receipts(subject: Path, report: Path) -> tuple[Path, Path]:
    subject.mkdir()
    (subject / "config.json").write_text('{"model_type":"gpt2"}\n', encoding="utf-8")
    receipt = with_receipt_digest(
        {"hashes": {"subject_tree_sha256": directory_sha256(subject)}}
    )
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    subject_receipt = subject / "training_receipt.json"
    subject_receipt.write_bytes(receipt_bytes)
    report.mkdir()
    copied_receipt = report / "training_receipt.json"
    copied_receipt.write_bytes(receipt_bytes)
    return subject_receipt, copied_receipt


def _verify_binding(
    fake_python: Path,
    subject: Path,
    copied_receipt: Path,
    *,
    fake_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    call = (
        'integration_run_source_archive_clean() { "$@"; }\n'
        "integration_verify_training_binding "
        f"{shlex.quote(str(fake_python))} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(REPO_ROOT / 'profiles.json'))} test-profile "
        f"{shlex.quote(str(subject))} {shlex.quote(str(copied_receipt))} 0"
    )
    env = os.environ.copy()
    env["FAKE_VERIFY_STATUS"] = str(fake_status)
    script = f"source {shlex.quote(str(HELPER))}\n{call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _finalize_binding(
    fake_python: Path,
    subject: Path,
    copied_receipt: Path,
    *,
    fake_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    report_out = copied_receipt.parent
    call = (
        'integration_run_source_archive_clean() { "$@"; }\n'
        "integration_finalize_training_binding "
        f"{shlex.quote(str(fake_python))} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(REPO_ROOT / 'profiles.json'))} test-profile "
        f"{shlex.quote(str(subject))} {shlex.quote(str(copied_receipt))} 0 "
        f"{shlex.quote(str(report_out))}"
    )
    env = os.environ.copy()
    env["FAKE_VERIFY_STATUS"] = str(fake_status)
    script = f"source {shlex.quote(str(HELPER))}\n{call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_finalized_binding(report_out: Path) -> subprocess.CompletedProcess[str]:
    call = (
        "integration_require_finalized_training_binding "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(report_out))}"
    )
    script = f"source {shlex.quote(str(HELPER))}\n{call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _stage_training_evidence_without_producer(
    *, report_out: Path, subject_dir: Path
) -> subprocess.CompletedProcess[str]:
    call = (
        'integration_run_source_archive_clean() { echo "proof producer invoked" >&2; return 91; }\n'
        "integration_stage_training_evidence "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(REPO_ROOT))} "
        f"{shlex.quote(str(REPO_ROOT / 'profiles.json'))} test-profile "
        f"{shlex.quote(str(subject_dir))} {shlex.quote(str(report_out))} 0 all"
    )
    script = f"source {shlex.quote(str(HELPER))}\n{call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_post_evaluation_binding_rechecks_profile_receipt_and_subject_tree(
    tmp_path: Path,
) -> None:
    fake_python = _write_fake_python(tmp_path)
    subject_receipt, copied_receipt = _write_bound_receipts(
        tmp_path / "subject", tmp_path / "report"
    )
    assert _mark(subject_receipt.parent).returncode == 0

    result = _verify_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert result.returncode == 0, result.stderr
    binding = json.loads(result.stdout)
    assert binding["schema"] == "invarlock.integration_training_binding.v1"
    assert binding["verified"] is True
    assert result.stdout.count("\n") == 1
    assert '"source": "fake-profile-verifier"' in result.stderr
    invocations = [
        json.loads(line)
        for line in (tmp_path / "verify-invocations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert invocations == [
        [
            "verify-training-profile",
            "test-profile",
            str(subject_receipt.parent),
            "--profiles-path",
            str(REPO_ROOT / "profiles.json"),
            "--repo-root",
            str(REPO_ROOT),
        ]
    ]

    (subject_receipt.parent / "config.json").write_text(
        '{"model_type":"tampered"}\n', encoding="utf-8"
    )
    tampered_subject = _verify_binding(
        fake_python, subject_receipt.parent, copied_receipt
    )
    assert tampered_subject.returncode != 0
    assert "Subject artifact tree no longer matches" in tampered_subject.stderr


def test_post_evaluation_binding_rejects_receipt_copy_drift_and_verify_failure(
    tmp_path: Path,
) -> None:
    fake_python = _write_fake_python(tmp_path)
    subject_receipt, copied_receipt = _write_bound_receipts(
        tmp_path / "subject", tmp_path / "report"
    )
    assert _mark(subject_receipt.parent).returncode == 0
    copied_receipt.write_bytes(subject_receipt.read_bytes() + b" ")

    drift = _verify_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert drift.returncode != 0
    assert "not byte-identical" in drift.stderr

    copied_receipt.write_bytes(subject_receipt.read_bytes())
    failed_verify = _verify_binding(
        fake_python, subject_receipt.parent, copied_receipt, fake_status=9
    )
    assert failed_verify.returncode == 9
    assert '"verified": true' not in failed_verify.stdout


def test_post_evaluation_binding_rejects_byte_identical_subject_replacement(
    tmp_path: Path,
) -> None:
    fake_python = _write_fake_python(tmp_path)
    subject_receipt, copied_receipt = _write_bound_receipts(
        tmp_path / "subject", tmp_path / "report"
    )
    assert _mark(subject_receipt.parent).returncode == 0
    original = tmp_path / "original-subject"
    subject_receipt.parent.rename(original)
    shutil.copytree(original, subject_receipt.parent)

    replaced = _verify_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert replaced.returncode != 0
    assert "Training subject identity changed" in replaced.stderr


def test_binding_finalization_invalidates_prior_success_on_failure(
    tmp_path: Path,
) -> None:
    fake_python = _write_fake_python(tmp_path)
    subject_receipt, copied_receipt = _write_bound_receipts(
        tmp_path / "subject", tmp_path / "report"
    )
    assert _mark(subject_receipt.parent).returncode == 0
    summary = copied_receipt.parent / "run_summary.txt"
    summary.write_text(
        "status: success\nlane_artifact_label: cuda-container-strict\n",
        encoding="utf-8",
    )
    (copied_receipt.parent / "evaluation.report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (copied_receipt.parent / "verify.json").write_text("{}\n", encoding="utf-8")

    completed = _finalize_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert completed.returncode == 0, completed.stderr
    binding = json.loads(
        (copied_receipt.parent / "training_binding.json").read_text(encoding="utf-8")
    )
    assert binding["schema"] == "invarlock.integration_training_binding.v1"
    assert binding["verified"] is True
    assert binding["evaluation_report_sha256"]
    assert binding["verify_artifact_sha256"]
    assert "training_binding_status: verified" in summary.read_text(encoding="utf-8")
    assert _require_finalized_binding(copied_receipt.parent).returncode == 0

    (subject_receipt.parent / "config.json").write_text(
        '{"model_type":"tampered"}\n', encoding="utf-8"
    )
    failed = _finalize_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert failed.returncode != 0
    assert not (copied_receipt.parent / "training_binding.json").exists()
    failed_summary = summary.read_text(encoding="utf-8")
    assert "status: failed" in failed_summary
    assert "status: success" not in failed_summary
    assert "training_binding_status: failed" in failed_summary


def test_training_evidence_staging_rejects_missing_finalized_binding_before_producer(
    tmp_path: Path,
) -> None:
    report_out = tmp_path / "report"
    report_out.mkdir()
    (report_out / "training_receipt.json").write_text("{}\n", encoding="utf-8")
    (report_out / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    (report_out / "verify.json").write_text("{}\n", encoding="utf-8")

    result = _stage_training_evidence_without_producer(
        report_out=report_out,
        subject_dir=tmp_path / "subject",
    )

    assert result.returncode != 0
    assert "requires a finalized post-evaluation binding" in result.stderr
    assert "proof producer invoked" not in result.stderr
    assert not (report_out / "training_evidence_proof.json").exists()
    assert not (report_out / "training_profile_snapshot.json").exists()
    assert "status: failed" in (report_out / "run_summary.txt").read_text(
        encoding="utf-8"
    )


def test_binding_augmentation_failure_atomically_invalidates_success(
    tmp_path: Path,
) -> None:
    fake_python = _write_fake_python(tmp_path)
    subject_receipt, copied_receipt = _write_bound_receipts(
        tmp_path / "subject", tmp_path / "report"
    )
    assert _mark(subject_receipt.parent).returncode == 0
    report_out = copied_receipt.parent
    summary = report_out / "run_summary.txt"
    summary.write_text(
        "status: success\nlane_artifact_label: cuda-container-strict\n",
        encoding="utf-8",
    )
    (report_out / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    assert not (report_out / "verify.json").exists()

    failed = _finalize_binding(fake_python, subject_receipt.parent, copied_receipt)

    assert failed.returncode != 0
    assert not (report_out / "training_binding.json").exists()
    failed_summary = summary.read_text(encoding="utf-8")
    assert failed_summary.startswith("status: failed\n")
    assert "status: success" not in failed_summary
    assert "training_binding_status: failed" in failed_summary


def test_real_training_runners_require_post_evaluation_binding_check() -> None:
    helper_text = HELPER.read_text(encoding="utf-8")
    staging_body = helper_text.split("integration_stage_training_evidence()", 1)[1]
    assert staging_body.count("integration_require_finalized_training_binding") == 2
    for runner in RUNNERS:
        subprocess.run(["bash", "-n", str(runner)], check=True)
        text = runner.read_text(encoding="utf-8")
        assert text.index('"${compare_cmd[@]}"') < text.index(
            "integration_finalize_training_binding"
        )
        assert text.index("integration_finalize_training_binding") < text.index(
            "integration_stage_training_evidence"
        )
        prepare_call = text[text.index("integration_prepare_training_output") :]
        assert '"$PYTHON_BIN" "$REPO_ROOT"' in prepare_call.splitlines()[1]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("runner", "profile_id"),
    [
        (RUNNERS[0], "tiny_gpt2_lora_v1"),
        (RUNNERS[1], "tiny_gpt2_full_ft_v1"),
    ],
)
def test_training_integration_runner_executes_verified_profile(
    runner: Path, profile_id: str, tmp_path: Path
) -> None:
    if os.environ.get("INVARLOCK_REQUIRE_REAL_TRAINING") != "1":
        pytest.skip("black-box training runners execute in the required CI lane")

    subject_dir = tmp_path / "subject"
    completed = subprocess.run(
        [
            str(runner),
            "--lane",
            "host",
            "--device",
            "cpu",
            "--training-profile",
            profile_id,
            "--subject-dir",
            str(subject_dir),
            "--fixture-dir",
            str(tmp_path / "fixture"),
            "--report-out",
            str(tmp_path / "report"),
            "--materialize-only",
            "--allow-network",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    receipt = json.loads(
        (subject_dir / "training_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["profile_id"] == profile_id
    assert (tmp_path / "fixture" / "fixture_summary.json").is_file()
    assert 'status": "verified' in completed.stderr
