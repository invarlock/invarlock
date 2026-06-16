from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "public_e2e"
RUNNER = EXAMPLE_DIR / "run_public_e2e_release_review.sh"
README = EXAMPLE_DIR / "README.md"


def test_public_e2e_runner_smoke(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    output_dir = tmp_path / "public-e2e"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            str(RUNNER),
            "--output-dir",
            str(output_dir),
            "--report-url",
            "https://example.test/evaluation.report.json",
            "--evidence-url",
            "https://example.test/evidence-pack",
            "--force",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "status: success" in result.stdout
    expected_files = [
        "evaluation.report.json",
        "runtime.manifest.json",
        "checkpoint_refs.json",
        "external_edit_summary.json",
        "invarlock-verify.json",
        "evaluation.html",
        "mlflow-tags.json",
        "model-card-invarlock.md",
        "release-review.md",
        "ci-summary.md",
        "run_command.txt",
        "source_run_command.txt",
        "run_summary.txt",
    ]
    for name in expected_files:
        assert (output_dir / name).is_file(), f"missing generated artifact: {name}"

    verify_payload = json.loads(
        (output_dir / "invarlock-verify.json").read_text(encoding="utf-8")
    )
    assert verify_payload["summary"]["ok"] is True
    assert verify_payload["results"][0]["id"] == str(
        output_dir / "evaluation.report.json"
    )

    tags = json.loads((output_dir / "mlflow-tags.json").read_text(encoding="utf-8"))[
        "tags"
    ]
    assert tags["invarlock.status"] == "pass"
    assert tags["invarlock.verifier_status"] == "pass"
    assert tags["invarlock.policy_profile"] == "release"

    html = (output_dir / "evaluation.html").read_text(encoding="utf-8")
    assert "<html" in html.lower()
    model_card = (output_dir / "model-card-invarlock.md").read_text(encoding="utf-8")
    assert "## InvarLock Evidence" in model_card
    assert "https://example.test/evaluation.report.json" in model_card
    assert "https://example.test/evidence-pack" in model_card
    release_review = (output_dir / "release-review.md").read_text(encoding="utf-8")
    assert "# InvarLock Release Review" in release_review
    assert "- Status: **PASS**" in release_review
    ci_summary = (output_dir / "ci-summary.md").read_text(encoding="utf-8")
    assert ci_summary.startswith("### InvarLock\n\n# InvarLock Release Review")
    run_command = (output_dir / "run_command.txt").read_text(encoding="utf-8")
    assert "run_public_e2e_release_review.sh" in run_command
    source_run_command = (output_dir / "source_run_command.txt").read_text(
        encoding="utf-8"
    )
    assert "invarlock evaluate" in source_run_command


def test_public_e2e_readme_scopes_claims() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Status: `reference-pattern`" in text
    assert "Script: runnable against checked-in public evidence." in text
    assert "copy/vendor `.github/actions/invarlock-report-gate/`" in text
    assert "does not regenerate the subject" in text
    assert "checkpoint" in text
    assert "push to MLflow" in text
    assert "update Hugging Face Hub" in text
    assert "approve a deployment" in text
    assert "subject checkpoint is not" in text
    assert "vendored in the repository" in text
