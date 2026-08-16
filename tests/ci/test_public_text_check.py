from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "checks" / "check_public_text.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_public_text", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def public_text():
    return _load_module()


@pytest.mark.parametrize(
    ("text", "rule_name"),
    [
        ("Run this on root@192.0.2.10.", "root_ssh_target"),
        ("Logs are under /Users/alice/project/run.log.", "absolute_home_path"),
        ("Use /root/runs/result.json.", "absolute_root_path"),
        ("Copy the .automation/session output.", "desktop_runtime_path"),
        ("Results are in /private/tmp/run-42/output.", "private_temp_path"),
        ("PATH=/private/tools:/usr/bin make verify", "full_path_environment"),
        ("api_key=abcdefghijklmnop", "credential_like_assignment"),
        ("Privacy gate passed for this report.", "review_process_status"),
        (
            "Checked by an automated reviewer before publishing.",
            "review_process_status",
        ),
        ("Adversarial\naudit completed.", "review_process_status"),
        ("Remote validation passed.", "avoidable_remote_validation_claim"),
        ("Added the engineering backlog lane.", "planning_or_workspace_note"),
        ("Added worktree-aware remote launch handling.", "planning_or_workspace_note"),
        ("This is the compact evaluator release focus.", "planning_or_workspace_note"),
        ("The matrix is not a release breadth target.", "planning_or_workspace_note"),
        (
            "The catalog is reviewed rather than quota-driven.",
            "planning_or_workspace_note",
        ),
        ("Reserved for a future claim.", "planning_or_workspace_note"),
        (
            "Increasing to 600 records would add more execution cost.",
            "planning_or_workspace_note",
        ),
    ],
)
def test_detects_nonpublic_operational_language(
    public_text, text: str, rule_name: str
) -> None:
    findings = public_text.findings_for_text("sample.md", text)

    assert rule_name in {finding.rule.name for finding in findings}


def test_accepts_capability_focused_public_language(public_text) -> None:
    text = """\
`make verify` passed.
Strict CUDA/container validation passed on a CUDA-capable host.
Artifacts use repository-relative paths and placeholder digests.
The integration branch receives dependency updates.
"""

    assert public_text.findings_for_text("sample.md", text) == []


def test_tracked_markdown_discovery_includes_every_repository_document(
    public_text,
) -> None:
    expected = {
        ROOT / raw_path.decode("utf-8")
        for raw_path in subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        if raw_path and Path(raw_path.decode("utf-8")).suffix.lower() == ".md"
    }

    discovered = set(public_text.tracked_markdown_paths(ROOT))

    assert discovered == expected
    assert ROOT / "CHANGELOG.md" in discovered


def test_repository_markdown_passes_the_public_text_check(public_text) -> None:
    paths = public_text.tracked_markdown_paths(ROOT)

    assert public_text.scan_paths(paths, ROOT) == []


def test_external_file_findings_do_not_expose_the_absolute_path(
    public_text, tmp_path: Path
) -> None:
    document = tmp_path / "private-location.md"
    document.write_text("Privacy gate passed.\n", encoding="utf-8")

    findings = public_text.scan_paths([document], ROOT)

    assert len(findings) == 1
    assert findings[0].source == "<external-file>"
    assert str(tmp_path) not in findings[0].source


def test_cli_reports_findings_and_fails(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    sensitive_target = "root@example.invalid"
    document.write_text(f"Validated from {sensitive_target}.\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), str(document)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "root_ssh_target" in result.stderr
    assert "Public text check failed: 1 finding(s)." in result.stderr
    assert sensitive_target not in result.stdout
    assert sensitive_target not in result.stderr


def test_main_scans_the_tracked_repository(public_text, capsys) -> None:
    expected_count = len(public_text.tracked_markdown_paths(ROOT))

    result = public_text.main(["--root", str(ROOT)])

    captured = capsys.readouterr()
    assert result == 0
    assert f"Public text check passed for {expected_count} file(s)." in captured.out
    assert captured.err == ""


def test_main_fails_closed_without_a_git_repository(
    public_text, tmp_path: Path, capsys
) -> None:
    result = public_text.main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "CalledProcessError" in captured.err
    assert str(tmp_path) not in captured.err
