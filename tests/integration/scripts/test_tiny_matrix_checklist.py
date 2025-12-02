import os
import subprocess
from pathlib import Path

import pytest


def test_tiny_gpt2_matrix_dry_run(tmp_path: Path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["GPT2_ID"] = "sshleifer/tiny-gpt2"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    # The script should complete without executing any commands and write a checklist
    subprocess.check_call(["bash", "scripts/run_tiny_all_matrix.sh"], env=env)
    checklist = Path(env["TMP_DIR"]) / "checklist.md"
    assert checklist.exists()
    text = checklist.read_text()
    assert "Certification Matrix" in text
    # Basic sanity: contains at least one invarlock certify command
    assert "invarlock certify" in text


def _read_profile_from_checklist(path: str) -> str:
    txt = Path(path).read_text()
    for line in txt.splitlines():
        if "invarlock certify" in line and "--profile" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "--profile" and i + 1 < len(parts):
                    return parts[i + 1]
    return ""


def _has_measured_cls(path: str) -> bool:
    txt = Path(path).read_text()
    return "distilbert_cls_measured" in txt


def test_checklist_uses_dev_profile_when_tiny_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["INVARLOCK_TINY_RELAX"] = "1"
    subprocess_result = subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: dev" in subprocess_result.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    prof = _read_profile_from_checklist(str(checklists[-1]))
    assert prof == "dev"


def test_checklist_defaults_to_ci_when_no_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env.pop("INVARLOCK_TINY_RELAX", None)
    subprocess_result = subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: ci" in subprocess_result.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    prof = _read_profile_from_checklist(str(checklists[-1]))
    assert prof == "ci"


def test_explicit_profile_overrides_relax(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "0"
    env["PROFILE"] = "ci"
    env["INVARLOCK_TINY_RELAX"] = "1"  # should NOT override explicit PROFILE
    res = subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Using profile: ci" in res.stdout
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    assert _read_profile_from_checklist(str(checklists[-1])) == "ci"


def test_measured_cls_included_only_when_requested(monkeypatch, tmp_path):
    # Default: not included
    env = os.environ.copy()
    env["RUN"] = "0"
    env["NET"] = "1"
    env.pop("INCLUDE_MEASURED_CLS", None)
    subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    checklists = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists, "No checklist generated"
    chk = str(checklists[-1])
    assert _has_measured_cls(chk) is False

    # With toggle: included
    env["INCLUDE_MEASURED_CLS"] = "1"
    subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    checklists2 = sorted(Path("tmp").glob("tiny_all_*/checklist.md"))
    assert checklists2, "No checklist generated"
    chk2 = str(checklists2[-1])
    assert _has_measured_cls(chk2) is True


pytestmark = pytest.mark.integration
