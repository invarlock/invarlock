import os
import shutil
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
    assert "Evaluation Matrix" in text
    # Basic sanity: contains at least one evaluate command
    assert "evaluate" in text and "--adapter hf_causal" in text


def _read_profile_from_checklist(path: str) -> str:
    txt = Path(path).read_text()
    for line in txt.splitlines():
        if "evaluate" in line and "--profile" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "--profile" and i + 1 < len(parts):
                    return parts[i + 1]
    return ""


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


def test_checklist_no_longer_advertises_distilbert_classification(tmp_path: Path):
    env = os.environ.copy()
    env["RUN"] = "0"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    checklist = Path(env["TMP_DIR"]) / "checklist.md"
    text = checklist.read_text(encoding="utf-8")
    assert "DistilBERT" not in text
    assert "classification" not in text.lower()


def test_net_bootstrap_prefers_cpu_torch_before_hf_extra() -> None:
    text = Path("scripts/run_tiny_all_matrix.sh").read_text(encoding="utf-8")
    cpu_torch_install = 'os.environ["TORCH_CPU_INDEX_URL"]'
    hf_install = '"invarlock[hf]"'

    assert "TORCH_CPU_INDEX_URL" in text
    assert "export TORCH_CPU_INDEX_URL" in text
    assert cpu_torch_install in text
    assert hf_install in text
    assert text.index(cpu_torch_install) < text.index(hf_install)


def test_matrix_prefers_local_runtime_image_when_available() -> None:
    text = Path("scripts/run_tiny_all_matrix.sh").read_text(encoding="utf-8")

    assert 'docker image inspect invarlock-runtime:local' in text
    assert 'export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"' in text


def test_run_mode_falls_back_to_python_module_when_console_script_missing(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_calls.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'echo "$@" >> {log_path}',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["RUN"] = "1"
    env["NET"] = "0"
    env["INVARLOCK_TINY_RELAX"] = "1"
    env["TMP_DIR"] = str(tmp_path / "tmp")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    installed_cli = shutil.which("invarlock")
    if installed_cli:
        cli_dir = str(Path(installed_cli).resolve().parent)
        path_parts = [
            part for part in path_parts if Path(part).resolve() != Path(cli_dir)
        ]
    env["PATH"] = os.pathsep.join([str(bin_dir), *path_parts])

    subprocess.run(
        ["bash", "scripts/run_tiny_all_matrix.sh"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8")
    assert "-m invarlock.cli evaluate" in calls


pytestmark = pytest.mark.integration
