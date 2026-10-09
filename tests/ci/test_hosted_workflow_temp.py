"""Keep hosted Ubuntu build and test files off the bounded /tmp RAM disk."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.stem)
def test_hosted_ubuntu_jobs_use_runner_temporary_storage(path: Path) -> None:
    workflow = yaml.safe_load(path.read_text())
    for name, job in workflow["jobs"].items():
        if "ubuntu-" not in str(job.get("runs-on", "")):
            continue
        if path.stem == "scorecards":
            # Publishing mode permits only the checkout and scorecard actions.
            assert len(job["steps"]) == 2
            assert all(
                step.get("env", {}).get("TMPDIR") == "${{ runner.temp }}"
                for step in job["steps"]
            )
        else:
            setup = job["steps"][0]
            assert "if" not in setup
            assert setup["name"] == "Use disk-backed temporary storage", name
            assert (
                setup["run"]
                == 'printf \'TMPDIR=%s\\n\' "$RUNNER_TEMP" >> "$GITHUB_ENV"'
            )


def test_temporary_storage_setup_exports_a_usable_path(tmp_path: Path) -> None:
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    setup = workflow["jobs"]["verify-full"]["steps"][0]
    runner_temp = tmp_path / "runner disk"
    runner_temp.mkdir()
    environment_file = tmp_path / "environment"
    subprocess.run(
        ["bash", "-e", "-c", setup["run"]],
        env={
            "PATH": os.defpath,
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(environment_file),
        },
        check=True,
    )
    line = environment_file.read_text()
    assert line.splitlines() == [f"TMPDIR={runner_temp}"]
    name, value = line.strip().split("=", 1)
    observed = subprocess.check_output(
        [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
        env={"PATH": os.defpath, name: value},
        text=True,
    )
    assert Path(observed.strip()).resolve() == runner_temp.resolve()
