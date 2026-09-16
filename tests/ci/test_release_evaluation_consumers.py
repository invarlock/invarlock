"""Keep the public evaluation consumers in both release artifact gates."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("job", "step_name", "core_install", "addin_install"),
    [
        (
            "build_check",
            "Install smoke from wheel",
            "dist/*.whl",
            "dist/addins/*.whl",
        ),
        (
            "published_install_smoke",
            "Install published wheels and smoke test",
            "wheelhouse/invarlock-*.whl",
            "wheelhouse/invarlock_*.whl",
        ),
    ],
)
def test_release_capture_runs_from_installed_core_before_addins(
    job, step_name, core_install, addin_install
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    step = next(
        step for step in workflow["jobs"][job]["steps"] if step.get("name") == step_name
    )
    script = step["run"]
    invocation = (
        'python scripts/release/core_wheel_consumers.py --cli "$(command -v invarlock)"'
    )
    assert invocation in script
    assert script.index(core_install) < script.index(invocation)
    assert script.index(invocation) < script.index(addin_install)
    assert "unset PYTHONPATH" in script[: script.index(invocation)]
    assert "export PYTHONSAFEPATH=1" in script[: script.index(invocation)]


def test_local_install_uses_shared_core_consumers_before_addins():
    script = (
        (ROOT / "Makefile").read_text().split("addins-install-smoke: dist-check", 1)[1]
    )
    invocation = '"$$smoke_venv/bin/python" scripts/release/core_wheel_consumers.py'
    assert script.index("dist/*.whl") < script.index(invocation)
    assert script.index(invocation) < script.index("dist/addins/*.whl")
    assert '--cli "$$smoke_venv/bin/invarlock"' in script
