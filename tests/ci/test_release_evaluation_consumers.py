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
    invocation = 'python captured-wheel-smoke.py --cli "${evaluation_cli}"'
    assert invocation in script
    assert script.index(core_install) < script.index(invocation)
    assert script.index(invocation) < script.index(addin_install)
    assert 'evaluation_cli="$(command -v invarlock)"' in script
    assert (
        'cp examples/captured-results/wheel_smoke.py "${quickstart_root}/captured-wheel-smoke.py"'
        in script
    )
    assert 'cd "${quickstart_root}"' in script[: script.index(invocation)]
    assert "unset PYTHONPATH" in script[: script.index(invocation)]
    assert "export PYTHONSAFEPATH=1" in script[: script.index(invocation)]
