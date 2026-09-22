"""Keep the public evaluation consumers in both release artifact gates."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("job", "step_name", "core_install"),
    [
        (
            "build_check",
            "Install smoke from wheel",
            "dist/*.whl",
        ),
        (
            "published_install_smoke",
            "Install published wheels and smoke test",
            "wheelhouse/invarlock-*.whl",
        ),
    ],
)
def test_release_consumers_run_from_the_installed_core_wheel(
    job, step_name, core_install
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
    assert "unset PYTHONPATH" in script[: script.index(invocation)]
    assert "export PYTHONSAFEPATH=1" in script[: script.index(invocation)]
