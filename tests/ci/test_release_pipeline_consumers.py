"""Keep the public pipeline consumer in both release artifact gates."""

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
def test_release_pipeline_runs_from_installed_core_before_addins(
    job, step_name, core_install, addin_install
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    step = next(
        step for step in workflow["jobs"][job]["steps"] if step.get("name") == step_name
    )
    script = step["run"]
    invocation = 'python pipeline-wheel-smoke.py --cli "${pipeline_cli}"'
    assert invocation in script
    assert script.index(core_install) < script.index(invocation)
    assert script.index(invocation) < script.index(addin_install)
    assert 'pipeline_cli="$(command -v invarlock-pipeline)"' in script
    assert (
        'cp examples/pipeline/wheel_smoke.py "${quickstart_root}/pipeline-wheel-smoke.py"'
        in script
    )
    assert 'cd "${quickstart_root}"' in script[: script.index(invocation)]
    assert "unset PYTHONPATH" in script[: script.index(invocation)]
    assert "export PYTHONSAFEPATH=1" in script[: script.index(invocation)]
