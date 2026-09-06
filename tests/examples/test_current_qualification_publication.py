from __future__ import annotations

import argparse
import errno
import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/evaluator-qualification"


@pytest.fixture(params=["inspect", "batch"])
def qualification(request, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    import runner_support

    kind = request.param
    spec = importlib.util.spec_from_file_location(
        f"qualification_publication_{kind}",
        EXAMPLE / "maintained" / f"qualify_{kind}.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = "0.3.254" if kind == "inspect" else "0.7.21"
    monkeypatch.setattr(runner_support.importlib.metadata, "version", lambda _: version)
    output = tmp_path / "qualified"
    arguments = {
        "cases": EXAMPLE / "cases.json",
        "schedule": EXAMPLE / "schedule.json",
        "output": output,
    }
    if kind == "batch":
        arguments["provider"] = "evidently"

    def install_capture(*, race=False, check_python=False):
        def run(command, **kwargs):
            if check_python:
                assert kwargs["env"]["UV_PYTHON"] == sys.executable
            args = argparse.Namespace(
                **{
                    name.replace("-", "_"): Path(
                        command[command.index("--" + name) + 1]
                    )
                    for name in (
                        "profile",
                        "schedule",
                        "cases",
                        "dependency-lock",
                        "raw-output",
                        "export",
                    )
                }
            )
            runner_support.finish_deterministic(
                args=args,
                entrypoint="publication test",
                scores=[1.0, 0.0],
                details=[{}, {}],
                environment=[],
            )
            if race:
                output.mkdir()

        monkeypatch.setattr(module.subprocess, "run", run)

    return module, arguments, install_capture


def test_current_qualification_does_not_replace_racing_empty_directory(qualification):
    module, arguments, install_capture = qualification
    install_capture(race=True)
    with pytest.raises(OSError) as error:
        module.execute(**arguments)
    assert error.value.errno == errno.EEXIST
    output = arguments["output"]
    assert output.is_dir() and list(output.iterdir()) == []
    assert list(output.parent.iterdir()) == [output]


def test_current_qualification_pins_the_invoking_python(qualification):
    module, arguments, install_capture = qualification
    install_capture(check_python=True)
    assert module.execute(**arguments)["scores"] == [1.0, 0.0]
