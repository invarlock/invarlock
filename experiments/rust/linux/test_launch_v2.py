"""Explicit provenance for a module absent from the original baseline bundle."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import launch_v2


def test_later_module_must_match_exact_prior_commit(tmp_path, monkeypatch):
    name = "optimization_runner.py"
    path = tmp_path / "experiments/rust/python/invarlock_rust_experiment" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen prior bytes")
    calls = []
    monkeypatch.setattr(
        launch_v2.subprocess,
        "check_output",
        lambda command: calls.append(command) or b"frozen prior bytes",
    )
    revision = launch_v2.harness_revision(tmp_path, name, None)
    assert revision["prior_sha256"] is None
    assert (
        revision["source_reference"]["commit"]
        == "e3caec51941990a2d620c00984c8b08e662e8b76"
    )
    assert calls[0][-1] == launch_v2.PRIOR_MEASURED_COMMIT + ":" + str(
        path.relative_to(tmp_path)
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from frozen prior revision"):
        launch_v2.harness_revision(tmp_path, name, None)


@pytest.mark.parametrize("name", ["worker.py", "optimization_runner.py"])
def test_existing_historical_binding_cannot_be_overridden(name, tmp_path):
    path = tmp_path / "experiments/rust/python/invarlock_rust_experiment" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="historical harness source changed"):
        launch_v2.harness_revision(tmp_path, name, "prior identity")
