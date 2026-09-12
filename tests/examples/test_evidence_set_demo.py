from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.evidence_sets.verification import verify_evidence_set

SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/judge-with-deterministic/demo.py"
)


def module():
    spec = importlib.util.spec_from_file_location("evidence_set_demo", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_documented_demo_preserves_an_inconclusive_result(
    tmp_path, monkeypatch, capsys
):
    demo = module()
    output = tmp_path / "demo"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--output", str(output)])
    with pytest.raises(SystemExit) as completed:
        demo.main()
    assert completed.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] and not result["accepted"]
    assert result["decision"] == "insufficient_evidence"
    assert (output / "verification.json").is_file()
    replay = verify_evidence_set(
        output / "evidence", recipient_policy=output / "recipient/composition.json"
    )
    assert replay.payload["verified"] and replay.exit_code == 7
    with pytest.raises(FileExistsError):
        demo.build(output)
