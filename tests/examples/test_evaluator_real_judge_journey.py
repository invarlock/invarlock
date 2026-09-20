"""Real retained ratings exercise the captured judge scorer without new calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/integrations/evaluator-parity/real_judge.py"
SPEC = importlib.util.spec_from_file_location("real_judge_journey_test", SCRIPT)
REAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REAL)


@pytest.fixture(scope="module")
def pilot():
    return REAL.retained("pilot")


@pytest.mark.parametrize("reference", ["heldout", "pilot"])
def test_complete_real_campaign_replays_and_preserves_original_outcome(
    tmp_path, reference, monkeypatch
):
    # Coverage runs the helper in-process; every command still uses the actual
    # CLI in an isolated subprocess with network blocked. The installed gate
    # separately enforces a clean core-only wheel interpreter via main().
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")
    original_run = subprocess.run

    def offline_run(*args, **kwargs):
        assert "INVARLOCK_ALLOW_NETWORK" not in kwargs["env"]
        return original_run(*args, **kwargs)

    monkeypatch.setattr(REAL.subprocess, "run", offline_run)
    result = REAL.journey(tmp_path, reference, sys.executable)
    assert result["workflow"] == "grounded_qa"
    assert result["cases"] == REAL.REFERENCES[reference]["cases"]
    assert result["trials"] == result["cases"] * 6
    assert result["new_calls"] == 0 and result["altered_measurement_rejected"]
    assert result["accepted"] == (reference == "heldout")
    assert result["decision"] == REAL.REFERENCES[reference]["decision"]
    origin = json.loads((tmp_path / "origin.json").read_bytes())
    assert origin["original_receipt_authenticated"]
    assert [row["exit_code"] for row in result["commands"]] == (
        [0, 0, 0, 0, 4] if reference == "heldout" else [0, 7, 7, 0, 4]
    )
    assert (tmp_path / "retained/evidence/measurements.json").read_bytes()
    assert (
        json.loads((tmp_path / "altered-measurement-refusal.json").read_bytes())[
            "verified"
        ]
        is False
    )


@pytest.mark.parametrize("field", ["cases", "trials", "decision"])
def test_independently_pinned_inventory_cannot_be_reinterpreted(monkeypatch, field):
    selected = dict(REAL.REFERENCES["pilot"])
    selected[field] = "pass" if field == "decision" else 1
    monkeypatch.setitem(REAL.REFERENCES, "pilot", selected)
    with pytest.raises(ValueError, match="independent inventory"):
        REAL.retained("pilot")


@pytest.mark.parametrize("document", ["plan", "analysis_policy"])
def test_recipe_rejects_changed_historical_bindings(pilot, document):
    documents = copy.deepcopy(pilot[1])
    key = "baseline_run_sha256" if document == "plan" else "plan_sha256"
    documents[document][key] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="recipe reconstruction"):
        REAL.recipe(documents)


@pytest.mark.parametrize("fault", ["authentication", "decision", "acceptance"])
def test_original_receipt_refusal_precedes_fresh_publication(
    tmp_path, monkeypatch, pilot, fault
):
    import invarlock.judge_measurements.acceptance as acceptance

    monkeypatch.setattr(REAL, "retained", lambda reference: pilot)
    result = SimpleNamespace(
        authenticated=fault != "authentication",
        verified=True,
        replayed=True,
        decision="pass" if fault == "decision" else "insufficient_evidence",
        accepted=fault == "acceptance",
    )
    monkeypatch.setattr(
        acceptance, "replay_signed_judge_verification_receipt", lambda *a, **k: result
    )
    with pytest.raises(ValueError, match="original retained judge"):
        REAL.prepare(tmp_path, "pilot")
    assert not (tmp_path / "signer.pem").exists()
    assert not (tmp_path / "trust.json").exists()


def test_command_failure_is_not_a_successful_journey(tmp_path, monkeypatch, pilot):
    monkeypatch.setattr(REAL, "prepare", lambda *a: (pilot[1], {}))
    monkeypatch.setattr(
        REAL.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 9, stdout="", stderr="deliberate command failure"
        ),
    )
    with pytest.raises(RuntimeError, match="deliberate command failure"):
        REAL.journey(tmp_path, "pilot", sys.executable)


@pytest.mark.parametrize("reference", ["both", "pilot"])
def test_cli_checks_recipient_and_selects_complete_campaigns(
    tmp_path, monkeypatch, capsys, reference
):
    identity = {"sdk_modules_absent": ["sdk"], "invarlock": "installed-wheel"}
    monkeypatch.setattr(REAL.PARITY, "installed_identity", lambda python: identity)
    calls = []

    def capture(output, selected, python):
        calls.append((output.name, selected, python))
        return {"reference": selected}

    monkeypatch.setattr(REAL, "journey", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--recipient-python",
            sys.executable,
            "--output",
            str(tmp_path / "new"),
            "--reference",
            reference,
        ],
    )
    REAL.main()
    result = json.loads(capsys.readouterr().out)
    expected = ["heldout", "pilot"] if reference == "both" else ["pilot"]
    assert [call[1] for call in calls] == expected
    assert result["recipient"] == identity
    assert result["references"] == [{"reference": value} for value in expected]
    with pytest.raises(FileExistsError):
        REAL.main()
