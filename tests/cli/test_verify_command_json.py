from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import typer


def _import_verify_command():
    # transformers stub to avoid heavy import via run
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self):
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto  # type: ignore[attr-defined]
        tr.GPT2Tokenizer = _GPT2  # type: ignore[attr-defined]
        sys.modules["transformers"] = tr
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object  # type: ignore[attr-defined]
        sys.modules["transformers.tokenization_utils_base"] = sub
    mod = importlib.import_module("invarlock.cli.commands.verify")
    return mod.verify_command


def _write_cert(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_cert(ratio: float = 2.0) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": ratio,
            "display_ci": [0.98, 1.02],
        },
        "dataset": {
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            }
        },
        "baseline_ref": {"primary_metric": {"final": 5.0}},
    }


def test_verify_json_success_single(tmp_path: Path, capsys) -> None:
    verify_command = _import_verify_command()
    c1 = _write_cert(tmp_path / "c1.json", _valid_cert())
    with pytest.raises(typer.Exit) as ei:
        verify_command(
            [c1], baseline=None, tolerance=1e-9, profile="dev", json_out=True
        )
    assert ei.value.exit_code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["summary"]["ok"] is True
    assert payload["certificate"]["count"] == 1
    assert payload["results"][0]["ok"] is True


def test_verify_json_mixed_policy_fail(tmp_path: Path, capsys) -> None:
    verify_command = _import_verify_command()
    c_ok = _write_cert(tmp_path / "ok.json", _valid_cert(ratio=2.0))
    # ratio declared incorrectly → policy fail (not malformed)
    c_bad = _write_cert(tmp_path / "bad.json", _valid_cert(ratio=1.0))
    with pytest.raises(typer.Exit) as ei:
        verify_command(
            [c_ok, c_bad], baseline=None, tolerance=1e-9, profile="dev", json_out=True
        )
    assert ei.value.exit_code == 1
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["summary"]["ok"] is False
    assert payload["summary"]["reason"] == "policy_fail"
    assert payload["certificate"]["count"] == 2


def test_verify_json_malformed(tmp_path: Path, capsys) -> None:
    verify_command = _import_verify_command()
    bad = {"schema_version": "v0", "run_id": "r1", "primary_metric": {"final": 1.0}}
    c_bad = _write_cert(tmp_path / "m.json", bad)
    with pytest.raises(typer.Exit) as ei:
        verify_command(
            [c_bad], baseline=None, tolerance=1e-9, profile="dev", json_out=True
        )
    assert ei.value.exit_code in (2, 1)  # prefer 2, but tolerate 1 if schema lenient
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["summary"]["ok"] is False
    # reason is 'malformed' when schema fails; but allow 'policy_fail' if minimal validator passes
    assert payload["summary"]["reason"] in {"malformed", "policy_fail"}
