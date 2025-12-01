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


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ci_missing_provider_digest_yields_policy_fail(tmp_path: Path, capsys) -> None:
    verify_command = _import_verify_command()
    cert = {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 2.0,
            "ratio_vs_baseline": 2.0,
            "display_ci": [1.0, 1.0],
        },
        # provenance missing provider_digest
        "provenance": {},
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
        "baseline_ref": {"primary_metric": {"final": 1.0}},
    }
    c = _write(tmp_path / "c.json", cert)
    with pytest.raises(typer.Exit) as ei:
        verify_command([c], baseline=None, tolerance=1e-9, profile="ci", json_out=True)
    assert ei.value.exit_code == 3  # InvarlockError → 3 in CI/Release
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert (
        payload["summary"]["ok"] is False
        and payload["summary"]["reason"] == "policy_fail"
    )


def test_ci_tokenizer_mismatch_policy_fail(tmp_path: Path, capsys) -> None:
    verify_command = _import_verify_command()
    cert = {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 2.0,
            "ratio_vs_baseline": 2.0,
            "display_ci": [1.0, 1.0],
        },
        "provenance": {
            "provider_digest": {"tokenizer_sha256": "AAA", "ids_sha256": "id"}
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
        "baseline_ref": {"primary_metric": {"final": 1.0}},
    }
    baseline = {
        "provenance": {
            "provider_digest": {"tokenizer_sha256": "BBB", "ids_sha256": "id"}
        }
    }
    c = _write(tmp_path / "c.json", cert)
    b = _write(tmp_path / "b.json", baseline)
    with pytest.raises(typer.Exit) as ei:
        verify_command([c], baseline=b, tolerance=1e-9, profile="ci", json_out=True)
    assert ei.value.exit_code == 3
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert (
        payload["summary"]["ok"] is False
        and payload["summary"]["reason"] == "policy_fail"
    )
