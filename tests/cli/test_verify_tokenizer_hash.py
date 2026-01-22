from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app

runner = CliRunner()


def _minimal_cert(token_hash_edited: str, token_hash_baseline: str) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "run-tokenizer-hash-test",
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "abcdef1234567890",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
            "tokenizer_hash": token_hash_edited,
        },
        "dataset": {
            "provider": "wikitext2",
            "split": "validation",
            "seq_len": 128,
            "stride": 128,
            "windows": {
                "preview": 2,
                "final": 2,
                "seed": 42,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "paired_windows": 2,
                    "coverage": {
                        "preview": {"used": 2, "required": 2, "ok": True},
                        "final": {"used": 2, "required": 2, "ok": True},
                    },
                },
            },
            "tokenizer": {"name": "gpt2", "hash": token_hash_edited},
        },
        "baseline_ref": {
            "run_id": "baseline-001",
            "model_id": "gpt2-small",
            "ppl_baseline": 10.0,
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "tokenizer_hash": token_hash_baseline,
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [10.0, 10.0],
        },
        # dataset section defined above
    }


def test_verify_tokenizer_hash_match(tmp_path: Path):
    cert = _minimal_cert("abc123", "abc123")
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    result = runner.invoke(app, ["verify", str(p)])
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_verify_tokenizer_hash_mismatch(tmp_path: Path):
    cert = _minimal_cert("abc123", "deadbeef")
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    result = runner.invoke(app, ["verify", str(p)])
    assert result.exit_code != 0
    assert "Tokenizer hash mismatch" in result.stdout


def test_verify_tokenizer_hash_skip_when_missing(tmp_path: Path):
    cert = _minimal_cert("", "abc123")
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    result = runner.invoke(app, ["verify", str(p)])
    # Missing edited hash → parity check skipped, should still succeed given other fields are fine
    assert result.exit_code == 0
