from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands import verify as verify_mod
from invarlock.cli.commands.verify import verify_command


def _write(p: Path, payload: dict) -> Path:
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_verify_ci_provider_parity_pass(tmp_path: Path, capsys) -> None:
    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    spectral_hash = verify_mod._measurement_contract_digest(spectral_contract)
    rmt_hash = verify_mod._measurement_contract_digest(rmt_contract)
    cert = {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            },
        },
        "provenance": {
            "provider_digest": {
                "ids_sha256": "ID",
                "tokenizer_sha256": "TOK",
                "masking_sha256": "MASK",
            }
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": spectral_hash,
            "measurement_contract_match": True,
        },
        "rmt": {
            "evaluated": True,
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": rmt_hash,
            "measurement_contract_match": True,
        },
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }
    baseline = {
        "provenance": {
            "provider_digest": {
                "ids_sha256": "ID",
                "tokenizer_sha256": "TOK",
                "masking_sha256": "MASK",
            }
        }
    }
    cp = _write(tmp_path / "c.json", cert)
    bp = _write(tmp_path / "b.json", baseline)
    with pytest.raises(typer.Exit) as ei:
        verify_command([cp], baseline=bp, profile="ci", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0
