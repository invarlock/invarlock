from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands import verify as verify_mod
from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp_path: Path, payload: dict, name: str = "cert.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal_ppl_certificate(
    *, ratio: float = 1.0, final: float = 10.0, baseline_final: float = 10.0
) -> dict:
    # Minimal schema-valid certificate for ppl-like metric
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
    return {
        "schema_version": "v1",
        "run_id": "run-xyz",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 9.5,
            "final": final,
            "ratio_vs_baseline": ratio,
            "display_ci": [ratio, ratio],
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
        "baseline_ref": {
            "run_id": "base-xyz",
            "model_id": "m",
            "primary_metric": {"kind": "ppl_causal", "final": baseline_final},
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "artifacts_extra": {},
    }


def test_verify_json_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = _minimal_ppl_certificate(ratio=1.0, final=10.0, baseline_final=10.0)
    ok["primary_metric"]["preview"] = 10.0  # ensure drift ratio == 1.0
    bad = _minimal_ppl_certificate(ratio=1.2, final=10.0, baseline_final=10.0)

    ok_path = _write_cert(tmp_path, ok, "ok.json")
    bad_path = _write_cert(tmp_path, bad, "bad.json")

    # Success case (JSON mode)
    with pytest.raises(typer.Exit) as ei_ok:
        verify_command(
            [ok_path], baseline=None, tolerance=1e-9, profile="dev", json_out=True
        )
    out_ok = capsys.readouterr().out
    payload_ok = json.loads(out_ok)
    assert payload_ok["summary"]["ok"] is True
    assert payload_ok["resolution"]["exit_code"] == 0
    assert getattr(ei_ok.value, "exit_code", getattr(ei_ok.value, "code", None)) == 0

    # Failure case due to ratio mismatch (JSON mode)
    with pytest.raises(typer.Exit) as ei_bad:
        verify_command(
            [bad_path], baseline=None, tolerance=1e-9, profile="dev", json_out=True
        )
    out_bad = capsys.readouterr().out
    payload_bad = json.loads(out_bad)
    assert payload_bad["summary"]["ok"] is False
    # reason can be policy_fail or malformed, but exit_code must be non-zero
    assert payload_bad["resolution"]["exit_code"] != 0
    assert getattr(ei_bad.value, "exit_code", getattr(ei_bad.value, "code", None)) != 0


def test_verify_recompute_dev_warning_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # PPL-like cert with missing evaluation_windows triggers dev-mode recompute warning path
    cert = _minimal_ppl_certificate()
    cert_path = _write_cert(tmp_path, cert)

    with pytest.raises(typer.Exit) as ei:
        verify_command([cert_path], baseline=None, profile="dev", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["resolution"]["exit_code"] in (0, 1, 2)
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) in (0, 1, 2)


def test_verify_human_success_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = _minimal_ppl_certificate(ratio=1.0, final=10.0, baseline_final=10.0)
    ok["primary_metric"]["preview"] = 10.0
    p = _write_cert(tmp_path, ok)
    # Human mode should not raise; prints a single-line success summary
    verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "VERIFY OK" in out


def test_verify_ci_profile_enforces_provider_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = _minimal_ppl_certificate()
    path = _write_cert(tmp_path, cert)
    # In CI profile, provider_digest is required and should raise with non-zero exit
    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile="ci", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["resolution"]["exit_code"] != 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_verify_json_failure_envelope_multiple(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad1 = _minimal_ppl_certificate(ratio=float("nan"))
    bad2 = _minimal_ppl_certificate(ratio=1.2)
    p1 = _write_cert(tmp_path, bad1, "c1.json")
    p2 = _write_cert(tmp_path, bad2, "c2.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p1, p2], baseline=None, profile="dev", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload.get("results"), list) and len(payload["results"]) == 2
    reasons = {r.get("reason") for r in payload["results"]}
    assert reasons.issubset({"malformed", "policy_fail"}) and reasons
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_verify_json_mixed_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = _minimal_ppl_certificate(ratio=1.0, final=10.0, baseline_final=10.0)
    ok["primary_metric"]["preview"] = 10.0
    bad = _minimal_ppl_certificate(ratio=1.2, final=10.0, baseline_final=10.0)
    p1 = _write_cert(tmp_path, ok, "ok.json")
    p2 = _write_cert(tmp_path, bad, "bad.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p1, p2], baseline=None, profile="dev", json_out=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["ok"] is False
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_verify_ci_tokenizer_mismatch_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Subject certificate includes provider_digest; baseline has different tokenizer hash
    cert = _minimal_ppl_certificate()
    cert.setdefault("provenance", {})["provider_digest"] = {
        "ids_sha256": "ID123",
        "tokenizer_sha256": "TOK-B",
    }
    cert_path = _write_cert(tmp_path, cert, "subject.json")
    baseline = {
        "provenance": {
            "provider_digest": {"ids_sha256": "ID456", "tokenizer_sha256": "TOK-A"}
        }
    }
    base_path = _write_cert(tmp_path, baseline, "baseline.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([cert_path], baseline=base_path, profile="ci", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    # Exit must be non-zero due to parity failure
    assert payload["resolution"]["exit_code"] != 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_validate_primary_metric_missing_block_direct() -> None:
    cert = {
        "schema_version": "v1",
        "run_id": "r",
        # primary_metric intentionally omitted to exercise early-return branch
    }
    errors = verify_mod._validate_primary_metric(cert)
    assert errors and "missing primary_metric block" in errors[0]


def test_verify_ppl_recompute_missing_windows_ci_profile(tmp_path: Path) -> None:
    cert = _minimal_ppl_certificate()
    # Provide provider_digest so CI profile reaches ppl recompute path
    cert.setdefault("provenance", {})["provider_digest"] = {
        "ids_sha256": "ID123",
        "tokenizer_sha256": "TOK-A",
    }
    cert_path = _write_cert(tmp_path, cert, "ppl_ci_missing_windows.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command(
            [cert_path],
            baseline=None,
            tolerance=1e-9,
            profile="ci",
            json_out=True,
        )
    exit_code = getattr(ei.value, "exit_code", getattr(ei.value, "code", None))
    assert exit_code != 0
