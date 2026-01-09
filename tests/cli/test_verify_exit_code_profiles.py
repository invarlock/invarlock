from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands import verify as verify_mod
from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp_path: Path, payload: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ppl_certificate(
    *,
    preview: float = 10.0,
    final: float = 10.0,
    ratio: float = 1.0,
    baseline_final: float = 10.0,
    include_provider_digest: bool = True,
    include_guard_overhead: bool = False,
) -> dict:
    spectral_contract = {"estimator": {"type": "power_iter", "iters": 4, "init": "ones"}}
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    spectral_hash = verify_mod._measurement_contract_digest(spectral_contract)
    rmt_hash = verify_mod._measurement_contract_digest(rmt_contract)

    cert: dict = {
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
            "preview": preview,
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
    }

    if include_provider_digest:
        cert.setdefault("provenance", {})["provider_digest"] = {
            "ids_sha256": "ID123",
            "tokenizer_sha256": "TOK-A",
        }

    if include_guard_overhead:
        cert["guard_overhead"] = {
            "skipped": True,
            "mode": "skipped",
            "source": "unit-test",
            "overhead_threshold": 0.01,
        }

    return cert


@pytest.mark.parametrize(
    ("profile", "expected_exit_code", "expected_ok"),
    [
        ("dev", 0, True),
        ("ci", 1, False),
        ("release", 1, False),
    ],
)
def test_verify_drift_band_exit_code_varies_by_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profile: str,
    expected_exit_code: int,
    expected_ok: bool,
) -> None:
    # Drift is only enforced in CI/Release; dev should ignore it.
    cert = _ppl_certificate(
        preview=10.0,
        final=12.0,
        ratio=1.2,
        baseline_final=10.0,
        include_provider_digest=True,
        include_guard_overhead=True,
    )
    path = _write_cert(tmp_path, cert, f"drift_{profile}.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile=profile, json_out=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["ok"] is expected_ok
    assert payload["resolution"]["exit_code"] == expected_exit_code
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == expected_exit_code


@pytest.mark.parametrize(
    ("profile", "expected_exit_code", "expected_ok"),
    [
        ("dev", 0, True),
        ("ci", 0, True),
        ("release", 1, False),
    ],
)
def test_verify_guard_overhead_exit_code_varies_by_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profile: str,
    expected_exit_code: int,
    expected_ok: bool,
) -> None:
    # guard_overhead is release-only enforcement.
    cert = _ppl_certificate(
        preview=10.0,
        final=10.0,
        ratio=1.0,
        baseline_final=10.0,
        include_provider_digest=True,
        include_guard_overhead=False,
    )
    path = _write_cert(tmp_path, cert, f"overhead_{profile}.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile=profile, json_out=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["ok"] is expected_ok
    assert payload["resolution"]["exit_code"] == expected_exit_code
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == expected_exit_code


@pytest.mark.parametrize(
    ("profile", "expected_exit_code"),
    [
        ("dev", 0),
        ("ci", 3),
        ("release", 3),
    ],
)
def test_verify_provider_digest_exit_code_varies_by_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profile: str,
    expected_exit_code: int,
) -> None:
    cert = _ppl_certificate(
        preview=10.0,
        final=10.0,
        ratio=1.0,
        baseline_final=10.0,
        include_provider_digest=False,
        include_guard_overhead=True,
    )
    path = _write_cert(tmp_path, cert, f"no_provider_digest_{profile}.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile=profile, json_out=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["resolution"]["exit_code"] == expected_exit_code
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == expected_exit_code


@pytest.mark.parametrize("profile", ["dev", "ci", "release"])
def test_verify_exit_code_2_when_any_certificate_is_malformed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profile: str,
) -> None:
    bad = _ppl_certificate(
        preview=10.0,
        final=10.0,
        ratio=float("nan"),
        baseline_final=10.0,
        include_provider_digest=True,
        include_guard_overhead=True,
    )
    good = _ppl_certificate(
        preview=10.0,
        final=10.0,
        ratio=1.0,
        baseline_final=10.0,
        include_provider_digest=True,
        include_guard_overhead=True,
    )
    p_bad = _write_cert(tmp_path, bad, "bad.json")
    p_good = _write_cert(tmp_path, good, "good.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([p_bad, p_good], baseline=None, profile=profile, json_out=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["ok"] is False
    assert payload["resolution"]["exit_code"] == 2
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 2
