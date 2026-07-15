from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command
from invarlock.eval.guard_metric_impact import (
    compute_guard_metric_impact,
)
from invarlock.reporting import verify_contract as verify_mod
from tests.cli.verify._support_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _write_runtime_manifest,
)
from tests.reporting._support_guard_metric_impact import attach_canonical_metric_impact


def _write_cert(tmp_path: Path, payload: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_runtime_manifest(path)
    return path


def _verify_release(path: Path) -> None:
    verify_command(
        [path],
        baseline=None,
        profile="release",
        json_out=True,
        expected_runtime_image_digest=_VALID_TEST_IMAGE_DIGEST,
    )


def _release_ready_cert(*, include_guard_metric_impact: bool) -> dict:
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
    cert: dict = {
        "schema_version": "v1",
        "run_id": "run-xyz",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "context": {"runtime": {"execution_mode": "container"}},
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
        "evaluation_windows": {
            "final": {
                "window_ids": [1],
                "logloss": [math.log(10.0)],
                "token_counts": [1],
            }
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0],
        },
        "primary_metric_tail": {
            "evaluated": True,
            "passed": True,
            "warned": False,
            "mode": "warn",
            "policy": {"quantile": 0.95},
            "stats": {"q95": 0.0},
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
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "primary_metric_tail_acceptable": True,
            "guard_metric_impact_acceptable": True,
        },
        "provenance": {"provider_digest": {"ids_sha256": "deadbeef"}},
    }

    if include_guard_metric_impact:
        attach_canonical_metric_impact(cert)

    return cert


def test_verify_release_fails_when_guard_metric_impact_missing(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=False)
    path = _write_cert(tmp_path, cert, "missing_metric_impact.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 1


def test_verify_release_rejects_explicit_metric_impact_skip_marker(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=True)
    cert["guard_metric_impact"] = {
        "skipped": True,
        "mode": "skipped",
        "evaluated": False,
        "passed": False,
        "source": "config:context.run.skip_guard_metric_impact_check",
        "skip_reason": "context.run.skip_guard_metric_impact_check",
        "degradation_limit": 0.01,
    }
    path = _write_cert(tmp_path, cert, "skipped_metric_impact.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 2


def test_verify_release_requires_metric_impact_evaluated_when_not_skipped(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=False)
    cert["guard_metric_impact"] = {
        "mode": "measured",
        "degradation_limit": 0.01,
        "evaluated": False,
    }
    path = _write_cert(tmp_path, cert, "not_evaluated.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 2


def test_verify_release_requires_degradation_when_evaluated(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=False)
    cert["guard_metric_impact"] = {
        "mode": "measured",
        "degradation_limit": 0.01,
        "evaluated": True,
    }
    path = _write_cert(tmp_path, cert, "missing_ratio.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 2


def test_verify_release_passes_with_canonical_metric_impact(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=True)
    path = _write_cert(tmp_path, cert, "with_metric_impact.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_release_fails_when_guard_metric_impact_gate_failed(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=False)
    cert["primary_metric"]["final"] = 10.2
    cert["primary_metric"]["ratio_vs_baseline"] = 1.02
    cert["evaluation_windows"]["final"]["logloss"] = [math.log(10.2)]
    attach_canonical_metric_impact(cert)
    impact = cert["guard_metric_impact"]
    measurement = compute_guard_metric_impact("ppl_causal", 10.0, 10.2)
    assert measurement is not None
    impact.update(measurement.to_metrics())
    impact["bare_facts"] = {
        **impact["bare_facts"],
        "weighted_logloss_sum": math.log(10.0),
    }
    impact["bare_report"] = {
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "final": {
            "window_ids": [1],
            "logloss": [math.log(10.0)],
            "token_counts": [1],
        },
        "status": "success",
    }
    impact["passed"] = False
    cert["validation"]["guard_metric_impact_acceptable"] = False
    path = _write_cert(tmp_path, cert, "failed_metric_impact_gate.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 1


def test_verify_release_fails_when_canonical_gate_flagged(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_metric_impact=True)
    cert["validation"]["spectral_stable"] = False
    path = _write_cert(tmp_path, cert, "failed_spectral_gate.json")

    with pytest.raises(typer.Exit) as ei:
        _verify_release(path)

    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 1
