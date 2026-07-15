from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_binding import verify_final_verdict_report_binding
from invarlock.evidence_pack_contracts.probes import (
    ProbeValidationError,
    build_probe_binding,
    load_probe_file,
    validate_probe_payload,
)
from invarlock.evidence_pack_snapshot import PackSnapshot


def _report() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "meta": {"model_id": "fixture/model", "adapter": "hf_causal", "profile": "ci"},
        "context": {"runtime": {"execution_mode": "container"}},
        "provenance": {"provider_digest": {"ids_sha256": "fixture-provider"}},
    }


def _binding(report: dict[str, object]) -> dict[str, object]:
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode() + b"\n"
    return build_probe_binding(report, "sha256:" + hashlib.sha256(raw).hexdigest())


def _rmt_probe(
    *, stable: bool = False, binding: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "schema": "invarlock/rmt-probe-v1",
        "probe": "rmt_cross_model_v1",
        "stable": stable,
        "passed": stable,
        "action": "continue" if stable else "abort",
        "stable_guard": stable,
        "epsilon_by_family": {"ffn": 0.01},
        "epsilon_default": 0.01,
        "epsilon_violations": []
        if stable
        else [
            {
                "family": "ffn",
                "module": "ffn",
                "edge_base": 1.0,
                "edge_cur": 2.0,
                "delta": 1.0,
                "allowed": 1.01,
                "epsilon": 0.01,
            }
        ],
        "violations": [],
        "metrics": {
            "stable": stable,
            "epsilon_default": 0.01,
            "epsilon_by_family": {"ffn": 0.01},
            "edge_base_by_family": {"ffn": 1.0},
            "edge_cur_by_family": {"ffn": 1.0 if stable else 2.0},
        },
        "binding": binding or _binding(_report()),
    }


def _ve_probe(*, binding: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": "invarlock/ve-probe-v1",
        "probe": "ve_probe_v1",
        "signal": True,
        "signal_reasons": [],
        "would_enable": True,
        "gate_reason": "enabled",
        "proposed_scales": 2,
        "ppl_no_ve": 10.0,
        "ppl_with_ve": 9.0,
        "abs_improvement": 1.0,
        "ab_gain": 0.1,
        "ratio_ci": [0.8, 0.9],
        "predictive_gate": {"would_enable": True, "reason": "enabled"},
        "calibration": {
            "windows": 12,
            "min_coverage": 10,
            "tier": "balanced",
            "profile": "ci",
        },
        "binding": binding or _binding(_report()),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_probe_contracts_accept_typed_v1_payloads() -> None:
    assert validate_probe_payload("rmt_probe.json", _rmt_probe())["stable"] is False
    assert validate_probe_payload("ve_probe.json", _ve_probe())["signal"] is True


@pytest.mark.parametrize(
    "filename,payload",
    [("rmt_probe.json", _rmt_probe()), ("ve_probe.json", _ve_probe())],
)
def test_probe_contract_rejects_neutral_extensions_and_host_paths(
    filename: str,
    payload: dict[str, object],
) -> None:
    extended = {**payload, "note": "unused"}
    with pytest.raises(ProbeValidationError, match="invalid"):
        validate_probe_payload(filename, extended)

    leaked = dict(payload)
    leaked["metrics" if filename == "rmt_probe.json" else "calibration"] = {
        "source": "/home/operator/private/model"
    }
    with pytest.raises(ProbeValidationError, match="host-local path"):
        validate_probe_payload(filename, leaked)


def test_probe_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "rmt_probe.json"
    path.write_text(
        '{"schema":"invarlock/rmt-probe-v1","schema":"other"}', encoding="utf-8"
    )
    with pytest.raises(ProbeValidationError, match="duplicate key"):
        load_probe_file(path)


def test_probe_is_structural_json_in_pack_snapshot(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    probe = pack / "reports/model/scenario/rmt_probe.json"
    probe.parent.mkdir(parents=True)
    probe.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    snapshot, errors = PackSnapshot.capture(pack)
    assert snapshot is None
    assert any("duplicate key" in error for error in errors)


def test_final_verdict_binds_probe_and_detects_mutation(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/model/scenario/evaluation.report.json"
    probe = report.parent / "rmt_probe.json"
    verdict = pack / "results/final_verdict.json"
    report_payload = _report()
    _write_json(report, report_payload)
    _write_json(probe, _rmt_probe(binding=_binding(report_payload)))
    _write_json(
        verdict,
        {
            "report_bindings": [
                {
                    "path": "reports/model/scenario/evaluation.report.json",
                    "report_sha256": _sha256(report),
                    "run_id": "run-1",
                    "probe_bindings": [
                        {
                            "path": "reports/model/scenario/rmt_probe.json",
                            "sha256": _sha256(probe),
                        }
                    ],
                }
            ]
        },
    )
    assert verify_final_verdict_report_binding(pack, require_binding=True) == []

    _write_json(probe, _rmt_probe(stable=True, binding=_binding(report_payload)))
    errors = verify_final_verdict_report_binding(pack, require_binding=True)
    assert any("probe_bindings do not match" in error for error in errors)


def test_final_verdict_rejects_unbound_probe(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/model/scenario/evaluation.report.json"
    probe = report.parent / "ve_probe.json"
    report_payload = _report()
    _write_json(report, report_payload)
    _write_json(probe, _ve_probe(binding=_binding(report_payload)))
    _write_json(
        pack / "results/final_verdict.json",
        {
            "report_bindings": [
                {
                    "path": "reports/model/scenario/evaluation.report.json",
                    "report_sha256": _sha256(report),
                }
            ]
        },
    )
    errors = verify_final_verdict_report_binding(pack, require_binding=True)
    assert any("probe_bindings do not match" in error for error in errors)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "other/model"),
        ("runtime", {"execution_mode": "host"}),
        ("toolchain", {"adapter": "other", "profile": "ci"}),
        ("provider_digest", {"ids_sha256": "other-provider"}),
    ],
)
def test_final_verdict_rejects_probe_identity_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/model/scenario/evaluation.report.json"
    probe = report.parent / "rmt_probe.json"
    report_payload = _report()
    _write_json(report, report_payload)
    binding = _binding(report_payload)
    binding[field] = value
    _write_json(probe, _rmt_probe(binding=binding))
    _write_json(pack / "results/final_verdict.json", {"report_bindings": []})
    errors = verify_final_verdict_report_binding(pack, require_binding=True)
    assert any("binding does not match" in error for error in errors)


def test_probe_contract_rejects_nested_string_and_nonfinite_values() -> None:
    typed = _rmt_probe()
    typed["metrics"] = {**typed["metrics"], "epsilon_default": "0.01"}
    with pytest.raises(ProbeValidationError, match="field types"):
        validate_probe_payload("rmt_probe.json", typed)

    typed = _rmt_probe()
    typed["epsilon_violations"] = [
        {
            "family": "ffn",
            "module": "ffn",
            "edge_base": 1.0,
            "edge_cur": float("inf"),
            "delta": 1.0,
            "allowed": 1.01,
            "epsilon": 0.01,
        }
    ]
    with pytest.raises(ProbeValidationError, match="field types"):
        validate_probe_payload("rmt_probe.json", typed)


def test_rmt_probe_contract_recomputes_epsilon_violation_math() -> None:
    typed = _rmt_probe()
    typed["epsilon_violations"][0]["allowed"] = 1.02
    with pytest.raises(ProbeValidationError, match="arithmetic"):
        validate_probe_payload("rmt_probe.json", typed)

    typed = _rmt_probe()
    typed["stable"] = True
    with pytest.raises(ProbeValidationError, match="status"):
        validate_probe_payload("rmt_probe.json", typed)

    # The ordinary subject guard and baseline-relative epsilon comparison are
    # distinct measurements. Their statuses may legitimately differ, but the
    # three representations of the ordinary guard must agree.
    typed = _rmt_probe()
    typed["passed"] = True
    typed["stable_guard"] = True
    typed["metrics"]["stable"] = True
    assert validate_probe_payload("rmt_probe.json", typed)["stable"] is False

    typed["passed"] = False
    with pytest.raises(ProbeValidationError, match="status"):
        validate_probe_payload("rmt_probe.json", typed)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("aggregate", "arithmetic"),
        ("policy", "epsilon policy"),
        ("missing-violation", "do not match family aggregates"),
        ("duplicate-family", "unknown or duplicate family"),
    ],
)
def test_rmt_probe_contract_binds_policy_aggregates_and_violation_set(
    mutation: str, match: str
) -> None:
    typed = _rmt_probe()
    if mutation == "aggregate":
        typed["metrics"]["edge_cur_by_family"]["ffn"] = 1.0
    elif mutation == "policy":
        typed["metrics"]["epsilon_default"] = 0.02
    elif mutation == "missing-violation":
        typed["epsilon_violations"] = []
        typed["stable"] = True
    else:
        typed["epsilon_violations"].append(dict(typed["epsilon_violations"][0]))

    with pytest.raises(ProbeValidationError, match=match):
        validate_probe_payload("rmt_probe.json", typed)


def test_ve_probe_contract_requires_positive_signal_semantics() -> None:
    typed = _ve_probe()
    typed["ab_gain"] = -0.1
    with pytest.raises(ProbeValidationError, match="positive measured gain"):
        validate_probe_payload("ve_probe.json", typed)

    typed = _ve_probe()
    typed["predictive_gate"] = {"would_enable": False, "reason": "enabled"}
    with pytest.raises(ProbeValidationError, match="enable decision"):
        validate_probe_payload("ve_probe.json", typed)

    typed = _ve_probe()
    typed["ratio_ci"] = [0.9, 0.8]
    with pytest.raises(ProbeValidationError, match="interval is reversed"):
        validate_probe_payload("ve_probe.json", typed)

    typed = _ve_probe()
    typed["ppl_with_ve"] = None
    with pytest.raises(ProbeValidationError, match="paired"):
        validate_probe_payload("ve_probe.json", typed)

    typed = _ve_probe()
    typed["ratio_ci"] = [-0.1, 0.9]
    with pytest.raises(ProbeValidationError, match="must be positive"):
        validate_probe_payload("ve_probe.json", typed)
