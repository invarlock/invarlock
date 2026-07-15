from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from invarlock.core.assurance_spectral_replay import replay_spectral_guard
from invarlock.core.guard_evidence import GuardEvidence
from invarlock.core.runner_runtime.guards import _normalize_guard_result
from invarlock.guards.spectral import SpectralGuard


class _TwoFamilyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(4, 4, bias=False)
        self.mlp_fc = nn.Linear(4, 4, bias=False)


def test_validate_retains_complete_module_replay_evidence() -> None:
    torch.manual_seed(17)
    model = _TwoFamilyModel()
    guard = SpectralGuard(degeneracy={"enabled": True})
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True

    result = guard.validate(model, adapter=None, context={})

    expected_modules = set(result.extras["final_metrics"])
    assert expected_modules
    assert set(result.extras["baseline_metrics"]["module_sigmas"]) == expected_modules
    assert set(result.extras["final_z_scores"]) == expected_modules
    assert set(result.extras["module_family_map"]) == expected_modules
    assert (
        set(result.extras["baseline_metrics"]["baseline_degeneracy"])
        == expected_modules
    )
    assert set(result.extras["final_degeneracy"]) == expected_modules
    assert result.metrics["baseline_modules"] == len(expected_modules)
    inventory = result.extras["measurement_inventory"]
    assert set(inventory) == {"prepare", "validate"}
    assert inventory["prepare"]["measured_modules"] == sorted(expected_modules)
    assert inventory["validate"]["measured_modules"] == sorted(expected_modules)
    assert {item["reason"] for item in inventory["prepare"]["excluded_modules"]} == {
        "missing_weight"
    }


def test_external_baseline_uses_same_bounded_cap_rule_as_local_validation() -> None:
    torch.manual_seed(19)
    model = _TwoFamilyModel()
    guard = SpectralGuard(
        max_caps=1,
        degeneracy={"enabled": False},
        family_caps={"attn": {"kappa": 0.0}, "ffn": {"kappa": 1e6}},
        multiple_testing={"method": "bonferroni", "alpha": 1.0, "m": 2},
    )
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True
    guard._external_baseline_required = True
    guard._external_baseline_ready = True

    with torch.no_grad():
        model.attn_proj.weight.mul_(2.0)

    result = guard.validate(model, adapter=None, context={})

    assert result.metrics["caps_applied"] == 1
    assert result.metrics["caps_exceeded"] is False
    assert result.passed is True
    assert result.decision == "monitor"


def test_external_baseline_still_blocks_when_required_evidence_is_unbound() -> None:
    torch.manual_seed(21)
    model = _TwoFamilyModel()
    guard = SpectralGuard(degeneracy={"enabled": False})
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True
    guard._external_baseline_required = True
    guard._external_baseline_ready = False
    guard._external_baseline_reason = "baseline_spectral_evidence_missing"

    result = guard.validate(model, adapter=None, context={})

    assert result.passed is False
    assert result.decision == "block"
    assert result.extras["supported"] is False
    assert result.extras["assurance_blocking"] is True
    assert result.extras["reason"] == "baseline_spectral_evidence_missing"


def test_real_producer_output_round_trips_through_independent_replay() -> None:
    torch.manual_seed(23)
    model = _TwoFamilyModel()
    guard = SpectralGuard(degeneracy={"enabled": True})
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True
    entry = {"name": "spectral", **guard.finalize(model)}
    modules = len(entry["final_metrics"])
    report = {
        "spectral": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "status": "pass",
            "evaluated": True,
            "caps_applied": 0,
            "max_caps": entry["policy"]["max_caps"],
            "caps_exceeded": False,
            "violations": [],
            "summary": {
                "status": "stable",
                "modules_checked": modules,
                "caps_applied": 0,
                "max_caps": entry["policy"]["max_caps"],
                "caps_exceeded": False,
            },
        }
    }

    assert replay_spectral_guard(report, entry, "guards[0]") == []

    entry["final_z_scores"][next(iter(entry["final_z_scores"]))] = 99.0
    assert "final_z_scores" in "\n".join(
        replay_spectral_guard(report, entry, "guards[0]")
    )


def test_selected_finding_persists_verified_weight_mutation_lifecycle() -> None:
    torch.manual_seed(29)
    model = _TwoFamilyModel()
    guard = SpectralGuard(
        correction_enabled=True,
        correction_cap_ratio=1.0,
        ignore_preview_inflation=False,
        degeneracy={"enabled": False},
        max_caps=1,
        family_caps={"attn": {"kappa": 0.0}, "ffn": {"kappa": 1e6}},
        multiple_testing={"method": "bonferroni", "alpha": 1.0, "m": 2},
    )
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True
    baseline = guard.baseline_sigmas["attn_proj"]
    with torch.no_grad():
        model.attn_proj.weight.mul_(10.0)

    result = guard.validate(model, adapter=None, context={})
    ledger = result.extras["correction_ledger"]

    assert ledger["policy_result"] == "corrections_applied"
    assert len(ledger["selected_findings"]) == 1
    correction = ledger["corrections"][0]
    assert correction["correction_id"] == "correction-0001:attn_proj"
    assert correction["module"] == "attn_proj"
    assert correction["attempted"] is True
    assert correction["mutation_applied"] is True
    assert correction["scale_factor"] == pytest.approx(0.1, rel=1e-5)
    assert correction["post_sigma"] == pytest.approx(baseline, rel=1e-5)
    assert correction["pre_weight_digest"] != correction["post_weight_digest"]
    assert ledger["post_correction_metrics"]["attn_proj"] == pytest.approx(
        result.extras["final_metrics"]["attn_proj"], rel=1e-7
    )
    assert (
        result.metrics["selected_budgeted_findings"] == result.metrics["caps_applied"]
    )
    assert result.metrics["cap_budget_exceeded"] == result.metrics["caps_exceeded"]
    assert result.metrics["corrections_applied"] == 1

    normalized = {"name": "spectral", **_normalize_guard_result(result)}
    evidence = GuardEvidence.from_result("spectral", normalized)
    assert evidence is not None
    entry = evidence.as_report_entry()
    report = {
        "spectral": {
            "supported": True,
            "passed": result.passed,
            "decision": result.decision,
            "status": "pass",
            "evaluated": True,
            "caps_applied": result.metrics["caps_applied"],
            "max_caps": result.metrics["max_caps"],
            "caps_exceeded": result.metrics["caps_exceeded"],
            "violations": [dict(item) for item in result.violations],
            "summary": {
                "status": "stable",
                "modules_checked": len(result.extras["final_metrics"]),
                "caps_applied": result.metrics["caps_applied"],
                "max_caps": result.metrics["max_caps"],
                "caps_exceeded": result.metrics["caps_exceeded"],
            },
        }
    }
    assert replay_spectral_guard(report, entry, "guards[0]") == []

    forged_correction = copy.deepcopy(entry)
    forged_correction["correction_ledger"]["corrections"][0]["mutation_applied"] = False
    assert "forged mutation state" in "\n".join(
        replay_spectral_guard(report, forged_correction, "guards[0]")
    )

    forged_digest = copy.deepcopy(entry)
    digest_correction = forged_digest["correction_ledger"]["corrections"][0]
    digest_correction["post_weight_digest"] = digest_correction["pre_weight_digest"]
    assert "did not change the weight digest" in "\n".join(
        replay_spectral_guard(report, forged_digest, "guards[0]")
    )

    noncanonical_id = copy.deepcopy(entry)
    noncanonical_id["correction_ledger"]["corrections"][0]["correction_id"] = "forged"
    assert "correction_id is not canonical" in "\n".join(
        replay_spectral_guard(report, noncanonical_id, "guards[0]")
    )

    forged_outcome = copy.deepcopy(entry)
    forged_outcome["correction_ledger"]["corrections"][0]["outcome"] = "success"
    assert "has invalid outcome" in "\n".join(
        replay_spectral_guard(report, forged_outcome, "guards[0]")
    )

    unsupported_field = copy.deepcopy(entry)
    unsupported_field["correction_ledger"]["legacy_field"] = []
    assert "contains unsupported fields" in "\n".join(
        replay_spectral_guard(report, unsupported_field, "guards[0]")
    )

    omitted_module = copy.deepcopy(entry)
    omitted_module["measurement_inventory"]["validate"]["measured_modules"].pop()
    assert "measured/excluded modules must exactly partition" in "\n".join(
        replay_spectral_guard(report, omitted_module, "guards[0]")
    )


def test_validate_re_resolves_live_modules_after_in_place_replacement() -> None:
    torch.manual_seed(31)
    model = _TwoFamilyModel()
    guard = SpectralGuard(
        correction_enabled=False,
        degeneracy={"enabled": False},
    )
    assert guard.prepare(model, adapter=None, calib=None, policy={})["ready"] is True
    replacement = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        replacement.weight.copy_(100.0 * torch.eye(4))
    model.attn_proj = replacement

    result = guard.validate(model, adapter=None, context={})

    assert result.extras["final_metrics"]["attn_proj"] > 50.0
    assert result.extras["measurement_inventory"]["validate"][
        "identity_changed_modules"
    ] == ["attn_proj"]
    assert result.passed is False
    assert result.decision == "block"
    assert result.metrics["identity_changed_modules"] == ["attn_proj"]
