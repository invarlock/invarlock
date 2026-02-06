from __future__ import annotations

import json
from pathlib import Path


def _load_scenarios() -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts/proof_packs/scenarios.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("scenarios", [])


def test_scenarios_include_intent_and_primary_guard_metadata() -> None:
    scenarios = _load_scenarios()

    allowed_intents = {
        "clean_control",
        "catastrophic_failure",
        "subtle_detectable",
        "fault_detection",
    }
    allowed_guards = {"invariants", "primary_metric", "spectral", "rmt"}

    assert scenarios, "scenario manifest must not be empty"

    for scenario in scenarios:
        scenario_id = str(scenario.get("id"))
        intent = scenario.get("intent")
        primary_guard = scenario.get("primary_guard")

        assert isinstance(intent, str), f"{scenario_id}: missing intent metadata"
        assert intent in allowed_intents, f"{scenario_id}: unknown intent={intent!r}"
        assert isinstance(primary_guard, str), (
            f"{scenario_id}: missing primary_guard metadata"
        )
        assert primary_guard in allowed_guards, (
            f"{scenario_id}: unknown primary_guard={primary_guard!r}"
        )


def test_scenarios_target_expected_guards_for_injection_probes() -> None:
    scenarios = _load_scenarios()
    by_id = {str(item.get("id")): item for item in scenarios}

    expected_primary_guard = {
        "nan_injection": "invariants",
        "inf_injection": "invariants",
        "shape_mismatch": "invariants",
        "missing_tensors": "invariants",
        "weight_tying_break": "invariants",
        "rmt_norm_noise": "rmt",
        "spectral_moderate_scale": "spectral",
    }

    for scenario_id, expected_guard in expected_primary_guard.items():
        assert scenario_id in by_id, f"{scenario_id} missing from scenarios manifest"
        assert by_id[scenario_id].get("primary_guard") == expected_guard
