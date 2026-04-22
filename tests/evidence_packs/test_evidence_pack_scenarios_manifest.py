from __future__ import annotations

import json
from pathlib import Path


def _load_scenarios() -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts/evidence_packs/scenarios.json"
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
    allowed_guards = {"invariants", "primary_metric", "spectral", "rmt", "variance"}

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
        "ve_mlp_scale_skew": "variance",
    }

    for scenario_id, expected_guard in expected_primary_guard.items():
        assert scenario_id in by_id, f"{scenario_id} missing from scenarios manifest"
        assert by_id[scenario_id].get("primary_guard") == expected_guard


def test_scenarios_require_direct_primary_guard_hits_for_demo_probes() -> None:
    scenarios = _load_scenarios()
    by_id = {str(item.get("id")): item for item in scenarios}

    required = {
        "nan_injection",
        "extreme_quant",
        "rmt_norm_noise",
        "spectral_moderate_scale",
        "ve_mlp_scale_skew",
    }
    for scenario_id in required:
        scenario = by_id.get(scenario_id)
        assert scenario is not None, f"{scenario_id} missing from scenarios manifest"
        requirements = scenario.get("requirements")
        assert isinstance(requirements, dict), (
            f"{scenario_id}: requirements must be a mapping"
        )
        assert requirements.get("primary_guard_required") is True, (
            f"{scenario_id}: primary_guard_required must be true"
        )

    for scenario_id in (
        "rmt_norm_noise",
        "spectral_moderate_scale",
        "ve_mlp_scale_skew",
    ):
        scenario = by_id[scenario_id]
        generation = scenario.get("generation")
        assert isinstance(generation, dict), f"{scenario_id}: generation must be dict"
        env = generation.get("env")
        assert isinstance(env, dict) and env, (
            f"{scenario_id}: generation.env must be a non-empty mapping"
        )
        for key, value in env.items():
            assert isinstance(key, str) and key.startswith("INVARLOCK_"), (
                f"{scenario_id}: invalid env key {key!r}"
            )
            assert isinstance(value, str), (
                f"{scenario_id}: env value must be string for {key}"
            )


def test_optional_model_specific_error_env_overrides_have_valid_shape() -> None:
    scenarios = _load_scenarios()

    for scenario in scenarios:
        scenario_id = str(scenario.get("id"))
        generation = scenario.get("generation")
        if not isinstance(generation, dict):
            continue
        env_by_model = generation.get("env_by_model")
        if env_by_model is None:
            continue
        assert isinstance(env_by_model, dict), (
            f"{scenario_id}: generation.env_by_model must be a mapping"
        )
        for model_name, override_env in env_by_model.items():
            assert isinstance(model_name, str) and model_name.strip(), (
                f"{scenario_id}: invalid env_by_model key {model_name!r}"
            )
            assert isinstance(override_env, dict) and override_env, (
                f"{scenario_id}: env_by_model[{model_name!r}] must be a non-empty mapping"
            )
            for key, value in override_env.items():
                assert isinstance(key, str) and key.startswith("INVARLOCK_"), (
                    f"{scenario_id}: invalid override env key {key!r}"
                )
                assert isinstance(value, str), (
                    f"{scenario_id}: override env value must be string for {key}"
                )


def test_deepseek_error_probe_overrides_are_present_and_narrower_than_defaults() -> None:
    scenarios = _load_scenarios()
    by_id = {str(item.get("id")): item for item in scenarios}
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    rmt_generation = by_id["rmt_norm_noise"]["generation"]
    assert isinstance(rmt_generation, dict)
    rmt_env = rmt_generation["env"]
    rmt_overrides = rmt_generation["env_by_model"][model_id]
    assert int(rmt_overrides["INVARLOCK_RMT_ANISO_MAX_PARAMS"]) < int(
        rmt_env["INVARLOCK_RMT_ANISO_MAX_PARAMS"]
    )
    assert int(rmt_overrides["INVARLOCK_RMT_ANISO_MAX_ROWS"]) < int(
        rmt_env["INVARLOCK_RMT_ANISO_MAX_ROWS"]
    )
    assert float(rmt_overrides["INVARLOCK_RMT_ANISO_ROW_FRAC"]) < float(
        rmt_env["INVARLOCK_RMT_ANISO_ROW_FRAC"]
    )
    assert float(rmt_overrides["INVARLOCK_RMT_ANISO_BLEND"]) < float(
        rmt_env["INVARLOCK_RMT_ANISO_BLEND"]
    )

    spectral_generation = by_id["spectral_moderate_scale"]["generation"]
    assert isinstance(spectral_generation, dict)
    spectral_env = spectral_generation["env"]
    spectral_overrides = spectral_generation["env_by_model"][model_id]
    assert float(spectral_overrides["INVARLOCK_SPECTRAL_SCALE_FACTOR"]) < float(
        spectral_env["INVARLOCK_SPECTRAL_SCALE_FACTOR"]
    )
    assert int(spectral_overrides["INVARLOCK_SPECTRAL_MAX_PARAMS"]) < int(
        spectral_env["INVARLOCK_SPECTRAL_MAX_PARAMS"]
    )


def test_missing_tensors_accepts_catastrophic_validation_failure_as_detection() -> None:
    scenarios = _load_scenarios()
    by_id = {str(item.get("id")): item for item in scenarios}

    scenario = by_id["missing_tensors"]
    requirements = scenario.get("requirements")
    assert isinstance(requirements, dict)
    detectors = requirements.get("detectors_any_of")
    assert isinstance(detectors, list)

    assert {
        "kind": "validation_flag",
        "flag": "primary_metric_acceptable",
        "expected": False,
    } in detectors
