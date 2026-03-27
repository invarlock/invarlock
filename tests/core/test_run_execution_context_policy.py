from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_execution_context_policy import (
    build_run_context_payload,
    build_run_execution_config_payloads,
)


def test_build_run_context_payload_merges_invariants_baseline_and_loss_context() -> None:
    cfg = SimpleNamespace(
        eval=SimpleNamespace(max_pm_ratio=1.5),
        dataset=SimpleNamespace(provider="wikitext2"),
        guards=SimpleNamespace(
            spectral={"enabled": True},
            rmt={},
            variance={},
            invariants={"profile_checks": ["existing"]},
        ),
        context={"custom": {"enabled": True}},
    )
    model_profile = SimpleNamespace(
        family="test",
        default_loss="ppl_causal",
        module_selectors={"decoder": ["x"]},
        invariants=["shape_ok", "existing"],
        cert_lints=[{"name": "lint"}],
    )

    payload = build_run_context_payload(
        cfg=cfg,
        profile="ci",
        pairing_schedule={"preview": {}, "final": {}},
        seed_bundle={"python": 43, "numpy": 43, "torch": 43},
        plugin_provenance={"adapter": {"name": "hf"}},
        run_id="run-1",
        baseline_report_data={
            "evaluation_windows": {
                "final": {
                    "window_ids": [1, 2],
                    "logloss": [0.1, 0.2],
                    "token_counts": [2, 2],
                }
            }
        },
        pm_acceptance_range=(0.9, 1.1),
        pm_drift_band=(0.95, 1.05),
        guard_overhead_threshold=0.02,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=True,
        to_serialisable_dict_fn=lambda obj: (
            dict(obj)
            if isinstance(obj, dict)
            else {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        ),
    )

    assert payload["guards"]["invariants"]["profile_checks"] == [
        "existing",
        "shape_ok",
    ]
    assert payload["run"]["tiny_relax"] is True
    assert payload["baseline_eval_windows"]["final"]["token_counts"] == [2, 2]
    assert payload["pm_acceptance_range"] == (0.9, 1.1)
    assert payload["pm_drift_band"] == (0.95, 1.05)
    assert payload["guard_overhead_threshold"] == 0.02
    assert payload["model_profile"]["module_selectors"] == {"decoder": ["x"]}
    assert payload["custom"] == {"enabled": True}
    assert payload["eval"]["loss"]["resolved_type"] == "ppl_causal"


class _PlanWrapper:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data


def test_build_run_execution_config_payloads_defaults_and_selector_injection() -> None:
    cfg = SimpleNamespace(
        auto=SimpleNamespace(enabled="yes", tier="fast", probes="4", target_pm_ratio="3"),
        edit=SimpleNamespace(plan=_PlanWrapper({"alpha": 1})),
    )
    model_profile = SimpleNamespace(module_selectors={"heads": [0, 1]})

    payloads = build_run_execution_config_payloads(cfg=cfg, model_profile=model_profile)

    assert payloads.auto_config == {
        "enabled": True,
        "tier": "fast",
        "probes": 4,
        "target_pm_ratio": 3.0,
    }
    assert payloads.edit_config == {
        "alpha": 1,
        "module_selectors": {"heads": [0, 1]},
    }


def test_build_run_execution_config_payloads_preserves_existing_module_selectors() -> None:
    cfg = SimpleNamespace(
        auto=SimpleNamespace(),
        edit=SimpleNamespace(plan={"module_selectors": {"heads": [9]}}),
    )
    model_profile = SimpleNamespace(module_selectors={"heads": [0, 1]})

    payloads = build_run_execution_config_payloads(cfg=cfg, model_profile=model_profile)

    assert payloads.auto_config == {
        "enabled": False,
        "tier": "balanced",
        "probes": 0,
        "target_pm_ratio": 2.0,
    }
    assert payloads.edit_config == {"module_selectors": {"heads": [9]}}
