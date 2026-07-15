from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_policy import (
    _baseline_eval_windows,
    _normalize_edit_plan,
    _normalize_profile_checks,
    _section_dict,
    build_run_context_payload,
    build_run_execution_config_payloads,
)


def test_normalize_profile_checks_handles_scalar_and_falsy_values() -> None:
    assert _normalize_profile_checks("shape_ok") == ["shape_ok"]
    assert _normalize_profile_checks(None) == []
    assert _normalize_profile_checks(0) == []


def test_section_dict_recovers_from_section_dispatch_errors() -> None:
    class _Config:
        guards = {"spectral": {"enabled": True}}

        def section(self, _name: str) -> object:
            raise TypeError("boom")

    assert _section_dict(_Config(), "guards") == {"spectral": {"enabled": True}}


def test_baseline_eval_windows_rejects_malformed_payloads_and_omits_bad_token_counts() -> (
    None
):
    assert _baseline_eval_windows(None) is None
    assert _baseline_eval_windows({"evaluation_windows": []}) is None
    assert _baseline_eval_windows({"evaluation_windows": {"final": []}}) is None
    assert (
        _baseline_eval_windows(
            {"evaluation_windows": {"final": {"window_ids": "bad", "logloss": []}}}
        )
        is None
    )
    assert _baseline_eval_windows(
        {
            "evaluation_windows": {
                "final": {
                    "window_ids": [1],
                    "logloss": [0.1],
                    "token_counts": "bad",
                }
            }
        }
    ) == {"final": {"window_ids": [1], "logloss": [0.1]}}


def test_build_run_context_payload_merges_invariants_baseline_and_loss_context() -> (
    None
):
    cfg = SimpleNamespace(
        model=SimpleNamespace(id="gpt2"),
        eval=SimpleNamespace(max_pm_ratio=1.5),
        dataset=SimpleNamespace(provider="wikitext2"),
        guards=SimpleNamespace(
            spectral={"enabled": True},
            rmt={},
            variance={},
            invariants={"profile_checks": ["existing"]},
        ),
        context={
            "custom": {"enabled": True},
            "model_id": "context-override-must-not-win",
        },
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
        guard_metric_degradation_limit=0.02,
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
    assert payload["guard_metric_degradation_limit"] == 0.02
    assert payload["model_profile"]["module_selectors"] == {"decoder": ["x"]}
    assert payload["model_id"] == "gpt2"
    assert payload["custom"] == {"enabled": True}
    assert payload["eval"]["loss"]["resolved_type"] == "ppl_causal"


def test_build_run_context_payload_includes_non_empty_assurance_section() -> None:
    cfg = SimpleNamespace(
        eval={},
        dataset={},
        guards={"spectral": {}, "rmt": {}, "variance": {}, "invariants": {}},
        assurance={"mode": "strict"},
        context={},
    )
    model_profile = SimpleNamespace(
        family="test",
        default_loss="ppl_causal",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )

    payload = build_run_context_payload(
        cfg=cfg,
        profile="ci",
        pairing_schedule=None,
        seed_bundle={},
        plugin_provenance={},
        run_id="run-assurance",
        baseline_report_data=None,
        pm_acceptance_range=None,
        pm_drift_band=None,
        guard_metric_degradation_limit=0.01,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
        to_serialisable_dict_fn=lambda obj: dict(obj) if isinstance(obj, dict) else {},
    )

    assert payload["assurance"] == {"mode": "strict"}


def test_build_run_context_payload_skips_invalid_baseline_and_non_dict_context() -> (
    None
):
    cfg = SimpleNamespace(
        eval=SimpleNamespace(max_pm_ratio=1.5),
        dataset=SimpleNamespace(provider="wikitext2"),
        guards=SimpleNamespace(
            spectral={},
            rmt={},
            variance={},
            invariants={"profile_checks": "shape_ok"},
        ),
        context="ignored",
    )
    model_profile = SimpleNamespace(
        family="test",
        default_loss="ppl_causal",
        module_selectors={},
        invariants=["shape_ok", "extra_ok"],
        cert_lints=[],
    )

    def _to_serialisable(obj: object) -> object:
        if obj == "ignored":
            return "not-a-dict"
        if isinstance(obj, dict):
            return dict(obj)
        return {
            key: value for key, value in vars(obj).items() if not key.startswith("_")
        }

    payload = build_run_context_payload(
        cfg=cfg,
        profile=None,
        pairing_schedule=None,
        seed_bundle={},
        plugin_provenance={},
        run_id="run-2",
        baseline_report_data={"evaluation_windows": {"final": {"window_ids": "bad"}}},
        pm_acceptance_range=None,
        pm_drift_band=None,
        guard_metric_degradation_limit=0.05,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
        to_serialisable_dict_fn=_to_serialisable,
    )

    assert payload["profile"] == ""
    assert "run" not in payload
    assert "baseline_eval_windows" not in payload
    assert "pm_drift_band" not in payload
    assert payload["guards"]["invariants"]["profile_checks"] == [
        "shape_ok",
        "extra_ok",
    ]
    assert payload["eval"]["max_pm_ratio"] == 1.5
    assert payload["eval"]["loss"]["resolved_type"] == "ppl_causal"


def test_build_run_context_payload_preserves_non_mapping_eval_payloads() -> None:
    cfg = SimpleNamespace(
        eval=SimpleNamespace(max_pm_ratio=1.5),
        dataset=SimpleNamespace(provider="wikitext2"),
        guards=SimpleNamespace(spectral={}, rmt={}, variance={}, invariants={}),
        context={},
    )
    model_profile = SimpleNamespace(
        family="test",
        default_loss="ppl_causal",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )

    def _to_serialisable(obj: object) -> object:
        if isinstance(obj, dict) and obj.get("max_pm_ratio") == 1.5:
            return "not-a-dict"
        if isinstance(obj, dict):
            return dict(obj)
        return {
            key: value for key, value in vars(obj).items() if not key.startswith("_")
        }

    payload = build_run_context_payload(
        cfg=cfg,
        profile="dev",
        pairing_schedule=None,
        seed_bundle={},
        plugin_provenance={},
        run_id="run-3",
        baseline_report_data=None,
        pm_acceptance_range=None,
        pm_drift_band=None,
        guard_metric_degradation_limit=0.01,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
        to_serialisable_dict_fn=_to_serialisable,
    )

    assert payload["eval"] == "not-a-dict"


def test_build_run_context_payload_ignores_non_mapping_serialised_context() -> None:
    cfg = SimpleNamespace(
        eval=SimpleNamespace(max_pm_ratio=1.5),
        dataset=SimpleNamespace(provider="wikitext2"),
        guards=SimpleNamespace(spectral={}, rmt={}, variance={}, invariants={}),
        context={"raw": True},
    )
    model_profile = SimpleNamespace(
        family="test",
        default_loss="ppl_causal",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )

    payload = build_run_context_payload(
        cfg=cfg,
        profile="dev",
        pairing_schedule=None,
        seed_bundle={},
        plugin_provenance={},
        run_id="run-4",
        baseline_report_data=None,
        pm_acceptance_range=None,
        pm_drift_band=None,
        guard_metric_degradation_limit=0.01,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
        to_serialisable_dict_fn=lambda obj: (
            "not-a-dict"
            if obj == {"raw": True}
            else dict(obj)
            if isinstance(obj, dict)
            else {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        ),
    )

    assert payload["profile"] == "dev"
    assert payload["eval"]["loss"]["resolved_type"] == "ppl_causal"
    assert "raw" not in payload


class _PlanWrapper:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data


def test_build_run_execution_config_payloads_defaults_and_selector_injection() -> None:
    cfg = SimpleNamespace(
        auto=SimpleNamespace(
            enabled="yes", tier="fast", probes="4", target_pm_ratio="3"
        ),
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


def test_build_run_execution_config_payloads_preserves_existing_module_selectors() -> (
    None
):
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


class _ItemsPlan:
    def items(self) -> list[tuple[str, int]]:
        return [("beta", 2)]

    def __iter__(self):
        return iter(self.items())


class _BadItemsPlan:
    def items(self) -> list[tuple[str, int]]:
        return [("gamma", 3)]

    def __iter__(self):
        raise ValueError("bad plan")


class _BadAuto:
    @property
    def enabled(self) -> bool:
        raise TypeError("bad enabled")

    @property
    def tier(self) -> str:
        raise ValueError("bad tier")

    @property
    def probes(self) -> str:
        raise ValueError("bad probes")

    @property
    def target_pm_ratio(self) -> str:
        raise TypeError("bad ratio")


class _RaisingGetDict(dict):
    def get(self, key, default=None):  # type: ignore[override]
        if key == "enabled":
            raise TypeError("bad enabled")
        if key == "tier":
            raise ValueError("bad tier")
        if key == "probes":
            raise TypeError("bad probes")
        if key == "target_pm_ratio":
            raise ValueError("bad ratio")
        return super().get(key, default)


def test_build_run_execution_config_payloads_supports_items_plan() -> None:
    cfg = SimpleNamespace(
        auto=SimpleNamespace(),
        edit=SimpleNamespace(plan=_ItemsPlan()),
    )
    model_profile = SimpleNamespace(module_selectors={"heads": [1, 2]})

    payloads = build_run_execution_config_payloads(cfg=cfg, model_profile=model_profile)

    assert payloads.edit_config == {
        "beta": 2,
        "module_selectors": {"heads": [1, 2]},
    }


def test_build_run_execution_config_payloads_falls_back_on_bad_auto_and_plan() -> None:
    cfg = SimpleNamespace(
        auto=_BadAuto(),
        edit=SimpleNamespace(plan=_BadItemsPlan()),
    )
    model_profile = SimpleNamespace(module_selectors=["not-a-dict"])

    payloads = build_run_execution_config_payloads(cfg=cfg, model_profile=model_profile)

    assert payloads.auto_config == {
        "enabled": False,
        "tier": "balanced",
        "probes": 0,
        "target_pm_ratio": 2.0,
    }
    assert payloads.edit_config == {}


def test_build_run_execution_config_payloads_handles_missing_edit_namespace() -> None:
    cfg = SimpleNamespace(auto=SimpleNamespace())
    model_profile = SimpleNamespace(module_selectors={})

    payloads = build_run_execution_config_payloads(cfg=cfg, model_profile=model_profile)

    assert payloads.auto_config == {
        "enabled": False,
        "tier": "balanced",
        "probes": 0,
        "target_pm_ratio": 2.0,
    }
    assert payloads.edit_config == {}


def test_normalize_edit_plan_returns_empty_for_non_mapping_objects() -> None:
    assert _normalize_edit_plan(object()) == {}


def test_build_run_execution_config_payloads_recovers_from_bad_auto_mapping_getters() -> (
    None
):
    class _Cfg:
        edit = SimpleNamespace(plan=object())

        def section(self, name: str) -> object:
            if name == "auto":
                return _RaisingGetDict()
            raise KeyError(name)

    payloads = build_run_execution_config_payloads(
        cfg=_Cfg(),
        model_profile=SimpleNamespace(module_selectors={}),
    )

    assert payloads.auto_config == {
        "enabled": False,
        "tier": "balanced",
        "probes": 0,
        "target_pm_ratio": 2.0,
    }
    assert payloads.edit_config == {}
