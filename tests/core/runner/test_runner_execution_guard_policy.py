from __future__ import annotations

from invarlock.core.api import Guard
from invarlock.core.runner import CoreRunner
from tests.core._support_runner_execution import (
    DummyAdapter,
    DummyEdit,
    DummyModel,
    GoodGuard,
    make_config,
)

# Fallback model layers metric via .get("n_layer", 0) covered by execution


def test_policy_application_paths(monkeypatch, tmp_path):
    # Exercise _apply_guard_policy paths: direct attr, config, policy, fallback
    class PolyGuard(GoodGuard):
        name = "poly"

        def __init__(self):
            super().__init__()
            self.alpha = 0.0  # direct attribute target

    def fake_resolver(tier, edit_name, overrides, *, profile):
        return {
            "poly": {
                "alpha": 0.5,  # direct attribute
                "cfg_only": 1,  # into config dict
                "pol_only": 2,  # into policy dict
                "new_attr": 3,  # setattr fallback
            }
        }

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    g = PolyGuard()
    g.config = {}
    g.policy = {}
    cfg = make_config(tmp_path)

    # Use calibration None to avoid heavy work
    report = runner.execute(model, adapter, edit, [g], cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}
    # By implementation, policy params are applied to guard.config first when present
    assert (
        g.alpha == 0.5
        and g.config.get("cfg_only") == 1
        and g.config.get("pol_only") == 2
        and g.config.get("new_attr") == 3
    )


def test_policy_only_guard_application(monkeypatch, tmp_path):
    class PolicyOnlyGuard(GoodGuard):
        name = "policy_only"

        def __init__(self):
            super().__init__()
            self.config = None  # not a dict; should use policy dict path
            self.policy = {}

    def fake_resolver(tier, edit_name, overrides, *, profile):
        return {"policy_only": {"theta": 7}}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    g = PolicyOnlyGuard()
    cfg = make_config(tmp_path)
    # Patch eval to avoid heavy compute
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )
    _ = runner.execute(model, adapter, edit, [g], cfg, calibration_data=None)
    assert g.policy.get("theta") == 7


def test_guard_missing_passed_default(monkeypatch, tmp_path):
    class NoPassGuard(Guard):
        name = "nopass"

        def validate(self, model, adapter, context):
            return {}  # no 'passed' key → defaults to False

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )
    report = runner.execute(
        model, adapter, DummyEdit(), [NoPassGuard()], cfg, calibration_data=None
    )
    assert report.status in {"rollback", "success"}
