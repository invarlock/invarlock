from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.guards.variance import VarianceGuard


@dataclass
class _DummyReport:
    meta: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    edit: Any = None


def test_variance_guard_handles_non_dict_base_calibration(monkeypatch):
    # Some probe/fixture environments may supply a malformed tier policy; the guard
    # should still be robust and merge caller calibration overrides.
    from invarlock.guards import policies as policies_mod

    real = policies_mod.get_variance_policy

    def fake(name: str = "balanced", *, use_yaml: bool = True):
        policy = dict(real(name, use_yaml=use_yaml))
        policy["calibration"] = None
        return policy

    monkeypatch.setattr(policies_mod, "get_variance_policy", fake)

    g = VarianceGuard(policy={"calibration": {"windows": 7}})
    assert g._policy["calibration"]["windows"] == 7


def test_variance_guard_allows_min_effect_lognll_none():
    g = VarianceGuard(policy={"min_effect_lognll": None})
    assert g._policy.get("min_effect_lognll") is None


def test_variance_guard_tap_none_defaults_to_mlp_projection():
    g = VarianceGuard(policy={"tap": None})
    assert g._tap_patterns == ["transformer.h.*.mlp.c_proj"]


def test_variance_guard_set_run_context_handles_empty_pairing_and_non_dict_edit():
    g = VarianceGuard()
    report = _DummyReport(
        context={
            "pairing_baseline": {
                "preview": {"window_ids": []},
                "final": {"window_ids": []},
            }
        },
        edit="not-a-dict",
    )
    g.set_run_context(report)
    assert g._pairing_reference == []
    assert g._pairing_digest is None


def test_variance_guard_set_run_context_handles_non_dict_deltas():
    g = VarianceGuard()
    report = _DummyReport(edit={"deltas": "not-a-dict"})
    g.set_run_context(report)
    assert g._params_changed is None
