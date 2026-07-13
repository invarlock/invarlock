from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import invarlock.guards.variance as variance_mod
from invarlock.guards.variance import VarianceGuard


@dataclass
class _DummyReport:
    meta: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    edit: Any = None


def test_variance_guard_handles_non_dict_base_calibration(monkeypatch):
    # Some probe/fixture environments may supply a malformed tier policy; the guard
    # should still be robust and merge caller calibration overrides.
    import invarlock.guards.policies as policies_mod

    real = policies_mod.get_variance_policy

    def fake(name: str = "balanced"):
        policy = dict(real(name))
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


def test_prepare_treats_uniterable_calibration_as_missing(monkeypatch):
    class _BadCalibration:
        def __iter__(self):
            raise TypeError("not iterable")

    g = VarianceGuard(policy={"max_calib": 10, "calibration": {"windows": 1}})
    model = torch.nn.Linear(1, 1)

    monkeypatch.setattr(g, "_resolve_target_modules", lambda *_args: {"linear": model})

    result = g.prepare(model, calib=_BadCalibration())

    assert result["ready"] is True
    assert g._calibration_stats["status"] == "insufficient"
    assert g._calibration_stats["coverage"] == 0


def test_finalize_ignores_evidence_dump_failures(monkeypatch):
    import invarlock.core.guard_evidence as guard_evidence

    expected = {"passed": True, "decision": "allow"}
    monkeypatch.setattr(
        variance_mod._variance_runtime,
        "finalize_guard",
        lambda *_args: expected,
    )
    monkeypatch.setattr(
        guard_evidence,
        "maybe_dump_guard_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dump failed")),
    )

    assert VarianceGuard().finalize(torch.nn.Linear(1, 1)) is expected
