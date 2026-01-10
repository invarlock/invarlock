import torch
import torch.nn as nn

import invarlock.guards.variance as variance_mod
from invarlock.guards.variance import VarianceGuard


class _AdapterDescribeDict:
    def describe(self, _model):  # noqa: ANN001
        return {"n_layer": 2}

    def get_layer_modules(self, _model, _i: int):  # noqa: ANN001
        return {
            "attn.c_proj": nn.Linear(2, 2, bias=False),
            "mlp.c_proj": nn.Linear(2, 2, bias=False),
        }


def test_refresh_calibration_defaults_coerces_non_dict_calibration() -> None:
    g = VarianceGuard(policy={"calibration": ["bad"]})
    calibration = g._policy.get("calibration")
    assert isinstance(calibration, dict)
    assert calibration["windows"] == 6
    assert calibration["min_coverage"] == 4


def test_compute_variance_scales_filters_raw_scales_via_scale_matches_target(
    monkeypatch,
) -> None:
    def fake_equalise(*_a, **_k):
        return {"block0.attn": 1.1}

    monkeypatch.setattr(variance_mod, "equalise_residual_variance", fake_equalise)

    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.0,
        "clamp": (0.5, 2.0),
        "min_abs_adjust": 0.0,
        "max_scale_step": 0.0,
        "topk_backstop": 0,
        # Focus includes the normalized ".c_proj" form.
        "target_modules": ["transformer.h.0.attn"],
        "max_adjusted_modules": 0,
    }
    g = VarianceGuard(policy=policy)
    # Use a slightly-mismatched key to force the fallback `_scale_matches_target` branch.
    g._target_modules = {"transformer.h.0.attn": nn.Linear(2, 2, bias=False)}
    monkeypatch.setattr(
        g, "_tensorize_calibration_batches", lambda batches: list(batches)
    )

    out = g._compute_variance_scales(nn.Linear(2, 2, bias=False), [])
    assert out.get("block0.attn") == 1.1


def test_resolve_target_modules_adapter_describe_dict_path_populates_targets() -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    targets = g._resolve_target_modules(nn.Module(), adapter=_AdapterDescribeDict())
    assert "transformer.h.0.attn.c_proj" in targets
    assert "transformer.h.1.mlp.c_proj" in targets


def test_compute_variance_scales_topk_backstop_injects_best_candidate(
    monkeypatch,
) -> None:
    def fake_equalise(*_a, **_k):
        return {"block0.attn": 1.3}

    monkeypatch.setattr(variance_mod, "equalise_residual_variance", fake_equalise)

    g = VarianceGuard(
        policy={
            "min_gain": 0.0,
            "scope": "both",
            "max_calib": 0,
            "deadband": 0.10,
            "clamp": (0.5, 2.0),
            "min_abs_adjust": 0.50,
            "max_scale_step": 0.0,
            "topk_backstop": 1,
            "max_adjusted_modules": 0,
        }
    )
    monkeypatch.setattr(
        g, "_tensorize_calibration_batches", lambda batches: list(batches)
    )

    out = g._compute_variance_scales(nn.Linear(2, 2, bias=False), [])
    assert out.get("block0.attn") == 1.3


def test_prepare_sets_focus_modules_stats_when_target_modules_policy_passed(
    monkeypatch,
) -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "calibration": {"windows": 0},
        }
    )
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda _m, _a: {"transformer.h.0.attn.c_proj": nn.Linear(2, 2, bias=False)},
    )

    g.prepare(
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=None,
        policy={"target_modules": ["transformer.h.0.attn"]},
    )
    assert g._stats.get("focus_modules") == ["transformer.h.0.attn.c_proj"]


def test_prepare_accepts_iterable_calibration_source(monkeypatch) -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 10,
            "calibration": {"windows": 0},
        }
    )
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda _m, _a: {"transformer.h.0.attn.c_proj": nn.Linear(2, 2, bias=False)},
    )

    seen: dict[str, int] = {}

    def fake_scales(_model, batches):  # noqa: ANN001
        seen["batches"] = len(batches)
        return {}

    monkeypatch.setattr(g, "_compute_variance_scales", fake_scales)
    calib = (torch.ones(1, 2) for _ in range(3))
    g.prepare(nn.Linear(2, 2, bias=False), adapter=None, calib=calib, policy=None)
    assert seen.get("batches") == 1


def test_prepare_monitor_mode_reason_insufficient_coverage(monkeypatch) -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "calibration": {"windows": 4, "min_coverage": 3, "seed": 7},
        }
    )
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda _m, _a: {"transformer.h.0.attn.c_proj": nn.Linear(2, 2, bias=False)},
    )
    monkeypatch.setattr(g, "_compute_variance_scales", lambda *_a, **_k: {"x": 1.2})

    def fake_ppl(*_a, **_k):
        return [10.0, 10.0], [2.0, 2.0], [100, 100]

    monkeypatch.setattr(g, "_compute_ppl_for_batches", fake_ppl)

    g.prepare(
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[torch.ones(1, 2) for _ in range(2)],
        policy=None,
    )
    assert g._predictive_gate_state.get("reason") == "insufficient_coverage"


def test_prepare_monitor_mode_reason_no_scales(monkeypatch) -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 9},
        }
    )
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda _m, _a: {"transformer.h.0.attn.c_proj": nn.Linear(2, 2, bias=False)},
    )
    monkeypatch.setattr(g, "_compute_variance_scales", lambda *_a, **_k: {})

    def fake_ppl(*_a, **_k):
        return [10.0, 10.0], [2.0, 2.0], [100, 100]

    monkeypatch.setattr(g, "_compute_ppl_for_batches", fake_ppl)

    g.prepare(
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[torch.ones(1, 2) for _ in range(2)],
        policy=None,
    )
    assert g._predictive_gate_state.get("reason") == "no_scales"


def test_prepare_monitor_mode_reason_ve_enable_failed(monkeypatch) -> None:
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 11},
        }
    )
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda _m, _a: {"transformer.h.0.attn.c_proj": nn.Linear(2, 2, bias=False)},
    )
    monkeypatch.setattr(g, "_compute_variance_scales", lambda *_a, **_k: {"x": 1.1})
    monkeypatch.setattr(g, "enable", lambda *_a, **_k: False)

    def fake_ppl(*_a, **_k):
        return [10.0, 10.0], [2.0, 2.0], [100, 100]

    monkeypatch.setattr(g, "_compute_ppl_for_batches", fake_ppl)

    g.prepare(
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[torch.ones(1, 2) for _ in range(2)],
        policy=None,
    )
    assert g._predictive_gate_state.get("reason") == "ve_enable_failed"
