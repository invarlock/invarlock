from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.core.assurance_contract import (
    build_assurance_section,
    strict_report_policy_errors,
)
from invarlock.core.run_policy import build_run_context_payload
from invarlock.edits import NoopEdit
from invarlock.guards.variance import VarianceGuard
from invarlock.reporting.guards_variance import _extract_variance_analysis
from tests.core._support_assurance_contract import _plugin_metadata, strict_report


class _TinyNoopModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([block])

    def forward(self, inputs, labels=None):  # noqa: ANN001
        _ = labels
        return self.transformer.h[0].mlp.c_proj(inputs)


def test_real_variance_finalize_shape_passes_strict_assurance(monkeypatch) -> None:
    report = strict_report()
    variance_fixture = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    pairing_digest = variance_fixture["metrics"]["ab_provenance"]["condition_a"][
        "pairing_digest"
    ]
    window_ids = list(
        variance_fixture["metrics"]["ab_provenance"]["condition_a"]["window_ids"]
    )
    delta_log = math.log(98.0) - math.log(100.0)
    condition_common = {
        "tag": "post_edit",
        "window_ids": window_ids,
        "window_count": 8,
        "target_fingerprint": "strict-target",
        "pairing_digest": pairing_digest,
        "consumed_pairing_digest": pairing_digest,
        "dataset_hash": "strict-dataset",
        "tokenizer_hash": "strict-tokenizer",
        "model_id": "strict-model",
        "run_seed": 123,
        "status": "evaluated",
    }
    guard = VarianceGuard(
        {
            "min_gain": 0.0,
            "scope": "both",
            "max_calib": 8,
            "tie_breaker_deadband": 0.005,
            "min_rel_gain": 0.001,
            "min_effect_lognll": 0.005,
            "predictive_one_sided": True,
            "seed": 123,
            "mode": "ci",
            "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
        }
    )
    guard._prepared = True
    guard._post_edit_evaluated = True
    guard._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    guard._scales = {"transformer.h.0.mlp.c_proj": 1.03}
    guard._ppl_no_ve = 100.0
    guard._ppl_with_ve = 98.0
    guard._ab_gain = 0.02
    guard._ratio_ci = (0.98, 0.98)
    guard._ab_windows_used = 8
    guard._ab_seed_used = 123
    guard._calibration_stats = {
        "status": "complete",
        "requested": 8,
        "coverage": 8,
        "min_coverage": 6,
        "seed": 123,
    }
    guard._predictive_gate_state = {
        "evaluated": True,
        "passed": True,
        "reason": "ci_gain_met",
        "delta_ci": [delta_log, delta_log],
        "gain_ci": [-delta_log, -delta_log],
        "mean_delta": delta_log,
    }
    guard._raw_scales_pre_edit = {"transformer.h.0.mlp.c_proj": 1.1}
    guard._raw_scales_post_edit = {"transformer.h.0.mlp.c_proj": 1.1}
    guard._stats = {
        "target_module_names": ["transformer.h.0.mlp.c_proj"],
        "proposed_scales_pre_edit": {"transformer.h.0.mlp.c_proj": 1.03},
        "proposed_scales_post_edit": {"transformer.h.0.mlp.c_proj": 1.03},
        "raw_scales_pre_edit_normalized": {"transformer.h.0.mlp.c_proj": 1.1},
        "raw_scales_post_edit_normalized": {"transformer.h.0.mlp.c_proj": 1.1},
        "predictive_gate": dict(guard._predictive_gate_state),
        "calibration": {"window_ids": window_ids},
        "target_fingerprint": "strict-target",
        "pairing_reference": {"digest": pairing_digest},
        "dataset_meta": {
            "dataset_hash": "strict-dataset",
            "tokenizer_hash": "strict-tokenizer",
        },
        "ab_provenance": {
            "condition_a": {**condition_common, "mode": "edited_no_ve"},
            "condition_b": {**condition_common, "mode": "virtual_ve"},
        },
        "ab_point_estimates": {
            "tag": "post_edit",
            "ppl_no_ve": 100.0,
            "ppl_with_ve": 98.0,
            "coverage": 8,
        },
        "ab_measurements": {
            "window_ids": list(window_ids),
            "condition_a": {
                "ppl": [100.0] * 8,
                "log_loss": [math.log(100.0)] * 8,
                "token_counts": [16] * 8,
            },
            "condition_b": {
                "ppl": [98.0] * 8,
                "log_loss": [math.log(98.0)] * 8,
                "token_counts": [16] * 8,
            },
            "ratio_bootstrap": {
                "method": "percentile_mean_ppl_ratio",
                "replicates": 500,
                "alpha": 0.05,
                "seed": 123,
            },
            "delta_log_bootstrap": {
                "method": "bca_paired_delta_log",
                "replicates": 500,
                "alpha": 0.05,
                "seed": 334,
                "weights": "condition_a_token_counts",
            },
            "ratio_ci": [0.98, 0.98],
            "delta_log_ci": [delta_log, delta_log],
        },
    }

    def enable(_model):
        guard._enabled = True
        return True

    def disable(_model):
        guard._enabled = False
        guard._last_restore_exact = True
        return True

    monkeypatch.setattr(guard, "_evaluate_ab_gate", lambda: (True, "criteria_met"))
    monkeypatch.setattr(guard, "enable", enable)
    monkeypatch.setattr(guard, "disable", disable)

    result = guard.finalize(nn.Linear(2, 2))

    assert result["passed"] is True
    assert result["metrics"]["ve_enabled_during_validation"] is True
    assert result["metrics"]["ve_enabled"] is False
    assert result["metrics"]["subject_restored_after_ab"] is True

    raw_guard = {
        **result,
        "name": "variance",
        "supported": True,
        "decision": "allow",
        "violations": [],
        "policy": dict(guard._policy),
    }
    report["edit"] = {"name": "quant_rtn"}
    report["plugins"]["edit"] = _plugin_metadata("edits", "quant_rtn")
    report["structure"]["params_changed"] = 123
    variance_index = next(
        index
        for index, entry in enumerate(report["guards"])
        if entry["name"] == "variance"
    )
    report["guards"][variance_index] = raw_guard
    report["variance"] = {
        **_extract_variance_analysis({"guards": [raw_guard]}),
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
    }
    report["resolved_policy"]["variance"] = dict(guard._policy)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)
    assert errors == [], "\n".join(errors)


def test_real_noop_pipeline_passes_strict_variance_assurance() -> None:
    model = _TinyNoopModel()
    report = strict_report()
    batches = [
        {"inputs": torch.ones(1, 2), "window_id": "preview::0"},
        {"inputs": torch.zeros(1, 2), "window_id": "preview::1"},
    ]
    guard = VarianceGuard(
        {
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "seed": 123,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 123},
        }
    )
    pairing_schedule = {
        phase: {"window_ids": list(report["evaluation_windows"][phase]["window_ids"])}
        for phase in ("preview", "final")
    }
    model_profile = SimpleNamespace(
        family="tiny",
        default_loss="ppl_causal",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )
    context = build_run_context_payload(
        cfg=SimpleNamespace(
            model=SimpleNamespace(id="strict-model"),
            eval={},
            dataset={},
            guards={
                "spectral": {},
                "rmt": {},
                "variance": {},
                "invariants": {},
            },
            context={},
        ),
        profile="release",
        pairing_schedule=pairing_schedule,
        seed_bundle={"python": 123},
        plugin_provenance={},
        run_id="strict-run",
        baseline_report_data=None,
        pm_acceptance_range=None,
        pm_drift_band=None,
        guard_metric_degradation_limit=0.0,
        model_profile=model_profile,
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
        to_serialisable_dict_fn=lambda value: (
            dict(value)
            if isinstance(value, dict)
            else {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        ),
    )
    context["dataset_meta"] = {
        "dataset_hash": "strict-dataset",
        "tokenizer_hash": "strict-tokenizer",
    }

    def run_report(edit: dict) -> SimpleNamespace:
        return SimpleNamespace(
            meta={"config": {}, "seed": 123},
            context=context,
            edit=edit,
        )

    guard.set_run_context(run_report({}))
    prepared = guard.prepare(model, adapter=None, calib=batches, policy=None)
    guard.before_edit(model)
    edit = NoopEdit().apply(
        model,
        SimpleNamespace(describe=lambda _model: {"family": "tiny"}),
    )
    guard.set_run_context(run_report(edit))
    guard.after_edit(model)
    result = guard.finalize(model)

    assert prepared["ready"] is True
    assert result["passed"] is True
    assert result["warnings"] == []
    assert result["errors"] == []
    assert result["metrics"]["predictive_gate"]["reason"] == ("no_adjustment_required")
    assert result["metrics"]["ab_provenance"]["condition_a"]["model_id"] == (
        "strict-model"
    )
    assert result["metrics"]["ab_provenance"]["condition_b"]["model_id"] == (
        "strict-model"
    )

    raw_guard = {
        **result,
        "name": "variance",
        "supported": True,
        "decision": "allow",
        "violations": [],
        "policy": dict(guard._policy),
    }
    report["context"].update(context)
    report["meta"].update(
        model_id="strict-model", seed=123, tokenizer_hash="strict-tokenizer"
    )
    report["edit"] = edit
    report["structure"] = dict(edit["deltas"])
    variance_index = next(
        index
        for index, entry in enumerate(report["guards"])
        if entry["name"] == "variance"
    )
    report["guards"][variance_index] = raw_guard
    report["variance"] = {
        **_extract_variance_analysis({"guards": [raw_guard]}),
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
    }
    report["resolved_policy"]["variance"] = dict(guard._policy)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)
    assert errors == [], "\n".join(errors)
