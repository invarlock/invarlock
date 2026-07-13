from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from invarlock.core.api import EditRuntime, RunReport
from invarlock.core.assurance_contract import (
    build_assurance_section,
    strict_report_policy_errors,
)
from invarlock.edits.quant_rtn import RTNQuantEdit
from invarlock.guards.variance import VarianceGuard
from invarlock.reporting.guards_variance import _extract_variance_analysis
from invarlock.reporting.run_report_payloads import build_guard_entries
from tests.core._support_assurance_contract import _plugin_metadata, strict_report


class _TinyEditedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([block])
        with torch.no_grad():
            block.mlp.c_proj.weight.copy_(
                torch.tensor([[0.501, 0.003], [0.002, 0.499]])
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.transformer.h[0].mlp.c_proj(inputs)


def _run_context(*, edit: dict[str, Any]) -> RunReport:
    return RunReport(
        meta={"model_id": "strict-model", "seed": 123},
        edit=edit,
        context={
            "dataset_meta": {
                "dataset_hash": "strict-dataset",
                "tokenizer_hash": "strict-tokenizer",
            },
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1, 2, 3]},
                "final": {"window_ids": [4, 5, 6, 7]},
            },
        },
    )


def test_variance_gain_lifecycle_restores_real_edit_and_passes_strict_verifier() -> (
    None
):
    model = _TinyEditedModel()
    target = model.transformer.h[0].mlp.c_proj
    original_weight = target.weight.detach().clone()
    batches = [
        {
            "inputs": torch.full((1, 2), 1.5),
            "window_id": (f"preview::{index}" if index < 4 else f"final::{index}"),
        }
        for index in range(8)
    ]
    policy = {
        "scope": "both",
        "min_gain": 0.0,
        "min_rel_gain": 0.001,
        "tie_breaker_deadband": 0.005,
        "min_effect_lognll": 0.001,
        "predictive_gate": True,
        "predictive_one_sided": True,
        "mode": "ci",
        "seed": 123,
        "max_calib": 80,
        "max_scale_step": 0.5,
        "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
    }
    guard = VarianceGuard(policy)

    guard.set_run_context(_run_context(edit={}))
    prepared = guard.prepare(model, adapter=None, calib=batches, policy=None)
    guard.before_edit(model)

    adapter = SimpleNamespace(describe=lambda _model: {"n_layer": 1})
    edit = RTNQuantEdit(scope="ffn", max_modules=1).apply(
        model,
        adapter,
        runtime=EditRuntime(profile="ci", include_runtime_debug=False),
    )
    edited_weight = target.weight.detach().clone()
    assert edit["deltas"]["params_processed"] == target.weight.numel()
    assert 0 < edit["deltas"]["params_changed"] <= target.weight.numel()
    assert not torch.equal(edited_weight, original_weight)

    guard.set_run_context(_run_context(edit=edit))
    guard.after_edit(model)
    assert torch.equal(target.weight, edited_weight)
    assert guard._scales
    applied_scale = next(iter(guard._scales.values()))
    assert applied_scale > 1.0

    result = guard.finalize(model)
    assert prepared["ready"] is True
    assert result["passed"] is True
    assert result["warnings"] == []
    assert result["errors"] == []
    assert result["metrics"]["predictive_gate"]["reason"] == "ci_gain_met"
    assert result["metrics"]["ve_enabled_during_validation"] is True
    assert result["metrics"]["ve_enabled"] is False
    assert result["metrics"]["subject_restored_after_ab"] is True
    assert torch.equal(target.weight, edited_weight)
    event_kinds = [event["kind"] for event in guard.diagnostic_records]
    assert "checkpoint_pushed" in event_kinds
    assert "checkpoint_popped" in event_kinds

    [variance_entry] = build_guard_entries(
        {
            "variance": {
                **result,
                "policy": guard.policy(),
                "supported": True,
                "violations": [],
            }
        }
    )
    report = strict_report()
    report["context"].update(_run_context(edit=edit).context)
    report["meta"].update(
        model_id="strict-model", seed=123, tokenizer_hash="strict-tokenizer"
    )
    report["edit"] = edit
    report["structure"] = dict(edit["deltas"])
    report["plugins"]["edit"] = _plugin_metadata("edits", "quant_rtn")
    variance_index = next(
        index
        for index, entry in enumerate(report["guards"])
        if entry["name"] == "variance"
    )
    report["guards"][variance_index] = variance_entry
    report["variance"] = {
        **_extract_variance_analysis({"guards": [variance_entry]}),
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
    }
    report["resolved_policy"]["variance"] = guard.policy()
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)
    assert errors == [], "\n".join(errors)
