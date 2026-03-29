from __future__ import annotations

import torch

from invarlock.guards import rmt_activation_runtime as runtime
from invarlock.guards import rmt_result_contract


class IndexedSource:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> int:
        return idx


class IterableOnly:
    def __iter__(self):
        return iter([10, 20, 30])


def test_collect_calibration_batches_supports_index_and_iterable_sources() -> None:
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        3,
        activation_sampling={"windows": {"indices_policy": "last"}},
    ) == [2, 3, 4]
    assert runtime.collect_calibration_batches(
        IterableOnly(),
        2,
    ) == [10, 20]
    assert runtime.collect_calibration_batches(object(), 2) == []


def test_prepare_activation_inputs_normalizes_and_falls_back_to_clone(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is not None
    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)


def test_prepare_and_after_edit_result_contract_helpers() -> None:
    prepare = rmt_result_contract.build_prepare_result(
        ready=True,
        baseline_metrics={"edge_risk_by_family": {"attn": 0.2}},
        policy_applied={"activation_required": True},
        preparation_time=1.25,
    )
    assert prepare == {
        "ready": True,
        "baseline_metrics": {"edge_risk_by_family": {"attn": 0.2}},
        "policy_applied": {"activation_required": True},
        "preparation_time": 1.25,
    }

    failed = rmt_result_contract.build_prepare_result(
        ready=False,
        baseline_metrics={},
        policy_applied={},
        preparation_time=0.5,
        error="Activation baseline unavailable",
    )
    assert failed["error"] == "Activation baseline unavailable"

    after = rmt_result_contract.build_after_edit_result(
        edge_risk_by_module={"layer": 0.2},
        edge_risk_by_family={"attn": 0.2},
        token_weight_total=12,
        batches_used=3,
    )
    assert after == {
        "analysis_source": "activations_edge_risk",
        "edge_risk_by_module": {"layer": 0.2},
        "edge_risk_by_family": {"attn": 0.2},
        "token_weight_total": 12,
        "batches_used": 3,
    }
