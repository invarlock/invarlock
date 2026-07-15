from __future__ import annotations

from types import SimpleNamespace

import torch

from invarlock.guards.spectral_correction import run_correction_lifecycle


class _BrokenDigestTensor(torch.Tensor):
    @staticmethod
    def __new__(cls) -> _BrokenDigestTensor:
        return torch.Tensor._make_subclass(cls, torch.ones(2, 2))

    def detach(self) -> torch.Tensor:
        raise RuntimeError("digest materialization failed")


def test_disabled_correction_blocks_when_weight_digest_cannot_be_materialized() -> None:
    guard = SimpleNamespace(
        correction_enabled=False,
        correction_cap_ratio=2.0,
        baseline_sigmas={"layer": 1.0},
        latest_z_scores={"layer": 3.0},
        latest_degeneracy={},
        _get_scoped_modules=lambda _model: [
            ("layer", SimpleNamespace(weight=_BrokenDigestTensor()))
        ],
    )

    final_metrics, ledger = run_correction_lifecycle(
        guard,
        object(),
        phase="validate",
        pre_correction_metrics={"layer": 2.0},
        selected_violations=[{"type": "sigma_drift", "module": "layer"}],
        multiple_testing_selection={"method": "bonferroni"},
    )

    assert final_metrics == {"layer": 2.0}
    assert ledger["policy_result"] == "evidence_incomplete"
    assert ledger["corrections"][0]["outcome"] == "evidence_missing"
    assert ledger["corrections"][0]["pre_weight_digest"] is None


def test_enabled_correction_records_control_failure_and_blocks() -> None:
    weight = torch.eye(2)
    guard = SimpleNamespace(
        correction_enabled=True,
        correction_cap_ratio=1.0,
        sigma_quantile=0.95,
        scope="all",
        baseline_sigmas={"layer": 1.0},
        target_sigma=1.0,
        latest_z_scores={"layer": 3.0},
        latest_degeneracy={},
        _get_scoped_modules=lambda _model: [("layer", SimpleNamespace(weight=weight))],
        _capture_sigmas=lambda _model, phase: {"layer": 2.0},
    )

    _, ledger = run_correction_lifecycle(
        guard,
        object(),
        phase="finalize",
        pre_correction_metrics={"layer": 2.0},
        selected_violations=[{"type": "sigma_drift", "module": "layer"}],
        multiple_testing_selection={},
        apply_spectral_control_fn=lambda _model, policy: {
            "corrections": [],
            "cap_result": {"failed_modules": [["layer", "cap failed"]]},
        },
    )

    correction = ledger["corrections"][0]
    assert ledger["policy_result"] == "correction_failed"
    assert correction["outcome"] == "correction_failed"
    assert correction["failure"] == "cap failed"
    assert correction["mutation_applied"] is False
