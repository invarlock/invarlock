from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.guards import rmt as R


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(4, 4, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_TinyBlock() for _ in range(n_layers)])


def test_rmt_detect_prints_improving_when_outliers_drop(monkeypatch, capsys) -> None:
    model = _TinyModel(n_layers=2)
    state = {"corrected": False}

    def fake_layer_svd_stats(_module, _bs=None, _bm=None, _name=""):  # noqa: ANN001
        if not state["corrected"]:
            ratio = 2.0
        else:
            ratio = 2.0 if str(_name).endswith("transformer.h.1.attn.c_proj") else 1.0
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": ratio,
            "worst_details": {
                "name": "attn.c_proj",
                "s_max": 2.0,
                "normalization": "mp_bulk_edge",
            },
        }

    def fake_apply(*_a, **_k):  # noqa: ANN001
        state["corrected"] = True

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    monkeypatch.setattr(R, "_apply_rmt_correction", fake_apply)

    R.rmt_detect(
        model,
        threshold=1.5,
        detect_only=False,
        correction_factor=0.9,
        verbose=True,
        max_iterations=2,
    )
    out = capsys.readouterr().out
    assert "RMT correction improving" in out


def test_rmt_detect_with_names_verbose_prints_more_layers_flagged(monkeypatch, capsys) -> None:
    model = _TinyModel(n_layers=5)

    def fake_layer_svd_stats(_layer, *_a, **_k):  # noqa: ANN001
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": 2.0,
            "worst_details": {"name": "attn.c_proj", "s_max": 2.0},
        }

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    R.rmt_detect_with_names(model, threshold=1.5, verbose=True)
    out = capsys.readouterr().out
    assert "... and 2 more layers flagged" in out


def test_apply_rmt_correction_fallback_scaling_on_svd_failure(monkeypatch, capsys) -> None:
    layer = nn.Linear(4, 4, bias=False)
    before = layer.weight.detach().clone()

    def boom(*_a, **_k):  # noqa: ANN001
        raise torch.linalg.LinAlgError("svd fail")

    monkeypatch.setattr(torch.linalg, "svdvals", boom)
    R._apply_rmt_correction(layer, factor=0.9, layer_name="layer", verbose=True)
    out = capsys.readouterr().out
    assert "fallback scaling" in out

    after = layer.weight.detach()
    assert torch.allclose(after, before * 0.9)


def test_rmt_guard_finalize_hydrates_edge_risk_from_calibration_batches(monkeypatch) -> None:
    guard = R.RMTGuard()
    guard.prepared = True
    guard._calibration_batches = [object()]

    monkeypatch.setattr(
        guard,
        "_compute_activation_edge_risk",
        lambda *_a, **_k: {
            "edge_risk_by_family": {"attn": 0.1},
            "edge_risk_by_module": {"m": 0.2},
        },
    )
    monkeypatch.setattr(guard, "_compute_epsilon_violations", lambda: [])

    guard.finalize(nn.Linear(2, 2, bias=False), adapter=None)
    assert guard.edge_risk_by_family.get("attn") == 0.1
