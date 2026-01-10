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


def test_iter_transformer_layers_skips_non_iterable_llama_layers() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = object()

    assert list(R._iter_transformer_layers(Model())) == []


def test_iter_transformer_layers_skips_non_iterable_bert_layers() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = object()

    assert list(R._iter_transformer_layers(Model())) == []


def test_rmt_detect_with_names_skips_non_iterable_gpt2_layers() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = object()

    out = R.rmt_detect_with_names(Model(), threshold=1.5, verbose=False)
    assert out["n_layers_flagged"] == 0


def test_rmt_detect_with_names_skips_non_iterable_llama_layers() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = object()

    out = R.rmt_detect_with_names(Model(), threshold=1.5, verbose=False)
    assert out["n_layers_flagged"] == 0


def test_rmt_detect_with_names_skips_non_iterable_bert_layers() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = object()

    out = R.rmt_detect_with_names(Model(), threshold=1.5, verbose=False)
    assert out["n_layers_flagged"] == 0


def test_rmt_detect_skips_modules_without_2d_weights_when_suffix_matches() -> None:
    class _WeirdProj(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

    class _WeirdBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.c_proj = _WeirdProj()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([_WeirdBlock()])

    out = R.rmt_detect(_Model(), detect_only=True)
    assert out["n_layers_flagged"] == 0


def test_rmt_detect_partial_baseline_deadband_branch_sets_outlier(monkeypatch) -> None:
    model = _TinyModel(n_layers=1)

    def fake_layer_svd_stats(_layer, *_a, **_k):  # noqa: ANN001
        return {
            "sigma_min": 1.0,
            "sigma_max": 10.0,
            "worst_ratio": 10.0,
            "worst_details": {"name": "attn.c_proj", "s_max": 10.0},
        }

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    out = R.rmt_detect(
        model,
        threshold=1.5,
        detect_only=True,
        deadband=0.10,
        baseline_sigmas={"transformer.h.0.attn.c_proj": 1.0},
        baseline_mp_stats={},
    )
    assert out["has_outliers"] is True


def test_rmt_detect_omits_details_when_worst_details_missing(monkeypatch) -> None:
    model = _TinyModel(n_layers=1)

    def fake_layer_svd_stats(_layer, *_a, **_k):  # noqa: ANN001
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": 2.0,
        }

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    out = R.rmt_detect(model, threshold=1.5, detect_only=True)
    assert out["per_layer"] and "details" not in out["per_layer"][0]


def test_rmt_detect_prints_stalled_when_outliers_do_not_improve(monkeypatch, capsys) -> None:
    model = _TinyModel(n_layers=2)

    def fake_layer_svd_stats(_layer, *_a, **_k):  # noqa: ANN001
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": 2.0,
            "worst_details": {"name": "attn.c_proj", "s_max": 2.0},
        }

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    monkeypatch.setattr(R, "_apply_rmt_correction", lambda *_a, **_k: None)

    R.rmt_detect(
        model,
        threshold=1.5,
        detect_only=False,
        correction_factor=0.9,
        verbose=True,
        max_iterations=2,
    )
    out = capsys.readouterr().out
    assert "RMT correction stalled" in out


def test_rmt_detect_improving_path_with_verbose_false_emits_no_message(
    monkeypatch, capsys
) -> None:
    model = _TinyModel(n_layers=2)
    state = {"corrected": False}

    def fake_layer_svd_stats(_module, _bs=None, _bm=None, _name=""):  # noqa: ANN001
        ratio = 2.0
        if state["corrected"] and str(_name).endswith("transformer.h.1.attn.c_proj"):
            ratio = 1.0
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": ratio,
            "worst_details": {"name": "attn.c_proj", "s_max": 2.0},
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
        verbose=False,
        max_iterations=2,
    )
    out = capsys.readouterr().out
    assert "RMT correction improving" not in out


def test_rmt_detect_logs_more_layers_when_over_three_outliers(monkeypatch, capsys) -> None:
    model = _TinyModel(n_layers=5)

    def fake_layer_svd_stats(_layer, *_a, **_k):  # noqa: ANN001
        return {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "worst_ratio": 2.0,
            "worst_details": {"name": "attn.c_proj", "s_max": 2.0},
        }

    monkeypatch.setattr(R, "layer_svd_stats", fake_layer_svd_stats)
    R.rmt_detect(model, threshold=1.5, detect_only=True, verbose=True)
    out = capsys.readouterr().out
    assert "more layers flagged" in out


def test_rmt_detect_target_layers_handles_missing_named_modules(monkeypatch) -> None:
    class Model(_TinyModel):
        def named_modules(self, *_a, **_k):  # noqa: ANN001
            return iter([])

    model = Model(n_layers=1)
    monkeypatch.setattr(
        R, "layer_svd_stats", lambda *_a, **_k: {"sigma_min": 1.0, "sigma_max": 1.0, "worst_ratio": 1.0}
    )

    out = R.rmt_detect(model, target_layers=["transformer_layer_0"], detect_only=True)
    assert out["n_layers_flagged"] == 0


def test_apply_rmt_correction_scales_tied_parameters() -> None:
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4) * 100.0)

    tied = nn.Parameter(layer.weight.detach().clone())

    class Adapter:
        def get_tying_map(self):  # noqa: ANN001
            return {"layer.weight": ["tied.weight"]}

        def get_parameter_by_name(self, name: str):  # noqa: ANN001
            return tied if name == "tied.weight" else None

    before = tied.detach().clone()
    R._apply_rmt_correction(layer, factor=0.9, layer_name="layer", adapter=Adapter())
    assert torch.allclose(tied.detach(), before) is False
