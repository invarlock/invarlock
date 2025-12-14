from __future__ import annotations

import torch

from invarlock.guards.spectral import SpectralGuard


class _TinySpectralModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(1, 1, bias=False)
        self.attn = torch.nn.Linear(1, 1, bias=False)
        self.embed = torch.nn.Embedding(1, 1)
        self.proj = torch.nn.Linear(1, 1, bias=False)


def _set_scalar_weight(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            module.weight.fill_(float(value))


def _prepare_guard(model: torch.nn.Module, guard: SpectralGuard) -> None:
    baseline_family_stats = {
        "ffn": {"mean": 1.0, "std": 1.0},
        "attn": {"mean": 1.0, "std": 1.0},
        "embed": {"mean": 1.0, "std": 1.0},
        "other": {"mean": 1.0, "std": 1.0},
    }
    guard.prepare(
        model=model,
        adapter=None,
        calib=None,
        policy={"baseline_family_stats": baseline_family_stats},
    )


def _run_case(method: str, *, max_caps: int = 10) -> dict[str, object]:
    model = _TinySpectralModel()
    # Baseline (mean=1.0)
    _set_scalar_weight(model.mlp, 1.0)
    _set_scalar_weight(model.attn, 1.0)
    _set_scalar_weight(model.embed, 1.0)
    _set_scalar_weight(model.proj, 1.0)

    guard = SpectralGuard(
        scope="all",
        deadband=0.0,
        max_caps=max_caps,
        family_caps={"ffn": 1.0, "attn": 1.0, "embed": 1.0, "other": 1.0},
        multiple_testing={"method": method, "alpha": 0.05, "m": 4},
    )
    _prepare_guard(model, guard)

    # Edited weights: sigma = mean + z, so z = sigma - mean (std=1.0).
    # P-values (two-sided): z=2.576→~0.01, 2.326→~0.02, 2.17→~0.03, 1.8→~0.072
    _set_scalar_weight(model.mlp, 1.0 + 2.576)  # ffn
    _set_scalar_weight(model.attn, 1.0 + 2.326)  # attn
    _set_scalar_weight(model.embed, 1.0 + 2.17)  # embed
    _set_scalar_weight(model.proj, 1.0 + 1.8)  # other

    return guard.validate(model=model, adapter=None, context={})


def test_spectral_multiple_testing_changes_selected_violations() -> None:
    bh = _run_case("bh")
    bonf = _run_case("bonferroni")

    bh_modules = {v.get("module") for v in bh.get("violations", [])}  # type: ignore[union-attr]
    bonf_modules = {v.get("module") for v in bonf.get("violations", [])}  # type: ignore[union-attr]

    assert bh_modules == {"mlp", "attn", "embed"}
    assert bonf_modules == {"mlp"}

    bh_mt = (bh.get("policy") or {}).get("multiple_testing")  # type: ignore[union-attr]
    assert isinstance(bh_mt, dict)
    assert bh_mt.get("method") == "bh"
    assert bh_mt.get("alpha") == 0.05
    assert bh_mt.get("m") == 4


def test_spectral_max_caps_applied_after_multiple_testing() -> None:
    # With max_caps=2: BH selects 3 families -> abort; Bonferroni selects 1 -> warn.
    bh = _run_case("bh", max_caps=2)
    bonf = _run_case("bonferroni", max_caps=2)

    assert bh.get("action") == "abort"
    assert bh.get("passed") is False

    assert bonf.get("action") == "warn"
    assert bonf.get("passed") is True
