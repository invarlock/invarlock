import pytest
import torch

import invarlock.guards as guards_pkg
import invarlock.guards.spectral as spectral_guard
import invarlock.guards.spectral_detection as spectral_detection
import invarlock.guards.spectral_measurement as spectral_measurement
import invarlock.guards.spectral_policy as spectral_policy
import invarlock.guards.spectral_runtime as spectral_runtime
from invarlock.core.exceptions import ValidationError

guards_pkg.spectral = spectral_guard
guards_pkg.spectral_detection = spectral_detection
guards_pkg.spectral_measurement = spectral_measurement


class _TinyModel:
    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        self._modules = modules

    def named_modules(self):
        yield from self._modules.items()


class _TinyWeightModule(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight


def test_normalize_family_caps_numeric_and_default_false():
    # Numeric shorthand becomes mapping with kappa
    caps = spectral_policy.normalize_family_caps({"ffn": 3.3})
    assert caps["ffn"]["kappa"] == 3.3
    # default=False yields empty mapping for invalid input
    assert spectral_policy.normalize_family_caps(None, default=False) == {}


def test_compute_sigma_max_quantized_int8_skips():
    W = torch.zeros(2, 2, dtype=torch.int8)
    assert spectral_measurement.compute_sigma_max(W) == 1.0


def test_should_process_module_scope_ffn_proj():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.zeros(2, 2)

    m = Mod()
    assert (
        spectral_detection.should_process_module("layer.c_proj", m, "ffn+proj") is True
    )
    assert (
        spectral_detection.should_process_module("layer.attn.c_proj", m, "attn") is True
    )


def test_spectral_prepare_with_aliases(monkeypatch):
    # Patch heavy functions to avoid real tensor ops
    monkeypatch.setattr(
        spectral_measurement,
        "capture_baseline_sigmas",
        lambda *a, **k: {"m": 1.0},
    )
    monkeypatch.setattr(
        spectral_detection,
        "classify_model_families",
        lambda *a, **k: {"m": "ffn"},
    )
    monkeypatch.setattr(
        spectral_detection,
        "compute_family_stats",
        lambda *a, **k: {"ffn": {"mean": 1.0, "std": 0.0}},
    )
    monkeypatch.setattr(spectral_measurement, "scan_model_gains", lambda *a, **k: {})
    monkeypatch.setattr(spectral_measurement, "auto_sigma_target", lambda *a, **k: 1.0)

    g = spectral_guard.SpectralGuard()

    class DummyModel:
        def named_modules(self):
            return iter([])

    policy = {
        "sigma_quantile": 0.9,
        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
        "estimator": {"iters": 1, "init": "e0"},
        "degeneracy": {
            "enabled": True,
            "stable_rank": {"warn_ratio": 0.75, "fatal_ratio": 0.5},
            "norm_collapse": {"warn_ratio": 0.5, "fatal_ratio": 0.25},
        },
        "baseline_family_stats": {"ffn": {"mean": 1.0, "std": 0.0}},
    }
    out = g.prepare(DummyModel(), object(), None, policy)
    assert out["ready"] is True
    assert g.config["sigma_quantile"] == 0.9
    assert "multiple_testing" in g.config
    assert g.estimator["iters"] == 1
    assert g.estimator["init"] == "e0"


def test_spectral_prepare_estimator_invalid_policy_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        spectral_measurement,
        "capture_baseline_sigmas",
        lambda *a, **k: {"m": 1.0},
    )
    monkeypatch.setattr(
        spectral_detection,
        "classify_model_families",
        lambda *a, **k: {"m": "ffn"},
    )
    monkeypatch.setattr(
        spectral_detection,
        "compute_family_stats",
        lambda *a, **k: {"ffn": {"mean": 1.0, "std": 0.0}},
    )
    monkeypatch.setattr(spectral_measurement, "scan_model_gains", lambda *a, **k: {})
    monkeypatch.setattr(spectral_measurement, "auto_sigma_target", lambda *a, **k: 1.0)

    g = spectral_guard.SpectralGuard()

    class DummyModel:
        def named_modules(self):
            return iter([])

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        g.prepare(
            DummyModel(),
            object(),
            None,
            {"estimator": {"iters": "bad", "init": "bad"}},
        )


def test_spectral_prepare_coerces_max_spectral_norm_from_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        spectral_measurement,
        "capture_baseline_sigmas",
        lambda *a, **k: {"m": 1.0},
    )
    monkeypatch.setattr(
        spectral_detection,
        "classify_model_families",
        lambda *a, **k: {"m": "ffn"},
    )
    monkeypatch.setattr(
        spectral_detection,
        "compute_family_stats",
        lambda *a, **k: {"ffn": {"mean": 1.0, "std": 0.0}},
    )
    monkeypatch.setattr(spectral_measurement, "scan_model_gains", lambda *a, **k: {})

    g = spectral_guard.SpectralGuard()

    class DummyModel:
        def named_modules(self):
            return iter([])

    out = g.prepare(DummyModel(), object(), None, {"max_spectral_norm": "2.5"})
    assert out["ready"] is True
    assert g.max_spectral_norm == 2.5
    assert g.config["max_spectral_norm"] == 2.5


def test_spectral_prepare_percentile_failure_falls_back_to_sigma_quantile(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        spectral_measurement,
        "capture_baseline_sigmas",
        lambda *a, **k: {"m": 1.0},
    )
    monkeypatch.setattr(
        spectral_detection,
        "classify_model_families",
        lambda *a, **k: {"m": "ffn"},
    )
    monkeypatch.setattr(
        spectral_detection,
        "compute_family_stats",
        lambda *a, **k: {"ffn": {"mean": 1.0, "std": 0.0}},
    )
    monkeypatch.setattr(spectral_measurement, "scan_model_gains", lambda *a, **k: {})
    monkeypatch.setattr(spectral_runtime.np, "percentile", lambda *_a, **_k: 1 / 0)

    g = spectral_guard.SpectralGuard(sigma_quantile=0.9)

    class DummyModel:
        def named_modules(self):
            return iter([])

    out = g.prepare(DummyModel(), object(), None, {})
    assert out["ready"] is True
    assert g.target_sigma == pytest.approx(g.sigma_quantile)


def test_spectral_guard_init_rejects_invalid_policy_and_accepts_valid_degeneracy() -> (
    None
):
    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_guard.SpectralGuard(
            estimator={"iters": "bad", "init": "bad"}, degeneracy="bad"
        )

    with pytest.raises(ValidationError, match="POLICY-PARAM-INVALID"):
        spectral_guard.SpectralGuard(estimator={"iters": -1, "init": "e0"})

    g_dict = spectral_guard.SpectralGuard(degeneracy={"enabled": False})
    assert g_dict.degeneracy["enabled"] is False


def test_spectral_set_run_context_captures_profile() -> None:
    g = spectral_guard.SpectralGuard()
    report = type("R", (), {"context": {"profile": "CI"}})()
    g.set_run_context(report)
    assert g._run_profile == "ci"
    report2 = type("R2", (), {"context": ["not", "dict"]})()
    g.set_run_context(report2)
    assert g._run_profile is None


def test_compute_sigma_max_additional_branches(monkeypatch) -> None:
    assert (
        spectral_measurement.compute_sigma_max("not_a_tensor", iters="bad", init="bad")
        == 1.0
    )
    assert spectral_measurement.compute_sigma_max(torch.empty((0, 3)), iters=1) == 0.0
    assert spectral_measurement.compute_sigma_max(torch.zeros(3), iters=1) == 0.0

    sigma = spectral_measurement.compute_sigma_max(torch.eye(2), iters=0, init="bad")
    assert sigma > 0.0

    monkeypatch.setattr(
        spectral_measurement,
        "power_iter_sigma_max",
        lambda *_a, **_k: 1 / 0,
    )
    assert spectral_measurement.compute_sigma_max(torch.eye(2), iters=1) == 1.0

    class BadNdim:
        @property
        def ndim(self):
            raise TypeError("bad ndim")

    assert spectral_measurement._is_matrix_weight(BadNdim()) is False
    assert spectral_measurement._scalarize_stat(1.25) == 1.25


def test_classify_module_family_moe_and_module_type_branches() -> None:
    linear = torch.nn.Linear(2, 2)
    assert spectral_detection.classify_module_family("router.gate", linear) == "router"
    assert (
        spectral_detection.classify_module_family("experts.block", linear)
        == "expert_ffn"
    )
    assert (
        spectral_detection.classify_module_family("layer.attn.c_proj", linear) == "attn"
    )
    assert spectral_detection.classify_module_family("layer.mlp.c_fc", linear) == "ffn"
    assert (
        spectral_detection.classify_module_family("layer.mlp.gate_proj", linear)
        == "ffn"
    )
    assert (
        spectral_detection.classify_module_family(
            "model.layers.0.mlp.gate_up_proj", linear
        )
        == "ffn"
    )
    assert (
        spectral_detection.classify_module_family(
            "model.layers.0.mamba.in_proj", linear
        )
        == "ffn"
    )
    assert (
        spectral_detection.classify_module_family(
            "model.layers.0.mamba.out_proj", linear
        )
        == "ffn"
    )
    assert (
        spectral_detection.classify_module_family(
            "model.layers.0.block_sparse_moe.gate", linear
        )
        == "router"
    )

    # module type based embedding classification
    assert (
        spectral_detection.classify_module_family(
            "layer.not_named_embed", torch.nn.Embedding(2, 2)
        )
        == "embed"
    )

    class _WeightOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros(2, 2)

    m = _WeightOnly()
    assert (
        spectral_detection.should_process_module("any.name", m, "unknown-scope") is True
    )


def test_classify_module_family_uses_embedding_module_type_without_name_hint() -> None:
    assert (
        spectral_detection.classify_module_family(
            "plain.module", torch.nn.Embedding(2, 2)
        )
        == "embed"
    )


def test_spectral_detection_family_iteration_and_unknown_scope_branches() -> None:
    class Conv1DModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros(2, 2)

    assert (
        spectral_detection.classify_module_family("layer.attn.proj", Conv1DModule())
        == "attn"
    )
    assert (
        spectral_detection.classify_module_family("layer.mlp.proj", Conv1DModule())
        == "ffn"
    )

    model = _TinyModel(
        {
            "keep": _TinyWeightModule(torch.eye(2)),
            "skip": _TinyWeightModule(torch.eye(2)),
        }
    )
    families = spectral_detection.classify_model_families(
        model,
        existing={"seed": "other"},
        should_process_module_fn=lambda name, *_args: name != "skip",
        classify_module_family_fn=lambda *_args: "ffn",
    )
    assert families == {"seed": "other", "keep": "ffn"}

    guard = type("Guard", (), {"scope": "unexpected"})()
    assert spectral_detection.should_check_module(
        guard, "plain.name", _TinyWeightModule(torch.eye(2))
    )


def test_spectral_detection_summary_and_degeneracy_error_branches(
    monkeypatch,
) -> None:
    summary = spectral_detection.summarize_family_z_scores(
        {"m": 1.0},
        {"m": "ffn"},
        {},
    )
    assert summary["ffn"]["violations"] == 0
    assert "kappa" not in summary["ffn"]

    stats = spectral_detection.compute_family_stats({"m": 2.0}, {"m": "ffn"})
    assert stats["ffn"]["max"] == 2.0

    guard = spectral_guard.SpectralGuard(scope="all", correction_enabled=False)
    guard.deadband = 0.0
    guard.ignore_preview_inflation = False
    guard.prepared = True
    guard.baseline_sigmas = {"plain.linear": 1.0}
    guard.baseline_family_stats = {"other": {"mean": 0.0, "std": 1.0}}
    guard.module_family_map = {"plain.linear": "other"}
    guard.family_caps = {"other": {"kappa": 10.0}}
    guard.target_sigma = 1.0
    guard.degeneracy = {
        "enabled": True,
        "stable_rank": {"warn_ratio": 0.9, "fatal_ratio": 0.8},
        "norm_collapse": {"warn_ratio": 0.9, "fatal_ratio": 0.8},
    }
    guard.baseline_degeneracy = {
        "plain.linear": {"stable_rank": 10.0, "norm_collapse": 1.0}
    }
    guard._log_event = lambda *_args, **_kwargs: None
    model = _TinyModel({"plain.linear": _TinyWeightModule(torch.eye(2))})

    monkeypatch.setattr(
        spectral_detection,
        "frobenius_norm_sq",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("sr boom")),
    )
    monkeypatch.setattr(
        spectral_detection,
        "row_col_norm_extrema",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("nc boom")),
    )

    violations = spectral_detection.detect_spectral_violations(
        guard,
        model,
        metrics={"plain.linear": 1.0},
        phase="finalize",
    )
    assert violations == []


def test_detect_spectral_violations_uses_default_kappa_and_degeneracy_defaults() -> (
    None
):
    guard = spectral_guard.SpectralGuard(scope="all", correction_enabled=False)
    guard.deadband = 0.0
    guard.ignore_preview_inflation = False
    guard.prepared = True
    guard.baseline_sigmas = {"plain.linear": 1.0}
    guard.baseline_family_stats = {"other": {"mean": 0.0, "std": 1.0}}
    guard.module_family_map = {"plain.linear": "other"}
    guard.family_caps = {"other": {"kappa": "bad"}}
    guard.target_sigma = 1.0
    guard.degeneracy = {
        "enabled": True,
        "stable_rank": {"warn_ratio": "bad"},
    }
    guard.baseline_degeneracy = {"plain.linear": {"stable_rank": 10.0}}

    model = _TinyModel({"plain.linear": _TinyWeightModule(torch.eye(2))})
    violations = guard._detect_spectral_violations(
        model,
        metrics={"plain.linear": 1.0},
        phase="finalize",
    )

    assert any(v["type"] == "degeneracy_stable_rank_drop" for v in violations)


def test_detect_spectral_violations_logs_module_errors() -> None:
    guard = spectral_guard.SpectralGuard(scope="all", correction_enabled=False)
    guard.deadband = 0.0
    guard.ignore_preview_inflation = False
    guard.prepared = True
    guard.baseline_sigmas = {}
    guard.baseline_family_stats = {}
    guard.module_family_map = {}
    guard.family_caps = {}
    guard.target_sigma = 1.0
    guard.degeneracy = {}
    guard.baseline_degeneracy = {}

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(operation: str, **kwargs: object) -> None:
        events.append((operation, kwargs))

    guard._log_event = _log_event

    model = _TinyModel({"plain.linear": _TinyWeightModule(torch.eye(2))})
    violations = spectral_detection.detect_spectral_violations(
        guard,
        model,
        metrics={},
        compute_sigma_max_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    assert violations == []
    assert events and events[0][0] == "violation_check_error"
