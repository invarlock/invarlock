from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import invarlock.eval.bench_policy as bench_policy
import invarlock.eval.metrics_activation as metrics_activation_mod
import invarlock.eval.metrics_aggregation as metrics_aggregation
import invarlock.eval.metrics_runtime as metrics_runtime_mod
import invarlock.eval.probes.mi as mi_mod
import invarlock.eval.probes.post_attention as post_attention
from invarlock.eval.metrics_support import MetricsConfig
from invarlock.eval.providers.vision_text import VisionTextProvider, _resolve_image_path


class _ExplodingGetDict(dict):
    def get(self, *_args, **_kwargs):  # type: ignore[override]
        raise TypeError("boom")


def test_bench_policy_extract_and_markdown_cover_error_and_dash_paths() -> None:
    report = {
        "metrics": {
            "primary_metric": _ExplodingGetDict(),
            "latency_ms_per_tok": 1.0,
            "memory_mb_peak": 2.0,
        },
        "meta": _ExplodingGetDict(),
    }

    extracted = bench_policy.MetricsAggregator.extract_core_metrics(report)
    assert np.isnan(extracted["primary_metric_preview"])
    assert np.isnan(extracted["primary_metric_final"])
    assert np.isnan(extracted["duration_s"])

    summary = bench_policy.BenchmarkSummary(
        config=bench_policy.BenchmarkConfig(
            edits=["quant_rtn"],
            tiers=["balanced"],
            probes=[0],
            output_dir=Path("bench"),
        ),
        scenarios=[
            bench_policy.ScenarioResult(
                config=bench_policy.ScenarioConfig(
                    edit="quant_rtn",
                    tier="balanced",
                    probes=0,
                ),
                metrics={
                    "primary_metric_overhead": 0.0,
                    "guard_overhead_time": float("nan"),
                    "guard_overhead_mem": 0.0,
                    "rmt_outliers_bare": 0,
                    "rmt_outliers_guarded": 0,
                },
                gates={"spike": True, "rmt": True, "quality": True},
            )
        ],
        overall_pass=True,
        timestamp="2026-04-08T00:00:00",
        execution_time_seconds=0.1,
    )

    markdown = bench_policy.generate_step14_markdown(summary)
    assert (
        "| quant_rtn | balanced | 0 | ✅ PASS | 🟢 +0.0% | - | 🟢 +0.0% |" in markdown
    )


class _LayeredContainer(nn.Module):
    def __init__(self, name: str, value: float) -> None:
        super().__init__()
        self.register_parameter(name, nn.Parameter(torch.full((2, 2), value)))


class _MismatchModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.Module()
        self.transformer.layers.register_parameter(
            "x_weight", nn.Parameter(torch.full((2, 2), value))
        )

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        yield (
            "transformer.layers.x.weight",
            self.transformer.layers._parameters["x_weight"],
        )  # type: ignore[index]


def test_metrics_aggregation_covers_non_matching_layer_names_and_missing_after_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _MismatchModel(0.0)
    after = _MismatchModel(1.0)

    deltas = metrics_aggregation.compute_parameter_deltas(before, after)
    assert deltas["params_changed"] == 4
    assert deltas["layers_modified"] == 0

    spectral_module = types.ModuleType("invarlock.guards.spectral_measurement")
    spectral_module.compute_spectral_norms = (
        lambda model, scope="ffn": {"layer_a": 2.0, "layer_b": 4.0}
        if model is before
        else {"layer_a": 3.0}
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "invarlock.guards.spectral_measurement",
        spectral_module,
    )

    changes = metrics_aggregation.analyze_spectral_changes(before, after)
    assert changes["layers_analyzed"] == 1
    assert "layer_b" not in changes["layer_changes"]
    assert changes["layer_changes"]["layer_a"]["ratio"] == 1.5


class _MiBlock(nn.Module):
    def __init__(self, hidden: int, mlp_dim: int) -> None:
        super().__init__()
        self.mlp = types.SimpleNamespace(c_fc=nn.Linear(hidden, mlp_dim, bias=False))


class _MiModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_layer=1)
        self.embedding = nn.Embedding(8, 1)
        self.transformer = SimpleNamespace(
            h=nn.ModuleList([_MiBlock(1, 1), _MiBlock(1, 1)])
        )
        self.out = nn.Linear(1, 8, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.transformer.h:
            x = block.mlp.c_fc(x)
        return SimpleNamespace(logits=self.out(x))


def test_mi_probe_covers_extra_layer_skip_and_sample_subselection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mi_mod,
        "mutual_info_regression",
        lambda features, targets, random_state=42: np.asarray(
            [float(features.mean()) + float(targets.mean()) * 0.0]
        ),
    )

    scores = mi_mod.compute_neuron_mi_scores(
        _MiModel(),
        [torch.randint(0, 8, (1, 10002), dtype=torch.long)],
        oracle_windows=1,
        device="cpu",
    )

    assert len(scores) == 1
    assert tuple(scores[0].shape) == (1,)


class _HeadScoreBlock(nn.Module):
    def __init__(self, hidden_size: int, *, include_ln: bool) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        if include_ln:
            self.ln_2 = nn.Identity()
        self.mlp = types.SimpleNamespace(
            c_fc=nn.Linear(hidden_size, hidden_size, bias=False)
        )


class _HeadScoreModel(nn.Module):
    def __init__(
        self, *, n_layers: int, blocks: list[_HeadScoreBlock], vocab: int = 16
    ):
        super().__init__()
        self.config = SimpleNamespace(n_layer=n_layers, n_head=2)
        self.embedding = nn.Embedding(vocab, 8)
        self.transformer = SimpleNamespace(h=nn.ModuleList(blocks))
        self.out = nn.Linear(8, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.transformer.h:
            x = block.mlp.c_fc(block.attn(x))
        return SimpleNamespace(logits=self.out(x))


def test_post_attention_probes_cover_missing_ln_extra_layers_and_none_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        post_attention.torch,
        "norm",
        lambda inp, p="fro", dim=None, keepdim=False, out=None, dtype=None: torch.ones(
            inp.size(2), device=inp.device
        )
        if isinstance(dim, tuple | list) and len(dim) == 3 and inp.dim() == 4
        else torch.linalg.vector_norm(inp),
    )

    head_model = _HeadScoreModel(
        n_layers=1,
        blocks=[
            _HeadScoreBlock(8, include_ln=False),
            _HeadScoreBlock(8, include_ln=False),
        ],
    )
    head_scores = post_attention.compute_post_attention_head_scores(
        head_model,
        [torch.randint(0, 16, (1, 4), dtype=torch.long)],
        calibration_windows=1,
        device="cpu",
    )
    assert tuple(head_scores["scores"].shape) == (1, 2)

    wanda_model = _HeadScoreModel(
        n_layers=1,
        blocks=[
            _HeadScoreBlock(8, include_ln=True),
            _HeadScoreBlock(8, include_ln=True),
        ],
    )
    for block in wanda_model.transformer.h:
        block.mlp.c_fc.weight.requires_grad_(False)
    wanda_scores = post_attention.compute_wanda_neuron_scores(
        wanda_model,
        [torch.randint(0, 16, (1, 4), dtype=torch.long)],
        calibration_windows=1,
        device="cpu",
    )
    assert torch.equal(
        wanda_scores["scores"],
        torch.zeros_like(wanda_scores["scores"]),
    )


def test_metrics_activation_progress_empty_mi_and_sigma_filter_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates = []
    config = MetricsConfig(progress_observer=updates.append, use_cache=False)
    metrics_activation_mod._emit_progress(  # noqa: SLF001
        config,
        phase="mi_gini_cpu",
        completed=2,
        total=4,
    )
    assert updates[0].completed == 2


class _TinyRuntimeModel(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, 2), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def test_metrics_runtime_invalid_label_branches_and_cuda_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyRuntimeModel().eval()
    batch = {
        "input_ids": torch.tensor([[9, 9, 9]], dtype=torch.long),
        "labels": torch.tensor([[9, 9, 9]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    with pytest.raises(metrics_runtime_mod.ValidationError):
        metrics_runtime_mod.compute_perplexity_strict(model, [batch], device="cpu")
    with pytest.raises(metrics_runtime_mod.ValidationError):
        metrics_runtime_mod.compute_perplexity(
            model, [batch], max_samples=1, device="cpu"
        )
    with pytest.raises(metrics_runtime_mod.ValidationError):
        metrics_runtime_mod.compute_ppl(
            model,
            metrics_runtime_mod.EvaluationWindow([[9, 9, 9]], [[1, 1, 1]], [0]),
            device="cpu",
        )

    class _Cuda:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def synchronize(self) -> None:
            self.calls.append("synchronize")

        def empty_cache(self) -> None:
            self.calls.append("empty_cache")

        def memory_allocated(self) -> float:
            self.calls.append("memory_allocated")
            return 8 * 1024 * 1024

        def reset_peak_memory_stats(self) -> None:
            self.calls.append("reset_peak_memory_stats")

    cuda = _Cuda()
    monkeypatch.setattr(metrics_runtime_mod.torch, "cuda", cuda, raising=False)
    cuda_device = SimpleNamespace(type="cuda")

    metrics_runtime_mod._maybe_cuda_synchronize(cuda_device)
    baseline_mb, process = metrics_runtime_mod._memory_measurement_baseline(cuda_device)
    current_mb = metrics_runtime_mod._current_memory_mb(cuda_device, process)
    metrics_runtime_mod._cleanup_memory_measurement_failure(cuda_device)

    assert baseline_mb == pytest.approx(8.0)
    assert current_mb == pytest.approx(8.0)
    assert process is None
    assert cuda.calls == [
        "synchronize",
        "empty_cache",
        "memory_allocated",
        "reset_peak_memory_stats",
        "memory_allocated",
        "empty_cache",
    ]

    class _NameSeries:
        def __init__(self, names: list[str]) -> None:
            self._names = names

        @property
        def str(self) -> _NameSeries:
            return self

        def contains(self, pattern: str, case: bool = False, regex: bool = True):
            del case, regex
            return np.asarray(
                [("embed" in name) or ("lm_head" in name) for name in self._names]
            )

    class _Frame:
        def __init__(self, names: list[str]) -> None:
            self._names = names
            self.columns = ["name"]

        def __getitem__(self, key):
            if isinstance(key, str) and key == "name":
                return _NameSeries(self._names)
            return self.__class__(
                [
                    name
                    for name, keep in zip(self._names, list(key), strict=False)
                    if keep
                ]
            )

        def __len__(self) -> int:
            return len(self._names)

    dep_manager = SimpleNamespace(
        is_available=lambda _name: True,
        get_module=lambda _name: (lambda _model, _batch: _Frame(["embed", "lm_head"])),
    )
    sigma_max = metrics_activation_mod._calculate_sigma_max(  # noqa: SLF001
        nn.Linear(2, 2),
        {"input_ids": torch.ones((1, 2), dtype=torch.long)},
        dep_manager,
        MetricsConfig(use_cache=False),
        torch.device("cpu"),
    )
    assert np.isnan(sigma_max)

    class _GainFrame(_Frame):
        @property
        def gain(self):
            return [0.5 * (idx + 1) for idx, _ in enumerate(self._names)]

    sigma_max_nonempty = metrics_activation_mod._calculate_sigma_max(  # noqa: SLF001
        nn.Linear(2, 2),
        {"input_ids": torch.ones((1, 2), dtype=torch.long)},
        SimpleNamespace(
            is_available=lambda _name: True,
            get_module=lambda _name: (
                lambda _model, _batch: _GainFrame(["embed", "block.0"])
            ),
        ),
        MetricsConfig(use_cache=False),
        torch.device("cpu"),
    )
    assert sigma_max_nonempty == pytest.approx(0.5)

    assert np.isnan(
        metrics_activation_mod._mi_gini_optimized_cpu_path(  # noqa: SLF001
            torch.empty((0, 2, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            max_per_layer=2,
            config=MetricsConfig(use_cache=False),
        )
    )


def test_metrics_runtime_covers_remaining_mps_and_vocab_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingWeightEmbedding:
        weight = None

    class _ZeroShapeEmbedding:
        weight = torch.zeros((0, 2))

    class _EmbeddingsMissingWeight(nn.Module):
        def get_input_embeddings(self):
            return _MissingWeightEmbedding()

    class _EmbeddingsZeroShape(nn.Module):
        def get_input_embeddings(self):
            return _ZeroShapeEmbedding()

    assert (
        metrics_runtime_mod._infer_model_vocab_size(_EmbeddingsMissingWeight()) is None
    )
    assert metrics_runtime_mod._infer_model_vocab_size(_EmbeddingsZeroShape()) is None

    sanitized_ids, sanitized_mask, sanitized_labels = (
        metrics_runtime_mod._sanitize_token_ids_for_model(
            torch.tensor([[9, 9]], dtype=torch.long),
            None,
            None,
            vocab_size=5,
            pad_token_id=0,
        )
    )
    assert sanitized_ids.tolist() == [[0, 0]]
    assert sanitized_mask is None
    assert sanitized_labels is None

    class _MaskedLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(4, 3)
            self.config = SimpleNamespace(
                model_type="bert", vocab_size=4, pad_token_id=0
            )

        def get_input_embeddings(self):
            return self.emb

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=True,
        ):
            del input_ids, attention_mask, token_type_ids, return_dict
            assert labels is not None
            return SimpleNamespace(loss=torch.tensor(0.5))

    strict_batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }
    assert (
        metrics_runtime_mod.compute_perplexity_strict(
            _MaskedLM().eval(),
            [strict_batch],
            device="cpu",
        )
        > 1.0
    )

    class _TinyRuntimeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(vocab_size=5, pad_token_id=0)
            self.emb = nn.Embedding(5, 2)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            del attention_mask, return_dict
            batch, seq_len = input_ids.shape
            logits = torch.zeros((batch, seq_len, 5), dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    original_tensor_to = torch.Tensor.to

    def _identity_to(self, *args, **kwargs):
        del args, kwargs
        return self

    monkeypatch.setattr(
        metrics_runtime_mod,
        "_resolve_eval_device",
        lambda *_args, **_kwargs: "mps",
    )
    monkeypatch.setattr(torch.Tensor, "to", _identity_to, raising=False)
    try:
        ppl = metrics_runtime_mod.compute_ppl(
            _TinyRuntimeModel().eval(),
            metrics_runtime_mod.EvaluationWindow([[1, 2, 3]], [[1, 1, 1]], [0]),
            device="mps",
        )
    finally:
        monkeypatch.setattr(torch.Tensor, "to", original_tensor_to, raising=False)

    assert ppl >= 1.0


def test_vision_text_provider_resolves_relative_paths_and_items_override(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"png")

    resolved = _resolve_image_path("img.png", base_dir=tmp_path)
    assert resolved == image_path
    assert _resolve_image_path(str(image_path), base_dir=tmp_path) == image_path

    provider = VisionTextProvider(items=[{"image": "img.png", "answer": "ok"}])
    assert provider._resolve_files() == []
