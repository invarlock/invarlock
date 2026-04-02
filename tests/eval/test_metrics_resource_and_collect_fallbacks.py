import types

import pytest
import torch
import torch.nn as nn

from invarlock.eval import metrics_runtime as runtime_mod
from invarlock.eval.metrics import MetricsConfig, ResourceManager, compute_perplexity
from invarlock.eval.metrics_activation import (
    _collect_activations,
    _perform_pre_eval_checks,
)


def test_resource_manager_gpu_memory_info_cuda_path(monkeypatch):
    cfg = MetricsConfig()
    rm = ResourceManager(cfg)
    # Force a CUDA-like device but make get_device_properties fail
    rm.device = types.SimpleNamespace(type="cuda")  # type: ignore[attr-defined]

    class DummyProps:
        total_memory = 1024 * 1024 * 1024

    class DummyCuda:
        def get_device_properties(self, idx):  # type: ignore[no-redef]
            return DummyProps()

        def memory_allocated(self):  # minimal API used in code
            return 0

        def is_available(self):
            return True

    monkeypatch.setattr(torch, "cuda", DummyCuda(), raising=False)
    # Should populate gpu_* keys without raising
    info = rm._get_memory_info()
    assert (
        "system_total_gb" in info
        and "gpu_total_gb" in info
        and info["gpu_total_gb"] > 0
    )


def test_collect_activations_malformed_batch_continues():
    class TinyLM(nn.Module):
        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            # Not reached due to malformed batch; define anyway
            if output_hidden_states:
                return types.SimpleNamespace(
                    hidden_states=[torch.randn(1, 4, 4) for _ in range(3)]
                )
            return types.SimpleNamespace(logits=torch.randn(1, 4, 5))

    # Dataloader yields a non-dict batch first to trigger inner except/continue
    def bad_then_stop():
        yield 123  # malformed batch triggers exception path

    out = _collect_activations(
        TinyLM().eval(),
        bad_then_stop(),
        MetricsConfig(),
        torch.device("cpu"),
    )
    # Expect empty collections and first_batch remains None due to early failure
    assert (
        out["hidden_states"] == []
        and out["fc1_activations"] == []
        and out["first_batch"] is None
    )


def test_sigma_max_no_columns_and_no_gain_values():
    # Provide a gains object without 'columns' attr and without 'gain'/'values'
    class GainsWeird:
        def __len__(self):
            return 2

        def __getitem__(self, mask):
            return self

    class DM:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model, first_batch):
                return GainsWeird()

            return scan_model_gains

    class Tiny(nn.Module):
        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            return types.SimpleNamespace(
                hidden_states=[torch.randn(1, 4, 4) for _ in range(3)]
            )

    from invarlock.eval.metrics_activation import _calculate_sigma_max

    out = _calculate_sigma_max(
        Tiny().eval(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(),
        MetricsConfig(),
        torch.device("cpu"),
    )
    # No gain values → NaN path
    assert isinstance(out, float) and (out != out)


def test_validate_dataloader_falsy_first_batch_raises_and_allow_empty_allows():
    class FalsyOnce:
        def __iter__(self):
            yield {}

    cfg_fail = MetricsConfig(allow_empty_data=False)
    import pytest

    from invarlock.eval.metrics import InputValidator, ValidationError

    with pytest.raises(ValidationError):
        InputValidator.validate_dataloader(FalsyOnce(), cfg_fail)
    # Now allow empty -> warning path
    InputValidator.validate_dataloader(
        FalsyOnce(), MetricsConfig(allow_empty_data=True)
    )


def test_resolve_eval_device_propagates_mps_backend_failures(monkeypatch):
    class BrokenMPS:
        @staticmethod
        def is_available():
            raise RuntimeError("mps boom")

    monkeypatch.setattr(torch.backends, "mps", BrokenMPS(), raising=False)

    with pytest.raises(RuntimeError, match="mps boom"):
        runtime_mod._resolve_eval_device(nn.Linear(1, 1), torch.device("mps"))


def test_infer_model_vocab_size_propagates_embedding_probe_failures():
    class BrokenEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()

        def get_input_embeddings(self):
            raise RuntimeError("embedding boom")

    with pytest.raises(RuntimeError, match="embedding boom"):
        runtime_mod._infer_model_vocab_size(BrokenEmbeddings())


def test_metrics_runtime_helper_resolution_and_pad_token_paths(monkeypatch):
    class Parameterless(nn.Module):
        def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise NotImplementedError

    class NoMPS:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch.backends, "mps", NoMPS(), raising=False)

    assert runtime_mod._resolve_eval_device(Parameterless(), None).type == "cpu"
    assert (
        runtime_mod._resolve_eval_device(nn.Linear(1, 1), torch.device("mps")).type
        == "cpu"
    )

    class EmbeddingsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 2)
            self.config = types.SimpleNamespace(pad_token_id=3, vocab_size=7)

        def get_input_embeddings(self):
            return self.emb

    class ModuleFallback(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_a = nn.Embedding(3, 2)
            self.emb_b = nn.Embedding(9, 2)

    class ConfigFallback(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=11, pad_token_id=99)

    assert runtime_mod._infer_model_vocab_size(EmbeddingsModel()) == 7
    assert runtime_mod._infer_model_vocab_size(ModuleFallback()) == 9
    assert runtime_mod._infer_model_vocab_size(ConfigFallback()) == 11
    assert runtime_mod._resolve_pad_token_id(EmbeddingsModel(), 7) == 3
    assert runtime_mod._resolve_pad_token_id(ConfigFallback(), 11) == 0
    assert (
        runtime_mod._resolve_pad_token_id(types.SimpleNamespace(config=None), None) == 0
    )


def test_runtime_perplexity_and_window_paths_warn_on_clamped_values(monkeypatch):
    class TinyLM(nn.Module):
        def __init__(self, vocab=5):
            super().__init__()
            self.emb = nn.Embedding(vocab, 4)
            self.head = nn.Linear(4, vocab)
            self.config = types.SimpleNamespace(vocab_size=vocab, pad_token_id=0)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            logits = self.head(self.emb(input_ids))
            return types.SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    window = runtime_mod.EvaluationWindow([[0, 1, 2]], [[1, 1, 1]], [0])
    model = TinyLM().eval()

    monkeypatch.setattr(runtime_mod.math, "exp", lambda _value: 0.5)
    assert (
        runtime_mod.compute_perplexity(model, [batch], max_samples=1, device="cpu")
        == 1.0
    )
    assert runtime_mod.compute_ppl(model, window, device="cpu") == 1.0

    monkeypatch.setattr(runtime_mod.math, "exp", lambda _value: float("inf"))
    assert runtime_mod.compute_perplexity(
        model, [batch], max_samples=1, device="cpu"
    ) == float("inf")
    assert runtime_mod.compute_ppl(model, window, device="cpu") == float("inf")


def test_runtime_token_sanitizer_and_raw_tensor_batches_cover_skip_paths():
    cleaned_ids, cleaned_mask, cleaned_labels = (
        runtime_mod._sanitize_token_ids_for_model(
            torch.tensor([[1, 9]]),
            None,
            torch.tensor([[1, 9]]),
            vocab_size=0,
            pad_token_id=0,
        )
    )
    assert cleaned_ids.tolist() == [[1, 9]]
    assert cleaned_mask is None
    assert cleaned_labels.tolist() == [[1, 9]]

    class TinyLM(nn.Module):
        def __init__(self, vocab=5):
            super().__init__()
            self.emb = nn.Embedding(vocab, 4)
            self.head = nn.Linear(4, vocab)
            self.config = types.SimpleNamespace(vocab_size=vocab, pad_token_id=0)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            logits = self.head(self.emb(input_ids))
            return types.SimpleNamespace(logits=logits)

    model = TinyLM().eval()
    dataloader = ["bad-batch", torch.tensor([[1]]), torch.tensor([[1, 2, 3]])]
    ppl = runtime_mod.compute_perplexity_strict(model, dataloader, device="cpu")
    assert ppl >= 1.0


def test_pre_eval_checks_dry_run_failure_and_compute_perplexity_no_valid_tokens():
    # Model raises during dry run forward to hit warning path
    class BadModel(nn.Module):
        def __init__(self, vocab=5):
            super().__init__()
            self.vocab = vocab
            self.config = types.SimpleNamespace(n_positions=2)

        def forward(self, *a, **k):
            raise RuntimeError("boom")

    dl = [{"input_ids": torch.randint(0, 5, (1, 3))}]
    _perform_pre_eval_checks(
        BadModel().eval(), dl, torch.device("cpu"), MetricsConfig()
    )

    # Compute perplexity with zero-valid tokens (all attention masked out) should continue then raise
    class TinyLM(nn.Module):
        def __init__(self, vocab=5):
            super().__init__()
            self.vocab = vocab

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab)
            return types.SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.randint(0, 5, (1, 4)),
        "attention_mask": torch.zeros(1, 4, dtype=torch.long),
    }
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_perplexity(TinyLM().eval(), [batch], max_samples=1, device="cpu")
