import types

import pytest
import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    ResourceManager,
    _collect_activations,
    _perform_pre_eval_checks,
    compute_perplexity,
)


def test_resource_manager_gpu_memory_info_cuda_branch(monkeypatch):
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
        MetricsConfig(progress_bars=False),
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

    out = __import__(
        "invarlock.eval.metrics", fromlist=["_calculate_sigma_max"]
    )._calculate_sigma_max(
        Tiny().eval(),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(),
        MetricsConfig(progress_bars=False),
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
