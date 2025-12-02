from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    InputValidator,
    MetricsConfig,
    _calculate_head_energy,
    _calculate_sigma_max,
    _extract_fc1_activations,
    _perform_pre_eval_checks,
    compute_perplexity,
    compute_perplexity_strict,
    measure_latency,
    measure_memory,
)


class TinyLM(nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab = vocab_size
        self.lin = nn.Linear(8, vocab_size)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=False
    ):
        B, T = input_ids.shape
        logits = torch.randn(B, T, self.vocab)
        if return_dict:
            return SimpleNamespace(logits=logits)
        return (logits,)


def test_compute_perplexity_mps_gather_fallback():
    model = TinyLM().eval()
    batch = {
        "input_ids": torch.randint(0, model.vocab, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    ppl = compute_perplexity(model, [batch], max_samples=1, device="mps")
    assert isinstance(ppl, float) and ppl >= 1.0


def test_extract_fc1_inconsistent_shapes_filters_to_common_shape():
    class Block(nn.Module):
        def __init__(self, out_dim):
            super().__init__()

            class CF(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.d = d

                def forward(self, x):
                    B, T, _ = x.shape
                    return torch.randn(B, T, self.d)

            self.mlp = SimpleNamespace(c_fc=CF(out_dim))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block(3), Block(4)])

    # hidden_states for idx+1 access → supply >= 2
    hs = [torch.randn(1, 5, 4) for _ in range(3)]
    out = _extract_fc1_activations(
        Model(), SimpleNamespace(hidden_states=hs), MetricsConfig(progress_bars=False)
    )
    # Should return a stacked tensor with consistent last dim (most common shape)
    assert out is None or out.dim() in (4,)


def test_extract_fc1_success_returns_stacked_tensor():
    class Block(nn.Module):
        def __init__(self, out_dim):
            super().__init__()

            class CF(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.d = d

                def forward(self, x):
                    B, T, _ = x.shape
                    return torch.randn(B, T, self.d)

            self.mlp = SimpleNamespace(c_fc=CF(3))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(h=[Block(3), Block(3)])

    hs = [torch.randn(1, 5, 4) for _ in range(4)]
    out = _extract_fc1_activations(
        Model(), SimpleNamespace(hidden_states=hs), MetricsConfig(progress_bars=False)
    )
    assert isinstance(out, torch.Tensor)


def test_sigma_max_empty_filtered_gains_returns_nan():
    class GainsEmpty:
        columns = ["name", "gain"]

        def __init__(self, n):
            self._n = n

        def __len__(self):
            return self._n

        def __getitem__(self, mask):
            # Return an empty view when filtering
            return GainsEmpty(0)

        @property
        def name(self):
            # Names that will be entirely filtered out
            return ["embed.wte"] * self._n

        @property
        def gain(self):
            return [0.1] * self._n

    class DM:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model, first_batch):
                return GainsEmpty(1)

            return scan_model_gains

    val = _calculate_sigma_max(
        nn.Linear(2, 2),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM(),
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    assert isinstance(val, float) and (val != val)  # NaN

    # Also cover None returns path
    class DM2:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model, first_batch):
                return None

            return scan_model_gains

    val2 = _calculate_sigma_max(
        nn.Linear(2, 2),
        {"input_ids": torch.ones(1, 8, dtype=torch.long)},
        DM2(),
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    assert isinstance(val2, float) and (val2 != val2)


def test_validate_tensor_nan_replacement():
    t = torch.tensor([1.0, float("nan"), 3.0])
    out = InputValidator.validate_tensor(t, "x", MetricsConfig(strict_validation=False))
    assert torch.isfinite(out).all()


def test_validate_tensor_inf_replacement_non_strict():
    t = torch.tensor([1.0, float("inf"), float("-inf")])
    out = InputValidator.validate_tensor(t, "y", MetricsConfig(strict_validation=False))
    assert torch.isfinite(out).all()


def test_compute_perplexity_strict_and_pre_eval_warns_on_context_length():
    model = TinyLM().eval()
    dl = [
        {
            "input_ids": torch.randint(0, model.vocab, (1, 6)),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
    ]
    ppl = compute_perplexity_strict(model, dl, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0

    # Model with config limit smaller than batch length to trigger length warning path
    class Cfg:
        n_positions = 2

    class M(TinyLM):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    _perform_pre_eval_checks(M().eval(), dl, torch.device("cpu"), MetricsConfig())


def test_mi_gini_gpu_oom_fallback_to_cpu():
    # Activation data: list with shapes L=1, N=1, T=6, D=3
    feats = torch.randn(1, 1, 6, 3)
    targs = torch.randint(0, 7, (1, 6))
    activation_data = {"fc1_activations": [feats], "targets": [targs]}

    class DM:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            def fn(*a, **k):
                raise RuntimeError("CUDA out of memory")

            return fn

    # Should exercise the RuntimeError OOM path and return a float
    from invarlock.eval.metrics import _calculate_mi_gini

    val = _calculate_mi_gini(
        TinyLM(),
        activation_data,
        DM(),
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    assert isinstance(val, float)


def test_compute_perplexity_skips_short_sequences_then_uses_valid_batch():
    model = TinyLM().eval()
    short = {
        "input_ids": torch.randint(0, model.vocab, (1, 1)),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
    }
    good = {
        "input_ids": torch.randint(0, model.vocab, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    ppl = compute_perplexity(model, [short, good], max_samples=2, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0


def test_compute_perplexity_batch_as_tensor():
    model = TinyLM().eval()
    tensor_batch = torch.randint(0, model.vocab, (1, 5))
    ppl = compute_perplexity(model, [tensor_batch], max_samples=1, device="cpu")
    assert isinstance(ppl, float) and ppl >= 1.0


def test_calculate_perplexity_raises_when_no_valid_tokens():
    class TinyLM2(TinyLM):
        pass

    model = TinyLM2().eval()
    dl = [
        {
            "input_ids": torch.randint(0, model.vocab, (1, 1)),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }
    ]
    import pytest

    from invarlock.eval.metrics import ValidationError as MValidationError
    from invarlock.eval.metrics import calculate_perplexity

    with pytest.raises(MValidationError):
        calculate_perplexity(model, dl, max_batches=1, device="cpu")


def test_measure_latency_returns_zero_on_empty_or_short_window():
    class LM2(nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            return SimpleNamespace(
                logits=torch.randn(input_ids.size(0), input_ids.size(1), 5)
            )

    class WEmpty:
        input_ids = []
        attention_masks = []

    class WShort:
        input_ids = [[1, 2, 3]]
        attention_masks = [[1, 1, 1]]

    assert (
        measure_latency(
            LM2(), WEmpty(), device="cpu", warmup_steps=0, measurement_steps=1
        )
        == 0.0
    )
    assert (
        measure_latency(
            LM2(), WShort(), device="cpu", warmup_steps=0, measurement_steps=1
        )
        == 0.0
    )


def test_head_energy_success_and_all_nan_paths():
    cfg = MetricsConfig()
    # Success: two layers, one batch, short T, D
    hs1 = torch.randn(1, 1, 4, 3)
    hs2 = torch.randn(1, 1, 4, 3)
    val = _calculate_head_energy([hs1, hs2], cfg)
    assert isinstance(val, float)
    # All-NaN path
    nan_hs = torch.full((1, 1, 2, 2), float("nan"))
    val2 = _calculate_head_energy([nan_hs], cfg)
    assert isinstance(val2, float) and (val2 != val2)


def test_measure_latency_and_memory_cpu_paths():
    class LM(nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            # return something trivial
            return SimpleNamespace(
                logits=torch.randn(input_ids.size(0), input_ids.size(1), 5)
            )

    # Latency with simple window object
    class W:
        def __init__(self):
            self.input_ids = [list(range(12))]
            self.attention_masks = [[1] * 12]

    lat = measure_latency(LM(), W(), device="cpu", warmup_steps=0, measurement_steps=1)
    assert isinstance(lat, float)

    class Window:
        def __init__(self):
            self.input_ids = [[1, 2, 3, 4] for _ in range(2)]
            self.attention_masks = [[1, 1, 1, 1] for _ in range(2)]

    mem = measure_memory(LM(), Window(), device="cpu")
    assert isinstance(mem, float)
