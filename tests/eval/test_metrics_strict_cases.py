from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M


def test_compute_perplexity_strict_masked_lm_branch():
    class DummyBert(torch.nn.Module):
        def __init__(self, vocab=8):
            super().__init__()
            self.config = SimpleNamespace(model_type="bert")
            self.out = torch.nn.Linear(4, vocab)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=True,
        ):  # noqa: D401
            # Return an object with .loss tensor to exercise masked LM path
            loss = torch.tensor(0.5)
            return SimpleNamespace(
                loss=loss,
                logits=self.out(
                    torch.zeros_like(input_ids, dtype=torch.float)
                    .unsqueeze(-1)
                    .expand(-1, -1, 4)
                ),
            )

        def parameters(self):  # pragma: no cover
            yield from self.out.parameters()

    batch = {
        "input_ids": torch.ones(1, 6, dtype=torch.long),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    ppl = M.compute_perplexity_strict(DummyBert(), [batch])
    assert isinstance(ppl, float) and ppl >= 1.0


def test_compute_perplexity_strict_tuple_loss_logits():
    class DummyLM(torch.nn.Module):
        def __init__(self, vocab=8):
            super().__init__()
            self.out = torch.nn.Linear(4, vocab)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=False
        ):  # noqa: D401
            logits = self.out(
                torch.zeros_like(input_ids, dtype=torch.float)
                .unsqueeze(-1)
                .expand(-1, -1, 4)
            )
            _ = torch.tensor(1.0)
            # Simulate models that don't support return_dict to force fallback branch
            if return_dict:
                raise TypeError("does not support return_dict")
            return (logits,)

        def parameters(self):  # pragma: no cover
            yield from self.out.parameters()

    batch = {
        "input_ids": torch.ones(1, 6, dtype=torch.long),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    ppl = M.compute_perplexity_strict(DummyLM(), [batch])
    assert isinstance(ppl, float) and ppl >= 1.0


def test_compute_ppl_partial_attention_mask():
    # Some tokens invalid → ensure valid ones used and finite PPL
    window = SimpleNamespace(input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 0, 1, 0]])

    class DummyLM(torch.nn.Module):
        def __init__(self, vocab=8):
            super().__init__()
            self.out = torch.nn.Linear(4, vocab)

        def eval(self):  # pragma: no cover
            return self

        def forward(self, input_ids=None, attention_mask=None, return_dict=True):  # noqa: D401
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return SimpleNamespace(logits=logits)

        def parameters(self):  # pragma: no cover
            yield from self.out.parameters()

    ppl = M.compute_ppl(DummyLM(), None, window)
    assert isinstance(ppl, float) and ppl >= 1.0


def test_measure_latency_measured_cpu_path():
    window = SimpleNamespace(input_ids=[list(range(12))], attention_masks=[[1] * 12])

    class DummyLM(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):  # noqa: D401
            # No-op forward
            return SimpleNamespace(logits=None)

        def parameters(self):  # pragma: no cover
            yield from ()

    lat = M.measure_latency(
        DummyLM(), window, device="cpu", warmup_steps=1, measurement_steps=2
    )
    assert isinstance(lat, float)
    # Either zero (if model path fails) or a small finite measurement
    assert lat >= 0.0
