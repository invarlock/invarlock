import sys
from types import ModuleType, SimpleNamespace

import torch

import invarlock.eval.data as data_mod
from invarlock.eval.data import WikiText2Provider


def test_wikitext2_scorer_with_mocked_transformers(monkeypatch):
    # Ensure torch is considered available
    monkeypatch.setattr(data_mod, "HAS_TORCH", True)
    monkeypatch.setenv("INVARLOCK_SCORES_BATCH_SIZE", "3")

    # Mock transformers.GPT2LMHeadModel
    class FakeModel:
        def __init__(self):
            self._device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, name):  # noqa: ARG003
            return cls()

        def eval(self):
            return self

        def to(self, device):
            self._device = device
            return self

        def __call__(self, input_ids, attention_mask=None):  # noqa: ARG002
            b, seq_len = input_ids.size(0), input_ids.size(1)
            vocab = 16
            logits = torch.randn(b, seq_len, vocab, device=self._device)
            return SimpleNamespace(logits=logits)

    fake_tf = ModuleType("transformers")
    fake_tf.GPT2LMHeadModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    # Patch CUDA/MPS availability to force CPU path
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: False, raising=True
        )

    # Instance under test
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()

    # Provide enough texts and a deterministic collector
    monkeypatch.setattr(pt, "load", lambda **kw: ["x" * 30] * 10)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        res = []
        for idx in indices:
            res.append((idx, [1, 2, 3, 0, 0], [1, 1, 1, 0, 0], 3))
        return res

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)

    # Run via windows to engage scorer; ensure batch override applied
    prev, final = pt.windows(
        tokenizer=SimpleNamespace(), seq_len=5, preview_n=4, final_n=3
    )
    assert len(prev) == 4 and len(final) == 3
    # Validate scorer profile and batch size override
    assert pt._last_batch_size_used == 3
    prof = pt.scorer_profile
    assert prof and prof["tokens_processed"] > 0 and prof["tokens_per_second"] >= 0.0


def test_wikitext2_scorer_exception_and_batch_default(monkeypatch):
    # Enable torch
    monkeypatch.setattr(data_mod, "HAS_TORCH", True)

    # Fake transformers that raises
    class RaisingModel:
        @classmethod
        def from_pretrained(cls, name):  # noqa: ARG003
            raise RuntimeError("boom")

    fake_tf = ModuleType("transformers")
    fake_tf.GPT2LMHeadModel = RaisingModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    pt = WikiText2Provider.__new__(WikiText2Provider)
    # Bypass __init__ dependency check; set attributes used by scorer
    pt._difficulty_model = None
    pt._difficulty_device = None
    pt._scorer_warmed = False
    pt._last_batch_size_used = 0
    pt._last_scorer_profile = None

    # Build minimal candidates
    candidates = [
        {
            "input_ids": [1, 2, 0, 0],
            "attention_mask": [1, 1, 0, 0],
            "dataset_index": i,
            "token_count": 2,
        }
        for i in range(3)
    ]
    ok = WikiText2Provider._score_candidates_with_model(pt, candidates)
    assert ok is False and pt._difficulty_model is False

    # Now set a working model and test default batch size (no env override)
    class FakeModel:
        def __init__(self):
            self._device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, name):  # noqa: ARG003
            return cls()

        def eval(self):
            return self

        def to(self, device):
            self._device = device
            return self

        def __call__(self, input_ids, attention_mask=None):  # noqa: ARG002
            b, seq_len = input_ids.size(0), input_ids.size(1)
            logits = torch.zeros(b, seq_len, 8, device=self._device)
            return SimpleNamespace(logits=logits)

    fake_tf2 = ModuleType("transformers")
    fake_tf2.GPT2LMHeadModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf2)

    # Reset flags
    pt._difficulty_model = None
    pt._difficulty_device = None
    pt._scorer_warmed = False
    ok2 = WikiText2Provider._score_candidates_with_model(pt, candidates)
    assert ok2 is True and pt._last_batch_size_used == 4  # default min batch size
    # Second call should skip warmup
    ok3 = WikiText2Provider._score_candidates_with_model(pt, candidates)
    assert ok3 is True
