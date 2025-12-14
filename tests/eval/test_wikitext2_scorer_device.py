import pytest

from invarlock.eval.data import WikiText2Provider


class DummyGPT2:
    def __init__(self):
        self.last_device = None

    def to(self, device):
        # Record the last device this model was moved to
        self.last_device = str(device)
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        import torch

        # Minimal logits tensor with the right shape
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, device=input_ids.device)

        class Out:
            def __init__(self, logits):
                self.logits = logits

        return Out(logits)


@pytest.fixture(autouse=True)
def _reset_wikitext2_cache():
    # Ensure class-level cache does not leak between tests
    WikiText2Provider._MODEL_CACHE = None
    WikiText2Provider._MODEL_DEVICE = None
    yield
    WikiText2Provider._MODEL_CACHE = None
    WikiText2Provider._MODEL_DEVICE = None


def _monkeypatch_gpt2(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch
    import transformers

    # Ensure tests never hit the network even if patching fails.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Force both CUDA and MPS to appear "available" so heuristics would pick them
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: True, raising=False
        )

    dummy = DummyGPT2()

    # Patch the classmethod to avoid HF downloads.
    monkeypatch.setattr(
        transformers.GPT2LMHeadModel,
        "from_pretrained",
        lambda *_, **__: dummy,
        raising=False,
    )

    return dummy


def test_wikitext2_scorer_respects_device_hint_cpu(monkeypatch):
    dummy = _monkeypatch_gpt2(monkeypatch)

    # No eval-device override
    monkeypatch.delenv("INVARLOCK_EVAL_DEVICE", raising=False)

    provider = WikiText2Provider(device_hint="cpu")

    candidates = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]

    ok = provider._score_candidates_with_model(candidates)
    assert ok is True

    # Difficulty model should live on CPU despite CUDA/MPS appearing available
    assert provider._difficulty_device is not None
    assert str(provider._difficulty_device).startswith("cpu")
    assert dummy.last_device is not None
    assert dummy.last_device.startswith("cpu")


def test_wikitext2_scorer_prefers_eval_env_over_hint(monkeypatch):
    dummy = _monkeypatch_gpt2(monkeypatch)

    # Force eval path to CPU even if hint prefers mps
    monkeypatch.setenv("INVARLOCK_EVAL_DEVICE", "cpu")

    provider = WikiText2Provider(device_hint="mps")

    candidates = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]

    ok = provider._score_candidates_with_model(candidates)
    assert ok is True

    # Env override wins: scorer should run on CPU
    assert provider._difficulty_device is not None
    assert str(provider._difficulty_device).startswith("cpu")
    assert dummy.last_device is not None
    assert dummy.last_device.startswith("cpu")


def test_wikitext2_scorer_moves_cached_model_on_new_hint(monkeypatch):
    dummy = _monkeypatch_gpt2(monkeypatch)
    monkeypatch.delenv("INVARLOCK_EVAL_DEVICE", raising=False)

    # First run: hint = cpu
    provider_cpu = WikiText2Provider(device_hint="cpu")
    candidates = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]
    ok = provider_cpu._score_candidates_with_model(candidates)
    assert ok is True
    first_device = str(provider_cpu._difficulty_device)
    assert first_device.startswith("cpu")

    # Second run: reuse cached model but request mps
    provider_mps = WikiText2Provider(device_hint="mps")
    ok = provider_mps._score_candidates_with_model(candidates)
    assert ok is True

    # Cached model should have been moved to the new device
    second_device = str(provider_mps._difficulty_device)
    assert dummy.last_device is not None
    if second_device != first_device:
        assert dummy.last_device.startswith("mps") or dummy.last_device.startswith(
            "cuda"
        )
    else:
        # Some torch builds cannot allocate on MPS/CUDA even when monkeypatched.
        assert second_device.startswith("cpu")
        assert dummy.last_device.startswith("cpu")
