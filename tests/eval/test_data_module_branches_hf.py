from invarlock.eval.data import EvaluationWindow, HFTextProvider


class DummyTok:
    def __init__(self, pad_id=0):
        self.pad_token_id = pad_id

    def encode(self, text, max_length, truncation, padding):  # noqa: ARG002
        ids = list(range(1, min(len(text) + 1, max_length + 1)))
        if len(ids) < max_length:
            ids += [self.pad_token_id] * (max_length - len(ids))
        return ids


def test_hf_text_provider_windows_success(monkeypatch):
    hp = HFTextProvider(
        dataset_name="dummy", config_name=None, text_field="text", max_samples=10
    )
    # Provide enough texts and stub tokenizer to avoid external deps
    monkeypatch.setattr(hp, "load", lambda **kw: ["hello world"] * 4)

    # Monkeypatch simple tokenizer to a deterministic minimal implementation
    def simple_tok(texts, tokenizer, seq_len, indices):  # noqa: ARG001
        ids = [[1] for _ in texts]
        masks = [[1] for _ in texts]
        return EvaluationWindow(ids, masks, indices[: len(ids)])

    monkeypatch.setattr(hp, "_simple_tokenize", simple_tok)
    prev, final = hp.windows(tokenizer=DummyTok(), seq_len=8, preview_n=2, final_n=2)
    assert isinstance(prev, EvaluationWindow) and len(prev) == 2 and len(final) == 2


def test_hf_text_provider_estimate_capacity(monkeypatch):
    hp = HFTextProvider(
        dataset_name="dummy", config_name=None, text_field="text", max_samples=7
    )
    monkeypatch.setattr(hp, "load", lambda **kw: ["x"] * 5)
    cap = hp.estimate_capacity(tokenizer=DummyTok(), seq_len=8, stride=4)
    assert cap["available_nonoverlap"] == 5 and cap["candidate_limit"] == 5


def test_hf_simple_tokenize_mapping_and_exception(monkeypatch):
    hp = HFTextProvider.__new__(HFTextProvider)

    # call _simple_tokenize directly with mapping tokenizer
    def tok_map(text, **kw):  # noqa: ARG002
        return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    out = HFTextProvider._simple_tokenize(hp, ["a", "b"], tok_map, 3, [0, 1])
    assert isinstance(out, EvaluationWindow) and len(out) == 2

    # Exception path (bad tokenizer)
    class BadTok:
        def encode(self, *a, **k):  # noqa: ARG002
            raise ValueError("bad")

    out2 = HFTextProvider._simple_tokenize(hp, ["a"], BadTok(), 3, [0])
    assert len(out2) == 0
