def test_hf_text_provider_windows_monkeypatched(monkeypatch):
    # Import module under test
    import invarlock.eval.data as data_mod
    import invarlock.eval.data_support as data_support_mod

    # Pretend datasets is available
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)

    # Stub load_dataset to return an iterable of rows with 'text'
    class DummyDS:
        def __iter__(self):
            for i in range(10):
                yield {"text": f"hello world {i}"}

        # emulate HF Dataset select() behavior lightly for cache warming
        def select(self, idxs):
            return self

        @property
        def cache_files(self):
            return []

    def fake_load_dataset(  # noqa: ARG001
        path, name=None, split=None, cache_dir=None, **kwargs
    ):
        return DummyDS()

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )

    # Create provider; verify load works without network
    prov = data_mod.HFTextProvider(
        dataset_name="dummy", text_field="text", max_samples=5
    )

    texts = prov.load(split="validation")
    assert len(texts) == 5

    # Simple tokenizer stub
    class T:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=8, padding="max_length"):  # noqa: ARG002
            suffix = int(text.rsplit(" ", 1)[-1])
            return [1, suffix + 2, suffix + 3]

    tokenizer = T()
    preview, final = prov.windows(tokenizer, seq_len=8, preview_n=3, final_n=2)
    assert len(preview) == 3
    assert len(final) == 2


def test_get_provider_hf_text_kwargs(monkeypatch):
    import invarlock.eval.data as data_mod
    import invarlock.eval.data_support as data_support_mod

    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)
    monkeypatch.setattr(
        data_support_mod, "load_dataset", lambda *a, **k: [], raising=False
    )

    prov = data_mod.get_provider(
        "hf_text",
        dataset_name="dummy",
        config_name="cnf",
        text_field="field",
        cache_dir="/tmp",
        max_samples=10,
    )
    assert isinstance(prov, data_mod.HFTextProvider)


def test_hf_text_provider_windows_respects_seeded_sampling(monkeypatch):
    import invarlock.eval.data as data_mod

    hp = data_mod.HFTextProvider(
        dataset_name="dummy", config_name=None, text_field="text", max_samples=10
    )
    monkeypatch.setattr(hp, "load", lambda **kw: [f"text-{i}" for i in range(10)])

    def simple_tok(texts, tokenizer, seq_len, indices):  # noqa: ARG001
        ids = [[position + 1, position + 2] for position in indices]
        masks = [[1, 1] for _ in texts]
        return data_mod.EvaluationWindow(ids, masks, list(indices))

    monkeypatch.setattr(hp, "_simple_tokenize", simple_tok)

    prev_a, fin_a = hp.windows(
        tokenizer=object(), seq_len=8, preview_n=3, final_n=2, seed=7
    )
    prev_b, fin_b = hp.windows(
        tokenizer=object(), seq_len=8, preview_n=3, final_n=2, seed=7
    )
    prev_c, fin_c = hp.windows(
        tokenizer=object(), seq_len=8, preview_n=3, final_n=2, seed=8
    )

    assert prev_a.indices == prev_b.indices
    assert fin_a.indices == fin_b.indices
    assert prev_a.indices != [0, 1, 2]
    assert fin_a.indices != [3, 4]
    assert prev_a.indices != prev_c.indices or fin_a.indices != fin_c.indices
    assert len(prev_a) == 3 and len(fin_a) == 2


def test_hf_text_provider_windows_keep_sampling_until_unique(monkeypatch):
    import invarlock.eval.data as data_mod
    import invarlock.eval.data_providers as data_providers_mod

    hp = data_mod.HFTextProvider(
        dataset_name="dummy", config_name=None, text_field="text", max_samples=16
    )
    texts = [
        "dup-a",
        "dup-a",
        "dup-a",
        "dup-b",
        "dup-b",
        "unique-0",
        "unique-1",
        "unique-2",
        "unique-3",
        "unique-4",
        "unique-5",
    ]
    monkeypatch.setattr(hp, "load", lambda **kw: texts)

    class NoShuffle:
        def __init__(self, seed):  # noqa: ARG002
            pass

        def shuffle(self, values):
            return None

    monkeypatch.setattr(data_providers_mod.random, "Random", NoShuffle)

    token_map = {
        "dup-a": [11, 12],
        "dup-b": [21, 22],
        "unique-0": [31, 32],
        "unique-1": [41, 42],
        "unique-2": [51, 52],
        "unique-3": [61, 62],
        "unique-4": [71, 72],
        "unique-5": [81, 82],
    }

    def simple_tok(texts, tokenizer, seq_len, indices):  # noqa: ARG001
        ids = [token_map[text] for text in texts]
        masks = [[1, 1] for _ in texts]
        return data_mod.EvaluationWindow(ids, masks, list(indices))

    monkeypatch.setattr(hp, "_simple_tokenize", simple_tok)

    prev, final = hp.windows(
        tokenizer=object(), seq_len=8, preview_n=3, final_n=3, seed=42
    )

    combined = prev.input_ids + final.input_ids
    assert len(prev) == 3
    assert len(final) == 3
    assert len({tuple(row) for row in combined}) == 6


def test_hf_text_provider_retries_with_invarlock_cache_on_lock_error(
    monkeypatch, tmp_path
):
    import invarlock.eval.data as data_mod
    import invarlock.eval.data_support as data_support_mod

    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)
    calls: list[str | None] = []

    def fake_load_dataset(path, name=None, split=None, cache_dir=None, **kwargs):  # noqa: ARG001
        calls.append(cache_dir)
        if len(calls) == 1:
            raise PermissionError(
                "Operation not permitted: '/Users/test/.cache/huggingface/datasets/sample.lock'"
            )
        return [{"text": "hello world from fallback cache"}]

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("INVARLOCK_HF_DATASETS_CACHE", raising=False)

    provider = data_mod.HFTextProvider(dataset_name="dummy", text_field="text")
    texts = provider.load(split="validation")

    assert texts == ["hello world from fallback cache"]
    assert calls[0] is None
    assert calls[1] == str(tmp_path / ".cache" / "invarlock" / "hf_datasets")
