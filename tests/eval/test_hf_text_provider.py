def test_hf_text_provider_windows_monkeypatched(monkeypatch):
    # Import module under test
    import invarlock.eval.data as data_mod

    # Pretend datasets is available
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)

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

    def fake_load_dataset(path, name=None, split=None, cache_dir=None):  # noqa: ARG001
        return DummyDS()

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset, raising=False)

    # Create provider; verify load works without network
    prov = data_mod.HFTextProvider(
        dataset_name="dummy", text_field="text", max_samples=5
    )

    texts = prov.load(split="validation")
    assert len(texts) == 5

    # Simple tokenizer stub
    class T:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=8):  # noqa: ARG002
            # return a few token ids
            return [1, 2, 3]

    tokenizer = T()
    preview, final = prov.windows(tokenizer, seq_len=8, preview_n=3, final_n=2)
    assert len(preview) == 3
    assert len(final) == 2


def test_get_provider_hf_text_kwargs(monkeypatch):
    import invarlock.eval.data as data_mod

    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)
    monkeypatch.setattr(data_mod, "load_dataset", lambda *a, **k: [], raising=False)

    prov = data_mod.get_provider(
        "hf_text",
        dataset_name="dummy",
        config_name="cnf",
        text_field="field",
        cache_dir="/tmp",
        max_samples=10,
    )
    assert isinstance(prov, data_mod.HFTextProvider)
