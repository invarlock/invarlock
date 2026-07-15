from __future__ import annotations

import invarlock.eval.data as data_mod
import invarlock.eval.data_providers as data_providers_mod
import invarlock.eval.data_support as data_support_mod


def test_wikitext2_fast_capacity_without_network(monkeypatch):
    # Bypass datasets check and load
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    monkeypatch.setattr(
        data_mod.WikiText2Provider, "_validate_dependencies", lambda self: None
    )
    prov = data_mod.WikiText2Provider()
    # Return fixed sample list to avoid datasets
    monkeypatch.setattr(
        prov, "load", lambda split="validation", max_samples=2000, **kw: ["a", "b", "c"]
    )
    # Fast capacity via fast_mode
    cap = prov.estimate_capacity(tokenizer=None, seq_len=16, stride=8, fast_mode=True)
    assert cap["available_nonoverlap"] == 3
    # Fast capacity via env flag
    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    cap2 = prov.estimate_capacity(tokenizer=None, seq_len=16, stride=8, fast_mode=False)
    assert cap2["available_nonoverlap"] == 3


def test_wikitext2_load_uses_namespaced_hf_dataset(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    monkeypatch.setattr(
        data_mod.WikiText2Provider, "_validate_dependencies", lambda self: None
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_stub(*args, **kwargs):  # noqa: ANN001
        calls.append((args, kwargs))
        return [
            {"text": "A long enough WikiText sample with alphabetic content."},
            {"text": ""},
        ]

    monkeypatch.setattr(
        data_providers_mod, "load_dataset_with_cache_fallback", load_stub
    )

    revision = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
    provider = data_mod.WikiText2Provider(revision=revision)
    texts = provider.load(max_samples=1)

    assert texts == ["A long enough WikiText sample with alphabetic content."]
    assert provider.dataset_name == "Salesforce/wikitext"
    assert provider.config_name == "wikitext-2-raw-v1"
    assert provider.revision == revision
    assert calls[0][0][:2] == ("Salesforce/wikitext", "wikitext-2-raw-v1")
    assert calls[0][1]["split"] == "validation[:1]"
    assert calls[0][1]["revision"] == revision


def test_hf_text_default_uses_namespaced_wikitext(monkeypatch):
    monkeypatch.setattr(
        data_providers_mod, "_require_load_dataset", lambda _message: object()
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_stub(*args, **kwargs):  # noqa: ANN001
        calls.append((args, kwargs))
        return [{"text": "sample"}]

    monkeypatch.setattr(
        data_providers_mod, "load_dataset_with_cache_fallback", load_stub
    )

    assert data_mod.HFTextProvider().load() == ["sample"]
    assert calls[0][1]["path"] == "Salesforce/wikitext"
