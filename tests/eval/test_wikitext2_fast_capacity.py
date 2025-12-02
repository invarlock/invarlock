from __future__ import annotations

import invarlock.eval.data as data_mod


def test_wikitext2_fast_capacity_without_network(monkeypatch):
    # Bypass datasets check and load
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    monkeypatch.setattr(
        data_mod.WikiText2Provider, "_validate_dependencies", lambda self: None
    )
    prov = data_mod.WikiText2Provider()
    # Return fixed sample list to avoid datasets
    monkeypatch.setattr(
        prov, "load", lambda split="validation", max_samples=2000, **kw: ["a", "b", "c"]
    )  # type: ignore[no-untyped-def]
    # Fast capacity via fast_mode
    cap = prov.estimate_capacity(tokenizer=None, seq_len=16, stride=8, fast_mode=True)
    assert cap["available_nonoverlap"] == 3
    # Fast capacity via env flag
    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    cap2 = prov.estimate_capacity(tokenizer=None, seq_len=16, stride=8, fast_mode=False)
    assert cap2["available_nonoverlap"] == 3
