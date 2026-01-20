import importlib


def test_default_provider_known_for_gpt2():
    mp = importlib.import_module("invarlock.model_profile")
    prof = mp.detect_model_profile("sshleifer/tiny-gpt2", adapter="hf_causal")
    assert prof.default_provider in {"wikitext2", "hf_text", "synthetic"}
    # GPT-style should prefer WT2
    assert prof.default_provider == "wikitext2"


def test_default_provider_known_for_mistral():
    mp = importlib.import_module("invarlock.model_profile")
    prof = mp.detect_model_profile(
        "mistralai/Mistral-7B-v0.1", adapter="hf_causal"
    )
    assert prof.default_provider in {"wikitext2", "hf_text", "synthetic"}
    assert prof.default_provider == "wikitext2"


def test_default_provider_known_for_bert():
    mp = importlib.import_module("invarlock.model_profile")
    prof = mp.detect_model_profile("prajjwal1/bert-tiny", adapter="hf_mlm")
    assert prof.default_provider in {"wikitext2", "hf_text", "synthetic"}
    # BERT/MLM should default to hf_text
    assert prof.default_provider == "hf_text"


def test_run_fallback_provider_is_known(monkeypatch):
    # Ensure run command fallback picks a known provider name (wikitext2)
    from invarlock.cli.commands import run as run_mod

    class DummyCfg:
        class dataset:
            provider = None

    # Use a dummy model_profile-like object
    dummy_profile = type("P", (), {"default_provider": None})()

    kind, provider, _ = run_mod._resolve_metric_and_provider(DummyCfg, dummy_profile)
    assert provider in {"wikitext2", "hf_text", "synthetic"}
    assert provider == "wikitext2"
