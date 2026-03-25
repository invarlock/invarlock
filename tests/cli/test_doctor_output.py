from __future__ import annotations


def test_provider_network_wikitext2_label() -> None:
    from invarlock.cli.constants import PROVIDER_NETWORK

    assert PROVIDER_NETWORK.get("wikitext2") == "cache"
