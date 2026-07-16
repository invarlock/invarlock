from __future__ import annotations

from invarlock_addins.gguf.conformance import conformance_payload


def test_gguf_addin_conforms_to_public_provider_abi() -> None:
    assert conformance_payload() == {
        "abi_version": "1",
        "errors": [],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": True,
        "provider": "llama_cpp",
    }
