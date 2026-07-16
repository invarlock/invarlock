from __future__ import annotations

from invarlock_addins.tensorrt_llm.conformance import conformance_payload


def test_tensorrt_llm_addin_conforms_to_public_provider_abi() -> None:
    assert conformance_payload() == {
        "abi_version": "1",
        "errors": [],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": True,
        "provider": "tensorrt_llm",
    }
