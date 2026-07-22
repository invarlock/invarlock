from __future__ import annotations

import json

import pytest
from invarlock_addins.gguf import conformance
from invarlock_addins.gguf.conformance import conformance_payload
from invarlock_addins.gguf.provider import LlamaCppProvider


def test_gguf_addin_conforms_to_public_provider_abi() -> None:
    assert conformance_payload() == {
        "abi_version": "1",
        "errors": [],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": True,
        "provider": "llama_cpp",
    }


def test_gguf_conformance_rejects_an_incomplete_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conformance, "LlamaCppProvider", object)

    assert conformance.conformance_payload() == {
        "abi_version": None,
        "errors": ["provider does not implement RuntimeProvider"],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": False,
        "provider": None,
    }


def test_gguf_conformance_reports_name_and_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriftedProvider(LlamaCppProvider):
        name = "drifted_llama_cpp"
        abi_version = "999"

    monkeypatch.setattr(conformance, "LlamaCppProvider", DriftedProvider)

    payload = conformance.conformance_payload()

    assert payload["ok"] is False
    assert payload["provider"] == "drifted_llama_cpp"
    assert payload["abi_version"] == "999"
    assert payload["errors"] == [
        "provider name must be llama_cpp",
        "provider ABI does not match the installed core",
    ]


@pytest.mark.parametrize("ok", [True, False])
def test_gguf_conformance_command_emits_json_and_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ok: bool,
) -> None:
    payload = {
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": ok,
    }
    monkeypatch.setattr(conformance, "conformance_payload", lambda: payload)

    assert conformance.main() == (0 if ok else 1)
    assert json.loads(capsys.readouterr().out) == payload
