from __future__ import annotations

import json

import pytest
from invarlock_addins.tensorrt_llm import conformance
from invarlock_addins.tensorrt_llm.conformance import conformance_payload
from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider


def test_tensorrt_llm_addin_conforms_to_public_provider_abi() -> None:
    assert conformance_payload() == {
        "abi_version": "1",
        "errors": [],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": True,
        "provider": "tensorrt_llm",
    }


def test_tensorrt_llm_conformance_rejects_incomplete_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conformance, "TensorRTLLMProvider", object)

    assert conformance_payload() == {
        "abi_version": None,
        "errors": ["provider does not implement RuntimeProvider"],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": False,
        "provider": None,
    }


def test_tensorrt_llm_conformance_reports_name_and_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TensorRTLLMProvider, "name", "wrong_provider")
    monkeypatch.setattr(TensorRTLLMProvider, "abi_version", "wrong-abi")

    payload = conformance_payload()

    assert payload["ok"] is False
    assert payload["provider"] == "wrong_provider"
    assert payload["abi_version"] == "wrong-abi"
    assert payload["errors"] == [
        "provider name must be tensorrt_llm",
        "provider ABI does not match the installed core",
    ]


def test_tensorrt_llm_conformance_main_emits_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert conformance.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setattr(
        conformance,
        "conformance_payload",
        lambda: {
            "abi_version": None,
            "errors": ["incomplete"],
            "format_version": "invarlock/runtime-provider-conformance-v1",
            "ok": False,
            "provider": None,
        },
    )
    assert conformance.main() == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
