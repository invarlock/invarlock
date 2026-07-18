from __future__ import annotations

from types import SimpleNamespace

import pytest
from invarlock_addins.multimodal import conformance
from invarlock_addins.multimodal.conformance import conformance_payload
from invarlock_addins.multimodal.provider import HFVisionTextProvider


def test_multimodal_addin_conforms_to_the_public_provider_abi() -> None:
    assert conformance_payload() == {
        "abi_version": "1",
        "errors": [],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": True,
        "provider": "hf_vision_text",
    }


def test_multimodal_conformance_rejects_a_non_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conformance, "HFVisionTextProvider", lambda: object())

    assert conformance_payload() == {
        "abi_version": None,
        "errors": ["provider does not implement RuntimeProvider"],
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": False,
        "provider": None,
    }


def test_multimodal_conformance_reports_identity_capability_and_preflight_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HFVisionTextProvider, "name", "unexpected")
    monkeypatch.setattr(HFVisionTextProvider, "abi_version", "unexpected")
    monkeypatch.setattr(
        HFVisionTextProvider,
        "capabilities",
        lambda _self: SimpleNamespace(tasks=(), metrics=()),
    )
    monkeypatch.setattr(
        conformance,
        "_exercise_host_input_preflight",
        lambda _provider: (_ for _ in ()).throw(ValueError("invalid media")),
    )

    payload = conformance_payload()

    assert payload["ok"] is False
    assert payload["errors"] == [
        "provider name must be hf_vision_text",
        "provider ABI does not match the installed core",
        "provider must expose only vision_text_generation",
        "provider must expose only exact_match",
        "host input preflight failed: invalid media",
    ]


def test_multimodal_conformance_main_returns_the_payload_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        conformance,
        "conformance_payload",
        lambda: {"ok": False, "errors": ["unavailable"]},
    )

    assert conformance.main() == 1
    assert capsys.readouterr().out == '{"errors":["unavailable"],"ok":false}\n'
