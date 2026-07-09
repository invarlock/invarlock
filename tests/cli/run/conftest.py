from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.runtime_security import RuntimeSecurityPolicy

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


class _DummyRunModel:
    def named_parameters(self):
        return []

    def named_buffers(self):
        return []


class _DummyRunAdapter:
    name = "hf_causal"

    def load_model(self, model_id: str, device: str | None = None, **_kwargs):
        return _DummyRunModel()


class _DummyRunRegistry:
    def get_adapter(self, name: str):
        return _DummyRunAdapter()

    def get_edit(self, name: str):
        return SimpleNamespace(name=name)

    def get_guard(self, name: str):
        return SimpleNamespace(name=name)

    def get_plugin_metadata(self, name: str, plugin_type: str):
        return {
            "name": name,
            "module": f"tests.{plugin_type}.{name}",
            "version": "test",
        }


@pytest.fixture(autouse=True)
def _default_run_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: _DummyRunRegistry(),
    )
    yield


@pytest.fixture(autouse=True)
def _default_run_host_execution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _VALID_TEST_IMAGE_DIGEST)
    monkeypatch.setattr(
        "invarlock.cli.config_execution.host_execution_allowed",
        lambda: True,
    )
    monkeypatch.setattr(
        "invarlock.cli.config_execution.resolve_shell_runtime_security_policy",
        lambda **_: RuntimeSecurityPolicy(allow_host_execution=True),
    )
    monkeypatch.setattr(
        "invarlock.cli.security_helpers.resolve_shell_runtime_security_policy",
        lambda **_: RuntimeSecurityPolicy(allow_host_execution=True),
    )
    yield
