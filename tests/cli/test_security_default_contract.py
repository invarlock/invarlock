from __future__ import annotations

from types import SimpleNamespace

import pytest

import invarlock.core.registry as registry_mod
import invarlock.runtime_security as runtime_security
from invarlock.cli.run_config import extract_model_load_kwargs
from invarlock.core.exceptions import InvarlockError


def test_third_party_plugin_discovery_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")

    def _entry_points_disabled() -> None:
        raise AssertionError("entry_points should not be called by default")

    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        _entry_points_disabled,
        raising=True,
    )

    registry = registry_mod.CoreRegistry()
    assert "hf_causal" in registry.list_adapters()
    assert "invariants" in registry.list_guards()


def test_model_trust_remote_code_requires_explicit_allow() -> None:
    cfg = SimpleNamespace(model_dump=lambda: {"model": {"trust_remote_code": True}})

    with pytest.raises(InvarlockError, match="REMOTE-CODE-DISABLED"):
        extract_model_load_kwargs(cfg, invarlock_error_cls=InvarlockError)


def test_container_launch_requires_local_image_when_network_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: False,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="runtime image"):
        runtime_security.build_container_command(["evaluate", "--help"])


def test_runtime_image_prefers_local_build_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: image == "invarlock-runtime:local",
        raising=True,
    )

    assert runtime_security.resolve_runtime_image() == "invarlock-runtime:local"


def test_runtime_image_defaults_to_registry_when_local_build_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: False,
        raising=True,
    )

    assert (
        runtime_security.resolve_runtime_image()
        == "ghcr.io/invarlock/invarlock-runtime:latest"
    )


def test_container_launch_uses_runtime_image_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    command = runtime_security.build_container_command(["evaluate", "--help"])

    assert command[-3:] == ["invarlock-runtime:local", "evaluate", "--help"]
    assert "python" not in command[command.index("invarlock-runtime:local") + 1 :]
