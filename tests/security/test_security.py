"""Security utility tests."""

from __future__ import annotations

import socket

import pytest

from invarlock import security


def test_network_guard_blocks_connections():
    """Network guard should prevent outbound socket connections."""
    guard = security.NetworkGuard()
    guard.install()

    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80), timeout=0.1)

    guard.restore()


def test_temporarily_allow_network_context():
    """Context manager should temporarily lift network restrictions."""
    security.enforce_network_policy(False)
    assert not security.network_policy_allows()

    with security.temporarily_allow_network():
        assert security.network_policy_allows()

    assert not security.network_policy_allows()
    security.enforce_network_policy(True)


def test_secure_tempdir_creates_secure_directory():
    """Secure tempdir should enforce 0o700 permissions and clean up."""
    with security.secure_tempdir() as tmp_path:
        assert tmp_path.exists()
        assert security.is_secure_path(tmp_path)
        marker = tmp_path / "marker.txt"
        marker.write_text("ok", encoding="utf-8")
        assert marker.exists()

    assert not tmp_path.exists()


def test_enforce_default_security_respects_environment(monkeypatch):
    """Environment variable should control network policy."""
    monkeypatch.delenv("INVARLOCK_ALLOW_NETWORK", raising=False)
    security.enforce_default_security()
    assert not security.network_policy_allows()

    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")
    security.enforce_default_security()
    assert security.network_policy_allows()

    security.enforce_network_policy(True)
