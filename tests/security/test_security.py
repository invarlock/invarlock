"""Security utility tests."""

from __future__ import annotations

import socket
import threading

import pytest

from invarlock import security


def test_network_guard_blocks_connections():
    """Network guard should prevent outbound socket connections."""
    guard = security.NetworkGuard()
    guard.install()

    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80), timeout=0.1)
    with pytest.raises(RuntimeError):
        socket.socket().connect_ex(("127.0.0.1", 9))
    with pytest.raises(RuntimeError):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", 9))

    guard.restore()


def test_temporarily_allow_network_context():
    """Context manager should temporarily lift network restrictions."""
    security.enforce_network_policy(False)
    assert not security.network_policy_allows()

    with security.temporarily_allow_network():
        assert security.network_policy_allows()

    assert not security.network_policy_allows()
    security.enforce_network_policy(True)


def test_temporarily_allow_network_nested_context() -> None:
    security.enforce_network_policy(False)
    assert not security.network_policy_allows()

    with security.temporarily_allow_network():
        assert security.network_policy_allows()
        with security.temporarily_allow_network():
            assert security.network_policy_allows()
        assert security.network_policy_allows()

    assert not security.network_policy_allows()
    security.enforce_network_policy(True)


def test_temporarily_allow_network_concurrent_contexts() -> None:
    security.enforce_network_policy(False)
    entered_a = threading.Event()
    entered_b = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()
    failures: list[Exception] = []

    def _worker(entered: threading.Event, release: threading.Event) -> None:
        try:
            with security.temporarily_allow_network():
                if not security.network_policy_allows():
                    raise RuntimeError("network should be allowed in temporary context")
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("worker release timed out")
        except Exception as exc:  # pragma: no cover - explicit capture for threads
            failures.append(exc)

    thread_a = threading.Thread(target=_worker, args=(entered_a, release_a))
    thread_b = threading.Thread(target=_worker, args=(entered_b, release_b))
    thread_a.start()
    thread_b.start()

    assert entered_a.wait(timeout=5)
    assert entered_b.wait(timeout=5)
    # Allowance should stay scoped to the worker threads.
    assert not security.network_policy_allows()

    release_a.set()
    thread_a.join(timeout=5)
    assert not thread_a.is_alive()
    assert not security.network_policy_allows()

    release_b.set()
    thread_b.join(timeout=5)
    assert not thread_b.is_alive()
    assert not security.network_policy_allows()
    assert not failures

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
