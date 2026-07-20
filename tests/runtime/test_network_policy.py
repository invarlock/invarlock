"""Security utility tests."""

from __future__ import annotations

import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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

    def _worker(entered: threading.Event, release: threading.Event) -> None:
        with security.temporarily_allow_network():
            if not security.network_policy_allows():
                raise RuntimeError("network should be allowed in temporary context")
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("worker release timed out")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_worker, entered_a, release_a)
        future_b = executor.submit(_worker, entered_b, release_b)

        assert entered_a.wait(timeout=5)
        assert entered_b.wait(timeout=5)
        # Allowance should stay scoped to the worker threads.
        assert not security.network_policy_allows()

        release_a.set()
        future_a.result(timeout=5)
        assert not security.network_policy_allows()

        release_b.set()
        future_b.result(timeout=5)
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


def test_network_guard_covers_blocked_and_scoped_socket_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeSocket:
        family = socket.AF_UNIX

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def connect(self, address: object) -> None:
            calls.append(("connect", address))

        def connect_ex(self, address: object) -> int:
            calls.append(("connect_ex", address))
            return 7

        def sendto(self, data: object, *args: object) -> int:
            calls.append(("sendto", (data, args)))
            return 3

        def sendmsg(
            self,
            buffers: Any,
            ancdata: Any = (),
            flags: int = 0,
            address: Any | None = None,
        ) -> int:
            calls.append(("sendmsg", (buffers, ancdata, flags, address)))
            return 5

    create_calls: list[tuple[object, object, object]] = []

    def fake_create_connection(
        address: object,
        timeout: object = None,
        source_address: object = None,
    ) -> object:
        create_calls.append((address, timeout, source_address))
        return object()

    monkeypatch.setattr(security.socket, "socket", FakeSocket)
    monkeypatch.setattr(security.socket, "create_connection", fake_create_connection)
    guard = security.NetworkGuard()
    guard.install()
    guarded = security.socket.socket()

    with pytest.raises(RuntimeError, match="Network access disabled"):
        guarded.connect(("127.0.0.1", 1))
    with pytest.raises(RuntimeError, match="Network access disabled"):
        guarded.connect_ex(("127.0.0.1", 1))
    with pytest.raises(RuntimeError, match="Network access disabled"):
        guarded.sendto(b"x", ("127.0.0.1", 1))
    with pytest.raises(RuntimeError, match="Network access disabled"):
        guarded.sendmsg([b"x"], address=("127.0.0.1", 1))
    with pytest.raises(RuntimeError, match="Network access disabled"):
        security.socket.create_connection(("127.0.0.1", 1))

    with security.temporarily_allow_network():
        guarded.connect(("127.0.0.1", 1))
        assert guarded.connect_ex(("127.0.0.1", 1)) == 7
        assert guarded.sendto(b"x", ("127.0.0.1", 1)) == 3
        assert guarded.sendmsg([b"x"], address=("127.0.0.1", 1)) == 5
        assert (
            security.socket.create_connection(
                ("127.0.0.1", 1), timeout=2, source_address=("127.0.0.1", 2)
            )
            is not None
        )

    # A non-network family and non-tuple address delegates without a scope.
    guarded.connect("local.sock")
    assert calls
    assert create_calls == [(("127.0.0.1", 1), 2, ("127.0.0.1", 2))]
    guard.restore()


def test_security_private_fallbacks_and_missing_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = security._NETWORK_ALLOW_SCOPE.set("not-an-int")
    try:
        assert not security._context_allows_network()
    finally:
        security._NETWORK_ALLOW_SCOPE.reset(token)

    class LocalSocket:
        family = socket.AF_UNIX

    assert not security._is_network_target(LocalSocket(), "local.sock")
    assert security._is_network_target(LocalSocket(), ("host", 1))
    assert not security.is_secure_path(tmp_path / "missing")

    guard = security.NetworkGuard()
    guard.restore()
    guard.install()
    guard.install()
    guard.restore()
