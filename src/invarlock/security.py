"""
InvarLock Security Utilities
========================

Runtime hardening helpers used by the CLI and automation surfaces.

- Network guard: disables outbound socket connections unless explicitly allowed.
- Secure temporary directory helper ensuring 0o700 permissions and cleanup.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import stat
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "NetworkGuard",
    "enforce_network_policy",
    "enforce_default_security",
    "temporarily_allow_network",
    "network_policy_allows",
    "secure_tempdir",
    "is_secure_path",
]

_NETWORK_DISABLED_ERROR = RuntimeError(
    "Network access disabled by InvarLock security policy. "
    "Set INVARLOCK_ALLOW_NETWORK=1 if connectivity is required."
)


class NetworkGuard:
    """Installable network guard that blocks outbound socket connections."""

    def __init__(self) -> None:
        self._installed = False
        self._original_socket_cls: type[socket.socket] | None = None
        self._original_create_connection: Callable[..., socket.socket] | None = None

    @property
    def installed(self) -> bool:
        """Whether the guard is currently blocking network access."""
        return self._installed

    def install(self) -> None:
        """Install the guard if not already active."""
        if self._installed:
            return

        self._original_socket_cls = socket.socket
        self._original_create_connection = socket.create_connection

        guard_error = _NETWORK_DISABLED_ERROR

        class GuardedSocket(socket.socket):
            """Socket subclass that blocks connect calls."""

            def connect(self_inner, address: Any) -> None:
                raise guard_error

        def guarded_create_connection(
            address: Any,
            timeout: float | None = None,
            source_address: Any | None = None,
        ) -> socket.socket:
            raise guard_error

        setattr(socket, "socket", GuardedSocket)  # noqa: B010
        setattr(socket, "create_connection", guarded_create_connection)  # noqa: B010
        self._installed = True

    def restore(self) -> None:
        """Restore the original socket implementations."""
        if not self._installed:
            return

        if self._original_socket_cls is not None:
            setattr(socket, "socket", self._original_socket_cls)  # noqa: B010
        if self._original_create_connection is not None:
            setattr(socket, "create_connection", self._original_create_connection)  # noqa: B010
        self._installed = False


_GUARD = NetworkGuard()
_NETWORK_POLICY_LOCK = threading.RLock()
_NETWORK_ALLOW_SCOPE_COUNT = 0
_NETWORK_BLOCKED = _GUARD.installed


def enforce_network_policy(allow: bool) -> None:
    """
    Apply the global network policy.

    Args:
        allow: When True the guard is removed, otherwise network access is blocked.
    """
    global _NETWORK_BLOCKED
    with _NETWORK_POLICY_LOCK:
        _NETWORK_BLOCKED = not bool(allow)
        if _NETWORK_BLOCKED:
            if _NETWORK_ALLOW_SCOPE_COUNT == 0:
                _GUARD.install()
        else:
            _GUARD.restore()


def network_policy_allows() -> bool:
    """Return True if outbound connections are currently permitted."""
    with _NETWORK_POLICY_LOCK:
        return not _GUARD.installed


def enforce_default_security() -> None:
    """
    Enforce default runtime security posture.

    Network access is denied unless INVARLOCK_ALLOW_NETWORK is set to a truthy value.
    """
    allow_env = os.environ.get("INVARLOCK_ALLOW_NETWORK", "")
    allow_network = allow_env.strip().lower() in {"1", "true", "yes", "on"}
    enforce_network_policy(allow_network)


@contextlib.contextmanager
def temporarily_allow_network() -> Iterator[None]:
    """
    Temporarily allow network access inside the context block.

    Restores the previous policy when exiting the context.
    """
    global _NETWORK_ALLOW_SCOPE_COUNT
    with _NETWORK_POLICY_LOCK:
        _NETWORK_ALLOW_SCOPE_COUNT += 1
        _GUARD.restore()
    try:
        yield
    finally:
        with _NETWORK_POLICY_LOCK:
            _NETWORK_ALLOW_SCOPE_COUNT = max(0, _NETWORK_ALLOW_SCOPE_COUNT - 1)
            if _NETWORK_ALLOW_SCOPE_COUNT == 0 and _NETWORK_BLOCKED:
                _GUARD.install()


@contextlib.contextmanager
def secure_tempdir(
    prefix: str = "invarlock-", base_dir: str | os.PathLike[str] | None = None
) -> Iterator[Path]:
    """
    Create a temporary directory with 0o700 permissions that is removed on exit.

    Args:
        prefix: Directory name prefix.
        base_dir: Optional base directory.

    Yields:
        Path to the secure temporary directory.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=base_dir))
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def is_secure_path(path: Path) -> bool:
    """
    Check whether a path has secure (0o700) permissions.

    Args:
        path: Path to validate.

    Returns:
        True if the path exists and has 0o700 permissions, False otherwise.
    """
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_IMODE(mode) == 0o700
