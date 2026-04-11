from pathlib import Path
from typing import Any, cast

import pytest

from invarlock.core.checkpoint import (
    CheckpointManager,
    PolicyCheckpoint,
    create_policy_checkpoint,
)
from invarlock.core.types import GuardOutcome


class DummyAdapter:
    def __init__(self, tmpdir: Path):
        self.tmpdir = tmpdir
        self.snapshots: list[str] = []
        self.restores: list[str] = []

    def snapshot(self, model: object) -> bytes:
        self.snapshots.append("bytes")
        return b"model-bytes"

    def restore(self, model: object, blob: bytes) -> None:
        assert blob == b"model-bytes"
        self.restores.append("bytes")

    def snapshot_chunked(self, model: object) -> str:
        path = self.tmpdir / "chunked_snapshot"
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunk_0.bin").write_bytes(b"data")
        self.snapshots.append("chunked")
        return str(path)

    def restore_chunked(self, model: object, snapshot_path: str) -> None:
        assert Path(snapshot_path).exists()
        self.restores.append("chunked")


def test_policy_checkpoint_bytes_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)
    adapter = DummyAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()

    cp = PolicyCheckpoint(model, adapter, policy)
    cp.create_checkpoint()
    assert cp.checkpoint_data and cp.checkpoint_data["mode"] == "bytes"

    # Rollback should call restore and succeed
    assert cp.rollback("test") is True
    assert adapter.restores == ["bytes"]
    cp.cleanup()  # should be a no-op for bytes


def test_policy_checkpoint_chunked_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    adapter = DummyAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()

    cp = PolicyCheckpoint(model, adapter, policy)
    cp.create_checkpoint()
    assert cp.checkpoint_data and cp.checkpoint_data["mode"] == "chunked"
    snapshot_path = cp.checkpoint_data["path"]
    assert Path(snapshot_path).exists()

    # Rollback should call restore_chunked and succeed
    assert cp.rollback("chunk_reason") is True
    assert adapter.restores == ["chunked"]

    # Cleanup should remove the chunked directory
    cp.cleanup()
    assert not Path(snapshot_path).exists()


def test_policy_checkpoint_should_rollback_logic():
    adapter = DummyAdapter(Path("/tmp"))
    model = object()
    policy = type("P", (), {"enable_auto_rollback": True})()
    cp = PolicyCheckpoint(model, adapter, policy)

    # Abort action takes priority
    outcomes = [
        GuardOutcome("a", True, decision="allow"),
        GuardOutcome("b", True, decision="block"),
    ]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "guard_abort"

    # Rollback action
    outcomes = [GuardOutcome("a", True, decision="rollback")]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "guard_rollback"

    # Auto rollback when enabled and a guard failed
    outcomes = [GuardOutcome("a", False, decision="allow")]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "auto_rollback"

    # Otherwise do not rollback
    outcomes = [GuardOutcome("a", True, decision="allow")]
    should, reason = cp.should_rollback(outcomes)
    assert not should and reason == ""


def test_policy_checkpoint_does_not_auto_rollback_when_policy_disabled() -> None:
    adapter = DummyAdapter(Path("/tmp"))
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()
    cp = PolicyCheckpoint(model, adapter, policy)

    should, reason = cp.should_rollback([GuardOutcome("a", False, decision="allow")])

    assert should is False
    assert reason == ""


def test_policy_checkpoint_rollback_guard_paths(tmp_path: Path):
    class CorruptBytesAdapter(DummyAdapter):
        def restore(self, model: object, blob: bytes | None) -> None:
            if blob is None:
                raise TypeError("missing blob")
            super().restore(model, blob)

    adapter = CorruptBytesAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()
    cp = PolicyCheckpoint(model, adapter, policy)

    # No checkpoint yet -> rollback returns False
    assert cp.rollback("nope") is False

    # Create bytes checkpoint then corrupt it to ensure graceful failure
    cp.create_checkpoint()
    checkpoint_data = cast(dict[str, Any], cp.checkpoint_data)
    checkpoint_data["blob"] = None
    assert cp.rollback("corrupt") is False


def test_policy_checkpoint_chunked_rollback_requires_path_and_restore(
    tmp_path: Path,
) -> None:
    class NoChunkRestoreAdapter(DummyAdapter):
        restore_chunked = None

    adapter = NoChunkRestoreAdapter(tmp_path)
    cp = PolicyCheckpoint(
        object(), adapter, type("P", (), {"enable_auto_rollback": False})()
    )
    cp.checkpoint_data = {"mode": "chunked", "path": ""}

    assert cp.rollback("missing_chunked_restore") is False


def test_policy_checkpoint_rollback_reraises_unexpected_errors(tmp_path: Path):
    class ExplodingAdapter(DummyAdapter):
        def restore(self, model: object, blob: bytes) -> None:
            raise AssertionError("explode")

    adapter = ExplodingAdapter(tmp_path)
    cp = PolicyCheckpoint(
        object(), adapter, type("P", (), {"enable_auto_rollback": False})()
    )
    cp.create_checkpoint()

    with pytest.raises(AssertionError, match="explode"):
        cp.rollback("unexpected")


def test_policy_checkpoint_cleanup_tolerates_missing_chunked_path(
    tmp_path: Path,
) -> None:
    adapter = DummyAdapter(tmp_path)
    cp = PolicyCheckpoint(
        object(), adapter, type("P", (), {"enable_auto_rollback": False})()
    )
    cp.checkpoint_data = {"mode": "chunked", "path": ""}

    cp.cleanup()

    assert cp.checkpoint_data is None


def test_create_policy_checkpoint_context_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    adapter = DummyAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()

    with create_policy_checkpoint(model, adapter, policy) as cp:
        assert cp.checkpoint_data and cp.checkpoint_data["mode"] == "chunked"
        checkpoint_data = cast(dict[str, Any], cp.checkpoint_data)
        path = Path(str(checkpoint_data["path"]))
        assert path.exists()
    # After context exit, path should be cleaned up
    assert not path.exists()


def test_checkpoint_manager_bytes_and_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = DummyAdapter(tmp_path)
    model = object()
    mgr = CheckpointManager()

    # Bytes
    monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)
    cid1 = mgr.create_checkpoint(model, adapter)
    assert cid1 in mgr.checkpoints
    assert mgr.restore_checkpoint(model, adapter, cid1) is True

    # Chunked
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    cid2 = mgr.create_checkpoint(model, adapter)
    assert cid2 in mgr.checkpoints
    assert mgr.restore_checkpoint(model, adapter, cid2) is True

    # Missing id
    assert mgr.restore_checkpoint(model, adapter, "missing") is False

    # Cleanup should remove chunked dirs and reset state
    # Ensure a chunked path still exists
    path = mgr.checkpoints[cid2]["path"]
    assert Path(path).exists()
    mgr.cleanup()
    assert mgr.checkpoints == {}
    assert mgr.next_id == 1
    assert not Path(path).exists()


def test_restore_checkpoint_chunked_missing_restore_chunked_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class ChunkedNoRestoreAdapter:
        def __init__(self, tmpdir: Path):
            self.tmpdir = tmpdir

        def snapshot_chunked(self, model: object) -> str:
            path = self.tmpdir / "chunked_snapshot_no_restore"
            path.mkdir(parents=True, exist_ok=True)
            (path / "chunk_0.bin").write_bytes(b"data")
            return str(path)

    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    adapter = ChunkedNoRestoreAdapter(tmp_path)
    model = object()
    mgr = CheckpointManager()

    cid = mgr.create_checkpoint(model, adapter)
    assert mgr.checkpoints[cid]["mode"] == "chunked"
    assert mgr.restore_checkpoint(model, adapter, cid) is False

    mgr.cleanup()


def test_checkpoint_manager_create_error_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class BadAdapter(DummyAdapter):
        def snapshot(self, model: object) -> bytes:
            raise RuntimeError("snap fail")

    adapter = BadAdapter(tmp_path)
    mgr = CheckpointManager()
    with pytest.raises(RuntimeError):
        _ = mgr.create_checkpoint(object(), adapter)


def test_checkpoint_manager_reraises_unexpected_restore_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class ExplodingAdapter(DummyAdapter):
        def restore(self, model: object, blob: bytes) -> None:
            raise AssertionError("explode")

    monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)
    adapter = ExplodingAdapter(tmp_path)
    mgr = CheckpointManager()
    checkpoint_id = mgr.create_checkpoint(object(), adapter)

    with pytest.raises(AssertionError, match="explode"):
        mgr.restore_checkpoint(object(), adapter, checkpoint_id)


def test_checkpoint_manager_cleanup_tolerates_missing_chunked_path() -> None:
    mgr = CheckpointManager()
    mgr.checkpoints["checkpoint_1"] = {"mode": "chunked", "path": ""}
    mgr.next_id = 2

    mgr.cleanup()

    assert mgr.checkpoints == {}
    assert mgr.next_id == 1
