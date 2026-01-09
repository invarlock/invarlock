from pathlib import Path

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

    def snapshot(self, model) -> bytes:  # type: ignore[no-untyped-def]
        self.snapshots.append("bytes")
        return b"model-bytes"

    def restore(self, model, blob: bytes) -> None:  # type: ignore[no-untyped-def]
        assert blob == b"model-bytes"
        self.restores.append("bytes")

    def snapshot_chunked(self, model) -> str:  # type: ignore[no-untyped-def]
        path = self.tmpdir / "chunked_snapshot"
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunk_0.bin").write_bytes(b"data")
        self.snapshots.append("chunked")
        return str(path)

    def restore_chunked(self, model, snapshot_path: str) -> None:  # type: ignore[no-untyped-def]
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
        GuardOutcome("a", True, action="none"),
        GuardOutcome("b", True, action="abort"),
    ]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "guard_abort"

    # Rollback action
    outcomes = [GuardOutcome("a", True, action="rollback")]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "guard_rollback"

    # Auto rollback when enabled and a guard failed
    outcomes = [GuardOutcome("a", False, action="none")]
    should, reason = cp.should_rollback(outcomes)
    assert should and reason == "auto_rollback"

    # Otherwise do not rollback
    outcomes = [GuardOutcome("a", True, action="none")]
    should, reason = cp.should_rollback(outcomes)
    assert not should and reason == ""


def test_policy_checkpoint_rollback_guard_paths(tmp_path: Path):
    adapter = DummyAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()
    cp = PolicyCheckpoint(model, adapter, policy)

    # No checkpoint yet -> rollback returns False
    assert cp.rollback("nope") is False

    # Create bytes checkpoint then corrupt it to ensure graceful failure
    cp.create_checkpoint()
    cp.checkpoint_data["blob"] = None  # type: ignore[index]
    assert cp.rollback("corrupt") is False


def test_create_policy_checkpoint_context_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
    adapter = DummyAdapter(tmp_path)
    model = object()
    policy = type("P", (), {"enable_auto_rollback": False})()

    with create_policy_checkpoint(model, adapter, policy) as cp:
        assert cp.checkpoint_data and cp.checkpoint_data["mode"] == "chunked"
        path = Path(cp.checkpoint_data["path"])  # type: ignore[index]
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

        def snapshot_chunked(self, model) -> str:  # type: ignore[no-untyped-def]
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
        def snapshot(self, model) -> bytes:  # type: ignore[override]
            raise RuntimeError("snap fail")

    adapter = BadAdapter(tmp_path)
    mgr = CheckpointManager()
    with pytest.raises(RuntimeError):
        _ = mgr.create_checkpoint(object(), adapter)
