from __future__ import annotations

import json
from pathlib import Path

from invarlock.core.events import EventLogger
from invarlock.core.types import LogLevel


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_event_logger_basic_and_sanitize(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path, auto_flush=True, run_id="RID", max_string_length=8)

    # Diverse payload covering branches
    class Obj:
        def __str__(self):
            return "OBJ"

    class ArrayLike:
        def tolist(self):
            return [1, 2, 3]

    payload = {
        "token": "abc",  # key-based redaction
        "note": "this contains password secret",  # value-based redaction
        "long": "x" * 50,  # truncation path
        "mapping": {"inner_token": "abc"},  # nested redaction
        "set_val": {1, 2},  # set → list
        "list_val": [b"b", 3.14, Obj()],  # sequence handling
        "bytes": b"hello",
        "array_like": ArrayLike(),  # tolist branch
        "none": None,
    }

    logger.log("test", "op", LogLevel.INFO, payload)
    logger.log_metric("runner", "throughput", 12.3, unit="sps")
    # unit omitted path
    logger.log_metric("runner", "latency", 1.0)
    logger.log_checkpoint("runner", "ckpt-1", "create")
    logger.log_error("guard", "fail", RuntimeError("boom"), context={"k": "v"})
    logger.close()

    lines = read_jsonl(log_path)
    # Expect session_start + 5 events + session_end
    assert len(lines) == 7

    # Find the payload event and verify sanitization
    evt = next(e for e in lines if e.get("operation") == "op")
    data = evt["data"]
    assert data["token"] == "<redacted>"
    assert data["note"] == "<redacted>"
    assert data["long"].startswith("<str len=")
    assert data["mapping"]["inner_token"] == "<redacted>"
    # set serialized to list
    assert sorted(data["set_val"]) == [1, 2]
    # bytes stringified placeholder in list
    assert (
        data["list_val"][0] == "<bytes len=1>" or data["list_val"][0] == "h"
    )  # serializer path tolerant
    # array_like converted
    assert data["array_like"] == [1, 2, 3]


def test_event_logger_context_manager_and_del(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLogger(log_path, auto_flush=False) as logger:
        logger.log("t", "x", LogLevel.INFO, {"a": 1})
    # file is closed and session_end written
    lines = read_jsonl(log_path)
    assert any(e.get("operation") == "session_end" for e in lines)

    # Create a logger without run_id to exercise branch
    log_path2 = tmp_path / "events2.jsonl"
    logger2 = EventLogger(log_path2, auto_flush=True)
    # simulate disabled file handle branch without disrupting context exit
    logger2._file = None  # type: ignore[attr-defined]
    logger2.log("t", "y", LogLevel.INFO, {"b": 2})
    logger2.close()
    lines2 = read_jsonl(log_path2)
    # Only session_start present since file handle disabled prevented writes
    assert lines2[0]["operation"] == "session_start"


def test_event_logger_serializer_branches() -> None:
    # Directly exercise serializer utility branches
    from invarlock.core.events import EventLogger

    logger = EventLogger(Path("/tmp/nonexistent.jsonl"), auto_flush=False)

    # tolist
    class A:
        def tolist(self):
            return [1]

    assert logger._json_serializer(A()) == [1]

    # __dict__ fallback
    class B:
        def __init__(self):
            self.x = 1

    assert isinstance(logger._json_serializer(B()), str)
    # set and bytes
    assert logger._json_serializer({1, 2}) in ([1, 2], [2, 1])
    assert isinstance(logger._json_serializer(b"x"), str)
