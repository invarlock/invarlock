import json
from pathlib import Path

import numpy as np

from invarlock.core.events import EventLogger
from invarlock.core.types import LogLevel


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_event_logger_writes_and_redacts(tmp_path: Path):
    log_path = tmp_path / "events" / "run.jsonl"
    with EventLogger(log_path, run_id="run-123", auto_flush=True) as logger:
        # Basic event
        logger.log("runner", "start", LogLevel.INFO, {"token": "abc", "msg": "hello"})

        # Various data types and redaction in values
        long_str = "x" * 600
        data = {
            "password": "super-secret",  # key-based redaction
            "note": "contains api_key and secret inside",  # value-based redaction
            "blob": b"bytes",
            "array": np.array([1, 2, 3]),
            "setlike": {1, 2, 3},
            "frozen": frozenset({4, 5}),
            "nested": {"email": "user@example.com", "text": long_str},
            "seq": [
                {"auth": "token-xyz"},
                [b"x", b"y"],
            ],
        }
        logger.log("runner", "data", LogLevel.DEBUG, data)

        # Convenience helpers
        logger.log_metric("runner", "latency", 12.5, unit="ms/token")
        logger.log_checkpoint("runner", "ckpt-1", "create")
        try:
            raise ValueError("boom")
        except ValueError as e:
            logger.log_error("runner", "oops", e, context={"hint": "check inputs"})

    # Logger context closed; file should exist with events including session_start and session_end
    records = _read_jsonl(log_path)
    assert any(r.get("operation") == "session_start" for r in records)
    assert any(r.get("operation") == "session_end" for r in records)
    # run_id present in events
    assert all(
        r.get("run_id") == "run-123"
        for r in records
        if r.get("operation") != "session_end"
    )

    # Find data event and verify redactions/truncation/serialization
    data_events = [r for r in records if r.get("operation") == "data"]
    assert len(data_events) == 1
    payload = data_events[0].get("data", {})
    assert payload["password"] == "<redacted>"
    assert payload["note"] == "<redacted>"
    assert payload["blob"] == "<bytes len=5>"
    assert payload["array"] == [1, 2, 3]
    assert sorted(payload["setlike"]) == [1, 2, 3]
    assert sorted(payload["frozen"]) == [4, 5]
    assert payload["nested"]["email"] == "<redacted>"
    assert payload["nested"]["text"].startswith("<str len=")
    # Sequence redaction and bytes handling
    assert payload["seq"][0]["auth"] == "<redacted>"
    assert payload["seq"][1][0].startswith("<bytes len=")
