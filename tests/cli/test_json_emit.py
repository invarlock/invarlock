from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import typer

from invarlock.cli._json import emit


@dataclass
class _Payload:
    a: int
    b: str


def test_emit_dataclass_adds_defaults_and_exits(capsys):
    with pytest.raises(typer.Exit) as ei:
        emit(_Payload(1, "x"), exit_code=0)
    assert ei.value.exit_code == 0
    out = capsys.readouterr().out.strip()
    obj = json.loads(out)
    assert obj["a"] == 1 and obj["b"] == "x"
    assert obj.get("component") == "cli"
    assert isinstance(obj.get("ts"), str)


def test_emit_dict_preserves_existing_fields(capsys):
    payload = {"component": "custom", "ts": "T", "k": 1}
    with pytest.raises(typer.Exit) as ei:
        emit(payload, exit_code=2)
    assert ei.value.exit_code == 2
    out = capsys.readouterr().out.strip()
    obj = json.loads(out)
    assert obj["k"] == 1
    # Should not overwrite existing values
    assert obj["component"] == "custom"
    assert obj["ts"] == "T"


def test_emit_non_mapping_payload(capsys):
    # Non-dict payload should still JSON-dump without defaults injection
    with pytest.raises(typer.Exit) as ei:
        emit([1, 2, 3], exit_code=3)
    assert ei.value.exit_code == 3
    out = capsys.readouterr().out.strip()
    obj = json.loads(out)
    assert obj == [1, 2, 3]
