from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.cli.run_runtime_evidence import (
    capture_backend_inventory,
    capture_runtime_quantization_proof,
)


class _ConfigError(Exception):
    pass


def test_backend_inventory_is_bound_to_context_and_event_directory(
    tmp_path: Path,
) -> None:
    run_config = SimpleNamespace(
        context={}, event_path=tmp_path / "run" / "events.jsonl"
    )
    calls: list[dict[str, object]] = []

    capture_backend_inventory(
        adapter=SimpleNamespace(name="hf_bnb"),
        cfg=object(),
        model=object(),
        run_config=run_config,
        extract_load_kwargs=lambda *_args, **_kwargs: {
            "quantization_config": {"load_in_8bit": True}
        },
        error_type=_ConfigError,
        build_inventory=lambda **kwargs: calls.append(kwargs) or {"backend": "bnb"},
        filename="backend-inventory.json",
    )

    assert calls[0]["adapter"] == "hf_bnb"
    assert calls[0]["quantization_config"] == {"load_in_8bit": True}
    assert run_config.context["_backend_inventory"] == {"backend": "bnb"}
    assert json.loads(
        (tmp_path / "run" / "backend-inventory.json").read_text(encoding="utf-8")
    ) == {"backend": "bnb"}


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("missing"),
        KeyError("missing"),
        TypeError("bad"),
        ValueError("bad"),
        _ConfigError("bad"),
    ],
)
def test_backend_inventory_uses_empty_quantization_config_when_loading_fails(
    error: Exception,
) -> None:
    observed: list[dict[str, object]] = []

    def fail(*_args, **_kwargs):
        raise error

    capture_backend_inventory(
        adapter=SimpleNamespace(name=None),
        cfg=object(),
        model=object(),
        run_config=SimpleNamespace(context=None, event_path=None),
        extract_load_kwargs=fail,
        error_type=_ConfigError,
        build_inventory=lambda **kwargs: observed.append(kwargs) or {"ok": True},
        filename="unused.json",
    )

    assert observed[0]["adapter"] == ""
    assert observed[0]["quantization_config"] == {}


def test_backend_inventory_none_and_unserializable_evidence_are_nonfatal(
    tmp_path: Path,
) -> None:
    common = {
        "adapter": object(),
        "cfg": object(),
        "model": object(),
        "extract_load_kwargs": lambda *_args, **_kwargs: {
            "quantization_config": "not-a-mapping"
        },
        "error_type": _ConfigError,
        "filename": "backend-inventory.json",
    }
    capture_backend_inventory(
        **common,
        run_config=SimpleNamespace(context={}, event_path=tmp_path / "events.jsonl"),
        build_inventory=lambda **_kwargs: None,
    )
    capture_backend_inventory(
        **common,
        run_config=SimpleNamespace(context={}, event_path=tmp_path / "events.jsonl"),
        build_inventory=lambda **_kwargs: {"not_json": object()},
    )
    assert not (tmp_path / "backend-inventory.json").exists()


def test_quantization_proof_is_bound_to_context_and_sidecar(tmp_path: Path) -> None:
    run_config = SimpleNamespace(context={}, event_path=tmp_path / "events.jsonl")
    writes: list[tuple[Path, dict[str, object]]] = []

    capture_runtime_quantization_proof(
        adapter=SimpleNamespace(name="gptq"),
        model=object(),
        run_config=run_config,
        build_proof=lambda **_kwargs: {"ok": True},
        write_sidecar=lambda path, proof: writes.append((path, proof)),
    )

    assert run_config.context["_runtime_quantization_proof"] == {"ok": True}
    assert writes == [(tmp_path, {"ok": True})]


@pytest.mark.parametrize("failure", ["build", "none", "write"])
def test_quantization_proof_capture_failures_do_not_block_normal_runs(
    tmp_path: Path,
    failure: str,
) -> None:
    context: object = {} if failure != "none" else None

    def build_proof(**_kwargs):
        if failure == "build":
            raise RuntimeError("backend inspection failed")
        if failure == "none":
            return None
        return {"ok": False}

    def write_sidecar(_path: Path, _proof: dict[str, object]) -> None:
        raise OSError("disk unavailable")

    capture_runtime_quantization_proof(
        adapter=SimpleNamespace(name=None),
        model=object(),
        run_config=SimpleNamespace(
            context=context, event_path=tmp_path / "events.jsonl"
        ),
        build_proof=build_proof,
        write_sidecar=write_sidecar,
    )

    if isinstance(context, dict) and failure == "write":
        assert context["_runtime_quantization_proof"] == {"ok": False}


def test_quantization_proof_without_event_path_updates_context_only() -> None:
    context: dict[str, object] = {}
    capture_runtime_quantization_proof(
        adapter=object(),
        model=object(),
        run_config=SimpleNamespace(context=context, event_path=None),
        build_proof=lambda **_kwargs: {"ok": True},
        write_sidecar=lambda *_args: pytest.fail("sidecar must not be written"),
    )
    assert context["_runtime_quantization_proof"] == {"ok": True}


def test_quantization_proof_writes_sidecar_without_mutable_context(
    tmp_path: Path,
) -> None:
    writes: list[Path] = []
    capture_runtime_quantization_proof(
        adapter=object(),
        model=object(),
        run_config=SimpleNamespace(context=None, event_path=tmp_path / "events.jsonl"),
        build_proof=lambda **_kwargs: {"ok": True},
        write_sidecar=lambda path, _proof: writes.append(path),
    )
    assert writes == [tmp_path]
