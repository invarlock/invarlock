from __future__ import annotations

import json
from importlib import import_module as stdlib_import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.cli import run_runtime_exec as run_runtime
from invarlock.core import runtime_quantization_proof as runtime_proof
from invarlock.core.runtime_quantization_proof import (
    RUNTIME_QUANTIZATION_PROOF_FILENAME,
)


class _DenseModel:
    def modules(self):
        return [self]


_BnbLinear8bitLt = type(
    "Linear8bitLt",
    (),
    {"__module__": "bitsandbytes.nn.modules"},
)


class _QuantizedModel:
    def modules(self):
        return [self, _BnbLinear8bitLt()]


def test_runtime_exec_persists_fail_closed_quantization_proof(tmp_path: Path) -> None:
    run_config = SimpleNamespace(
        context={},
        event_path=tmp_path / "events.jsonl",
    )

    run_runtime._capture_runtime_quantization_proof(
        adapter=SimpleNamespace(name="hf_bnb"),
        model=_DenseModel(),
        run_config=run_config,
    )

    proof_path = tmp_path / RUNTIME_QUANTIZATION_PROOF_FILENAME
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["reason"] == "no_recognized_quantized_runtime_types"
    assert run_config.context["_runtime_quantization_proof"] == payload


def test_runtime_exec_persists_live_quantization_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def import_for_test(module_name: str):
        if module_name == "bitsandbytes.nn.modules":
            return SimpleNamespace(Linear8bitLt=_BnbLinear8bitLt)
        return stdlib_import_module(module_name)

    monkeypatch.setattr(runtime_proof, "import_module", import_for_test)
    run_config = SimpleNamespace(
        context={},
        event_path=tmp_path / "events.jsonl",
    )

    run_runtime._capture_runtime_quantization_proof(
        adapter=SimpleNamespace(name="hf_bnb"),
        model=_QuantizedModel(),
        run_config=run_config,
    )

    payload = json.loads(
        (tmp_path / RUNTIME_QUANTIZATION_PROOF_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["ok"] is True
    assert payload["recognized_quantized_runtime_type_count"] == 1


def test_runtime_quantization_proof_capture_is_non_fatal(monkeypatch) -> None:
    run_config = SimpleNamespace(context={}, event_path=Path("events.jsonl"))

    def fail_write(*_args, **_kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(
        run_runtime,
        "write_runtime_quantization_proof_sidecar",
        fail_write,
    )

    run_runtime._capture_runtime_quantization_proof(
        adapter=SimpleNamespace(name="hf_bnb"),
        model=_DenseModel(),
        run_config=run_config,
    )

    assert run_config.context["_runtime_quantization_proof"]["ok"] is False


def test_execute_guarded_run_captures_runtime_quantization_proof(monkeypatch) -> None:
    model = _DenseModel()
    captured: list[object] = []
    monkeypatch.setattr(
        run_runtime,
        "_capture_runtime_quantization_proof",
        lambda *, model, **_kwargs: captured.append(model),
    )
    monkeypatch.setattr(run_runtime, "release_process_memory", lambda: None)

    report, returned_model = run_runtime.execute_guarded_run(
        runner=SimpleNamespace(execute=lambda **_kwargs: {"status": "success"}),
        adapter=SimpleNamespace(name="hf_bnb"),
        model=model,
        cfg=SimpleNamespace(model=SimpleNamespace(id="demo")),
        edit_op=None,
        run_config=SimpleNamespace(event_path=None, context={}),
        guards=[],
        calibration_data=[],
        auto_config=None,
        edit_config={},
        preview_count=1,
        final_count=1,
        restore_fn=None,
        resolved_device="cpu",
        skip_model_load=True,
    )

    assert report == {"status": "success"}
    assert returned_model is model
    assert captured == [model]
