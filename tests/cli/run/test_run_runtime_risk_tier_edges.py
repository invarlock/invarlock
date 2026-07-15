from __future__ import annotations

import json
import logging
import warnings
from types import SimpleNamespace

import pytest

from invarlock.cli import run_execution
from invarlock.cli import run_runtime_exec as runtime
from invarlock.cli import run_runtime_warnings as warning_mod
from invarlock.cli.commands import doctor
from invarlock.cli.run_runtime_snapshot import SnapshotRestoreFailed
from invarlock.core.checkpoint_identity import CheckpointIdentityError
from invarlock.core.exceptions import ValidationError


def test_emit_and_postprocess_run_outputs_preserve_paths(monkeypatch, tmp_path):
    events = []
    report_path = tmp_path / "report.json"
    report_module = SimpleNamespace(
        save_report=lambda report, out_dir, formats, filename_prefix: {
            "json": report_path
        }
    )
    monkeypatch.setattr(
        run_execution.importlib, "import_module", lambda _name: report_module
    )
    monkeypatch.setattr(
        run_execution,
        "_event",
        lambda _console, tag, message, **kwargs: events.append((tag, message, kwargs)),
    )

    saved = run_execution.emit_run_outputs(
        report={"status": "success"},
        out_dir=tmp_path,
        filename_prefix="report",
        console=object(),
    )
    assert saved == {"json": str(report_path)}

    completed = run_execution.postprocess_and_summarize(
        report={"status": "success"},
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        console=object(),
        saved_files=saved,
    )
    assert completed is saved
    assert any("Events:" in message for _tag, message, _kwargs in events)

    monkeypatch.setattr(
        run_execution,
        "emit_run_outputs",
        lambda **_kwargs: {"json": str(report_path)},
    )
    without_event = run_execution.postprocess_and_summarize(
        report={"status": "success"},
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=None),
        console=object(),
        saved_files=None,
    )
    assert without_event == {"json": str(report_path)}


def test_run_execution_output_helpers_delegate_exact_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(run_execution, "console", "console")
    monkeypatch.setattr(
        run_execution.run_execution_output_mod,
        "emit_console_line",
        lambda *args, **kwargs: calls.append(("line", args, kwargs)),
    )
    monkeypatch.setattr(
        run_execution.run_execution_output_mod,
        "emit_console_blank_line",
        lambda *args: calls.append(("blank", args, {})),
    )
    monkeypatch.setattr(
        run_execution.run_execution_output_mod,
        "begin_progress_step",
        lambda *args: calls.append(("begin", args, {})),
    )
    monkeypatch.setattr(
        run_execution.run_execution_output_mod,
        "transition_progress_step",
        lambda *args, **kwargs: calls.append(("transition", args, kwargs)),
    )

    run_execution._emit_console_line("hello", markup=True)
    run_execution._emit_console_blank_line()
    run_execution._begin_progress_step("load_model")
    run_execution._transition_progress_step(
        "load_model",
        from_tag="INIT",
        from_message="loaded",
        to_key="execute",
        from_emoji="ok",
    )

    assert calls[0] == ("line", ("console", "hello"), {"markup": True})
    assert calls[1][0:2] == ("blank", ("console",))
    assert calls[2][0:2] == ("begin", ("console", "load_model"))
    assert calls[3][2]["to_key"] == "execute"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (None, 1),
        (SimpleNamespace(code="baseline_windows_missing", error=None), 3),
        (SimpleNamespace(code="unknown_edit", error=None), 2),
        (SimpleNamespace(code="schema_invalid_run_report", error=None), 2),
        (SimpleNamespace(code="torch_missing", error=None), 1),
    ],
)
def test_exit_code_for_structured_failures(failure, expected):
    assert run_execution._exit_code_for_failure(failure, profile="ci") == expected


def test_validation_failure_uses_stable_shell_exit_code():
    error = ValidationError(code="E402", message="Invalid tier: impossible")
    failure = SimpleNamespace(code="validation", message=error.message, error=error)
    assert run_execution._exit_code_for_failure(failure, profile="ci") == 1


def test_optional_runtime_cache_and_memory_recovery_paths(monkeypatch):
    loaded = {"psutil": object(), "torch": object()}
    monkeypatch.setattr(runtime, "psutil", None)
    monkeypatch.setattr(runtime, "torch", None)
    monkeypatch.setattr(runtime, "_import_optional_module", loaded.__getitem__)
    runtime.reset_optional_runtime_caches()
    assert runtime.psutil is loaded["psutil"]
    assert runtime.torch is loaded["torch"]

    monkeypatch.setattr(
        runtime.gc,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("gc unavailable")),
    )
    monkeypatch.setattr(runtime, "get_torch", lambda: None)
    monkeypatch.setattr(runtime, "_malloc_trim", lambda: True)
    runtime.release_process_memory()

    monkeypatch.setattr(
        runtime,
        "release_process_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    runtime.free_model_memory(object())
    runtime.free_model_memory(None)


def test_malloc_trim_success_missing_symbol_and_loader_failure(monkeypatch):
    class _Trim:
        argtypes = None
        restype = None

        def __call__(self, _amount):
            return 1

    trim = _Trim()
    monkeypatch.setattr(
        runtime, "CDLL", lambda _name: SimpleNamespace(malloc_trim=trim)
    )
    assert runtime._malloc_trim() is True
    assert trim.argtypes == [runtime.c_size_t]
    assert trim.restype is runtime.c_int

    monkeypatch.setattr(runtime, "CDLL", lambda _name: SimpleNamespace())
    assert runtime._malloc_trim() is False
    monkeypatch.setattr(
        runtime,
        "CDLL",
        lambda _name: (_ for _ in ()).throw(OSError("no libc")),
    )
    assert runtime._malloc_trim() is False


def test_load_model_rejects_missing_id_after_both_config_paths_fail():
    class _BrokenConfig:
        @property
        def model(self):
            raise AttributeError("no model")

        def model_dump(self):
            raise ValueError("no dump")

    with pytest.raises(ValueError, match="Missing model.id"):
        runtime.load_model_with_cfg(object(), _BrokenConfig(), "cpu")


def test_load_model_signature_failure_still_honors_local_only(monkeypatch):
    calls = []

    class _Config:
        model = SimpleNamespace(id="model-id")

        def model_dump(self):
            raise ValueError("unavailable")

    class _Adapter:
        def load_model(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

    monkeypatch.setattr(
        "invarlock.cli.run_config.extract_model_load_kwargs", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        runtime.inspect,
        "signature",
        lambda _callable: (_ for _ in ()).throw(ValueError("opaque callable")),
    )

    result = runtime.load_model_with_cfg(
        _Adapter(), _Config(), "cpu", prefer_local_files_only=True
    )
    assert result == "loaded"
    assert calls == [
        (("model-id",), {"device": "cpu", "prefer_local_files_only": True})
    ]


def test_load_model_strict_local_only_and_filtered_kwargs(monkeypatch):
    calls = []

    class _Config:
        model = SimpleNamespace(id="model-id")

        def model_dump(self):
            return {"model": {"id": "model-id"}}

    class _Adapter:
        def load_model(
            self, model_id, device, *, prefer_local_files_only=False, revision=None
        ):
            calls.append((model_id, device, prefer_local_files_only, revision))
            return "loaded"

    monkeypatch.setattr(
        "invarlock.cli.run_config.extract_model_load_kwargs",
        lambda *_a, **_k: {"unknown": 1, "revision": "commit"},
    )
    result = runtime.load_model_with_cfg(
        _Adapter(), _Config(), "cpu", prefer_local_files_only=True
    )

    assert result == "loaded"
    assert calls == [("model-id", "cpu", True, "commit")]


def test_load_model_rejects_malformed_typed_identity():
    class _Config:
        model = SimpleNamespace(id="model-id")

        def model_dump(self):
            return {"model": {"id": "model-id", "model_identity": {"kind": "bad"}}}

    with pytest.raises(CheckpointIdentityError, match="malformed"):
        runtime.load_model_with_cfg(object(), _Config(), "cpu")


@pytest.mark.parametrize("context", [None, {"run_id": ""}])
def test_execute_guarded_reload_handles_absent_run_context(monkeypatch, context):
    loaded_model = object()
    release_calls = []
    monkeypatch.setattr(runtime, "load_model_with_cfg", lambda *_a, **_k: loaded_model)
    monkeypatch.setattr(runtime, "_capture_backend_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime, "_capture_runtime_quantization_proof", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        runtime, "release_process_memory", lambda: release_calls.append("released")
    )
    runner = SimpleNamespace(execute=lambda **kwargs: {"model": kwargs["model"]})
    run_config = SimpleNamespace(verbose=False, event_path=None)
    if context is not None:
        run_config.context = context

    report, returned_model = runtime.execute_guarded_run(
        runner=runner,
        adapter=object(),
        model=None,
        cfg=object(),
        edit_op=object(),
        run_config=run_config,
        guards=[],
        calibration_data=[],
        auto_config=None,
        edit_config={},
        preview_count=1,
        final_count=1,
        restore_fn=None,
        resolved_device="cpu",
        snapshot_provenance=None,
    )

    assert report == {"model": loaded_model}
    assert returned_model is loaded_model
    assert release_calls == ["released"]


def test_execute_guarded_run_restore_failure_is_typed(monkeypatch):
    monkeypatch.setattr(runtime, "_capture_backend_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime, "_capture_runtime_quantization_proof", lambda **_kwargs: None
    )

    with pytest.raises(SnapshotRestoreFailed, match="restore failed"):
        runtime.execute_guarded_run(
            runner=object(),
            adapter=object(),
            model=object(),
            cfg=object(),
            edit_op=object(),
            run_config=SimpleNamespace(verbose=False, event_path=None),
            guards=[],
            calibration_data=[],
            auto_config=None,
            edit_config={},
            preview_count=1,
            final_count=1,
            restore_fn=lambda: (_ for _ in ()).throw(RuntimeError("restore failed")),
            resolved_device="cpu",
        )


class _RawStream:
    encoding = "utf-8"
    errors = "strict"
    buffer = object()
    closed = False

    def __init__(self) -> None:
        self.writes = []

    def fileno(self) -> int:
        return 7

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def write(self, value):
        self.writes.append(value)
        return 11

    def flush(self) -> None:
        raise OSError("closed")


def test_filtered_warning_stream_preserves_file_protocol_and_fallback_write():
    raw = _RawStream()
    sink = []
    stream = warning_mod.FilteredWarningStream(
        raw, [warning_mod.re.compile("suppress me")], sink
    )

    assert stream.encoding == "utf-8"
    assert stream.errors == "strict"
    assert stream.buffer is raw.buffer
    assert stream.closed is False
    assert stream.fileno() == 7
    assert stream.isatty() is True
    assert stream.writable() is False
    assert stream.write(b"visible\n") == len("visible\n")
    assert stream.write("suppress me\nvisible too\n") == len(
        "suppress me\nvisible too\n"
    )
    stream.flush()

    class _Unrenderable:
        def __str__(self):
            raise ValueError("cannot render")

    assert stream.write(_Unrenderable()) == 11
    assert sink == ["suppress me"]
    assert raw.writes[-1].__class__.__name__ == "_Unrenderable"


def test_suppressed_warning_is_recorded_and_environment_restored(monkeypatch, tmp_path):
    event_path = tmp_path / "events" / "warnings.jsonl"
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "info")
    monkeypatch.setattr(
        warning_mod.warnings,
        "formatwarning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("format")),
    )

    with warning_mod.suppress_noisy_warnings(
        "release", event_path=event_path, context={"run_id": "run-1"}
    ):
        warnings.warn("loss_type=None is unrecognized", UserWarning, stacklevel=1)

    payload = json.loads(event_path.read_text().strip())
    assert payload["operation"] == "suppressed"
    assert payload["data"]["count"] == 1
    assert payload["data"]["run_id"] == "run-1"
    assert warning_mod.os.environ["TRANSFORMERS_VERBOSITY"] == "info"


def test_warning_log_filter_tolerates_bad_log_record(monkeypatch):
    class _BadRecord(logging.LogRecord):
        def getMessage(self):
            raise ValueError("bad message")

    class _CaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = _CaptureHandler()
    logger = logging.getLogger("datasets")
    previous_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(handler)
    try:
        with warning_mod.suppress_noisy_warnings("ci"):
            record = _BadRecord("datasets", logging.WARNING, __file__, 1, "x", (), None)
            logger.handle(record)
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagate

    assert handler.records == [record]


def test_doctor_version_and_optional_report_failures_are_nonfatal(monkeypatch):
    monkeypatch.setattr(
        doctor.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing package")),
    )
    assert doctor._doctor_load_invarlock_version() == "unknown"
    monkeypatch.setattr(
        doctor,
        "load_explicit_report_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")),
    )
    assert doctor._doctor_load_optional_report_payload("report.json") is None
    monkeypatch.setattr(
        doctor,
        "load_explicit_report_input",
        lambda *_args, **_kwargs: (None, {"valid": False}, None, True),
    )
    assert doctor._doctor_load_optional_report_payload("report.json") is None


def test_doctor_inventory_status_preserves_actionable_detail():
    rows = [
        ("needs_extra", None, None, "Needs extra"),
        ("degraded", None, "runtime probe failed", "runtime probe failed"),
        ("custom", None, None, "custom"),
    ]
    for status, required_extra, detail, expected in rows:
        row = SimpleNamespace(
            mode="adapter",
            status=status,
            required_extra=required_extra,
            detail=detail,
        )
        assert doctor._doctor_inventory_status_action(row) == expected


def test_doctor_tiny_relax_env_probe_failure_defaults_to_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        doctor,
        "_doctor_tiny_relax_enabled",
        lambda: (_ for _ in ()).throw(ValueError("bad environment")),
    )
    monkeypatch.setattr(
        doctor,
        "build_tiny_relax_finding",
        lambda **kwargs: calls.append(kwargs) or None,
    )
    accumulator = doctor.DoctorAccumulator()
    doctor._doctor_apply_tiny_relax(
        subject_report=None,
        baseline_report=None,
        json_out=True,
        accumulator=accumulator,
    )
    assert calls[0]["env_enabled"] is False
    assert accumulator.findings == []
