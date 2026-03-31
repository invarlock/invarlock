from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli import output as output_mod
from invarlock.cli import run_execution as run_execution_mod
from invarlock.cli import run_execution_output as run_execution_output_mod
from invarlock.core.run_orchestrator import (
    RunCleanupStatusEvent,
    RunDiagnosticEvent,
    RunEvaluationReportFailedEvent,
    RunExecutionFailure,
    RunFailureEvent,
    RunLoadModelOnceEvent,
    RunRetryAttemptStartedEvent,
    RunRetryValidationErrorEvent,
    RunTelemetryFailedEvent,
)
from tests.cli.run._internal_cli import internal_run_app as cli
from tests.cli.support import RecordingConsole


def _cfg(tmp_path: Path, *, provider: str = "synthetic") -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: auto
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: __PROVIDER__
  id: __PROVIDER__
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  metric: { kind: ppl_causal }
  loss: { type: auto }

output:
  dir: runs
""".replace("__PROVIDER__", provider)
    )
    return str(p)


def _common_stubs(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "1")

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(
            execute=lambda **k: SimpleNamespace(
                edit={"deltas": {"params_changed": 0}},
                metrics={"window_overlap_fraction": 0.0, "window_match_fraction": 1.0},
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            ),
            estimate_capacity=lambda **kw: {
                "available_unique": 100,
                "available_nonoverlap": 100,
                "total_tokens": 1000,
                "dedupe_rate": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
            "tokhash123",
        ),
    )


def test_run_ci_uses_semantic_prefixes_no_emojis(tmp_path: Path, monkeypatch) -> None:
    _common_stubs(monkeypatch)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")

    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(
        cli,
        [
            "run",
            "-c",
            cfg,
            "--profile",
            "ci",
            "--style",
            "audit",
            "--allow-host-execution",
        ],
    )
    assert r.exit_code == 0
    s = r.stdout
    assert "[INIT]" in s or "[EXEC]" in s or "[DATA]" in s
    for emoji in [
        "🚀",
        "📋",
        "🔧",
        "🛡️",
        "📜",
        "✅",
        "❌",
        "⚠️",
        "📊",
        "📚",
        "💾",
        "🧹",
        "✂️",
        "🧪",
        "🏁",
    ]:
        assert emoji not in s


def test_run_audit_uses_structured_dataset_events_without_emojis(
    tmp_path: Path, monkeypatch
) -> None:
    _common_stubs(monkeypatch)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")

    def _provider_factory(kind: str, **kwargs):
        _ = kwargs
        assert kind == "wikitext2"

        def _windows(**kw):
            _ = kw
            return (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )

        return SimpleNamespace(windows=_windows)

    monkeypatch.setattr("invarlock.eval.data.get_provider", _provider_factory)

    cfg = _cfg(tmp_path, provider="wikitext2")
    r = CliRunner().invoke(
        cli,
        [
            "run",
            "-c",
            cfg,
            "--profile",
            "ci",
            "--style",
            "audit",
            "--allow-host-execution",
        ],
    )
    assert r.exit_code == 0
    s = r.stdout
    assert "[DATA] Loading dataset: wikitext2" in s
    assert "📚" not in s
    assert "📊" not in s


def test_output_style_resolution_paths(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    audit = output_mod.OutputStyle(name="audit")
    friendly = output_mod.OutputStyle(name="friendly")
    assert audit.audit is True
    assert audit.emojis is False
    assert friendly.audit is False
    assert friendly.emojis is True

    assert output_mod.normalize_style(None) is None
    assert output_mod.normalize_style("   ") is None
    assert output_mod.normalize_style("friendly") == "friendly"
    assert output_mod.normalize_style("unknown") is None

    assert output_mod.resolve_style_name(None, "release") == "audit"
    assert output_mod.resolve_style_name(None, "dev") == "friendly"
    assert output_mod.resolve_style_name(" friendly ", "release") == "friendly"

    monkeypatch.setenv("NO_COLOR", "1")
    resolved = output_mod.resolve_output_style(
        style=None,
        profile="dev",
        progress=True,
        timing=True,
        no_color=False,
    )
    assert resolved.name == "friendly"
    assert resolved.progress is True
    assert resolved.timing is True
    assert resolved.color is False

    explicit = output_mod.resolve_output_style(
        style="audit",
        profile="dev",
        progress=False,
        timing=False,
        no_color=True,
    )
    assert explicit.name == "audit"
    assert explicit.color is False


def test_output_printing_and_style_assignment() -> None:
    fallback_console = RecordingConsole(fail_with_kwargs=True)
    output_mod._safe_console_print(
        fallback_console, "hello", style="green", markup=False
    )
    assert fallback_console.calls == [(("hello",), {})]

    recorder = RecordingConsole()
    audit = output_mod.OutputStyle(name="audit", color=True)
    friendly = output_mod.OutputStyle(name="friendly", color=True)
    no_color = output_mod.OutputStyle(name="friendly", color=False)

    output_mod.print_event(recorder, "pass", "ok", style=audit)
    output_mod.print_event(recorder, "fail", "bad", style=audit)
    output_mod.print_event(recorder, "warn", "careful", style=audit)
    output_mod.print_event(recorder, "metric", "ratio=1.2", style=audit)
    output_mod.print_event(
        recorder,
        "",
        "uses default tag",
        style=friendly,
        emoji="✅",
        console_style="magenta",
    )
    output_mod.print_event(recorder, "info", "still plain", style=audit)
    output_mod.print_event(recorder, "info", "plain", style=no_color, emoji="🎯")

    lines = [call[0][0] for call in recorder.calls]
    kwargs = [call[1] for call in recorder.calls]
    assert lines[0] == "[PASS] ok"
    assert kwargs[0]["style"] == "green"
    assert kwargs[1]["style"] == "red"
    assert kwargs[2]["style"] == "yellow"
    assert kwargs[3]["style"] == "cyan"
    assert lines[4] == "✅ uses default tag"
    assert kwargs[4]["style"] == "magenta"
    assert lines[5] == "[INFO] still plain"
    assert kwargs[5]["style"] is None
    assert lines[6] == "🎯 plain"
    assert kwargs[6]["style"] is None

    assert output_mod.format_event_line("", "message", style=audit) == "[INFO] message"
    assert (
        output_mod.format_event_line("warn", "notice  ", style=friendly, emoji="⚠️")
        == "⚠️ notice"
    )


def test_run_execution_output_style_resets_console_color_state(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    fake_console = SimpleNamespace(no_color=False, _invarlock_output_style=None)
    monkeypatch.setattr(run_execution_mod, "console", fake_console, raising=True)

    no_color_request = SimpleNamespace(
        style=None,
        profile="ci",
        progress=False,
        timing=False,
        no_color=True,
    )
    color_request = SimpleNamespace(
        style=None,
        profile="dev",
        progress=False,
        timing=False,
        no_color=False,
    )

    no_color_style = run_execution_mod._resolve_shell_output_style(no_color_request)
    assert no_color_style.color is False
    assert fake_console.no_color is True

    color_style = run_execution_mod._resolve_shell_output_style(color_request)
    assert color_style.color is True
    assert fake_console.no_color is False


def test_output_timing_helpers_cover_progress_and_summary(monkeypatch) -> None:
    console = RecordingConsole()
    style = output_mod.OutputStyle(name="audit", progress=True, timing=True, color=True)
    perf_values = iter([10.0, 12.25, 20.0, 20.5])
    monkeypatch.setattr(output_mod, "perf_counter", lambda: next(perf_values))

    timings: dict[str, float] = {}
    with output_mod.timed_step(
        console=console,
        style=style,
        timings=timings,
        key="phase",
        tag="pass",
        message="phase",
        emoji="✅",
    ):
        pass
    assert timings["phase"] == 2.25
    assert console.calls[-1][0][0] == "[PASS] phase done (2.25s)"

    no_progress = output_mod.OutputStyle(
        name="friendly", progress=False, timing=False, color=True
    )
    with output_mod.timed_step(
        console=console,
        style=no_progress,
        timings=None,
        key="silent",
        tag="info",
        message="skip",
    ):
        pass
    assert "silent" not in timings

    before_summary = len(console.calls)
    output_mod.print_timing_summary(
        console,
        {"phase": 2.25},
        style=output_mod.OutputStyle(name="audit", timing=False),
        order=[("Phase", "phase")],
    )
    assert len(console.calls) == before_summary

    output_mod.print_timing_summary(
        console,
        {"phase": 2.25},
        style=style,
        order=[("Phase", "phase"), ("Missing", "missing")],
        extra_lines=["  extra"],
    )
    rendered = [call[0][0] for call in console.calls[before_summary:]]
    assert rendered[0] == ""
    assert rendered[1] == "TIMING SUMMARY"
    assert rendered[2].startswith("  Phase")
    assert rendered[-1] == "  extra"


def test_run_execution_output_helpers_cover_fallback_and_progress_paths(
    monkeypatch,
) -> None:
    fallback_console = RecordingConsole(fail_with_kwargs=True)
    run_execution_output_mod.emit_console_line(
        fallback_console,
        "plain line",
        markup=False,
    )
    run_execution_output_mod.emit_console_line(
        fallback_console,
        "[bold]markup[/bold]",
        markup=True,
    )
    run_execution_output_mod.emit_console_blank_line(fallback_console)
    assert fallback_console.lines == ["plain line", "[bold]markup[/bold]", ""]

    idle_console = RecordingConsole()
    run_execution_output_mod.begin_progress_step(idle_console, "noop")
    run_execution_output_mod.complete_progress_step(
        idle_console,
        "noop",
        tag="PASS",
        message="noop",
    )
    assert idle_console.calls == []

    progress_console = RecordingConsole()
    progress_console._invarlock_output_style = SimpleNamespace(progress=True)
    perf_values = iter([10.0, 12.25, 20.0])
    monkeypatch.setattr(output_mod, "perf_counter", lambda: next(perf_values))

    run_execution_output_mod.begin_progress_step(progress_console, "load")
    run_execution_output_mod.complete_progress_step(
        progress_console,
        "load",
        tag="PASS",
        message="Loading model",
        emoji="✅",
    )
    run_execution_output_mod.transition_progress_step(
        progress_console,
        "missing",
        from_tag="INIT",
        from_message="Warmup",
        to_key="execute",
        from_emoji="🔧",
    )
    assert "load" in progress_console._invarlock_progress_completed
    assert progress_console._invarlock_progress_steps["execute"] == 20.0
    assert "Loading model done (2.25s)" in progress_console.lines[0]


def test_run_execution_event_rendering_covers_split_owner_branches(monkeypatch) -> None:
    console = RecordingConsole()
    console._invarlock_output_style = output_mod.OutputStyle(
        name="audit",
        progress=True,
        timing=True,
        color=True,
    )
    monkeypatch.setattr(output_mod, "perf_counter", lambda: 50.0)

    run_execution_output_mod.render_run_execution_event(
        console,
        RunDiagnosticEvent(code="export_tokenizer_missing"),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunDiagnosticEvent(code="export_failed"),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunDiagnosticEvent(
            code="snapshot_restore_fallback",
            context={"error": "boom"},
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunDiagnosticEvent(
            code="retry_validation_telemetry_summary",
            context={"summary": "telemetry summary"},
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunDiagnosticEvent(
            summary="plain diagnostic",
            level="warn",
            context={"emoji": "⚠️"},
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunLoadModelOnceEvent(model_id="gpt2"),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunRetryAttemptStartedEvent(attempt=2, max_attempts=4),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunTelemetryFailedEvent(error="disk full"),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunEvaluationReportFailedEvent(gate_codes=("pm", "spectral")),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunRetryValidationErrorEvent(summary="schema mismatch"),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunCleanupStatusEvent(removed=False),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(code="edit_name_missing", context={})
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(
                code="unknown_edit",
                context={"edit_name": "bad-edit"},
            )
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(
                code="baseline_windows_missing",
                summary="baseline windows unavailable",
            )
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(
                code="config_file_missing",
                context={"path": "/tmp/missing.yaml"},
            )
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(code="schema_invalid_run_report", context={})
        ),
    )
    run_execution_output_mod.render_run_execution_event(
        console,
        RunFailureEvent(
            failure=RunExecutionFailure(
                code="pipeline_failed",
                summary="core pipeline failure",
            )
        ),
    )

    output = console.joined()
    assert "tokenizer artifacts" in output
    assert "unexpected error" in output
    assert "switching to reload-per-attempt" in output
    assert "↳ boom" in output
    assert "telemetry summary" in output
    assert "plain diagnostic" in output
    assert "Loading model once: gpt2" in output
    assert "Retry attempt 2/4" in output
    assert "Telemetry export failed: disk full" in output
    assert "FAILED gates: pm, spectral" in output
    assert "schema mismatch" in output
    assert "Cleanup: skipped" in output
    assert "Edit configuration must specify a non-empty `edit.name`." in output
    assert "Unknown edit 'bad-edit'." in output
    assert "baseline windows unavailable" in output
    assert "Configuration file not found: /tmp/missing.yaml" in output
    assert "Schema invalid: run report structure failed validation" in output
    assert "Pipeline execution failed: core pipeline failure" in output
