from __future__ import annotations

import importlib
from io import StringIO


def test_cli_main_module_imports() -> None:
    mod = importlib.import_module("invarlock.cli.__main__")
    assert hasattr(mod, "main")


def test_output_helpers_fallback_print_and_no_color(monkeypatch) -> None:
    from invarlock.cli.output import (
        make_console,
        print_event,
        print_timing_summary,
        resolve_output_style,
    )

    class DummyConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, *args) -> None:  # noqa: ANN002
            self.lines.append(" ".join(str(a) for a in args))

    dummy = DummyConsole()
    styled = resolve_output_style(
        style="audit",
        profile="ci",
        progress=False,
        timing=True,
        no_color=False,
    )
    print_event(dummy, "PASS", "hello", style=styled, emoji="✅")
    print_timing_summary(dummy, {"step": 1.0}, style=styled, order=[("Step", "step")])
    assert any("[PASS]" in line for line in dummy.lines)
    assert any("TIMING SUMMARY" in line for line in dummy.lines)

    buf = StringIO()
    monkeypatch.delenv("NO_COLOR", raising=False)
    console_color = make_console(file=buf, force_terminal=True)
    console_color.print("hello", style="red")
    assert "\x1b[" in buf.getvalue()

    buf = StringIO()
    monkeypatch.setenv("NO_COLOR", "1")
    console_plain = make_console(file=buf, force_terminal=True)
    console_plain.print("hello", style="red")
    assert "\x1b[" not in buf.getvalue()


def test_doctor_helpers_get_adapter_rows(monkeypatch) -> None:
    from invarlock.cli import doctor_helpers

    class DummyRegistry:
        def list_adapters(self):  # noqa: ANN001
            return [
                "hf_causal",
                "hf_gptq",
                "hf_awq",
                "hf_bnb",
                "hf_causal_onnx",
                "hf_auto",
                "plugin_adapter",
            ]

        def get_plugin_info(self, name, plugin_type):  # noqa: ANN001,ARG002
            if name == "plugin_adapter":
                return {"module": "my_plugin.adapters"}
            return {"module": f"invarlock.adapters.{name}"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(doctor_helpers._platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor_helpers.importlib.util, "find_spec", lambda _n: None)

    rows = doctor_helpers.get_adapter_rows()
    by_name = {row["name"]: row for row in rows}

    assert by_name["hf_gptq"]["status"] == "unsupported"
    assert by_name["hf_gptq"]["enable"] == "Linux-only"
    assert by_name["hf_awq"]["status"] == "unsupported"
    assert by_name["hf_causal_onnx"]["status"] == "needs_extra"
    assert "invarlock[onnx]" in by_name["hf_causal_onnx"]["enable"]
    assert by_name["hf_auto"]["mode"] == "auto-matcher"
    assert by_name["plugin_adapter"]["origin"] == "plugin"
