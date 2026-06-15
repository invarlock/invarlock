from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import invarlock.cli.evaluate_report_phase as phase_mod
from invarlock.cli.evaluate_report_phase import emit_evaluation_report_phase
from invarlock.cli.output import resolve_output_style
from invarlock.core.exceptions import ConfigError, MetricsError, ValidationError


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "edited_report": tmp_path / "edited.report.json",
        "baseline_report_path": tmp_path / "baseline.report.json",
        "report_out": tmp_path / "reports",
        "baseline": "gpt2",
        "subject": "distilgpt2",
        "baseline_eff_adapter": "hf_causal",
        "subject_eff_adapter": "hf_causal",
        "profile_name": "dev",
        "tier_name": "balanced",
        "preset": None,
        "out": str(tmp_path / "runs"),
        "edit_config": None,
        "edit_label": "noop",
        "allow_network": False,
        "allow_remote_code": False,
        "allow_third_party_plugins": False,
        "execution_mode": "host",
        "assurance_mode": "off",
        "defer_report_rendering": False,
        "console": Console(file=None),
        "output_style": resolve_output_style(
            style="audit",
            profile="dev",
            progress=False,
            timing=False,
            no_color=True,
        ),
        "timings": {},
        "info_fn": lambda *_args, **_kwargs: None,
    }


def test_evaluate_report_phase_calls_report_contract_with_render_optional(
    tmp_path: Path,
) -> None:
    report_calls: list[dict[str, Any]] = []
    manifest_calls: list[dict[str, Any]] = []

    def generate_reports(  # noqa: A002, ANN001
        *, run, format, baseline, output, render_optional
    ):
        report_calls.append(
            {
                "run": run,
                "format": format,
                "baseline": baseline,
                "output": output,
                "render_optional": render_optional,
            }
        )

    def emit_runtime_manifest(path, **kwargs):  # noqa: ANN001
        manifest_calls.append({"path": path, **kwargs})

    kwargs = _base_kwargs(tmp_path)
    emit_evaluation_report_phase(
        **kwargs,
        fail_fn=lambda *_args, **_kwargs: None,
        generate_reports_fn=generate_reports,
        emit_runtime_manifest_fn=emit_runtime_manifest,
        manifest_execution_fn=lambda **_kwargs: None,
    )

    assert report_calls == [
        {
            "run": str(tmp_path / "edited.report.json"),
            "format": "report",
            "baseline": str(tmp_path / "baseline.report.json"),
            "output": str(tmp_path / "reports"),
            "render_optional": True,
        }
    ]
    assert "evaluation_report" in kwargs["timings"]
    assert manifest_calls[0]["path"] == tmp_path / "reports" / "evaluation.report.json"
    assert manifest_calls[0]["config_payload"]["baseline"] == "gpt2"


def test_evaluate_report_phase_disables_optional_rendering_when_deferred(
    tmp_path: Path,
) -> None:
    report_calls: list[dict[str, Any]] = []

    def generate_reports(**kwargs):  # noqa: ANN001
        report_calls.append(dict(kwargs))

    kwargs = _base_kwargs(tmp_path)
    kwargs["defer_report_rendering"] = True
    emit_evaluation_report_phase(
        **kwargs,
        fail_fn=lambda *_args, **_kwargs: None,
        generate_reports_fn=generate_reports,
        emit_runtime_manifest_fn=lambda *_args, **_kwargs: None,
        manifest_execution_fn=lambda **_kwargs: None,
    )

    assert report_calls[0]["render_optional"] is False


def test_evaluate_report_phase_emits_manifest_after_timed_report_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    manifest_calls: list[dict[str, Any]] = []
    execution = object()

    @contextmanager
    def timed_step(*, timings, key, **_kwargs) -> Iterator[None]:
        events.append("timed-enter")
        try:
            yield
        finally:
            timings[key] = 1.5
            events.append("timed-exit")

    def generate_reports(**_kwargs):  # noqa: ANN001
        events.append("generate")

    def emit_runtime_manifest(path, **kwargs):  # noqa: ANN001
        events.append("manifest")
        manifest_calls.append({"path": path, **kwargs})

    def manifest_execution(**kwargs):  # noqa: ANN001
        assert kwargs == {
            "execution_mode": "container",
            "allow_network": True,
            "allow_remote_code": True,
            "allow_third_party_plugins": True,
        }
        return execution

    monkeypatch.setattr(phase_mod.cli_output, "timed_step", timed_step, raising=True)
    kwargs = _base_kwargs(tmp_path)
    kwargs.update(
        {
            "allow_network": True,
            "allow_remote_code": True,
            "allow_third_party_plugins": True,
            "execution_mode": "container",
            "assurance_mode": "strict",
            "preset": "configs/evaluate/current-supported.yaml",
            "edit_config": "edits/noop.yaml",
            "edit_label": "noop-edit",
        }
    )
    emit_evaluation_report_phase(
        **kwargs,
        fail_fn=lambda *_args, **_kwargs: None,
        generate_reports_fn=generate_reports,
        emit_runtime_manifest_fn=emit_runtime_manifest,
        manifest_execution_fn=manifest_execution,
    )

    assert events == ["timed-enter", "generate", "timed-exit", "manifest"]
    assert kwargs["timings"]["evaluation_report"] == 1.5
    assert manifest_calls == [
        {
            "path": tmp_path / "reports" / "evaluation.report.json",
            "config_payload": {
                "command": "evaluate",
                "baseline": "gpt2",
                "subject": "distilgpt2",
                "baseline_adapter": "hf_causal",
                "subject_adapter": "hf_causal",
                "profile": "dev",
                "tier": "balanced",
                "preset": "configs/evaluate/current-supported.yaml",
                "out": str(tmp_path / "runs"),
                "report_out": tmp_path / "reports",
                "edit_config": "edits/noop.yaml",
                "edit_label": "noop-edit",
                "allow_network": True,
                "allow_remote_code": True,
                "allow_third_party_plugins": True,
                "execution_mode": "container",
                "assurance": "strict",
                "defer_report_rendering": False,
            },
            "extra": {
                "command": "evaluate",
                "profile": "dev",
                "tier": "balanced",
                "execution_mode": "container",
                "assurance": "strict",
            },
            "execution": execution,
        }
    ]


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (ConfigError("bad_config", "bad report config"), "bad report config"),
        (MetricsError("bad_metric", "bad report metric"), "bad report metric"),
        (
            ValidationError("bad_validation", "bad report validation"),
            "bad report validation",
        ),
    ],
)
def test_evaluate_report_phase_routes_report_errors_to_fail_handler(
    tmp_path: Path, exc: Exception, message: str
) -> None:
    failures: list[tuple[str, int]] = []
    manifest_calls: list[object] = []

    def generate_reports(**_kwargs):  # noqa: ANN001
        raise exc

    def fail_fn(message: str, *, exit_code: int) -> None:
        failures.append((message, exit_code))

    emit_evaluation_report_phase(
        **_base_kwargs(tmp_path),
        fail_fn=fail_fn,
        generate_reports_fn=generate_reports,
        emit_runtime_manifest_fn=lambda *args, **kwargs: manifest_calls.append(
            (args, kwargs)
        ),
        manifest_execution_fn=lambda **_kwargs: None,
    )

    assert failures == [(message, 1)]
    assert manifest_calls == []


def test_evaluate_report_phase_bubbles_unexpected_report_errors(
    tmp_path: Path,
) -> None:
    manifest_calls: list[object] = []

    def generate_reports(**_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        emit_evaluation_report_phase(
            **_base_kwargs(tmp_path),
            fail_fn=lambda *_args, **_kwargs: None,
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=lambda *args, **kwargs: manifest_calls.append(
                (args, kwargs)
            ),
            manifest_execution_fn=lambda **_kwargs: None,
        )

    assert manifest_calls == []
