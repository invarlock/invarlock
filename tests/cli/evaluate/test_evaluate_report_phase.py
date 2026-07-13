from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import invarlock.cli.evaluate_report_phase as phase_mod
from invarlock.cli.evaluate_report_phase import (
    EvaluationReportRequest,
    EvaluationReportRuntime,
    emit_evaluation_report_phase,
)
from invarlock.cli.output import resolve_output_style
from invarlock.core.exceptions import ConfigError, MetricsError, ValidationError


def _base_request(tmp_path: Path, **overrides: Any) -> EvaluationReportRequest:
    report_root = tmp_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    resolved_config = tmp_path / "resolved-subject.yaml"
    resolved_config.write_text("model: {}\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "edited_report": tmp_path / "edited.report.json",
        "resolved_subject_config": resolved_config,
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
    }
    payload.update(overrides)
    return EvaluationReportRequest(**payload)


def _base_runtime(
    *,
    timings: dict[str, float] | None = None,
    fail_fn: Any | None = None,
    generate_reports_fn: Any | None = None,
    emit_runtime_manifest_fn: Any | None = None,
    manifest_execution_fn: Any | None = None,
    finalize_clean_selection_report_fn: Any | None = None,
    finalize_clean_pruning_selection_report_fn: Any | None = None,
) -> EvaluationReportRuntime:
    return EvaluationReportRuntime(
        console=Console(file=None),
        output_style=resolve_output_style(
            style="audit",
            profile="dev",
            progress=False,
            timing=False,
            no_color=True,
        ),
        timings=timings if timings is not None else {},
        info_fn=lambda *_args, **_kwargs: None,
        fail_fn=fail_fn or (lambda *_args, **_kwargs: None),
        generate_reports_fn=generate_reports_fn or (lambda **_kwargs: None),
        emit_runtime_manifest_fn=emit_runtime_manifest_fn
        or (lambda *_args, **_kwargs: None),
        manifest_execution_fn=manifest_execution_fn or (lambda **_kwargs: None),
        finalize_clean_selection_report_fn=finalize_clean_selection_report_fn,
        finalize_clean_pruning_selection_report_fn=(
            finalize_clean_pruning_selection_report_fn
        ),
    )


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

    timings: dict[str, float] = {}
    emit_evaluation_report_phase(
        _base_request(tmp_path),
        _base_runtime(
            timings=timings,
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=emit_runtime_manifest,
        ),
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
    assert "evaluation_report" in timings
    assert manifest_calls[0]["path"] == tmp_path / "reports" / "evaluation.report.json"
    assert manifest_calls[0]["config_path"] == (
        tmp_path / "reports" / "resolved-config.yaml"
    )


def test_evaluate_report_phase_disables_optional_rendering_when_deferred(
    tmp_path: Path,
) -> None:
    report_calls: list[dict[str, Any]] = []

    def generate_reports(**kwargs):  # noqa: ANN001
        report_calls.append(dict(kwargs))

    emit_evaluation_report_phase(
        _base_request(tmp_path, defer_report_rendering=True),
        _base_runtime(generate_reports_fn=generate_reports),
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
    timings: dict[str, float] = {}
    request = _base_request(
        tmp_path,
        allow_network=True,
        allow_remote_code=True,
        allow_third_party_plugins=True,
        execution_mode="container",
        assurance_mode="strict",
        preset="configs/evaluate/current-supported.yaml",
        edit_config="edits/noop.yaml",
        edit_label="noop-edit",
    )
    emit_evaluation_report_phase(
        request,
        _base_runtime(
            timings=timings,
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=emit_runtime_manifest,
            manifest_execution_fn=manifest_execution,
        ),
    )

    assert events == ["timed-enter", "generate", "timed-exit", "manifest"]
    assert timings["evaluation_report"] == 1.5
    assert manifest_calls == [
        {
            "path": tmp_path / "reports" / "evaluation.report.json",
            "config_path": tmp_path / "reports" / "resolved-config.yaml",
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


def test_evaluate_report_phase_finalizes_clean_selection_before_manifest(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manifest_calls: list[dict[str, Any]] = []
    context = object()
    link = {"candidate_id": "candidate", "repeat_index": 0}

    def generate_reports(**_kwargs: Any) -> None:
        events.append("generate")

    def finalize(path: Path, *, context: object) -> dict[str, object]:
        events.append("finalize")
        assert path == tmp_path / "reports" / "evaluation.report.json"
        assert context is not None
        return link

    def emit_runtime_manifest(path: Path, **kwargs: Any) -> None:
        events.append("manifest")
        manifest_calls.append({"path": path, **kwargs})

    emit_evaluation_report_phase(
        _base_request(tmp_path, clean_selection_context=context),
        _base_runtime(
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=emit_runtime_manifest,
            finalize_clean_selection_report_fn=finalize,
        ),
    )

    assert events == ["generate", "finalize", "manifest"]
    assert manifest_calls[0]["extra"]["clean_selection_execution"] == link


def test_evaluate_report_phase_fails_closed_when_clean_selection_cannot_finalize(
    tmp_path: Path,
) -> None:
    failures: list[tuple[str, int]] = []
    manifest_calls: list[object] = []

    def finalizer(_path: Path, *, context: object) -> dict[str, object]:
        assert context is not None
        raise ValueError("missing evaluator provenance")

    def fail_fn(message: str, *, exit_code: int) -> None:
        failures.append((message, exit_code))

    emit_evaluation_report_phase(
        _base_request(tmp_path, clean_selection_context=object()),
        _base_runtime(
            fail_fn=fail_fn,
            finalize_clean_selection_report_fn=finalizer,
            emit_runtime_manifest_fn=lambda *args, **kwargs: manifest_calls.append(
                (args, kwargs)
            ),
        ),
    )

    assert failures == [
        (
            "Could not finalize receipt-bound clean-selection evidence: "
            "missing evaluator provenance",
            1,
        )
    ]
    assert manifest_calls == []


def test_evaluate_report_phase_finalizes_clean_pruning_selection_before_manifest(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manifest_calls: list[dict[str, Any]] = []
    link = {"candidate_id": "prune-25", "repeat_index": 0}

    def generate_reports(**_kwargs: Any) -> None:
        events.append("generate")

    def finalize(path: Path, *, context: object) -> dict[str, object]:
        events.append("finalize")
        assert path == tmp_path / "reports" / "evaluation.report.json"
        assert context is not None
        return link

    def emit_runtime_manifest(path: Path, **kwargs: Any) -> None:
        events.append("manifest")
        manifest_calls.append({"path": path, **kwargs})

    emit_evaluation_report_phase(
        _base_request(tmp_path, clean_pruning_selection_context=object()),
        _base_runtime(
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=emit_runtime_manifest,
            finalize_clean_pruning_selection_report_fn=finalize,
        ),
    )

    assert events == ["generate", "finalize", "manifest"]
    assert manifest_calls[0]["extra"]["clean_pruning_selection_execution"] == link


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
        _base_request(tmp_path),
        _base_runtime(
            fail_fn=fail_fn,
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=lambda *args, **kwargs: manifest_calls.append(
                (args, kwargs)
            ),
        ),
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
            _base_request(tmp_path),
            _base_runtime(
                generate_reports_fn=generate_reports,
                emit_runtime_manifest_fn=lambda *args, **kwargs: manifest_calls.append(
                    (args, kwargs)
                ),
            ),
        )

    assert manifest_calls == []


def test_evaluate_report_phase_rejects_competing_selection_receipts(
    tmp_path: Path,
) -> None:
    failures: list[tuple[str, int]] = []

    emit_evaluation_report_phase(
        _base_request(
            tmp_path,
            clean_selection_context=object(),
            clean_pruning_selection_context=object(),
        ),
        _base_runtime(
            fail_fn=lambda message, *, exit_code: failures.append((message, exit_code))
        ),
    )

    assert failures == [
        (
            "Generic clean selection and clean pruning selection cannot both "
            "finalize one report.",
            1,
        )
    ]


@pytest.mark.parametrize(
    ("request_field", "runtime_field", "expected_message"),
    [
        (
            "clean_selection_context",
            "finalize_clean_selection_report_fn",
            "Clean-selection evaluator finalizer is unavailable.",
        ),
        (
            "clean_pruning_selection_context",
            "finalize_clean_pruning_selection_report_fn",
            "Clean-pruning-selection evaluator finalizer is unavailable.",
        ),
    ],
)
def test_evaluate_report_phase_requires_selection_finalizers(
    tmp_path: Path,
    request_field: str,
    runtime_field: str,
    expected_message: str,
) -> None:
    failures: list[tuple[str, int]] = []
    request = _base_request(tmp_path, **{request_field: object()})
    runtime = _base_runtime(
        fail_fn=lambda message, *, exit_code: failures.append((message, exit_code)),
        **{runtime_field: None},
    )

    emit_evaluation_report_phase(request, runtime)

    assert failures == [(expected_message, 1)]


@pytest.mark.parametrize(
    ("request_field", "runtime_field", "expected_message"),
    [
        (
            "clean_selection_context",
            "finalize_clean_selection_report_fn",
            "Clean-selection evaluator finalizer did not return a manifest link.",
        ),
        (
            "clean_pruning_selection_context",
            "finalize_clean_pruning_selection_report_fn",
            "Clean-pruning-selection evaluator finalizer did not return a manifest link.",
        ),
    ],
)
def test_evaluate_report_phase_rejects_non_mapping_selection_links(
    tmp_path: Path,
    request_field: str,
    runtime_field: str,
    expected_message: str,
) -> None:
    failures: list[tuple[str, int]] = []
    request = _base_request(tmp_path, **{request_field: object()})
    runtime = _base_runtime(
        fail_fn=lambda message, *, exit_code: failures.append((message, exit_code)),
        **{runtime_field: lambda *_args, **_kwargs: None},
    )

    emit_evaluation_report_phase(request, runtime)

    assert failures == [(expected_message, 1)]


def test_evaluate_report_phase_fails_closed_when_pruning_selection_raises(
    tmp_path: Path,
) -> None:
    failures: list[tuple[str, int]] = []

    def fail_finalizer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("receipt digest mismatch")

    emit_evaluation_report_phase(
        _base_request(tmp_path, clean_pruning_selection_context=object()),
        _base_runtime(
            fail_fn=lambda message, *, exit_code: failures.append((message, exit_code)),
            finalize_clean_pruning_selection_report_fn=fail_finalizer,
        ),
    )

    assert failures == [
        (
            "Could not finalize receipt-bound clean-pruning-selection evidence: "
            "receipt digest mismatch",
            1,
        )
    ]
