from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import click
import pytest

import invarlock.cli.commands.run as run_mod
import invarlock.cli.run_execution as run_exec_mod
from invarlock.cli.commands import evaluate as mod
from tests.cli.support import RecordingConsole


def _stub_run_dir(out_dir: Path, name: str = "report.json") -> Path:
    ts = out_dir / "20250101_000000"
    ts.mkdir(parents=True, exist_ok=True)
    report_path = ts / name
    report_path.write_text(
        json.dumps({"meta": {}, "metrics": {}, "data": {}}), encoding="utf-8"
    )
    return report_path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_run_command_with_paths(
    path_by_out_dir: dict[str, Path | None],
    *,
    run_calls: list[dict[str, object]] | None = None,
    validator: Callable[[dict[str, object], str], None] | None = None,
) -> Callable[..., str | None]:
    def _fake_run(**kwargs):
        if run_calls is not None:
            run_calls.append(kwargs)
        out_name = Path(kwargs["out"]).name
        if validator is not None:
            validator(kwargs, out_name)
        if out_name not in path_by_out_dir:
            raise AssertionError(f"Unexpected run output dir: {kwargs['out']}")
        report_path = path_by_out_dir[out_name]
        return str(report_path) if report_path is not None else None

    return _fake_run


def _valid_baseline_report_payload(
    *,
    adapter: str = "hf_causal",
    profile: str = "dev",
    tier: str = "balanced",
    edit_name: str = "noop",
    evaluation_windows: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "edit": {"name": edit_name},
        "meta": {"adapter": adapter},
        "context": {"profile": profile, "auto": {"tier": tier}},
        "evaluation_windows": evaluation_windows
        or {
            "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
        },
    }


def _prepare_evaluate_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    return src, edt


def _assert_baseline_report_validation_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payload: object | None = None,
    raw_text: str | None = None,
    profile: str = "dev",
    tier: str = "balanced",
) -> click.exceptions.Exit:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_path = Path("baseline.json")
    if raw_text is not None:
        baseline_path.write_text(raw_text, encoding="utf-8")
    else:
        assert payload is not None
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_path),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile=profile,
            tier=tier,
        )

    return exc.value


__all__ = [
    "RecordingConsole",
    "_assert_baseline_report_validation_exit",
    "_fake_run_command_with_paths",
    "_prepare_evaluate_paths",
    "_stub_run_dir",
    "_valid_baseline_report_payload",
    "_write_json",
    "mod",
    "run_exec_mod",
    "run_mod",
]
