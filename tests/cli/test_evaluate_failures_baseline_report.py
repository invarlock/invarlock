from __future__ import annotations

import os
from pathlib import Path

import click
import pytest

from tests.cli._support_evaluate_failures import (
    _assert_baseline_report_validation_exit,
    _prepare_evaluate_paths,
    _valid_baseline_report_payload,
    _write_json,
    mod,
    run_mod,
)


def test_evaluate_rejects_baseline_report_directory_even_with_report_json(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = tmp_path / "baseline-run"
    baseline_dir.mkdir()
    _write_json(
        baseline_dir / "report.json",
        _valid_baseline_report_payload(),
    )
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: run_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 2
    assert run_calls == []


def test_evaluate_rejects_baseline_report_directory_without_report_json(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = tmp_path / "baseline-run"
    baseline_dir.mkdir()
    _write_json(
        baseline_dir / "20250101_000000.json",
        _valid_baseline_report_payload(),
    )
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: run_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 2
    assert run_calls == []


def test_evaluate_baseline_report_invalid_json_exits(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        raw_text="{not-json",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_json_object(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=["not-a-dict"],
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_noop_edit(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(edit_name="quant_rtn"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_adapter_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(adapter="hf_awq"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_profile_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(profile="release"),
        profile="dev",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_tier_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(tier="strict"),
        tier="balanced",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_evaluation_windows_payload(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload={
            **_valid_baseline_report_payload(),
            "evaluation_windows": None,
        },
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_non_regular_file(
    monkeypatch, tmp_path: Path
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo unavailable on this platform")

    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_fifo = tmp_path / "baseline.pipe"
    os.mkfifo(baseline_fifo)

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            baseline_report=str(baseline_fifo),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 2


def test_evaluate_baseline_report_requires_preview_payload(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": None,
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_matching_window_lengths(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
                "final": {
                    "window_ids": ["final-0", "final-1"],
                    "input_ids": [[4, 5, 6]],
                },
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_supplied_baseline_report_path_must_exist(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            baseline_report="missing.json",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 2


def test_evaluate_supplied_baseline_report_directory_requires_a_report_file(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = Path("baseline-dir")
    baseline_dir.mkdir()
    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 2


def test_evaluate_baseline_report_rejects_non_mapping_meta_and_context(
    monkeypatch, tmp_path: Path
) -> None:
    baseline_payload = {
        **_valid_baseline_report_payload(),
        "meta": "bad-meta",
        "context": "bad-context",
    }
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=baseline_payload,
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_model_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(model_id="wrong-model"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_assurance_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(assurance_mode="strict"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_dataset_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _valid_baseline_report_payload()
    data = payload["data"]
    assert isinstance(data, dict)
    data = dict(data)
    data["seed"] = 999
    payload["data"] = data
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=payload,
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_nonempty_preview_window_ids(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": [], "input_ids": [[1, 2, 3]]},
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_nonempty_preview_input_ids(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": []},
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2
