from __future__ import annotations

import pytest
import typer

from invarlock.cli.run_pairing import validate_and_harvest_baseline_schedule
from invarlock.core.exceptions import InvarlockError
from tests.cli.run._support_run_baseline_harvest import _Cfg


def test_baseline_harvest_success() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "tokenizer_hash": "tok",
            "window_plan": {"k": 1},
        }
    }
    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=None,
    )
    assert out["effective_preview"] == 1 and out["effective_final"] == 1
    assert out["dataset_meta"]["loss_type"] == "causal"
    assert out["window_plan"] == {"k": 1}


def test_baseline_harvest_mismatch_raises() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 4,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }
    with pytest.raises(typer.Exit) as exc:
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert exc.value.exit_code == 1


def test_baseline_harvest_adjustment_prints(capsys) -> None:
    cfg = _Cfg()
    # Make config expect different counts to trigger adjustment message
    cfg.dataset.preview_n = 2
    cfg.dataset.final_n = 2
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    class _Console:
        def print(self, *args, **kwargs):  # pragma: no cover - exercised by capture
            print(*args)

    validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=_Console(),
        event_fn=lambda console, tag, message, **kwargs: console.print(message),
    )
    captured = capsys.readouterr().out
    assert "Adjusting evaluation window counts" in captured


def test_baseline_harvest_uses_dataset_attr_and_ignores_mask_assignment_failures(
    capsys,
) -> None:
    cfg = _Cfg()
    cfg.dataset.preview_n = 2
    cfg.dataset.final_n = 2
    cfg.dataset.provider = None
    cfg.dataset.dataset = "wikitext2"

    class _NoSetDict(dict):
        def __setitem__(self, key, value):  # noqa: ANN001
            raise TypeError("read only")

    pairing = {
        "preview": _NoSetDict({"input_ids": [[0, 1]], "window_ids": [1]}),
        "final": _NoSetDict({"input_ids": [[2, 3]], "window_ids": [2]}),
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    class _Console:
        def print(self, *args, **kwargs):  # pragma: no cover - capture helper
            print(*args)

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=_Console(),
        event_fn=lambda console, tag, message, **kwargs: console.print(message),
    )
    captured = capsys.readouterr().out
    assert "Adjusting evaluation window counts" in captured
    assert out["dataset_meta"]["loss_type"] == "causal"


def test_baseline_dataset_mismatch_emits_exit():
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1]], "window_ids": [1]},
        "final": {"input_ids": [[2, 3]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "synthetic",
            "split": "validation",
        }
    }

    class CaptureConsole:
        def __init__(self):
            self.messages: list[str] = []

        def print(self, msg):
            self.messages.append(str(msg))

    console = CaptureConsole()
    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=console,
            event_fn=lambda console, tag, message, **kwargs: console.print(message),
        )
    assert any("dataset mismatch" in msg for msg in console.messages)


def test_baseline_split_mismatch_raises() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1]], "window_ids": [1]},
        "final": {"input_ids": [[2, 3]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "train",
        }
    }
    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


def test_baseline_harvest_duplicate_and_overlap_failures() -> None:
    cfg = _Cfg()
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1], [1]], "window_ids": [1, 2]},
                "final": {"input_ids": [[2]], "window_ids": [3]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


def test_baseline_harvest_typed_failure_for_missing_sections() -> None:
    with pytest.raises(InvarlockError, match="missing preview/final"):
        validate_and_harvest_baseline_schedule(
            _Cfg(),
            {},
            {},
            tokenizer_hash=None,
            resolved_loss_type="classification",
            typed_failures=True,
        )


def test_baseline_harvest_text_schedule_failure_edges() -> None:
    cfg = _Cfg()
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {
                    "input_ids": [[1, 2]],
                    "window_ids": [0],
                    "labels": [[1, 2], [3, 4]],
                },
                "final": {"input_ids": [[3, 4]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1], [2]], "window_ids": [0, 0]},
                "final": {"input_ids": [[3]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1]], "window_ids": [0]},
                "final": {"input_ids": [[2], [3]], "window_ids": [1, 1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    class _ExplodingGet(dict):
        def get(self, *_args, **_kwargs):  # type: ignore[override]
            raise TypeError("boom")

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": _ExplodingGet({"input_ids": [[1]], "window_ids": [0]}),
                "final": {"input_ids": [[2]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )


def test_baseline_harvest_accepts_explicit_helper_injection() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "tokenizer_hash": "tok",
            "window_plan": {"k": 1},
        }
    }

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=None,
        canonical_dataset_id_fn=lambda value: str(value),
        tensor_or_list_to_ints_fn=lambda value: [int(v) for v in value],
        hash_sequences_fn=lambda seqs: f"hash:{len(seqs)}",
        invarlock_error_cls=RuntimeError,
    )

    assert out["effective_preview"] == 1
    assert out["effective_final"] == 1
