from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from invarlock.cli.commands.run import _validate_and_harvest_baseline_schedule


class _Cfg:
    def __init__(self):
        self.dataset = SimpleNamespace(
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )


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
    out = _validate_and_harvest_baseline_schedule(
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
        _validate_and_harvest_baseline_schedule(
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

    _validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=_Console(),
    )
    captured = capsys.readouterr().out
    assert "Adjusting evaluation window counts" in captured


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
        _validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=console,
        )
    assert any("dataset mismatch" in msg for msg in console.messages)


def test_baseline_tokenizer_hash_mismatch():
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
            "split": "validation",
            "tokenizer_hash": "tokA",
        }
    }
    with pytest.raises(typer.Exit) as exc:
        _validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash="tokB",
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert exc.value.exit_code == 1
