from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from invarlock.cli.run_pairing import validate_and_harvest_baseline_schedule
from invarlock.core.exceptions import InvarlockError


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

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1]], "window_ids": [1]},
                "final": {"input_ids": [[2], [2]], "window_ids": [2, 3]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1]], "window_ids": [1]},
                "final": {"input_ids": [[1]], "window_ids": [2]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


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
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash="tokB",
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert exc.value.exit_code == 1


def test_baseline_harvest_preview_hash_mismatch_warns_in_dev_profile() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "preview_hash": "deadbeef",
        }
    }

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        profile="dev",
        baseline_path_str="baseline.json",
        console=None,
    )
    assert out["effective_preview"] == 1


def test_baseline_harvest_preview_hash_mismatch_fails_in_ci() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "preview_hash": "deadbeef",
        }
    }

    with pytest.raises(InvarlockError) as ei:
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            profile="ci",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert str(ei.value).startswith("[INVARLOCK:E001]")


def test_baseline_harvest_final_hash_mismatch_warns_in_dev_profile() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "final_hash": "deadbeef",
        }
    }

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        profile="dev",
        baseline_path_str="baseline.json",
        console=None,
    )
    assert out["effective_final"] == 1


def test_baseline_harvest_final_hash_mismatch_fails_in_ci() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "final_hash": "deadbeef",
        }
    }

    with pytest.raises(InvarlockError) as ei:
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            profile="ci",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert str(ei.value).startswith("[INVARLOCK:E001]")


def test_baseline_harvest_dataset_hash_mismatch_warns_in_dev_profile() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "dataset_hash": "deadbeef",
        }
    }

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        profile="dev",
        baseline_path_str="baseline.json",
        console=None,
    )
    assert out["effective_preview"] == 1


def test_baseline_harvest_dataset_hash_mismatch_fails_in_ci() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
            "dataset_hash": "deadbeef",
        }
    }

    with pytest.raises(InvarlockError) as ei:
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            profile="ci",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert str(ei.value).startswith("[INVARLOCK:E001]")


def test_baseline_harvest_missing_final_section_fails() -> None:
    cfg = _Cfg()
    pairing = {"preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]}}
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
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


def test_baseline_harvest_empty_input_ids_fails() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[]], "window_ids": [0]},
        "final": {"input_ids": [[1, 2, 3]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
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


def test_baseline_harvest_window_id_length_mismatch_fails() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0, 1]},
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


def test_baseline_harvest_attention_mask_row_count_mismatch_fails() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {
            "input_ids": [[0, 1, 2], [3, 4, 5]],
            "window_ids": [0, 1],
            "attention_masks": [[1, 1, 1]],
        },
        "final": {"input_ids": [[6, 7, 8]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
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


def test_baseline_harvest_attention_mask_row_not_list_fails() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {
            "input_ids": [[0, 1, 2], [3, 4, 5]],
            "window_ids": [0, 1],
            "attention_masks": [1, [1, 1, 1]],
        },
        "final": {"input_ids": [[6, 7, 8]], "window_ids": [2]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
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


def test_baseline_harvest_missing_preview_section_raises_in_ci() -> None:
    cfg = _Cfg()
    pairing = {"final": {"input_ids": [[3, 4, 5]], "window_ids": [2]}}
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }

    with pytest.raises(InvarlockError) as exc:
        validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            profile="ci",
            baseline_path_str="baseline.json",
            console=None,
        )
    assert exc.value.code == "E001"


def test_baseline_harvest_labels_length_mismatch_fails() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {
            "input_ids": [[0, 1, 2]],
            "window_ids": [0],
            "labels": [[-100, -100]],
        },
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


def test_baseline_harvest_sequence_and_stride_mismatch_fail() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
        "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 16,
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


def test_baseline_harvest_duplicate_ids_and_overlap_fail() -> None:
    cfg = _Cfg()
    pairing = {
        "preview": {"input_ids": [[0, 1, 2], [0, 1, 2]], "window_ids": [0, 1]},
        "final": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
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
