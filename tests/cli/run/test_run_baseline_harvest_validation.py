from __future__ import annotations

import pytest
import typer

from invarlock.cli.run_pairing import validate_and_harvest_baseline_schedule
from invarlock.core.exceptions import InvarlockError
from tests.cli.run._support_run_baseline_harvest import _Cfg


def test_baseline_harvest_text_hash_mismatch_warns_in_dev() -> None:
    cfg = _Cfg()
    messages: list[str] = []

    class _Console:
        def print(self, message):  # noqa: ANN001
            messages.append(str(message))

    out = validate_and_harvest_baseline_schedule(
        cfg,
        {
            "preview": {
                "input_ids": [[1, 2]],
                "window_ids": [1],
                "attention_masks": [1, 1],
            },
            "final": {
                "input_ids": [[3, 4]],
                "window_ids": [2],
                "attention_masks": [[1, 1]],
            },
        },
        {
            "data": {
                "seq_len": 8,
                "stride": 8,
                "dataset": "wikitext2",
                "split": "validation",
                "preview_hash": "wrong-preview",
                "final_hash": "wrong-final",
                "dataset_hash": "wrong-dataset",
            }
        },
        tokenizer_hash=None,
        resolved_loss_type="causal",
        profile="dev",
        console=_Console(),
        event_fn=lambda console, tag, message, **kwargs: console.print(message),
    )

    assert out["dataset_meta"]["loss_type"] == "causal"
    assert any("preview_hash mismatch" in message for message in messages)
    assert any("final_hash mismatch" in message for message in messages)
    assert any("dataset_hash mismatch" in message for message in messages)


def test_baseline_harvest_text_invalid_section_and_window_id_failures() -> None:
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
            {"preview": {"input_ids": [[1]]}, "final": {"input_ids": [[2]]}},
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[1]], "window_ids": ["bad"]},
                "final": {"input_ids": [[2]], "window_ids": [2]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )


def test_baseline_tokenizer_hash_mismatch() -> None:
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


def test_baseline_harvest_additional_text_validation_failures() -> None:
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
                    "input_ids": [[0, 1, 2]],
                    "window_ids": [0],
                    "labels": [[1], [2]],
                },
                "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"input_ids": [[0, 1, 2]], "window_ids": [0]},
                "final": {"input_ids": [[3, 4, 5]], "window_ids": [1, 1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )

    class _BrokenSection:
        def get(self, *_args, **_kwargs):
            raise TypeError("section boom")

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": _BrokenSection(),
                "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
        )
