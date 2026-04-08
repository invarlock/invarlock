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


def test_baseline_harvest_multimodal_success_populates_defaults() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
    cfg.dataset.split = "validation"

    out = validate_and_harvest_baseline_schedule(
        cfg,
        {
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"id": "ex-1", "prompt": "what?"}],
                "processor_sha256": "proc-123",
            },
            "final": {
                "records": [{"id": "ex-2", "prompt": "where?"}],
            },
        },
        {
            "data": {
                "dataset": "vision_text",
                "split": "validation",
                "provider_kind": "vision_text",
            },
            "provenance": {"provider_digest": "bad-type"},
        },
        tokenizer_hash=None,
        resolved_loss_type="classification",
        console=None,
    )

    assert out["effective_preview"] == 1
    assert out["effective_final"] == 1
    assert out["dataset_meta"]["provider_kind"] == "vision_text"
    assert out["dataset_meta"]["processor_sha256"] == "proc-123"
    assert out["dataset_meta"]["loss_type"] == "classification"
    assert out["window_plan"]["profile"] == "vision_text"
    assert out["calibration_data"] == []


def test_baseline_harvest_multimodal_dev_hash_mismatch_warns() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"

    messages: list[str] = []

    class _Console:
        def print(self, message):  # noqa: ANN001
            messages.append(str(message))

    console = _Console()

    out = validate_and_harvest_baseline_schedule(
        cfg,
        {
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"id": "ex-1"}],
            },
            "final": {
                "example_ids": ["ex-2"],
                "records": [{"id": "ex-2"}],
            },
        },
        {
            "data": {
                "dataset": "vision_text",
                "split": "validation",
                "preview_hash": "wrong-preview",
                "final_hash": "wrong-final",
                "dataset_hash": "wrong-dataset",
            }
        },
        tokenizer_hash=None,
        resolved_loss_type="classification",
        profile="dev",
        console=console,
        event_fn=lambda console, tag, message, **kwargs: console.print(message),
    )

    assert out["dataset_meta"]["loss_type"] == "classification"
    assert any("preview_hash mismatch" in message for message in messages)
    assert any("final_hash mismatch" in message for message in messages)
    assert any("dataset_hash mismatch" in message for message in messages)


def test_baseline_harvest_multimodal_record_id_mismatch_raises() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
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
                    "example_ids": ["ex-1"],
                    "records": [{"id": "wrong-id"}],
                },
                "final": {
                    "example_ids": ["ex-2"],
                    "records": [{"id": "ex-2"}],
                },
            },
            {"data": {"dataset": "vision_text", "split": "validation"}},
            tokenizer_hash=None,
            resolved_loss_type="classification",
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


def test_baseline_harvest_multimodal_duplicate_and_overlap_failures() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
    baseline = {"data": {"dataset": "vision_text", "split": "validation"}}

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"example_ids": ["ex-1", "ex-1"], "records": []},
                "final": {"example_ids": ["ex-2"], "records": []},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"example_ids": ["ex-1"], "records": []},
                "final": {"example_ids": ["ex-1"], "records": []},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )


def test_baseline_harvest_multimodal_empty_example_id_raises() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"records": [{"id": ""}]},
                "final": {"example_ids": ["ex-2"], "records": [{"id": "ex-2"}]},
            },
            {"data": {"dataset": "vision_text", "split": "validation"}},
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )


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


def test_baseline_harvest_multimodal_success() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = "vision_text"
    pairing = {
        "preview": {
            "example_ids": ["ex-1"],
            "records": [{"id": "ex-1", "prompt": "what?"}],
            "processor_sha256": "proc-123",
        },
        "final": {
            "example_ids": ["ex-2"],
            "records": [{"id": "ex-2", "prompt": "where?"}],
        },
    }
    baseline = {
        "data": {
            "dataset": "vision_text",
            "split": "validation",
            "provider_kind": "vision_text",
            "window_plan": {"profile": "vision_text"},
        },
        "provenance": {"provider_digest": {"processor_sha256": "proc-123"}},
    }

    out = validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="classification",
        baseline_path_str="baseline.json",
        console=None,
    )

    assert out["effective_preview"] == 1
    assert out["effective_final"] == 1
    assert out["dataset_meta"]["provider_kind"] == "vision_text"
    assert out["dataset_meta"]["processor_sha256"] == "proc-123"
    assert out["window_plan"] == {"profile": "vision_text"}


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
