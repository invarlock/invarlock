from __future__ import annotations

import pytest
import typer

from invarlock.cli import run_pairing_baseline as run_pairing_baseline_mod
from invarlock.cli.run_pairing import validate_and_harvest_baseline_schedule
from tests.cli.run._support_run_baseline_harvest import _Cfg


def test_baseline_harvest_internal_hash_and_multimodal_failure_edges() -> None:
    assert run_pairing_baseline_mod._BaselineScheduleValidator._hash_tokens([]) == b""

    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
    baseline = {"data": {"dataset": "vision_text", "split": "validation"}}

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"records": [1]},
                "final": {"example_ids": ["ex-2"], "records": [{"id": "ex-2"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"records": []},
                "final": {"example_ids": ["ex-2"], "records": [{"id": "ex-2"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {
                    "example_ids": ["ex-1", "ex-2"],
                    "records": [{"id": "ex-1"}],
                },
                "final": {"example_ids": ["ex-3"], "records": [{"id": "ex-3"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"example_ids": ["ex-1"], "records": [{"id": "ex-1"}]},
                "final": {
                    "example_ids": ["ex-2", "ex-2"],
                    "records": [{"id": "ex-2"}, {"id": "ex-2"}],
                },
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
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


def test_baseline_harvest_multimodal_accepts_example_ids_without_records() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
    cfg.dataset.split = "validation"

    out = validate_and_harvest_baseline_schedule(
        cfg,
        {
            "preview": {
                "example_ids": ["ex-1"],
                "processor_sha256": "proc-123",
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
                "provider_kind": "vision_text",
            }
        },
        tokenizer_hash=None,
        resolved_loss_type="classification",
        console=None,
    )

    assert out["effective_preview"] == 1
    assert out["effective_final"] == 1
    assert out["dataset_meta"]["provider_kind"] == "vision_text"
    assert out["dataset_meta"]["processor_sha256"] == "proc-123"


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


def test_baseline_harvest_additional_multimodal_validation_failures() -> None:
    cfg = _Cfg()
    cfg.dataset.provider = None
    cfg.dataset.dataset = "vision_text"
    baseline = {"data": {"dataset": "vision_text", "split": "validation"}}

    assert run_pairing_baseline_mod._BaselineScheduleValidator._hash_tokens([]) == b""

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"records": ["bad-record"]},
                "final": {"example_ids": ["ex-2"], "records": [{"id": "ex-2"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"records": []},
                "final": {"example_ids": ["ex-2"], "records": [{"id": "ex-2"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {
                    "example_ids": ["ex-1", "ex-2"],
                    "records": [{"id": "ex-1"}],
                },
                "final": {"example_ids": ["ex-3"], "records": [{"id": "ex-3"}]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )

    with pytest.raises(typer.Exit):
        validate_and_harvest_baseline_schedule(
            cfg,
            {
                "preview": {"example_ids": ["ex-1"], "records": [{"id": "ex-1"}]},
                "final": {"example_ids": ["ex-2", "ex-2"], "records": []},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="classification",
        )
