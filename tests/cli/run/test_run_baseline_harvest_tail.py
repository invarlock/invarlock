from __future__ import annotations

import pytest
import typer

from invarlock.cli.run_pairing import validate_and_harvest_baseline_schedule
from invarlock.core.exceptions import InvarlockError
from tests.cli.run._support_run_baseline_harvest import _Cfg


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


def test_baseline_harvest_actual_count_and_mask_mismatch_failures() -> None:
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
                    "actual_token_counts": [1, 2],
                },
                "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
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
                "preview": {
                    "input_ids": [[0, 1, 2]],
                    "window_ids": [0],
                    "attention_masks": [[1, 1]],
                },
                "final": {"input_ids": [[3, 4, 5]], "window_ids": [1]},
            },
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )
