from __future__ import annotations

import pytest
import typer

from invarlock.cli.commands.evaluate import evaluate_command


def test_clean_pruning_selection_flags_are_all_or_none() -> None:
    with pytest.raises(typer.BadParameter, match="Clean pruning selection requires"):
        evaluate_command(
            baseline="baseline",
            subject="subject",
            clean_pruning_selection_config="config.json",
        )


def test_generic_and_pruning_selection_flags_are_mutually_exclusive() -> None:
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        evaluate_command(
            baseline="baseline",
            subject="subject",
            clean_selection_config="generic-config.json",
            clean_pruning_selection_config="pruning-config.json",
        )
