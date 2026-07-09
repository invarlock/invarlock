from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import assert_single_run_output_artifacts
from tests.cli.run._support_run_plugins import (
    plugin_provenance_cfg as _cfg,
)
from tests.cli.run._support_run_plugins import (
    plugin_provenance_common_ce as _common_ce,
)


@pytest.mark.parametrize("indices_type", ["list", "tuple", "generator"])
def test_provider_indices_various_types(tmp_path: Path, indices_type: str):
    cfg = _cfg(tmp_path, 1, 1)

    def _indices():
        return (i for i in [0])

    class Provider:
        def windows(self, **kwargs):
            if indices_type == "list":
                idx = [0]
            elif indices_type == "tuple":
                idx = (0,)
            else:
                idx = _indices()
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=idx
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=idx
            )
            return prev, fin

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    assert_single_run_output_artifacts(tmp_path)
