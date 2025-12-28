from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from invarlock.cli.commands.run import _enforce_provider_parity, run_command
from invarlock.cli.errors import InvarlockError


def test_enforce_provider_parity_mask_mismatch_raises_invarlock_error():
    subj = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-A"}
    base = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "mask-B"}
    with pytest.raises(InvarlockError) as ei:
        _enforce_provider_parity(subj, base, profile="ci")
    s = str(ei.value)
    assert s.startswith("[INVARLOCK:E003]")
    assert "MASK-PARITY-MISMATCH" in s


def test_enforce_provider_parity_tokenizer_mismatch_raises_invarlock_error():
    subj = {"ids_sha256": "ids", "tokenizer_sha256": "abc"}
    base = {"ids_sha256": "ids", "tokenizer_sha256": "def"}
    with pytest.raises(InvarlockError) as ei:
        _enforce_provider_parity(subj, base, profile="release")
    s = str(ei.value)
    assert s.startswith("[INVARLOCK:E002]")
    assert "TOKENIZER-DIGEST-MISMATCH" in s


def test_enforce_provider_parity_ids_mismatch_raises_invarlock_error():
    subj = {"ids_sha256": "ids-a", "tokenizer_sha256": "tok"}
    base = {"ids_sha256": "ids-b", "tokenizer_sha256": "tok"}
    with pytest.raises(InvarlockError) as ei:
        _enforce_provider_parity(subj, base, profile="ci")
    s = str(ei.value)
    assert s.startswith("[INVARLOCK:E006]")
    assert "IDS-DIGEST-MISMATCH" in s


def test_enforce_provider_parity_missing_digest_raises_invarlock_error():
    subj = {"masking_sha256": "m"}
    base = {"tokenizer_sha256": "abc"}
    with pytest.raises(InvarlockError) as ei:
        _enforce_provider_parity(subj, base, profile="ci")
    s = str(ei.value)
    assert s.startswith("[INVARLOCK:E004]")
    assert "PROVIDER-DIGEST-MISSING" in s


def test_run_command_emits_e003_prefix_and_exit_code(tmp_path: Path):
    # Baseline with fixed tokenizer and different masks to trigger E003
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5]]},
                },
            }
        )
    )

    # Minimal config YAML
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  id: gpt2
  adapter: hf_gpt2
dataset:
  provider: dummy
  seq_len: 8
  stride: 8
  preview_n: 1
  final_n: 1
edit:
  name: noop
guards:
  order: []
context:
  profile: release
        """
    )

    # Patch provider to return windows with same tokenizer but different masks
    from unittest.mock import patch

    with ExitStack() as stack:
        # get_provider returns windows with attention masks
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(input_ids=[[4, 5]], attention_masks=[[1, 1]]),
                    )
                ),
            )
        )
        # Force tokenizer hash to match baseline so only mask parity differs
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        # Make CoreRunner.execute a no-op object with essentials
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

        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                out=str(tmp_path / "runs"),
                baseline=str(baseline),
                until_pass=False,
            )
        assert ei.value.exit_code == 3
