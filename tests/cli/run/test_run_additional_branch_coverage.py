# ruff: noqa: I001,E402,F811
from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands import run as run_mod
from invarlock.cli.errors import InvarlockError
from invarlock.core.exceptions import ConfigError, DataError, ValidationError


def test_should_measure_overhead_respects_env_and_profile(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_SKIP_OVERHEAD_CHECK", raising=False)
    assert run_mod._should_measure_overhead("ci") == (True, False)
    assert run_mod._should_measure_overhead("release") == (True, False)
    assert run_mod._should_measure_overhead("dev") == (False, False)

    monkeypatch.setenv("INVARLOCK_SKIP_OVERHEAD_CHECK", "1")
    assert run_mod._should_measure_overhead("ci") == (False, True)


def test_persist_ref_masks_writes_artifact_when_present(tmp_path: Path) -> None:
    core_report = {
        "edit": {"artifacts": {"mask_payload": {"keep_indices": [1, 2, 3]}}},
    }
    out = run_mod._persist_ref_masks(core_report, tmp_path)
    assert isinstance(out, Path)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["keep_indices"] == [1, 2, 3]
    assert isinstance(payload.get("meta", {}).get("generated_at"), str)


def test_persist_ref_masks_returns_none_when_missing_payload(tmp_path: Path) -> None:
    assert run_mod._persist_ref_masks({}, tmp_path) is None
    assert run_mod._persist_ref_masks({"edit": {}}, tmp_path) is None
    assert run_mod._persist_ref_masks({"edit": {"artifacts": {}}}, tmp_path) is None


def test_resolve_exit_code_covers_known_exceptions() -> None:
    assert (
        run_mod._resolve_exit_code(ConfigError(code="E0", message="x"), profile="ci")
        == 2
    )
    assert (
        run_mod._resolve_exit_code(
            ValidationError(code="E0", message="x"), profile=None
        )
        == 2
    )
    assert (
        run_mod._resolve_exit_code(DataError(code="E0", message="x"), profile="release")
        == 2
    )
    assert (
        run_mod._resolve_exit_code(
            ValueError("Invalid RunReport structure"), profile=None
        )
        == 2
    )
    assert (
        run_mod._resolve_exit_code(
            InvarlockError(code="E0", message="x"), profile="release"
        )
        == 3
    )
    assert (
        run_mod._resolve_exit_code(
            InvarlockError(code="E0", message="x"), profile="dev"
        )
        == 1
    )


def test_extract_pairing_schedule_sanitizes_attention_and_labels() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "input_ids": [[1, 2, 3]],
                # No attention_masks → fallback generated from input_ids
                "labels": [[5, 6]],  # shorter than input_ids → padded
                "masked_token_counts": [2],
                "actual_token_counts": [3],
            },
            "final": {
                "window_ids": [2],
                "input_ids": [[9, 0]],
                "attention_masks": [[1, 0]],
                "labels": [[7, 8, 9]],  # longer than input_ids → truncated
            },
        }
    }
    sched = run_mod._extract_pairing_schedule(report)
    assert isinstance(sched, dict)
    assert sched["preview"]["attention_masks"] == [[1, 1, 1]]
    assert sched["preview"]["labels"] == [[5, 6, -100]]
    assert sched["preview"]["masked_token_counts"] == [2]
    assert sched["preview"]["actual_token_counts"] == [3]
    assert sched["final"]["labels"] == [[7, 8]]


def test_extract_pairing_schedule_returns_none_on_invalid_shapes() -> None:
    assert run_mod._extract_pairing_schedule(None) is None
    assert run_mod._extract_pairing_schedule({"evaluation_windows": "nope"}) is None
    assert (
        run_mod._extract_pairing_schedule(
            {"evaluation_windows": {"preview": {"input_ids": "bad"}, "final": {}}}
        )
        is None
    )


def test_extract_pairing_schedule_rejects_non_int_window_ids() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "window_ids": ["bad"],
                "input_ids": [[1, 2, 3]],
                "attention_masks": [[1, 1, 1]],
            },
            "final": {"input_ids": [[1]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_apply_mlm_masks_covers_mask_random_and_original_branches(monkeypatch) -> None:
    # Force masking decision for each position.
    monkeypatch.setattr(run_mod.random, "random", lambda: 0.0)

    r_values = iter([0.0, 0.85, 0.95])

    class _FakeRandom:
        def __init__(self, _seed):  # noqa: ANN001
            pass

        def random(self) -> float:
            return float(next(r_values))

        def randint(self, a: int, b: int) -> int:  # noqa: ARG002
            return 7

    monkeypatch.setattr(run_mod.random, "Random", _FakeRandom)

    class _Tok:
        vocab_size = 100
        mask_token_id = 999

    records = [{"window_id": "w0", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=_Tok(),
        mask_prob=1.0,
        seed=0,
        random_token_prob=0.1,
        original_token_prob=0.1,
        prefix="p",
    )
    assert total == 3
    assert counts == [3]
    assert records[0]["labels"] == [1, 2, 3]
    assert records[0]["input_ids"][0] == 999  # mask token branch
    assert records[0]["input_ids"][1] == 7  # random token branch
    assert records[0]["input_ids"][2] == 3  # original token branch


def test_apply_mlm_masks_candidate_positions_empty_leaves_all_unmasked() -> None:
    class _Tok:
        vocab_size = 10
        mask_token_id = 999

    records = [{"input_ids": [1, 2], "attention_mask": [0, 0]}]
    total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=_Tok(),
        mask_prob=1.0,
        seed=0,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="p",
    )
    assert total == 0
    assert counts == [0]
    assert records[0]["labels"] == [-100, -100]
    assert records[0]["mlm_masked"] == 0
