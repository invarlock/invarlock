# ruff: noqa: I001,E402,F811
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from invarlock.cli import run_execution as masking_mod
from invarlock.cli.run_execution import persist_ref_masks
from invarlock.cli.run_pairing import (
    _compute_mask_positions_digest,
    _hash_sequences,
    compute_provider_digest,
    enforce_provider_parity,
    extract_pairing_schedule,
)
from invarlock.core.exceptions import resolve_command_exit_code
from invarlock.core.exceptions import (
    ConfigError,
    DataError,
    InvarlockError,
    ValidationError,
)
from invarlock.core.run_policy import choose_dataset_split, should_measure_overhead


def test_should_measure_overhead_respects_config_and_profile() -> None:
    assert should_measure_overhead("ci", {}) == (True, False, None)
    assert should_measure_overhead("release", {}) == (True, False, None)
    assert should_measure_overhead("dev", {}) == (False, False, None)

    assert should_measure_overhead(
        "ci", {"context": {"run": {"skip_overhead_check": True}}}
    ) == (False, True, "config:context.run.skip_overhead_check")


def test_persist_ref_masks_writes_artifact_when_present(tmp_path: Path) -> None:
    core_report = {
        "edit": {"artifacts": {"mask_payload": {"keep_indices": [1, 2, 3]}}},
    }
    out = persist_ref_masks(core_report, tmp_path)
    assert isinstance(out, Path)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["keep_indices"] == [1, 2, 3]
    assert isinstance(payload.get("meta", {}).get("generated_at"), str)


def test_persist_ref_masks_returns_none_when_missing_payload(tmp_path: Path) -> None:
    assert persist_ref_masks({}, tmp_path) is None
    assert persist_ref_masks({"edit": {}}, tmp_path) is None
    assert persist_ref_masks({"edit": {"artifacts": {}}}, tmp_path) is None


def test_persist_ref_masks_from_dict_and_object_preserves_generated_at(
    tmp_path: Path,
) -> None:
    payload = {"keep": [1, 2], "meta": {"generated_at": "existing-ts"}}
    core_report = {"edit": {"artifacts": {"mask_payload": payload}}}

    mask_path = persist_ref_masks(core_report, tmp_path)
    assert mask_path == tmp_path / "artifacts" / "edit_masks" / "masks.json"
    written = json.loads(mask_path.read_text(encoding="utf-8"))
    assert written == payload
    assert mask_path.read_text(encoding="utf-8").endswith("\n")

    obj = SimpleNamespace(edit={"artifacts": {"mask_payload": {"keep": [3]}}})
    object_mask_path = persist_ref_masks(obj, tmp_path)
    object_written = json.loads(object_mask_path.read_text(encoding="utf-8"))
    assert object_written["keep"] == [3]
    assert "generated_at" in object_written["meta"]


@pytest.mark.parametrize(
    "core_report",
    [
        {},
        {"edit": []},
        {"edit": {}},
        {"edit": {"artifacts": []}},
        {"edit": {"artifacts": {}}},
        {"edit": {"artifacts": {"mask_payload": {}}}},
        {"edit": {"artifacts": {"mask_payload": []}}},
    ],
)
def test_persist_ref_masks_rejects_missing_sections(
    tmp_path: Path, core_report: object
) -> None:
    assert persist_ref_masks(core_report, tmp_path) is None


def test_choose_dataset_split_behaviors() -> None:
    split, used_fallback = choose_dataset_split(
        requested="test", available=["train", "test"]
    )
    assert split == "test"
    assert used_fallback is False

    split, used_fallback = choose_dataset_split(
        requested=None, available=["val", "train"]
    )
    assert split in {"validation", "val"}
    assert used_fallback is True

    split, used_fallback = choose_dataset_split(requested=None, available=None)
    assert split == "validation"
    assert used_fallback is True


def test_hash_sequences_stability() -> None:
    digest = _hash_sequences([[1, 2, 3], [4, 5]])
    assert isinstance(digest, str)
    assert len(digest) == 32


def test_hash_sequences_respects_boundaries() -> None:
    assert _hash_sequences([[1, 2], [3]]) != _hash_sequences([[1], [2, 3]])


def test_compute_mask_positions_digest_roundtrip() -> None:
    windows = {
        "preview": {"labels": [np.array([-100, 2, -100], dtype=np.int32)]},
        "final": {"labels": [np.array([-100, -100, 7], dtype=np.int32)]},
    }
    digest = _compute_mask_positions_digest(windows)
    assert isinstance(digest, str)
    assert len(digest) > 0
    assert _compute_mask_positions_digest({"preview": {"labels": []}}) is None


def test_enforce_provider_parity_missing_tokenizer_digest_in_ci_raises() -> None:
    with pytest.raises(InvarlockError) as excinfo:
        enforce_provider_parity(
            {"ids_sha256": "abc"}, {"ids_sha256": "def"}, profile="ci"
        )
    assert excinfo.value.code == "E004"


def test_enforce_provider_parity_uses_explicit_error_class() -> None:
    class _CustomParityError(Exception):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args)
            self.kwargs = kwargs

    with pytest.raises(_CustomParityError):
        enforce_provider_parity(
            {"ids_sha256": "abc"},
            {"ids_sha256": "def"},
            profile="ci",
            invarlock_error_cls=_CustomParityError,
        )


def test_resolve_exit_code_covers_known_exceptions() -> None:
    assert (
        resolve_command_exit_code(ConfigError(code="E0", message="x"), profile="ci")
        == 2
    )
    assert (
        resolve_command_exit_code(ValidationError(code="E0", message="x"), profile=None)
        == 2
    )
    assert (
        resolve_command_exit_code(DataError(code="E0", message="x"), profile="release")
        == 2
    )
    assert (
        resolve_command_exit_code(
            ValueError("Invalid RunReport structure"), profile=None
        )
        == 2
    )
    assert (
        resolve_command_exit_code(
            InvarlockError(code="E0", message="x"), profile="release"
        )
        == 3
    )
    assert (
        resolve_command_exit_code(InvarlockError(code="E0", message="x"), profile="dev")
        == 1
    )


def test_resolve_exit_code_invarlockerror_profiles() -> None:
    err = InvarlockError(code="E005", message="boom")
    assert resolve_command_exit_code(err, profile="ci") == 3
    assert resolve_command_exit_code(err, profile="release") == 3
    assert resolve_command_exit_code(err, profile="dev") == 1


def test_resolve_exit_code_schema_validation_types() -> None:
    for err in (
        ConfigError(code="E201", message="cfg"),
        ValidationError(code="E202", message="val"),
        DataError(code="E203", message="data"),
    ):
        assert resolve_command_exit_code(err, profile="dev") == 2
        assert resolve_command_exit_code(err, profile="ci") == 2
        assert resolve_command_exit_code(err, profile="release") == 2


def test_resolve_exit_code_invalid_runreport_value_error_special_case() -> None:
    err = ValueError("Invalid RunReport: shape mismatch")
    assert resolve_command_exit_code(err, profile="dev") == 2


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
    sched = extract_pairing_schedule(report)
    assert isinstance(sched, dict)
    assert sched["preview"]["attention_masks"] == [[1, 1, 1]]
    assert sched["preview"]["labels"] == [[5, 6, -100]]
    assert sched["preview"]["masked_token_counts"] == [2]
    assert sched["preview"]["actual_token_counts"] == [3]
    assert sched["final"]["labels"] == [[7, 8]]


def test_extract_pairing_schedule_returns_none_on_invalid_shapes() -> None:
    assert extract_pairing_schedule(None) is None
    assert extract_pairing_schedule({"evaluation_windows": "nope"}) is None
    assert (
        extract_pairing_schedule(
            {"evaluation_windows": {"preview": {"input_ids": "bad"}, "final": {}}}
        )
        is None
    )


def test_extract_pairing_schedule_accepts_explicit_tensor_helper() -> None:
    calls: list[object] = []

    def _helper(raw):  # noqa: ANN001
        calls.append(raw)
        return [int(value) for value in raw]

    report = {
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2]], "window_ids": [1]},
            "final": {"input_ids": [[3, 4]], "window_ids": [2]},
        }
    }

    sched = extract_pairing_schedule(report, tensor_or_list_to_ints_fn=_helper)

    assert sched is not None
    assert calls == [[1, 2], [3, 4]]


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
    assert extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_reraises_unexpected_window_id_errors() -> None:
    class _BadWindowId:
        def __int__(self) -> int:
            raise RuntimeError("boom")

    report = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [_BadWindowId()],
                "input_ids": [[1, 2, 3]],
                "attention_masks": [[1, 1, 1]],
            },
            "final": {"input_ids": [[1]]},
        }
    }

    with pytest.raises(RuntimeError, match="boom"):
        extract_pairing_schedule(report)


def test_extract_pairing_schedule_single_row_fallbacks() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 0, 2]],
                "attention_masks": [1, 0, 1],
                "labels": [7, 8],
                "masked_token_counts": 4,
                "actual_token_counts": 5,
            },
            "final": {
                "input_ids": [[3, 4]],
                "attention_masks": [1, 1],
                "labels": [9],
                "masked_token_counts": 2,
                "actual_token_counts": 2,
            },
        }
    }
    sched = extract_pairing_schedule(report)
    assert isinstance(sched, dict)
    assert sched["preview"]["window_ids"] == [0]
    assert sched["preview"]["attention_masks"] == [[1, 0, 1]]
    assert sched["preview"]["labels"] == [[7, 8, -100]]
    assert sched["preview"]["masked_token_counts"] == [4]
    assert sched["preview"]["actual_token_counts"] == [5]
    assert sched["final"]["window_ids"] == [1]
    assert sched["final"]["attention_masks"] == [[1, 1]]
    assert sched["final"]["labels"] == [[9, -100]]


def test_extract_pairing_schedule_ignores_non_list_attention_masks_and_prefers_window_processor_sha() -> (
    None
):
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 0, 2]],
                "window_ids": [1],
                "attention_masks": object(),
                "processor_sha256": "proc-window",
            },
            "final": {
                "input_ids": [[3, 4]],
                "window_ids": [2],
                "attention_masks": [[1, 1]],
            },
        },
        "meta": {"processor_sha256": "proc-meta", "tokenizer_hash": "tok-123"},
    }

    sched = extract_pairing_schedule(report)
    assert isinstance(sched, dict)
    assert sched["preview"]["attention_masks"] == [[1, 0, 1]]

    digest = compute_provider_digest(
        report,
        compute_mask_positions_digest_fn=lambda _windows: None,
    )
    assert digest is not None
    assert digest["processor_sha256"] == "proc-window"


def test_extract_pairing_schedule_rejects_malformed_sections() -> None:
    assert (
        extract_pairing_schedule(
            {"evaluation_windows": {"preview": "bad", "final": {}}}
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {"evaluation_windows": {"preview": {"input_ids": []}, "final": {}}}
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {"input_ids": [[1]], "window_ids": [1, 2]},
                    "final": {"input_ids": [[2]]},
                }
            }
        )
        is None
    )


def test_extract_pairing_schedule_supports_multimodal_records() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"id": "ex-1", "prompt": "what?"}],
                "processor_sha256": "proc-123",
            },
            "final": {
                "records": [{"id": "ex-2", "prompt": "where?"}],
            },
        }
    }

    sched = extract_pairing_schedule(report)

    assert sched == {
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
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1], [2]],
                        "window_ids": [1, 2],
                        "actual_token_counts": [1],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )


def test_extract_pairing_schedule_prefers_multimodal_input_records() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"id": "ex-1", "prediction": "red", "references": ["red"]}],
                "input_records": [
                    {
                        "id": "ex-1",
                        "image_path": "/tmp/red.ppm",
                        "prompt": "what color?",
                        "answers": ["red"],
                    }
                ],
            },
            "final": {
                "example_ids": ["ex-2"],
                "records": [
                    {"id": "ex-2", "prediction": "green", "references": ["green"]}
                ],
                "input_records": [
                    {
                        "id": "ex-2",
                        "image_path": "/tmp/green.ppm",
                        "prompt": "what color?",
                        "answers": ["green"],
                    }
                ],
            },
        }
    }

    sched = extract_pairing_schedule(report)

    assert sched == {
        "preview": {
            "example_ids": ["ex-1"],
            "records": [
                {
                    "id": "ex-1",
                    "image_path": "/tmp/red.ppm",
                    "prompt": "what color?",
                    "answers": ["red"],
                }
            ],
        },
        "final": {
            "example_ids": ["ex-2"],
            "records": [
                {
                    "id": "ex-2",
                    "image_path": "/tmp/green.ppm",
                    "prompt": "what color?",
                    "answers": ["green"],
                }
            ],
        },
    }


def test_compute_provider_digest_uses_meta_processor_fallback() -> None:
    digest = compute_provider_digest(
        {
            "evaluation_windows": {
                "preview": {"window_ids": [1]},
                "final": {"window_ids": [2]},
            },
            "meta": {"processor_sha256": "proc-123", "tokenizer_hash": "tok-123"},
        },
        compute_mask_positions_digest_fn=lambda _windows: None,
    )

    assert digest is not None
    assert digest["tokenizer_sha256"] == "tok-123"
    assert digest["processor_sha256"] == "proc-123"


def test_extract_pairing_schedule_falls_back_for_malformed_attention_rows() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2, 0], [3, 4, 0]],
                "window_ids": [1, 2],
                "attention_masks": [1, 0, 1],
            },
            "final": {"input_ids": [[5, 6]], "window_ids": [3]},
        }
    }
    sched = extract_pairing_schedule(report)
    assert isinstance(sched, dict)
    assert sched["preview"]["attention_masks"] == [[1, 1, 0], [1, 1, 0]]
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {"input_ids": [[1]], "window_ids": ["bad"]},
                    "final": {"input_ids": [[2]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1]],
                        "window_ids": [1],
                        "actual_token_counts": [True],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1]],
                        "window_ids": [1],
                        "masked_token_counts": ["bad-count"],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1]],
                        "window_ids": [1],
                        "actual_token_counts": [-1],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1]],
                        "window_ids": [1],
                        "actual_token_counts": True,
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1]],
                        "window_ids": [1],
                        "attention_masks": [[1, 1]],
                    },
                    "final": {"input_ids": [[2]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1], [2]],
                        "window_ids": [1, 2],
                        "labels": [[1], [2], [3]],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {
                        "input_ids": [[1], [2]],
                        "window_ids": [1, 2],
                        "masked_token_counts": [1],
                    },
                    "final": {"input_ids": [[3]]},
                }
            }
        )
        is None
    )
    assert (
        extract_pairing_schedule(
            {
                "evaluation_windows": {
                    "preview": {"input_ids": [[1]], "window_ids": [1]},
                    "final": {
                        "input_ids": [[2]],
                        "window_ids": [2],
                        "attention_masks": [1, 1],
                    },
                }
            }
        )
        is None
    )


def test_extract_pairing_schedule_rejects_multimodal_length_mismatch() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "example_ids": ["ex-1", "ex-2"],
                "records": [{"id": "ex-1"}],
            },
            "final": {
                "example_ids": ["ex-3"],
                "records": [{"id": "ex-3"}],
            },
        }
    }

    assert extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_skips_non_dict_multimodal_records() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "example_ids": ["ex-1"],
                "records": ["skip-me"],
            },
            "final": {
                "records": [{"id": "ex-2"}],
            },
        }
    }

    assert extract_pairing_schedule(report) == {
        "preview": {"example_ids": ["ex-1"]},
        "final": {
            "example_ids": ["ex-2"],
            "records": [{"id": "ex-2"}],
        },
    }


def test_extract_pairing_schedule_rejects_attention_mask_row_count_mismatch() -> None:
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "attention_masks": [[1, 1]],
            },
            "final": {"input_ids": [[5, 6]], "attention_masks": [[1, 1]]},
        }
    }

    assert extract_pairing_schedule(report) is None


def test_apply_mlm_masks_handles_mask_random_and_original_modes(monkeypatch) -> None:
    # Force masking decision for each position.
    monkeypatch.setattr(masking_mod.random, "random", lambda: 0.0)

    r_values = iter([0.0, 0.85, 0.95])

    class _FakeRandom:
        def __init__(self, _seed):  # noqa: ANN001
            pass

        def random(self) -> float:
            return float(next(r_values))

        def randint(self, a: int, b: int) -> int:  # noqa: ARG002
            return 7

    monkeypatch.setattr(masking_mod.random, "Random", _FakeRandom)

    class _Tok:
        vocab_size = 100
        mask_token_id = 999

    records = [{"window_id": "w0", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    total, counts = masking_mod._apply_mlm_masks(
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
    total, counts = masking_mod._apply_mlm_masks(
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
