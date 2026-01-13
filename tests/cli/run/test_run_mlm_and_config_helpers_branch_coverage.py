# ruff: noqa: I001,E402,F811
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from invarlock.cli.commands import run as run_mod


def test_coerce_mapping_covers_multiple_sources_and_failures() -> None:
    assert run_mod._coerce_mapping({"a": 1}) == {"a": 1}

    obj_with_data = SimpleNamespace(_data={"b": 2})
    assert run_mod._coerce_mapping(obj_with_data) == {"b": 2}

    class _DataAttrRaises:
        def __getattribute__(self, name: str):  # noqa: ANN001
            if name == "_data":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

        def model_dump(self):  # noqa: ANN001
            return {"c": 3}

    assert run_mod._coerce_mapping(_DataAttrRaises()) == {"c": 3}

    class _ModelDumpRaises:
        def model_dump(self):  # noqa: ANN001
            raise RuntimeError("boom")

    inst = _ModelDumpRaises()
    inst.x = 1  # type: ignore[attr-defined]
    assert run_mod._coerce_mapping(inst) == {"x": 1}

    class _Slots:
        __slots__ = ()

    assert run_mod._coerce_mapping(_Slots()) == {}

    class _C:
        x = 1

    assert run_mod._coerce_mapping(_C) == {}


def test_resolve_pm_acceptance_range_parses_cfg_and_env(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    assert run_mod._resolve_pm_acceptance_range(None) == {}
    assert run_mod._resolve_pm_acceptance_range({}) == {}

    cfg = {"primary_metric": {"acceptance_range": {"min": "bad", "max": "1.2"}}}
    out = run_mod._resolve_pm_acceptance_range(cfg)
    assert out == {"min": 0.95, "max": 1.2}

    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "0")
    out2 = run_mod._resolve_pm_acceptance_range(cfg)
    assert out2 == {"min": 0.95, "max": 1.1}

    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "1.2")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "1.1")
    out3 = run_mod._resolve_pm_acceptance_range(cfg)
    assert out3 == {"min": 1.2, "max": 1.2}

    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "bad")
    out4 = run_mod._resolve_pm_acceptance_range(cfg)
    assert out4["max"] == 1.2


def test_resolve_pm_acceptance_range_ignores_invalid_cfg_max(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    cfg = {"primary_metric": {"acceptance_range": {"min": "1.0", "max": "bad"}}}
    out = run_mod._resolve_pm_acceptance_range(cfg)
    assert out == {"min": 1.0, "max": 1.1}


def test_resolve_pm_acceptance_range_covers_outer_exception(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    def _boom(_cfg):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(run_mod, "_coerce_mapping", _boom)
    assert (
        run_mod._resolve_pm_acceptance_range(
            {"primary_metric": {"acceptance_range": {"min": 0.9, "max": 1.1}}}
        )
        == {}
    )


def test_choose_dataset_split_covers_fallback_and_exception_branch() -> None:
    split, used = run_mod._choose_dataset_split(
        requested="train", available=["validation"]
    )
    assert split == "train"
    assert used is False

    split, used = run_mod._choose_dataset_split(
        requested=None, available=["val", "train"]
    )
    assert split == "val"
    assert used is True

    split, used = run_mod._choose_dataset_split(
        requested=None, available=["zzz", "aaa"]
    )
    assert split == "aaa"
    assert used is True

    split, used = run_mod._choose_dataset_split(requested=None, available=None)
    assert split == "validation"
    assert used is True

    class _BadStr(str):
        def __len__(self) -> int:
            raise RuntimeError("boom")

    split, used = run_mod._choose_dataset_split(
        requested=_BadStr("x"), available=["validation"]
    )
    assert split == "validation"
    assert used is True


def test_compute_mask_positions_digest_covers_none_digest_and_exception() -> None:
    assert (
        run_mod._compute_mask_positions_digest(
            {
                "preview": {"labels": [[-100, -100]]},
                "final": {"labels": [[-100]]},
            }
        )
        is None
    )

    digest = run_mod._compute_mask_positions_digest(
        {
            "preview": {"labels": [[-100, 5]]},
            "final": {"labels": [[-100]]},
        }
    )
    assert isinstance(digest, str) and len(digest) == 32

    class _BadDict(dict):
        def get(self, *_a, **_k):  # noqa: ANN001
            raise RuntimeError("boom")

    assert run_mod._compute_mask_positions_digest(_BadDict()) is None


def test_tensor_or_list_to_ints_covers_tolist_numpy_iterable_and_exceptions(
    monkeypatch,
) -> None:
    class _WithList:
        def tolist(self):  # noqa: ANN001
            return [1, 2]

    monkeypatch.setattr(run_mod, "torch", object())
    assert run_mod._tensor_or_list_to_ints(_WithList()) == [1, 2]

    class _WithIterable:
        def tolist(self):  # noqa: ANN001
            return (1, 2)

    assert run_mod._tensor_or_list_to_ints(_WithIterable()) == [1, 2]

    class _BadRaw:
        def __iter__(self):  # noqa: ANN001
            raise RuntimeError("boom")

    class _WithBad:
        def tolist(self):  # noqa: ANN001
            return _BadRaw()

    assert run_mod._tensor_or_list_to_ints(_WithBad()) == []

    monkeypatch.setattr(run_mod, "torch", None)
    assert run_mod._tensor_or_list_to_ints(np.array([1, 2])) == [1, 2]
    assert run_mod._tensor_or_list_to_ints(range(3)) == [0, 1, 2]

    class _BadIter:
        def __iter__(self):  # noqa: ANN001
            raise RuntimeError("boom")

    assert run_mod._tensor_or_list_to_ints(_BadIter()) == []


def test_apply_mlm_masks_zero_prob_sets_labels_and_counts() -> None:
    records = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=object(),
        mask_prob=0.0,
        seed=0,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="p",
    )
    assert total == 0
    assert counts == [0]
    assert records[0]["labels"] == [-100, -100, -100]
    assert records[0]["mlm_masked"] == 0


def test_apply_mlm_masks_requires_mask_token_id() -> None:
    class _Tok:
        vocab_size = 10
        mask_token_id = None

    records = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
    with pytest.raises(RuntimeError):
        run_mod._apply_mlm_masks(
            records,
            tokenizer=_Tok(),
            mask_prob=0.5,
            seed=0,
            random_token_prob=0.0,
            original_token_prob=0.0,
            prefix="p",
        )


def test_apply_mlm_masks_forces_one_mask_and_handles_special_id_exceptions(
    monkeypatch,
) -> None:
    class _IntRaises:
        def __int__(self) -> int:
            raise TypeError("no")

    class _AllSpecialRaises:
        def __iter__(self):  # noqa: ANN001
            raise RuntimeError("boom")

    class _Tok:
        vocab_size = "10"
        mask_token_id = _IntRaises()
        cls_token_id = _IntRaises()
        all_special_ids = _AllSpecialRaises()

    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)

    records = [
        {
            "window_id": "w0",
            "input_ids": [5, 6],
            "attention_mask": [1, 1],
        }
    ]
    total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=_Tok(),
        mask_prob=0.5,
        seed=7,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="p",
    )
    assert total == 1
    assert counts == [1]
    assert records[0]["mlm_masked"] == 1
    assert records[0]["labels"][1] == 6
    assert records[0]["input_ids"][1] == 0


def test_tokenizer_digest_covers_get_vocab_vocab_fallback_and_unknown() -> None:
    class _TokGetVocab:
        def get_vocab(self):  # noqa: ANN001
            return {"a": 1, 2: 3, None: 4}

    digest = run_mod._tokenizer_digest(_TokGetVocab())
    assert isinstance(digest, str) and len(digest) == 64

    class _TokVocabList:
        vocab = [("a", 1), ("b", 2)]
        name_or_path = "x"
        eos_token = "</s>"
        pad_token = "</s>"
        vocab_size = 2

    digest2 = run_mod._tokenizer_digest(_TokVocabList())
    assert isinstance(digest2, str) and len(digest2) == 64

    class _TokBad:
        def get_vocab(self):  # noqa: ANN001
            raise RuntimeError("boom")

        vocab = [("a", 1), "bad"]
        name_or_path = "x"
        eos_token = "</s>"
        pad_token = "</s>"
        vocab_size = "2"

    digest3 = run_mod._tokenizer_digest(_TokBad())
    assert isinstance(digest3, str) and len(digest3) == 64

    class _TokUnserializable:
        name_or_path = object()

    assert run_mod._tokenizer_digest(_TokUnserializable()) == "unknown-tokenizer"
