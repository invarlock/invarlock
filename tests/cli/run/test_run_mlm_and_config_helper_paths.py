# ruff: noqa: I001,E402,F811
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from invarlock.cli import run_execution as masking_mod
from invarlock.cli import run_pairing as pairing_mod
from invarlock.cli import run_config as run_serial_mod
from invarlock.core.exceptions import ConfigError
from invarlock.core import run_policy as run_policy_mod
from invarlock.core.run_policy import GUARD_OVERHEAD_THRESHOLD
from invarlock.core.run_policy import choose_dataset_split
from invarlock.core.run_policy import resolve_guard_overhead_threshold
from invarlock.core.run_policy import resolve_pm_acceptance_range


def test_coerce_mapping_covers_multiple_sources_and_failures() -> None:
    assert run_serial_mod._coerce_mapping({"a": 1}) == {"a": 1}

    obj_with_data = SimpleNamespace(_data={"b": 2})
    assert run_serial_mod._coerce_mapping(obj_with_data) == {"b": 2}

    class _DataAttrRaises:
        def __getattribute__(self, name: str):  # noqa: ANN001
            if name == "_data":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

        def model_dump(self):  # noqa: ANN001
            return {"c": 3}

    with pytest.raises(RuntimeError, match="boom"):
        run_serial_mod._coerce_mapping(_DataAttrRaises())

    class _ModelDumpRaises:
        def model_dump(self):  # noqa: ANN001
            raise RuntimeError("boom")

    inst = _ModelDumpRaises()
    inst.x = 1
    with pytest.raises(RuntimeError, match="boom"):
        run_serial_mod._coerce_mapping(inst)

    class _Slots:
        __slots__ = ()

    assert run_serial_mod._coerce_mapping(_Slots()) == {}

    class _C:
        x = 1

    assert run_serial_mod._coerce_mapping(_C) == {}


def test_resolve_pm_acceptance_range_parses_cfg_and_ignores_env(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    assert resolve_pm_acceptance_range(None) == {}
    assert resolve_pm_acceptance_range({}) == {}

    cfg = {"primary_metric": {"acceptance_range": {"min": "bad", "max": "1.2"}}}
    with pytest.raises(ConfigError, match="acceptance_range.min"):
        resolve_pm_acceptance_range(cfg)

    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MIN", "-1")
    monkeypatch.setenv("INVARLOCK_PM_ACCEPTANCE_MAX", "0")
    with pytest.raises(ConfigError, match="acceptance_range.min"):
        resolve_pm_acceptance_range(cfg)


def test_resolve_pm_acceptance_range_ignores_invalid_cfg_max(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    cfg = {"primary_metric": {"acceptance_range": {"min": "1.0", "max": "bad"}}}
    with pytest.raises(ConfigError, match="acceptance_range.max"):
        resolve_pm_acceptance_range(cfg)


def test_resolve_pm_acceptance_range_clamps_invalid_bounds(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    with pytest.raises(ConfigError, match="acceptance_range.min"):
        resolve_pm_acceptance_range(
            {"primary_metric": {"acceptance_range": {"min": -0.1, "max": 1.2}}}
        )

    with pytest.raises(ConfigError, match="acceptance_range.max"):
        resolve_pm_acceptance_range(
            {"primary_metric": {"acceptance_range": {"min": 1.0, "max": 0.0}}}
        )

    with pytest.raises(ConfigError, match="greater than or equal to min"):
        resolve_pm_acceptance_range(
            {"primary_metric": {"acceptance_range": {"min": 1.2, "max": 1.1}}}
        )


def test_resolve_pm_acceptance_range_covers_outer_exception(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    def _boom(_cfg):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(run_policy_mod, "coerce_mapping", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        resolve_pm_acceptance_range(
            {"primary_metric": {"acceptance_range": {"min": 0.9, "max": 1.1}}}
        )


def test_resolve_guard_overhead_threshold_from_config() -> None:
    assert resolve_guard_overhead_threshold(None) == pytest.approx(
        GUARD_OVERHEAD_THRESHOLD
    )
    assert resolve_guard_overhead_threshold(
        {"primary_metric": {"overhead_threshold": 0.025}}
    ) == pytest.approx(0.025)
    with pytest.raises(ConfigError, match="overhead_threshold"):
        resolve_guard_overhead_threshold(
            {"primary_metric": {"overhead_threshold": "bad"}}
        )
    with pytest.raises(ConfigError, match="overhead_threshold"):
        resolve_guard_overhead_threshold({"primary_metric": {"overhead_threshold": -1}})


def test_choose_dataset_split_covers_fallback_and_exception_path() -> None:
    split, used = choose_dataset_split(requested="train", available=["validation"])
    assert split == "train"
    assert used is False

    split, used = choose_dataset_split(requested=None, available=["val", "train"])
    assert split == "val"
    assert used is True

    split, used = choose_dataset_split(requested=None, available=["zzz", "aaa"])
    assert split == "aaa"
    assert used is True

    split, used = choose_dataset_split(requested=None, available=None)
    assert split == "validation"
    assert used is True

    class _BadStr(str):
        def __len__(self) -> int:
            raise RuntimeError("boom")

    split, used = choose_dataset_split(requested=_BadStr("x"), available=["validation"])
    assert split == "x"
    assert used is False


def test_compute_mask_positions_digest_covers_none_digest_and_exception() -> None:
    assert (
        pairing_mod._compute_mask_positions_digest(
            {
                "preview": {"labels": [[-100, -100]]},
                "final": {"labels": [[-100]]},
            }
        )
        is None
    )

    digest = pairing_mod._compute_mask_positions_digest(
        {
            "preview": {"labels": [[-100, 5]]},
            "final": {"labels": [[-100]]},
        }
    )
    assert isinstance(digest, str) and len(digest) == 32

    class _BadDict(dict):
        def get(self, *_a, **_k):  # noqa: ANN001
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pairing_mod._compute_mask_positions_digest(_BadDict())


def test_tensor_or_list_to_ints_covers_tolist_numpy_iterable_and_exceptions(
    monkeypatch,
) -> None:
    class _WithList:
        def tolist(self):  # noqa: ANN001
            return [1, 2]

    monkeypatch.setattr(pairing_mod, "torch", object())
    assert pairing_mod._tensor_or_list_to_ints(_WithList()) == [1, 2]

    class _WithIterable:
        def tolist(self):  # noqa: ANN001
            return (1, 2)

    assert pairing_mod._tensor_or_list_to_ints(_WithIterable()) == [1, 2]

    class _BadRaw:
        def __iter__(self):  # noqa: ANN001
            raise RuntimeError("boom")

    class _WithBad:
        def tolist(self):  # noqa: ANN001
            return _BadRaw()

    with pytest.raises(RuntimeError, match="boom"):
        pairing_mod._tensor_or_list_to_ints(_WithBad())

    monkeypatch.setattr(pairing_mod, "torch", None)
    assert pairing_mod._tensor_or_list_to_ints(np.array([1, 2])) == [1, 2]
    assert pairing_mod._tensor_or_list_to_ints(range(3)) == [0, 1, 2]

    class _BadIter:
        def __iter__(self):  # noqa: ANN001
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pairing_mod._tensor_or_list_to_ints(_BadIter())


def test_apply_mlm_masks_zero_prob_sets_labels_and_counts() -> None:
    records = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    total, counts = masking_mod._apply_mlm_masks(
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
        masking_mod._apply_mlm_masks(
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

    monkeypatch.setattr(masking_mod.random, "random", lambda: 1.0)

    records = [
        {
            "window_id": "w0",
            "input_ids": [5, 6],
            "attention_mask": [1, 1],
        }
    ]
    total, counts = masking_mod._apply_mlm_masks(
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

    digest = masking_mod._tokenizer_digest(_TokGetVocab())
    assert isinstance(digest, str) and len(digest) == 64

    class _TokVocabList:
        vocab = [("a", 1), ("b", 2)]
        name_or_path = "x"
        eos_token = "</s>"
        pad_token = "</s>"
        vocab_size = 2

    digest2 = masking_mod._tokenizer_digest(_TokVocabList())
    assert isinstance(digest2, str) and len(digest2) == 64

    class _TokBad:
        def get_vocab(self):  # noqa: ANN001
            raise RuntimeError("boom")

        vocab = [("a", 1), "bad"]
        name_or_path = "x"
        eos_token = "</s>"
        pad_token = "</s>"
        vocab_size = "2"

    digest3 = masking_mod._tokenizer_digest(_TokBad())
    assert isinstance(digest3, str) and len(digest3) == 64

    class _TokUnserializable:
        name_or_path = object()

    digest4 = masking_mod._tokenizer_digest(_TokUnserializable())
    assert isinstance(digest4, str) and len(digest4) == 64

    class _TokExplodes:
        @property
        def name_or_path(self):  # noqa: ANN201
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        masking_mod._tokenizer_digest(_TokExplodes())
