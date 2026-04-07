from __future__ import annotations

from types import SimpleNamespace

from invarlock.eval.window_planning import (
    _tensor_or_list_to_ints,
    _window_records,
    choose_first_token_sufficient_candidate,
    resolve_effective_windows,
)


def _make_window(
    count: int,
    token_count: int,
    base: int,
    *,
    duplicate_tail: bool,
) -> SimpleNamespace:
    input_ids = [
        [base + offset + idx for idx in range(token_count)] for offset in range(count)
    ]
    attention_masks = [[1] * token_count for _ in range(count)]
    if duplicate_tail and count >= 20:
        input_ids[-1] = list(input_ids[0])
        input_ids[-2] = list(input_ids[1])
    return SimpleNamespace(input_ids=input_ids, attention_masks=attention_masks)


class _DedupingProvider:
    def windows(
        self,
        *,
        seq_len: int,
        preview_n: int,
        final_n: int,
        **_: object,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        token_count = 3 if seq_len <= 16 else max(1, seq_len // 3)
        return (
            _make_window(preview_n, token_count, 100, duplicate_tail=True),
            _make_window(final_n, token_count, 200, duplicate_tail=True),
        )


class _CapacityProvider:
    def windows(
        self,
        *,
        seq_len: int,
        preview_n: int,
        final_n: int,
        **_: object,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        token_count = {512: 200, 768: 300}.get(seq_len, 32)
        return (
            _make_window(preview_n, token_count, 10, duplicate_tail=False),
            _make_window(final_n, token_count, 1000, duplicate_tail=False),
        )


class _FloorFailProvider:
    def windows(
        self,
        *,
        preview_n: int,
        final_n: int,
        **_: object,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        preview = _make_window(preview_n, 3, 500, duplicate_tail=False)
        final = _make_window(final_n, 3, 800, duplicate_tail=False)
        preview.input_ids = [[1, 2, 3] for _ in range(preview_n)]
        final.input_ids = [[4, 5, 6] for _ in range(final_n)]
        return preview, final


def test_resolve_effective_windows_records_dedupe_adjustments_and_token_totals() -> (
    None
):
    result = resolve_effective_windows(
        data_provider=_DedupingProvider(),
        tokenizer=object(),
        seq_len=8,
        stride=8,
        preview_n=20,
        final_n=20,
        seed=42,
        split="validation",
        requested_preview=20,
        requested_final=20,
        profile="ci",
    )

    assert result["actual_preview"] == 15
    assert result["actual_final"] == 15
    assert result["preview_total_tokens"] == 45
    assert result["final_total_tokens"] == 45
    assert result["dedupe_adjustments"] == [{"deficit": 4, "proposed_per_arm": 15}]


def test_choose_first_token_sufficient_candidate_selects_first_viable_schedule() -> (
    None
):
    result = choose_first_token_sufficient_candidate(
        data_provider=_CapacityProvider(),
        tokenizer=object(),
        split="validation",
        seed=42,
        candidates=[
            {"seq_len": 512, "stride": 512, "preview_n": 2, "final_n": 2},
            {"seq_len": 768, "stride": 768, "preview_n": 2, "final_n": 2},
        ],
        min_tokens_target=1000,
        headroom_ratio=1.05,
        profile="ci",
    )

    assert result["status"] == "selected"
    assert result["selected"]["seq_len"] == 768
    assert result["selected"]["total_tokens"] == 1200
    assert result["selected"]["tokens_floor_met"] is True
    assert result["candidates"][0]["reason"] == "below_token_floor"


def test_choose_first_token_sufficient_candidate_reports_no_candidate_on_dedupe_floor() -> (
    None
):
    result = choose_first_token_sufficient_candidate(
        data_provider=_FloorFailProvider(),
        tokenizer=object(),
        split="validation",
        seed=42,
        candidates=[
            {"seq_len": 8, "stride": 8, "preview_n": 10, "final_n": 10},
        ],
        min_tokens_target=1000,
        profile="ci",
    )

    assert result["status"] == "no_candidate"
    assert result["selected"] is None
    assert (
        "Unable to construct non-overlapping windows"
        in result["candidates"][0]["reason"]
    )


def test_tensor_or_list_to_ints_rejects_bool_elements() -> None:
    assert _tensor_or_list_to_ints([True, 2]) == []


def test_tensor_or_list_to_ints_recovers_after_runtime_coercion_failures() -> None:
    class _TensorLike:
        def detach(self):
            raise RuntimeError("detach unavailable")

        def cpu(self):
            raise RuntimeError("cpu unavailable")

        def tolist(self):
            return [1, 2, 3]

    assert _tensor_or_list_to_ints(_TensorLike()) == [1, 2, 3]


def test_window_records_coerces_bad_dataset_index_to_none() -> None:
    window = SimpleNamespace(
        input_ids=[[1, 2, 3]],
        attention_masks=[[1, 1, 1]],
        indices=["bad-index"],
    )

    records, total_tokens = _window_records(window)

    assert total_tokens == 3
    assert records == [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "dataset_index": None,
        }
    ]
