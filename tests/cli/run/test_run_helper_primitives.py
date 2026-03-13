from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from rich.console import Console

from invarlock.cli.commands import run as run_mod


def test_choose_dataset_split_variants():
    assert run_mod._choose_dataset_split(
        requested="custom", available=["validation"]
    ) == ("custom", False)
    split, used_fallback = run_mod._choose_dataset_split(
        requested=None, available=["train", "dev"]
    )
    assert split == "dev"
    assert used_fallback
    split2, used2 = run_mod._choose_dataset_split(requested=None, available=["xsplit"])
    assert split2 == "xsplit"
    assert used2
    assert run_mod._choose_dataset_split(requested=None, available=None) == (
        "validation",
        True,
    )


def test_suppress_noisy_warnings_release_profile() -> None:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        run_mod._apply_warning_filters("release")
        warnings.warn(
            "`loss_type=None` was set in the config but it is unrecognized.",
            UserWarning,
            stacklevel=2,
        )
        warnings.warn("some other warning", UserWarning, stacklevel=2)
    assert len(records) == 1
    assert "some other warning" in str(records[0].message)


def test_resolve_metric_override_takes_precedence() -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(provider="wikitext2"),
        eval=SimpleNamespace(metric=SimpleNamespace(kind="ppl_causal")),
    )
    model_profile = SimpleNamespace(
        default_metric="ppl_causal",
        default_provider="wikitext2",
    )
    metric_kind, provider_kind, _opts = run_mod._resolve_metric_and_provider(
        cfg,
        model_profile,
        resolved_loss_type="causal",
        metric_kind_override="accuracy",
    )
    assert metric_kind == "accuracy"
    assert provider_kind == "wikitext2"


def test_resolve_pm_acceptance_range_missing_min_uses_default(monkeypatch):
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    out = run_mod._resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"max": 1.2}}}
    )
    assert out == {"min": 0.95, "max": 1.2}


def test_resolve_pm_acceptance_range_missing_max_uses_default(monkeypatch):
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MIN", raising=False)
    monkeypatch.delenv("INVARLOCK_PM_ACCEPTANCE_MAX", raising=False)

    out = run_mod._resolve_pm_acceptance_range(
        {"primary_metric": {"acceptance_range": {"min": 0.9}}}
    )
    assert out == {"min": 0.9, "max": 1.1}


def test_persist_ref_masks_roundtrip(tmp_path):
    assert run_mod._persist_ref_masks({}, tmp_path) is None
    payload = {
        "edit": {
            "artifacts": {
                "mask_payload": {
                    "indices": [1, 2, 3],
                    "meta": {"note": "kept"},
                }
            }
        }
    }
    mask_path = run_mod._persist_ref_masks(payload, tmp_path)
    assert mask_path is not None
    data = json.loads(mask_path.read_text(encoding="utf-8"))
    assert data["indices"] == [1, 2, 3]
    assert "generated_at" in data["meta"]


def test_resolve_exit_code_cases():
    assert (
        run_mod._resolve_exit_code(ValueError("Invalid RunReport data"), profile=None)
        == 2
    )
    invarlock_exc = run_mod.InvarlockError(code="E001", message="boom")
    assert run_mod._resolve_exit_code(invarlock_exc, profile="release") == 3

    class BadProfile:
        def __bool__(self):
            return True

    assert run_mod._resolve_exit_code(RuntimeError("x"), profile=BadProfile()) == 1


def test_extract_model_load_kwargs_rejects_removed_keys():
    from invarlock.cli.config import InvarLockConfig

    cfg = InvarLockConfig(
        {
            "model": {
                "id": "foo",
                "adapter": "dummy",
                "device": "cuda",
                "torch_dtype": "float16",
            }
        }
    )

    with pytest.raises(run_mod.InvarlockError) as excinfo:
        _ = run_mod._extract_model_load_kwargs(cfg)

    assert excinfo.value.code == "E007"
    assert excinfo.value.details.get("removed_keys") == ["torch_dtype"]


def test_hash_and_mask_digests():
    digest = run_mod._hash_sequences([[1, 2, 3], [4, 5]])
    assert digest == "e08215eb1a73f6d493dfb9f17c0de613"
    windows = {
        "preview": {"labels": [[-100, 42, -100], [3, -100]]},
        "final": {"labels": [[-100, -100]]},
    }
    mask_digest = run_mod._compute_mask_positions_digest(windows)
    assert mask_digest == "bb77b7fbd60b8716abfbcec6f3e2e822"
    assert (
        run_mod._compute_mask_positions_digest({"preview": {"labels": [[-100]]}})
        is None
    )
    # Rows with no tokens should be ignored without tripping the digest logic
    empty_row = {
        "preview": {"labels": [[], [-100, -100]]},
        "final": {"labels": [[-100]]},
    }
    assert run_mod._compute_mask_positions_digest(empty_row) is None


def test_tensor_or_list_to_ints_variants(monkeypatch):
    fake_tensor = SimpleNamespace(tolist=lambda: [1, 2, 3])
    monkeypatch.setattr(run_mod, "torch", SimpleNamespace(), raising=False)
    assert run_mod._tensor_or_list_to_ints(fake_tensor) == [1, 2, 3]
    monkeypatch.setattr(run_mod, "torch", None, raising=False)
    arr = np.array([4, 5])
    assert run_mod._tensor_or_list_to_ints(arr) == [4, 5]
    gen = (v for v in [6, 7])
    assert run_mod._tensor_or_list_to_ints(gen) == [6, 7]
    assert run_mod._tensor_or_list_to_ints(object()) == []


def test_apply_mlm_masks_zero_path_and_missing_tokenizer():
    records = [{"input_ids": [10, 11], "attention_mask": [1, 1]}]
    tokenizer = SimpleNamespace(mask_token_id=0)
    masked, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.0,
        seed=123,
        random_token_prob=0.1,
        original_token_prob=0.1,
        prefix="test",
    )
    assert masked == 0
    assert counts == [0]
    assert records[0]["labels"] == [-100, -100]

    missing_mask_tokenizer = SimpleNamespace(mask_token_id=None)
    with pytest.raises(RuntimeError):
        run_mod._apply_mlm_masks(
            [{"input_ids": [1], "attention_mask": [1]}],
            tokenizer=missing_mask_tokenizer,
            mask_prob=0.5,
            seed=0,
            random_token_prob=0.0,
            original_token_prob=0.0,
            prefix="test",
        )


def test_apply_mlm_masks_masks_tokens():
    tokenizer = SimpleNamespace(
        mask_token_id=99,
        vocab_size=1000,
        cls_token_id=0,
        sep_token_id=1,
        pad_token_id=2,
        all_special_ids=[0, 1, 2],
    )
    records = [
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1]},
    ]
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=1.0,
        seed=123,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="mask",
    )
    assert masked_total == counts[0] == records[0]["mlm_masked"]
    assert any(label != -100 for label in records[0]["labels"])


def test_apply_mlm_masks_skips_attention_gaps_and_specials():
    tokenizer = SimpleNamespace(
        mask_token_id=42,
        vocab_size=100,
        all_special_ids=[101],
        pad_token_id=0,
    )
    records = [
        {
            "window_id": "special:0",
            "input_ids": [101, 7, 9],
            "attention_mask": [1, 0, 1],
        }
    ]
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=1.0,
        seed=3,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="special",
    )
    assert masked_total == counts[0]
    labels = records[0]["labels"]
    # Attention gap and special tokens stay unmasked
    assert labels[0] == -100
    assert labels[1] == -100
    assert labels[2] != -100


def test_apply_mlm_masks_no_candidates_skips_fallback(monkeypatch):
    tokenizer = SimpleNamespace(
        mask_token_id=9,
        vocab_size=50,
        all_special_ids=[123],
    )
    records = [
        {
            "window_id": "special-only",
            "input_ids": [123],
            "attention_mask": [1],
        }
    ]
    # Force the stochastic mask sampling to skip masking entirely
    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=5,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="special",
    )
    assert masked_total == 0
    assert counts == [0]
    assert records[0]["mlm_masked"] == 0


def test_apply_mlm_masks_fallback_random_replacement(monkeypatch):
    tokenizer = SimpleNamespace(
        mask_token_id=77,
        vocab_size=50,
        all_special_ids=[],
    )
    records = [
        {
            "input_ids": [11, 13],
            "attention_mask": [1, 1],
        }
    ]
    # Ensure the primary sampling path never masks tokens so fallback triggers
    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=2,
        random_token_prob=0.1,
        original_token_prob=0.05,
        prefix="fallback",
    )
    assert masked_total == counts[0] == records[0]["mlm_masked"] == 1
    labels = records[0]["labels"]
    assert labels[1] == 13  # original token captured before replacement
    assert records[0]["input_ids"][1] == 36  # rng.randrange branch executed


def test_apply_mlm_masks_fallback_masks_token_alt(monkeypatch):
    # Force no initial masks so the fallback path triggers
    tokenizer = SimpleNamespace(mask_token_id=55, vocab_size=100, all_special_ids=[])
    records = [
        {"input_ids": [11, 13], "attention_mask": [1, 1], "window_id": "fb"},
    ]
    # Ensure the primary sampling skips masking entirely
    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=7,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="fb",
    )
    assert masked_total == counts[0] == 1
    # Fallback should choose mask_token_id branch when probs sum to 0.0
    assert 55 in records[0]["input_ids"]


def test_apply_mlm_masks_fallback_keep_original(monkeypatch):
    # Force primary sampling to skip masking entirely so fallback triggers
    tokenizer = SimpleNamespace(mask_token_id=77, vocab_size=100, all_special_ids=[])
    records = [
        {"input_ids": [11, 13], "attention_mask": [1, 1], "window_id": "fb2"},
    ]
    # Primary path never masks
    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)

    class FixedRNG:
        def __init__(self, *_a, **_k):  # noqa: D401
            pass

        def random(self):
            return 0.99  # ensure else branch in fallback → keep original token

    # Make Random() produce a generator with fixed random()
    monkeypatch.setattr(run_mod.random, "Random", FixedRNG)

    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=2,
        random_token_prob=0.1,
        original_token_prob=0.8,
        prefix="fb2",
    )
    assert masked_total == counts[0] == 1
    # Fallback chose to keep original token (no mask/replacement), but labels capture original
    assert records[0]["labels"].count(-100) == len(records[0]["labels"]) - 1
    # Input id at masked position remains one of the originals
    assert set(records[0]["input_ids"]) <= {11, 13, 77}


def test_choose_dataset_split_sort_fallback():
    # Alias present → prefers alias 'dev'
    split, used = run_mod._choose_dataset_split(
        requested=None, available=["zebra", "dev", "apple"]
    )
    assert split == "dev" and used is True
    # No alias present → sorted fallback to first element
    split2, used2 = run_mod._choose_dataset_split(
        requested=None, available=["zebra", "apple"]
    )
    assert split2 == "apple" and used2 is True


def test_apply_mlm_masks_original_token_branch(monkeypatch):
    tokenizer = SimpleNamespace(
        mask_token_id=101,
        vocab_size=128,
        all_special_ids=[],
    )
    records = [
        {"input_ids": [21, 22], "attention_mask": [1, 1], "window_id": "keep"},
    ]
    calls = {"n": 0}

    def fake_random():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1.0

    monkeypatch.setattr(run_mod.random, "random", fake_random)
    masked_total, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=1.0,
        seed=11,
        random_token_prob=0.0,
        original_token_prob=1.0,
        prefix="keep",
    )
    assert masked_total == counts[0] == 1
    assert records[0]["labels"][0] == 21
    assert records[0]["input_ids"][0] == 21


def test_apply_mlm_masks_fallback_masks_token(monkeypatch):
    tokenizer = SimpleNamespace(
        mask_token_id=55,
        vocab_size=100,
        all_special_ids=[],
    )
    records = [
        {"input_ids": [7, 8], "attention_mask": [1, 1]},
    ]
    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    masked, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.2,
        seed=5,
        random_token_prob=0.0,
        original_token_prob=0.0,
        prefix="fallback",
    )
    assert masked == counts[0] == 1
    assert 55 in records[0]["input_ids"]


def test_apply_mlm_masks_fallback_random_branch(monkeypatch):
    """Ensure fallback branch can replace tokens via rng.randrange."""
    tokenizer = SimpleNamespace(mask_token_id=77, vocab_size=100, all_special_ids=[])
    records = [
        {"input_ids": [9, 13], "attention_mask": [1, 1], "window_id": "fallback"},
    ]

    class FakeRandom:
        def __init__(self, seed):
            self.seed = seed

        def random(self):
            return 0.85  # Between the first and second thresholds

        def randrange(self, limit):
            return min(5, max(0, limit - 1))

    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    monkeypatch.setattr(run_mod.random, "Random", FakeRandom)
    masked, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=9,
        random_token_prob=0.2,
        original_token_prob=0.1,
        prefix="fallback",
    )
    assert masked == counts[0] == 1
    # Should have replaced token via rng.randrange rather than mask_token_id
    assert records[0]["input_ids"][1] == 5


def test_apply_mlm_masks_fallback_keeps_original(monkeypatch):
    """Fallback should leave token unchanged when both branches skip."""
    tokenizer = SimpleNamespace(mask_token_id=77, vocab_size=100, all_special_ids=[])
    records = [
        {"input_ids": [3, 4], "attention_mask": [1, 1], "window_id": "skip"},
    ]

    class FakeRandom:
        def __init__(self, seed):
            self.seed = seed

        def random(self):
            return 0.99  # Above both thresholds

        def randrange(self, limit):
            raise AssertionError("should not be called")

    monkeypatch.setattr(run_mod.random, "random", lambda: 1.0)
    monkeypatch.setattr(run_mod.random, "Random", FakeRandom)
    masked, counts = run_mod._apply_mlm_masks(
        records,
        tokenizer=tokenizer,
        mask_prob=0.5,
        seed=3,
        random_token_prob=0.2,
        original_token_prob=0.1,
        prefix="skip",
    )
    assert masked == counts[0] == 1
    assert records[0]["input_ids"][1] == 4


def test_tokenizer_digest_prefers_get_vocab():
    class FakeMap:
        def items(self):
            return [("b", 2), ("a", 1)]

    class FakeTokenizer:
        def get_vocab(self):
            return FakeMap()

    digest = run_mod._tokenizer_digest(FakeTokenizer())
    expected_pairs = [("a", 1), ("b", 2)]
    import hashlib

    payload = json.dumps(expected_pairs, separators=(",", ":")).encode()
    assert digest == hashlib.sha256(payload).hexdigest()


def test_tokenizer_digest_fallbacks():
    class VocabListTokenizer:
        vocab = [("c", 3), ("d", 4)]

    digest_list = run_mod._tokenizer_digest(VocabListTokenizer())
    payload = json.dumps([("c", 3), ("d", 4)], separators=(",", ":")).encode()
    import hashlib

    assert digest_list == hashlib.sha256(payload).hexdigest()

    class AttrTokenizer:
        name_or_path = "attr"
        eos_token = "</s>"
        pad_token = "<pad>"
        vocab_size = 321

    digest_attr = run_mod._tokenizer_digest(AttrTokenizer())
    assert digest_attr != "unknown-tokenizer"


def test_tokenizer_digest_unknown_path():
    class Explode:
        def __getattr__(self, item):
            raise RuntimeError("boom")

    digest = run_mod._tokenizer_digest(Explode())
    assert digest == "unknown-tokenizer"


def test_tokenizer_digest_get_vocab_items_not_callable():
    # get_vocab exists but returns an object with non-callable 'items' attribute
    class WeirdMap:
        items = 123  # not callable

    class Tok:
        def __init__(self):
            self.name_or_path = "weird"
            self.vocab_size = 10

        def get_vocab(self):
            return WeirdMap()

    # Should skip the get_vocab branch and fall back to attrs digest path
    digest = run_mod._tokenizer_digest(Tok())
    assert isinstance(digest, str) and len(digest) == 64


def test_tokenizer_digest_get_vocab_missing_items():
    class Tok:
        def get_vocab(self):
            return SimpleNamespace()

        vocab = [("x", 1)]

    digest = run_mod._tokenizer_digest(Tok())
    payload = json.dumps([("x", 1)], separators=(",", ":")).encode()
    import hashlib

    assert digest == hashlib.sha256(payload).hexdigest()


def test_validate_and_harvest_baseline_schedule_mismatch(monkeypatch):
    # Minimal cfg stub
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(seq_len=128, stride=64, preview_n=2, final_n=1)
    )
    pairing = {
        "preview": {"input_ids": [[1, 2]]},
        "final": {"input_ids": [[3, 4]]},
    }
    # Baseline meta with mismatched seq_len/stride
    baseline = {"data": {"seq_len": 256, "stride": 32, "dataset": "wikitext2"}}

    with pytest.raises(run_mod.typer.Exit):
        run_mod._validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


def test_validate_and_harvest_baseline_schedule_counts_adjust():
    # Console capture
    outputs: list[str] = []

    class CaptureConsole:
        def print(self, *args, **kwargs):  # noqa: D401
            outputs.append(" ".join(str(a) for a in args))

    # cfg requests different counts than the schedule
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(seq_len=128, stride=64, preview_n=5, final_n=5)
    )
    pairing = {
        "preview": {
            "window_ids": [0, 1],
            "input_ids": [[1, 2], [3, 4]],
            "attention_masks": [[1, 1], [1, 1]],
        },
        "final": {
            "window_ids": [2],
            "input_ids": [[5, 6]],
            "attention_masks": [[1, 1]],
        },
    }
    baseline = {"data": {"seq_len": 128, "stride": 64, "dataset": "wikitext2"}}

    out = run_mod._validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=CaptureConsole(),
    )
    # Effective counts match schedule, not cfg
    assert out["preview_count"] == 2 and out["final_count"] == 1
    # Warning line captured
    assert any("Adjusting evaluation window counts" in s for s in outputs)


def test_validate_and_harvest_baseline_schedule_duplicate_preview_ids():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(seq_len=128, stride=64, preview_n=2, final_n=1)
    )
    pairing = {
        "preview": {
            "window_ids": [0, 0],
            "input_ids": [[1, 2], [3, 4]],
            "attention_masks": [[1, 1], [1, 1]],
        },
        "final": {
            "window_ids": [1],
            "input_ids": [[5, 6]],
            "attention_masks": [[1, 1]],
        },
    }
    baseline = {"data": {"seq_len": 128, "stride": 64, "dataset": "wikitext2"}}

    with pytest.raises(typer.Exit):
        run_mod._validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


def test_validate_and_harvest_baseline_schedule_overlap_ids_wraps_masks():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(seq_len=128, stride=64, preview_n=1, final_n=1)
    )
    pairing = {
        "preview": {
            "window_ids": [0],
            "input_ids": [[1, 2]],
            "attention_masks": [1, 1],
        },
        "final": {
            "window_ids": [0],
            "input_ids": [[3, 4]],
            "attention_masks": [1, 1],
        },
    }
    baseline = {"data": {"seq_len": 128, "stride": 64, "dataset": "wikitext2"}}

    with pytest.raises(typer.Exit):
        run_mod._validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=None,
        )


def test_tokenizer_digest_filters_non_string_keys():
    class WeirdTokenizer:
        def get_vocab(self):
            class Map:
                def items(self):
                    return [(1, 5), ("b", 3)]

            return Map()

    digest = run_mod._tokenizer_digest(WeirdTokenizer())
    expected = sorted([("1", 5), ("b", 3)])
    payload = json.dumps(expected, separators=(",", ":")).encode()
    import hashlib

    assert digest == hashlib.sha256(payload).hexdigest()


def test_tokenizer_digest_unknown_fallback():
    class ErrorTokenizer:
        def __getattr__(self, item):
            raise RuntimeError("boom")

    assert run_mod._tokenizer_digest(ErrorTokenizer()) == "unknown-tokenizer"


def test_extract_pairing_schedule_success():
    report = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [0],
                "input_ids": [[1, 2, 3]],
                "attention_masks": [[1, 1, 1]],
                "labels": [[9, 9]],
                "masked_token_counts": [2],
                "actual_token_counts": [3],
            },
            "final": {
                "window_ids": [1],
                "input_ids": [[4, 5]],
                "attention_masks": [[1, 0]],
            },
        }
    }
    schedule = run_mod._extract_pairing_schedule(report)
    assert schedule is not None
    assert schedule["preview"]["window_ids"] == [0]
    assert schedule["preview"]["masked_token_counts"] == [2]
    assert schedule["preview"]["actual_token_counts"] == [3]
    # Labels padded to match input length
    assert schedule["preview"]["labels"][0] == [9, 9, -100]
    assert schedule["final"]["attention_masks"][0] == [1, 0]


def test_extract_pairing_schedule_missing_section():
    assert run_mod._extract_pairing_schedule(None) is None
    assert run_mod._extract_pairing_schedule({"evaluation_windows": {}}) is None


def test_extract_pairing_schedule_defaults_and_truncation():
    report = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [0],
                "input_ids": [[1, 0]],
                "labels": [[9, 8, 7]],
            },
            "final": {
                "window_ids": [1],
                "input_ids": [[4, 5]],
            },
        }
    }
    schedule = run_mod._extract_pairing_schedule(report)
    assert schedule is not None
    assert schedule["final"]["attention_masks"][0] == [1, 1]
    assert schedule["preview"]["labels"][0] == [9, 8]


def test_extract_pairing_schedule_autofills_ids_masks_and_wraps_scalars():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2, 0]],
                "labels": [100, -100, -100],
                "masked_token_counts": 2,
                "actual_token_counts": 3,
            },
            "final": {
                "input_ids": [[3, 4]],
                "attention_masks": [1, 1],
            },
        }
    }

    schedule = run_mod._extract_pairing_schedule(report)
    assert schedule is not None
    assert schedule["preview"]["window_ids"] == [0]
    assert schedule["final"]["window_ids"] == [1]
    assert schedule["preview"]["attention_masks"] == [[1, 1, 0]]
    assert schedule["final"]["attention_masks"] == [[1, 1]]
    assert schedule["preview"]["labels"] == [[100, -100, -100]]
    assert schedule["preview"]["masked_token_counts"] == [2]
    assert schedule["preview"]["actual_token_counts"] == [3]


def test_extract_pairing_schedule_rejects_window_id_length_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {"window_ids": [0, 1], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": [2], "input_ids": [[4, 5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_empty_input_ids():
    report = {
        "evaluation_windows": {
            "preview": {"input_ids": []},
            "final": {"input_ids": [[1, 2, 3]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_falls_back_for_non_2d_attention_masks():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 0], [2, 3]],
                "attention_masks": [1, 1],
            },
            "final": {"input_ids": [[4, 0]]},
        }
    }

    schedule = run_mod._extract_pairing_schedule(report)
    assert schedule is not None
    assert schedule["preview"]["attention_masks"] == [[1, 0], [1, 1]]
    assert schedule["final"]["attention_masks"] == [[1, 0]]


def test_extract_pairing_schedule_rejects_attention_mask_row_count_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "attention_masks": [[1, 1]],
            },
            "final": {"input_ids": [[5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_attention_mask_token_length_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "attention_masks": [[1, 1], [1]],
            },
            "final": {"input_ids": [[5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_labels_row_count_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "labels": [[-100, -100]],
            },
            "final": {"input_ids": [[5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_masked_token_counts_row_count_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "masked_token_counts": [0],
            },
            "final": {"input_ids": [[5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_actual_token_counts_row_count_mismatch():
    report = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[1, 2], [3, 4]],
                "actual_token_counts": [2],
            },
            "final": {"input_ids": [[5, 6]]},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_extract_pairing_schedule_rejects_missing_final_section():
    report = {
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2, 3]]},
            "final": {},
        }
    }
    assert run_mod._extract_pairing_schedule(report) is None


def test_resolve_device_invalid(monkeypatch):
    class Cfg:
        model = SimpleNamespace(device=None)
        output = SimpleNamespace(dir=None)

    console = Console(file=io.StringIO())
    monkeypatch.setattr(
        "invarlock.cli.device.resolve_device", lambda name: name.upper(), raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config",
        lambda device: (False, "unsupported"),
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        run_mod._resolve_device_and_output(
            Cfg(), device="cuda", out=None, console=console
        )
    assert exc.value.exit_code == 1


def test_resolve_device_output_fallback(monkeypatch, tmp_path):
    class CfgOutIgnored:
        model = SimpleNamespace(device=None)
        output = SimpleNamespace(dir=None)
        out = SimpleNamespace(dir=str(tmp_path / "ignored"))

    console = Console(file=io.StringIO())
    monkeypatch.setattr(
        "invarlock.cli.device.resolve_device", lambda name: "cpu", raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config",
        lambda device: (True, ""),
        raising=False,
    )
    monkeypatch.chdir(tmp_path)
    resolved, out_dir = run_mod._resolve_device_and_output(
        CfgOutIgnored(), device=None, out=None, console=console
    )
    assert resolved == "cpu"
    assert out_dir == Path("runs")

    class CfgFallback:
        model = SimpleNamespace(device=None)
        output = SimpleNamespace(dir=None)

    monkeypatch.chdir(tmp_path)
    resolved2, out_dir2 = run_mod._resolve_device_and_output(
        CfgFallback(), device=None, out=None, console=console
    )
    assert out_dir2.name == "runs"
    assert out_dir2.exists()


def test_resolve_provider_and_split_with_obj_items_variant(monkeypatch):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            provider=SimpleNamespace(
                kind="synthetic",
                items=lambda: [("extra_arg", 3)],
            )
        )
    )

    class Provider:
        def available_splits(self):
            return ["custom_split"]

    captured_kwargs = {}

    def fake_get_provider(name, **kwargs):
        captured_kwargs.update(kwargs)
        return Provider()

    provider, split, used = run_mod._resolve_provider_and_split(
        cfg,
        SimpleNamespace(default_provider=None),
        get_provider_fn=fake_get_provider,
        console=Console(file=io.StringIO()),
    )
    assert isinstance(provider, Provider)
    assert split == "custom_split" and used is True
    assert captured_kwargs == {"extra_arg": 3}


def test_resolve_provider_and_split_available_raises_variant(monkeypatch):
    cfg = SimpleNamespace(dataset=SimpleNamespace(provider="synthetic"))

    class Provider:
        def available_splits(self):
            raise RuntimeError("boom")

    provider, split, used = run_mod._resolve_provider_and_split(
        cfg,
        SimpleNamespace(default_provider=None),
        get_provider_fn=lambda *a, **k: Provider(),
        console=Console(file=io.StringIO()),
    )
    assert isinstance(provider, Provider)
    assert split == "validation" and used is True


def test_resolve_provider_and_split_default_provider():
    cfg = SimpleNamespace(dataset=SimpleNamespace(provider=None, split=None))

    class Profile:
        default_provider = "synthetic"

    class Provider:
        def available_splits(self):
            return []

    provider, split, used = run_mod._resolve_provider_and_split(
        cfg,
        Profile(),
        get_provider_fn=lambda *a, **k: Provider(),
        console=Console(file=io.StringIO()),
    )
    assert isinstance(provider, Provider)
    assert split == "validation" and used is True


def test_resolve_provider_and_split_items_populate_kwargs():
    class ProviderObj:
        kind = "hf_text"

        def items(self):
            return [("kind", "hf_text"), ("text_field", "body"), ("max_samples", 4)]

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(provider=ProviderObj(), split="train")
    )
    captured_kwargs = {}

    class Provider:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def available_splits(self):
            return ["dev", "train"]

    provider, split, used = run_mod._resolve_provider_and_split(
        cfg,
        SimpleNamespace(default_provider=None),
        get_provider_fn=lambda name, **kwargs: Provider(**kwargs),
        console=Console(file=io.StringIO()),
    )
    assert isinstance(provider, Provider)
    assert split == "train" and used is False
    assert captured_kwargs == {"text_field": "body", "max_samples": 4}


def test_enforce_provider_parity_noop_profile():
    # Should not raise outside CI/Release profiles
    run_mod._enforce_provider_parity({}, {}, profile="dev")


def test_enforce_provider_parity_missing_digest_raises():
    with pytest.raises(run_mod.InvarlockError) as exc:
        run_mod._enforce_provider_parity({}, {}, profile="ci")
    assert exc.value.code == "E004"


def test_enforce_provider_parity_tokenizer_mismatch():
    subject = {"ids_sha256": "ids", "tokenizer_sha256": "abc"}
    baseline = {"ids_sha256": "ids", "tokenizer_sha256": "xyz"}
    with pytest.raises(run_mod.InvarlockError) as exc:
        run_mod._enforce_provider_parity(subject, baseline, profile="release")
    assert exc.value.code == "E002"


def test_enforce_provider_parity_mask_mismatch():
    subject = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "m1"}
    baseline = {"ids_sha256": "ids", "tokenizer_sha256": "abc", "masking_sha256": "m2"}
    with pytest.raises(run_mod.InvarlockError) as exc:
        run_mod._enforce_provider_parity(subject, baseline, profile="ci")
    assert exc.value.code == "E003"


def test_validate_harvest_schedule_seq_stride_mismatch(monkeypatch):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )
    )
    pairing = {
        "preview": {
            "input_ids": [[0, 1]],
            "window_ids": [0],
            "attention_masks": [[1, 1]],
        },
        "final": {
            "input_ids": [[2, 3]],
            "window_ids": [1],
            "attention_masks": [[1, 1]],
        },
    }
    baseline = {
        "data": {
            "seq_len": 6,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }
    console = Console(file=io.StringIO())
    with pytest.raises(typer.Exit):
        run_mod._validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=console,
        )


def test_validate_harvest_schedule_count_adjust(monkeypatch):
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            preview_n=2,
            final_n=2,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )
    )
    pairing = {
        "preview": {
            "input_ids": [[0, 1]],
            "window_ids": [0],
            "attention_masks": [[1, 1]],
        },
        "final": {
            "input_ids": [[2, 3]],
            "window_ids": [1],
            "attention_masks": [[1, 1]],
        },
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "validation",
        }
    }
    out = io.StringIO()
    console = Console(file=out)
    result = run_mod._validate_and_harvest_baseline_schedule(
        cfg,
        pairing,
        baseline,
        tokenizer_hash=None,
        resolved_loss_type="causal",
        baseline_path_str="baseline.json",
        console=console,
    )
    text = out.getvalue()
    assert "Adjusting evaluation window counts" in text
    assert result["effective_preview"] == 1 and result["effective_final"] == 1


def test_run_bare_control_non_finite_warns(monkeypatch):
    # Patch CoreRunner to return non-finite PPL in CI profile
    class DummyRunner:
        def execute(self, **kwargs):  # noqa: D401, ARG002
            # Minimal report-like object
            return {
                "status": "ok",
                "metrics": {"primary_metric": {"preview": float("nan"), "final": 0.0}},
            }

    # Minimal adapter and config stubs
    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    # Patch CoreRunner in its source module so local import sees the stub
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(), raising=False
    )
    # Also stub out snapshot extractor to avoid side-effects
    monkeypatch.setattr(
        run_mod, "_extract_pm_snapshot_for_overhead", lambda *a, **k: {"ok": True}
    )

    out = io.StringIO()
    console = Console(file=out)
    payload = run_mod._run_bare_control(
        adapter=Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=Cfg(),
        model=None,
        run_config=SimpleNamespace(context={}),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=console,
        resolved_loss_type="seq2seq",
        profile_normalized="ci",
    )
    text = out.getvalue()
    assert "non-finite" in text
    # Ensure seq2seq branch executed by checking payload exists (no exception)
    assert isinstance(payload, dict) and payload.get("source") == "ci_profile"


def test_run_bare_control_mlm_branch(monkeypatch):
    # Exercise the 'mlm' pm_kind selection path
    class DummyRunner:
        def execute(self, **kwargs):  # noqa: D401, ARG002
            return {
                "status": "ok",
                "metrics": {"primary_metric": {"preview": 0.0, "final": 0.0}},
            }

    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(), raising=False
    )
    monkeypatch.setattr(
        run_mod, "_extract_pm_snapshot_for_overhead", lambda *a, **k: {"ok": True}
    )

    payload = run_mod._run_bare_control(
        adapter=Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=Cfg(),
        model=None,
        run_config=SimpleNamespace(context={}),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=Console(file=io.StringIO()),
        resolved_loss_type="mlm",
        profile_normalized="dev",
    )
    assert isinstance(payload, dict) and payload.get("source") == "dev_profile"


def test_run_bare_control_frees_private_reload_model(monkeypatch):
    events: list[tuple[str, object | None]] = []
    private_model = object()
    shared_model = object()

    class DummyRunner:
        def execute(self, **kwargs):  # noqa: D401, ARG002
            events.append(("execute", kwargs.get("model")))
            return {
                "status": "ok",
                "metrics": {"primary_metric": {"preview": 1.0, "final": 1.0}},
            }

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(), raising=False
    )
    monkeypatch.setattr(
        run_mod, "_extract_pm_snapshot_for_overhead", lambda *a, **k: {}
    )

    def _load_model_with_cfg(adapter, cfg, device, **_kwargs):  # noqa: ARG001
        events.append(("load", None))
        return private_model

    monkeypatch.setattr(run_mod, "_load_model_with_cfg", _load_model_with_cfg)
    monkeypatch.setattr(
        run_mod, "_free_model_memory", lambda m: events.append(("free", m))
    )

    run_mod._run_bare_control(
        adapter=SimpleNamespace(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=SimpleNamespace(model=SimpleNamespace(id="dummy")),
        model=None,
        run_config=SimpleNamespace(context={}),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=Console(file=io.StringIO()),
        resolved_loss_type="causal",
        profile_normalized="ci",
    )

    assert [e[0] for e in events].count("free") == 1
    assert events.index(("execute", private_model)) < events.index(
        ("free", private_model)
    )

    events.clear()
    monkeypatch.setattr(
        run_mod,
        "_load_model_with_cfg",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected load")),
    )
    run_mod._run_bare_control(
        adapter=SimpleNamespace(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=SimpleNamespace(model=SimpleNamespace(id="dummy")),
        model=shared_model,
        run_config=SimpleNamespace(context={}),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=lambda: events.append(("restore", None)),
        console=Console(file=io.StringIO()),
        resolved_loss_type="causal",
        profile_normalized="ci",
    )
    assert ("execute", shared_model) in events
    assert not any(kind == "free" for kind, _ in events)


def test_run_bare_control_skip_model_load_uses_stub(monkeypatch):
    seen: dict[str, object] = {}

    class DummyRunner:
        def execute(self, **kwargs):  # noqa: D401, ARG002
            seen["model"] = kwargs.get("model")
            return {
                "status": "ok",
                "metrics": {"primary_metric": {"preview": 1.0, "final": 1.0}},
            }

    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(), raising=False
    )
    monkeypatch.setattr(
        run_mod, "_extract_pm_snapshot_for_overhead", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        run_mod,
        "_load_model_with_cfg",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected load")),
    )

    payload = run_mod._run_bare_control(
        adapter=Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=Cfg(),
        model=None,
        run_config=SimpleNamespace(context={}, event_path=None),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=Console(file=io.StringIO()),
        resolved_loss_type="causal",
        profile_normalized="dev",
        skip_model_load=True,
    )
    assert isinstance(payload, dict)
    assert getattr(seen.get("model"), "name", "") == "bare_stub_model"


def test_run_bare_control_ci_finite_does_not_warn(monkeypatch):
    class DummyReport:
        status = "ok"
        metrics = {"primary_metric": {"preview": 1.0, "final": 1.0}}

    class DummyRunner:
        def execute(self, **kwargs):  # noqa: D401, ARG002
            return DummyReport()

    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(), raising=False
    )
    monkeypatch.setattr(
        run_mod, "_extract_pm_snapshot_for_overhead", lambda *a, **k: {}
    )

    out = io.StringIO()
    payload = run_mod._run_bare_control(
        adapter=Adapter(),
        edit_op=SimpleNamespace(name="noop"),
        cfg=Cfg(),
        model=None,
        run_config=SimpleNamespace(context={}, event_path=None),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=Console(file=out),
        resolved_loss_type="causal",
        profile_normalized="ci",
    )
    assert isinstance(payload, dict)
    assert "non-finite" not in out.getvalue()


def test_run_bare_control_restore_fn_failure_raises() -> None:
    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    with pytest.raises(run_mod._SnapshotRestoreFailed):
        run_mod._run_bare_control(
            adapter=Adapter(),
            edit_op=SimpleNamespace(name="noop"),
            cfg=Cfg(),
            model=object(),
            run_config=SimpleNamespace(context={}, event_path=None),
            calibration_data=[],
            auto_config={},
            edit_config={},
            preview_count=1,
            final_count=1,
            seed_bundle={"python": 0, "numpy": 0, "torch": None},
            resolved_device="cpu",
            restore_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            console=Console(file=io.StringIO()),
            resolved_loss_type="causal",
            profile_normalized="ci",
        )


def test_compute_provider_digest_paths():
    # None cases: missing windows or wrong types
    assert run_mod._compute_provider_digest({}) is None
    assert run_mod._compute_provider_digest({"evaluation_windows": None}) is None

    # Mixed sections: preview not dict triggers continue; final contributes ids
    report = {
        "evaluation_windows": {
            "preview": None,
            "final": {"window_ids": ["f1", "f2"], "labels": [[-100, 1], [-100, -100]]},
        },
        "meta": {"tokenizer_hash": "abc"},
    }
    digest = run_mod._compute_provider_digest(report)
    assert digest and "ids_sha256" in digest and digest.get("tokenizer_sha256") == "abc"


def test_compute_provider_digest_data_tokenizer_fallback():
    report = {
        "evaluation_windows": {
            "preview": {"window_ids": ["p"], "labels": [[0, -100]]},
            "final": {"window_ids": ["f"]},
        },
        "data": {"tokenizer_hash": "xyz"},
    }
    digest = run_mod._compute_provider_digest(report)
    assert digest and digest.get("tokenizer_sha256") == "xyz"


def test_compute_provider_digest_int_ids_no_masking():
    from invarlock.utils.digest import hash_json

    report = {
        "evaluation_windows": {
            "preview": {"window_ids": [0, 1]},
            "final": {"window_ids": [2]},
        },
        "meta": {"tokenizer_hash": "tok"},
    }
    digest = run_mod._compute_provider_digest(report)
    assert digest is not None
    assert digest.get("ids_sha256") == hash_json([0, 1, 2])
    assert digest.get("tokenizer_sha256") == "tok"
    assert "masking_sha256" not in digest


def test_compute_provider_digest_no_window_ids_returns_tokenizer_only():
    report = {
        "evaluation_windows": {
            "preview": {"labels": [[-100, -100]]},
            "final": {"labels": [[-100]]},
        },
        "data": {"tokenizer_hash": "tok"},
    }
    digest = run_mod._compute_provider_digest(report)
    assert digest == {"tokenizer_sha256": "tok"}


def test_run_bare_control_passes_edit_op_to_core_runner(monkeypatch):
    # Ensure the bare-control path uses the same edit op as the guarded run
    sentinel = SimpleNamespace(name="sentinel_edit")

    class DummyRunner:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def execute(self, **kwargs):  # noqa: D401, ARG002
            self.kwargs = dict(kwargs)
            # Minimal report-like object with finite primary metric
            return {
                "status": "ok",
                "metrics": {"primary_metric": {"preview": 1.0, "final": 2.0}},
            }

    dummy = DummyRunner()

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: dummy, raising=False
    )

    class Adapter:
        def load_model(self, model_id, device=None):  # noqa: ARG002
            return object()

    class Cfg:
        class _Model:
            id = "dummy"

        model = _Model()

    payload = run_mod._run_bare_control(
        adapter=Adapter(),
        edit_op=sentinel,
        cfg=Cfg(),
        model=None,
        run_config=SimpleNamespace(context={}),
        calibration_data=[],
        auto_config={},
        edit_config={},
        preview_count=1,
        final_count=1,
        seed_bundle={"python": 0, "numpy": 0, "torch": None},
        resolved_device="cpu",
        restore_fn=None,
        console=Console(file=io.StringIO()),
        resolved_loss_type="causal",
        profile_normalized="ci",
    )

    assert isinstance(payload, dict)
    assert dummy.kwargs is not None
    # The bare-control run must receive the same edit op used for the guarded run
    assert dummy.kwargs.get("edit") is sentinel


def test_validate_harvest_schedule_split_mismatch():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            preview_n=1,
            final_n=1,
            seq_len=8,
            stride=8,
            provider="wikitext2",
            split="validation",
        )
    )
    pairing = {
        "preview": {"input_ids": [[0, 1]], "window_ids": [0]},
        "final": {"input_ids": [[2, 3]], "window_ids": [1]},
    }
    baseline = {
        "data": {
            "seq_len": 8,
            "stride": 8,
            "dataset": "wikitext2",
            "split": "test",
        }
    }
    with pytest.raises(typer.Exit):
        run_mod._validate_and_harvest_baseline_schedule(
            cfg,
            pairing,
            baseline,
            tokenizer_hash=None,
            resolved_loss_type="causal",
            baseline_path_str="baseline.json",
            console=Console(file=io.StringIO()),
        )


def test_choose_dataset_split_fallback():
    split, used = run_mod._choose_dataset_split(requested=None, available=None)
    assert split == "validation" and used is True
    split2, used2 = run_mod._choose_dataset_split(
        requested=None, available=["train", "dev"]
    )
    assert split2 == "dev" and used2 is True


def test_persist_ref_masks_missing_paths(tmp_path: Path):
    empty = {}
    assert run_mod._persist_ref_masks(empty, tmp_path) is None
    core_report = SimpleNamespace(edit={})
    assert run_mod._persist_ref_masks(core_report, tmp_path) is None


def test_resolve_provider_and_split_with_obj_items(monkeypatch):
    # Provider object exposing kind and items(), plus available_splits on instance
    class ProviderObj:
        kind = "hf_text"

        def items(self):
            # Only non-empty, non-None should propagate into kwargs
            return [("text_field", "text"), ("max_samples", 3), ("empty", "")]

    cfg = SimpleNamespace(dataset=SimpleNamespace(provider=ProviderObj()))

    class FakeProvider:
        def available_splits(self):
            return ["train", "custom"]

    def fake_get_provider(name, **kwargs):  # noqa: ARG001
        # kwargs should include text_field/max_samples from items()
        assert kwargs.get("text_field") == "text"
        assert kwargs.get("max_samples") == 3
        return FakeProvider()

    console = Console(file=io.StringIO())
    prov, split, used_fallback = run_mod._resolve_provider_and_split(
        cfg,
        model_profile=SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=fake_get_provider,
        provider_kwargs=None,
        console=console,
    )
    assert hasattr(prov, "available_splits")
    # No alias present; falls back to first sorted available split
    assert split in {"custom", "train"}
    assert used_fallback is True


def test_resolve_provider_and_split_available_raises(monkeypatch):
    # Provider object with only kind; available_splits raises to exercise except path
    class ProviderObj:
        kind = "synthetic"

    class FakeProvider:
        def available_splits(self):  # noqa: D401
            raise RuntimeError("boom")

    cfg = SimpleNamespace(dataset=SimpleNamespace(provider=ProviderObj()))

    def fake_get_provider(name, **kwargs):  # noqa: ARG001
        return FakeProvider()

    console = Console(file=io.StringIO())
    _, split, used = run_mod._resolve_provider_and_split(
        cfg,
        model_profile=SimpleNamespace(default_provider=None),
        get_provider_fn=fake_get_provider,
        provider_kwargs=None,
        console=console,
    )
    # With no available splits, falls back to validation
    assert split == "validation" and used is True


def test_resolve_provider_and_split_plain_string_provider(monkeypatch):
    # Provider given as plain string; available_splits has no alias → sorted fallback
    cfg = SimpleNamespace(dataset=SimpleNamespace(provider="hf_text"))

    class FakeProvider:
        def available_splits(self):
            return ["zebra", "apple"]

    def fake_get_provider(name, **kwargs):  # noqa: ARG001
        assert name == "hf_text"
        return FakeProvider()

    console = Console(file=io.StringIO())
    _, split, used = run_mod._resolve_provider_and_split(
        cfg,
        model_profile=SimpleNamespace(default_provider=None),
        get_provider_fn=fake_get_provider,
        provider_kwargs=None,
        console=console,
    )
    assert split == "apple" and used is True


def test_resolve_provider_and_split_missing_kind_fallback(monkeypatch):
    # Provider object missing 'kind' attribute triggers model_profile fallback
    class ProviderObj:
        def items(self):  # noqa: D401
            return [("unused", 1)]

    cfg = SimpleNamespace(dataset=SimpleNamespace(provider=ProviderObj()))

    def fake_get_provider(name, **kwargs):  # noqa: ARG001
        # Should pick default from model_profile
        assert name == "wikitext2"

        class P:
            pass

        return P()

    console = Console(file=io.StringIO())
    _, split, used = run_mod._resolve_provider_and_split(
        cfg,
        model_profile=SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=fake_get_provider,
        provider_kwargs=None,
        console=console,
    )
    assert split == "validation" and used is True


def test_prepare_config_for_run_invalid_tier_and_probes(monkeypatch):
    # Patch config helpers used inside the function
    class DummyCfg:
        def __init__(self, data):
            self._data = data

        def model_dump(self):  # noqa: D401
            return dict(self._data)

    monkeypatch.setattr(
        "invarlock.cli.config.load_config",
        lambda path: DummyCfg({"auto": {}}),
        raising=False,
    )
    # Pass-through apply_profile and apply_edit_override
    monkeypatch.setattr(
        "invarlock.cli.config.apply_profile", lambda cfg, p: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.cli.config.apply_edit_override", lambda cfg, e: cfg, raising=False
    )
    # Invalid tier triggers Exit(1)
    with pytest.raises(typer.Exit):
        run_mod._prepare_config_for_run(
            config_path="dummy.yaml",
            profile=None,
            edit=None,
            tier="invalid_tier",
            probes=None,
            console=Console(file=io.StringIO()),
        )
    # Invalid probes triggers Exit(1)
    with pytest.raises(typer.Exit):
        run_mod._prepare_config_for_run(
            config_path="dummy.yaml",
            profile=None,
            edit=None,
            tier=None,
            probes=99,
            console=Console(file=io.StringIO()),
        )
