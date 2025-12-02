from __future__ import annotations

import random

from invarlock.cli.commands.run import _hash_sequences


def _gen_token_sequences(
    seed: int, n: int, min_len: int = 3, max_len: int = 8
) -> list[list[int]]:
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    for _ in range(n):
        L = rng.randint(min_len, max_len)
        seqs.append([rng.randint(1, 20) for __ in range(L)])
    return seqs


def test_schedule_digest_is_deterministic_for_same_seed():
    a = _gen_token_sequences(seed=123, n=5)
    b = _gen_token_sequences(seed=123, n=5)
    assert a == b
    ha = _hash_sequences(a)
    hb = _hash_sequences(b)
    assert ha == hb


# Removed legacy pairability helper coverage; pairing is enforced via guard digests during runs.
