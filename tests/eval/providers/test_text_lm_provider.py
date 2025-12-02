from __future__ import annotations

import copy

from invarlock.cli.commands.run import (
    _compute_mask_positions_digest,
    _compute_provider_digest,
)


def _mk_windows(
    labels_preview, labels_final, window_ids_preview=None, window_ids_final=None
):
    return {
        "preview": {
            "labels": labels_preview,
            "window_ids": window_ids_preview or list(range(len(labels_preview))),
        },
        "final": {
            "labels": labels_final,
            "window_ids": window_ids_final or list(range(len(labels_final))),
        },
    }


def test_mask_positions_digest_stable_and_changes_with_positions():
    # labels: -100 means not masked; positive values indicate masked positions
    win1 = _mk_windows(
        labels_preview=[[-100, 5, -100], [7, -100]],
        labels_final=[[-100, -100, 11]],
    )
    d1 = _compute_mask_positions_digest(win1)
    d1_repeat = _compute_mask_positions_digest(copy.deepcopy(win1))
    assert isinstance(d1, str) and d1
    assert d1 == d1_repeat

    # Flip a single mask position → digest must change
    win2 = _mk_windows(
        labels_preview=[[-100, 5, -100], [-100, -100]],  # changed second preview sample
        labels_final=[[-100, -100, 11]],
    )
    d2 = _compute_mask_positions_digest(win2)
    assert d1 != d2


def test_provider_digest_includes_ids_and_tokenizer_and_masking():
    report = {
        "meta": {"tokenizer_hash": "abc123"},
        "evaluation_windows": _mk_windows(
            labels_preview=[[-100, 5, -100]],
            labels_final=[[-100, -100, 11]],
            window_ids_preview=[10],
            window_ids_final=[20],
        ),
    }
    digest = _compute_provider_digest(report)
    assert isinstance(digest, dict)
    assert digest.get("tokenizer_sha256") == "abc123"
    assert isinstance(digest.get("ids_sha256"), str) and len(digest["ids_sha256"]) >= 16
    assert (
        isinstance(digest.get("masking_sha256"), str)
        and len(digest["masking_sha256"]) >= 16
    )
