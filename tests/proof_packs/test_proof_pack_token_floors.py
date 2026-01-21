from __future__ import annotations

from pathlib import Path


def test_proof_pack_default_cert_min_windows_is_high_enough() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "scripts/proof_packs/lib/task_functions.sh").read_text(
        encoding="utf-8"
    )

    # Proof-pack runs rely on balanced-tier token floors (min_tokens=50k). WT-2
    # windows can be short (padding), so the CI defaults should scale up for
    # short sequence lengths unless the caller overrides INVARLOCK_CERT_MIN_WINDOWS.
    assert "_default_ci_min_windows" in text
    assert "default_windows=256" in text
    assert "default_windows=352" in text
    assert "INVARLOCK_CERT_MIN_WINDOWS" in text


def test_proof_pack_7b_uses_short_seq_len_for_throughput() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "scripts/proof_packs/lib/validation_suite.sh").read_text(
        encoding="utf-8"
    )

    # The suite's 7B config should avoid long padded sequences on short-text
    # datasets (WT-2), which wastes compute and slows tuning/runs.
    assert 'echo "512:512:' in text
