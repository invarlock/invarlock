from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_json import sha256_prefixed
from scripts.evidence_packs.python.editing.training_profile_snapshot import (
    TRAINING_PROFILE_SNAPSHOT_SCHEMA,
    TrainingProfileSnapshotError,
    produce_training_profile_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = REPO_ROOT / "scripts/evidence_packs/training_profiles.json"


def test_profile_snapshot_is_deterministic_and_requires_explicit_scope(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tiny_gpt2_lora_v1.json"

    result = produce_training_profile_snapshot(
        profile_id="tiny_gpt2_lora_v1",
        scope="attn",
        output_path=output,
        profiles_path=PROFILES_PATH,
        repo_root=REPO_ROOT,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == TRAINING_PROFILE_SNAPSHOT_SCHEMA
    assert payload["profile_id"] == "tiny_gpt2_lora_v1"
    assert payload["scope"] == "attn"
    assert result["snapshot_sha256"] == sha256_prefixed(output.read_bytes())

    first_bytes = output.read_bytes()
    assert (
        produce_training_profile_snapshot(
            profile_id="tiny_gpt2_lora_v1",
            scope="attn",
            output_path=output,
            profiles_path=PROFILES_PATH,
            repo_root=REPO_ROOT,
        )
        == result
    )
    assert output.read_bytes() == first_bytes

    with pytest.raises(TrainingProfileSnapshotError, match="scope"):
        produce_training_profile_snapshot(
            profile_id="tiny_gpt2_lora_v1",
            scope="derived-from-module-name",
            output_path=tmp_path / "invalid.json",
            profiles_path=PROFILES_PATH,
            repo_root=REPO_ROOT,
        )


def test_profile_snapshot_refuses_to_replace_different_content(tmp_path: Path) -> None:
    output = tmp_path / "tiny_gpt2_full_ft_v1.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(TrainingProfileSnapshotError, match="refusing to overwrite"):
        produce_training_profile_snapshot(
            profile_id="tiny_gpt2_full_ft_v1",
            scope="all",
            output_path=output,
            profiles_path=PROFILES_PATH,
            repo_root=REPO_ROOT,
        )
