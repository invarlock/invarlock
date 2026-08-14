from __future__ import annotations

import pytest

from examples.integrations.evaluator_transaction import model_profiles


def test_quick_profile_uses_the_immutable_qwen35_08b_pair() -> None:
    profile = model_profiles.model_profile("quick")

    assert profile.profile_id == "qwen35-0.8b-base-to-post-trained-cpu-v1"
    assert profile.device == "cpu"
    assert profile.dtype == "float32"
    assert profile.batch_size == 8
    assert [
        (snapshot.role, snapshot.repository, snapshot.revision, snapshot.model_type)
        for snapshot in profile.snapshots
    ] == [
        (
            "baseline",
            "Qwen/Qwen3.5-0.8B-Base",
            "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "qwen3_5",
        ),
        (
            "subject",
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            "qwen3_5",
        ),
    ]
    assert {
        snapshot.role: (
            snapshot.checkpoint_tree_sha256,
            snapshot.tokenizer_contract_sha256,
        )
        for snapshot in profile.snapshots
    } == {
        "baseline": (
            "sha256:d9a7f63f71b0a8825121c1d5fb6531f4e334b0b6b889f3bd223b551fc545d25f",
            "7ada77f663f15f6943662b56a8dcea510f475dfd48d31418781b0a5e938066f0",
        ),
        "subject": (
            "sha256:d6866dbe2ec16212b927ca14045a2caefe6bc2a272958506678eefbb809a4b9a",
            "d2404e21ad9a6346678434df047fa1a4dc2b37b0a88e2b9aaecdfe38bd6ca284",
        ),
    }
    for snapshot in profile.snapshots:
        assert all(
            item.byte_length > 0 and len(item.sha256) == 64
            for item in snapshot.files
        )
        assert {
            "config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "model.safetensors-00001-of-00001.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        } <= {item.name for item in snapshot.files}


def test_flagship_profile_is_the_immutable_qwen35_9b_cuda_comparison() -> None:
    profile = model_profiles.model_profile("flagship")

    assert profile.profile_id == "qwen35-9b-base-to-post-trained-bf16-singleton-v1"
    assert profile.device == "cuda"
    assert profile.dtype == "bfloat16"
    # Singleton inference prevents evaluator-specific batch composition from
    # changing borderline BF16 token decisions.
    assert profile.batch_size == 1
    assert [
        (snapshot.role, snapshot.repository, snapshot.revision)
        for snapshot in profile.snapshots
    ] == [
        (
            "baseline",
            "Qwen/Qwen3.5-9B-Base",
            "68c46c4b3498877f3ef123c856ecfde50c39f404",
        ),
        (
            "subject",
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        ),
    ]
    expected_identities = {
        "baseline": (
            "sha256:20f0af4e87fa4fb226b702f7de1b1f21bf738a687fe9834cc0abda8964861dfe",
            "7ff212d57b99bc9eba792a4ab0b32c080164f3d402ce898d00680d9df551b107",
        ),
        "subject": (
            "sha256:a73abe2d4664cef43cf774e975ad86f614faf57a7e9e63ae660e42e4245bcbf7",
            "a4dc0cc2bd8621a72a232a4889a8887b7d05482c7df8e6d42ac4c014cdbdad94",
        ),
    }
    for snapshot in profile.snapshots:
        names = {item.name for item in snapshot.files}
        assert {
            "config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            *(
                f"model.safetensors-{index:05d}-of-00004.safetensors"
                for index in range(1, 5)
            ),
        } <= names
        assert all(
            item.byte_length > 0 and len(item.sha256) == 64 for item in snapshot.files
        )
        assert (
            snapshot.checkpoint_tree_sha256,
            snapshot.tokenizer_contract_sha256,
        ) == expected_identities[snapshot.role]


def test_model_profile_lookup_rejects_unmaintained_profiles() -> None:
    try:
        model_profiles.model_profile("large")
    except ValueError as exc:
        assert str(exc) == "unknown evaluator model profile: large"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("an unknown profile was accepted")


def test_model_profile_rejects_an_unknown_snapshot_role() -> None:
    profile = model_profiles.model_profile("quick")

    with pytest.raises(ValueError, match="unknown evaluator model role: candidate"):
        profile.snapshot("candidate")


def test_portability_profile_is_the_recent_gemma4_12b_qat_comparison() -> None:
    profile = model_profiles.model_profile("portability")

    assert profile.profile_id == ("gemma4-12b-it-to-qat-q4-bf16-singleton-v1")
    assert profile.device == "cuda"
    assert profile.dtype == "bfloat16"
    assert profile.batch_size == 1
    assert [snapshot.repository for snapshot in profile.snapshots] == [
        "google/gemma-4-12B-it",
        "google/gemma-4-12B-it-qat-q4_0-unquantized",
    ]
    assert all(
        snapshot.model_type == "gemma4_unified" for snapshot in profile.snapshots
    )
    assert [snapshot.revision for snapshot in profile.snapshots] == [
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "b6ed86275a6a5735884e208bfed95b445a684ca2",
    ]
    assert all(len(snapshot.files) >= 4 for snapshot in profile.snapshots)
    assert all(
        sum(item.byte_length for item in snapshot.files) > 23_900_000_000
        for snapshot in profile.snapshots
    )
    expected_identities = {
        "baseline": (
            "sha256:4b242ffea3b93942d347ff7c9c1982a0ec94b8a86e11ad94ccc41f0923da41dc",
            "3ee6db2a73bdd7e427cab96d315a5cfd3adde1e17143159481bef0317246fe21",
        ),
        "subject": (
            "sha256:107c7a1581a1215a5443429340a4d7618649e5a95d1cee9d4a93356885f35cd9",
            "3ee6db2a73bdd7e427cab96d315a5cfd3adde1e17143159481bef0317246fe21",
        ),
    }
    assert {
        snapshot.role: (
            snapshot.checkpoint_tree_sha256,
            snapshot.tokenizer_contract_sha256,
        )
        for snapshot in profile.snapshots
    } == expected_identities
