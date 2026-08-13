from __future__ import annotations

from examples.integrations.evaluator_transaction import model_profiles


def test_quick_profile_remains_the_small_cpu_ci_path() -> None:
    profile = model_profiles.model_profile("quick")

    assert profile.device == "cpu"
    assert profile.dtype == "float32"
    assert profile.batch_size == 8
    assert [snapshot.repository for snapshot in profile.snapshots] == [
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B",
    ]


def test_flagship_profile_is_the_immutable_qwen35_9b_cuda_comparison() -> None:
    profile = model_profiles.model_profile("flagship")

    assert (
        profile.profile_id
        == "qwen35-9b-base-to-post-trained-bf16-singleton-v1"
    )
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


def test_portability_profile_is_the_recent_gemma4_12b_cuda_comparison() -> None:
    profile = model_profiles.model_profile("portability")

    assert profile.profile_id == (
        "gemma4-12b-base-to-post-trained-bf16-singleton-v1"
    )
    assert profile.device == "cuda"
    assert profile.dtype == "bfloat16"
    assert profile.batch_size == 1
    assert [snapshot.repository for snapshot in profile.snapshots] == [
        "google/gemma-4-12B",
        "google/gemma-4-12B-it",
    ]
    assert all(snapshot.model_type == "gemma4_unified" for snapshot in profile.snapshots)
    assert [snapshot.revision for snapshot in profile.snapshots] == [
        "023679ed352de9bb66cc873c9009ce3482585c08",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    ]
    assert all(len(snapshot.files) >= 4 for snapshot in profile.snapshots)
    assert all(
        sum(item.byte_length for item in snapshot.files) > 23_900_000_000
        for snapshot in profile.snapshots
    )
