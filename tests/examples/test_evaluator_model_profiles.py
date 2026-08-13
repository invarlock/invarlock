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

    assert profile.profile_id == "qwen35-9b-base-to-post-trained-bf16-v1"
    assert profile.device == "cuda"
    assert profile.dtype == "bfloat16"
    assert profile.batch_size == 4
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


def test_model_profile_lookup_rejects_unmaintained_profiles() -> None:
    try:
        model_profiles.model_profile("large")
    except ValueError as exc:
        assert str(exc) == "unknown evaluator model profile: large"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("an unknown profile was accepted")
