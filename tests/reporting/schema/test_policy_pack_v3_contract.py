from __future__ import annotations

import copy

import jsonschema
import pytest

from invarlock.policy_pack import (
    BEHAVIORAL_POLICY_PACK_FORMAT,
    build_behavioral_policy_pack,
    verify_policy_pack,
)
from invarlock.public_contracts import load_policy_pack_schema


def _dataset_identity() -> dict[str, object]:
    return {
        "provider": "local_jsonl",
        "dataset_name": "partner-regression-v1",
        "config_name": None,
        "revision": "a" * 40,
        "split": "validation",
    }


def _binding(
    provider_name: str,
    artifact_format: str,
    *,
    marker: str,
) -> dict[str, object]:
    return {
        "provider_name": provider_name,
        "artifact_format": artifact_format,
        "artifact_identity_sha256": marker * 64,
        "outer_image_digest": "sha256:" + marker * 64,
        "execution_settings_sha256": "e" * 64,
    }


def _pack() -> dict[str, object]:
    return build_behavioral_policy_pack(
        tier="balanced",
        schedule_sha256="f" * 64,
        baseline=_binding("hf_transformers", "hf_snapshot", marker="a"),
        subject=_binding("llama_cpp", "gguf", marker="b"),
        metric_kind="exact_match",
        minimum_subject_score=0.7,
        maximum_regression=0.05,
        dataset_identity=_dataset_identity(),
    )


def test_behavioral_policy_pack_round_trip_and_schema() -> None:
    pack = _pack()

    assert pack["format"] == BEHAVIORAL_POLICY_PACK_FORMAT
    assert pack["resolved_policy"] == {}
    assert "guard_authority" not in pack["resolved_policy"]
    assert verify_policy_pack(pack) == []
    jsonschema.validate(pack, load_policy_pack_schema())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda pack: pack["behavioral_claim"].update(  # type: ignore[union-attr]
                {"claim_set": "invarlock-weight-edit-regression-v2"}
            ),
            "claim_set",
        ),
        (
            lambda pack: pack["behavioral_claim"].update(  # type: ignore[union-attr]
                {"baseline": _binding("llama-cpp", "hf_snapshot", marker="a")}
            ),
            "provider_name must be canonical",
        ),
        (
            lambda pack: pack["behavioral_claim"].update(  # type: ignore[union-attr]
                {"schedule_sha256": "A" * 64}
            ),
            "schedule_sha256 must be a lowercase sha256 digest",
        ),
        (
            lambda pack: pack["behavioral_claim"]["subject"].update(  # type: ignore[index,union-attr]
                {"outer_image_digest": "B" * 64}
            ),
            "outer_image_digest must be a sha256 image digest",
        ),
        (
            lambda pack: pack["behavioral_claim"]["required_capabilities"].update(  # type: ignore[index,union-attr]
                {"evidence_surfaces": ["behavior", "build"]}
            ),
            "behavior and tokenizer",
        ),
        (
            lambda pack: pack["behavioral_claim"]["metric_policy"].update(  # type: ignore[index,union-attr]
                {"maximum_regression": 1.1}
            ),
            "finite number in [0, 1]",
        ),
        (
            lambda pack: pack["behavioral_claim"]["metric_policy"].update(  # type: ignore[index,union-attr]
                {"kind": "multiple_choice_accuracy"}
            ),
            "must be exact_match",
        ),
        (
            lambda pack: pack["compatibility"].pop("dataset_identity"),  # type: ignore[union-attr]
            "dataset_identity is required",
        ),
    ],
)
def test_behavioral_policy_pack_rejects_unauthorized_or_incomplete_claim(
    mutate, message: str
) -> None:
    pack = _pack()
    mutate(pack)

    assert any(message in error for error in verify_policy_pack(pack))


def test_behavioral_policy_digest_binds_every_authorization_field() -> None:
    pack = _pack()
    changed = copy.deepcopy(pack)
    changed["behavioral_claim"]["metric_policy"]["minimum_subject_score"] = 0.8

    errors = verify_policy_pack(changed)

    assert any("policy digest mismatch" in error for error in errors)


def test_v1_v2_cannot_smuggle_behavioral_authority() -> None:
    pack = _pack()
    pack["format"] = "policy-pack-v2"
    pack["resolved_policy"] = {
        "guard_authority": {
            "spectral": "observe",
            "rmt": "observe",
            "variance": "enforce",
        }
    }

    errors = verify_policy_pack(pack)

    assert any("only for policy-pack-v3" in error for error in errors)


def test_behavioral_builder_rejects_malformed_directed_binding_and_metric() -> None:
    with pytest.raises(ValueError, match="provider_name must be canonical"):
        build_behavioral_policy_pack(
            tier="balanced",
            schedule_sha256="f" * 64,
            baseline=_binding("llama-cpp", "hf_snapshot", marker="a"),
            subject=_binding("llama_cpp", "gguf", marker="b"),
            metric_kind="exact_match",
            minimum_subject_score=0.7,
            maximum_regression=0.05,
            dataset_identity=_dataset_identity(),
        )

    with pytest.raises(ValueError, match="must be exact_match"):
        build_behavioral_policy_pack(
            tier="balanced",
            schedule_sha256="f" * 64,
            baseline=_binding("hf_transformers", "hf_snapshot", marker="a"),
            subject=_binding("llama_cpp", "gguf", marker="b"),
            metric_kind="multiple_choice_accuracy",
            minimum_subject_score=0.7,
            maximum_regression=0.05,
            dataset_identity=_dataset_identity(),
        )
