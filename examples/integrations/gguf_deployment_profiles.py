"""Closed model-family profiles for the BF16-to-GGUF deployment journey."""

from __future__ import annotations

from dataclasses import dataclass

from examples.integrations.evaluator_transaction.corpora import (
    CorpusProfile,
    corpus_profile,
    independent_canary_corpus_profile,
    independent_canary_records,
    qualification_records,
)
from examples.integrations.evaluator_transaction.model_profiles import (
    Snapshot,
    SnapshotFile,
    model_profile,
)

DEFAULT_DEPLOYMENT_PROFILE = "qwen35-9b"


@dataclass(frozen=True, slots=True)
class DeploymentProfile:
    """Immutable source, corpus, transformation, and runtime selection."""

    key: str
    source: Snapshot
    corpus: CorpusProfile
    intermediate_name: str
    subject_name: str
    observation_id: str
    quantization: str = "Q5_K_M"
    baseline_device: str = "cuda"
    subject_device: str = "cpu"


_MINISTRAL3_8B = Snapshot(
    role="source",
    repository="mistralai/Ministral-3-8B-Instruct-2512-BF16",
    revision="f6fae9795746f63c9be8344932f01275f3c63734",
    model_type="mistral3",
    files=(
        SnapshotFile(
            "chat_template.jinja",
            11_912,
            "74eeb55fd3341286ec3fd44e902b7120721acc81cd394e96b431f85e93a1ea56",
        ),
        SnapshotFile(
            "config.json",
            1_579,
            "3953b565385881fe1e80980fa8f797ffef5dbcc170851b5f90d60ddcb1905f58",
        ),
        SnapshotFile(
            "model.safetensors.index.json",
            52_675,
            "639a9c27a864b0be21e9e0fd1313f120d24c45563723797ebf611c6fce75b822",
        ),
        SnapshotFile(
            "model-00001-of-00004.safetensors",
            4_984_292_952,
            "95f4da19c81e6a06d4f0c61cac3dfdd85ef463a7955473dc4c07e570fd2342f9",
        ),
        SnapshotFile(
            "model-00002-of-00004.safetensors",
            4_999_804_256,
            "18275e4c8413d0ed4b0cb8380fe1fd4faeee34189960fdd92ef5b0d3f4ed1b98",
        ),
        SnapshotFile(
            "model-00003-of-00004.safetensors",
            4_915_917_680,
            "980d1d3635ae29a6e629e2f7ad589c882eba270c3f66da74da4f8ccb11845687",
        ),
        SnapshotFile(
            "model-00004-of-00004.safetensors",
            2_936_108_304,
            "fa80e52af465c5644b18b623eeeebb1d50e47dafbee73816bf26f2473f88d369",
        ),
        SnapshotFile(
            "params.json",
            1_098,
            "81b8377f36c5b3d60900b333214d85160a0500b84576e969092c6fd214f69538",
        ),
        SnapshotFile(
            "special_tokens_map.json",
            147_094,
            "0a5c981e8c5c6f8886ee007a6d4543a0be6b221cb9ca32a8709384a4c6fc8cbb",
        ),
        SnapshotFile(
            "tekken.json",
            16_753_784,
            "600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843",
        ),
        SnapshotFile(
            "tokenizer.json",
            17_078_128,
            "d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135",
        ),
        SnapshotFile(
            "tokenizer_config.json",
            198_094,
            "f59f7294e4f26383d0ea93840fe21cf197784be0842a8301a0343e8c34ed0d6d",
        ),
    ),
    checkpoint_tree_sha256=(
        "sha256:6cbddcebc289550569cc3f6a93676a8f4f605d8574b8aec8448d61594a283996"
    ),
    tokenizer_contract_sha256=(
        "b9e3906504b6235b5c289fe9d3f7a86512f968dfe300771ec660080924615dbc"
    ),
)


def _qwen35_profile() -> DeploymentProfile:
    source = model_profile("flagship").snapshot("subject")
    return DeploymentProfile(
        key="qwen35-9b",
        source=source,
        corpus=corpus_profile("flagship"),
        intermediate_name="Qwen3.5-9B-BF16.gguf",
        subject_name="Qwen3.5-9B-Q5_K_M.gguf",
        observation_id="qwen35-9b-bf16-to-gguf-q5-k-m",
    )


def _ministral3_profile() -> DeploymentProfile:
    return DeploymentProfile(
        key="ministral3-8b",
        source=_MINISTRAL3_8B,
        corpus=independent_canary_corpus_profile(),
        intermediate_name="Ministral-3-8B-Instruct-BF16.gguf",
        subject_name="Ministral-3-8B-Instruct-Q5_K_M.gguf",
        observation_id="ministral3-8b-bf16-to-gguf-q5-k-m",
    )


def deployment_profile_keys() -> tuple[str, ...]:
    return (DEFAULT_DEPLOYMENT_PROFILE, "ministral3-8b")


def deployment_profile(key: str = DEFAULT_DEPLOYMENT_PROFILE) -> DeploymentProfile:
    """Return one closed deployment profile after checking shared invariants."""

    try:
        profile = {
            DEFAULT_DEPLOYMENT_PROFILE: _qwen35_profile,
            "ministral3-8b": _ministral3_profile,
        }[key]()
    except KeyError as exc:
        raise ValueError(f"unknown GGUF deployment profile: {key}") from exc
    if (
        profile.key != key
        or profile.corpus.record_count != 400
        or profile.quantization != "Q5_K_M"
        or profile.baseline_device != "cuda"
        or profile.subject_device != "cpu"
        or profile.source.checkpoint_tree_sha256 is None
        or profile.source.tokenizer_contract_sha256 is None
        or not profile.intermediate_name.endswith("-BF16.gguf")
        or not profile.subject_name.endswith("-Q5_K_M.gguf")
        or "/" in profile.intermediate_name
        or "/" in profile.subject_name
    ):
        raise RuntimeError("GGUF deployment profile invariants are invalid")
    return profile


def deployment_records(profile: DeploymentProfile) -> list[dict[str, str]]:
    """Return the exact rendered records declared by a deployment profile."""

    if profile.key == DEFAULT_DEPLOYMENT_PROFILE:
        return qualification_records(profile.corpus)
    if profile.key == "ministral3-8b":
        return independent_canary_records()
    raise ValueError(f"unknown GGUF deployment record profile: {profile.key}")


__all__ = [
    "DEFAULT_DEPLOYMENT_PROFILE",
    "DeploymentProfile",
    "deployment_profile",
    "deployment_profile_keys",
    "deployment_records",
]
