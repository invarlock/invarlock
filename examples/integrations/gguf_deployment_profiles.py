"""Closed model-family profiles for the BF16-to-GGUF deployment journey."""

from __future__ import annotations

from dataclasses import dataclass

from examples.integrations.evaluator_transaction.corpora import (
    CorpusProfile,
    corpus_profile,
    independent_canary_corpus_profile,
    independent_canary_records,
    qualification_records,
    qwen38_27b_corpus_profile,
    qwen38_27b_records,
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

_QWEN38_27B = Snapshot(
    role="source",
    repository="Qwen/Qwen3.8-27B",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_type="qwen3_5",
    files=(
        SnapshotFile(
            "chat_template.jinja",
            8_952,
            "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        ),
        SnapshotFile(
            "config.json",
            4_312,
            "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
        ),
        SnapshotFile(
            "merges.txt",
            3_353_259,
            "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        ),
        SnapshotFile(
            "model-00001-of-00018.safetensors",
            3_966_730_552,
            "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
        ),
        SnapshotFile(
            "model-00002-of-00018.safetensors",
            3_043_080_328,
            "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
        ),
        SnapshotFile(
            "model-00003-of-00018.safetensors",
            2_542_796_952,
            "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
        ),
        SnapshotFile(
            "model-00004-of-00018.safetensors",
            3_988_973_152,
            "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
        ),
        SnapshotFile(
            "model-00005-of-00018.safetensors",
            2_099_339_864,
            "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
        ),
        SnapshotFile(
            "model-00006-of-00018.safetensors",
            3_979_553_696,
            "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
        ),
        SnapshotFile(
            "model-00007-of-00018.safetensors",
            2_108_759_344,
            "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
        ),
        SnapshotFile(
            "model-00008-of-00018.safetensors",
            3_979_553_696,
            "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
        ),
        SnapshotFile(
            "model-00009-of-00018.safetensors",
            2_108_759_344,
            "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
        ),
        SnapshotFile(
            "model-00010-of-00018.safetensors",
            3_979_553_696,
            "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
        ),
        SnapshotFile(
            "model-00011-of-00018.safetensors",
            2_108_759_344,
            "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
        ),
        SnapshotFile(
            "model-00012-of-00018.safetensors",
            3_979_553_696,
            "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
        ),
        SnapshotFile(
            "model-00013-of-00018.safetensors",
            2_108_759_344,
            "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
        ),
        SnapshotFile(
            "model-00014-of-00018.safetensors",
            3_979_553_696,
            "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
        ),
        SnapshotFile(
            "model-00015-of-00018.safetensors",
            2_108_759_344,
            "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
        ),
        SnapshotFile(
            "model-00016-of-00018.safetensors",
            3_979_564_040,
            "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
        ),
        SnapshotFile(
            "model-00017-of-00018.safetensors",
            2_108_759_344,
            "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
        ),
        SnapshotFile(
            "model-00018-of-00018.safetensors",
            3_392_197_344,
            "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
        ),
        SnapshotFile(
            "model.safetensors.index.json",
            112_216,
            "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
        ),
        SnapshotFile(
            "tokenizer.json",
            12_809_320,
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        ),
        SnapshotFile(
            "tokenizer_config.json",
            17_928,
            "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
        ),
        SnapshotFile(
            "vocab.json",
            6_722_759,
            "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        ),
    ),
    checkpoint_tree_sha256=(
        "sha256:2556be511aff126ec5cb1c0d1be9776cc958bc312e859705cb7e2f6d1eb97e7a"
    ),
    tokenizer_contract_sha256=(
        "3b6e697fdb642963dd8dd07ffe7e5e60b4e3710b252d0a5511488adb8cc8e0ea"
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


def _qwen38_profile() -> DeploymentProfile:
    return DeploymentProfile(
        key="qwen38-27b",
        source=_QWEN38_27B,
        corpus=qwen38_27b_corpus_profile(),
        intermediate_name="Qwen3.8-27B-BF16.gguf",
        subject_name="Qwen3.8-27B-Q5_K_M.gguf",
        observation_id="qwen38-27b-bf16-to-gguf-q5-k-m",
    )


def deployment_profile_keys() -> tuple[str, ...]:
    return (DEFAULT_DEPLOYMENT_PROFILE, "qwen38-27b", "ministral3-8b")


def deployment_profile(key: str = DEFAULT_DEPLOYMENT_PROFILE) -> DeploymentProfile:
    """Return one closed deployment profile after checking shared invariants."""

    try:
        profile = {
            DEFAULT_DEPLOYMENT_PROFILE: _qwen35_profile,
            "qwen38-27b": _qwen38_profile,
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
    if profile.key == "qwen38-27b":
        return qwen38_27b_records()
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
