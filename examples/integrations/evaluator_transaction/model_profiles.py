"""Immutable model and execution profiles for evaluator transactions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    name: str
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    role: str
    repository: str
    revision: str
    model_type: str
    files: tuple[SnapshotFile, ...]
    checkpoint_tree_sha256: str | None = None
    tokenizer_contract_sha256: str | None = None

    @property
    def locator(self) -> str:
        return f"hf://{self.repository}@{self.revision}"

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{filename}"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    key: str
    profile_id: str
    device: str
    dtype: str
    batch_size: int
    torch_num_threads: int
    snapshots: tuple[Snapshot, Snapshot]

    def snapshot(self, role: str) -> Snapshot:
        for value in self.snapshots:
            if value.role == role:
                return value
        raise ValueError(f"unknown evaluator model role: {role}")


_QWEN35_08B_SHARED = (
    SnapshotFile(
        "config.json",
        2_907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    SnapshotFile(
        "merges.txt",
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    SnapshotFile(
        "vocab.json",
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
)

_QUICK = ModelProfile(
    key="quick",
    profile_id="qwen35-0.8b-base-to-post-trained-cpu-v1",
    device="cpu",
    dtype="float32",
    batch_size=8,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="Qwen/Qwen3.5-0.8B-Base",
            revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            model_type="qwen3_5",
            files=(
                *_QWEN35_08B_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    50_900,
                    "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
                ),
                SnapshotFile(
                    "model.safetensors-00001-of-00001.safetensors",
                    1_746_942_600,
                    "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    12_807_196,
                    "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    16_712,
                    "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:d9a7f63f71b0a8825121c1d5fb6531f4e334b0b6b889f3bd223b551fc545d25f"
            ),
            tokenizer_contract_sha256=(
                "7ada77f663f15f6943662b56a8dcea510f475dfd48d31418781b0a5e938066f0"
            ),
        ),
        Snapshot(
            role="subject",
            repository="Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            model_type="qwen3_5",
            files=(
                SnapshotFile(
                    "chat_template.jinja",
                    7_755,
                    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                ),
                *_QWEN35_08B_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    50_900,
                    "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
                ),
                SnapshotFile(
                    "model.safetensors-00001-of-00001.safetensors",
                    1_746_942_600,
                    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    12_807_982,
                    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    16_709,
                    "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:d6866dbe2ec16212b927ca14045a2caefe6bc2a272958506678eefbb809a4b9a"
            ),
            tokenizer_contract_sha256=(
                "d2404e21ad9a6346678434df047fa1a4dc2b37b0a88e2b9aaecdfe38bd6ca284"
            ),
        ),
    ),
)

_QWEN35_SHARED = (
    SnapshotFile(
        "config.json",
        3_126,
        "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
    ),
    SnapshotFile(
        "merges.txt",
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    SnapshotFile(
        "vocab.json",
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
)

_FLAGSHIP = ModelProfile(
    key="flagship",
    profile_id="qwen35-9b-base-to-post-trained-bf16-singleton-v1",
    device="cuda",
    dtype="bfloat16",
    batch_size=1,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="Qwen/Qwen3.5-9B-Base",
            revision="68c46c4b3498877f3ef123c856ecfde50c39f404",
            model_type="qwen3_5",
            files=(
                *_QWEN35_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    79_657,
                    "026b9d9fe03f19fd065f2a2f56a332c67640878106c0ca6be2f60c655ed5a8c1",
                ),
                SnapshotFile(
                    "model.safetensors-00001-of-00004.safetensors",
                    5_276_436_216,
                    "862bf7bba8a50145d19d0ae463931fae515284024736592a73a336bc4dfa54ee",
                ),
                SnapshotFile(
                    "model.safetensors-00002-of-00004.safetensors",
                    5_335_161_576,
                    "bace8e115e11ca93c22f0352a60d2fb0c76ac6d7d1c2993c143b7ad2b6c8868c",
                ),
                SnapshotFile(
                    "model.safetensors-00003-of-00004.safetensors",
                    5_368_717_376,
                    "63a021ac0011cbfc66166e77103327a8b45dee95832e36551f6b4c3337448959",
                ),
                SnapshotFile(
                    "model.safetensors-00004-of-00004.safetensors",
                    3_325_995_704,
                    "1a643bbed669266917b5058b5d3f660c03233599249ff7d8fd083decfe662ae0",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    12_807_196,
                    "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    16_713,
                    "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:20f0af4e87fa4fb226b702f7de1b1f21bf738a687fe9834cc0abda8964861dfe"
            ),
            tokenizer_contract_sha256=(
                "7ff212d57b99bc9eba792a4ab0b32c080164f3d402ce898d00680d9df551b107"
            ),
        ),
        Snapshot(
            role="subject",
            repository="Qwen/Qwen3.5-9B",
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            model_type="qwen3_5",
            files=(
                SnapshotFile(
                    "chat_template.jinja",
                    7_756,
                    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
                ),
                *_QWEN35_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    79_657,
                    "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5",
                ),
                SnapshotFile(
                    "model.safetensors-00001-of-00004.safetensors",
                    5_276_436_216,
                    "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
                ),
                SnapshotFile(
                    "model.safetensors-00002-of-00004.safetensors",
                    5_335_161_512,
                    "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
                ),
                SnapshotFile(
                    "model.safetensors-00003-of-00004.safetensors",
                    5_368_717_440,
                    "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
                ),
                SnapshotFile(
                    "model.safetensors-00004-of-00004.safetensors",
                    3_325_995_712,
                    "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    12_807_982,
                    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    16_710,
                    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:a73abe2d4664cef43cf774e975ad86f614faf57a7e9e63ae660e42e4245bcbf7"
            ),
            tokenizer_contract_sha256=(
                "a4dc0cc2bd8621a72a232a4889a8887b7d05482c7df8e6d42ac4c014cdbdad94"
            ),
        ),
    ),
)

_PORTABILITY = ModelProfile(
    key="portability",
    profile_id="gemma4-12b-it-to-qat-q4-bf16-singleton-v1",
    device="cuda",
    dtype="bfloat16",
    batch_size=1,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="google/gemma-4-12B-it",
            revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            model_type="gemma4_unified",
            files=(
                SnapshotFile(
                    "chat_template.jinja",
                    18_683,
                    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
                ),
                SnapshotFile(
                    "config.json",
                    4_423,
                    "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9",
                ),
                SnapshotFile(
                    "model.safetensors",
                    23_919_549_408,
                    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    32_169_626,
                    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    3_089,
                    "a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:4b242ffea3b93942d347ff7c9c1982a0ec94b8a86e11ad94ccc41f0923da41dc"
            ),
            tokenizer_contract_sha256=(
                "3ee6db2a73bdd7e427cab96d315a5cfd3adde1e17143159481bef0317246fe21"
            ),
        ),
        Snapshot(
            role="subject",
            repository="google/gemma-4-12B-it-qat-q4_0-unquantized",
            revision="b6ed86275a6a5735884e208bfed95b445a684ca2",
            model_type="gemma4_unified",
            files=(
                SnapshotFile(
                    "chat_template.jinja",
                    18_683,
                    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
                ),
                SnapshotFile(
                    "config.json",
                    4_310,
                    "a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077",
                ),
                SnapshotFile(
                    "model.safetensors",
                    23_919_549_408,
                    "26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    32_169_626,
                    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    3_089,
                    "a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:107c7a1581a1215a5443429340a4d7618649e5a95d1cee9d4a93356885f35cd9"
            ),
            tokenizer_contract_sha256=(
                "3ee6db2a73bdd7e427cab96d315a5cfd3adde1e17143159481bef0317246fe21"
            ),
        ),
    ),
)


def model_profile(key: str) -> ModelProfile:
    try:
        return {
            "quick": _QUICK,
            "flagship": _FLAGSHIP,
            "portability": _PORTABILITY,
        }[key]
    except KeyError as exc:
        raise ValueError(f"unknown evaluator model profile: {key}") from exc


__all__ = ["ModelProfile", "Snapshot", "SnapshotFile", "model_profile"]
