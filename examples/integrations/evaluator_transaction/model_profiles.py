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


_QWEN3_SHARED = (
    SnapshotFile(
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    SnapshotFile(
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)

_QUICK = ModelProfile(
    key="quick",
    profile_id="qwen3-0.6b-base-to-post-trained-cpu-v1",
    device="cpu",
    dtype="float32",
    batch_size=8,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="Qwen/Qwen3-0.6B-Base",
            revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
            model_type="qwen3",
            files=(
                SnapshotFile(
                    "config.json",
                    727,
                    "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
                ),
                *_QWEN3_SHARED,
                SnapshotFile(
                    "model.safetensors",
                    1_192_135_096,
                    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    7_031_645,
                    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    9_678,
                    "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:eddb974cecb32ecf6bfaec2a19ecfbb32c73be9f7c38c7b54d551cd8ef66bd75"
            ),
            tokenizer_contract_sha256=(
                "c5f0898f912c7d953302779f61c86026b3cea05561a9520b6209e82b9d650581"
            ),
        ),
        Snapshot(
            role="subject",
            repository="Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            model_type="qwen3",
            files=(
                SnapshotFile(
                    "config.json",
                    726,
                    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
                ),
                *_QWEN3_SHARED,
                SnapshotFile(
                    "model.safetensors",
                    1_503_300_328,
                    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    11_422_654,
                    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    9_732,
                    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                ),
            ),
            checkpoint_tree_sha256=(
                "sha256:f97b7ac0717847938aed654bf671a93a28cf13413e37d29040ebad85564f6346"
            ),
            tokenizer_contract_sha256=(
                "ddf5fc73d604adf713f3d2fa98a9229c9dc05abb0881b33e636d15a5616dcd02"
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
    profile_id="gemma4-12b-base-to-post-trained-bf16-singleton-v1",
    device="cuda",
    dtype="bfloat16",
    batch_size=1,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="google/gemma-4-12B",
            revision="023679ed352de9bb66cc873c9009ce3482585c08",
            model_type="gemma4_unified",
            files=(
                SnapshotFile(
                    "config.json",
                    4_383,
                    "14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098",
                ),
                SnapshotFile(
                    "model.safetensors",
                    23_919_549_408,
                    "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a",
                ),
                SnapshotFile(
                    "tokenizer.json",
                    32_170_070,
                    "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
                ),
                SnapshotFile(
                    "tokenizer_config.json",
                    888,
                    "522a38334973725dba8f7c645195b19dda0c284f403f43273f77837679ba2eab",
                ),
            ),
        ),
        Snapshot(
            role="subject",
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
