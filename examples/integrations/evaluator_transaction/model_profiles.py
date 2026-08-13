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

_GRANITE_SHARED = (
    SnapshotFile(
        "merges.txt",
        916_646,
        "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
    ),
    SnapshotFile(
        "special_tokens_map.json",
        579,
        "c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e",
    ),
    SnapshotFile(
        "tokenizer.json",
        7_153_421,
        "e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253",
    ),
    SnapshotFile(
        "tokenizer_config.json",
        17_659,
        "a5ec5daab12ba090a90f3dd169c8f9c275557013a87b9c1258dc7cb497a35c86",
    ),
    SnapshotFile(
        "vocab.json",
        1_612_704,
        "8af71076de8b0b626eed0f4c984faf0a7c062479164b2a31308a948524d4f69c",
    ),
)


def _granite_weights(digests: tuple[str, ...]) -> tuple[SnapshotFile, ...]:
    sizes = (
        4_931_823_752,
        4_805_655_352,
        4_684_967_104,
        4_805_655_392,
        4_805_655_400,
        4_684_967_136,
        4_805_655_400,
        4_805_655_400,
        4_684_967_136,
        4_805_655_400,
        4_805_655_400,
        4_805_655_400,
        4_684_967_144,
        2_297_808_752,
    )
    if len(digests) != len(sizes):
        raise ValueError("Granite snapshot shard metadata is incomplete")
    return tuple(
        SnapshotFile(
            f"model-{index:05d}-of-00014.safetensors", size, digest
        )
        for index, (size, digest) in enumerate(zip(sizes, digests, strict=True), 1)
    )


_GRANITE_BASE_WEIGHTS = _granite_weights(
    (
        "22cdb9955764879404d1cbbd744520ec53c7986bf9c3e760533796ed8131ace4",
        "e11b406e444df0be6c34eeb2fd4352478a8c512b7e63934b31621d02f611123c",
        "1c159dabf6acc53763935a03435fd13d6e34fdcd8d5aacdfc87f6f45c0cb94f4",
        "f56db2d9910619adece710cce488ab1809aac152274d1b9c469246bc4c013ad9",
        "043bf16b004cb7a837aeea7805736940f6778f7f9eada8ceb0209086c022a1c4",
        "9739029ca98cbab6038feb8dcc0b05883c0ba994aaba1d24c4bade08f4b8bece",
        "6923892e50ded8070735cec45508a55640039f3e3f01ece3d6f322ce8db55f20",
        "15e1a862fb9693f6d00ca07f38192393dd2cb1a413bfdb3de5e1b57efc2dafb3",
        "7c04b55d66236d1e5995bb16e8df03fdf4fd6f6302ec693136f601a253417744",
        "f6cd5063fbb77f96a44805f5c78fc5b8cfd80b82d1e9f522d3db4c7703e4b560",
        "ca406c8f3da58f26b7e243009f1bd1fa85b3c3432890712940335d9c560bd64b",
        "9d4ef298f04e8b44a2a2467a9d0646bb4dab2605c5f2662e98ef1fa62aaaf6e8",
        "d508219e378cdec3ecc92d2cc9c83a2a7688cd4b5b0305e6d38c8add9a93a4ec",
        "d299279ef4a04d12c39e223cb2a138df0c80cb433fb250912359baf70a89d0a8",
    )
)

_GRANITE_SUBJECT_WEIGHTS = _granite_weights(
    (
        "f45359dac7e0fbffde4f4261d9959f86113e2347ef90e75e9b47f628c969ed38",
        "10586e2ebd97b828d2ddbba49f9bc066d10cddf71a352f177e09491b5c891f5d",
        "5b13bb3c47222bbbedda637125eae5300fd102bf4a9536313e6e736b2e37de02",
        "9febd93455f138d2b08088131901a7b772e4a5c7b2a4c6cb73c524ed7b2a638c",
        "c0b86c90323a9e80b229e99e727995d27e8d3d717cd20c0b1fbe7f09f036679e",
        "07e746cdc60458973fbca6e2bbe14cf86990b04082b8020eb5fde24757566113",
        "e99acbbdcba299f93b0b0b42ba9519626f086bed34593dbcc36b513195bf48d7",
        "b9bde838b687e90aeb80d62cd35080cfbd8cb7c366e15601a4e058a348f1baac",
        "06afd739e8c459b22d09dde380949ac29bb3efa008e932bd49496fc0863a120f",
        "df18281f1f1e1ee2f0a2406ee5f18dba3c82127572b0ec61e995cb3f5f99bb13",
        "15dd3482b3bb3ba2f498572ac370a15c8c9c59bfcc3c1fa0a730c133a03b9ea6",
        "480b31305085ca70d6ceaf7294a38b008395a0466527333043a8daf42643536d",
        "a8e8cbbe40341a49618ec39c2f657c4ee88f82beed252b157c55d97036b7b4ef",
        "b4bc589e873c34c93586f5ce84446f0cee838ddd4e847493a618868815d8a8cb",
    )
)

_PORTABILITY = ModelProfile(
    key="portability",
    profile_id="granite4-h-small-base-to-post-trained-bf16-singleton-v1",
    device="cuda",
    dtype="bfloat16",
    batch_size=1,
    torch_num_threads=1,
    snapshots=(
        Snapshot(
            role="baseline",
            repository="ibm-granite/granite-4.0-h-small-base",
            revision="4b4faa80c56f74d6fbd8ff305646aa32dd53d600",
            model_type="granitemoehybrid",
            files=(
                SnapshotFile(
                    "config.json",
                    1_800,
                    "3557fc92ef2dcc113c85e3eb1d0898d5d33aa866ac294e6682d587d7ca9498d4",
                ),
                *_GRANITE_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    48_925,
                    "6284b19b6f5b8fe785027406fdafe68eb5eef824667535ace82b30bad6841563",
                ),
                *_GRANITE_BASE_WEIGHTS,
            ),
            tokenizer_contract_sha256=(
                "ae4db62486dcf8936c7c42e0e9e596f93b4f2da452834b12a71effb422d002a1"
            ),
        ),
        Snapshot(
            role="subject",
            repository="ibm-granite/granite-4.0-h-small",
            revision="b8c0982bab7fde4eb48110f5a069527c008fab39",
            model_type="granitemoehybrid",
            files=(
                SnapshotFile(
                    "chat_template.jinja",
                    6_418,
                    "9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce",
                ),
                SnapshotFile(
                    "config.json",
                    1_799,
                    "8616e9f0b30e6fac9696f7c1e1dbd08f1a850ac4af0de6353f7d6009043702ae",
                ),
                *_GRANITE_SHARED,
                SnapshotFile(
                    "model.safetensors.index.json",
                    48_915,
                    "4dbfb53a28571da2a058a3fcf3464582fc73a43ac358f3c9ca9dd728d65e6478",
                ),
                *_GRANITE_SUBJECT_WEIGHTS,
            ),
            tokenizer_contract_sha256=(
                "b86599de52621d29cfb2f1c64561d076ccb851e3edccccd22a0e700771cc8e06"
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
