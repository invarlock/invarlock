from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from invarlock.runtime_providers import tensorrt_llm_identity


def _record_stat(*, atime_ns: int = 1, mtime_ns: int = 2):
    class RecordStat:
        st_dev = 1
        st_ino = 2
        st_mode = 0o100600
        st_size = 3
        st_ctime_ns = 4

        def __init__(self) -> None:
            self.st_atime_ns = atime_ns
            self.st_mtime_ns = mtime_ns

    return RecordStat()


def test_authenticated_record_comparison_ignores_read_driven_atime_only() -> None:
    initial = (tensorrt_llm_identity._FileRecord("rank0.engine", 3, _record_stat()),)
    atime_changed = (replace(initial[0], initial_stat=_record_stat(atime_ns=9)),)
    mtime_changed = (replace(initial[0], initial_stat=_record_stat(mtime_ns=9)),)

    assert tensorrt_llm_identity._same_authenticated_records(initial, atime_changed)
    assert not tensorrt_llm_identity._same_authenticated_records(initial, mtime_changed)


def _config(
    *,
    world_size: int = 1,
    build: dict[str, object] | None = None,
    architecture: str = "LlamaForCausalLM",
) -> dict[str, object]:
    return {
        "version": "1.0.0",
        "pretrained_config": {
            "architecture": architecture,
            "dtype": "float16",
            "mapping": {
                "world_size": world_size,
                "tp_size": world_size,
                "pp_size": 1,
            },
            "num_hidden_layers": 2,
        },
        "build_config": build
        if build is not None
        else {
            "max_batch_size": 8,
            "max_input_len": 128,
            "max_seq_len": 256,
        },
    }


def _bundle(
    root: Path,
    *,
    world_size: int = 1,
    config: dict[str, object] | None = None,
    indent: int | None = None,
) -> Path:
    root.mkdir()
    payload = config if config is not None else _config(world_size=world_size)
    root.joinpath("config.json").write_text(
        json.dumps(payload, indent=indent), encoding="utf-8"
    )
    for rank in range(world_size):
        root.joinpath(f"rank{rank}.engine").write_bytes(
            f"serialized-engine-rank-{rank}".encode()
        )
    return root


def _read(
    path: Path,
    *,
    capability: str = "9.0",
    tokenizer_metadata_sha256: str = "a" * 64,
):
    return tensorrt_llm_identity.read_tensorrt_llm_artifact_identity(
        path,
        target_compute_capability=capability,
        tokenizer_metadata_sha256=tokenizer_metadata_sha256,
    )


def test_identity_accepts_closed_multi_rank_layout_and_hides_bundle_name(
    tmp_path: Path,
) -> None:
    first = _bundle(tmp_path / "sensitive-model-name", world_size=2)
    second = tmp_path / "renamed"
    second.mkdir()
    for source in first.iterdir():
        second.joinpath(source.name).write_bytes(source.read_bytes())

    identity = _read(first)
    renamed = _read(second)

    assert identity == renamed
    assert identity.target_compute_capability == "9.0"
    assert identity.tokenizer_metadata_sha256 == "a" * 64
    assert identity.bundle_name == (
        f"tensorrt-llm-sha256-{identity.engine_bundle_tree_sha256}"
    )
    assert "sensitive-model-name" not in identity.bundle_name
    for digest in (
        identity.engine_bundle_tree_sha256,
        identity.file_inventory_sha256,
        identity.builder_config_sha256,
        identity.tokenizer_metadata_sha256,
        identity.engine_metadata_sha256,
    ):
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")


def test_identity_partitions_tree_builder_metadata_tokenizer_and_compute_capability(
    tmp_path: Path,
) -> None:
    base = _read(_bundle(tmp_path / "base", indent=None))
    whitespace = _read(_bundle(tmp_path / "whitespace", indent=2))
    build_changed = _read(
        _bundle(
            tmp_path / "build",
            config=_config(build={"max_batch_size": 16}),
        )
    )
    metadata_changed = _read(
        _bundle(
            tmp_path / "metadata",
            config=_config(architecture="QwenForCausalLM"),
        )
    )
    capability_changed = _read(_bundle(tmp_path / "capability"), capability="8.9")
    tokenizer_changed = _read(
        _bundle(tmp_path / "tokenizer"), tokenizer_metadata_sha256="b" * 64
    )

    assert whitespace.engine_bundle_tree_sha256 != base.engine_bundle_tree_sha256
    assert whitespace.file_inventory_sha256 != base.file_inventory_sha256
    assert whitespace.builder_config_sha256 == base.builder_config_sha256
    assert whitespace.engine_metadata_sha256 == base.engine_metadata_sha256

    assert build_changed.builder_config_sha256 != base.builder_config_sha256
    assert build_changed.engine_metadata_sha256 == base.engine_metadata_sha256
    assert metadata_changed.engine_metadata_sha256 != base.engine_metadata_sha256
    assert metadata_changed.builder_config_sha256 == base.builder_config_sha256

    assert capability_changed.engine_bundle_tree_sha256 == (
        base.engine_bundle_tree_sha256
    )
    assert capability_changed.file_inventory_sha256 == base.file_inventory_sha256
    assert capability_changed.builder_config_sha256 == base.builder_config_sha256
    assert capability_changed.engine_metadata_sha256 != base.engine_metadata_sha256

    assert tokenizer_changed.engine_bundle_tree_sha256 == (
        base.engine_bundle_tree_sha256
    )
    assert tokenizer_changed.file_inventory_sha256 == base.file_inventory_sha256
    assert tokenizer_changed.builder_config_sha256 == base.builder_config_sha256
    assert tokenizer_changed.tokenizer_metadata_sha256 == "b" * 64
    assert tokenizer_changed.engine_metadata_sha256 != base.engine_metadata_sha256


@pytest.mark.parametrize(
    ("world_size", "remove", "add", "message"),
    [
        (1, "config.json", None, "missing 'config.json'"),
        (2, "rank1.engine", None, "missing 'rank1.engine'"),
        (1, None, "rank1.engine", "unsupported entry 'rank1.engine'"),
        (1, None, "rank00.engine", "unsupported entry 'rank00.engine'"),
        (1, None, "notes.txt", "unsupported entry 'notes.txt'"),
    ],
)
def test_identity_rejects_incomplete_or_extended_layout(
    tmp_path: Path,
    world_size: int,
    remove: str | None,
    add: str | None,
    message: str,
) -> None:
    bundle = _bundle(tmp_path / "bundle", world_size=world_size)
    if remove is not None:
        bundle.joinpath(remove).unlink()
    if add is not None:
        bundle.joinpath(add).write_bytes(b"unsupported")

    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError, match=message):
        _read(bundle)


def test_identity_rejects_empty_engine_and_invalid_world_size(tmp_path: Path) -> None:
    empty = _bundle(tmp_path / "empty")
    empty.joinpath("rank0.engine").write_bytes(b"")
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="must not be empty"
    ):
        _read(empty)

    for index, value in enumerate((0, -1, True, 257, "1")):
        config = _config()
        mapping = config["pretrained_config"]["mapping"]  # type: ignore[index]
        mapping["world_size"] = value
        bundle = _bundle(tmp_path / f"world-size-{index}", config=config)
        with pytest.raises(
            tensorrt_llm_identity.TensorRTLLMIdentityError, match="world_size"
        ):
            _read(bundle)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"version":"1","version":"2","pretrained_config":{},"build_config":{}}',
        b'{"version":"1","pretrained_config":{"value":NaN},"build_config":{"x":1}}',
        b'{"version":"1","pretrained_config":{"value":Infinity},'
        b'"build_config":{"x":1}}',
        b'{"version":"1","pretrained_config":{"value":1e400},"build_config":{"x":1}}',
        b'{"version":"1","pretrained_config":{"value":"\\ud800"},'
        b'"build_config":{"x":1}}',
        b"\xff",
        b"[]",
    ],
)
def test_identity_rejects_non_strict_json(tmp_path: Path, payload: bytes) -> None:
    bundle = _bundle(tmp_path / "bundle")
    bundle.joinpath("config.json").write_bytes(payload)

    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError):
        _read(bundle)


def test_identity_rejects_unbounded_json_depth_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = _config()
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(5):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    deep["build_config"] = nested
    deep_bundle = _bundle(tmp_path / "deep", config=deep)
    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_JSON_DEPTH", 3)
    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError, match="depth"):
        _read(deep_bundle)

    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_JSON_DEPTH", 64)
    sized_bundle = _bundle(tmp_path / "sized")
    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_JSON_BYTES", 8)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="JSON-size"
    ):
        _read(sized_bundle)


def test_identity_rejects_file_count_size_and_total_byte_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count_bundle = _bundle(tmp_path / "count")
    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_FILE_COUNT", 1)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="file count"
    ):
        _read(count_bundle)

    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_FILE_COUNT", 256)
    size_bundle = _bundle(tmp_path / "size")
    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_FILE_BYTES", 8)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="file-size"
    ):
        _read(size_bundle)

    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_FILE_BYTES", 1024**2)
    total_bundle = _bundle(tmp_path / "total")
    monkeypatch.setattr(tensorrt_llm_identity, "_MAX_TOTAL_BYTES", 8)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="total bytes"
    ):
        _read(total_bundle)


def test_identity_rejects_symlinks_hardlinks_and_special_files(tmp_path: Path) -> None:
    symlink_bundle = _bundle(tmp_path / "symlink")
    symlink_bundle.joinpath("rank0.engine").unlink()
    symlink_bundle.joinpath("rank0.engine").symlink_to("config.json")
    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError, match="symlink"):
        _read(symlink_bundle)

    hardlink_bundle = _bundle(tmp_path / "hardlink")
    os.link(
        hardlink_bundle / "rank0.engine",
        hardlink_bundle / "rank1.engine",
    )
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="hard-linked"
    ):
        _read(hardlink_bundle)

    special_bundle = _bundle(tmp_path / "special")
    os.mkfifo(special_bundle / "extra.pipe")
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="regular file"
    ):
        _read(special_bundle)


def test_identity_rejects_casefold_collisions_and_nested_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collision = _bundle(tmp_path / "collision")
    original_listdir = os.listdir

    def colliding_listdir(descriptor: int) -> list[str]:
        entries = original_listdir(descriptor)
        if "rank0.engine" in entries:
            return [*entries, "RANK0.ENGINE"]
        return entries

    monkeypatch.setattr(os, "listdir", colliding_listdir)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="casefold collision"
    ):
        _read(collision)
    monkeypatch.setattr(os, "listdir", original_listdir)

    nested = _bundle(tmp_path / "nested")
    nested.joinpath("lora").mkdir()
    nested.joinpath("lora", "adapter.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError,
        match="unsupported entry 'lora/adapter.json'",
    ):
        _read(nested)


def test_identity_rejects_root_and_intermediate_symlink(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError, match="symlink"):
        _read(root_link)

    parent = tmp_path / "parent"
    parent.mkdir()
    intermediate = parent / "linked"
    intermediate.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError, match="symlink"):
        _read(intermediate)

    other = tmp_path / "other"
    other.mkdir()
    traversal = other / ".." / "bundle"
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="traversal"
    ):
        _read(traversal)


def test_identity_rejects_mutation_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    original_hash = tensorrt_llm_identity._hash_file
    calls = 0

    def mutate_after_first(root_descriptor: int, record):
        nonlocal calls
        result = original_hash(root_descriptor, record)
        calls += 1
        if calls == 1:
            bundle.joinpath("rank0.engine").write_bytes(b"changed")
        return result

    monkeypatch.setattr(tensorrt_llm_identity, "_hash_file", mutate_after_first)
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError, match="changed before hashing"
    ):
        _read(bundle)


@pytest.mark.parametrize("capability", ["9", "9.0.0", "sm90", "", "09.0", "100.000"])
def test_identity_rejects_noncanonical_compute_capability(
    tmp_path: Path, capability: str
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError,
        match="major.minor",
    ):
        _read(bundle, capability=capability)


@pytest.mark.parametrize(
    "digest",
    ["", "a" * 63, "a" * 65, "A" * 64, "sha256:" + "a" * 64, "g" * 64],
)
def test_identity_rejects_noncanonical_tokenizer_digest(
    tmp_path: Path, digest: str
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    with pytest.raises(
        tensorrt_llm_identity.TensorRTLLMIdentityError,
        match="lowercase sha256 digest",
    ):
        _read(bundle, tokenizer_metadata_sha256=digest)


def test_identity_errors_never_disclose_absolute_bundle_path(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "sensitive-bundle-name")
    bundle.joinpath("rank0.engine").unlink()

    with pytest.raises(tensorrt_llm_identity.TensorRTLLMIdentityError) as captured:
        _read(bundle)

    message = str(captured.value)
    assert str(tmp_path) not in message
    assert "sensitive-bundle-name" not in message
    assert "rank0.engine" in message
