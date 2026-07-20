from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)
from scripts import tensorrt_llm_canary_preflight as preflight

DIGEST = "sha256:" + ("a" * 64)


def _tokenizer_contract() -> dict[str, object]:
    return {
        "add_special_tokens": False,
        "clean_up_tokenization_spaces": False,
        "eos_token_id": 1,
        "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
        "pad_token_id": 0,
        "skip_special_tokens": True,
        "tokenizer_json": {"model": {"type": "BPE"}, "version": "1.0"},
    }


def _valid_inputs(
    tmp_path: Path,
    *,
    input_root: Path | None = None,
) -> dict[str, str]:
    root = input_root or (tmp_path / "inputs")
    engine = root / "engines" / "candidate"
    engine.mkdir(parents=True)
    engine.joinpath("config.json").write_text(
        json.dumps(
            {
                "build_config": {
                    "max_batch_size": 8,
                    "max_input_len": 128,
                    "max_seq_len": 256,
                },
                "pretrained_config": {
                    "architecture": "LlamaForCausalLM",
                    "dtype": "float16",
                    "mapping": {"pp_size": 1, "tp_size": 1, "world_size": 1},
                },
                "version": "1.0.0",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    engine.joinpath("rank0.engine").write_bytes(b"serialized-engine-fixture")
    tokenizer = root / "contracts" / "tokenizer.json"
    tokenizer.parent.mkdir()
    tokenizer.write_text(
        json.dumps(_tokenizer_contract(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    engine_tree_sha256 = read_tensorrt_llm_artifact_identity(
        engine,
        target_compute_capability="9.0",
        tokenizer_metadata_sha256=tokenizer_sha256,
    ).engine_bundle_tree_sha256
    return {
        "image": DIGEST,
        "image_digest": DIGEST,
        "input_root": str(root),
        "engine_bundle": "engines/candidate",
        "tokenizer_contract": "contracts/tokenizer.json",
        "expected_engine_tree_sha256": engine_tree_sha256,
        "expected_tokenizer_sha256": tokenizer_sha256,
        "expected_output_sha256": "d" * 64,
        "tmpfs_gib": "8",
    }


@pytest.mark.parametrize(
    "image",
    (DIGEST, f"registry.example/invarlock/runtime@{DIGEST}"),
)
def test_validate_authenticates_inputs_and_returns_canonical_root(
    tmp_path: Path,
    image: str,
) -> None:
    arguments = _valid_inputs(tmp_path)
    arguments["image"] = image

    canonical_root = preflight.validate(**arguments)

    assert canonical_root == str(Path(arguments["input_root"]).absolute())


@pytest.mark.parametrize(
    "image",
    (
        "candidate image@" + DIGEST,
        "registry.example/invarlock,candidate@" + DIGEST,
        "registry.example/invarlock/candidate@@" + DIGEST,
        "@" + DIGEST,
        "registry.example/invarlock/candidate@" + DIGEST + "@" + DIGEST,
        "registry.example/invarlock/\tcandidate@" + DIGEST,
    ),
)
def test_validate_rejects_invalid_oci_image_references(
    tmp_path: Path,
    image: str,
) -> None:
    arguments = _valid_inputs(tmp_path)
    arguments["image"] = image

    with pytest.raises(preflight.CanaryPreflightError, match="IMAGE must be"):
        preflight.validate(**arguments)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {"image": "sha256:short", "image_digest": "sha256:short"},
            "canonical sha256 image digest",
        ),
        ({"image": "candidate:mutable"}, "IMAGE must be"),
        (
            {"expected_engine_tree_sha256": "B" * 64},
            "lowercase sha256 digest",
        ),
        ({"expected_engine_tree_sha256": "0" * 64}, "engine bundle does not match"),
        ({"expected_tokenizer_sha256": "0" * 64}, "tokenizer contract does not match"),
        ({"tmpfs_gib": "eight"}, "integer from 4 to 64"),
        ({"tmpfs_gib": "65"}, "integer from 4 to 64"),
        ({"input_root": "relative/inputs"}, "absolute path"),
        (
            {"input_root": "/definitely/missing/invarlock-canary-inputs"},
            "existing non-symlink directory",
        ),
    ),
)
def test_validate_rejects_invalid_or_mismatched_inputs(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    arguments = _valid_inputs(tmp_path)
    arguments.update(updates)

    with pytest.raises(preflight.CanaryPreflightError, match=message):
        preflight.validate(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("engine_bundle", ""),
        ("engine_bundle", "../candidate"),
        ("engine_bundle", "/absolute/candidate"),
        ("engine_bundle", "engines//candidate"),
        ("tokenizer_contract", "contracts\\tokenizer.json"),
        ("tokenizer_contract", "C:/tokenizer.json"),
        ("tokenizer_contract", "contracts/tokenizer.json\n"),
    ),
)
def test_validate_rejects_nonportable_resource_paths(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    arguments = _valid_inputs(tmp_path)
    arguments[field] = value

    with pytest.raises(preflight.CanaryPreflightError, match="portable relative path"):
        preflight.validate(**arguments)


def test_validate_rejects_comma_and_symlinked_parent_roots(tmp_path: Path) -> None:
    comma_arguments = _valid_inputs(
        tmp_path / "comma-case",
        input_root=tmp_path / "comma-case" / "inputs,unsafe",
    )
    with pytest.raises(preflight.CanaryPreflightError, match="must not contain"):
        preflight.validate(**comma_arguments)

    real_parent = tmp_path / "real-parent"
    linked_arguments = _valid_inputs(
        tmp_path / "link-case",
        input_root=real_parent / "inputs",
    )
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_arguments["input_root"] = str(linked_parent / "inputs")
    with pytest.raises(preflight.CanaryPreflightError, match="symlink"):
        preflight.validate(**linked_arguments)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_engine", "engine identity cannot be authenticated"),
        ("engine_symlink", "engine identity cannot be authenticated"),
        ("engine_file", "engine identity cannot be authenticated"),
        ("tokenizer_symlink", "TOKENIZER_CONTRACT must exist"),
        ("tokenizer_directory", "stable regular file"),
    ),
)
def test_validate_rejects_unsafe_resource_layouts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    arguments = _valid_inputs(tmp_path)
    root = Path(arguments["input_root"])
    engine = root / arguments["engine_bundle"]
    tokenizer = root / arguments["tokenizer_contract"]
    if mutation == "missing_engine":
        arguments["engine_bundle"] = "engines/missing"
    elif mutation == "engine_symlink":
        for child in engine.iterdir():
            child.unlink()
        engine.rmdir()
        target = tmp_path / "external-engine"
        target.mkdir()
        engine.symlink_to(target, target_is_directory=True)
    elif mutation == "engine_file":
        for child in engine.iterdir():
            child.unlink()
        engine.rmdir()
        engine.write_text("not a directory\n", encoding="utf-8")
    elif mutation == "tokenizer_symlink":
        tokenizer.unlink()
        target = tmp_path / "external-tokenizer.json"
        target.write_text("{}\n", encoding="utf-8")
        tokenizer.symlink_to(target)
    else:
        tokenizer.unlink()
        tokenizer.mkdir()

    with pytest.raises(preflight.CanaryPreflightError, match=message):
        preflight.validate(**arguments)


def test_validate_rejects_malformed_closed_tokenizer_contract(tmp_path: Path) -> None:
    arguments = _valid_inputs(tmp_path)
    tokenizer = Path(arguments["input_root"]) / arguments["tokenizer_contract"]
    tokenizer.write_text('{"unexpected":true}\n', encoding="utf-8")
    arguments["expected_tokenizer_sha256"] = hashlib.sha256(
        tokenizer.read_bytes()
    ).hexdigest()

    with pytest.raises(preflight.CanaryPreflightError, match="fields are not closed"):
        preflight.validate(**arguments)


def test_main_prints_only_the_authenticated_canonical_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = _valid_inputs(tmp_path)
    argv: list[str] = []
    for name, value in arguments.items():
        argv.extend(("--" + name.replace("_", "-"), value))

    assert preflight.main(argv) == 0
    assert capsys.readouterr().out == f"{Path(arguments['input_root']).absolute()}\n"


def test_secure_directory_authentication_requires_platform_nofollow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(preflight.os, "O_NOFOLLOW")
    with pytest.raises(preflight.CanaryPreflightError, match="nofollow"):
        preflight._directory_flags()


def test_input_root_rejects_explicit_traversal(tmp_path: Path) -> None:
    value = str(tmp_path / "child" / ".." / "inputs")
    with pytest.raises(preflight.CanaryPreflightError, match="traversal"):
        preflight._canonical_input_root(value)


def test_tokenizer_contract_size_bound_and_main_failure_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _valid_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_MAX_TOKENIZER_CONTRACT_BYTES", 0)
    with pytest.raises(preflight.CanaryPreflightError, match="size bound"):
        preflight.validate(**arguments)

    argv: list[str] = []
    arguments["image"] = "mutable:latest"
    for name, value in arguments.items():
        argv.extend(("--" + name.replace("_", "-"), value))
    with pytest.raises(SystemExit):
        preflight.main(argv)
