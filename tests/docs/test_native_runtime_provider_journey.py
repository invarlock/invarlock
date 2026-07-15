from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.commands.runtime_behavior import runtime_behavior_app
from invarlock.core.runtime_provider import load_runtime_behavioral_schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "integrations" / "runtime_providers"


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _json(name: str) -> object:
    return json.loads((EXAMPLE_ROOT / name).read_text(encoding="utf-8"))


def _write_fake_native_tool(path: Path) -> None:
    path.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
from __future__ import annotations

import os
import sys
from pathlib import Path

args = sys.argv[1:]
commands = (
    "build-schedule",
    "prepare-binding",
    "build-policy",
    "run-side",
    "verify-pair",
)
command = next((candidate for candidate in commands if candidate in args), None)
if command is None:
    raise SystemExit(0)


def option(name: str) -> str:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit(f"missing fake-tool option: {name}") from None


raw_output = option("--receipt" if command == "verify-pair" else "--out")
work = Path(os.environ["INVARLOCK_NATIVE_WORK_DIR"])
if command == "prepare-binding":
    output = work / "bindings" / Path(raw_output).name
    role = output.name.removesuffix("-binding.json")
    stage = f"prepare-{role}"
elif command == "run-side":
    role = option("--role")
    output = work / "sides" / role
    stage = f"run-{role}"
else:
    output = Path(raw_output)
    stage = command

if os.environ.get("FAKE_OMIT_STAGE") == stage:
    raise SystemExit(0)

if command == "run-side":
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "evaluation.report.json",
        "model-artifact.identity.json",
        "runtime-behavior.config.json",
        "runtime-provider.receipt.json",
        "runtime-scoring.observation.json",
        "runtime.manifest.json",
    ):
        (output / name).write_text("{}", encoding="utf-8")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}", encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _fake_wrapper_environment(
    tmp_path: Path,
    tool: Path,
    *,
    subject_provider: str = "llama_cpp",
    subject_gpu: str | None = None,
) -> dict[str, str]:
    records = tmp_path / "records.json"
    records.write_text("[]", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "baseline.gguf"
    artifact.write_bytes(b"fake-gguf")
    settings = tmp_path / "baseline-settings.json"
    settings.write_text(json.dumps({"artifact_sha256": "1" * 64}), encoding="utf-8")
    environment = {
        "PATH": os.environ["PATH"],
        "CONTAINER_ENGINE": str(tool),
        "INVARLOCK_CLI": str(tool),
        "PYTHON_BIN": sys.executable,
        "INVARLOCK_RECORDS": str(records),
        "INVARLOCK_DATASET_IDENTITY": str(dataset),
        "INVARLOCK_NATIVE_WORK_DIR": str(tmp_path / "output"),
        "INVARLOCK_BASELINE_PROVIDER": "llama_cpp",
        "INVARLOCK_BASELINE_IMAGE_DIGEST": "sha256:" + ("1" * 64),
        "INVARLOCK_BASELINE_ARTIFACT": str(artifact),
        "INVARLOCK_BASELINE_SETTINGS": str(settings),
        "INVARLOCK_SUBJECT_PROVIDER": subject_provider,
        "INVARLOCK_SUBJECT_IMAGE_DIGEST": "sha256:" + ("2" * 64),
    }
    if subject_provider == "llama_cpp":
        environment.update(
            {
                "INVARLOCK_SUBJECT_ARTIFACT": str(artifact),
                "INVARLOCK_SUBJECT_SETTINGS": str(settings),
            }
        )
    else:
        engine = tmp_path / "subject-engine"
        engine.mkdir()
        tensor_settings = tmp_path / "subject-settings.json"
        tensor_settings.write_text(
            json.dumps({"engine_bundle_tree_sha256": "3" * 64}),
            encoding="utf-8",
        )
        tokenizer = tmp_path / "tokenizer.json"
        tokenizer.write_text("{}", encoding="utf-8")
        environment.update(
            {
                "INVARLOCK_SUBJECT_ARTIFACT": str(engine),
                "INVARLOCK_SUBJECT_SETTINGS": str(tensor_settings),
                "INVARLOCK_SUBJECT_TOKENIZER_CONTRACT": str(tokenizer),
            }
        )
        if subject_gpu is not None:
            environment["INVARLOCK_SUBJECT_GPU"] = subject_gpu
    return environment


def test_native_runtime_example_templates_are_closed_and_parseable() -> None:
    dataset = _json("dataset-identity.json")
    assert isinstance(dataset, dict)
    assert set(dataset) == {
        "config_name",
        "dataset_name",
        "provider",
        "revision",
        "split",
    }
    assert dataset == {
        "config_name": "gguf-f16-vs-tensorrt-llm-fp16-one-token",
        "dataset_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "provider": "invarlock_native_fixture",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "split": "release-canary",
    }

    records = _json("behavioral-records.json")
    assert isinstance(records, list) and len(records) == 5
    assert all(
        isinstance(record, dict)
        and set(record) == {"expected_output", "input_text", "record_id"}
        for record in records
    )

    llama = _json("llama-cpp-settings.example.json")
    assert isinstance(llama, dict)
    assert set(llama) == {
        "artifact_byte_length",
        "artifact_sha256",
        "backend_binary_sha256",
        "backend_source_sha256",
        "backend_version",
        "batch_size",
        "context_length",
        "gguf_metadata_sha256",
        "max_output_tokens",
        "seed",
        "tensor_inventory_sha256",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
    assert llama["batch_size"] == 1
    assert llama["context_length"] == 8
    assert llama["max_output_tokens"] == 1
    assert llama["seed"] == 0

    tensorrt = _json("tensorrt-llm-settings.example.json")
    assert isinstance(tensorrt, dict)
    assert set(tensorrt) == {
        "backend_build_sha256",
        "backend_version",
        "batch_size",
        "builder_config_sha256",
        "context_length",
        "engine_bundle_tree_sha256",
        "engine_metadata_sha256",
        "file_inventory_sha256",
        "max_output_tokens",
        "runner_binary_sha256",
        "seed",
        "target_compute_capability",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
    assert tensorrt["batch_size"] == 1
    assert tensorrt["context_length"] == 8
    assert tensorrt["max_output_tokens"] == 1
    assert tensorrt["seed"] == 0

    tokenizer = _json("tensorrt-llm-tokenizer-contract.example.json")
    assert isinstance(tokenizer, dict)
    assert set(tokenizer) == {
        "add_special_tokens",
        "clean_up_tokenization_spaces",
        "eos_token_id",
        "format_version",
        "pad_token_id",
        "skip_special_tokens",
        "tokenizer_json",
    }
    assert tokenizer["format_version"] == (
        "invarlock/tensorrt-llm-tokenizer-contract-v1"
    )


def test_native_runtime_wrapper_is_syntax_checked_and_fail_closed() -> None:
    wrapper = EXAMPLE_ROOT / "run_native_pair.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text(encoding="utf-8")

    for command in (
        "build-schedule",
        "prepare-binding",
        "build-policy",
        "run-side",
        "verify-pair",
    ):
        assert command in text
    for boundary in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "INVARLOCK_CONTAINER_EXECUTION=1",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST",
        '--user "$HOST_UID:$HOST_GID"',
        'HOST_UID="$(id -u)"',
        'HOST_GID="$(id -g)"',
        'required_output_file "paired receipt"',
        "required_side_bundle baseline",
        "required_side_bundle subject",
    ):
        assert boundary in text
    assert "eval " not in text
    assert "runtime not probed" not in text


@pytest.mark.parametrize(
    ("stage", "missing_label"),
    [
        ("build-schedule", "behavioral schedule"),
        ("prepare-baseline", "baseline binding"),
        ("prepare-subject", "subject binding"),
        ("build-policy", "acceptance policy pack"),
        ("run-baseline", "baseline side"),
        ("run-subject", "subject side"),
        ("verify-pair", "paired receipt"),
    ],
)
def test_native_runtime_wrapper_rejects_each_missing_stage_output(
    tmp_path: Path,
    stage: str,
    missing_label: str,
) -> None:
    tool = tmp_path / "fake-native-tool"
    _write_fake_native_tool(tool)
    environment = _fake_wrapper_environment(tmp_path, tool)
    environment["FAKE_OMIT_STAGE"] = stage

    result = subprocess.run(
        ["bash", str(EXAMPLE_ROOT / "run_native_pair.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert missing_label in result.stderr
    assert "Native runtime pair verified" not in result.stdout


@pytest.mark.parametrize(
    "selector",
    ["all", "0", "device=-1", "device=GPU-short", "device=0,1"],
)
def test_native_runtime_wrapper_rejects_noncanonical_gpu_selector(
    tmp_path: Path,
    selector: str,
) -> None:
    tool = tmp_path / "fake-native-tool"
    _write_fake_native_tool(tool)
    environment = _fake_wrapper_environment(
        tmp_path,
        tool,
        subject_provider="tensorrt_llm",
        subject_gpu=selector,
    )

    result = subprocess.run(
        ["bash", str(EXAMPLE_ROOT / "run_native_pair.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "must be device=<nonnegative-index> or device=<GPU-UUID>" in result.stderr
    assert "Native runtime pair verified" not in result.stdout


@pytest.mark.parametrize(
    "selector",
    ["device=0", "device=GPU-1234567890abcdef1234"],
)
def test_native_runtime_wrapper_accepts_canonical_gpu_selector_until_postcondition(
    tmp_path: Path,
    selector: str,
) -> None:
    tool = tmp_path / "fake-native-tool"
    _write_fake_native_tool(tool)
    environment = _fake_wrapper_environment(
        tmp_path,
        tool,
        subject_provider="tensorrt_llm",
        subject_gpu=selector,
    )
    environment["FAKE_OMIT_STAGE"] = "verify-pair"

    result = subprocess.run(
        ["bash", str(EXAMPLE_ROOT / "run_native_pair.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "paired receipt" in result.stderr
    assert "GPU" not in result.stderr
    assert "Native runtime pair verified" not in result.stdout


def test_native_runtime_schedule_templates_build_through_installed_command(
    tmp_path: Path,
) -> None:
    output = tmp_path / "behavioral-schedule.json"
    result = CliRunner().invoke(
        runtime_behavior_app,
        [
            "build-schedule",
            "--records",
            str(EXAMPLE_ROOT / "behavioral-records.json"),
            "--dataset-identity",
            str(EXAMPLE_ROOT / "dataset-identity.json"),
            "--out",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["ok"] is True
    assert response["record_count"] == 5
    schedule = load_runtime_behavioral_schedule(output)
    assert schedule.schedule_sha256 == response["schedule_sha256"]


def test_native_runtime_wrapper_rejects_template_digest_before_docker(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"not-used-because-preflight-stops-first")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"artifact_sha256": "0" * 64}), encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    records = tmp_path / "records.json"
    records.write_text("[]", encoding="utf-8")
    zero_image = "sha256:" + ("0" * 64)
    environment = {
        "PATH": os.environ["PATH"],
        "CONTAINER_ENGINE": "true",
        "INVARLOCK_CLI": "true",
        "PYTHON_BIN": os.environ.get("PYTHON", "python3"),
        "INVARLOCK_RECORDS": str(records),
        "INVARLOCK_DATASET_IDENTITY": str(dataset),
        "INVARLOCK_NATIVE_WORK_DIR": str(tmp_path / "output"),
        "INVARLOCK_BASELINE_PROVIDER": "llama_cpp",
        "INVARLOCK_BASELINE_IMAGE_DIGEST": zero_image,
        "INVARLOCK_BASELINE_ARTIFACT": str(artifact),
        "INVARLOCK_BASELINE_SETTINGS": str(settings),
        "INVARLOCK_SUBJECT_PROVIDER": "llama_cpp",
        "INVARLOCK_SUBJECT_IMAGE_DIGEST": zero_image,
        "INVARLOCK_SUBJECT_ARTIFACT": str(artifact),
        "INVARLOCK_SUBJECT_SETTINGS": str(settings),
    }
    result = subprocess.run(
        ["bash", str(EXAMPLE_ROOT / "run_native_pair.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "still contains the template digest" in result.stderr
    assert not (tmp_path / "output").exists()


def test_native_runtime_operator_path_is_discoverable() -> None:
    getting_started = _read("docs/user-guide/getting-started.md")
    integrations = _read("docs/user-guide/integrations.md")
    docs_hub = _read("docs/README.md")
    navigation = _read("mkdocs.yml")
    reference = _read("docs/reference/runtime-providers.md")
    example_index = _read("examples/integrations/README.md")

    assert "Native Runtime Providers" in getting_started
    assert "invarlock advanced runtime-behavior" in getting_started
    assert "native-runtime-providers.md" in integrations
    assert "native-runtime-providers.md" in docs_hub
    assert (
        "Native Runtime Providers: user-guide/native-runtime-providers.md" in navigation
    )
    assert "native-runtime-providers.md" in reference
    assert "runtime_providers/" in example_index


def test_native_runtime_docs_keep_fixture_and_release_asset_claims_narrow() -> None:
    example = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    guide = _read("docs/user-guide/native-runtime-providers.md")
    reference = _read("docs/reference/runtime-providers.md")
    normalized_example = " ".join(example.split())

    assert "not retained proof" in example
    assert "No five-record pair receipt is published" in example
    assert "only persistent writable bind mount" in normalized_example
    assert example.count('--user "$(id -u):$(id -g)"') == 2
    assert "scripts/release/runtime_release_evidence.py build" in guide
    assert "scripts/release/runtime_release_evidence.py validate" in guide
    assert "release-asset carrier" in guide
    assert "not source-tree evidence" in guide
    assert "--qualification llama_cpp:cpu-reference=" in guide
    assert "--qualification tensorrt_llm:pair-a=" in guide
    assert "--qualification tensorrt_llm:pair-b=" in guide
    assert "--expected-qualification llama_cpp:cpu-reference" in guide
    assert "--expected-qualification tensorrt_llm:pair-a" in guide
    assert "--expected-qualification tensorrt_llm:pair-b" in guide
    assert "does not by itself prove independent execution" in guide
    assert "provider-owned `inspect-inputs` command" in reference
    assert "do not hand-assemble" in reference


def test_native_runtime_example_uses_installed_inspection_command() -> None:
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.count("advanced runtime-behavior inspect-inputs") == 2
    assert "derive_native_settings.py" not in readme
    assert "advanced runtime-behavior run-side" in (
        _read("docs/reference/runtime-providers.md")
    )


def test_runtime_provider_capability_reference_matches_contract_fields() -> None:
    text = _read("docs/reference/contracts.md")

    assert (
        "artifact formats, tasks, metrics, execution modes, required extras and "
        "images, platform constraints, evidence surfaces, claim sets, and degraded "
        "or unavailable modes" in text
    )
    assert "precision modes, and device kinds" not in text
