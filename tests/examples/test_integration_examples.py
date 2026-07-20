from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from examples.integrations import launch, qwen3_profile
from examples.integrations import run as integration

ZERO_DIGEST = "sha256:" + ("0" * 64)
REAL_QWEN3_LOADER = qwen3_profile.load_model_and_tokenizer


@pytest.fixture(autouse=True)
def tiny_qwen3_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused tests offline while production uses the pinned checkpoint."""

    def load_model_and_tokenizer(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        import tokenizers

        torch.manual_seed(integration._SEED)  # type: ignore[attr-defined]
        vocabulary = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "alpha": 4,
            "beta": 5,
            "target": 6,
            "other": 7,
        }
        backend = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
        )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer = transformers.PreTrainedTokenizerFast(  # type: ignore[attr-defined]
            tokenizer_object=backend,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        config = transformers.Qwen3Config(  # type: ignore[attr-defined]
            vocab_size=8,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=False,
            tie_word_embeddings=False,
        )
        return transformers.Qwen3ForCausalLM(config), tokenizer  # type: ignore[attr-defined]

    monkeypatch.setattr(
        qwen3_profile, "load_model_and_tokenizer", load_model_and_tokenizer
    )


def test_hf_preparation_creates_closed_distinct_transaction(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "hf",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = (
        (paths.evaluation / "inputs/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(records) == 50
    parsed_records = [json.loads(record) for record in records]
    assert len({record["prompt"] for record in parsed_records}) == 50
    assert {record["expected"] for record in parsed_records} == {" target"}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        paths.evaluation / "models/baseline", local_files_only=True
    )
    for record in parsed_records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=True).input_ids
        target_ids = tokenizer(record["expected"], add_special_tokens=False).input_ids
        decoded_prompt = tokenizer.decode(
            prompt_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        decoded_continuation = tokenizer.decode(
            prompt_ids + target_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        assert decoded_continuation == decoded_prompt + record["expected"]
    assert request["comparison"]["dataset"]["name"] == "hf-transformers-smoke"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == "hf_transformers"
    assert request["comparison"]["subject"]["runtime"]["provider"] == "hf_transformers"
    baseline_locator = request["comparison"]["baseline"]["artifact"]["locator"]
    subject_locator = request["comparison"]["subject"]["artifact"]["locator"]
    assert baseline_locator.startswith(
        f"hf://{qwen3_profile.MODEL_ID}@{qwen3_profile.MODEL_REVISION}#"
    )
    assert subject_locator.startswith(
        "generated://invarlock-example/hf-transformers-subject@sha256:"
    )
    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["method"] == "causal-output-row-fit"
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]
    assert anchors["baseline_runtime_digest"] == ZERO_DIGEST
    assert paths.evidence_key.stat().st_mode & 0o777 == 0o600
    assert paths.verifier_key.stat().st_mode & 0o777 == 0o600
    for checkpoint in ("baseline", "subject"):
        model_root = paths.evaluation / "models" / checkpoint
        assert model_root.stat().st_mode & 0o005 == 0o005
        assert all(
            path.stat().st_mode & (0o005 if path.is_dir() else 0o004)
            for path in model_root.rglob("*")
        )


def test_hf_preparation_can_author_an_exact_match_transaction(tmp_path: Path) -> None:
    paths, _anchors = integration._prepare_workspace(
        tmp_path / "hf-exact-match",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
        metric="exact_match",
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (paths.evaluation / "inputs/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    policy = json.loads(
        (paths.evaluation / "inputs/acceptance.json").read_text(encoding="utf-8")
    )
    assert {record["expected"] for record in records} == {"target"}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generated_outputs: dict[str, set[str]] = {}
    for role in ("baseline", "subject"):
        checkpoint = paths.evaluation / "models" / role
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        encoded = tokenizer(
            [record["prompt"] for record in records],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_outputs[role] = set(
            tokenizer.batch_decode(
                generated[:, -1:],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
        )
    assert generated_outputs["baseline"] != {"target"}
    assert generated_outputs["subject"] == {"target"}
    assert request["comparison"]["metric"] == "exact_match"
    assert policy == {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": 0.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 50,
                }
            }
        }
    }


def test_peft_preparation_trains_serializes_reloads_and_merges(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "peft",
        integration="peft-lora",
        runtime_image_digest=ZERO_DIGEST,
    )

    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["library"] == "peft"
    assert summary["library_version"] == "0.19.1"
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["target_modules"] == ["q_proj", "v_proj"]
    assert summary["training_record_count"] == 50
    assert summary["training_steps"] == 12
    assert summary["final_loss"] < summary["initial_loss"]
    assert (paths.root / "upstream/peft-adapter/adapter_model.safetensors").is_file()
    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    assert request["observations"] == [
        {
            "id": "peft-lora-subject-transformation",
            "kind": "artifact_transformation",
            "scope": "subject",
            "path": "inputs/subject-transformation.json",
        }
    ]
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]


def test_torchao_preparation_quantizes_and_materializes_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_loader = qwen3_profile.load_model_and_tokenizer

    def load_bfloat16_model(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        model, tokenizer = tiny_loader(torch=torch, transformers=transformers)
        return model.to(dtype=torch.bfloat16), tokenizer  # type: ignore[attr-defined]

    monkeypatch.setattr(qwen3_profile, "load_model_and_tokenizer", load_bfloat16_model)
    paths, anchors = integration._prepare_workspace(
        tmp_path / "torchao",
        integration="torchao-int8",
        runtime_image_digest=ZERO_DIGEST,
    )

    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (paths.evaluation / "inputs/acceptance.json").read_text(encoding="utf-8")
    )
    assert summary["library"] == "torchao"
    assert summary["library_version"] == "0.17.0"
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["quantization"] == {
        "configuration": "Int8WeightOnlyConfig(version=2)",
        "excluded_modules": ["lm_head"],
        "materialization": "dequantize-dense-state-v1",
        "selected_module_type": "torch.nn.Linear",
    }
    assert len(summary["quantized_tensors"]) >= 7
    assert "lm_head.weight" not in summary["quantized_tensors"]
    assert summary["quantized_tensor_count"] == len(summary["quantized_tensors"])
    import torch as pytorch
    from transformers import AutoModelForCausalLM

    persisted = AutoModelForCausalLM.from_pretrained(
        paths.evaluation / "models/subject", local_files_only=True
    ).state_dict()
    commitment = hashlib.sha256()
    for name in summary["quantized_tensors"]:
        tensor = persisted[name].contiguous()
        descriptor = integration.canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            }
        )
        payload = tensor.view(pytorch.uint8).numpy().tobytes(order="C")
        commitment.update(len(descriptor).to_bytes(8, "big"))
        commitment.update(descriptor)
        commitment.update(len(payload).to_bytes(8, "big"))
        commitment.update(payload)
    assert summary["dequantized_tensor_state_sha256"] == (
        "sha256:" + commitment.hexdigest()
    )
    assert summary["dequantized_tensor_state_loaded_exact"] is True
    assert summary["dequantized_tensor_state_save_reload_exact"] is True
    observation = summary["live_kernel_observation"]
    assert observation["authority"] == "observation"
    assert observation["device"] == "cpu"
    assert observation["record_count"] == 50
    assert 0 <= observation["top1_agreement_count"] <= 50
    assert math.isfinite(observation["max_abs_next_token_logit_delta"])
    assert math.isfinite(observation["mean_abs_next_token_logit_delta"])
    assert (
        observation["max_abs_next_token_logit_delta"]
        >= (observation["mean_abs_next_token_logit_delta"])
    )
    assert observation["mean_abs_next_token_logit_delta"] >= 0.0
    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    assert request["observations"][0]["id"] == ("torchao-int8-subject-transformation")
    assert (
        policy["resolved_policy"]["metrics"]["normalized_nll_per_utf8_byte"][
            "ratio_max"
        ]
        == 1.01
    )
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]


def test_torchao_preparation_rejects_nonfinite_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_loader = qwen3_profile.load_model_and_tokenizer

    def load_nonfinite_model(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        model, tokenizer = tiny_loader(torch=torch, transformers=transformers)
        with torch.no_grad():  # type: ignore[attr-defined]
            model.model.embed_tokens.weight[0, 0] = float("nan")
        return model, tokenizer

    monkeypatch.setattr(qwen3_profile, "load_model_and_tokenizer", load_nonfinite_model)
    with pytest.raises(RuntimeError, match="produced a non-finite tensor"):
        integration._prepare_workspace(
            tmp_path / "torchao-nonfinite",
            integration="torchao-int8",
            runtime_image_digest=ZERO_DIGEST,
        )


def test_preparation_rejects_existing_and_unknown_workspace(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="workspace already exists"):
        integration._prepare_workspace(
            existing,
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
        )
    paths = integration._paths(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="unsupported integration"):
        integration._create_checkpoints(paths, "unknown")


def test_execute_invokes_public_commands_and_checks_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = integration._paths(tmp_path)
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"comparison": {"value": 0.75}, "verdict": "pass"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(integration, "_run", lambda command: commands.append(command))

    integration._execute(
        paths,
        container_engine="docker",
        runtime_image="example:current",
        runtime_image_digest=ZERO_DIGEST,
        runtime_device="cpu",
    )

    assert [command[3] for command in commands] == ["evaluate", "verify", "report"]
    assert "--trust-profile" in commands[1]
    assert "--html" in commands[2]

    report.write_text(
        json.dumps({"comparison": {"value": "bad"}, "verdict": "pass"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not produce a passing ratio"):
        integration._execute(
            paths,
            container_engine="docker",
            runtime_image="example:current",
            runtime_image_digest=ZERO_DIGEST,
            runtime_device="cpu",
        )


def test_command_runner_surfaces_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 3, stdout="out\n", stderr="bad\n"
        ),
    )
    with pytest.raises(RuntimeError, match="status 3"):
        integration._run(["false"])


def test_run_main_prepare_only_and_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = integration.main(
        [
            "hf-transformers",
            "--workspace",
            str(tmp_path / "prepared"),
            "--runtime-image-digest",
            ZERO_DIGEST,
            "--prepare-only",
        ]
    )
    assert prepared == 0

    existing = tmp_path / "existing"
    existing.mkdir()
    assert (
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(existing),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit, match="full execution requires"):
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(tmp_path / "missing-image"),
                "--runtime-image-digest",
                ZERO_DIGEST,
            ]
        )


def test_launch_helpers_require_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(("a" * 40, ""))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    assert launch._require_committed_checkout(tmp_path) == "a" * 40

    replies = iter(("b" * 40, " M tracked.py"))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    with pytest.raises(RuntimeError, match="commit or stash"):
        launch._require_committed_checkout(tmp_path)


def test_runtime_image_builds_from_authenticated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    commands: list[list[str]] = []
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "sha256:" + ("d" * 64)
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    image, digest = launch._runtime_image(
        repository=repository,
        build_root=build,
        container_engine="docker",
        dockerfile="addins/example/Dockerfile",
        image_prefix="custom-example",
    )
    assert image == "sha256:" + ("d" * 64)
    assert digest == "sha256:" + ("d" * 64)
    assert any("authenticated_runtime_build.py" in " ".join(item) for item in commands)
    build_command = next(
        item for item in commands if "authenticated_runtime_build.py" in " ".join(item)
    )
    assert "addins/example/Dockerfile" in build_command
    assert "custom-example:" + "c" * 12 in build_command
    assert build_command[build_command.index("--statement") + 1] == str(
        build / "runtime-build.json"
    )

    def invalid_inspect(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "not-a-digest"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", invalid_inspect)
    with pytest.raises(RuntimeError, match="sha256 image ID"):
        launch._runtime_image(
            repository=repository,
            build_root=build,
            container_engine="docker",
        )


def test_launch_main_dispatches_prepare_and_full_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    assert (
        launch.main(
            [
                "hf-transformers",
                "--prepare-only",
                "--workspace",
                str(tmp_path / "prepare"),
            ]
        )
        == 0
    )
    assert "--prepare-only" in commands[-1]

    runtime_calls: list[dict[str, object]] = []

    def fake_runtime(**kwargs: object) -> tuple[str, str]:
        runtime_calls.append(kwargs)
        return "example:current", "sha256:" + ("e" * 64)

    monkeypatch.setattr(launch, "_runtime_image", fake_runtime)
    assert (
        launch.main(
            [
                "peft-lora",
                "--workspace",
                str(tmp_path / "full"),
                "--runtime-device",
                "cuda:1",
            ]
        )
        == 0
    )
    assert "example:current" in commands[-1]
    assert runtime_calls[0]["dockerfile"] == "runtime/Dockerfile.cuda"
    assert runtime_calls[0]["image_prefix"] == "invarlock-example-runtime-cuda"
    assert commands[-1][commands[-1].index("--runtime-device") + 1] == "cuda:1"

    with pytest.raises(SystemExit):
        launch._parser().parse_args(["hf-transformers", "--runtime-device", "gpu:1"])

    existing = tmp_path / "existing"
    existing.mkdir()
    assert launch.main(["hf-transformers", "--workspace", str(existing)]) == 2


def test_launch_runner_reports_subprocess_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, stdout="", stderr="failed"
        ),
    )
    with pytest.raises(RuntimeError, match="status 7"):
        launch._run(["bad"], cwd=tmp_path)


def test_qwen3_profile_uses_an_immutable_official_revision() -> None:
    assert qwen3_profile.MODEL_ID == "Qwen/Qwen3-0.6B"
    assert qwen3_profile.MODEL_REVISION == ("c1899de289a04d12100db370d81485cdf75e47ca")
    assert qwen3_profile.PEFT_TARGET_MODULES == ("q_proj", "v_proj")


def test_qwen3_loader_passes_the_pin_to_model_and_tokenizer() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class Config:
        use_cache = True

    class Model:
        config = Config()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> Tokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> Model:
            calls.append(("model", model_id, kwargs))
            return Model()

    Transformers = type(
        "Transformers",
        (),
        {"AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModel},
    )

    model, _tokenizer = REAL_QWEN3_LOADER(torch=object(), transformers=Transformers)
    assert model.config.use_cache is False
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[1] == qwen3_profile.MODEL_ID for call in calls)
    assert all(call[2]["revision"] == qwen3_profile.MODEL_REVISION for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)
    assert calls[1][2]["use_safetensors"] is True
    assert calls[1][2]["dtype"] == "auto"
