from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from examples.integrations import gguf_deployment as example
from examples.integrations.evaluator_transaction.model_profiles import SnapshotFile
from examples.integrations.gguf_llama_cpp import PendingTrust


def _completed(
    command: list[str], stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _pending_trust() -> PendingTrust:
    return PendingTrust(
        anchors={"schedule_digest": "sha256:" + "2" * 64},
        policy_bytes=b"{}\n",
        external=False,
        trust_root=None,
        verifier_key_bytes=None,
        evidence_fingerprint="sha256:" + "3" * 64,
        verifier_fingerprint="sha256:" + "4" * 64,
    )


def test_make_target_installs_the_hf_runtime_and_gguf_addin() -> None:
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
        encoding="utf-8"
    )
    target = makefile.split("example-gguf-deployment:", 1)[1].split("\n\n", 1)[0]

    assert "--with '.[hf]'" in target
    assert "--with ./addins/gguf" in target
    assert "examples.integrations.gguf_deployment" in target


def test_profile_uses_one_pinned_post_trained_9b_source_and_flagship_corpus() -> None:
    profile = example.deployment_profile()

    assert profile.source.repository == "Qwen/Qwen3.5-9B"
    assert profile.source.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert profile.source.checkpoint_tree_sha256 == (
        "sha256:a73abe2d4664cef43cf774e975ad86f614faf57a7e9e63ae660e42e4245bcbf7"
    )
    assert profile.corpus.key == "flagship"
    assert profile.corpus.record_count == 400
    assert profile.quantization == "Q5_K_M"
    assert profile.baseline_device == "cuda"
    assert profile.subject_device == "cpu"


def test_conversion_and_quantization_commands_are_immutable_and_networkless(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-model"
    source.mkdir()
    archive = tmp_path / "llama.cpp.tar.gz"
    archive.write_bytes(b"source")
    output = tmp_path / "output"
    output.mkdir()
    image = "sha256:" + "a" * 64

    conversion = example._conversion_command(
        "docker",
        image,
        source_checkpoint=source,
        source_archive=archive,
        output_root=output,
    )
    assert conversion[:3] == ["docker", "run", "--rm"]
    assert conversion[conversion.index("--network") + 1] == "none"
    assert "--read-only" in conversion
    assert conversion[conversion.index("--entrypoint") + 1] == "python"
    assert image in conversion
    assert "convert_hf_to_gguf.py" in conversion[-1]
    assert "--outtype" in conversion[-1]
    assert "bf16" in conversion[-1]

    intermediate = output / example._INTERMEDIATE_NAME
    quantization = example._quantization_command(
        "docker",
        image,
        intermediate=intermediate,
        output_root=output,
    )
    assert quantization[quantization.index("--network") + 1] == "none"
    assert quantization[quantization.index("--entrypoint") + 1] == (
        "/opt/llama.cpp/llama-quantize"
    )
    assert quantization[-1] == "Q5_K_M"
    assert "--allow-requantize" not in quantization


def test_conversion_rejects_missing_or_unchanged_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "llama.cpp.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(
        example, "_LLAMA_SOURCE_SHA256", example.hashlib.sha256(b"archive").hexdigest()
    )
    output = tmp_path / "output"
    output.mkdir()
    commands: list[list[str]] = []

    def missing(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", missing)
    with pytest.raises(RuntimeError, match="BF16 GGUF"):
        example._convert_and_quantize(
            tmp_path,
            source_checkpoint=source,
            source_archive=archive,
            output_root=output,
            container_engine="docker",
            conversion_image_id="sha256:" + "a" * 64,
            gguf_image_id="sha256:" + "b" * 64,
        )
    assert len(commands) == 1

    def unchanged(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if len(commands) == 1:
            (output / example._INTERMEDIATE_NAME).write_bytes(b"same")
        else:
            (output / example._SUBJECT_NAME).write_bytes(b"same")
        return _completed(command)

    commands.clear()
    monkeypatch.setattr(example.launch, "_run", unchanged)
    with pytest.raises(RuntimeError, match="distinct GGUF"):
        example._convert_and_quantize(
            tmp_path,
            source_checkpoint=source,
            source_archive=archive,
            output_root=output,
            container_engine="docker",
            conversion_image_id="sha256:" + "a" * 64,
            gguf_image_id="sha256:" + "b" * 64,
        )
    assert not (output / example._INTERMEDIATE_NAME).exists()


def test_checkpoint_staging_removes_a_tree_that_fails_final_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = b'{"model_type":"qwen3_5"}'
    snapshot = replace(
        example.deployment_profile().source,
        files=(
            SnapshotFile(
                name="config.json",
                byte_length=len(config),
                sha256=hashlib.sha256(config).hexdigest(),
            ),
        ),
        checkpoint_tree_sha256="sha256:" + "0" * 64,
    )
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(
        example.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(config),
    )

    with pytest.raises(RuntimeError, match="tree is not pinned"):
        example._stage_source_checkpoint(models, snapshot)

    assert not (models / "source").exists()


def test_transaction_binds_cross_runtime_source_lineage_and_predeclared_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "transaction"
    runtime_root = root / "evaluation" / "runtime"
    source = runtime_root / "models" / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    subject = runtime_root / "models" / example._SUBJECT_NAME
    subject.write_bytes(b"GGUF-subject")
    transformation = {
        "format": example._TRANSFORMATION_FORMAT,
        "source": {"checkpoint_tree_sha256": "sha256:" + "1" * 64},
        "subject": {"sha256": "2" * 64},
    }
    baseline_spec = {
        "model_id": "Qwen/Qwen3.5-9B",
        "settings": {
            "batch_size": 1,
            "checkpoint_tree_sha256": "sha256:" + "1" * 64,
            "context_length": 1024,
            "max_output_tokens": 1,
            "offline": True,
            "seed": 20_260_716,
            "timeout_seconds": 300,
            "tokenizer_metadata_sha256": "3" * 64,
        },
    }
    subject_spec = {"model_id": "gguf-sha256-model.gguf", "settings": {}}
    monkeypatch.setattr(
        example,
        "_artifact_anchor",
        lambda provider, *_args, **_kwargs: (
            "sha256:" + ("a" if provider == "hf_transformers" else "b") * 64
        ),
    )

    paths, _pending = example._prepare_transaction(
        root,
        runtime_root=runtime_root,
        source_checkpoint=source,
        subject=subject,
        baseline_spec=baseline_spec,
        subject_spec=subject_spec,
        transformation=transformation,
        baseline_image_id="sha256:" + "c" * 64,
        subject_image_id="sha256:" + "d" * 64,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    comparison = request["comparison"]
    assert comparison["baseline"]["runtime"]["provider"] == "hf_transformers"
    assert comparison["subject"]["runtime"]["provider"] == "llama_cpp"
    assert comparison["baseline"]["artifact"]["locator"] == (
        "hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert comparison["subject"]["artifact"]["locator"].startswith(
        "derived://Qwen/Qwen3.5-9B@c2022362"
    )
    assert comparison["dataset"]["name"] == "TIGER-Lab/MMLU-Pro/qwen35-no-think"
    policy = json.loads(paths.independent_policy.read_text(encoding="utf-8"))
    assert policy["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -2.0,
        "maximum_interval_width_pp": 10.0,
        "minimum_record_count": 400,
        "minimum_side_accuracy": 0.2,
    }
    observation = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    assert observation == transformation
    assert request["observations"][0]["scope"] == "subject"


def test_execute_selects_independent_images_and_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = example._paths(tmp_path / "transaction")
    paths.trusted_inputs.parent.mkdir(parents=True, exist_ok=True)
    paths.trusted_inputs.write_text('{"anchors":{}}\n', encoding="utf-8")
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "baseline": {"mean_score": 0.5},
                "comparison": {"value": -1.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.49},
                "verdict": "pass",
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool = False,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "--preflight" in command:
            return _completed(
                command, json.dumps({"request_digest": "sha256:" + "9" * 64})
            )
        if "verify" in command:
            paths.receipt.parent.mkdir(parents=True, exist_ok=True)
            paths.receipt.write_text(
                json.dumps(
                    {
                        "statement": {
                            "verdict": {
                                "integrity_ok": True,
                                "ok": True,
                                "policy_verdict": "pass",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
        if "report" in command:
            paths.html_report.write_text("<html></html>", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", run)
    example._execute(
        tmp_path,
        paths,
        runtime_root=tmp_path / "runtime",
        container_engine="docker",
        baseline_image_id="sha256:" + "a" * 64,
        subject_image_id="sha256:" + "b" * 64,
        pending_trust=_pending_trust(),
    )

    evaluate = commands[0]
    assert evaluate[evaluate.index("--baseline-runtime-image") + 1] == (
        "sha256:" + "a" * 64
    )
    assert evaluate[evaluate.index("--subject-runtime-image") + 1] == (
        "sha256:" + "b" * 64
    )
    assert evaluate[evaluate.index("--baseline-runtime-device") + 1] == "cuda"
    assert evaluate[evaluate.index("--subject-runtime-device") + 1] == "cpu"


def test_execute_can_retain_an_independently_verified_policy_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = example._paths(tmp_path / "transaction")
    paths.trusted_inputs.parent.mkdir(parents=True, exist_ok=True)
    paths.trusted_inputs.write_text('{"anchors":{}}\n', encoding="utf-8")
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "baseline": {"mean_score": 0.5},
                "comparison": {"value": -3.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.47},
                "verdict": "fail",
            }
        ),
        encoding="utf-8",
    )

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--preflight" in command:
            return _completed(
                command, json.dumps({"request_digest": "sha256:" + "9" * 64})
            )
        if "report" in command:
            paths.html_report.write_text("<html></html>", encoding="utf-8")
        return _completed(command)

    def verify(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        paths.receipt.parent.mkdir(parents=True, exist_ok=True)
        paths.receipt.write_text(
            json.dumps(
                {
                    "statement": {
                        "verdict": {
                            "integrity_ok": True,
                            "ok": False,
                            "policy_verdict": "fail",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            int(example.EvidencePackStatus.REPORTS),
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(example.launch, "_run", run)
    monkeypatch.setattr(example, "run_bounded_command", verify)

    example._execute(
        tmp_path,
        paths,
        runtime_root=tmp_path / "runtime",
        container_engine="docker",
        baseline_image_id="sha256:" + "a" * 64,
        subject_image_id="sha256:" + "b" * 64,
        pending_trust=_pending_trust(),
        allow_policy_fail=True,
    )
