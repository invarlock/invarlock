from __future__ import annotations

import hashlib
import io
import json
import stat
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


def _baseline_spec() -> dict[str, object]:
    profile = example.deployment_profile()
    return {
        "model_id": profile.source.repository,
        "settings": {
            "batch_size": 1,
            "checkpoint_tree_sha256": profile.source.checkpoint_tree_sha256,
            "context_length": profile.corpus.context_length,
            "immutable_revision": profile.source.revision,
            "max_output_tokens": 1,
            "offline": True,
            "seed": example._SEED,
            "timeout_seconds": example._TIMEOUT_SECONDS,
            "tokenizer_metadata_sha256": profile.source.tokenizer_contract_sha256,
        },
    }


def _subject_spec(subject: Path) -> dict[str, object]:
    digest = hashlib.sha256(subject.read_bytes()).hexdigest()
    return {
        "model_id": f"gguf-sha256-{digest}.gguf",
        "settings": {
            "artifact_byte_length": subject.stat().st_size,
            "artifact_sha256": digest,
            "backend_binary_sha256": "2" * 64,
            "backend_source_sha256": "3" * 64,
            "backend_version": "b10015 (12127def)",
            "batch_size": 1,
            "context_length": example.deployment_profile().corpus.context_length,
            "gguf_metadata_sha256": "4" * 64,
            "max_output_tokens": 1,
            "seed": example._SEED,
            "tensor_inventory_sha256": "5" * 64,
            "timeout_seconds": example._TIMEOUT_SECONDS,
            "tokenizer_metadata_sha256": "6" * 64,
        },
    }


def test_make_target_installs_the_hf_runtime_and_gguf_addin() -> None:
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
        encoding="utf-8"
    )
    target = makefile.split("example-gguf-deployment:", 1)[1].split("\n\n", 1)[0]

    assert "--with '.[hf]'" in target
    assert "--with ./addins/gguf" in target
    assert "examples.integrations.gguf_deployment" in target


def test_requested_workspace_is_owner_only_before_any_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "retained"
    observed: list[int] = []

    def stop() -> example.DeploymentProfile:
        observed.append(stat.S_IMODE(workspace.stat().st_mode))
        raise RuntimeError("stop after workspace check")

    monkeypatch.setattr(example, "deployment_profile", stop)

    assert (
        example.main(
            [
                "--workspace",
                str(workspace),
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    assert observed == [0o700]


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
    profile = example.deployment_profile()
    baseline_image_id = "sha256:" + "c" * 64
    subject_image_id = "sha256:" + "d" * 64
    transformation = example._transformation_document(
        profile=profile,
        conversion=example.ConversionResult(
            subject=subject,
            intermediate_sha256="1" * 64,
            intermediate_byte_length=18_407_320_864,
            subject_sha256=hashlib.sha256(subject.read_bytes()).hexdigest(),
            subject_byte_length=subject.stat().st_size,
        ),
        conversion_image_id=baseline_image_id,
        gguf_image_id=subject_image_id,
    )
    baseline_spec = _baseline_spec()
    subject_spec = _subject_spec(subject)
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
        baseline_image_id=baseline_image_id,
        subject_image_id=subject_image_id,
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


def test_transaction_rejects_a_transformation_not_bound_to_the_current_subject(
    tmp_path: Path,
) -> None:
    profile = example.deployment_profile()
    runtime_root = tmp_path / "runtime"
    source = runtime_root / "models/source"
    source.mkdir(parents=True)
    subject = runtime_root / "models" / example._SUBJECT_NAME
    subject.write_bytes(b"original-subject")
    baseline_image_id = "sha256:" + "a" * 64
    subject_image_id = "sha256:" + "b" * 64
    transformation = example._transformation_document(
        profile=profile,
        conversion=example.ConversionResult(
            subject=subject,
            intermediate_sha256="1" * 64,
            intermediate_byte_length=18_407_320_864,
            subject_sha256=hashlib.sha256(subject.read_bytes()).hexdigest(),
            subject_byte_length=subject.stat().st_size,
        ),
        conversion_image_id=baseline_image_id,
        gguf_image_id=subject_image_id,
    )
    subject.write_bytes(b"replaced-subject")

    with pytest.raises(RuntimeError, match="transformation subject identity"):
        example._prepare_transaction(
            tmp_path / "transaction",
            runtime_root=runtime_root,
            source_checkpoint=source,
            subject=subject,
            baseline_spec=_baseline_spec(),
            subject_spec=_subject_spec(subject),
            transformation=transformation,
            baseline_image_id=baseline_image_id,
            subject_image_id=subject_image_id,
        )


def test_transaction_rejects_a_runtime_spec_not_bound_to_the_subject_bytes(
    tmp_path: Path,
) -> None:
    profile = example.deployment_profile()
    runtime_root = tmp_path / "runtime"
    source = runtime_root / "models/source"
    source.mkdir(parents=True)
    subject = runtime_root / "models" / example._SUBJECT_NAME
    subject.write_bytes(b"GGUF-subject")
    subject_sha256 = hashlib.sha256(subject.read_bytes()).hexdigest()
    baseline_image_id = "sha256:" + "a" * 64
    subject_image_id = "sha256:" + "b" * 64
    transformation = example._transformation_document(
        profile=profile,
        conversion=example.ConversionResult(
            subject=subject,
            intermediate_sha256="1" * 64,
            intermediate_byte_length=18_407_320_864,
            subject_sha256=subject_sha256,
            subject_byte_length=subject.stat().st_size,
        ),
        conversion_image_id=baseline_image_id,
        gguf_image_id=subject_image_id,
    )

    with pytest.raises(RuntimeError, match="subject runtime specification"):
        example._prepare_transaction(
            tmp_path / "transaction",
            runtime_root=runtime_root,
            source_checkpoint=source,
            subject=subject,
            baseline_spec=_baseline_spec(),
            subject_spec={
                **_subject_spec(subject),
                "model_id": "gguf-sha256-" + "0" * 64 + ".gguf",
            },
            transformation=transformation,
            baseline_image_id=baseline_image_id,
            subject_image_id=subject_image_id,
        )


@pytest.mark.parametrize(
    ("side", "setting", "replacement", "message"),
    (
        ("baseline", "seed", 7, "baseline runtime specification"),
        ("baseline", "offline", False, "baseline runtime specification"),
        ("subject", "context_length", 512, "subject runtime specification"),
        ("subject", "timeout_seconds", 30, "subject runtime specification"),
    ),
)
def test_transaction_rejects_execution_settings_outside_the_closed_profile(
    tmp_path: Path,
    side: str,
    setting: str,
    replacement: object,
    message: str,
) -> None:
    profile = example.deployment_profile()
    runtime_root = tmp_path / "runtime"
    source = runtime_root / "models/source"
    source.mkdir(parents=True)
    subject = runtime_root / "models" / example._SUBJECT_NAME
    subject.write_bytes(b"GGUF-subject")
    baseline_image_id = "sha256:" + "a" * 64
    subject_image_id = "sha256:" + "b" * 64
    transformation = example._transformation_document(
        profile=profile,
        conversion=example.ConversionResult(
            subject=subject,
            intermediate_sha256="1" * 64,
            intermediate_byte_length=18_407_320_864,
            subject_sha256=hashlib.sha256(subject.read_bytes()).hexdigest(),
            subject_byte_length=subject.stat().st_size,
        ),
        conversion_image_id=baseline_image_id,
        gguf_image_id=subject_image_id,
    )
    baseline_spec = _baseline_spec()
    subject_spec = _subject_spec(subject)
    selected = baseline_spec if side == "baseline" else subject_spec
    settings = selected["settings"]
    assert isinstance(settings, dict)
    settings[setting] = replacement

    with pytest.raises(RuntimeError, match=message):
        example._prepare_transaction(
            tmp_path / "transaction",
            runtime_root=runtime_root,
            source_checkpoint=source,
            subject=subject,
            baseline_spec=baseline_spec,
            subject_spec=subject_spec,
            transformation=transformation,
            baseline_image_id=baseline_image_id,
            subject_image_id=subject_image_id,
        )


def test_transaction_rejects_a_symlinked_subject_before_hashing(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    source = runtime_root / "models/source"
    source.mkdir(parents=True)
    target = tmp_path / "outside.gguf"
    target.write_bytes(b"outside-subject")
    subject = runtime_root / "models" / example._SUBJECT_NAME
    subject.symlink_to(target)

    with pytest.raises(RuntimeError, match="Q5_K_M GGUF"):
        example._prepare_transaction(
            tmp_path / "transaction",
            runtime_root=runtime_root,
            source_checkpoint=source,
            subject=subject,
            baseline_spec={},
            subject_spec={},
            transformation={},
            baseline_image_id="sha256:" + "a" * 64,
            subject_image_id="sha256:" + "b" * 64,
        )


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
