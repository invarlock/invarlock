from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> Any:
    return _load(
        "tensorrt_outreach_example",
        Path(__file__).resolve().parents[2]
        / "examples/integrations/tensorrt-llm/run.py",
    )


@pytest.fixture(scope="module")
def showcase() -> Any:
    return _load(
        "tensorrt_outreach_showcase",
        Path(__file__).resolve().parents[2]
        / "examples/integrations/tensorrt-llm/showcase.py",
    )


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    root.mkdir()
    for name in ("baseline-engine", "subject-engine"):
        (root / name).mkdir()
        (root / name / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer-contract.json").write_text("{}\n", encoding="utf-8")
    (root / "records.jsonl").write_text(
        '{"id":"1","prompt":"The sky is","expected":" blue"}\n',
        encoding="utf-8",
    )
    (root / "policy.json").write_text(
        '{"resolved_policy":{"metrics":{"exact_match":{"delta_min_pp":-1}}}}',
        encoding="utf-8",
    )
    return root


def _inspection() -> dict[str, object]:
    return {
        role: {
            "artifact_identity_sha256": "sha256:" + digit * 64,
            "model_id": f"tensorrt-llm-{role}",
            "settings": {"engine_bundle_tree_sha256": digit * 64},
        }
        for role, digit in (("baseline", "1"), ("subject", "2"))
    }


@pytest.fixture
def prepare_helper(monkeypatch: pytest.MonkeyPatch) -> Any:
    modules = {
        name: ModuleType(name)
        for name in (
            "modelopt",
            "modelopt.torch",
            "modelopt.torch.export",
            "modelopt.torch.quantization",
            "tensorrt_llm",
            "tensorrt_llm.mapping",
            "tensorrt_llm.models",
            "tensorrt_llm.models.modeling_utils",
            "torch",
            "transformers",
        )
    }

    exported: list[tuple[Any, str, Any, Path, int, int]] = []

    class FakeTensor:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str) -> FakeTensor:
            self.devices.append(device)
            return self

    class FakeModel:
        def __init__(self) -> None:
            self.quantized = False
            self.devices: list[str] = []
            self.batches: list[dict[str, Any]] = []

        def to(self, device: str) -> FakeModel:
            self.devices.append(device)
            return self

        def __call__(self, **batch: Any) -> None:
            self.batches.append(batch)

    class FakeTokenizer:
        is_fast = True
        eos_token_id = 2
        pad_token_id = None
        backend_tokenizer = SimpleNamespace(to_str=lambda: '{"model":{}}')

        def __call__(self, prompts: list[str], **_kwargs: Any) -> dict[str, Any]:
            assert prompts
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    def export(
        model: Any,
        decoder_type: str,
        dtype: Any,
        export_dir: Path,
        inference_tensor_parallel: int,
        inference_pipeline_parallel: int,
    ) -> None:
        exported.append(
            (
                model,
                decoder_type,
                dtype,
                export_dir,
                inference_tensor_parallel,
                inference_pipeline_parallel,
            )
        )
        export_dir.mkdir(mode=0o700)
        quantization = "FP8" if model.quantized else None
        (export_dir / "config.json").write_text(
            json.dumps({"quantization": {"quant_algo": quantization}}) + "\n",
            encoding="utf-8",
        )

    tokenizer = FakeTokenizer()
    dtype = object()
    model = FakeModel()
    modules["modelopt.torch.export"].export_tensorrt_llm_checkpoint = export  # type: ignore[attr-defined]
    modules["modelopt.torch.quantization"].FP8_DEFAULT_CFG = {  # type: ignore[attr-defined]
        "algorithm": "max"
    }

    def quantize(model: FakeModel, _config: Any, *, forward_loop: Any) -> None:
        model.quantized = True
        forward_loop()

    modules["modelopt.torch.quantization"].quantize = quantize  # type: ignore[attr-defined]
    modules["torch"].bfloat16 = dtype  # type: ignore[attr-defined]
    modules["torch"].inference_mode = FakeInferenceMode  # type: ignore[attr-defined]
    modules["transformers"].AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: tokenizer
    )
    modules["transformers"].AutoModelForCausalLM = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: model
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    helper = _load(
        "tensorrt_outreach_prepare",
        Path(__file__).resolve().parents[2]
        / "examples/integrations/tensorrt-llm/prepare.py",
    )
    helper._test_exported = exported
    helper._test_model = model
    helper._test_dtype = dtype
    helper._test_tensor_type = FakeTensor
    return helper


def test_showcase_uses_one_pinned_qwen3_family(showcase: Any) -> None:
    assert showcase._MODEL == (
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
    )
    assert showcase._VARIANTS == {"baseline": "none", "subject": "fp8"}


@pytest.mark.parametrize("entrypoint", ("run.py", "showcase.py"))
def test_public_entrypoints_start_as_direct_scripts(entrypoint: str) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(repository / "src"),
            str(repository / "addins/tensorrt_llm/src"),
            str(repository),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/integrations/tensorrt-llm" / entrypoint),
            "--help",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_image_requires_immutable_identity(example: Any) -> None:
    digest = "sha256:" + "a" * 64
    assert example._image(digest) == (digest, digest)
    reference = "registry.example/runtime@" + digest
    assert example._image(reference) == (reference, digest)
    with pytest.raises(ValueError, match="immutable"):
        example._image("runtime:latest")


def test_resource_root_is_closed_and_never_overwritten(
    example: Any, inputs: Path
) -> None:
    assert example._root(inputs) == inputs
    (inputs / "invarlock-tensorrt-example-request.yaml").write_text(
        "existing", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="already exists"):
        example._root(inputs)


def test_resource_root_rejects_symlinked_artifacts(example: Any, inputs: Path) -> None:
    tokenizer = inputs / "tokenizer-contract.json"
    tokenizer.rename(inputs / "real-tokenizer.json")
    tokenizer.symlink_to("real-tokenizer.json")
    with pytest.raises(ValueError, match="missing or unsafe"):
        example._root(inputs)


def test_inspect_runs_real_offline_gpu_probe_and_validates_output(
    example: Any,
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.extend(command)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(command, 0, json.dumps(_inspection()), "")

    monkeypatch.setattr(example.subprocess, "run", run)
    digest = "sha256:" + "a" * 64
    assert example._inspect(inputs, digest, digest, "cuda:1") == _inspection()
    assert seen[seen.index("--network") + 1] == "none"
    assert seen[seen.index("--gpus") + 1] == "device=1"
    assert "--read-only" in seen and "--pull=never" in seen
    assert "65532:65532" in seen
    assert f"INVARLOCK_RUNTIME_IMAGE_DIGEST={digest}" in seen
    assert f"type=bind,src={inputs},dst=/resources,readonly" in seen
    assert seen[-6:] == [
        "--context-length",
        "1024",
        "--max-output-tokens",
        "1",
        "--timeout-seconds",
        "300",
    ]


def test_inspect_surfaces_runtime_diagnostic(
    example: Any,
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "unsupported tokenizer contract"
        ),
    )
    digest = "sha256:" + "a" * 64
    with pytest.raises(RuntimeError, match="unsupported tokenizer contract"):
        example._inspect(inputs, digest, digest, "cuda:0")


def test_inspect_rejects_same_engine_identity(
    example: Any, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _inspection()
    payload["subject"]["artifact_identity_sha256"] = payload["baseline"][  # type: ignore[index]
        "artifact_identity_sha256"
    ]
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps(payload), ""
        ),
    )
    digest = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="must be distinct"):
        example._inspect(inputs, digest, digest, "cuda:0")


def test_prepare_closes_request_keys_and_independent_trust(
    example: Any, inputs: Path, tmp_path: Path
) -> None:
    digest = "sha256:" + "a" * 64
    paths = example._prepare(
        inputs,
        tmp_path / "output",
        _inspection(),
        digest,
        ("hf://owner/baseline@rev", "hf://owner/subject@rev"),
    )
    request = yaml.safe_load(paths["request"].read_text(encoding="utf-8"))
    assert request["comparison"]["baseline"]["runtime"]["provider"] == ("tensorrt_llm")
    assert request["comparison"]["subject"]["artifact"]["path"] == ("subject-engine")
    assert request["comparison"]["metric"] == "exact_match"
    trust = json.loads(paths["trust"].read_text(encoding="utf-8"))
    assert trust["anchors"]["baseline_artifact_digest"] == "sha256:" + "1" * 64
    assert trust["anchors"]["subject_runtime_digest"] == digest
    assert trust["anchors"]["schedule_digest"].startswith("sha256:")
    assert stat.S_IMODE(paths["signer"].stat().st_mode) == 0o600
    assert stat.S_IMODE(paths["verifier"].stat().st_mode) == 0o600


def test_execute_runs_preflight_evaluate_verify_report_with_provider_resources(
    example: Any,
    inputs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        name: tmp_path / name
        for name in ("request", "evidence", "signer", "trust", "receipt", "report")
    }
    report = paths["evidence"] / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        '{"baseline":{"mean_score":0.5},'
        '"subject":{"mean_score":0.49},"verdict":"pass"}\n',
        encoding="utf-8",
    )
    paths["receipt"].write_text(
        '{"statement":{"verdict":{"ok":true,"integrity_ok":true,'
        '"policy_verdict":"pass"}}}\n',
        encoding="utf-8",
    )
    paths["report"].write_text("<html>comparison</html>\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(
        command: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        assert check
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(example.subprocess, "run", run)
    digest = "sha256:" + "a" * 64
    example._execute(inputs, paths, digest, digest, ("cuda:0", "cuda:1"))
    assert len(calls) == 4
    assert "--preflight" in calls[0][0] and calls[0][0][3] == "evaluate"
    assert "--preflight" not in calls[1][0] and calls[1][0][3] == "evaluate"
    assert calls[2][0][3] == "verify" and "--trust-profile" in calls[2][0]
    assert calls[3][0][3] == "report"
    for command, environment in calls[:2]:
        assert command[command.index("--baseline-runtime-device") + 1] == "cuda:0"
        assert command[command.index("--subject-runtime-device") + 1] == "cuda:1"
        assert environment["INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT"] == str(inputs)

    report.write_text(
        '{"baseline":{"mean_score":0.39},'
        '"subject":{"mean_score":0.49},"verdict":"pass"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline engine solved fewer"):
        example._execute(inputs, paths, digest, digest, ("cuda:0", "cuda:1"))

    paths["receipt"].unlink()
    with pytest.raises(ValueError, match="missing verified outputs"):
        example._execute(inputs, paths, digest, digest, ("cuda:0", "cuda:1"))

    paths["receipt"].write_text("{}\n", encoding="utf-8")
    report.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid outputs"):
        example._execute(inputs, paths, digest, digest, ("cuda:0", "cuda:1"))


def test_main_is_one_inspect_prepare_execute_transaction(
    example: Any, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []
    digest = "sha256:" + "a" * 64
    paths = {name: inputs / name for name in ("evidence", "receipt")}
    monkeypatch.setattr(
        example,
        "_inspect",
        lambda *_args: observed.append("inspect") or _inspection(),
    )
    monkeypatch.setattr(
        example,
        "_prepare",
        lambda *_args: observed.append("prepare") or paths,
    )
    monkeypatch.setattr(example, "_execute", lambda *_args: observed.append("execute"))
    assert (
        example.main(
            [
                "--runtime-image",
                digest,
                "--resource-root",
                str(inputs),
                "--baseline-locator",
                "hf://baseline@rev",
                "--subject-locator",
                "hf://subject@rev",
            ]
        )
        == 0
    )
    assert observed == ["inspect", "prepare", "execute"]


def test_image_probe_uses_addin_provider_and_official_runner(
    monkeypatch: pytest.MonkeyPatch, inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    packages = {
        name: ModuleType(name)
        for name in (
            "invarlock_addins",
            "invarlock_addins.tensorrt_llm",
            "invarlock_addins.tensorrt_llm.execution",
            "invarlock_addins.tensorrt_llm.provider",
            "invarlock_addins.tensorrt_llm.session",
        )
    }

    class Bindings:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class Provider:
        def inspect_runtime_spec(self, bindings: Any, **settings: Any) -> Any:
            role = bindings.engine_bundle_path.name.split("-", 1)[0]
            return SimpleNamespace(model_id=role, settings=settings)

        def identify_artifact(self, spec: Any) -> Any:
            return spec

    packages[
        "invarlock_addins.tensorrt_llm.execution"
    ].official_tensorrt_llm_runner_path = lambda: Path("/runner")  # type: ignore[attr-defined]
    packages["invarlock_addins.tensorrt_llm.provider"].TensorRTLLMProvider = Provider  # type: ignore[attr-defined]
    packages[
        "invarlock_addins.tensorrt_llm.session"
    ].TensorRTLLMRuntimeBindings = Bindings  # type: ignore[attr-defined]
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    import invarlock.runtime_security_helpers as security

    monkeypatch.setattr(security, "strict_container_boundary_present", lambda: True)
    monkeypatch.setattr(
        "invarlock.core.runtime_provider.artifact_identity_sha256",
        lambda _value: "7" * 64,
    )
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", digest)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", digest)
    helper = _load(
        "tensorrt_outreach_inspect",
        Path(__file__).resolve().parents[2]
        / "examples/integrations/tensorrt-llm/engine_inspect.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_inspect.py",
            "--resource-root",
            str(inputs),
            "--context-length",
            "128",
            "--max-output-tokens",
            "2",
            "--timeout-seconds",
            "30",
        ],
    )
    helper.main()
    output = json.loads(capsys.readouterr().out)
    assert output["baseline"]["model_id"] == "baseline"
    assert output["subject"]["model_id"] == "subject"


def test_showcase_workspace_download_and_input_materialization(
    showcase: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = showcase._create_workspace(tmp_path / "showcase")
    assert paths.workspace == tmp_path / "showcase"
    with pytest.raises(FileExistsError, match="already exists"):
        showcase._create_workspace(paths.workspace)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    canonical = showcase._create_workspace(alias_parent / "canonical-showcase")
    assert canonical.workspace == real_parent / "canonical-showcase"

    def download(**arguments: Any) -> str:
        destination = Path(arguments["local_dir"])
        destination.mkdir()
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setattr(showcase, "snapshot_download", download)
    model = showcase._download(paths)
    assert model == paths.models / "qwen3-0.6b"
    assert (model / "model.safetensors").stat().st_mode & 0o777 == 0o644

    monkeypatch.setattr(
        showcase,
        "snapshot_download",
        lambda **_arguments: str(tmp_path),
    )
    with pytest.raises(RuntimeError, match="unexpected destination"):
        showcase._download(paths)

    class Tokenizer:
        expected = ""

        def __call__(self, value: str, **_kwargs: Any) -> dict[str, list[int]]:
            self.expected = value
            return {"input_ids": [1]}

        def decode(self, _ids: list[int], **_kwargs: Any) -> str:
            return self.expected

    tokenizer = Tokenizer()
    for role in showcase._VARIANTS:
        directory = paths.work / role
        directory.mkdir()
        (directory / f"{role}.tokenizer-contract.json").write_bytes(b"same")
    showcase._prepare_inputs(paths, tokenizer=tokenizer)
    records = (
        (paths.resources / "records.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(records) == 102
    parsed_records = [json.loads(record) for record in records]
    assert len({record["id"] for record in parsed_records}) == 102
    assert len({record["prompt"] for record in parsed_records}) == 102
    assert all(record["expected"].startswith(" ") for record in parsed_records)
    policy = json.loads((paths.resources / "policy.json").read_text())
    assert (
        policy["resolved_policy"]["metrics"]["exact_match"]["minimum_record_count"]
        == 102
    )
    assert policy["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -10.0,
        "maximum_interval_width_pp": 20.0,
        "minimum_record_count": 102,
    }
    (paths.work / "subject/subject.tokenizer-contract.json").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="share one tokenizer"):
        showcase._prepare_inputs(paths, tokenizer=tokenizer)


def test_showcase_rejects_non_lossless_exact_match_targets(
    showcase: Any, tmp_path: Path
) -> None:
    paths = showcase._create_workspace(tmp_path / "showcase")
    for role in showcase._VARIANTS:
        directory = paths.work / role
        directory.mkdir()
        (directory / f"{role}.tokenizer-contract.json").write_bytes(b"same")

    class Tokenizer:
        def __call__(self, _value: str, **_kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": [1, 2]}

        def decode(self, _ids: list[int], **_kwargs: Any) -> str:
            return " other"

    with pytest.raises(RuntimeError, match="losslessly decoded token"):
        showcase._prepare_inputs(paths, tokenizer=Tokenizer())


def test_showcase_policy_accepts_moderate_discordance_at_102_records() -> None:
    from invarlock.paired_exact_match import paired_exact_match_statistics

    baseline = [True] * 40 + [True] * 10 + [False] * 12 + [False] * 40
    subject = [True] * 40 + [False] * 10 + [True] * 12 + [False] * 40
    statistics = paired_exact_match_statistics(baseline, subject)

    interval = statistics.effect_size_confidence_interval
    assert interval.lower_pp >= -10.0
    assert interval.upper_pp - interval.lower_pp <= 20.0


def test_showcase_policy_rejects_high_discordance_at_102_records() -> None:
    from invarlock.paired_exact_match import paired_exact_match_statistics

    baseline = [True] * 26 + [True] * 25 + [False] * 25 + [False] * 26
    subject = [True] * 26 + [False] * 25 + [True] * 25 + [False] * 26
    interval = paired_exact_match_statistics(
        baseline, subject
    ).effect_size_confidence_interval

    assert interval.upper_pp - interval.lower_pp > 20.0


def test_showcase_container_build_and_transaction_commands(
    showcase: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = showcase._create_workspace(tmp_path / "showcase")
    (paths.models / "qwen3-0.6b").mkdir()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(showcase.subprocess, "run", run)
    digest = "sha256:" + "a" * 64
    showcase._container_build(
        paths,
        role="baseline",
        device="1",
        image=digest,
        container_engine="docker",
    )
    command = calls[-1][0]
    assert stat.S_IMODE((paths.work / "baseline").stat().st_mode) == 0o777
    assert command[command.index("--gpus") + 1] == "device=1"
    assert "--network" in command and "none" in command
    assert "LD_LIBRARY_PATH=/usr/local/tensorrt/lib" in command
    assert "/resources/baseline-engine" in command
    assert command[command.index("--quantization") + 1] == "none"
    with pytest.raises(ValueError, match="nonnegative"):
        showcase._container_build(
            paths,
            role="subject",
            device="cuda:1",
            image=digest,
            container_engine="docker",
        )

    showcase._run_transaction(
        paths,
        image=digest,
        devices=("0", "1"),
        container_engine="docker",
    )
    transaction, options = calls[-1]
    assert transaction[1].endswith("tensorrt-llm/run.py")
    assert transaction[transaction.index("--baseline-device") + 1] == "cuda:0"
    assert transaction[transaction.index("--subject-device") + 1] == "cuda:1"
    assert options["env"]["INVARLOCK_CONTAINER_ENGINE"] == "docker"


def test_showcase_main_runs_two_downloads_and_builds(
    showcase: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = showcase._create_workspace(tmp_path / "prepared")
    events: list[str] = []
    monkeypatch.setattr(showcase, "_require_committed_checkout", lambda _root: "c" * 40)
    monkeypatch.setattr(showcase, "_create_workspace", lambda _value: paths)
    monkeypatch.setattr(
        showcase,
        "_download",
        lambda _paths: events.append("download"),
    )
    monkeypatch.setattr(
        showcase,
        "_runtime_image",
        lambda **_arguments: ("sha256:" + "a" * 64,) * 2,
    )
    monkeypatch.setattr(
        showcase,
        "_container_build",
        lambda _paths, **values: events.append(f"build-{values['role']}"),
    )
    monkeypatch.setattr(
        showcase, "_prepare_inputs", lambda _paths: events.append("inputs")
    )
    monkeypatch.setattr(
        showcase,
        "_run_transaction",
        lambda _paths, **_values: events.append("transaction"),
    )
    assert showcase.main(["--workspace", str(tmp_path / "ignored")]) == 0
    assert events[0] == "download"
    assert set(events[1:3]) == {"build-baseline", "build-subject"}
    assert events[-2:] == ["inputs", "transaction"]
    assert "Complete TensorRT-LLM" in capsys.readouterr().out

    assert showcase.main(["--baseline-device", "0", "--subject-device", "0"]) == 2
    assert "distinct GPU" in capsys.readouterr().err
    monkeypatch.setattr(
        showcase,
        "_runtime_image",
        lambda **_arguments: ("sha256:" + "a" * 64, "sha256:" + "b" * 64),
    )
    (paths.workspace / "runtime-build").rmdir()
    assert showcase.main([]) == 2
    assert "not immutable" in capsys.readouterr().err


def test_showcase_rejects_dirty_source_before_downloads(
    showcase: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads: list[str] = []
    monkeypatch.setattr(
        showcase,
        "_require_committed_checkout",
        lambda _root: (_ for _ in ()).throw(RuntimeError("tracked source is dirty")),
    )
    monkeypatch.setattr(
        showcase,
        "_download",
        lambda _paths: downloads.append("download"),
    )

    assert showcase.main([]) == 2
    assert downloads == []
    assert "tracked source is dirty" in capsys.readouterr().err


def test_prepare_helper_contract_conversion_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_helper: Any
) -> None:
    prepare = prepare_helper
    contract = json.loads(prepare._canonical_tokenizer_contract(tmp_path))
    assert contract["pad_token_id"] == 2
    assert contract["tokenizer_json"] == {"model": {}}

    checkpoint = tmp_path / "checkpoint"
    prepare._convert(
        tmp_path,
        checkpoint,
        quantization="none",
        calibration_records=None,
    )
    assert checkpoint.is_dir()
    assert prepare._test_exported == [
        (
            prepare._test_model,
            "qwen",
            prepare._test_dtype,
            checkpoint,
            1,
            1,
        )
    ]

    fp8_records = tmp_path / "records.json"
    fp8_records.write_text(
        '[{"id":"one","prompt":"first"},{"id":"two","prompt":"second"}]',
        encoding="utf-8",
    )
    fp8_checkpoint = tmp_path / "fp8-checkpoint"
    prepare._convert(
        tmp_path,
        fp8_checkpoint,
        quantization="fp8",
        calibration_records=fp8_records,
    )
    assert prepare._test_model.quantized is True
    assert prepare._test_model.devices == ["cuda"]
    assert len(prepare._test_model.batches) == 1
    assert all(
        tensor.devices == ["cuda"] for tensor in prepare._test_model.batches[0].values()
    )
    with pytest.raises(RuntimeError, match="requires calibration"):
        prepare._convert(
            tmp_path,
            tmp_path / "missing-calibration",
            quantization="fp8",
            calibration_records=None,
        )

    engine = tmp_path / "engine"
    monkeypatch.setattr(prepare.shutil, "which", lambda _name: "/trtllm-build")

    def build(command: list[str], *, check: bool) -> None:
        assert check and "--max_input_len" in command
        assert command[command.index("--output_timing_cache") + 1] == str(
            checkpoint.parent / "model.cache"
        )
        assert command[command.index("--gemm_plugin") + 1] == "bfloat16"
        engine.mkdir()
        (engine / "config.json").write_text("{}", encoding="utf-8")
        (engine / "rank0.engine").write_bytes(b"engine")

    monkeypatch.setattr(prepare.subprocess, "run", build)
    prepare._build(checkpoint, engine, quantization="none")

    fp8_engine = tmp_path / "fp8-engine"

    def build_fp8(command: list[str], *, check: bool) -> None:
        assert check
        assert command[command.index("--gemm_plugin") + 1] == "disable"
        fp8_engine.mkdir()
        (fp8_engine / "config.json").write_text("{}", encoding="utf-8")
        (fp8_engine / "rank0.engine").write_bytes(b"engine")

    monkeypatch.setattr(prepare.subprocess, "run", build_fp8)
    prepare._build(fp8_checkpoint, fp8_engine, quantization="fp8")
    monkeypatch.setattr(prepare.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        prepare._build(checkpoint, tmp_path / "unused", quantization="none")


def test_prepare_helper_main_and_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_helper: Any
) -> None:
    prepare = prepare_helper
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = tmp_path / "checkpoint"
    engine = tmp_path / "engine"
    contract = tmp_path / "contract.json"
    monkeypatch.setattr(
        prepare, "_canonical_tokenizer_contract", lambda _model: b"contract"
    )

    monkeypatch.setattr(
        prepare,
        "_convert",
        lambda _model, path, **_kwargs: path.mkdir(),
    )
    monkeypatch.setattr(
        prepare,
        "_build",
        lambda _checkpoint, path, **_kwargs: path.mkdir(),
    )
    arguments = [
        "--model",
        str(model),
        "--checkpoint",
        str(checkpoint),
        "--engine",
        str(engine),
        "--tokenizer-contract",
        str(contract),
        "--quantization",
        "none",
    ]
    assert prepare.main(arguments) == 0
    assert contract.read_bytes() == b"contract"
    with pytest.raises(RuntimeError, match="already exists"):
        prepare.main(arguments)
